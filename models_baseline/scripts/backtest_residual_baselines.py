#!/usr/bin/env python3
"""Expanding-window monthly backtest for residual baselines.

For each month from --initial-train-end+1 month onward:
  - train on all games strictly before the window start
  - predict on games within the window
  - emit predictions.csv (matches existing baseline schema + market columns)

Usage example:
  python backtest_residual_baselines.py --model logistic --use-market feature
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling_common import (
    DEFAULT_FEATURES_DB,
    DEFAULT_OUTPUT_ROOT,
    LABEL_COLUMN,
    evaluate,
    load_table,
    prepare_model_frame,
)
from residual_common import (
    MARKET_KEEP_COLS,
    USE_MARKET_CHOICES,
    attach_market,
    disagree_metrics,
    feat_cols_for_mode,
)
from train_residual_baselines import (
    fit_logistic,
    fit_mlp,
    fit_xgboost,
    set_seed,
)

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "backtest_residual"
FITTERS = {"logistic": fit_logistic, "xgboost": fit_xgboost, "mlp": fit_mlp}


def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    periods = pd.to_datetime(dates).dt.to_period("M").drop_duplicates().sort_values()
    return [period.to_timestamp() for period in periods]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("logistic", "xgboost", "mlp"), required=True)
    p.add_argument("--use-market", choices=USE_MARKET_CHOICES, default="residual")
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--table", default="model_games")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--initial-train-end", default="2023-12-31")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    # torch-shared
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--val-fraction", type=float, default=0.15)
    # loss
    p.add_argument("--loss", choices=("bce", "ev", "mixed"), default="bce")
    p.add_argument("--ev-weight", type=float, default=0.0,
                   help="lambda for mixed loss: BCE + lambda * EV_loss")
    # MLP
    p.add_argument("--hidden-dims", default="128,64,32,16")
    p.add_argument("--dropout", type=float, default=0.10)
    # XGBoost
    p.add_argument("--xgb-n-estimators", type=int, default=200)
    p.add_argument("--xgb-max-depth", type=int, default=2)
    p.add_argument("--xgb-learning-rate", type=float, default=0.05)
    p.add_argument("--xgb-subsample", type=float, default=0.9)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.output_dir or (
        DEFAULT_OUTPUT_DIR / f"{args.model}_{args.use_market}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.features_db, args.table)
    df, features = prepare_model_frame(df, args.table, args.min_games_before)
    df = attach_market(df)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    feat_cols = feat_cols_for_mode(features, args.use_market)
    use_offset = args.use_market == "residual"

    initial_train_end = pd.Timestamp(args.initial_train_end)
    windows = [
        s
        for s in month_starts(df.loc[df["game_date"] > initial_train_end, "game_date"])
        if s > initial_train_end
    ]
    if not windows:
        raise ValueError("No backtest windows found after initial_train_end.")

    all_predictions = []
    window_metrics = []

    for i, window_start in enumerate(windows, 1):
        window_end = window_start + pd.offsets.MonthBegin(1)
        train_df = df[df["game_date"] < window_start].copy()
        test_df = df[
            (df["game_date"] >= window_start) & (df["game_date"] < window_end)
        ].copy()
        if train_df.empty or test_df.empty:
            print(f"[w{i:02d}/{len(windows)}] {window_start.date()}: skipped (empty)")
            continue

        train_prob, test_prob, summary = FITTERS[args.model](
            args, train_df, test_df, feat_cols, use_offset
        )

        metrics = evaluate(test_df[LABEL_COLUMN], test_prob)
        metrics.update(
            {
                "window_start": window_start.date().isoformat(),
                "window_end": (window_end - pd.Timedelta(days=1)).date().isoformat(),
                "train_n": int(len(train_df)),
            }
        )

        preds = test_df[
            [
                "game_id",
                "game_date",
                "home_team_id",
                "away_team_id",
                LABEL_COLUMN,
                *MARKET_KEEP_COLS,
            ]
        ].copy()
        preds["pred_home_win_prob"] = test_prob
        preds["window_start"] = window_start.date().isoformat()
        all_predictions.append(preds)

        # disagree(.60) for this window
        dg = disagree_metrics(preds, 0.60)
        metrics.update(
            {
                "disagree60_bets": dg["bets"],
                "disagree60_wins": dg["wins"],
                "disagree60_roi": dg["roi"],
            }
        )
        window_metrics.append(metrics)
        print(
            f"[w{i:02d}/{len(windows)}] {window_start.date()}: "
            f"train_n={len(train_df)} test_n={len(test_df)} "
            f"BCE={metrics['log_loss']:.4f} acc={metrics['accuracy']:.3f} "
            f"d60_bets={dg['bets']} d60_roi={dg['roi']*100:+.2f}%"
        )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    overall = evaluate(predictions_df[LABEL_COLUMN], predictions_df["pred_home_win_prob"])
    overall_d60 = disagree_metrics(predictions_df, 0.60)

    config = {
        "model": args.model,
        "use_market": args.use_market,
        "use_offset": use_offset,
        "feat_cols": feat_cols,
        "n_features_input": len(feat_cols),
        "initial_train_end": args.initial_train_end,
        "n_windows": len(window_metrics),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "overall_metrics.json").write_text(
        json.dumps({"overall": overall, "disagree60": overall_d60}, indent=2) + "\n"
    )
    pd.DataFrame(window_metrics).to_csv(out_dir / "window_metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "predictions.csv", index=False)

    print()
    print(json.dumps({"overall": overall, "disagree60": overall_d60}, indent=2))
    print(f"Saved backtest artifacts to {out_dir}")


if __name__ == "__main__":
    main()
