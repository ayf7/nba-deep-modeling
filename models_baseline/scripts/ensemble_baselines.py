#!/usr/bin/env python3
"""Ensemble baseline backtest predictions.

Combines per-game predictions from multiple baseline backtests into a single
ensemble prediction. Three combination strategies:

  average         Arithmetic mean of base-model probabilities.
  weighted        Per-model weights optimized on a rolling calibration window
                  (the most recent --calibration-months of training data).
  stacked         Logistic regression meta-learner retrained each window on
                  the base-model predictions from prior months.

All strategies produce output artifacts compatible with the standard backtest
schema (predictions.csv, overall_metrics.json, window_metrics.csv), so they
can be fed directly into evaluate_betting_strategy.py.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from modeling_common import DEFAULT_OUTPUT_ROOT, evaluate

MODELS = ("logistic", "xgboost", "mlp")
LABEL_COLUMN = "label_home_win"


def load_backtest_predictions(artifact_root: Path, models: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    frames = {}
    for model in models:
        path = artifact_root / f"backtest_{model}" / "predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing backtest predictions: {path}")
        df = pd.read_csv(path)
        df["game_date"] = pd.to_datetime(df["game_date"])
        frames[model] = df
    return frames


def merge_predictions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_model = list(frames.keys())[0]
    merged = frames[base_model][
        ["game_id", "game_date", "home_team_id", "away_team_id", LABEL_COLUMN, "window_start"]
    ].copy()
    for model, df in frames.items():
        merged = merged.merge(
            df[["game_id", "pred_home_win_prob"]].rename(
                columns={"pred_home_win_prob": f"prob_{model}"}
            ),
            on="game_id",
            how="inner",
        )
    return merged.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def ensemble_average(merged: pd.DataFrame, models: tuple[str, ...]) -> np.ndarray:
    prob_cols = [f"prob_{m}" for m in models]
    return merged[prob_cols].mean(axis=1).to_numpy()


def optimize_weights(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid_steps: int = 21,
) -> np.ndarray:
    n_models = probs.shape[1]
    best_loss = float("inf")
    best_weights = np.ones(n_models) / n_models

    if n_models == 2:
        for w0 in np.linspace(0, 1, grid_steps):
            weights = np.array([w0, 1 - w0])
            blended = probs @ weights
            blended = np.clip(blended, 1e-7, 1 - 1e-7)
            loss = log_loss(y_true, blended, labels=[0, 1])
            if loss < best_loss:
                best_loss = loss
                best_weights = weights
    elif n_models == 3:
        for w0, w1 in product(
            np.linspace(0, 1, grid_steps), np.linspace(0, 1, grid_steps)
        ):
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9:
                continue
            weights = np.array([w0, w1, max(w2, 0.0)])
            blended = probs @ weights
            blended = np.clip(blended, 1e-7, 1 - 1e-7)
            loss = log_loss(y_true, blended, labels=[0, 1])
            if loss < best_loss:
                best_loss = loss
                best_weights = weights
    else:
        best_weights = np.ones(n_models) / n_models

    return best_weights


def ensemble_weighted(
    merged: pd.DataFrame,
    models: tuple[str, ...],
    calibration_months: int,
) -> tuple[np.ndarray, list[dict]]:
    prob_cols = [f"prob_{m}" for m in models]
    windows = sorted(merged["window_start"].unique())
    all_probs = np.full(len(merged), np.nan)
    weight_log = []

    for window in windows:
        test_mask = merged["window_start"] == window
        window_ts = pd.Timestamp(window)
        cal_start = window_ts - pd.DateOffset(months=calibration_months)
        cal_mask = (
            (merged["game_date"] >= cal_start)
            & (merged["game_date"] < window_ts)
        )

        if cal_mask.sum() < 20:
            weights = np.ones(len(models)) / len(models)
        else:
            cal_probs = merged.loc[cal_mask, prob_cols].to_numpy()
            cal_y = merged.loc[cal_mask, LABEL_COLUMN].to_numpy()
            weights = optimize_weights(cal_y, cal_probs)

        blended = merged.loc[test_mask, prob_cols].to_numpy() @ weights
        all_probs[test_mask.to_numpy()] = blended
        weight_log.append({
            "window_start": window,
            "calibration_n": int(cal_mask.sum()),
            **{f"weight_{m}": float(w) for m, w in zip(models, weights)},
        })

    return np.clip(all_probs, 1e-7, 1 - 1e-7), weight_log


def ensemble_stacked(
    merged: pd.DataFrame,
    models: tuple[str, ...],
) -> tuple[np.ndarray, list[dict]]:
    prob_cols = [f"prob_{m}" for m in models]
    windows = sorted(merged["window_start"].unique())
    all_probs = np.full(len(merged), np.nan)
    coef_log = []

    for window in windows:
        test_mask = merged["window_start"] == window
        window_ts = pd.Timestamp(window)
        train_mask = merged["game_date"] < window_ts

        if train_mask.sum() < 30:
            blended = merged.loc[test_mask, prob_cols].mean(axis=1).to_numpy()
            coefs = {f"coef_{m}": float("nan") for m in models}
        else:
            train_x = merged.loc[train_mask, prob_cols].to_numpy()
            train_y = merged.loc[train_mask, LABEL_COLUMN].to_numpy()
            meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            meta.fit(train_x, train_y)
            test_x = merged.loc[test_mask, prob_cols].to_numpy()
            blended = meta.predict_proba(test_x)[:, 1]
            coefs = {f"coef_{m}": float(c) for m, c in zip(models, meta.coef_[0])}
            coefs["intercept"] = float(meta.intercept_[0])

        all_probs[test_mask.to_numpy()] = blended
        coef_log.append({"window_start": window, "train_n": int(train_mask.sum()), **coefs})

    return np.clip(all_probs, 1e-7, 1 - 1e-7), coef_log


def build_outputs(
    merged: pd.DataFrame,
    ensemble_probs: np.ndarray,
    strategy: str,
    config: dict,
    output_dir: Path,
    extra_log: list[dict] | None = None,
) -> dict:
    predictions = merged[
        ["game_id", "game_date", "home_team_id", "away_team_id", LABEL_COLUMN, "window_start"]
    ].copy()
    predictions["pred_home_win_prob"] = ensemble_probs

    overall = evaluate(predictions[LABEL_COLUMN], ensemble_probs)

    window_metrics = []
    for window in sorted(merged["window_start"].unique()):
        mask = merged["window_start"] == window
        y = merged.loc[mask, LABEL_COLUMN]
        p = ensemble_probs[mask.to_numpy()]
        wm = evaluate(y, p)
        wm["window_start"] = window
        wm["window_end"] = (
            pd.Timestamp(window) + pd.offsets.MonthEnd(0)
        ).date().isoformat()
        window_metrics.append(wm)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall, indent=2) + "\n"
    )
    pd.DataFrame(window_metrics).to_csv(output_dir / "window_metrics.csv", index=False)
    predictions["game_date"] = predictions["game_date"].dt.strftime("%Y-%m-%d")
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    if extra_log:
        pd.DataFrame(extra_log).to_csv(
            output_dir / f"{strategy}_details.csv", index=False
        )

    return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble baseline backtest predictions."
    )
    parser.add_argument(
        "--strategy",
        choices=("average", "weighted", "stacked", "all"),
        default="all",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to artifacts/ensemble_<strategy>.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        help="Base models to ensemble.",
    )
    parser.add_argument(
        "--calibration-months",
        type=int,
        default=6,
        help="Months of calibration data for weighted strategy.",
    )
    return parser.parse_args()


def run_strategy(
    strategy: str,
    merged: pd.DataFrame,
    models: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict] | None]:
    if strategy == "average":
        return ensemble_average(merged, models), None
    elif strategy == "weighted":
        return ensemble_weighted(merged, models, args.calibration_months)
    elif strategy == "stacked":
        return ensemble_stacked(merged, models)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    args = parse_args()
    models = tuple(args.models)

    frames = load_backtest_predictions(args.artifact_root, models)
    merged = merge_predictions(frames)
    print(f"Merged {len(merged)} games from {len(models)} models: {', '.join(models)}")

    strategies = ["average", "weighted", "stacked"] if args.strategy == "all" else [args.strategy]

    for strategy in strategies:
        probs, extra_log = run_strategy(strategy, merged, models, args)

        output_dir = args.output_dir or (args.artifact_root / f"ensemble_{strategy}")
        if args.strategy == "all":
            output_dir = args.artifact_root / f"ensemble_{strategy}"

        config = {
            "strategy": strategy,
            "base_models": list(models),
            "calibration_months": args.calibration_months if strategy == "weighted" else None,
        }
        overall = build_outputs(merged, probs, strategy, config, output_dir, extra_log)

        print(f"\n=== {strategy} ensemble ===")
        print(json.dumps(overall, indent=2))
        print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
