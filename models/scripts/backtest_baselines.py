#!/usr/bin/env python3
"""Run expanding-window backtests for baseline models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from modeling_common import (
    DEFAULT_FEATURES_DB,
    DEFAULT_OUTPUT_ROOT,
    LABEL_COLUMN,
    build_preprocessor,
    evaluate,
    feature_columns,
    load_table,
    prepare_model_frame,
)


def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    periods = pd.to_datetime(dates).dt.to_period("M").drop_duplicates().sort_values()
    return [period.to_timestamp() for period in periods]


def make_classifier(args: argparse.Namespace) -> tuple[Any, bool, dict[str, Any]]:
    if args.model == "logistic":
        params = {"max_iter": args.max_iter}
        return LogisticRegression(**params), True, params

    if args.model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with "
                "`/home/ayf7/trading/env/bin/pip install xgboost` or update the conda env."
            ) from exc

        params = {
            "n_estimators": args.xgb_n_estimators,
            "max_depth": args.xgb_max_depth,
            "learning_rate": args.xgb_learning_rate,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": args.random_state,
        }
        return XGBClassifier(**params), False, params

    raise ValueError(f"Unsupported model: {args.model}")


def mlp_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hidden_dim_1": args.mlp_hidden_dim_1,
        "hidden_dim_2": args.mlp_hidden_dim_2,
        "dropout": args.mlp_dropout,
        "learning_rate": args.mlp_learning_rate,
        "weight_decay": args.mlp_weight_decay,
        "batch_size": args.mlp_batch_size,
        "epochs": args.mlp_epochs,
        "validation_fraction": args.mlp_validation_fraction,
        "patience": args.mlp_patience,
        "random_state": args.random_state,
        "device": args.mlp_device,
    }


def fit_predict_mlp(
    args: argparse.Namespace,
    features: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
        from train_baseline_mlp import (
            TwoLayerMLP,
            predict_probabilities,
            set_seed,
            split_train_validation,
            train_model,
        )
    except ImportError as exc:
        raise ImportError(
            "PyTorch MLP dependencies are not available. Install CPU torch with "
            "`pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir`."
        ) from exc

    if args.mlp_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")

    set_seed(args.random_state)
    device = torch.device(args.mlp_device)
    train_fit_df, validation_df = split_train_validation(train_df, args.mlp_validation_fraction)
    training_args = argparse.Namespace(
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        batch_size=args.mlp_batch_size,
        epochs=args.mlp_epochs,
        patience=args.mlp_patience,
        random_state=args.random_state,
    )

    preprocessor = build_preprocessor(features, scale=True)
    train_fit_x = preprocessor.fit_transform(train_fit_df[features]).astype("float32")
    validation_x = preprocessor.transform(validation_df[features]).astype("float32")
    test_x = preprocessor.transform(test_df[features]).astype("float32")

    model = TwoLayerMLP(
        input_dim=len(features),
        hidden_dim_1=args.mlp_hidden_dim_1,
        hidden_dim_2=args.mlp_hidden_dim_2,
        dropout=args.mlp_dropout,
    ).to(device)
    training_summary = train_model(
        model,
        train_fit_x,
        train_fit_df[LABEL_COLUMN].to_numpy(dtype="float32"),
        validation_x,
        validation_df[LABEL_COLUMN].to_numpy(dtype="float32"),
        training_args,
        device,
    )
    probabilities = predict_probabilities(model, test_x, device, args.mlp_batch_size)
    return probabilities, {**mlp_params(args), **training_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanding-window model backtests.")
    parser.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    parser.add_argument("--table", default="model_games")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for artifacts. Defaults to "
            "models/artifacts/backtest_<model>."
        ),
    )
    parser.add_argument("--model", choices=("logistic", "xgboost", "mlp"), default="logistic")
    parser.add_argument("--min-games-before", type=int, default=10)
    parser.add_argument(
        "--initial-train-end",
        default="2023-12-31",
        help="Train on games up to and including this date before the first test window.",
    )
    parser.add_argument(
        "--window",
        choices=("month",),
        default="month",
        help="Backtest window size.",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--xgb-n-estimators", type=int, default=200)
    parser.add_argument("--xgb-max-depth", type=int, default=2)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--mlp-hidden-dim-1", type=int, default=64)
    parser.add_argument("--mlp-hidden-dim-2", type=int, default=32)
    parser.add_argument("--mlp-dropout", type=float, default=0.10)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.001)
    parser.add_argument("--mlp-weight-decay", type=float, default=0.001)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-validation-fraction", type=float, default=0.15)
    parser.add_argument("--mlp-patience", type=int, default=15)
    parser.add_argument("--mlp-device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"backtest_{args.model}"
    df = load_table(args.features_db, args.table)
    df, features = prepare_model_frame(df, args.table, args.min_games_before)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    initial_train_end = pd.Timestamp(args.initial_train_end)
    windows = [
        start
        for start in month_starts(df.loc[df["game_date"] > initial_train_end, "game_date"])
        if start > initial_train_end
    ]
    if not windows:
        raise ValueError("No backtest windows found after initial_train_end.")

    all_predictions = []
    window_metrics = []
    model_params: dict[str, Any] | None = None

    for window_start in windows:
        window_end = window_start + pd.offsets.MonthBegin(1)
        train_df = df[df["game_date"] < window_start].copy()
        test_df = df[(df["game_date"] >= window_start) & (df["game_date"] < window_end)].copy()
        if train_df.empty or test_df.empty:
            continue

        if args.model == "mlp":
            probabilities, params = fit_predict_mlp(args, features, train_df, test_df)
        else:
            classifier, scale, params = make_classifier(args)
            model = Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(features, scale=scale)),
                    ("classifier", classifier),
                ]
            )
            model.fit(train_df[features], train_df[LABEL_COLUMN])
            probabilities = model.predict_proba(test_df[features])[:, 1]
        model_params = params
        metrics = evaluate(test_df[LABEL_COLUMN], probabilities)
        if args.model == "mlp":
            metrics.update(
                {
                    "mlp_epochs_completed": params["epochs_completed"],
                    "mlp_best_epoch": params["best_epoch"],
                    "mlp_best_validation_loss": params["best_validation_loss"],
                }
            )
        metrics.update(
            {
                "window_start": window_start.date().isoformat(),
                "window_end": (window_end - pd.Timedelta(days=1)).date().isoformat(),
                "train_n": int(len(train_df)),
                "train_start": train_df["game_date"].min().date().isoformat(),
                "train_end": train_df["game_date"].max().date().isoformat(),
            }
        )
        window_metrics.append(metrics)

        predictions = test_df[
            ["game_id", "game_date", "home_team_id", "away_team_id", LABEL_COLUMN]
        ].copy()
        predictions["pred_home_win_prob"] = probabilities
        predictions["window_start"] = window_start.date().isoformat()
        all_predictions.append(predictions)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    overall_metrics = evaluate(predictions_df[LABEL_COLUMN], predictions_df["pred_home_win_prob"])

    config = {
        "features_db": str(args.features_db),
        "table": args.table,
        "model": args.model,
        "min_games_before": args.min_games_before,
        "initial_train_end": args.initial_train_end,
        "window": args.window,
        "feature_columns": features,
        "model_params": model_params,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall_metrics, indent=2) + "\n"
    )
    pd.DataFrame(window_metrics).to_csv(output_dir / "window_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    print(json.dumps(overall_metrics, indent=2))
    print(f"Saved backtest artifacts to {output_dir}")


if __name__ == "__main__":
    main()
