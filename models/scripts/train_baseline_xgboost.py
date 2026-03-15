#!/usr/bin/env python3
"""Train the XGBoost baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from modeling_common import DEFAULT_FEATURES_DB, DEFAULT_OUTPUT_ROOT, fit_and_write_run


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "baseline_xgboost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost baseline.")
    parser.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    parser.add_argument("--table", default="model_games")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--min-games-before", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Install it with "
            "`/home/ayf7/trading/env/bin/pip install xgboost` or update the conda env."
        ) from exc

    args = parse_args()
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": args.random_state,
    }
    fit_and_write_run(
        model_name="baseline_xgboost",
        classifier=XGBClassifier(**params),
        features_db=args.features_db,
        table=args.table,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        min_games_before=args.min_games_before,
        scale=False,
        model_params=params,
    )


if __name__ == "__main__":
    main()
