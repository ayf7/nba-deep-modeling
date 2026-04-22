#!/usr/bin/env python3
"""Train the logistic-regression baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from modeling_common import DEFAULT_FEATURES_DB, DEFAULT_OUTPUT_ROOT, fit_and_write_run


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "baseline_logistic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic-regression baseline.")
    parser.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    parser.add_argument("--table", default="model_games")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--min-games-before", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = {"max_iter": args.max_iter}
    fit_and_write_run(
        model_name="baseline_logistic_regression",
        classifier=LogisticRegression(**params),
        features_db=args.features_db,
        table=args.table,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        min_games_before=args.min_games_before,
        scale=True,
        model_params=params,
    )


if __name__ == "__main__":
    main()
