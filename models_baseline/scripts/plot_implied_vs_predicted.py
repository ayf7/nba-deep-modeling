#!/usr/bin/env python3
"""Scatter plot: market-implied home-win probability vs model-predicted home-win probability.

For each predictions CSV produced by `backtest_baselines.py`, join to
`game_moneyline_odds` and plot one panel showing how predicted probabilities
align with vig-removed implied probabilities.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modeling_common import DEFAULT_OUTPUT_ROOT, PROJECT_ROOT


DEFAULT_CORE_DB = PROJECT_ROOT / "data" / "artifacts" / "nba_core.sqlite"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "implied_vs_predicted.png"

DEFAULT_PREDICTIONS = [
    DEFAULT_OUTPUT_ROOT / "backtest_xgboost_t24feb_baseline" / "predictions.csv",
    DEFAULT_OUTPUT_ROOT / "backtest_xgboost_t24feb_v1" / "predictions.csv",
    DEFAULT_OUTPUT_ROOT / "backtest_xgboost_t24feb_v2" / "predictions.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=DEFAULT_PREDICTIONS,
        help="One or more predictions.csv paths.",
    )
    parser.add_argument(
        "--label",
        nargs="+",
        default=None,
        help="Display label per predictions file (defaults to parent directory name).",
    )
    parser.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Scatter point transparency.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=14.0,
    )
    parser.add_argument(
        "--joined-csv",
        type=Path,
        default=None,
        help="Optional path to write the merged predictions+odds dataframe.",
    )
    return parser.parse_args()


def load_odds(core_db: Path) -> pd.DataFrame:
    with sqlite3.connect(core_db) as conn:
        odds = pd.read_sql_query(
            """
            SELECT
                game_id,
                home_implied_prob_normalized,
                away_implied_prob_normalized,
                home_avg_decimal_odds,
                away_avg_decimal_odds,
                has_moneyline_odds
            FROM game_moneyline_odds
            WHERE has_moneyline_odds = 1
            """,
            conn,
        )
    odds["game_id"] = odds["game_id"].astype(str)
    return odds


def load_and_join(predictions_path: Path, odds: pd.DataFrame) -> pd.DataFrame:
    preds = pd.read_csv(predictions_path)
    preds["game_id"] = preds["game_id"].astype(str)
    merged = preds.merge(odds, on="game_id", how="inner")
    merged = merged.dropna(
        subset=["home_implied_prob_normalized", "pred_home_win_prob", "label_home_win"]
    ).reset_index(drop=True)
    return merged


def metric_summary(df: pd.DataFrame) -> dict:
    x = df["home_implied_prob_normalized"].to_numpy()
    y = df["pred_home_win_prob"].to_numpy()
    residual = y - x
    return {
        "n": int(len(df)),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]) if len(df) > 1 else float("nan"),
        "mean_residual": float(residual.mean()),
        "std_residual": float(residual.std(ddof=0)),
        "mean_abs_residual": float(np.abs(residual).mean()),
    }


def panel_title(label: str, summary: dict) -> str:
    return (
        f"{label}\n"
        f"n={summary['n']}  "
        f"r={summary['pearson_r']:.3f}  "
        f"mean(pred-implied)={summary['mean_residual']:+.3f}  "
        f"MAE={summary['mean_abs_residual']:.3f}"
    )


def grid_shape(n: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    return rows, cols


def plot_panels(
    frames: list[tuple[str, pd.DataFrame]],
    output: Path,
    alpha: float,
    point_size: float,
    max_cols: int = 3,
) -> None:
    n = len(frames)
    rows, cols = grid_shape(n, max_cols=max_cols)
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(5.0 * cols, 5.0 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, (label, df) in zip(flat_axes, frames):
        summary = metric_summary(df)

        wins = df[df["label_home_win"] == 1]
        losses = df[df["label_home_win"] == 0]
        ax.scatter(
            losses["home_implied_prob_normalized"],
            losses["pred_home_win_prob"],
            s=point_size,
            alpha=alpha,
            color="#d62728",
            label="home lost",
            edgecolors="none",
        )
        ax.scatter(
            wins["home_implied_prob_normalized"],
            wins["pred_home_win_prob"],
            s=point_size,
            alpha=alpha,
            color="#2ca02c",
            label="home won",
            edgecolors="none",
        )
        ax.plot([0, 1], [0, 1], color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.axhline(0.5, color="grey", linewidth=0.6, alpha=0.4)
        ax.axvline(0.5, color="grey", linewidth=0.6, alpha=0.4)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Market implied prob (vig-removed, home)")
        ax.set_title(panel_title(label, summary), fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.3)

    for ax in flat_axes[len(frames):]:
        ax.set_visible(False)

    for r in range(rows):
        flat_axes[r * cols].set_ylabel("Model predicted prob (home win)")
    flat_axes[0].legend(loc="upper left", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def derive_labels(paths: list[Path], explicit: list[str] | None) -> list[str]:
    if explicit is None:
        return [p.parent.name for p in paths]
    if len(explicit) != len(paths):
        raise ValueError(
            f"Got {len(explicit)} labels for {len(paths)} predictions files."
        )
    return explicit


def main() -> None:
    args = parse_args()

    if not args.core_db.exists():
        raise FileNotFoundError(f"Core DB not found: {args.core_db}")

    labels = derive_labels(args.predictions, args.label)
    odds = load_odds(args.core_db)

    frames: list[tuple[str, pd.DataFrame]] = []
    merged_rows: list[pd.DataFrame] = []
    for label, path in zip(labels, args.predictions):
        if not path.exists():
            raise FileNotFoundError(f"Predictions not found: {path}")
        df = load_and_join(path, odds)
        if df.empty:
            raise ValueError(f"No rows after joining odds for {path}")
        frames.append((label, df))

        summary = metric_summary(df)
        print(
            f"{label}: n={summary['n']} "
            f"pearson_r={summary['pearson_r']:.4f} "
            f"mean_residual={summary['mean_residual']:+.4f} "
            f"mae={summary['mean_abs_residual']:.4f}"
        )

        if args.joined_csv is not None:
            tagged = df.copy()
            tagged.insert(0, "model", label)
            merged_rows.append(tagged)

    plot_panels(frames, args.output, alpha=args.alpha, point_size=args.point_size)
    print(f"Wrote {args.output}")

    if args.joined_csv is not None and merged_rows:
        out = pd.concat(merged_rows, ignore_index=True)
        args.joined_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.joined_csv, index=False)
        print(f"Wrote {args.joined_csv}")


if __name__ == "__main__":
    main()
