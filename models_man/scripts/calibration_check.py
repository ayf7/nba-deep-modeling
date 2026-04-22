#!/usr/bin/env python3
"""Calibration analysis on a predictions.csv.

Two views:
  1. Pure calibration: bucketed predicted vs actual home-win rate. Reports ECE,
     MCE, Brier, accuracy; plots a reliability diagram.
  2. Market EV: when game_moneyline_odds is joinable, computes realized return
     of a flat-$1 strategy that bets the side the model favors over the market,
     swept across edge thresholds. Reports edge needed before flat betting is
     break-even and at typical -110.

Defaults to the v3 predictions but works on any predictions.csv with columns
game_id, label_home_win, pred_home_win_prob.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORE_DB = PROJECT_ROOT / "data" / "artifacts" / "nba_core.sqlite"
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "models_man" / "artifacts" / "man_v3_t24jan_v24mar" / "predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--edge-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
        help="Edge thresholds to sweep for the market-EV analysis (pred − implied).",
    )
    return parser.parse_args()


def load_odds(core_db: Path) -> pd.DataFrame:
    with sqlite3.connect(core_db) as conn:
        odds = pd.read_sql_query(
            """
            SELECT game_id,
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


def calibration_table(probs: np.ndarray, labels: np.ndarray, bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin_lo": edges[b],
                    "bin_hi": edges[b + 1],
                    "n": 0,
                    "mean_pred": np.nan,
                    "actual_rate": np.nan,
                    "abs_error": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": n,
                "mean_pred": float(probs[mask].mean()),
                "actual_rate": float(labels[mask].mean()),
                "abs_error": float(abs(probs[mask].mean() - labels[mask].mean())),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, table: pd.DataFrame) -> dict:
    total = len(probs)
    weighted = (table["n"] / total) * table["abs_error"].fillna(0.0)
    ece = float(weighted.sum())
    mce = float(table["abs_error"].max(skipna=True))
    brier = float(np.mean((probs - labels) ** 2))
    accuracy = float(((probs >= 0.5).astype(int) == labels).mean())
    base_rate = float(labels.mean())
    return {
        "n": int(total),
        "base_rate_home_win": base_rate,
        "accuracy": accuracy,
        "brier": brier,
        "ece": ece,
        "mce": mce,
    }


def plot_calibration(
    table: pd.DataFrame, metrics: dict, label: str, output: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.5, label="perfect")
    valid = table[table["n"] > 0]
    sizes = 8 + (valid["n"] / valid["n"].max()) * 220
    ax.scatter(
        valid["mean_pred"],
        valid["actual_rate"],
        s=sizes,
        alpha=0.7,
        color="#2ca02c",
        edgecolors="black",
        linewidths=0.5,
        label="bucket (size ∝ n)",
    )
    ax.plot(valid["mean_pred"], valid["actual_rate"], color="#2ca02c", alpha=0.5, linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted P(home win)")
    ax.set_ylabel("Empirical home-win rate")
    ax.set_title(
        f"{label}\n"
        f"n={metrics['n']}  brier={metrics['brier']:.4f}  "
        f"ece={metrics['ece']:.4f}  mce={metrics['mce']:.4f}  "
        f"acc={metrics['accuracy']:.4f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def market_ev_sweep(
    df: pd.DataFrame, edge_thresholds: list[float]
) -> pd.DataFrame:
    """For each edge threshold, bet the side the model favors when |pred − implied| ≥ thresh.

    Profit per bet (home): home_won * (home_decimal − 1) − (1 − home_won) * 1
    Profit per bet (away): (1 − home_won) * (away_decimal − 1) − home_won * 1
    """
    rows = []
    pred = df["pred_home_win_prob"].to_numpy()
    implied = df["home_implied_prob_normalized"].to_numpy()
    label = df["label_home_win"].to_numpy()
    home_decimal = df["home_avg_decimal_odds"].to_numpy()
    away_decimal = df["away_avg_decimal_odds"].to_numpy()
    edge = pred - implied  # signed; positive = bet home, negative = bet away

    for thresh in edge_thresholds:
        bet_home = edge >= thresh
        bet_away = edge <= -thresh
        n_bets = int(bet_home.sum() + bet_away.sum())
        if n_bets == 0:
            rows.append(
                {
                    "edge_threshold": thresh,
                    "n_bets": 0,
                    "n_home_bets": 0,
                    "n_away_bets": 0,
                    "hit_rate": np.nan,
                    "avg_decimal_odds": np.nan,
                    "total_profit": 0.0,
                    "roi_per_bet": np.nan,
                }
            )
            continue
        home_profit = label * (home_decimal - 1) - (1 - label) * 1
        away_profit = (1 - label) * (away_decimal - 1) - label * 1
        profit = np.where(bet_home, home_profit, np.where(bet_away, away_profit, 0.0))
        total = float(profit[bet_home | bet_away].sum())
        hits = int(((bet_home & (label == 1)) | (bet_away & (label == 0))).sum())
        avg_odds = float(
            np.concatenate([home_decimal[bet_home], away_decimal[bet_away]]).mean()
        )
        rows.append(
            {
                "edge_threshold": thresh,
                "n_bets": n_bets,
                "n_home_bets": int(bet_home.sum()),
                "n_away_bets": int(bet_away.sum()),
                "hit_rate": hits / n_bets,
                "avg_decimal_odds": avg_odds,
                "total_profit": total,
                "roi_per_bet": total / n_bets,
            }
        )
    return pd.DataFrame(rows)


def market_calibration_table(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Per pred-prob bucket: mean implied, mean pred, residual, n."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    pred = df["pred_home_win_prob"].to_numpy()
    bin_idx = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": n,
                "mean_pred": float(df["pred_home_win_prob"].to_numpy()[mask].mean()),
                "mean_implied": float(
                    df["home_implied_prob_normalized"].to_numpy()[mask].mean()
                ),
                "actual_rate": float(df["label_home_win"].to_numpy()[mask].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pred_minus_implied"] = out["mean_pred"] - out["mean_implied"]
        out["actual_minus_implied"] = out["actual_rate"] - out["mean_implied"]
    return out


def fmt_table(df: pd.DataFrame, float_cols: list[str], int_cols: list[str]) -> str:
    formatted = df.copy()
    for c in float_cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    for c in int_cols:
        if c in formatted.columns:
            formatted[c] = formatted[c].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    return formatted.to_string(index=False)


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions not found: {args.predictions}")

    output_dir = args.output_dir or args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.predictions.parent.name

    preds = pd.read_csv(args.predictions)
    preds["game_id"] = preds["game_id"].astype(str)
    needed = {"game_id", "label_home_win", "pred_home_win_prob"}
    missing = needed - set(preds.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {sorted(missing)}")

    probs = preds["pred_home_win_prob"].to_numpy()
    labels = preds["label_home_win"].to_numpy().astype(int)

    cal = calibration_table(probs, labels, args.bins)
    metrics = calibration_metrics(probs, labels, cal)

    print(f"=== Calibration: {label} ===")
    print(f"n={metrics['n']}  base_rate(home win)={metrics['base_rate_home_win']:.4f}")
    print(
        f"accuracy={metrics['accuracy']:.4f}  brier={metrics['brier']:.4f}  "
        f"ECE={metrics['ece']:.4f}  MCE={metrics['mce']:.4f}"
    )
    print()
    print(
        fmt_table(
            cal,
            float_cols=["bin_lo", "bin_hi", "mean_pred", "actual_rate", "abs_error"],
            int_cols=["n"],
        )
    )

    cal.to_csv(output_dir / "calibration.csv", index=False)
    plot_calibration(cal, metrics, label, output_dir / "calibration.png")
    (output_dir / "calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    if not args.core_db.exists():
        print(f"\n(no core DB at {args.core_db}, skipping market-EV section)")
        return

    odds = load_odds(args.core_db)
    merged = preds.merge(odds, on="game_id", how="inner").dropna(
        subset=[
            "home_implied_prob_normalized",
            "home_avg_decimal_odds",
            "away_avg_decimal_odds",
        ]
    )
    if merged.empty:
        print("\n(no market odds joinable, skipping market-EV section)")
        return

    print(f"\n=== Market alignment: {len(merged)} of {len(preds)} games have odds ===")
    market_cal = market_calibration_table(merged, args.bins)
    print(
        fmt_table(
            market_cal,
            float_cols=[
                "bin_lo",
                "bin_hi",
                "mean_pred",
                "mean_implied",
                "actual_rate",
                "pred_minus_implied",
                "actual_minus_implied",
            ],
            int_cols=["n"],
        )
    )
    market_cal.to_csv(output_dir / "market_calibration.csv", index=False)

    sweep = market_ev_sweep(merged, args.edge_thresholds)
    print("\n=== Flat-$1 EV sweep (bet side model favors over market) ===")
    print(
        fmt_table(
            sweep,
            float_cols=[
                "edge_threshold",
                "hit_rate",
                "avg_decimal_odds",
                "total_profit",
                "roi_per_bet",
            ],
            int_cols=["n_bets", "n_home_bets", "n_away_bets"],
        )
    )
    sweep.to_csv(output_dir / "market_ev_sweep.csv", index=False)

    print(f"\n[saved] {output_dir}/calibration.{{csv,png}}")
    print(f"[saved] {output_dir}/market_calibration.csv")
    print(f"[saved] {output_dir}/market_ev_sweep.csv")


if __name__ == "__main__":
    main()
