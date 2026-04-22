#!/usr/bin/env python3
"""Two diagnostic plots from a predictions.csv with home/away scores.

A. score_scatter.png:
   X = actual points, Y = predicted points. One blue dot per game for the
   home team, one orange dot per game for the away team. Diagonal = perfect
   prediction.

B. score_field.png:
   X = home points, Y = away points. For each game, a faint segment from
   (actual_h, actual_a) to (pred_h, pred_a). The y=x line splits the plane
   into home-win (below) and home-loss (above) regions; arrows that cross
   it represent sign-flipped winner predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


REQUIRED = [
    "pred_home_pts", "pred_away_pts",
    "actual_home_pts", "actual_away_pts",
    "label_home_win",
    "margin_mu", "actual_margin",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True,
                   help="Predictions CSV with pred/actual home/away pts.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write score_scatter.png and score_field.png.")
    p.add_argument("--label", type=str, default=None,
                   help="Title prefix (default: predictions parent dir name).")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--point-size", type=float, default=12.0)
    return p.parse_args()


def plot_scatter(df: pd.DataFrame, label: str, out: Path, alpha: float, ps: float) -> None:
    h_a = df["actual_home_pts"].to_numpy()
    h_p = df["pred_home_pts"].to_numpy()
    a_a = df["actual_away_pts"].to_numpy()
    a_p = df["pred_away_pts"].to_numpy()

    home_mae = float(np.mean(np.abs(h_p - h_a)))
    away_mae = float(np.mean(np.abs(a_p - a_a)))
    home_bias = float(np.mean(h_p - h_a))
    away_bias = float(np.mean(a_p - a_a))
    home_r = float(np.corrcoef(h_a, h_p)[0, 1])
    away_r = float(np.corrcoef(a_a, a_p)[0, 1])

    lo = float(min(h_a.min(), a_a.min(), h_p.min(), a_p.min())) - 5
    hi = float(max(h_a.max(), a_a.max(), h_p.max(), a_p.max())) + 5

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(h_a, h_p, s=ps, alpha=alpha, color="#1f77b4",
               label=f"home  (r={home_r:.3f}, MAE={home_mae:.2f}, bias={home_bias:+.2f})",
               edgecolors="none")
    ax.scatter(a_a, a_p, s=ps, alpha=alpha, color="#ff7f0e",
               label=f"away  (r={away_r:.3f}, MAE={away_mae:.2f}, bias={away_bias:+.2f})",
               edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", alpha=0.6,
            label="perfect")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual points")
    ax.set_ylabel("Predicted points")
    ax.set_title(f"{label}: predicted vs actual points  (n={len(df)})")
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_field(df: pd.DataFrame, label: str, out: Path, alpha: float) -> None:
    h_a = df["actual_home_pts"].to_numpy()
    h_p = df["pred_home_pts"].to_numpy()
    a_a = df["actual_away_pts"].to_numpy()
    a_p = df["pred_away_pts"].to_numpy()
    win = df["label_home_win"].to_numpy().astype(bool)

    # Each game: segment (actual_h, actual_a) -> (pred_h, pred_a) in (home_pts, away_pts) plane.
    starts = np.stack([h_a, a_a], axis=1)
    ends = np.stack([h_p, a_p], axis=1)
    segs = np.stack([starts, ends], axis=1)  # (N, 2, 2)

    # Sign-flip: actual margin > 0 (home won) but pred margin <= 0 (model picks away), or vice versa.
    actual_margin = h_a - a_a
    pred_margin = h_p - a_p
    flipped = np.sign(actual_margin) != np.sign(pred_margin)

    lo = float(min(h_a.min(), a_a.min(), h_p.min(), a_p.min())) - 5
    hi = float(max(h_a.max(), a_a.max(), h_p.max(), a_p.max())) + 5

    fig, ax = plt.subplots(figsize=(7.5, 7))
    # Color segments by whether the prediction sign-flips the winner.
    colors = np.where(flipped, "#d62728", "#7f7f7f")  # red if flipped, grey otherwise
    lc = LineCollection(segs, colors=colors, alpha=alpha, linewidths=0.8)
    ax.add_collection(lc)

    # Overlay: small dot at actual (truth), tiny dot at pred (arrowhead substitute).
    ax.scatter(h_a, a_a, s=6, color="#2ca02c", alpha=0.55,
               edgecolors="none", label="actual")
    ax.scatter(h_p, a_p, s=4, color="#1f77b4", alpha=0.40,
               edgecolors="none", label="predicted")

    # Diagonals: y=x (even game) and a few constant-margin lines for orientation.
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0,
            alpha=0.7, label="home pts = away pts")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Home points")
    ax.set_ylabel("Away points")

    flip_rate = float(flipped.mean())
    ax.set_title(
        f"{label}: actual → predicted score per game  "
        f"(n={len(df)}, sign-flipped winner: {flip_rate:.1%})"
    )
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_margin(df: pd.DataFrame, label: str, out: Path, alpha: float, ps: float) -> None:
    """Predicted margin vs actual margin. One dot per game. Diagonal = perfect.
    Off-diagonal axes (top-left, bottom-right quadrants) = sign-flipped winner.
    """
    actual = df["actual_margin"].to_numpy()
    pred = df["margin_mu"].to_numpy()

    r = float(np.corrcoef(actual, pred)[0, 1])
    mae = float(np.mean(np.abs(pred - actual)))
    bias = float(np.mean(pred - actual))
    pred_std = float(np.std(pred))
    actual_std = float(np.std(actual))

    flipped = np.sign(actual) != np.sign(pred)
    flip_rate = float(flipped.mean())

    lo = float(min(actual.min(), pred.min())) - 5
    hi = float(max(actual.max(), pred.max())) + 5
    rng = max(abs(lo), abs(hi))
    lo, hi = -rng, rng

    fig, ax = plt.subplots(figsize=(7.5, 7))
    # Quadrant shading: light red where the winner sign would flip.
    ax.axhspan(0, hi, xmin=0, xmax=0.5, facecolor="#fbecec", alpha=0.5, zorder=0)
    ax.axhspan(lo, 0, xmin=0.5, xmax=1.0, facecolor="#fbecec", alpha=0.5, zorder=0)

    colors = np.where(flipped, "#d62728", "#1f77b4")
    ax.scatter(actual, pred, s=ps, alpha=alpha, c=colors, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--",
            alpha=0.7, label="perfect")
    ax.axhline(0, color="grey", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.6, alpha=0.5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual margin (home − away)")
    ax.set_ylabel("Predicted margin (margin_mu)")
    ax.set_title(
        f"{label}: predicted vs actual margin  (n={len(df)})\n"
        f"r={r:.3f}  MAE={mae:.2f}  bias={bias:+.2f}  "
        f"std(pred)={pred_std:.2f} vs std(actual)={actual_std:.2f}  "
        f"sign-flip={flip_rate:.1%}"
    )
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"predictions CSV missing required columns: {missing}")

    label = args.label or args.predictions.parent.name

    out_a = args.output_dir / "score_scatter.png"
    out_b = args.output_dir / "score_field.png"
    out_c = args.output_dir / "margin_scatter.png"

    plot_scatter(df, label, out_a, alpha=args.alpha, ps=args.point_size)
    print(f"Wrote {out_a}")

    plot_field(df, label, out_b, alpha=args.alpha)
    print(f"Wrote {out_b}")

    plot_margin(df, label, out_c, alpha=args.alpha, ps=args.point_size)
    print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()
