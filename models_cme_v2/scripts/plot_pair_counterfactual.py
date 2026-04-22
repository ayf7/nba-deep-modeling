#!/usr/bin/env python3
"""Plot CME-v2 Lakers March 2026 pair counterfactual results.

Compares baseline vs Luka-OUT, LeBron-OUT, and Luka+LeBron-OUT. The top two
panels mirror the v5 pair counterfactual plot; the bottom panel adds the
CME-v2 predicted Lakers points delta.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ART = Path(__file__).resolve().parents[1] / "artifacts" / "multi_player_cf_2026_03_pair"
DEFAULT_CSV = ART / "lakers_march_multi_player.csv"
DEFAULT_OUT = ART / "lakers_march_pair_bars.png"

PROB_BARS = [
    ("baseline", "Baseline", "#1f77b4"),
    ("luka_out", "Luka OUT", "#d62728"),
    ("lebron_out", "LeBron OUT", "#9467bd"),
    ("luka_and_lebron_out", "Luka + LeBron OUT", "#8b0000"),
]
DELTA_BARS = [
    ("luka", "Luka", "#d62728"),
    ("lebron", "LeBron", "#9467bd"),
    ("luka_and_lebron", "Luka + LeBron", "#8b0000"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    labels = [
        f"{d.strftime('%m-%d')} {'vs' if h == 1 else '@'} {opp}"
        for d, h, opp in zip(df["game_date"], df["lakers_home"], df["opponent_abbr"])
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(15, 11), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )

    x = np.arange(len(df))

    # Top panel: per-game win probabilities.
    n = len(PROB_BARS)
    width = 0.8 / n
    for i, (key, label, color) in enumerate(PROB_BARS):
        col = "p_lakers_baseline" if key == "baseline" else f"p_lakers_{key}"
        offset = (i - (n - 1) / 2) * width
        ax1.bar(x + offset, df[col], width, label=label, color=color)
    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("P(Lakers win)")
    ax1.set_ylim(0, 1)
    ax1.set_title(
        "CME-v2 Lakers March 2026 counterfactual: baseline vs Luka / LeBron / both OUT"
    )
    ax1.legend(loc="lower right", ncol=4, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.3)

    # Middle panel: win-probability deltas.
    nd = len(DELTA_BARS)
    dwidth = 0.8 / nd
    for i, (key, label, color) in enumerate(DELTA_BARS):
        offset = (i - (nd - 1) / 2) * dwidth
        ax2.bar(x + offset, df[f"delta_{key}"], dwidth, label=label, color=color)
    additive_prob = df["delta_luka"] + df["delta_lebron"]
    pair_offset = ((nd - 1) - (nd - 1) / 2) * dwidth
    ax2.scatter(
        x + pair_offset, additive_prob, marker="x", color="black", s=40,
        zorder=5, label="Additive (solo sum)",
    )
    ax2.axhline(0.0, color="black", linewidth=0.6)
    ax2.set_ylabel("Delta P(Lakers win)\n(player OUT - baseline)")
    prob_summary = []
    for key, label, _ in DELTA_BARS:
        prob_summary.append(f"{label}: mean={df[f'delta_{key}'].mean():+.3f}")
    prob_summary.append(
        f"Additive mean={additive_prob.mean():+.3f}; "
        f"joint excess={df['delta_luka_and_lebron'].mean() - additive_prob.mean():+.3f}"
    )
    ax2.set_title(" | ".join(prob_summary), fontsize=10)
    ax2.legend(loc="lower right", ncol=4, framealpha=0.95)
    ax2.grid(axis="y", alpha=0.3)

    # Bottom panel: predicted Lakers points deltas.
    for i, (key, label, color) in enumerate(DELTA_BARS):
        offset = (i - (nd - 1) / 2) * dwidth
        ax3.bar(
            x + offset, df[f"delta_lakers_pts_{key}"],
            dwidth, label=label, color=color,
        )
    additive_pts = df["delta_lakers_pts_luka"] + df["delta_lakers_pts_lebron"]
    ax3.scatter(
        x + pair_offset, additive_pts, marker="x", color="black", s=40,
        zorder=5, label="Additive (solo sum)",
    )
    ax3.axhline(0.0, color="black", linewidth=0.6)
    ax3.set_ylabel("Delta Lakers predicted pts\n(player OUT - baseline)")
    pts_summary = []
    for key, label, _ in DELTA_BARS:
        pts_summary.append(f"{label}: mean={df[f'delta_lakers_pts_{key}'].mean():+.1f}")
    pts_summary.append(
        f"Additive mean={additive_pts.mean():+.1f}; "
        f"joint excess={df['delta_lakers_pts_luka_and_lebron'].mean() - additive_pts.mean():+.1f}"
    )
    ax3.set_title(" | ".join(pts_summary), fontsize=10)
    ax3.legend(loc="lower right", ncol=4, framealpha=0.95)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
