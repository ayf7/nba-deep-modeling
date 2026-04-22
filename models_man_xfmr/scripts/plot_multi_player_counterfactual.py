#!/usr/bin/env python3
"""Plot Lakers win prob per March 2026 game across multiple counterfactuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts" / "multi_player_cf_2026_03"
CSV_PATH = ART / "lakers_march_multi_player.csv"
OUT_PATH = ART / "lakers_march_multi_player_bars.png"

PLAYER_LABELS = [
    ("baseline",     "Baseline",         "#1f77b4"),
    ("luka_out",     "Luka OUT",         "#d62728"),
    ("lebron_out",   "LeBron OUT",       "#9467bd"),
    ("ayton_out",    "Ayton OUT",        "#2ca02c"),
    ("smart_out",    "Smart OUT",        "#ff7f0e"),
]
DELTA_PLAYERS = [
    ("luka",    "Luka",     "#d62728"),
    ("lebron",  "LeBron",   "#9467bd"),
    ("ayton",   "Ayton",    "#2ca02c"),
    ("smart",   "Smart",    "#ff7f0e"),
]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    labels = [
        f"{d.strftime('%m-%d')} {'vs' if h == 1 else '@'} {opp}"
        for d, h, opp in zip(df["game_date"], df["lakers_home"], df["opponent_abbr"])
    ]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    # ---- Top panel: per-game probabilities ----
    n_bars = len(PLAYER_LABELS)
    x = np.arange(len(df))
    width = 0.8 / n_bars
    for i, (key, label, color) in enumerate(PLAYER_LABELS):
        col = "p_lakers_baseline" if key == "baseline" else f"p_lakers_{key}"
        offset = (i - (n_bars - 1) / 2) * width
        ax1.bar(x + offset, df[col], width, label=label, color=color)

    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("P(Lakers win)")
    ax1.set_ylim(0, 1)
    ax1.set_title(
        "v5 win-prob per Lakers March 2026 game — baseline vs each player forced OUT"
    )
    ax1.legend(loc="lower right", ncol=4, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.3)

    # ---- Bottom panel: per-game deltas ----
    n_dbars = len(DELTA_PLAYERS)
    dwidth = 0.8 / n_dbars
    for i, (key, label, color) in enumerate(DELTA_PLAYERS):
        offset = (i - (n_dbars - 1) / 2) * dwidth
        deltas = df[f"delta_{key}"]
        ax2.bar(x + offset, deltas, dwidth, label=label, color=color)

    ax2.axhline(0.0, color="black", linewidth=0.6)
    ax2.set_ylabel("Δ P(Lakers win)\n(player OUT − baseline)")

    # title with summary stats
    summary = []
    for key, label, _ in DELTA_PLAYERS:
        mean = df[f"delta_{key}"].mean()
        summary.append(f"{label}: μ={mean:+.3f}")
    ax2.set_title(" │ ".join(summary), fontsize=10)
    ax2.legend(loc="lower right", ncol=3, framealpha=0.95)
    ax2.grid(axis="y", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
