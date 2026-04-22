#!/usr/bin/env python3
"""Plot Lakers win prob per March 2026 game across pair counterfactual.

Compares baseline vs Luka-OUT, LeBron-OUT, Luka+LeBron-OUT.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts" / "multi_player_cf_2026_03_pair"
CSV_PATH = ART / "lakers_march_multi_player.csv"
OUT_PATH = ART / "lakers_march_pair_bars.png"

PROB_BARS = [
    ("baseline",          "Baseline",               "#1f77b4"),
    ("luka_out",          "Luka OUT",               "#d62728"),
    ("lebron_out",        "LeBron OUT",             "#9467bd"),
    ("luka_and_lebron_out", "Luka + LeBron OUT",    "#8b0000"),
]
DELTA_BARS = [
    ("luka",             "Luka",                "#d62728"),
    ("lebron",           "LeBron",              "#9467bd"),
    ("luka_and_lebron",  "Luka + LeBron",       "#8b0000"),
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
    n = len(PROB_BARS)
    x = np.arange(len(df))
    width = 0.8 / n
    for i, (key, label, color) in enumerate(PROB_BARS):
        col = "p_lakers_baseline" if key == "baseline" else f"p_lakers_{key}"
        offset = (i - (n - 1) / 2) * width
        ax1.bar(x + offset, df[col], width, label=label, color=color)
    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("P(Lakers win)")
    ax1.set_ylim(0, 1)
    ax1.set_title(
        "v5 win-prob per Lakers March 2026 game — baseline vs Luka / LeBron / both forced OUT"
    )
    ax1.legend(loc="lower right", ncol=4, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.3)

    # ---- Bottom panel: deltas (incl additive reference) ----
    nd = len(DELTA_BARS)
    dwidth = 0.8 / nd
    for i, (key, label, color) in enumerate(DELTA_BARS):
        offset = (i - (nd - 1) / 2) * dwidth
        deltas = df[f"delta_{key}"]
        ax2.bar(x + offset, deltas, dwidth, label=label, color=color)

    # Additive reference: Luka + LeBron solo (sum), plotted as a black "x" marker per game
    additive = df["delta_luka"] + df["delta_lebron"]
    # Place markers where the pair bar would be (rightmost group)
    pair_offset = ((nd - 1) - (nd - 1) / 2) * dwidth
    ax2.scatter(x + pair_offset, additive, marker="x", color="black", s=40,
                zorder=5, label="Additive (Luka+LeBron solo)")

    ax2.axhline(0.0, color="black", linewidth=0.6)
    ax2.set_ylabel("Δ P(Lakers win)\n(player OUT − baseline)")

    summary_parts = []
    for key, label, _ in DELTA_BARS:
        m = df[f"delta_{key}"].mean()
        summary_parts.append(f"{label}: μ={m:+.3f}")
    add_mean = additive.mean()
    joint_mean = df["delta_luka_and_lebron"].mean()
    summary_parts.append(f"Additive expected: μ={add_mean:+.3f}  →  excess vs additive: {joint_mean - add_mean:+.3f}")
    ax2.set_title(" │ ".join(summary_parts), fontsize=10)
    ax2.legend(loc="lower right", ncol=4, framealpha=0.95)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
