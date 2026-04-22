#!/usr/bin/env python3
"""Plot Lakers win prob per March 2026 game: baseline vs starting-5 OUT.

Compares observed joint Δ against the additive expectation (sum of solos).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts" / "multi_player_cf_2026_03_starting5"
CSV_PATH = ART / "lakers_march_multi_player.csv"
OUT_PATH = ART / "lakers_march_starting5_bars.png"

SOLO_NAMES = ["luka", "lebron", "ayton", "smart", "rui"]
SOLO_LABEL = "Sum of solos (additive expected)"
JOINT_LABEL = "Starting 5 OUT (observed joint)"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    labels = [
        f"{d.strftime('%m-%d')} {'vs' if h == 1 else '@'} {opp}"
        for d, h, opp in zip(df["game_date"], df["lakers_home"], df["opponent_abbr"])
    ]

    additive = sum(df[f"delta_{n}"] for n in SOLO_NAMES)
    joint = df["delta_starting5"]
    excess = joint - additive

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 3]}
    )

    # ---- Top: per-game probability ----
    x = np.arange(len(df))
    width = 0.4
    ax1.bar(x - width / 2, df["p_lakers_baseline"], width,
            label="Baseline", color="#1f77b4")
    ax1.bar(x + width / 2, df["p_lakers_starting5_out"], width,
            label="Starting 5 OUT (Luka, LeBron, Ayton, Smart, Rui)",
            color="#5c0000")
    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("P(Lakers win)")
    ax1.set_ylim(0, 1)
    base_mean = df["p_lakers_baseline"].mean()
    s5_mean = df["p_lakers_starting5_out"].mean()
    ax1.set_title(
        f"v5 win-prob per Lakers March 2026 game — baseline vs starting 5 OUT  "
        f"(mean: {base_mean:.3f} → {s5_mean:.3f})"
    )
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.3)

    # ---- Bottom: deltas (additive vs observed joint) ----
    ax2.bar(x - width / 2, additive, width,
            label=SOLO_LABEL, color="#aaaaaa")
    ax2.bar(x + width / 2, joint, width,
            label=JOINT_LABEL, color="#5c0000")

    # Annotate excess (super-additivity) per game
    for i, (a, j) in enumerate(zip(additive, joint)):
        y = min(a, j) - 0.005
        ax2.text(x[i], y, f"{j - a:+.3f}", ha="center", va="top",
                 fontsize=7, color="#000")

    ax2.axhline(0.0, color="black", linewidth=0.6)
    ax2.set_ylabel("Δ P(Lakers win)\n(player(s) OUT − baseline)")
    solo_str = "  ".join(
        f"{n.title()}: {df[f'delta_{n}'].mean():+.3f}" for n in SOLO_NAMES
    )
    ax2.set_title(
        f"Additive: μ={additive.mean():+.3f}    "
        f"Joint: μ={joint.mean():+.3f}    "
        f"Excess (super-add.): μ={excess.mean():+.3f}\n"
        f"Solo means:  {solo_str}",
        fontsize=10,
    )
    ax2.legend(loc="lower right", framealpha=0.95)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
