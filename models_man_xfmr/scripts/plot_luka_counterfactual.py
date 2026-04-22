#!/usr/bin/env python3
"""Bar chart of Lakers win prob per March 2026 game: baseline vs Luka-OUT."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts" / "luka_counterfactual_2026_03"
CSV_PATH = ART / "lakers_march_predictions.csv"
OUT_PATH = ART / "lakers_march_bars.png"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    labels = [
        f"{d.strftime('%m-%d')} {'vs' if h == 1 else '@'} {opp}"
        for d, h, opp in zip(df["game_date"], df["lakers_home"], df["opponent_abbr"])
    ]
    baseline = df["p_lakers_baseline"].to_numpy()
    luka_out = df["p_lakers_luka_out"].to_numpy()

    x = np.arange(len(df))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - width / 2, baseline, width,
                label="Baseline (real statuses)", color="#1f77b4")
    b2 = ax.bar(x + width / 2, luka_out, width,
                label="Counterfactual (Luka OUT)", color="#d62728")

    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("P(Lakers win)")
    ax.set_ylim(0, 1)
    ax.set_title(
        "v5 win-prob per Lakers March 2026 game — baseline vs Luka forced OUT\n"
        f"mean Δ = {(luka_out - baseline).mean():+.3f}    "
        f"(range [{(luka_out - baseline).min():+.3f}, {(luka_out - baseline).max():+.3f}])"
    )
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    # Annotate delta above each pair
    for i, d in enumerate(luka_out - baseline):
        y = max(baseline[i], luka_out[i]) + 0.02
        ax.text(x[i], y, f"{d:+.3f}", ha="center", va="bottom",
                fontsize=7, color="#444")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=140)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
