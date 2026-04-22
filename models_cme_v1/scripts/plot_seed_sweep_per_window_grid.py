#!/usr/bin/env python3
"""Per-window OOS EV heatmap grid for the cme_v1 5-seed backtest.

Rows = seeds (7 / 11 / 23 / 42 / 47), cols = backtest windows.
Same per-cell layout as docs/plots/per_window_grid.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "models_baseline" / "scripts"))
from plot_per_window_grid import (  # noqa: E402
    SMOOTH_BW, SMOOTH_MIN_EFFECTIVE_N, draw_cell, load_odds, load_preds,
)

OUT = REPO / "docs" / "plots" / "cme_v1_seed_sweep_per_window_grid.png"

SEEDS = [
    ("seed 7",  REPO / "models_cme_v1" / "artifacts" / "backtest_v1_seed7"  / "predictions.csv"),
    ("seed 11", REPO / "models_cme_v1" / "artifacts" / "backtest_v1_seed11" / "predictions.csv"),
    ("seed 23", REPO / "models_cme_v1" / "artifacts" / "backtest_v1_seed23" / "predictions.csv"),
    ("seed 42", REPO / "models_cme_v1" / "artifacts" / "backtest_v1"        / "predictions.csv"),
    ("seed 47", REPO / "models_cme_v1" / "artifacts" / "backtest_v1_seed47" / "predictions.csv"),
]


def main() -> None:
    odds = load_odds()
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap = "RdYlGn"

    seed_data = []
    common_windows = None
    for name, path in SEEDS:
        df = load_preds(str(path), odds)
        ws = set(df["window_start"].unique())
        common_windows = ws if common_windows is None else common_windows & ws
        seed_data.append((name, df))
    windows = sorted(common_windows)
    print(f"Common windows: {len(windows)}")

    nrows = len(SEEDS)
    ncols = len(windows)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.25 * ncols + 1.0, 1.45 * nrows + 0.8),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.18, "wspace": 0.06})

    for r, (name, df) in enumerate(seed_data):
        for c, ws in enumerate(windows):
            sub = df[df["window_start"] == ws]
            ax = axes[r][c]
            draw_cell(ax, sub, norm, cmap,
                      show_xticks=(r == nrows - 1),
                      show_yticks=(c == 0))
            if r == 0:
                ax.set_title(pd.Timestamp(ws).strftime("%Y-%m"), fontsize=7, pad=3)
            if c == 0:
                ax.set_ylabel(name, fontsize=10, weight="bold", rotation=90,
                              labelpad=8)

    # Right-side totals box per row (naive ROI + disagree(0.60) HOME/AWAY split)
    import numpy as np
    for r, (name, df) in enumerate(seed_data):
        p = df["mod_p"].to_numpy()
        hf = df["mkt_p"].to_numpy() >= 0.5
        h_dec = df["home_avg_decimal_odds"].to_numpy()
        a_dec = df["away_avg_decimal_odds"].to_numpy()
        home_won = (df["label_home_win"] == 1).to_numpy()
        bh = (p >= 0.60) & (~hf)
        ba = ((1 - p) >= 0.60) & hf
        h_pf = np.where(home_won[bh], h_dec[bh] - 1, -1) if bh.any() else np.array([])
        a_pf = np.where(~home_won[ba], a_dec[ba] - 1, -1) if ba.any() else np.array([])
        n_naive = len(df)
        naive_roi = df["bet_profit"].mean() * 100
        h_n, h_roi = int(bh.sum()), (float(h_pf.mean()) * 100 if bh.any() else 0.0)
        a_n, a_roi = int(ba.sum()), (float(a_pf.mean()) * 100 if ba.any() else 0.0)
        ax = axes[r][-1]
        ax.text(1.04, 0.5,
                f"NAIVE  n={n_naive}\n  ROI={naive_roi:+5.1f}%\n\n"
                f"DIS(0.60):\n"
                f" HOME n={h_n}\n   ROI={h_roi:+5.1f}%\n"
                f" AWAY n={a_n}\n   ROI={a_roi:+5.1f}%",
                transform=ax.transAxes, fontsize=6.2, va="center", ha="left",
                family="monospace",
                bbox=dict(facecolor="#f7f7f7", edgecolor="#bbb",
                          boxstyle="round,pad=0.4"))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.45, pad=0.10)
    cbar.set_label("Mean ROI per bet within cell\n(green=+EV, red=-EV)", fontsize=8)

    fig.suptitle(
        "cme_v1 5-seed per-window OOS EV heatmap (rows=seed, cols=window).  "
        "Each cell is a separately trained expanding-window model.\n"
        f"Gaussian-smoothed (σ={SMOOTH_BW}, min eff n={SMOOTH_MIN_EFFECTIVE_N:.0f}).  "
        "Per-cell n / ROI = naive 'bet whichever side model favors'.  "
        "x = market implied (home), y = model predicted (home)",
        fontsize=10, y=0.998)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"Wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
