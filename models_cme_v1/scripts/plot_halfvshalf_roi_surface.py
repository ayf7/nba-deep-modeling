#!/usr/bin/env python3
"""Cross-window stability check for cme_v1 seed 42's OOS ROI topology.

Pool the 3210 OOS predictions into two halves by window-date:
  - first half = windows 1..9 (chronologically earlier)
  - second half = windows 10..18

For each half, fit a single Gaussian-smoothed ROI surface in (mkt_p, mod_p)
space. Plot side by side + a diff panel + a per-cell scatter. Report Pearson
correlation between the two surfaces on cells where both have enough effective
support.

If the green regions overlap (r >> 0): topology is forward-predictable, and a
historical ROI surface from past windows is a usable feature.
If the green regions move (r ~ 0): cross-seed similarity is fixed-test-distribution
artifact, not transferable signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "models_baseline" / "scripts"))
from plot_per_window_grid import gaussian_roi, load_odds, load_preds  # noqa: E402

PREDS = REPO / "models_cme_v1" / "artifacts" / "backtest_v1" / "predictions.csv"
OUT = REPO / "docs" / "plots" / "cme_v1_seed42_halfvshalf_roi_surface.png"


def main() -> None:
    odds = load_odds()
    df = load_preds(str(PREDS), odds)
    windows = sorted(df["window_start"].unique())
    print(f"Total windows: {len(windows)}, total games (with odds): {len(df)}")

    mid = len(windows) // 2  # 9
    first_w = windows[:mid]
    second_w = windows[mid:]
    df_a = df[df["window_start"].isin(first_w)].copy()
    df_b = df[df["window_start"].isin(second_w)].copy()
    print(f"First half:  windows {pd.Timestamp(first_w[0]).strftime('%Y-%m')}..{pd.Timestamp(first_w[-1]).strftime('%Y-%m')}  n={len(df_a)}")
    print(f"Second half: windows {pd.Timestamp(second_w[0]).strftime('%Y-%m')}..{pd.Timestamp(second_w[-1]).strftime('%Y-%m')}  n={len(df_b)}")

    surf_a = gaussian_roi(df_a)
    surf_b = gaussian_roi(df_b)
    diff = surf_b - surf_a

    valid = (~np.isnan(surf_a)) & (~np.isnan(surf_b))
    a_flat = surf_a[valid]
    b_flat = surf_b[valid]
    r = float(np.corrcoef(a_flat, b_flat)[0, 1])
    print(f"\nPearson r over {valid.sum()} valid grid cells: {r:+.3f}")

    # ROI consistency in fixed disagree boxes
    def slice_roi(d, mkt_lo, mkt_hi, mod_lo, mod_hi):
        m = ((d["mkt_p"] >= mkt_lo) & (d["mkt_p"] < mkt_hi)
             & (d["mod_p"] >= mod_lo) & (d["mod_p"] < mod_hi))
        return int(m.sum()), (float(d.loc[m, "bet_profit"].mean()) * 100 if m.any() else float("nan"))

    print()
    print(f"{'slice':<22}{'first':>20}{'second':>20}")
    for tag, args in [
        ("naive (all)", (0, 1, 0, 1)),
        ("TL (mkt<.5, mod>.60)", (0.0, 0.5, 0.60, 1.0)),
        ("BR (mkt>=.5, mod<.40)", (0.5, 1.0, 0.0, 0.40)),
        ("center (.4<mod<.6)", (0.0, 1.0, 0.40, 0.60)),
    ]:
        na, ra = slice_roi(df_a, *args)
        nb, rb = slice_roi(df_b, *args)
        print(f"{tag:<22}  n={na:>4} ROI={ra:+6.2f}%   n={nb:>4} ROI={rb:+6.2f}%")

    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.18)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap = "RdYlGn"

    titles = [
        f"FIRST half (windows 1..{mid})\n{pd.Timestamp(first_w[0]).strftime('%Y-%m')}..{pd.Timestamp(first_w[-1]).strftime('%Y-%m')}  n={len(df_a)}",
        f"SECOND half (windows {mid+1}..{len(windows)})\n{pd.Timestamp(second_w[0]).strftime('%Y-%m')}..{pd.Timestamp(second_w[-1]).strftime('%Y-%m')}  n={len(df_b)}",
    ]
    for i, (surf, dh, title) in enumerate([
        (surf_a, df_a, titles[0]),
        (surf_b, df_b, titles[1]),
    ]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(surf, origin="lower", extent=[0, 1, 0, 1],
                  cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")
        wins = dh[dh["bet_profit"] > 0]
        loss = dh[dh["bet_profit"] < 0]
        ax.scatter(loss["mkt_p"], loss["mod_p"], s=2, color="black", alpha=0.10,
                   marker="x", linewidths=0.3)
        ax.scatter(wins["mkt_p"], wins["mod_p"], s=2, color="black", alpha=0.10,
                   marker="o", linewidths=0)
        ax.plot([0, 1], [0, 1], "--", color="black", lw=0.5, alpha=0.4)
        ax.add_patch(Rectangle((0, 0.60), 0.5, 0.40, fill=False, edgecolor="blue", lw=0.8))
        ax.add_patch(Rectangle((0.5, 0), 0.5, 0.40, fill=False, edgecolor="purple", lw=0.8))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("market implied (home)")
        if i == 0:
            ax.set_ylabel("model predicted (home)")
        ax.set_title(title, fontsize=10)

    # Diff panel
    ax_diff = fig.add_subplot(gs[0, 2])
    diff_norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    im_d = ax_diff.imshow(diff, origin="lower", extent=[0, 1, 0, 1],
                          cmap="RdBu_r", norm=diff_norm, aspect="auto",
                          interpolation="bilinear")
    ax_diff.plot([0, 1], [0, 1], "--", color="black", lw=0.5, alpha=0.4)
    ax_diff.set_xlim(0, 1); ax_diff.set_ylim(0, 1)
    ax_diff.set_xlabel("market implied (home)")
    ax_diff.set_title("SECOND - FIRST  (blue: 2nd more +EV here)", fontsize=10)
    fig.colorbar(im_d, ax=ax_diff, fraction=0.04, pad=0.02)

    # Scatter panel: per-cell (a, b)
    ax_sc = fig.add_subplot(gs[0, 3])
    ax_sc.scatter(a_flat, b_flat, s=2, color="black", alpha=0.15)
    lim = max(0.5, max(abs(a_flat).max(), abs(b_flat).max()))
    ax_sc.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=0.6)
    ax_sc.axhline(0, color="grey", lw=0.4); ax_sc.axvline(0, color="grey", lw=0.4)
    ax_sc.set_xlim(-lim, lim); ax_sc.set_ylim(-lim, lim)
    ax_sc.set_xlabel("first-half smoothed ROI per cell")
    ax_sc.set_ylabel("second-half smoothed ROI per cell")
    ax_sc.set_title(f"Per-cell stability  r={r:+.3f}", fontsize=10)

    fig.suptitle(
        f"cme_v1 seed 42: cross-window stability of OOS EV topology.  "
        f"First 9 vs last 9 expanding-window backtests, Gaussian-smoothed surfaces.",
        fontsize=11, y=1.02)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
