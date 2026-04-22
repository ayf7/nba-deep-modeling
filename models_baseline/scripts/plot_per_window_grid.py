#!/usr/bin/env python3
"""Per-window per-model OOS EV heatmap grid.

Rows = models (logistic / xgboost / mlp / v5 / cme_v1)
Cols = backtest windows (each col is a separately trained model)

Each cell is the same EV heatmap as docs/plots/ev_heatmap.png but computed
on only the ~178 OOS games inside that single window. Smoothing is loosened
(larger kernel, lower min effective n) to compensate for the sparser data.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

CORE = Path("/home/ayf7/trading/data/artifacts/nba_core.sqlite")
OUT = Path("/home/ayf7/trading/docs/plots/per_window_grid.png")

MODELS = [
    ("logistic", "/home/ayf7/trading/models_baseline/artifacts/backtest_logistic/predictions.csv"),
    ("xgboost",  "/home/ayf7/trading/models_baseline/artifacts/backtest_xgboost/predictions.csv"),
    ("mlp",      "/home/ayf7/trading/models_baseline/artifacts/backtest_mlp/predictions.csv"),
    ("v5",       "/home/ayf7/trading/models_man_xfmr/artifacts/backtest_v5_pstats_2stream_apoiss10/predictions.csv"),
    ("cme_v1",   "/home/ayf7/trading/models_cme_v1/artifacts/backtest_v1/predictions.csv"),
    ("cme_v2",   "/home/ayf7/trading/models_cme_v2/artifacts/backtest_s_tt/predictions.csv"),
]

# Looser smoothing for sparse per-window data (~178 games each)
SMOOTH_RES = 100
SMOOTH_BW = 0.09          # was 0.05 in main plot; widen to compensate for fewer points
SMOOTH_MIN_EFFECTIVE_N = 3.0   # was 8.0


def load_odds():
    with sqlite3.connect(CORE) as con:
        odds = pd.read_sql_query(
            """SELECT game_id, home_avg_decimal_odds, away_avg_decimal_odds,
                      home_implied_prob_normalized
               FROM game_moneyline_odds WHERE has_moneyline_odds=1""", con)
    odds["game_id"] = odds["game_id"].astype(str)
    return odds


def load_preds(path, odds):
    df = pd.read_csv(path, usecols=["game_id", "label_home_win",
                                    "pred_home_win_prob", "window_start"])
    df["game_id"] = df["game_id"].astype(str)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.merge(odds, on="game_id")

    p = df["pred_home_win_prob"].to_numpy()
    home_won = (df["label_home_win"] == 1).to_numpy()
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()
    bet_home = p >= 0.5
    win = np.where(bet_home, home_won, ~home_won)
    odds_used = np.where(bet_home, h_dec, a_dec)
    df["bet_profit"] = np.where(win, odds_used - 1, -1)
    df["mkt_p"] = df["home_implied_prob_normalized"]
    df["mod_p"] = p
    return df


def gaussian_roi(df, res=SMOOTH_RES, bw=SMOOTH_BW,
                 min_eff_n=SMOOTH_MIN_EFFECTIVE_N):
    xs = df["mkt_p"].to_numpy(dtype=np.float32)
    ys = df["mod_p"].to_numpy(dtype=np.float32)
    pf = df["bet_profit"].to_numpy(dtype=np.float32)
    if len(xs) == 0:
        return np.full((res, res), np.nan, dtype=np.float32)

    grid = np.linspace(0, 1, res, dtype=np.float32)
    XX, YY = np.meshgrid(grid, grid)
    mean_roi = np.full((res, res), np.nan, dtype=np.float32)
    inv_2bw2 = 1.0 / (2.0 * bw * bw)
    for j in range(res):
        dx2 = (XX[j, :, None] - xs[None, :]) ** 2
        dy2 = (YY[j, :, None] - ys[None, :]) ** 2
        w = np.exp(-(dx2 + dy2) * inv_2bw2)
        wsum = w.sum(axis=1)
        wpf = (w * pf[None, :]).sum(axis=1)
        mask = wsum >= min_eff_n
        mean_roi[j, mask] = wpf[mask] / wsum[mask]
    return mean_roi


def draw_cell(ax, df, norm, cmap, *, show_xticks, show_yticks):
    smooth = gaussian_roi(df)
    ax.imshow(smooth, origin="lower", extent=[0, 1, 0, 1],
              cmap=cmap, norm=norm, aspect="auto",
              interpolation="bilinear")
    # raw bet outcomes as small markers (so empty regions are visible)
    wins = df[df["bet_profit"] > 0]
    loss = df[df["bet_profit"] < 0]
    ax.scatter(loss["mkt_p"], loss["mod_p"], s=2, color="black", alpha=0.30,
               marker="x", linewidths=0.4)
    ax.scatter(wins["mkt_p"], wins["mod_p"], s=2, color="black", alpha=0.30,
               marker="o", linewidths=0)

    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.4, alpha=0.45)
    ax.add_patch(Rectangle((0, 0.60), 0.5, 0.40, fill=False,
                           edgecolor="blue", lw=0.5))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 0.40, fill=False,
                           edgecolor="purple", lw=0.5))

    # disagree(0.60) ROI within each box (HOME-dog top-left, AWAY-dog bottom-right)
    p = df["mod_p"].to_numpy()
    mkt = df["mkt_p"].to_numpy()
    home_won = (df["label_home_win"] == 1).to_numpy()
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()
    bh = (p >= 0.60) & (mkt < 0.5)
    ba = ((1 - p) >= 0.60) & (mkt >= 0.5)
    h_pf = np.where(home_won[bh], h_dec[bh] - 1, -1) if bh.any() else np.array([])
    a_pf = np.where(~home_won[ba], a_dec[ba] - 1, -1) if ba.any() else np.array([])
    n_h, n_a = int(bh.sum()), int(ba.sum())
    roi_h = float(h_pf.mean()) * 100 if n_h else 0.0
    roi_a = float(a_pf.mean()) * 100 if n_a else 0.0

    def _label(n, roi):
        if n == 0:
            return "n=0", "#888"
        return f"n={n}\nROI={roi:+.0f}%", ("#1a9641" if roi > 0 else "#d7191c")

    txt_h, col_h = _label(n_h, roi_h)
    txt_a, col_a = _label(n_a, roi_a)
    ax.text(0.04, 0.97, txt_h, transform=ax.transAxes,
            fontsize=5.5, va="top", ha="left", color=col_h, weight="bold",
            bbox=dict(facecolor="white", alpha=0.78, pad=0.6, edgecolor="none"))
    ax.text(0.96, 0.03, txt_a, transform=ax.transAxes,
            fontsize=5.5, va="bottom", ha="right", color=col_a, weight="bold",
            bbox=dict(facecolor="white", alpha=0.78, pad=0.6, edgecolor="none"))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(labelsize=4, length=2)
    if not show_xticks:
        ax.set_xticklabels([])
    if not show_yticks:
        ax.set_yticklabels([])


def main():
    odds = load_odds()
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap = "RdYlGn"

    model_data = []
    windows_set = None
    for name, path in MODELS:
        df = load_preds(path, odds)
        ws = set(df["window_start"].unique())
        windows_set = ws if windows_set is None else windows_set & ws
        model_data.append((name, df))
    windows = sorted(windows_set)
    print(f"Common windows: {len(windows)}")

    nrows = len(MODELS)
    ncols = len(windows)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.25 * ncols + 1.0, 1.45 * nrows + 0.8),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.18, "wspace": 0.06})

    for r, (name, df) in enumerate(model_data):
        for c, ws in enumerate(windows):
            sub = df[df["window_start"] == ws]
            ax = axes[r][c]
            draw_cell(ax, sub, norm, cmap,
                      show_xticks=(r == nrows - 1),
                      show_yticks=(c == 0))
            if r == 0:
                ax.set_title(pd.Timestamp(ws).strftime("%Y-%m"),
                             fontsize=7, pad=3)
            if c == 0:
                ax.set_ylabel(name, fontsize=10, weight="bold", rotation=90,
                              labelpad=8)

    # Right-side TOTALs box per row, plus disagree(0.60) HOME/AWAY split
    for r, (name, df) in enumerate(model_data):
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
        h_n, h_roi = bh.sum(), (h_pf.mean() * 100 if bh.sum() else 0.0)
        a_n, a_roi = ba.sum(), (a_pf.mean() * 100 if ba.sum() else 0.0)
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

    # Single shared colorbar across all cells
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        shrink=0.45, pad=0.10)
    cbar.set_label("Mean ROI per bet within cell\n(green=+EV, red=-EV)",
                   fontsize=8)

    fig.suptitle(
        "Per-window OOS EV heatmap, by model (rows) × backtest window (cols).  "
        "Each window is a separately trained model.\n"
        f"Gaussian-smoothed (σ={SMOOTH_BW}, min eff n={SMOOTH_MIN_EFFECTIVE_N:.0f}; "
        "looser than the all-windows plot due to ~178 games/cell).  "
        "Per-cell n / ROI = naive 'bet whichever side model favors'.  "
        "x = market implied (home), y = model predicted (home)",
        fontsize=10, y=0.998)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"Wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
