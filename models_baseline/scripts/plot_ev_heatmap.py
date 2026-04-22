#!/usr/bin/env python3
"""EV heatmap across baselines + transformer family.

For each game we'd hypothetically bet the side the model favors. We bin in
2D (market implied prob, model predicted prob) and color each cell by the
empirical ROI per bet — green = +EV, red = -EV, gray = noisy/empty.

Layout: 2 rows × 4 cols. Top row = plain baselines (logistic / xgboost / mlp).
Bottom row = transformer family (v5 / v6).
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
OUT = Path("/home/ayf7/trading/docs/plots/ev_heatmap.png")
OUT_HOMEAWAY = Path("/home/ayf7/trading/docs/plots/ev_heatmap_home_vs_away_dog.png")

# 5 models, two groups. "baselines" use only the 41 tabular features (no
# market, no lineup); "transformers" use the lineup pipeline.
MODELS = {
    "logistic": "/home/ayf7/trading/models_baseline/artifacts/backtest_logistic/predictions.csv",
    "xgboost":  "/home/ayf7/trading/models_baseline/artifacts/backtest_xgboost/predictions.csv",
    "mlp":      "/home/ayf7/trading/models_baseline/artifacts/backtest_mlp/predictions.csv",
    "v5":       "/home/ayf7/trading/models_man_xfmr/artifacts/backtest_v5_pstats_2stream_apoiss10/predictions.csv",
    "cme_v1":   "/home/ayf7/trading/models_cme_v1/artifacts/backtest_v1/predictions.csv",
    "cme_v2":   "/home/ayf7/trading/models_cme_v2/artifacts/backtest_s_tt/predictions.csv",
}
# top row = 3 baselines, bottom row = transformer-family (v5 + cme_v1 + cme_v2)
LAYOUT = [
    ["logistic", "xgboost", "mlp"],
    ["v5", "cme_v1", "cme_v2"],
]

GRID = 12  # bins per axis (~22 games/cell on average across 3164 games)
MIN_BIN = 12  # require >= this many games before showing a color

# Gaussian smoothing parameters
SMOOTH_RES = 200  # pixels per axis on smoothed plot
SMOOTH_BW = 0.05  # bandwidth (std dev) of Gaussian kernel in [0,1] units
SMOOTH_MIN_EFFECTIVE_N = 8.0  # mask cells with less than this many effective samples


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
                                    "pred_home_win_prob"])
    df["game_id"] = df["game_id"].astype(str)
    df = df.merge(odds, on="game_id")
    p = df["pred_home_win_prob"].to_numpy()
    home_won = (df["label_home_win"] == 1).to_numpy()
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()

    bet_home = p >= 0.5
    win = np.where(bet_home, home_won, ~home_won)
    odds_used = np.where(bet_home, h_dec, a_dec)
    df["bet_profit"] = np.where(win, odds_used - 1, -1)
    df["bet_home"] = bet_home
    df["mkt_home"] = df["home_implied_prob_normalized"]
    df["mod_home"] = p
    return df


def binned_roi(df, grid=GRID, min_n=MIN_BIN):
    """Return (xedges, yedges, mean_roi, n) where x=mkt_p, y=mod_p."""
    edges = np.linspace(0, 1, grid + 1)
    n, _, _ = np.histogram2d(df["mkt_home"], df["mod_home"],
                             bins=[edges, edges])
    s, _, _ = np.histogram2d(df["mkt_home"], df["mod_home"],
                             bins=[edges, edges],
                             weights=df["bet_profit"])
    mean = np.where(n >= min_n, s / np.maximum(n, 1), np.nan)
    return edges, mean.T, n.T  # transpose so y is rows (model_p)


def gaussian_roi(df, res=SMOOTH_RES, bw=SMOOTH_BW,
                 min_eff_n=SMOOTH_MIN_EFFECTIVE_N):
    """Gaussian-kernel-smoothed ROI surface.

    For each pixel we compute weighted mean of bet_profit using a 2D
    isotropic Gaussian kernel centered at the pixel's (mkt_p, model_p).
    Pixels where the effective sample size (sum of weights) is below
    min_eff_n are masked.
    """
    xs = df["mkt_home"].to_numpy(dtype=np.float32)
    ys = df["mod_home"].to_numpy(dtype=np.float32)
    pf = df["bet_profit"].to_numpy(dtype=np.float32)

    grid = np.linspace(0, 1, res, dtype=np.float32)
    XX, YY = np.meshgrid(grid, grid)  # XX: market, YY: model; shape (res, res)

    # Vectorize over pixels in chunks to bound memory: each pixel needs
    # weights of length n_games. Process by row.
    mean_roi = np.full((res, res), np.nan, dtype=np.float32)
    eff_n = np.zeros((res, res), dtype=np.float32)
    inv_2bw2 = 1.0 / (2.0 * bw * bw)

    for j in range(res):
        # Compute (res, n) weights for this entire row
        dx2 = (XX[j, :, None] - xs[None, :]) ** 2  # (res, n)
        dy2 = (YY[j, :, None] - ys[None, :]) ** 2  # (res, n)
        w = np.exp(-(dx2 + dy2) * inv_2bw2)        # (res, n)
        wsum = w.sum(axis=1)                        # (res,)
        wpf = (w * pf[None, :]).sum(axis=1)         # (res,)
        eff_n[j, :] = wsum
        mask = wsum >= min_eff_n
        mean_roi[j, mask] = wpf[mask] / wsum[mask]
    return mean_roi, eff_n


def _draw_heatmap(ax, name, df, norm, cmap, *, show_ylabel):
    smooth_roi, _eff_n = gaussian_roi(df)
    n_total = len(df)
    roi_total = df["bet_profit"].mean()

    im = ax.imshow(smooth_roi, origin="lower", extent=[0, 1, 0, 1],
                   cmap=cmap, norm=norm, aspect="auto",
                   interpolation="bilinear")

    wins = df[df["bet_profit"] > 0]
    loss = df[df["bet_profit"] < 0]
    ax.scatter(loss["mkt_home"], loss["mod_home"], s=3, color="black",
               alpha=0.10, marker="x", linewidths=0.4)
    ax.scatter(wins["mkt_home"], wins["mod_home"], s=3, color="black",
               alpha=0.10, marker="o", linewidths=0)

    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.7, alpha=0.5)

    ax.add_patch(Rectangle((0, 0.60), 0.5, 0.40, fill=False,
                           edgecolor="blue", lw=1.3, linestyle="-",
                           label="disagree(0.60) HOME-dog"))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 0.40, fill=False,
                           edgecolor="purple", lw=1.3, linestyle="-",
                           label="disagree(0.60) AWAY-dog"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("market implied prob (home)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("model predicted prob (home)", fontsize=9)
    ax.set_title(f"{name}  (n={n_total}, naive-bet ROI={roi_total*100:+.2f}%)",
                 fontsize=10)
    ax.tick_params(labelsize=8)
    return im


def plot_main():
    odds = load_odds()
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap = "RdYlGn"

    nrows = len(LAYOUT)
    ncols = max(len(row) for row in LAYOUT)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.6 * nrows),
                             sharex=True, sharey=True)
    if nrows == 1:
        axes = [axes]

    last_im = None
    for r, row in enumerate(LAYOUT):
        for c in range(ncols):
            ax = axes[r][c]
            name = row[c] if c < len(row) else None
            if name is None:
                ax.axis("off")
                continue
            df = load_preds(MODELS[name], odds)
            last_im = _draw_heatmap(ax, name, df, norm, cmap,
                                    show_ylabel=(c == 0))
            if r == 0 and c == 0:
                ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)

    # row labels at far left
    row_labels = ["Plain baselines\n(41 tabular feats only)",
                  "Transformer family\n(lineup + tabular)"]
    for r, row in enumerate(LAYOUT):
        if r < len(row_labels):
            axes[r][0].text(-0.27, 0.5, row_labels[r],
                            transform=axes[r][0].transAxes,
                            ha="center", va="center", rotation=90,
                            fontsize=11, weight="bold", color="#333")

    cbar = fig.colorbar(last_im, ax=[ax for row in axes for ax in row],
                        shrink=0.55, pad=0.015)
    cbar.set_label("Mean ROI per bet within cell\n(green = +EV, red = -EV)",
                   fontsize=10)
    fig.suptitle(
        f"EV heatmap (Gaussian-smoothed, σ={SMOOTH_BW}, "
        f"min effective n={SMOOTH_MIN_EFFECTIVE_N:.0f}):  "
        f"empirical ROI from naive 'bet whichever side model favors'\n"
        "x = market implied prob (home),  y = model predicted prob (home)",
        y=0.995, fontsize=12)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")
    plt.close(fig)


def _draw_home_vs_away_dog(ax, name, df, pthr_grid, *, show_ylabel, show_legend):
    home_roi, home_n, away_roi, away_n = [], [], [], []
    p = df["mod_home"].to_numpy()
    hf = df["mkt_home"].to_numpy() >= 0.5
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()
    home_won = (df["label_home_win"] == 1).to_numpy()
    for pthr in pthr_grid:
        bh = (p >= pthr) & (~hf)
        ba = ((1 - p) >= pthr) & hf
        h_pf = np.where(home_won[bh], h_dec[bh] - 1, -1) if bh.any() else np.array([])
        a_pf = np.where(~home_won[ba], a_dec[ba] - 1, -1) if ba.any() else np.array([])
        home_roi.append(h_pf.mean() if len(h_pf) else np.nan)
        home_n.append(len(h_pf))
        away_roi.append(a_pf.mean() if len(a_pf) else np.nan)
        away_n.append(len(a_pf))

    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0.60, color="gray", linestyle="--", lw=0.8,
               label="p_thr=0.60")
    ax.plot(pthr_grid, np.array(home_roi) * 100, "-o", color="C2",
            label="HOME-dog disagree", markersize=3)
    ax.plot(pthr_grid, np.array(away_roi) * 100, "-o", color="C3",
            label="AWAY-dog disagree", markersize=3)
    ax2 = ax.twinx()
    ax2.bar(pthr_grid, home_n, width=0.008, alpha=0.15, color="C2")
    ax2.bar(pthr_grid, [-x for x in away_n], width=0.008, alpha=0.15,
            color="C3")
    ax2.tick_params(labelsize=7)
    ax2.set_ylabel("bet count (+home / −away)", fontsize=7)

    ax.set_xlabel("disagree threshold p_thr", fontsize=9)
    ax.set_xlim(0.50, 0.80)
    if show_ylabel:
        ax.set_ylabel("ROI per bet (%)", fontsize=9)
    ax.set_title(name, fontsize=10)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)


def plot_home_vs_away_dog():
    """ROI vs model_p, separated by HOME-dog vs AWAY-dog."""
    odds = load_odds()
    pthr_grid = np.linspace(0.50, 0.80, 31)

    nrows = len(LAYOUT)
    ncols = max(len(row) for row in LAYOUT)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows),
                             sharey=True)
    if nrows == 1:
        axes = [axes]

    for r, row in enumerate(LAYOUT):
        for c in range(ncols):
            ax = axes[r][c]
            name = row[c] if c < len(row) else None
            if name is None:
                ax.axis("off")
                continue
            df = load_preds(MODELS[name], odds)
            _draw_home_vs_away_dog(ax, name, df, pthr_grid,
                                   show_ylabel=(c == 0),
                                   show_legend=(r == 0 and c == 0))

    row_labels = ["Plain baselines", "Transformer family"]
    for r, row in enumerate(LAYOUT):
        if r < len(row_labels):
            axes[r][0].text(-0.30, 0.5, row_labels[r],
                            transform=axes[r][0].transAxes,
                            ha="center", va="center", rotation=90,
                            fontsize=11, weight="bold", color="#333")

    fig.suptitle("Disagree-rule ROI by p_thr — HOME-dog vs AWAY-dog",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_HOMEAWAY, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_HOMEAWAY}")
    plt.close(fig)


if __name__ == "__main__":
    plot_main()
    plot_home_vs_away_dog()
