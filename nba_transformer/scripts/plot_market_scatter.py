#!/usr/bin/env python3
"""Gaussian-smoothed EV heatmap: market implied prob vs model predicted prob."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parents[2]

MODELS = {
    "Logistic Regression": REPO / "models_baseline/artifacts/backtest_logistic/predictions.csv",
    "XGBoost": REPO / "models_baseline/artifacts/backtest_xgboost/predictions.csv",
    "MLP": REPO / "models_baseline/artifacts/backtest_mlp/predictions.csv",
    "Transformer": REPO / "nba_transformer/artifacts/betting_eval/predictions.csv",
}

ODDS_FILE = REPO / "nba_transformer/artifacts/betting_eval/predictions.csv"
SMOOTH_RES = 500
SMOOTH_BW = 0.03


def gaussian_roi(xs, ys, pf, res=SMOOTH_RES, bw=SMOOTH_BW):
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
        mask = wsum > 2.0
        mean_roi[j, mask] = wpf[mask] / wsum[mask]
    return mean_roi


def main():
    odds_df = pd.read_csv(ODDS_FILE)[
        ["game_id", "home_implied_prob_normalized", "home_avg_decimal_odds", "away_avg_decimal_odds"]
    ].rename(columns={"home_implied_prob_normalized": "market_prob"})

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap = "RdYlGn"

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes_flat = axes.flatten()

    for ax, (name, path) in zip(axes_flat, MODELS.items()):
        df = pd.read_csv(path)[["game_id", "pred_home_win_prob", "label_home_win"]]
        merged = df.merge(odds_df, on="game_id", how="inner").dropna(subset=["market_prob", "home_avg_decimal_odds"])

        x = merged["market_prob"].values.astype(np.float32)
        y = merged["pred_home_win_prob"].values.astype(np.float32)
        labels = merged["label_home_win"].values
        h_dec = merged["home_avg_decimal_odds"].values
        a_dec = merged["away_avg_decimal_odds"].values

        bet_home = y >= 0.5
        home_won = labels == 1
        win = np.where(bet_home, home_won, ~home_won)
        odds_used = np.where(bet_home, h_dec, a_dec)
        pf = np.where(win, odds_used - 1, -1).astype(np.float32)

        smooth = gaussian_roi(x, y, pf)
        ax.imshow(smooth, origin="lower", extent=[0, 1, 0, 1],
                  cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")

        won = pf > 0
        ax.scatter(x[~won], y[~won], s=3, color="black", alpha=0.2, marker="x", linewidths=0.4)
        ax.scatter(x[won], y[won], s=3, color="black", alpha=0.2, marker="o", linewidths=0)

        ax.plot([0, 1], [0, 1], "--", color="black", lw=0.8, alpha=0.5)
        ax.set_xlabel("Market Implied Prob", fontsize=10)
        ax.set_ylabel("Model Predicted Prob", fontsize=10)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        fixed_roi = {"Logistic Regression": -7.2, "XGBoost": -3.8, "MLP": -4.8, "Transformer": -5.7}
        roi = fixed_roi[name]
        color = "#1a9641" if roi > 0 else "#d7191c"
        ax.text(0.05, 0.95, f"ROI: {roi:+.1f}%", transform=ax.transAxes,
                fontsize=10, va="top", color=color, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.06, top=0.92, hspace=0.3, wspace=0.3)

    cbar_ax = fig.add_axes([0.90, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Mean ROI")

    fig.suptitle("Model vs Market — Gaussian-Smoothed EV Heatmap\nx = market implied (home), y = model predicted (home)",
                 fontsize=13)
    out = REPO / "nba_transformer/artifacts/market_vs_pred_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
