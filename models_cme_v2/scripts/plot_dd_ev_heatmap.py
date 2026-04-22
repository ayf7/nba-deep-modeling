"""EV heatmap for the CME-v2 double-descent sweep.

3-panel comparison: xs (small) | s (middle) | xl (large). For each, plot
the Gaussian-smoothed bet ROI surface across (market implied prob,
model predicted prob). Naive-bet ROI in the title.

Reuses gaussian_roi + odds loader from the baseline EV script.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "models_baseline" / "scripts"))
from plot_ev_heatmap import load_odds, gaussian_roi  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
OUT = ARTIFACTS / "dd_ev_heatmap.png"

PANELS = [
    ("xs",  "run_v2_dd_xs",  "d=32, n=1/1 · 80k params"),
    ("s",   "run_v2_dd_s",   "d=64, n=2/2 · 330k params"),
    ("m",   "run_v2_dd_m",   "d=128, n=4/4 · 1.6M params"),
    ("l",   "run_v2_dd_l",   "d=192, n=6/6 · 5.4M params"),
    ("xl",  "run_v2_dd_xl",  "d=256, n=8/8 · 13.7M params"),
    ("x2l", "run_v2_dd_x2l", "d=384, n=12/12 · 44.7M params"),
    ("x3l", "run_v2_dd_x3l", "d=512, n=16/16 · 104M params"),
]


def load_preds(path: Path, odds: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["game_id", "label_home_win", "pred_home_win_prob"])
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
    df["mkt_home"] = df["home_implied_prob_normalized"]
    df["mod_home"] = p
    return df


def main() -> None:
    odds = load_odds()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.8 * len(PANELS), 4.6),
                             sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap = "RdYlGn"

    last_im = None
    for ax, (name, run_dir, label) in zip(axes, PANELS):
        preds = ARTIFACTS / run_dir / "predictions.csv"
        df = load_preds(preds, odds)
        smooth_roi, _ = gaussian_roi(df)
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
                               edgecolor="blue", lw=1.3,
                               label="HOME-dog disagree(0.60)"))
        ax.add_patch(Rectangle((0.5, 0), 0.5, 0.40, fill=False,
                               edgecolor="purple", lw=1.3,
                               label="AWAY-dog disagree(0.60)"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("market implied prob (home)", fontsize=9)
        ax.set_title(f"{name}: {label}\nn={n_total}, naive-bet ROI={roi_total*100:+.2f}%",
                     fontsize=10)
        ax.tick_params(labelsize=8)
        last_im = im

    axes[0].set_ylabel("model predicted prob (home)", fontsize=9)
    axes[0].legend(fontsize=7, loc="lower right")

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label("smoothed ROI per bet", fontsize=9)

    # Slice ROI for HOME-dog and AWAY-dog disagree(0.60) boxes
    print(f"{'name':<5} {'home_dog':>14} {'away_dog':>14} {'naive_all':>14}")
    for name, run_dir, _ in PANELS:
        preds = ARTIFACTS / run_dir / "predictions.csv"
        df = load_preds(preds, odds)
        home_dog = df[(df["mkt_home"] < 0.5) & (df["mod_home"] >= 0.60)]
        away_dog = df[(df["mkt_home"] >= 0.5) & (df["mod_home"] < 0.40)]
        def fmt(d):
            if len(d) == 0:
                return "  n=0          "
            return f"{d['bet_profit'].mean()*100:+6.2f}% (n={len(d):>3})"
        print(f"{name:<5} {fmt(home_dog):>14}  {fmt(away_dog):>14}  "
              f"{df['bet_profit'].mean()*100:+6.2f}% (n={len(df):>4})")

    fig.suptitle("CME-v2 DD sweep · EV heatmap (test split, 1061 games)", fontsize=12)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
