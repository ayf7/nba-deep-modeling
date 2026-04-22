#!/usr/bin/env python3
"""Per-window EV heatmap grid: one or more train rows + test row.

Requires a backtest run with --save-train-preds, which writes
predictions_train.csv alongside predictions.csv.

By default plots a single train row (all training data) on top and the
OOS test row on bottom. Pass --train-recent-months N to restrict the
train row to games within the last N months before each window_start.
Pass multiple values (e.g. --train-recent-months 3 2 1) to plot one
train row per N, ordered top-to-bottom in the order given, with test
appended at the bottom.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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

DEFAULT_RUN_DIR = REPO / "models_cme_v1" / "artifacts" / "backtest_v1_trainpreds"
DEFAULT_OUT = REPO / "docs" / "plots" / "cme_v1_seed42_trainvstest_per_window_grid.png"
DEFAULT_LABEL = "cme_v1 seed 42"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                   help="Directory containing predictions.csv + predictions_train.csv")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--label", default=DEFAULT_LABEL,
                   help="Title prefix (e.g., 'logistic', 'xgboost').")
    p.add_argument("--train-recent-months", type=int, nargs="+", default=[0],
                   help="One or more recency windows in months. 0 means use all "
                        "training data. Multiple values produce stacked train rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    preds_test = run_dir / "predictions.csv"
    preds_train = run_dir / "predictions_train.csv"
    _run(preds_train, preds_test, args.out, args.label,
         train_recent_months_list=args.train_recent_months)


def _load_dates(preds_train: Path) -> pd.DataFrame:
    dates = pd.read_csv(preds_train, usecols=["game_id", "window_start", "game_date"])
    dates["game_id"] = dates["game_id"].astype(str)
    dates["window_start"] = pd.to_datetime(dates["window_start"])
    dates["game_date"] = pd.to_datetime(dates["game_date"])
    return dates


def _filter_train_recent(df_train: pd.DataFrame, dates: pd.DataFrame,
                         months: int) -> pd.DataFrame:
    """Restrict df_train to the single month exactly `months` back from window_start.

    months=1 -> games in [window_start - 1mo, window_start)
    months=2 -> games in [window_start - 2mo, window_start - 1mo)
    months=N -> games in [window_start - N mo, window_start - (N-1) mo)
    months=0 -> all training data
    """
    df = df_train.merge(dates, on=["game_id", "window_start"], how="left")
    if months <= 0:
        return df.copy()
    lower = df["window_start"] - pd.DateOffset(months=months)
    upper = df["window_start"] - pd.DateOffset(months=months - 1)
    return df[(df["game_date"] >= lower) & (df["game_date"] < upper)].copy()


def _row_label(months: int) -> str:
    if months <= 0:
        return "train\n(all)"
    return f"train\nm-{months}"


def _run(preds_train: Path, preds_test: Path, out_path: Path, label: str,
         train_recent_months_list: list[int]) -> None:
    odds = load_odds()
    df_train_full = load_preds(str(preds_train), odds)
    df_test = load_preds(str(preds_test), odds)
    dates = _load_dates(preds_train)

    train_rows: list[tuple[str, pd.DataFrame]] = []
    for m in train_recent_months_list:
        if m <= 0:
            sub = df_train_full
        else:
            sub = _filter_train_recent(df_train_full, dates, m)
        print(f"train m={m}: n={len(sub)}")
        train_rows.append((_row_label(m), sub))

    common = set(df_test["window_start"].unique())
    for _, sub in train_rows:
        common &= set(sub["window_start"].unique())
    windows = sorted(common)
    print(f"Common windows: {len(windows)}, test n={len(df_test)}")

    rows = train_rows + [("test", df_test)]
    nrows = len(rows)
    ncols = len(windows)

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap = "RdYlGn"

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.25 * ncols + 1.0, 1.45 * nrows + 0.8),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.22, "wspace": 0.06})
    if nrows == 1:
        axes = np.array([axes])

    for r, (name, df) in enumerate(rows):
        for c, ws in enumerate(windows):
            sub = df[df["window_start"] == ws]
            ax = axes[r][c]
            draw_cell(ax, sub, norm, cmap,
                      show_xticks=(r == nrows - 1),
                      show_yticks=(c == 0))
            if r == 0:
                ax.set_title(pd.Timestamp(ws).strftime("%Y-%m"), fontsize=7, pad=3)
            if c == 0:
                ax.set_ylabel(name, fontsize=8.5, weight="bold", rotation=90,
                              labelpad=8)

    # Right-edge totals per row
    for r, (name, df) in enumerate(rows):
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
        naive_roi = df["bet_profit"].mean() * 100 if n_naive else 0.0
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
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.55, pad=0.10)
    cbar.set_label("Mean ROI per bet within cell\n(green=+EV, red=-EV)", fontsize=8)

    fig.suptitle(
        f"{label} · per-window EV heatmaps.  "
        "Each column = one separately trained expanding-window model.\n"
        "Train rows: predictions on a recency-restricted slice of in-sample data.  "
        "Test row: held-out next-month OOS predictions.  "
        f"Gaussian-smoothed (σ={SMOOTH_BW}, min eff n={SMOOTH_MIN_EFFECTIVE_N:.0f}).  "
        "Per-cell n / ROI = naive 'bet whichever side model favors'.",
        fontsize=10, y=0.998)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    print(f"Wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
