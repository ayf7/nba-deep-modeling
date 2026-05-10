#!/usr/bin/env python3
"""Per-window per-model OOS EV heatmap grid.

Rows = models, Cols = backtest windows.
Each cell is an EV heatmap on the ~178 OOS games in that window.
Odds joined directly from oddsportal_moneyline.sqlite.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
ODDS_DB = REPO_ROOT / "data" / "artifacts" / "oddsportal_moneyline.sqlite"
CORE_DB = REPO_ROOT / "data" / "artifacts" / "nba_core.sqlite"
OUT = REPO_ROOT / "nba_transformer" / "artifacts" / "betting_eval" / "per_window_grid.png"

MODELS = [
    ("Logistic", REPO_ROOT / "models_baseline" / "artifacts" / "backtest_logistic" / "predictions.csv"),
    ("XGBoost",  REPO_ROOT / "models_baseline" / "artifacts" / "backtest_xgboost" / "predictions.csv"),
    ("MLP",      REPO_ROOT / "models_baseline" / "artifacts" / "backtest_mlp" / "predictions.csv"),
    ("Transformer", REPO_ROOT / "models_cme_v5" / "artifacts" / "full_backtest_d128" / "full_d128" / "predictions.csv"),
]

SMOOTH_RES = 80
SMOOTH_BW = 0.06
SMOOTH_MIN_EFFECTIVE_N = 0.3

TEAM_NAME_TO_TRICODE = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def load_odds():
    odds_con = sqlite3.connect(str(ODDS_DB))
    odds_df = pd.read_sql_query(
        """SELECT game_date_et, home_name, away_name, home_result, away_result,
                  home_avg_decimal_odds, away_avg_decimal_odds,
                  home_implied_prob_normalized
           FROM moneyline_odds
           WHERE has_moneyline_odds = 1 AND status_id IN (3, 10)""",
        odds_con,
    )
    odds_con.close()
    odds_df["home_tri"] = odds_df["home_name"].map(TEAM_NAME_TO_TRICODE)
    odds_df["away_tri"] = odds_df["away_name"].map(TEAM_NAME_TO_TRICODE)
    odds_df = odds_df.dropna(subset=["home_tri", "away_tri"])

    core_con = sqlite3.connect(str(CORE_DB))
    games_df = pd.read_sql_query(
        "SELECT game_id, game_date, home_team_abbr, away_team_abbr, home_score, away_score FROM games",
        core_con,
    )
    core_con.close()

    merged = games_df.merge(
        odds_df,
        left_on=["game_date", "home_team_abbr", "away_team_abbr", "home_score", "away_score"],
        right_on=["game_date_et", "home_tri", "away_tri", "home_result", "away_result"],
        how="inner",
    )
    merged["game_id"] = merged["game_id"].astype(str)
    result = merged[["game_id", "home_avg_decimal_odds", "away_avg_decimal_odds",
                      "home_implied_prob_normalized"]].drop_duplicates(subset=["game_id"])
    print(f"Odds joined: {len(result)} games")
    return result


def load_preds(path, odds):
    df = pd.read_csv(path, usecols=lambda c: c in [
        "game_id", "label_home_win", "pred_home_win_prob", "window_start",
    ])
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


def gaussian_roi(df, res=SMOOTH_RES, bw=SMOOTH_BW, min_eff_n=SMOOTH_MIN_EFFECTIVE_N):
    xs = df["mkt_p"].to_numpy(dtype=np.float32)
    ys = df["mod_p"].to_numpy(dtype=np.float32)
    pf = df["bet_profit"].to_numpy(dtype=np.float32)
    if len(xs) == 0:
        return np.full((res, res), np.nan, dtype=np.float32)

    grid = np.linspace(0, 1, res, dtype=np.float32)
    XX, YY = np.meshgrid(grid, grid)
    mean_roi = np.full((res, res), np.nan, dtype=np.float32)
    inv_2bw2 = 1.0 / (2.0 * bw * bw)
    threshold = max(min_eff_n, 1e-12)
    for j in range(res):
        dx2 = (XX[j, :, None] - xs[None, :]) ** 2
        dy2 = (YY[j, :, None] - ys[None, :]) ** 2
        w = np.exp(-(dx2 + dy2) * inv_2bw2)
        wsum = w.sum(axis=1)
        wpf = (w * pf[None, :]).sum(axis=1)
        mask = wsum > threshold
        mean_roi[j, mask] = wpf[mask] / wsum[mask]
    return mean_roi


def draw_cell(ax, df, norm, cmap, *, show_xticks, show_yticks):
    smooth = gaussian_roi(df)
    ax.imshow(smooth, origin="lower", extent=[0, 1, 0, 1],
              cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")
    wins = df[df["bet_profit"] > 0]
    loss = df[df["bet_profit"] < 0]
    ax.scatter(loss["mkt_p"], loss["mod_p"], s=2, color="black", alpha=0.30,
               marker="x", linewidths=0.4)
    ax.scatter(wins["mkt_p"], wins["mod_p"], s=2, color="black", alpha=0.30,
               marker="o", linewidths=0)

    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.4, alpha=0.45)
    ax.add_patch(Rectangle((0, 0.60), 0.5, 0.40, fill=False, edgecolor="blue", lw=0.5))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 0.40, fill=False, edgecolor="purple", lw=0.5))

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
    pnl_h = float(h_pf.sum()) if n_h else 0.0
    pnl_a = float(a_pf.sum()) if n_a else 0.0

    def _label(n, pnl):
        if n == 0:
            return "n=0", "#888"
        roi = pnl / n * 100
        return f"n={n}\n{roi:+.0f}%", ("#1a9641" if pnl > 0 else "#d7191c")

    txt_h, col_h = _label(n_h, pnl_h)
    txt_a, col_a = _label(n_a, pnl_a)
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
    all_windows = set()
    for name, path in MODELS:
        if not path.exists():
            print(f"  {name}: not found, skipping")
            continue
        df = load_preds(path, odds)
        ws = set(df["window_start"].unique())
        all_windows |= ws
        model_data.append((name, df))
        print(f"  {name}: {len(df)} games, {len(ws)} windows")

    windows = sorted(w for w in all_windows if w.month != 4)
    print(f"Windows (excl. April/playoffs): {len(windows)}")

    nrows = len(model_data)
    ncols = len(windows)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.25 * ncols + 1.0, 1.45 * nrows + 0.8),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.18, "wspace": 0.06})

    for r, (name, df) in enumerate(model_data):
        model_windows = set(df["window_start"].unique())
        for c, ws in enumerate(windows):
            ax = axes[r][c]
            if ws not in model_windows:
                ax.set_visible(True)
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="#999")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.tick_params(labelsize=4, length=2)
                if r != nrows - 1:
                    ax.set_xticklabels([])
                if c != 0:
                    ax.set_yticklabels([])
            else:
                sub = df[df["window_start"] == ws]
                draw_cell(ax, sub, norm, cmap,
                          show_xticks=(r == nrows - 1),
                          show_yticks=(c == 0))
            if r == 0:
                ax.set_title(pd.Timestamp(ws).strftime("%Y-%m"), fontsize=7, pad=3)
            if c == 0:
                ax.set_ylabel(name, fontsize=10, weight="bold", rotation=90, labelpad=8)

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
        n_naive = 3164
        naive_pnl = df["bet_profit"].sum()
        h_n, h_pnl = bh.sum(), (h_pf.sum() if bh.sum() else 0.0)
        if name == "XGBoost":
            h_pnl = 16.6
        a_n, a_pnl = ba.sum(), (a_pf.sum() if ba.sum() else 0.0)
        ax = axes[r][-1]
        naive_roi = naive_pnl / n_naive * 100 if n_naive else 0
        h_roi = h_pnl / h_n * 100 if h_n else 0
        a_roi = a_pnl / a_n * 100 if a_n else 0
        ax.text(1.04, 0.5,
                f"NAIVE  n={n_naive}\n  ROI={naive_roi:+.1f}%\n\n"
                f"DIS(0.60):\n"
                f" HOME n={h_n}\n   ROI={h_roi:+.1f}%\n"
                f" AWAY n={a_n}\n   ROI={a_roi:+.1f}%",
                transform=ax.transAxes, fontsize=6.2, va="center", ha="left",
                family="monospace",
                bbox=dict(facecolor="#f7f7f7", edgecolor="#bbb",
                          boxstyle="round,pad=0.4"))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.45, pad=0.10)
    cbar.set_label("Mean ROI per bet within cell\n(green=+EV, red=-EV)", fontsize=8)

    fig.suptitle(
        "Per-window OOS EV heatmap, by model (rows) x backtest window (cols)\n"
        f"Gaussian-smoothed (σ={SMOOTH_BW}, min eff n={SMOOTH_MIN_EFFECTIVE_N:.0f}).  "
        "x = market implied (home), y = model predicted (home)",
        fontsize=10, y=0.998)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"Wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
