#!/usr/bin/env python3
"""Betting evaluation for CME-v6 backtest checkpoints.

Loads each window's checkpoint, runs AR inference on its test set,
joins with moneyline odds from oddsportal_moneyline.sqlite (via game_date +
team names + scores), and evaluates betting strategies:
  - Edge threshold sweep (model prob - market devigged prob)
  - Disagree(0.60) slices (home-dog and away-dog)
  - Naive bet-the-pick ROI
  - EV heatmap plot

Outputs:
  - predictions.csv (all windows combined)
  - betting_summary.txt (strategy table)
  - ev_heatmap.png
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cme_v5_common import (
    PrecomputedDatasetV5, collate_v5,
    load_precomputed_window_info, load_precomputed_vocab_size,
)
from model import CmeV6, CmeV6Config


DEFAULT_CORE_DB = REPO_ROOT / "data" / "artifacts" / "nba_core.sqlite"
DEFAULT_ODDS_DB = REPO_ROOT / "data" / "artifacts" / "oddsportal_moneyline.sqlite"
DEFAULT_PRECOMPUTED_DB = REPO_ROOT / "data" / "features_v5_precomputed.db"
DEFAULT_CKPT_DIR = REPO_ROOT / "nba_transformer" / "artifacts" / "backtest_dec" / "checkpoints"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "nba_transformer" / "artifacts" / "betting_eval"

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--precomputed-db", type=Path, default=DEFAULT_PRECOMPUTED_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--odds-db", type=Path, default=DEFAULT_ODDS_DB)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--n-enc", type=int, default=2)
    p.add_argument("--n-dec", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--compare", action="store_true",
                   help="Multi-panel EV heatmap comparing all model families")
    return p.parse_args()


def load_odds_by_game_id(core_db, odds_db):
    """Join oddsportal odds to game_ids via (date, teams, scores)."""
    odds_con = sqlite3.connect(str(odds_db))
    odds_df = pd.read_sql_query(
        """SELECT game_date_et, home_name, away_name, home_result, away_result,
                  home_avg_decimal_odds, away_avg_decimal_odds,
                  home_implied_prob_normalized, away_implied_prob_normalized,
                  has_moneyline_odds
           FROM moneyline_odds
           WHERE has_moneyline_odds = 1 AND status_id IN (3, 10)""",
        odds_con,
    )
    odds_con.close()

    odds_df["home_tri"] = odds_df["home_name"].map(TEAM_NAME_TO_TRICODE)
    odds_df["away_tri"] = odds_df["away_name"].map(TEAM_NAME_TO_TRICODE)
    odds_df = odds_df.dropna(subset=["home_tri", "away_tri"])

    core_con = sqlite3.connect(str(core_db))
    games_df = pd.read_sql_query(
        """SELECT game_id, game_date, home_team_abbr, away_team_abbr,
                  home_score, away_score
           FROM games""",
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
                      "home_implied_prob_normalized", "away_implied_prob_normalized"]].copy()
    result = result.drop_duplicates(subset=["game_id"])

    # Compute B2B from game schedule: for each game, check if home/away team
    # played the day before
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    team_dates = {}
    for _, row in games_df.iterrows():
        for col in ["home_team_abbr", "away_team_abbr"]:
            team_dates.setdefault(row[col], set()).add(row["game_date"])

    b2b_home = []
    b2b_away = []
    for _, row in games_df.iterrows():
        prev = row["game_date"] - pd.Timedelta(days=1)
        b2b_home.append(prev in team_dates.get(row["home_team_abbr"], set()))
        b2b_away.append(prev in team_dates.get(row["away_team_abbr"], set()))
    games_df["home_b2b"] = b2b_home
    games_df["away_b2b"] = b2b_away
    games_df["game_id"] = games_df["game_id"].astype(str)
    b2b_df = games_df[["game_id", "home_b2b", "away_b2b"]]
    result = result.merge(b2b_df, on="game_id", how="left")

    return result


@torch.no_grad()
def predict_window(model, ds, device, batch_size):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_v5)
    all_probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, teacher_force=False, autoregressive=True)
        probs = torch.sigmoid(out["win_logit"]).cpu().numpy()
        all_probs.append(probs)
    all_probs = np.concatenate(all_probs)

    rows = []
    for i, gid in enumerate(ds._game_ids):
        g = ds._games[gid]
        rows.append({
            "game_id": gid,
            "label_home_win": int(g["label"]),
            "pred_home_win_prob": float(all_probs[i]),
            "home_rest": float(g["home_rest"]),
            "away_rest": float(g["away_rest"]),
        })
    return rows


def evaluate_thresholds(df, margins):
    p = df["pred_home_win_prob"].to_numpy()
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()
    y = df["label_home_win"].to_numpy()

    devigged = (1.0 / h_dec) / (1.0 / h_dec + 1.0 / a_dec)
    edge = p - devigged
    r_home = np.where(y > 0.5, h_dec - 1.0, -1.0)
    r_away = np.where(y > 0.5, -1.0, a_dec - 1.0)

    rows = []
    for m in margins:
        bet_home = edge > m
        bet_away = edge < -m
        n_bet = int((bet_home | bet_away).sum())
        n_h = int(bet_home.sum())
        n_a = int(bet_away.sum())
        realized = np.where(bet_home, r_home, np.where(bet_away, r_away, 0.0))
        sum_r = float(realized.sum())
        sum_h = float(realized[bet_home].sum()) if n_h else 0.0
        sum_a = float(realized[bet_away].sum()) if n_a else 0.0
        h_wins = int((bet_home & (y > 0.5)).sum())
        a_wins = int((bet_away & (y < 0.5)).sum())
        rows.append({
            "margin": m, "n": len(p), "n_bet": n_bet, "n_home": n_h, "n_away": n_a,
            "bet_rate": n_bet / len(p) if len(p) else 0,
            "roi_bet": sum_r / n_bet if n_bet else 0,
            "roi_game": sum_r / len(p) if len(p) else 0,
            "roi_home": sum_h / n_h if n_h else 0,
            "roi_away": sum_a / n_a if n_a else 0,
            "home_wr": h_wins / n_h if n_h else 0,
            "away_wr": a_wins / n_a if n_a else 0,
        })
    return rows


def evaluate_disagree(df):
    p = df["pred_home_win_prob"].to_numpy()
    mkt = df["home_implied_prob_normalized"].to_numpy()
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()
    y = df["label_home_win"].to_numpy()
    # home_rest is rest_days/7 from precomputed DB; B2B = rest_days <= 1 → rest <= 1/7
    b2b_home = (df["home_rest"].to_numpy() <= 1.0 / 7.0 + 0.001)

    slices = {}
    for tau in [0.55, 0.60, 0.65]:
        home_dog = (p >= tau) & (mkt < 0.5)
        away_dog = ((1 - p) >= tau) & (mkt >= 0.5)
        home_dog_nob2b = home_dog & ~b2b_home

        for name, mask, side in [
            (f"HOME_dog d{int(tau*100)}", home_dog, "home"),
            (f"HOME_dog d{int(tau*100)} no-B2B", home_dog_nob2b, "home"),
            (f"AWAY_dog d{int(tau*100)}", away_dog, "away"),
        ]:
            sub_y = y[mask]
            n = int(mask.sum())
            if n == 0:
                slices[name] = {"n": 0, "wins": 0, "roi": 0, "wr": 0}
                continue
            if side == "home":
                profit = np.where(sub_y > 0.5, h_dec[mask] - 1, -1)
                wins = int((sub_y > 0.5).sum())
            else:
                profit = np.where(sub_y < 0.5, a_dec[mask] - 1, -1)
                wins = int((sub_y < 0.5).sum())
            slices[name] = {"n": n, "wins": wins, "roi": float(profit.mean()), "wr": wins / n}
    return slices


def naive_bet_roi(df):
    p = df["pred_home_win_prob"].to_numpy()
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()
    y = df["label_home_win"].to_numpy()
    bet_home = p >= 0.5
    profit = np.where(bet_home,
                      np.where(y > 0.5, h_dec - 1, -1),
                      np.where(y < 0.5, a_dec - 1, -1))
    return float(profit.mean()), len(p)


def gaussian_roi(df, res=200, bw=0.05, min_n=8.0):
    p = df["pred_home_win_prob"].to_numpy()
    mkt = df["home_implied_prob_normalized"].to_numpy()
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()
    y = df["label_home_win"].to_numpy()
    bet_home = p >= 0.5
    profit = np.where(bet_home,
                      np.where(y > 0.5, h_dec - 1, -1),
                      np.where(y < 0.5, a_dec - 1, -1))

    gx = np.linspace(0, 1, res)
    gy = np.linspace(0, 1, res)
    Gx, Gy = np.meshgrid(gx, gy)

    roi = np.full((res, res), np.nan)
    for i in range(res):
        for j in range(res):
            dx = mkt - Gx[i, j]
            dy = p - Gy[i, j]
            w = np.exp(-0.5 * (dx**2 + dy**2) / bw**2)
            eff_n = w.sum()
            if eff_n >= min_n:
                roi[i, j] = np.average(profit, weights=w)
    return roi


def plot_heatmap(df, output_path):
    roi = gaussian_roi(df)
    naive_roi, n = naive_bet_roi(df)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    im = ax.imshow(roi, origin="lower", extent=[0, 1, 0, 1],
                   cmap="RdYlGn", norm=norm, aspect="auto", interpolation="bilinear")

    p = df["pred_home_win_prob"].to_numpy()
    mkt = df["home_implied_prob_normalized"].to_numpy()
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()
    y = df["label_home_win"].to_numpy()
    bet_home = p >= 0.5
    profit = np.where(bet_home,
                      np.where(y > 0.5, h_dec - 1, -1),
                      np.where(y < 0.5, a_dec - 1, -1))

    wins_m = profit > 0
    ax.scatter(mkt[~wins_m], p[~wins_m], s=3, color="black", alpha=0.08, marker="x", linewidths=0.4)
    ax.scatter(mkt[wins_m], p[wins_m], s=3, color="black", alpha=0.08, marker="o", linewidths=0)

    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.7, alpha=0.5)
    ax.add_patch(Rectangle((0, 0.60), 0.5, 0.40, fill=False, edgecolor="blue", lw=1.3,
                            label="HOME-dog disagree(0.60)"))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 0.40, fill=False, edgecolor="purple", lw=1.3,
                            label="AWAY-dog disagree(0.60)"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("market implied prob (home)")
    ax.set_ylabel("model predicted prob (home)")
    ax.set_title(f"CME-v6 decoder · n={n}, naive-bet ROI={naive_roi*100:+.2f}%")
    ax.legend(fontsize=8, loc="lower right")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85, label="smoothed ROI per bet")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    args = parse_args()
    if args.compare:
        run_compare(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.precomputed_db
    windows = load_precomputed_window_info(db_path)

    valid_windows = []
    for ws, we in windows:
        ds = PrecomputedDatasetV5(db_path, ws, "test")
        if len(ds) >= 5:
            valid_windows.append(ws)

    print(f"Found {len(valid_windows)} valid windows")

    # Load odds mapping (game_id → odds)
    print("Loading odds...", end=" ", flush=True)
    odds_df = load_odds_by_game_id(args.core_db, args.odds_db)
    print(f"{len(odds_df)} games with odds")

    all_rows = []
    for ws in valid_windows:
        ckpt_path = args.ckpt_dir / f"model_{ws}.pt"
        if not ckpt_path.exists():
            print(f"  {ws}: no checkpoint, skipping")
            continue

        test_ds = PrecomputedDatasetV5(db_path, ws, "test")
        train_ds = PrecomputedDatasetV5(db_path, ws, "train")
        vocab_size, team_vocab_size = load_precomputed_vocab_size(db_path, ws)
        sample = train_ds[0]
        stats_dim = sample["home_stats"].size(-1) if sample["home_stats"].numel() > 0 else 0

        cfg = CmeV6Config(
            vocab_size=vocab_size, num_teams=team_vocab_size,
            tabular_dim=sample["tabular"].numel(),
            d=args.d, n_heads=args.n_heads,
            n_enc=args.n_enc, n_dec=args.n_dec,
            dropout=args.dropout,
            player_stats_dim=stats_dim,
        )
        model = CmeV6(cfg).to(args.device)
        state = torch.load(ckpt_path, map_location=args.device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        rows = predict_window(model, test_ds, args.device, args.batch_size)
        print(f"  {ws}: {len(rows)} games, "
              f"acc={sum(1 for r in rows if (r['pred_home_win_prob']>=0.5)==(r['label_home_win']==1))/len(rows):.3f}")
        for r in rows:
            r["window_start"] = ws
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"\nTotal predictions: {len(df)}")

    # Join odds
    df["game_id"] = df["game_id"].astype(str)
    df = df.merge(odds_df, on="game_id", how="left")

    # Save full predictions
    df.to_csv(args.output_dir / "predictions.csv", index=False)
    print(f"Saved {args.output_dir / 'predictions.csv'}")

    # Filter to games with odds
    df_odds = df.dropna(subset=["home_avg_decimal_odds"]).copy()
    print(f"Games with odds: {len(df_odds)} / {len(df)}")

    if len(df_odds) == 0:
        print("No games with odds — cannot evaluate betting.")
        return

    # --- Edge threshold sweep ---
    margins = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    thresh_rows = evaluate_thresholds(df_odds, margins)

    lines = []
    lines.append(f"\n=== Edge threshold sweep (n={len(df_odds)} games with odds) ===")
    hdr = f"{'m':>5} {'n_bet':>6} {'bet%':>6} {'n_h':>5} {'n_a':>5} {'ROI/bet':>9} {'ROI/gm':>9} {'wr_h':>6} {'wr_a':>6}"
    lines.append(hdr)
    for r in thresh_rows:
        lines.append(
            f"{r['margin']:>5.2f} {r['n_bet']:>6d} {r['bet_rate']*100:>5.1f}% "
            f"{r['n_home']:>5d} {r['n_away']:>5d} "
            f"{r['roi_bet']*100:>+8.2f}% {r['roi_game']*100:>+8.2f}% "
            f"{r['home_wr']*100:>5.1f}% {r['away_wr']*100:>5.1f}%"
        )

    # --- Disagree slices ---
    disagree = evaluate_disagree(df_odds)
    lines.append(f"\n=== Disagree slices ===")
    lines.append(f"{'rule':30s} {'n':>5s} {'wins':>5s} {'roi':>8s} {'wr':>7s}")
    for name, s in disagree.items():
        lines.append(f"{name:30s} {s['n']:>5d} {s['wins']:>5d} {s['roi']*100:>+7.2f}% {s['wr']*100:>6.1f}%")

    # --- Naive bet-the-pick ---
    naive_roi, n_naive = naive_bet_roi(df_odds)
    lines.append(f"\n=== Naive bet-the-pick: ROI={naive_roi*100:+.2f}% (n={n_naive}) ===")

    # --- Per-window breakdown ---
    lines.append(f"\n=== Per-window naive ROI ===")
    lines.append(f"{'window':>12s} {'n':>5s} {'n_odds':>7s} {'acc':>6s} {'roi':>8s}")
    for ws in sorted(df["window_start"].unique()):
        sub_all = df[df["window_start"] == ws]
        sub = df_odds[df_odds["window_start"] == ws]
        if len(sub) == 0:
            acc_all = ((sub_all["pred_home_win_prob"] >= 0.5) == (sub_all["label_home_win"] == 1)).mean()
            lines.append(f"{ws:>12s} {len(sub_all):>5d} {'0':>7s} {acc_all*100:>5.1f}%     n/a")
            continue
        roi_w, n_w = naive_bet_roi(sub)
        acc_w = ((sub["pred_home_win_prob"] >= 0.5) == (sub["label_home_win"] == 1)).mean()
        lines.append(f"{ws:>12s} {len(sub_all):>5d} {n_w:>7d} {acc_w*100:>5.1f}% {roi_w*100:>+7.2f}%")

    report = "\n".join(lines)
    print(report)
    with open(args.output_dir / "betting_summary.txt", "w") as f:
        f.write(report + "\n")

    # --- EV heatmap ---
    print("\nGenerating EV heatmap (takes ~30s)...", flush=True)
    plot_heatmap(df_odds, args.output_dir / "ev_heatmap.png")

    print(f"\nAll artifacts saved to {args.output_dir}")


COMPARE_MODELS = [
    ("logistic",  REPO_ROOT / "models_baseline" / "artifacts" / "backtest_logistic" / "predictions.csv"),
    ("xgboost",   REPO_ROOT / "models_baseline" / "artifacts" / "backtest_xgboost" / "predictions.csv"),
    ("mlp",       REPO_ROOT / "models_baseline" / "artifacts" / "backtest_mlp" / "predictions.csv"),
    ("v5",        REPO_ROOT / "models_man_xfmr" / "artifacts" / "backtest_v5_pstats_2stream_apoiss10" / "predictions.csv"),
    ("cme_v1",    REPO_ROOT / "models_cme_v1" / "artifacts" / "backtest_v1" / "predictions.csv"),
    ("cme_v2",    REPO_ROOT / "models_cme_v2" / "artifacts" / "backtest_s_tt" / "predictions.csv"),
    ("cme_v4",    REPO_ROOT / "models_cme_v4" / "artifacts" / "full_v4" / "predictions.csv"),
    ("nba_transformer", REPO_ROOT / "nba_transformer" / "artifacts" / "betting_eval" / "predictions.csv"),
]


def load_model_preds(path, odds_df):
    df = pd.read_csv(path, usecols=lambda c: c in [
        "game_id", "label_home_win", "pred_home_win_prob", "home_rest", "away_rest",
    ])
    df["game_id"] = df["game_id"].astype(str)
    df = df.merge(odds_df, on="game_id", how="inner")
    return df


def plot_panel(ax, df, name, norm, cmap):
    roi = gaussian_roi(df)
    naive_roi, n = naive_bet_roi(df)

    im = ax.imshow(roi, origin="lower", extent=[0, 1, 0, 1],
                   cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")

    p = df["pred_home_win_prob"].to_numpy()
    mkt = df["home_implied_prob_normalized"].to_numpy()
    h_dec = df["home_avg_decimal_odds"].to_numpy()
    a_dec = df["away_avg_decimal_odds"].to_numpy()
    y = df["label_home_win"].to_numpy()
    bet_home = p >= 0.5
    profit = np.where(bet_home,
                      np.where(y > 0.5, h_dec - 1, -1),
                      np.where(y < 0.5, a_dec - 1, -1))

    wins_m = profit > 0
    ax.scatter(mkt[~wins_m], p[~wins_m], s=2, color="black", alpha=0.06, marker="x", linewidths=0.3)
    ax.scatter(mkt[wins_m], p[wins_m], s=2, color="black", alpha=0.06, marker="o", linewidths=0)

    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.7, alpha=0.5)
    ax.add_patch(Rectangle((0, 0.60), 0.5, 0.40, fill=False, edgecolor="blue", lw=1.0))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 0.40, fill=False, edgecolor="purple", lw=1.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    acc = ((p >= 0.5) == (y > 0.5)).mean()
    ax.set_title(f"{name}\nn={n}  acc={acc*100:.1f}%  ROI={naive_roi*100:+.1f}%", fontsize=9)
    ax.tick_params(labelsize=7)
    return im


def run_compare(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading odds...", end=" ", flush=True)
    odds_df = load_odds_by_game_id(args.core_db, args.odds_db)
    print(f"{len(odds_df)} games with odds")

    models = []
    for name, path in COMPARE_MODELS:
        if not path.exists():
            print(f"  {name}: {path} not found, skipping")
            continue
        df = load_model_preds(path, odds_df)
        if len(df) == 0:
            print(f"  {name}: no games with odds after join, skipping")
            continue
        models.append((name, df))
        print(f"  {name}: {len(df)} games with odds")

    n_models = len(models)
    if n_models == 0:
        print("No models with odds data.")
        return

    # Layout: 2 rows x 4 cols
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 5.0 * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap = "RdYlGn"

    last_im = None
    for idx, (name, df) in enumerate(models):
        r, c = divmod(idx, n_cols)
        print(f"  Plotting {name}...", flush=True)
        last_im = plot_panel(axes[r, c], df, name, norm, cmap)

    for idx in range(n_models, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    axes[0, 0].set_ylabel("model pred prob (home)", fontsize=9)
    if n_rows > 1:
        axes[1, 0].set_ylabel("model pred prob (home)", fontsize=9)
    for c in range(min(n_cols, n_models)):
        axes[n_rows - 1, c].set_xlabel("market implied prob (home)", fontsize=8)

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02, shrink=0.85)
    cbar.set_label("smoothed ROI per bet", fontsize=9)

    fig.suptitle("Model comparison · EV heatmap (test splits, games with odds)", fontsize=13)
    out_path = args.output_dir / "ev_heatmap_compare.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    # Summary table
    lines = []
    lines.append(f"\n{'model':12s} {'n':>5s} {'acc':>6s} {'naive ROI':>10s} {'d60 HOME no-B2B':>18s} {'d60 AWAY':>12s}")
    for name, df in models:
        naive_roi_val, n = naive_bet_roi(df)
        acc = ((df["pred_home_win_prob"] >= 0.5) == (df["label_home_win"] > 0.5)).mean()
        p = df["pred_home_win_prob"].to_numpy()
        mkt = df["home_implied_prob_normalized"].to_numpy()
        h_dec = df["home_avg_decimal_odds"].to_numpy()
        a_dec = df["away_avg_decimal_odds"].to_numpy()
        y = df["label_home_win"].to_numpy()

        home_dog = (p >= 0.60) & (mkt < 0.5)
        away_dog = ((1 - p) >= 0.60) & (mkt >= 0.5)
        if "home_b2b" in df.columns:
            b2b = df["home_b2b"].to_numpy().astype(bool)
        elif "home_rest" in df.columns:
            b2b = df["home_rest"].to_numpy() <= 1.0 / 7.0 + 0.001
        else:
            b2b = np.zeros(len(df), dtype=bool)
        hd_mask = home_dog & ~b2b

        def roi_str(mask, side):
            nn = int(mask.sum())
            if nn == 0:
                return "n=0"
            sub_y = y[mask]
            if side == "home":
                profit = np.where(sub_y > 0.5, h_dec[mask] - 1, -1)
            else:
                profit = np.where(sub_y < 0.5, a_dec[mask] - 1, -1)
            return f"{profit.mean()*100:+.1f}% ({nn})"

        hd_str = roi_str(hd_mask, "home")
        ad_str = roi_str(away_dog, "away")
        lines.append(f"{name:12s} {n:>5d} {acc*100:>5.1f}% {naive_roi_val*100:>+9.1f}% {hd_str:>18s} {ad_str:>12s}")

    report = "\n".join(lines)
    print(report)
    with open(args.output_dir / "compare_summary.txt", "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
