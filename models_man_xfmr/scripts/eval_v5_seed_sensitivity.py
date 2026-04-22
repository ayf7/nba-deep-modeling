#!/usr/bin/env python3
"""Head-to-head EV analysis on disagree(0.60) HOME-dog across v5 seeds.

Loads predictions for v5 seeds {7, 11, 23, 47} (and the original v5 reference
backtest), joins with odds + rest-days, and reports per-seed and pooled ROI
for the d60_HOME_dog slice with and without the B2B home-rest filter.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("/home/ayf7/trading/models_man_xfmr/artifacts")
SEEDS = {
    "v5_ref":  ART / "backtest_v5_pstats_2stream_apoiss10" / "predictions.csv",
    "seed7":   ART / "backtest_v5_seed7"  / "predictions.csv",
    "seed11":  ART / "backtest_v5_seed11" / "predictions.csv",
    "seed23":  ART / "backtest_v5_seed23" / "predictions.csv",
    "seed47":  ART / "backtest_v5_seed47" / "predictions.csv",
}
CORE = Path("/home/ayf7/trading/data/artifacts/nba_core.sqlite")
FEATS = Path("/home/ayf7/trading/data/artifacts/features.sqlite")


def load_odds_and_rest():
    with sqlite3.connect(CORE) as con:
        odds = pd.read_sql_query(
            """SELECT game_id, home_avg_decimal_odds, away_avg_decimal_odds,
                      home_implied_prob_normalized
               FROM game_moneyline_odds WHERE has_moneyline_odds=1""", con)
    with sqlite3.connect(FEATS) as con:
        rest = pd.read_sql_query(
            "SELECT game_id, home_rest_days FROM model_games", con)
    odds["game_id"] = odds["game_id"].astype(str)
    rest["game_id"] = rest["game_id"].astype(str)
    return odds.merge(rest, on="game_id")


def home_dog_stats(df, mask, label, n_boot=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    sub = df[mask]
    if len(sub) == 0:
        return {"label": label, "n": 0, "roi": 0.0, "win_rate": 0.0,
                "lo": 0.0, "hi": 0.0}
    h_dec = sub["home_avg_decimal_odds"].to_numpy()
    won = sub["label_home_win"].to_numpy() == 1
    profit = np.where(won, h_dec - 1, -1)
    n = len(profit)
    roi = profit.mean()
    boot = np.array([profit[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"label": label, "n": n, "wins": int(won.sum()),
            "roi": float(roi), "win_rate": float(won.mean()),
            "lo": float(lo), "hi": float(hi)}


def main():
    od = load_odds_and_rest()
    all_dfs = {}
    print(f"{'seed':>10s}  {'rule':28s} {'n':>4s} {'wins':>4s} "
          f"{'roi':>9s} {'win_r':>7s} {'95%-CI':>22s}")
    print("-" * 95)

    pooled_hd = []
    pooled_hd_b2b = []

    for name, path in SEEDS.items():
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        df = pd.read_csv(path, usecols=["game_id", "label_home_win",
                                        "pred_home_win_prob"])
        df["game_id"] = df["game_id"].astype(str)
        df = df.merge(od, on="game_id", how="inner")
        df["b2b_home"] = (df["home_rest_days"].fillna(99) <= 1).astype(int)
        p = df["pred_home_win_prob"].to_numpy()
        mkt = df["home_implied_prob_normalized"].to_numpy()
        hd = (p >= 0.60) & (mkt < 0.5)
        hd_b2b = hd & (df["b2b_home"] == 0)

        rng = np.random.default_rng(hash(name) & 0xFFFF)
        s1 = home_dog_stats(df, hd, "d60_HOME_dog", rng=rng)
        s2 = home_dog_stats(df, hd_b2b, "d60_HOME_dog B2B-filt", rng=rng)

        for s in (s1, s2):
            ci = f"[{s['lo']*100:+5.2f}%,{s['hi']*100:+5.2f}%]"
            print(f"{name:>10s}  {s['label']:28s} {s['n']:>4d} {s['wins']:>4d} "
                  f"{s['roi']*100:>+7.2f}% {s['win_rate']*100:>6.1f}% {ci:>22s}")

        all_dfs[name] = df
        if name != "v5_ref":  # pool across the 4 new seeds only
            pooled_hd.append(df[hd][["home_avg_decimal_odds", "label_home_win"]])
            pooled_hd_b2b.append(df[hd_b2b][["home_avg_decimal_odds", "label_home_win"]])

    print()
    print("=== pooled across seeds {7,11,23,47} (concatenated bets) ===")
    rng = np.random.default_rng(123)
    for label, parts in [("d60_HOME_dog (pooled)", pooled_hd),
                         ("d60_HOME_dog B2B-filt (pooled)", pooled_hd_b2b)]:
        cat = pd.concat(parts, ignore_index=True)
        h = cat["home_avg_decimal_odds"].to_numpy()
        won = cat["label_home_win"].to_numpy() == 1
        profit = np.where(won, h - 1, -1)
        n = len(profit)
        if n == 0:
            print(f"{label:32s} n=0")
            continue
        roi = profit.mean()
        boot = np.array([profit[rng.integers(0, n, n)].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{label:32s} n={n:4d} wins={int(won.sum()):4d} "
              f"roi={roi*100:+7.2f}% win_r={won.mean()*100:5.1f}%  "
              f"95%-CI=[{lo*100:+.2f}%, {hi*100:+.2f}%]")

    # Cross-seed agreement: per-game how often does the d60 HOME-dog flag fire?
    print()
    print("=== seed-agreement on d60_HOME_dog flag ===")
    seed_names = [n for n in SEEDS if n != "v5_ref" and n in all_dfs]
    if len(seed_names) >= 2:
        ref_df = all_dfs[seed_names[0]][["game_id"]].copy()
        flag_cols = []
        for n in seed_names:
            d = all_dfs[n].set_index("game_id")
            flag = ((d["pred_home_win_prob"] >= 0.60)
                    & (d["home_implied_prob_normalized"] < 0.5)).astype(int)
            ref_df[n] = ref_df["game_id"].map(flag).fillna(0).astype(int)
            flag_cols.append(n)
        ref_df["k_seeds_flag"] = ref_df[flag_cols].sum(axis=1)
        # join odds + label for k-seeds = 4 (unanimous) ROI
        merged = ref_df.merge(od, on="game_id", how="inner")
        merged = merged.merge(
            all_dfs[seed_names[0]][["game_id", "label_home_win", "b2b_home"]],
            on="game_id", how="inner")
        for k in [1, 2, 3, 4]:
            mask = merged["k_seeds_flag"] >= k
            sub = merged[mask]
            if len(sub) == 0:
                continue
            h = sub["home_avg_decimal_odds"].to_numpy()
            won = sub["label_home_win"].to_numpy() == 1
            profit = np.where(won, h - 1, -1)
            print(f"  k>={k}: n={len(sub):4d}  wins={int(won.sum()):4d}  "
                  f"roi={profit.mean()*100:+7.2f}%  win_r={won.mean()*100:5.1f}%")
            sub_b2b = sub[sub["b2b_home"] == 0]
            if len(sub_b2b) > 0:
                hb = sub_b2b["home_avg_decimal_odds"].to_numpy()
                wb = sub_b2b["label_home_win"].to_numpy() == 1
                pb = np.where(wb, hb - 1, -1)
                print(f"    + B2B-filt: n={len(sub_b2b):4d}  wins={int(wb.sum()):4d}  "
                      f"roi={pb.mean()*100:+7.2f}%  win_r={wb.mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
