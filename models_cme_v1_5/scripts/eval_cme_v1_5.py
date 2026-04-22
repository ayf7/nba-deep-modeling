#!/usr/bin/env python3
"""Evaluate CME-v1.5 selector against v5 disagree(0.60) baseline.

Reads predictions.csv from the cme_v1.5 backtest. For each decision rule
(argmax, p_max_bet > tau for several tau, learned-bet AND model_p in disagree
slice), reports n / ROI / win-rate.

Compares against v5 backtest disagree(0.60) HOME-dog (B2B-filtered) as the
known baseline.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

CME15_PRED = Path("/home/ayf7/trading/models_cme_v1_5/artifacts/backtest_v1_5/predictions.csv")
V5_PRED = Path("/home/ayf7/trading/models_man_xfmr/artifacts/backtest_v5_pstats_2stream_apoiss10/predictions.csv")
CORE = Path("/home/ayf7/trading/data/artifacts/nba_core.sqlite")


def load_odds_and_rest():
    with sqlite3.connect(CORE) as con:
        odds = pd.read_sql_query(
            """SELECT game_id, home_avg_decimal_odds, away_avg_decimal_odds,
                      home_implied_prob_normalized
               FROM game_moneyline_odds WHERE has_moneyline_odds=1""", con)
        rest = pd.read_sql_query(
            "SELECT game_id, home_rest_days, away_rest_days FROM games", con)
    odds["game_id"] = odds["game_id"].astype(str)
    rest["game_id"] = rest["game_id"].astype(str)
    return odds.merge(rest, on="game_id")


def slice_stats(df, mask, side, label="", h_dec_col="home_avg_decimal_odds",
                a_dec_col="away_avg_decimal_odds"):
    """side: 'home' or 'away' or 'auto' (use sel_action)."""
    sub = df[mask]
    if len(sub) == 0:
        return {"label": label, "n": 0, "wins": 0, "losses": 0,
                "roi": 0.0, "win_rate": 0.0}
    h_dec = sub[h_dec_col].to_numpy()
    a_dec = sub[a_dec_col].to_numpy()
    won = sub["label_home_win"].to_numpy() == 1
    if side == "home":
        profit = np.where(won, h_dec - 1, -1)
        nwins = int(won.sum())
    elif side == "away":
        profit = np.where(~won, a_dec - 1, -1)
        nwins = int((~won).sum())
    elif side == "auto":
        a = sub["sel_action"].to_numpy()
        profit = np.where(a == 0, np.where(won, h_dec - 1, -1),
                          np.where(a == 1, np.where(~won, a_dec - 1, -1), 0))
        nwins = int(((a == 0) & won).sum() + ((a == 1) & ~won).sum())
    else:
        raise ValueError(side)
    n = len(sub)
    return {"label": label, "n": n, "wins": nwins, "losses": n - nwins,
            "roi": float(profit.mean()), "win_rate": nwins / n}


def main():
    if not CME15_PRED.exists():
        print(f"[skip] {CME15_PRED} doesn't exist yet")
        return
    cme = pd.read_csv(CME15_PRED)
    cme["game_id"] = cme["game_id"].astype(str)
    print(f"[load] cme_v1.5 predictions: n={len(cme)}, has_odds={int(cme['has_odds'].sum())}")

    od = load_odds_and_rest()
    df = cme.merge(od, on="game_id", how="left")
    df = df[df["has_odds"] == 1].copy()
    df["p_max_bet"] = df[["sel_p_home", "sel_p_away"]].max(axis=1)
    df["b2b_home"] = (df["home_rest_days"].fillna(99) <= 1).astype(int)

    print(f"[merged] n_with_odds = {len(df)}")
    print()
    print("=== CME-v1.5 selector: action distribution ===")
    print(df["sel_action"].value_counts().sort_index().to_dict(),
          " (0=home,1=away,2=no_bet)")
    print(f"sel_p_nobet quantiles: "
          f"{df['sel_p_nobet'].quantile([0.1,0.5,0.9]).round(3).to_dict()}")
    print(f"p_max_bet quantiles : "
          f"{df['p_max_bet'].quantile([0.1,0.5,0.9]).round(3).to_dict()}")
    print()

    rows = []
    # argmax
    rows.append(slice_stats(df, df["sel_action"] != 2, "auto", "argmax"))
    # threshold-based: bet whichever leg is higher when p_max_bet > tau
    for tau in [0.20, 0.25, 0.30, 0.35, 0.40]:
        m = df["p_max_bet"] > tau
        d = df.copy()
        d["sel_action"] = np.where(d["sel_p_home"] > d["sel_p_away"], 0, 1)
        rows.append(slice_stats(d[m], np.ones(int(m.sum()), dtype=bool), "auto",
                                f"p_max_bet>{tau}"))
    # learned-bet AND in disagree(0.60) HOME-dog slice (combo signal)
    p = df["pred_home_win_prob"].to_numpy()
    mkt = df["mkt_p_home"].to_numpy()
    home_dog = (p >= 0.60) & (mkt < 0.5)
    away_dog = ((1 - p) >= 0.60) & (mkt >= 0.5)

    rows.append(slice_stats(df[home_dog], np.ones(int(home_dog.sum()), dtype=bool),
                            "home", "d60_HOME_dog (no selector)"))
    rows.append(slice_stats(df[home_dog & (df["sel_action"] == 0)],
                            np.ones(int((home_dog & (df["sel_action"] == 0)).sum()), dtype=bool),
                            "home", "d60_HOME_dog AND sel=home"))
    rows.append(slice_stats(df[home_dog & (df["b2b_home"] == 0)],
                            np.ones(int((home_dog & (df["b2b_home"] == 0)).sum()), dtype=bool),
                            "home", "d60_HOME_dog B2B-filt"))

    rows.append(slice_stats(df[away_dog], np.ones(int(away_dog.sum()), dtype=bool),
                            "away", "d60_AWAY_dog (no selector)"))

    print("=== CME-v1.5 bet slices ===")
    print(f"{'rule':40s} {'n':>5s} {'wins':>5s} {'roi':>8s} {'win_r':>7s}")
    for r in rows:
        print(f"{r['label']:40s} {r['n']:>5d} {r['wins']:>5d} "
              f"{r['roi']*100:>+7.2f}% {r['win_rate']*100:>6.1f}%")

    # v5 baseline for direct comparison
    print()
    if V5_PRED.exists():
        v5 = pd.read_csv(V5_PRED)
        v5["game_id"] = v5["game_id"].astype(str)
        v5 = v5.merge(od, on="game_id", how="inner")
        p5 = v5["pred_home_win_prob"].to_numpy()
        mkt5 = v5["home_implied_prob_normalized"].to_numpy()
        v5_b2b = (v5["home_rest_days"].fillna(99) <= 1).astype(int)
        v5_hd = (p5 >= 0.60) & (mkt5 < 0.5)
        print("=== v5 baseline (same period) ===")
        v5_stats = slice_stats(v5[v5_hd], np.ones(int(v5_hd.sum()), dtype=bool),
                               "home", "v5 d60_HOME_dog")
        print(f"{v5_stats['label']:40s} {v5_stats['n']:>5d} {v5_stats['wins']:>5d} "
              f"{v5_stats['roi']*100:>+7.2f}% {v5_stats['win_rate']*100:>6.1f}%")
        v5_stats_b2b = slice_stats(v5[v5_hd & (v5_b2b == 0)],
                                   np.ones(int((v5_hd & (v5_b2b == 0)).sum()), dtype=bool),
                                   "home", "v5 d60_HOME_dog B2B-filt")
        print(f"{v5_stats_b2b['label']:40s} {v5_stats_b2b['n']:>5d} {v5_stats_b2b['wins']:>5d} "
              f"{v5_stats_b2b['roi']*100:>+7.2f}% {v5_stats_b2b['win_rate']*100:>6.1f}%")

    # Bootstrap CI for the headline argmax slice
    rng = np.random.default_rng(0)
    bet_mask = df["sel_action"] != 2
    sub = df[bet_mask]
    if len(sub) > 30:
        a = sub["sel_action"].to_numpy()
        won = sub["label_home_win"].to_numpy() == 1
        h = sub["home_avg_decimal_odds"].to_numpy()
        ad = sub["away_avg_decimal_odds"].to_numpy()
        profit = np.where(a == 0, np.where(won, h - 1, -1),
                          np.where(won == False, ad - 1, -1))
        boot = np.array([profit[rng.integers(0, len(profit), len(profit))].mean()
                        for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"\n[argmax CI] ROI={profit.mean()*100:+.2f}%  95%-CI=[{lo*100:+.2f}%, {hi*100:+.2f}%]")


if __name__ == "__main__":
    main()
