#!/usr/bin/env python3
"""Analyze games where models disagree with the market.

For each model: find games where model says HOME but market says AWAY (and vice versa).
Break down by: rest days, home/away favorite, actual outcome, team strength.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3

REPO = Path(__file__).resolve().parents[2]

MODELS = {
    "Logistic": REPO / "models_baseline/artifacts/backtest_logistic/predictions.csv",
    "XGBoost": REPO / "models_baseline/artifacts/backtest_xgboost/predictions.csv",
    "MLP": REPO / "models_baseline/artifacts/backtest_mlp/predictions.csv",
    "Transformer": REPO / "models_cme_v5/artifacts/full_backtest_d128/full_d128/predictions.csv",
}

ODDS_FILE = REPO / "nba_transformer/artifacts/betting_eval/predictions.csv"
CORE_DB = REPO / "data" / "artifacts" / "nba_core.sqlite"


def load_with_odds(path, odds_df):
    df = pd.read_csv(path)
    df["game_id"] = df["game_id"].astype(str)
    merged = df.merge(odds_df, on="game_id", how="inner").dropna(subset=["market_prob"])
    return merged


def analyze_disagree(df, model_name):
    """Find games where model and market disagree on who wins."""
    p = df["pred_home_win_prob"].values
    mkt = df["market_prob"].values
    label = df["label_home_win"].values

    # Model says home, market says away
    home_dog_disagree = (p >= 0.5) & (mkt < 0.5)
    # Model says away, market says home
    away_dog_disagree = (p < 0.5) & (mkt >= 0.5)

    all_disagree = home_dog_disagree | away_dog_disagree

    n_total = len(df)
    n_disagree = all_disagree.sum()
    n_home_dog = home_dog_disagree.sum()
    n_away_dog = away_dog_disagree.sum()

    # Win rate when disagreeing
    if n_home_dog > 0:
        home_dog_correct = label[home_dog_disagree].mean()
    else:
        home_dog_correct = float("nan")
    if n_away_dog > 0:
        away_dog_correct = (1 - label[away_dog_disagree]).mean()
    else:
        away_dog_correct = float("nan")

    # Confidence in disagree region
    home_dog_conf = p[home_dog_disagree].mean() if n_home_dog > 0 else float("nan")
    away_dog_conf = (1 - p[away_dog_disagree]).mean() if n_away_dog > 0 else float("nan")

    # Market odds in disagree region
    home_dog_mkt = mkt[home_dog_disagree].mean() if n_home_dog > 0 else float("nan")
    away_dog_mkt = (1 - mkt[away_dog_disagree]).mean() if n_away_dog > 0 else float("nan")

    # Rest days analysis (if available)
    has_rest = "home_rest" in df.columns and "away_rest" in df.columns
    rest_info = ""
    if has_rest:
        hr = df["home_rest"].values
        ar = df["away_rest"].values
        b2b_home = hr < 0.2  # rest is normalized, ~0.14 = 1 day
        b2b_away = ar < 0.2

        disagree_b2b_rate = (b2b_home[all_disagree] | b2b_away[all_disagree]).mean() if n_disagree > 0 else 0
        baseline_b2b_rate = (b2b_home | b2b_away).mean()
        rest_info = f"  B2B rate in disagree: {disagree_b2b_rate:.1%} vs baseline {baseline_b2b_rate:.1%}"

    return {
        "model": model_name,
        "n_total": n_total,
        "n_disagree": n_disagree,
        "disagree_pct": n_disagree / n_total * 100,
        "n_home_dog": n_home_dog,
        "n_away_dog": n_away_dog,
        "home_dog_correct": home_dog_correct,
        "away_dog_correct": away_dog_correct,
        "home_dog_model_conf": home_dog_conf,
        "away_dog_model_conf": away_dog_conf,
        "home_dog_mkt_prob": home_dog_mkt,
        "away_dog_mkt_prob": away_dog_mkt,
        "rest_info": rest_info,
    }


def main():
    odds_df = pd.read_csv(ODDS_FILE)[
        ["game_id", "home_implied_prob_normalized", "home_avg_decimal_odds",
         "away_avg_decimal_odds", "home_rest", "away_rest"]
    ].rename(columns={"home_implied_prob_normalized": "market_prob"})
    odds_df["game_id"] = odds_df["game_id"].astype(str)

    print(f"{'='*80}")
    print("DISAGREE ANALYSIS: Where do models contradict the market?")
    print(f"{'='*80}\n")

    all_results = []
    for name, path in MODELS.items():
        df = load_with_odds(path, odds_df)
        r = analyze_disagree(df, name)
        all_results.append(r)

    # Print comparison table
    print(f"{'Model':>12s} {'Disagree%':>10s} {'HomeDog':>8s} {'AwayDog':>8s} "
          f"{'HD Acc':>7s} {'AD Acc':>7s} {'HD Conf':>7s} {'AD Conf':>7s}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:>12s} {r['disagree_pct']:>9.1f}% "
              f"{r['n_home_dog']:>8d} {r['n_away_dog']:>8d} "
              f"{r['home_dog_correct']:>6.1%} {r['away_dog_correct']:>6.1%} "
              f"{r['home_dog_model_conf']:>6.1%} {r['away_dog_model_conf']:>6.1%}")
        if r["rest_info"]:
            print(f"             {r['rest_info']}")

    # Deeper dive: what games does the transformer disagree on that others don't?
    print(f"\n{'='*80}")
    print("TRANSFORMER-UNIQUE DISAGREES")
    print(f"{'='*80}\n")

    xfmr_df = load_with_odds(MODELS["Transformer"], odds_df)
    xgb_df = load_with_odds(MODELS["XGBoost"], odds_df)

    # Merge on game_id
    compare = xfmr_df[["game_id", "pred_home_win_prob", "label_home_win"]].rename(
        columns={"pred_home_win_prob": "xfmr_prob"}
    ).merge(
        xgb_df[["game_id", "pred_home_win_prob"]].rename(
            columns={"pred_home_win_prob": "xgb_prob"}
        ), on="game_id",
    ).merge(odds_df[["game_id", "market_prob"]], on="game_id")

    # Games where transformer disagrees with market but XGBoost agrees
    xfmr_disagree = (compare["xfmr_prob"] >= 0.5) != (compare["market_prob"] >= 0.5)
    xgb_agree = (compare["xgb_prob"] >= 0.5) == (compare["market_prob"] >= 0.5)
    unique_disagree = xfmr_disagree & xgb_agree

    n_unique = unique_disagree.sum()
    if n_unique > 0:
        sub = compare[unique_disagree]
        xfmr_correct = ((sub["xfmr_prob"] >= 0.5) == (sub["label_home_win"] == 1)).mean()
        xgb_correct = ((sub["xgb_prob"] >= 0.5) == (sub["label_home_win"] == 1)).mean()
        print(f"Games where Transformer disagrees with market but XGBoost agrees: {n_unique}")
        print(f"  Transformer accuracy on these: {xfmr_correct:.1%}")
        print(f"  XGBoost accuracy on these:     {xgb_correct:.1%}")
        print(f"  → Transformer is {'RIGHT more often' if xfmr_correct > xgb_correct else 'WRONG more often'}")

        # What's the market confidence on these?
        mkt_conf = np.where(sub["market_prob"] >= 0.5, sub["market_prob"], 1 - sub["market_prob"])
        print(f"  Market confidence on these: {mkt_conf.mean():.1%} (how sure the market was)")

    # Reverse: XGBoost disagrees but Transformer agrees
    xgb_disagree = (compare["xgb_prob"] >= 0.5) != (compare["market_prob"] >= 0.5)
    xfmr_agree = (compare["xfmr_prob"] >= 0.5) == (compare["market_prob"] >= 0.5)
    reverse = xgb_disagree & xfmr_agree

    n_reverse = reverse.sum()
    if n_reverse > 0:
        sub2 = compare[reverse]
        xfmr_correct2 = ((sub2["xfmr_prob"] >= 0.5) == (sub2["label_home_win"] == 1)).mean()
        xgb_correct2 = ((sub2["xgb_prob"] >= 0.5) == (sub2["label_home_win"] == 1)).mean()
        print(f"\nGames where XGBoost disagrees with market but Transformer agrees: {n_reverse}")
        print(f"  Transformer accuracy on these: {xfmr_correct2:.1%}")
        print(f"  XGBoost accuracy on these:     {xgb_correct2:.1%}")

    # Prediction spread comparison
    print(f"\n{'='*80}")
    print("PREDICTION SPREAD")
    print(f"{'='*80}\n")
    for name, path in MODELS.items():
        df = load_with_odds(path, odds_df)
        p = df["pred_home_win_prob"].values
        spread = np.std(p)
        pct_extreme = ((p > 0.7) | (p < 0.3)).mean()
        print(f"  {name:12s}: std={spread:.3f}, % extreme (>0.7 or <0.3): {pct_extreme:.1%}")


if __name__ == "__main__":
    main()
