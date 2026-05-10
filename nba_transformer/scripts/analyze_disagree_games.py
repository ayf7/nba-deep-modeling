#!/usr/bin/env python3
"""Sample and characterize home-dog disagree games — model says home, market says away."""
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3

REPO = Path(__file__).resolve().parents[2]
ODDS_FILE = REPO / "nba_transformer/artifacts/betting_eval/predictions.csv"
CORE_DB = REPO / "data" / "artifacts" / "nba_core.sqlite"

MODELS = {
    "XGBoost": REPO / "models_baseline/artifacts/backtest_xgboost/predictions.csv",
    "Transformer": REPO / "models_cme_v5/artifacts/full_backtest_d128/full_d128/predictions.csv",
}


def main():
    odds_df = pd.read_csv(ODDS_FILE)
    odds_df["game_id"] = odds_df["game_id"].astype(str)

    core = sqlite3.connect(str(CORE_DB))
    games = pd.read_sql_query(
        "SELECT game_id, game_date, home_team_abbr, away_team_abbr, home_score, away_score FROM games",
        core,
    )
    core.close()
    games["game_id"] = games["game_id"].astype(str)
    games["margin"] = games["home_score"] - games["away_score"]

    for model_name, path in MODELS.items():
        df = pd.read_csv(path)
        df["game_id"] = df["game_id"].astype(str)

        merged = df.merge(
            odds_df[["game_id", "home_implied_prob_normalized", "home_avg_decimal_odds",
                      "away_avg_decimal_odds", "home_rest", "away_rest"]],
            on="game_id",
        ).dropna(subset=["home_implied_prob_normalized"])

        merged = merged.merge(games[["game_id", "game_date", "home_team_abbr", "away_team_abbr",
                                      "home_score", "away_score", "margin"]], on="game_id")

        p = merged["pred_home_win_prob"].values
        mkt = merged["home_implied_prob_normalized"].values

        # HOME-DOG DISAGREE: model says home (p >= 0.5), market says away (mkt < 0.5)
        home_dog = (p >= 0.5) & (mkt < 0.5)
        hd = merged[home_dog].copy()
        hd["model_conf"] = hd["pred_home_win_prob"]
        hd["mkt_home_prob"] = hd["home_implied_prob_normalized"]
        hd["won"] = hd["label_home_win"] == 1
        hd["pnl"] = np.where(hd["won"], hd["home_avg_decimal_odds"] - 1, -1)

        print(f"\n{'='*90}")
        print(f"HOME-DOG DISAGREE — {model_name}")
        print(f"{'='*90}")
        print(f"Total: {len(hd)} games, Win rate: {hd['won'].mean():.1%}, "
              f"P&L: {hd['pnl'].sum():+.1f}u, ROI: {hd['pnl'].mean()*100:+.1f}%")

        # By team
        print(f"\n--- By HOME team (who model backs as underdog) ---")
        team_stats = hd.groupby("home_team_abbr").agg(
            n=("won", "size"),
            wins=("won", "sum"),
            pnl=("pnl", "sum"),
            avg_odds=("home_avg_decimal_odds", "mean"),
            avg_model=("model_conf", "mean"),
            avg_mkt=("mkt_home_prob", "mean"),
        ).sort_values("n", ascending=False)
        team_stats["win_rate"] = team_stats["wins"] / team_stats["n"]
        team_stats["roi"] = team_stats["pnl"] / team_stats["n"] * 100
        print(f"{'Team':>5s} {'n':>4s} {'W%':>6s} {'P&L':>7s} {'ROI':>7s} {'AvgOdds':>8s} {'Model':>6s} {'Mkt':>6s}")
        for team, row in team_stats.head(15).iterrows():
            print(f"{team:>5s} {row['n']:>4.0f} {row['win_rate']:>5.1%} {row['pnl']:>+6.1f}u "
                  f"{row['roi']:>+6.1f}% {row['avg_odds']:>7.2f} {row['avg_model']:>5.1%} {row['avg_mkt']:>5.1%}")

        # By confidence bucket
        print(f"\n--- By model confidence ---")
        hd["conf_bucket"] = pd.cut(hd["model_conf"], bins=[0.5, 0.55, 0.60, 0.65, 0.70, 1.0],
                                    labels=["50-55%", "55-60%", "60-65%", "65-70%", "70%+"])
        conf_stats = hd.groupby("conf_bucket", observed=True).agg(
            n=("won", "size"), wins=("won", "sum"), pnl=("pnl", "sum"),
            avg_odds=("home_avg_decimal_odds", "mean"),
        )
        conf_stats["win_rate"] = conf_stats["wins"] / conf_stats["n"]
        conf_stats["roi"] = conf_stats["pnl"] / conf_stats["n"] * 100
        for bucket, row in conf_stats.iterrows():
            print(f"  {bucket:>7s}: n={row['n']:>3.0f}, W={row['win_rate']:>5.1%}, "
                  f"P&L={row['pnl']:>+6.1f}u, ROI={row['roi']:>+6.1f}%, avgOdds={row['avg_odds']:.2f}")

        # By market confidence (how big the underdog)
        print(f"\n--- By market underdog strength ---")
        hd["mkt_bucket"] = pd.cut(hd["mkt_home_prob"], bins=[0, 0.30, 0.35, 0.40, 0.45, 0.50],
                                   labels=["<30%", "30-35%", "35-40%", "40-45%", "45-50%"])
        mkt_stats = hd.groupby("mkt_bucket", observed=True).agg(
            n=("won", "size"), wins=("won", "sum"), pnl=("pnl", "sum"),
            avg_odds=("home_avg_decimal_odds", "mean"),
        )
        mkt_stats["win_rate"] = mkt_stats["wins"] / mkt_stats["n"]
        mkt_stats["roi"] = mkt_stats["pnl"] / mkt_stats["n"] * 100
        for bucket, row in mkt_stats.iterrows():
            print(f"  {bucket:>7s}: n={row['n']:>3.0f}, W={row['win_rate']:>5.1%}, "
                  f"P&L={row['pnl']:>+6.1f}u, ROI={row['roi']:>+6.1f}%, avgOdds={row['avg_odds']:.2f}")

        # Rest days pattern
        print(f"\n--- By rest pattern ---")
        hd["home_b2b"] = hd["home_rest"] < 0.2
        hd["away_b2b"] = hd["away_rest"] < 0.2
        hd["rest_pattern"] = "normal"
        hd.loc[hd["home_b2b"] & ~hd["away_b2b"], "rest_pattern"] = "home_b2b"
        hd.loc[~hd["home_b2b"] & hd["away_b2b"], "rest_pattern"] = "away_b2b"
        hd.loc[hd["home_b2b"] & hd["away_b2b"], "rest_pattern"] = "both_b2b"
        rest_stats = hd.groupby("rest_pattern").agg(
            n=("won", "size"), wins=("won", "sum"), pnl=("pnl", "sum"),
        )
        rest_stats["win_rate"] = rest_stats["wins"] / rest_stats["n"]
        rest_stats["roi"] = rest_stats["pnl"] / rest_stats["n"] * 100
        for pattern, row in rest_stats.iterrows():
            print(f"  {pattern:>12s}: n={row['n']:>3.0f}, W={row['win_rate']:>5.1%}, "
                  f"P&L={row['pnl']:>+6.1f}u, ROI={row['roi']:>+6.1f}%")

        # Margin distribution — when model is right vs wrong
        print(f"\n--- Score margins ---")
        correct = hd[hd["won"]]
        wrong = hd[~hd["won"]]
        print(f"  When correct (home wins): avg margin +{correct['margin'].mean():.1f}, "
              f"median +{correct['margin'].median():.0f}")
        print(f"  When wrong (home loses):  avg margin {wrong['margin'].mean():.1f}, "
              f"median {wrong['margin'].median():.0f}")

        # Sample some games
        print(f"\n--- Sample games (highest model confidence disagrees) ---")
        date_col = "game_date_x" if "game_date_x" in hd.columns else "game_date"
        sample = hd.nlargest(10, "model_conf")
        print(f"{'Date':>12s} {'Home':>5s} {'Away':>5s} {'Model':>6s} {'Mkt':>6s} "
              f"{'Score':>9s} {'Odds':>6s} {'Won':>4s}")
        for _, r in sample.iterrows():
            score = f"{int(r['home_score'])}-{int(r['away_score'])}"
            dt = r.get(date_col, "")
            print(f"{str(dt):>12s} {r['home_team_abbr']:>5s} {r['away_team_abbr']:>5s} "
                  f"{r['model_conf']:>5.1%} {r['mkt_home_prob']:>5.1%} {score:>9s} "
                  f"{r['home_avg_decimal_odds']:>5.2f} {'Y' if r['won'] else 'N':>4s}")


if __name__ == "__main__":
    main()
