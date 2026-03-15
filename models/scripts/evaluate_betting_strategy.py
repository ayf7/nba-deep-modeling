#!/usr/bin/env python3
"""Evaluate moneyline betting strategies from model predictions."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from modeling_common import DEFAULT_OUTPUT_ROOT, PROJECT_ROOT


DEFAULT_CORE_DB = PROJECT_ROOT / "data" / "artifacts" / "nba_core.sqlite"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "betting_strategy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate moneyline betting strategies.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strategy",
        choices=(
            "edge_threshold",
            "expected_value",
            "predicted_winner",
            "random",
            "favorite",
            "underdog",
        ),
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Minimum model-vs-market probability edge for edge_threshold.",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=0.02,
        help="Minimum expected profit per $1 staked for expected_value.",
    )
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used by the random baseline strategy.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    required = {"game_id", "label_home_win", "pred_home_win_prob"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {', '.join(missing)}")
    predictions["game_id"] = predictions["game_id"].astype(str)
    return predictions


def load_moneyline_context(core_db: Path) -> pd.DataFrame:
    with sqlite3.connect(core_db) as conn:
        return pd.read_sql_query(
            """
            SELECT
                g.game_id,
                g.game_date,
                g.home_team_abbr,
                g.away_team_abbr,
                g.home_score,
                g.away_score,
                g.home_win,
                o.home_avg_decimal_odds,
                o.away_avg_decimal_odds,
                o.home_implied_prob_normalized,
                o.away_implied_prob_normalized,
                o.has_moneyline_odds
            FROM games g
            JOIN game_moneyline_odds o USING (game_id)
            WHERE o.has_moneyline_odds = 1
            """,
            conn,
        )


def join_predictions_to_odds(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    odds["game_id"] = odds["game_id"].astype(str)
    joined = predictions.merge(
        odds,
        on="game_id",
        how="left",
        suffixes=("_pred", ""),
    )
    joined["eligible_for_betting"] = joined["has_moneyline_odds"].fillna(0).astype(int)
    joined = joined[joined["eligible_for_betting"] == 1].copy()
    if joined.empty:
        raise ValueError("No predictions matched rows with moneyline odds.")
    return joined


def add_strategy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pred_away_win_prob"] = 1 - df["pred_home_win_prob"]
    df["home_edge"] = df["pred_home_win_prob"] - df["home_implied_prob_normalized"]
    df["away_edge"] = df["pred_away_win_prob"] - df["away_implied_prob_normalized"]
    df["home_ev"] = df["pred_home_win_prob"] * df["home_avg_decimal_odds"] - 1
    df["away_ev"] = df["pred_away_win_prob"] * df["away_avg_decimal_odds"] - 1
    return df


def choose_side(
    home_score: pd.Series,
    away_score: pd.Series,
    minimum: float,
) -> np.ndarray:
    choose_home = (home_score > minimum) & (home_score >= away_score)
    choose_away = (away_score > minimum) & (away_score > home_score)
    return np.select([choose_home, choose_away], ["home", "away"], default="none")


def make_decisions(
    df: pd.DataFrame,
    strategy: str,
    threshold: float,
    min_ev: float,
    random_seed: int,
) -> pd.DataFrame:
    df = add_strategy_columns(df)
    if strategy == "edge_threshold":
        df["decision"] = choose_side(df["home_edge"], df["away_edge"], threshold)
        df["decision_score"] = np.where(
            df["decision"] == "home",
            df["home_edge"],
            np.where(df["decision"] == "away", df["away_edge"], np.nan),
        )
    elif strategy == "expected_value":
        df["decision"] = choose_side(df["home_ev"], df["away_ev"], min_ev)
        df["decision_score"] = np.where(
            df["decision"] == "home",
            df["home_ev"],
            np.where(df["decision"] == "away", df["away_ev"], np.nan),
        )
    elif strategy == "predicted_winner":
        df["decision"] = np.where(df["pred_home_win_prob"] >= 0.5, "home", "away")
        df["decision_score"] = np.where(
            df["decision"] == "home",
            df["pred_home_win_prob"],
            df["pred_away_win_prob"],
        )
    elif strategy == "random":
        rng = np.random.default_rng(random_seed)
        df["decision"] = rng.choice(["home", "away"], size=len(df))
        df["decision_score"] = 0.5
    elif strategy == "favorite":
        df["decision"] = np.where(
            df["home_avg_decimal_odds"] <= df["away_avg_decimal_odds"],
            "home",
            "away",
        )
        df["decision_score"] = np.where(
            df["decision"] == "home",
            df["home_implied_prob_normalized"],
            df["away_implied_prob_normalized"],
        )
    elif strategy == "underdog":
        df["decision"] = np.where(
            df["home_avg_decimal_odds"] >= df["away_avg_decimal_odds"],
            "home",
            "away",
        )
        df["decision_score"] = np.where(
            df["decision"] == "home",
            df["home_avg_decimal_odds"],
            df["away_avg_decimal_odds"],
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return df


def settle_bets(decisions: pd.DataFrame, stake: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = decisions.copy()
    decisions["bet_decimal_odds"] = np.where(
        decisions["decision"] == "home",
        decisions["home_avg_decimal_odds"],
        np.where(decisions["decision"] == "away", decisions["away_avg_decimal_odds"], np.nan),
    )
    decisions["bet_won"] = np.where(
        decisions["decision"] == "home",
        decisions["label_home_win"] == 1,
        np.where(decisions["decision"] == "away", decisions["label_home_win"] == 0, False),
    )
    decisions["stake"] = np.where(decisions["decision"] == "none", 0.0, stake)
    decisions["profit"] = np.where(
        decisions["decision"] == "none",
        0.0,
        np.where(decisions["bet_won"], stake * (decisions["bet_decimal_odds"] - 1), -stake),
    )
    decisions["cumulative_profit"] = decisions["profit"].cumsum()
    decisions["running_peak_profit"] = decisions["cumulative_profit"].cummax()
    decisions["drawdown"] = decisions["cumulative_profit"] - decisions["running_peak_profit"]
    bets = decisions[decisions["decision"] != "none"].copy()
    return decisions, bets


def summarize(decisions: pd.DataFrame, bets: pd.DataFrame) -> dict[str, float | int | None]:
    eligible_games = int(len(decisions))
    bet_count = int(len(bets))
    total_staked = float(bets["stake"].sum()) if bet_count else 0.0
    net_profit = float(bets["profit"].sum()) if bet_count else 0.0
    wins = int(bets["bet_won"].sum()) if bet_count else 0
    summary = {
        "eligible_games": eligible_games,
        "bets_placed": bet_count,
        "bet_rate": bet_count / eligible_games if eligible_games else None,
        "wins": wins,
        "losses": bet_count - wins,
        "win_rate": wins / bet_count if bet_count else None,
        "total_staked": total_staked,
        "net_profit": net_profit,
        "roi": net_profit / total_staked if total_staked else None,
        "avg_profit_per_bet": net_profit / bet_count if bet_count else None,
        "max_drawdown": float(decisions["drawdown"].min()) if eligible_games else None,
        "avg_decision_score": float(bets["decision_score"].mean()) if bet_count else None,
    }
    return summary


def monthly_summary(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()
    monthly = bets.copy()
    monthly["month"] = pd.to_datetime(monthly["game_date"]).dt.to_period("M").astype(str)
    return (
        monthly.groupby("month")
        .agg(
            bets=("game_id", "count"),
            wins=("bet_won", "sum"),
            total_staked=("stake", "sum"),
            net_profit=("profit", "sum"),
            avg_decision_score=("decision_score", "mean"),
        )
        .reset_index()
        .assign(
            win_rate=lambda frame: frame["wins"] / frame["bets"],
            roi=lambda frame: frame["net_profit"] / frame["total_staked"],
        )
    )


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    odds = load_moneyline_context(args.core_db)
    joined = join_predictions_to_odds(predictions, odds)
    decisions = make_decisions(
        joined,
        args.strategy,
        args.threshold,
        args.min_ev,
        args.random_seed,
    )
    decisions, bets = settle_bets(decisions, args.stake)
    summary = summarize(decisions, bets)
    summary.update(
        {
            "predictions": str(args.predictions),
            "core_db": str(args.core_db),
            "strategy": args.strategy,
            "threshold": args.threshold,
            "min_ev": args.min_ev,
            "stake": args.stake,
            "random_seed": args.random_seed,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    decisions.to_csv(args.output_dir / "decisions.csv", index=False)
    bets.to_csv(args.output_dir / "bets.csv", index=False)
    monthly_summary(bets).to_csv(args.output_dir / "monthly_summary.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved betting artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
