"""Shared utilities for market-conditioned (residual) baselines.

Three modes of using market info:
  off       : control -- no market info at all
  feature   : market_logit appended to feature vector (model decides how to use)
  residual  : final logit = market_logit + delta_logit, delta_logit = f(features).
              Forces gradient onto deviations-from-market only.

market_logit = log(p / (1-p)) where p = home_implied_prob_normalized.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from modeling_common import LABEL_COLUMN, PROJECT_ROOT

CORE_DB = PROJECT_ROOT / "data" / "artifacts" / "nba_core.sqlite"
ODDS_TABLE = "game_moneyline_odds"
MARKET_LOGIT_COL = "market_logit"
USE_MARKET_CHOICES = ("off", "feature", "residual")
MARKET_KEEP_COLS = (
    "home_implied_prob_normalized",
    "away_implied_prob_normalized",
    "home_avg_decimal_odds",
    "away_avg_decimal_odds",
    MARKET_LOGIT_COL,
)


def load_market_data() -> pd.DataFrame:
    """Pull odds + implied probs for games that have moneyline data."""
    with sqlite3.connect(CORE_DB) as conn:
        df = pd.read_sql_query(
            f"""SELECT game_id,
                       home_implied_prob_normalized,
                       away_implied_prob_normalized,
                       home_avg_decimal_odds,
                       away_avg_decimal_odds
                FROM {ODDS_TABLE}
                WHERE has_moneyline_odds = 1""",
            conn,
        )
    df["game_id"] = df["game_id"].astype(str)
    eps = 1e-6
    p = np.clip(
        df["home_implied_prob_normalized"].to_numpy(dtype=np.float64),
        eps,
        1.0 - eps,
    )
    df[MARKET_LOGIT_COL] = np.log(p / (1.0 - p))
    return df


def attach_market(df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join market columns to a feature frame; drop games missing odds."""
    market = load_market_data()
    out = df.copy()
    out["game_id"] = out["game_id"].astype(str)
    n_before = len(out)
    out = out.merge(market, on="game_id", how="inner")
    n_after = len(out)
    if n_after < n_before:
        print(
            f"[attach_market] dropped {n_before - n_after} games missing odds "
            f"({n_before} -> {n_after})"
        )
    return out


def disagree_metrics(df: pd.DataFrame, pthr: float = 0.60) -> dict:
    """ROI / win-rate for disagree(p>=pthr)-with-market bets.

    Required columns: pred_home_win_prob, label_home_win,
    home_implied_prob_normalized, home_avg_decimal_odds, away_avg_decimal_odds.
    """
    p = df["pred_home_win_prob"].to_numpy()
    hf = df["home_implied_prob_normalized"].to_numpy() >= 0.5
    bet_home = (p >= pthr) & (~hf)
    bet_away = ((1 - p) >= pthr) & hf
    home_won = (df[LABEL_COLUMN] == 1).to_numpy()
    h_pf = np.where(
        bet_home & home_won,
        df["home_avg_decimal_odds"] - 1,
        np.where(bet_home & ~home_won, -1, 0),
    )
    a_pf = np.where(
        bet_away & ~home_won,
        df["away_avg_decimal_odds"] - 1,
        np.where(bet_away & home_won, -1, 0),
    )
    profit = h_pf + a_pf
    bets = int((bet_home | bet_away).sum())
    wins = int(((bet_home & home_won) | (bet_away & ~home_won)).sum())
    return {
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets if bets else 0.0,
        "roi": float(profit.sum()) / bets if bets else 0.0,
    }


def feat_cols_for_mode(features: list[str], use_market: str) -> list[str]:
    """Decide which input columns the model sees.

    off       : just 41 features (control)
    feature   : 41 features + market_logit (model decides)
    residual  : just 41 features (market enters via skip-add to the logit)
    """
    if use_market not in USE_MARKET_CHOICES:
        raise ValueError(f"use_market must be one of {USE_MARKET_CHOICES}")
    if use_market == "feature":
        return features + [MARKET_LOGIT_COL]
    return list(features)
