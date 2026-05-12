#!/usr/bin/env python3
"""Elo rating system for NBA game prediction.

Online rating model with margin-of-victory updates, home-court advantage,
and season regression. Produces backtest-compatible output artifacts.

Calibration: grid search over (K, HCA, season_reversion, scale, alpha) on
the last two pre-backtest seasons, scored by log loss. The scale parameter
controls how aggressively rating differences map to probabilities (higher =
more compressed toward 50%). The alpha parameter blends Elo predictions
with market-implied probabilities when available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from modeling_common import DEFAULT_OUTPUT_ROOT, evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORE_DB = PROJECT_ROOT / "data" / "artifacts" / "nba_core.sqlite"


class EloSystem:
    def __init__(
        self,
        k: float = 20.0,
        hca: float = 100.0,
        season_reversion: float = 0.75,
        scale: float = 400.0,
        initial_elo: float = 1500.0,
    ):
        self.k = k
        self.hca = hca
        self.season_reversion = season_reversion
        self.scale = scale
        self.initial_elo = initial_elo
        self.ratings: dict[str, float] = {}

    def get_rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, self.initial_elo)

    def predict(self, home_id: str, away_id: str) -> float:
        diff = self.get_rating(home_id) - self.get_rating(away_id) + self.hca
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def mov_multiplier(self, margin: int, elo_diff: float) -> float:
        return math.log(abs(margin) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

    def update(
        self, home_id: str, away_id: str, home_score: int, away_score: int
    ) -> None:
        expected = self.predict(home_id, away_id)
        actual = 1.0 if home_score > away_score else 0.0
        margin = home_score - away_score
        elo_diff = self.get_rating(home_id) - self.get_rating(away_id) + self.hca
        if actual == 0.0:
            elo_diff = -elo_diff
        mov_mult = self.mov_multiplier(margin, elo_diff)
        shift = self.k * mov_mult * (actual - expected)
        self.ratings[home_id] = self.get_rating(home_id) + shift
        self.ratings[away_id] = self.get_rating(away_id) - shift

    def regress_to_mean(self) -> None:
        for team_id in list(self.ratings):
            self.ratings[team_id] = (
                self.season_reversion * self.ratings[team_id]
                + (1 - self.season_reversion) * self.initial_elo
            )

    def reset(self) -> None:
        self.ratings.clear()


def load_games(db_path: Path) -> pd.DataFrame:
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT game_id, season, game_date, home_team_id, away_team_id, "
            "home_score, away_score, home_win FROM games ORDER BY game_date, game_id",
            conn,
        )
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_market_probs(db_path: Path) -> dict[str, float]:
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        rows = pd.read_sql_query(
            "SELECT game_id, home_implied_prob_normalized "
            "FROM game_moneyline_odds WHERE has_moneyline_odds = 1",
            conn,
        )
    return dict(zip(rows["game_id"].astype(str), rows["home_implied_prob_normalized"]))


def run_elo_pass(
    games: pd.DataFrame,
    k: float,
    hca: float,
    season_reversion: float,
    scale: float = 400.0,
) -> tuple[EloSystem, list[dict]]:
    elo = EloSystem(k=k, hca=hca, season_reversion=season_reversion, scale=scale)
    records = []
    prev_season = None

    for row in games.itertuples(index=False):
        if prev_season is not None and row.season != prev_season:
            elo.regress_to_mean()
        prev_season = row.season

        pred = elo.predict(row.home_team_id, row.away_team_id)
        records.append(
            {
                "game_id": row.game_id,
                "game_date": row.game_date,
                "home_team_id": row.home_team_id,
                "away_team_id": row.away_team_id,
                "label_home_win": row.home_win,
                "pred_home_win_prob": pred,
                "home_elo": elo.get_rating(row.home_team_id),
                "away_elo": elo.get_rating(row.away_team_id),
            }
        )
        elo.update(row.home_team_id, row.away_team_id, row.home_score, row.away_score)

    return elo, records


def calibrate(
    games: pd.DataFrame,
    market_probs: dict[str, float],
    backtest_start: str,
    cal_seasons: int = 2,
) -> dict[str, float]:
    backtest_ts = pd.Timestamp(backtest_start)
    pre_backtest = games[games["game_date"] < backtest_ts]
    defaults = {"k": 20.0, "hca": 100.0, "season_reversion": 0.75, "scale": 400.0, "alpha": 1.0}
    if pre_backtest.empty:
        return defaults

    all_seasons = sorted(pre_backtest["season"].unique())
    if len(all_seasons) <= cal_seasons:
        cal_season_cutoff = all_seasons[0]
    else:
        cal_season_cutoff = all_seasons[-cal_seasons]

    k_values = [15, 20, 25, 30]
    hca_values = [40, 60, 80, 100, 120]
    reversion_values = [0.6, 0.7, 0.75, 0.8, 0.85]
    scale_values = [400, 480, 560, 640, 720]
    alpha_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Stage 1: find best K, HCA, reversion, scale with alpha=1.0 (pure Elo)
    stage1_total = len(k_values) * len(hca_values) * len(reversion_values) * len(scale_values)
    print(f"Stage 1: calibrating over {stage1_total} combos (K × HCA × reversion × scale)...")

    best_loss = float("inf")
    best_params = defaults.copy()

    for k in k_values:
        for hca in hca_values:
            for rev in reversion_values:
                for scale in scale_values:
                    _, records = run_elo_pass(pre_backtest, k, hca, rev, scale)
                    df = pd.DataFrame(records)
                    cal_mask = df["game_date"].dt.year >= cal_season_cutoff
                    if cal_mask.sum() < 50:
                        continue
                    cal_df = df[cal_mask]
                    loss = log_loss(
                        cal_df["label_home_win"],
                        cal_df["pred_home_win_prob"].clip(1e-7, 1 - 1e-7),
                        labels=[0, 1],
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_params = {
                            "k": float(k), "hca": float(hca),
                            "season_reversion": float(rev), "scale": float(scale),
                            "alpha": 1.0,
                        }

    print(f"Stage 1 best: K={best_params['k']}, HCA={best_params['hca']}, "
          f"reversion={best_params['season_reversion']}, scale={best_params['scale']} "
          f"(log_loss={best_loss:.4f})")

    # Stage 2: fix K/HCA/reversion, search scale × alpha with market blend
    _, records = run_elo_pass(
        pre_backtest, best_params["k"], best_params["hca"],
        best_params["season_reversion"], best_params["scale"],
    )
    df = pd.DataFrame(records)
    cal_mask = df["game_date"].dt.year >= cal_season_cutoff
    cal_df = df[cal_mask].copy()
    cal_df["game_id"] = cal_df["game_id"].astype(str)
    cal_df["market_prob"] = cal_df["game_id"].map(market_probs)
    has_market = cal_df["market_prob"].notna()

    if has_market.sum() >= 50:
        stage2_total = len(scale_values) * len(alpha_values)
        print(f"Stage 2: calibrating over {stage2_total} combos (scale × alpha) "
              f"with {has_market.sum()} market-matched games...")

        for scale in scale_values:
            _, records2 = run_elo_pass(
                pre_backtest, best_params["k"], best_params["hca"],
                best_params["season_reversion"], scale,
            )
            df2 = pd.DataFrame(records2)
            cal_df2 = df2[df2["game_date"].dt.year >= cal_season_cutoff].copy()
            cal_df2["game_id"] = cal_df2["game_id"].astype(str)
            cal_df2["market_prob"] = cal_df2["game_id"].map(market_probs)
            has_mkt2 = cal_df2["market_prob"].notna()
            cal_sub = cal_df2[has_mkt2]
            if len(cal_sub) < 50:
                continue

            for alpha in alpha_values:
                blended = alpha * cal_sub["pred_home_win_prob"] + (1 - alpha) * cal_sub["market_prob"]
                blended = blended.clip(1e-7, 1 - 1e-7)
                loss = log_loss(cal_sub["label_home_win"], blended, labels=[0, 1])
                if loss < best_loss:
                    best_loss = loss
                    best_params["scale"] = float(scale)
                    best_params["alpha"] = float(alpha)
    else:
        print("Stage 2: not enough market data for alpha calibration, keeping alpha=1.0")

    print(f"Final best: K={best_params['k']}, HCA={best_params['hca']}, "
          f"reversion={best_params['season_reversion']}, scale={best_params['scale']}, "
          f"alpha={best_params['alpha']} (log_loss={best_loss:.4f})")
    return best_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Elo rating system backtest.")
    parser.add_argument("--db", type=Path, default=DEFAULT_CORE_DB)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backtest-start", default="2024-01-01")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--k", type=float, default=20.0)
    parser.add_argument("--hca", type=float, default=100.0)
    parser.add_argument("--reversion", type=float, default=0.75)
    parser.add_argument("--scale", type=float, default=400.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / "backtest_elo")

    games = load_games(args.db)
    market_probs = load_market_probs(args.db)
    print(f"Loaded {len(games)} games ({games['season'].nunique()} seasons), "
          f"{len(market_probs)} with market odds")

    if args.no_calibrate:
        params = {
            "k": args.k, "hca": args.hca, "season_reversion": args.reversion,
            "scale": args.scale, "alpha": args.alpha,
        }
        print(f"Using provided params: K={params['k']}, HCA={params['hca']}, "
              f"reversion={params['season_reversion']}, scale={params['scale']}, "
              f"alpha={params['alpha']}")
    else:
        params = calibrate(games, market_probs, args.backtest_start)

    alpha = params.pop("alpha")
    _, records = run_elo_pass(games, **params)
    all_df = pd.DataFrame(records)

    backtest_ts = pd.Timestamp(args.backtest_start)
    preds = all_df[all_df["game_date"] >= backtest_ts].copy()

    if alpha < 1.0:
        preds["game_id_str"] = preds["game_id"].astype(str)
        preds["market_prob"] = preds["game_id_str"].map(market_probs)
        has_market = preds["market_prob"].notna()
        preds.loc[has_market, "pred_home_win_prob"] = (
            alpha * preds.loc[has_market, "pred_home_win_prob"]
            + (1 - alpha) * preds.loc[has_market, "market_prob"]
        )
        preds.drop(columns=["game_id_str", "market_prob"], inplace=True)
        print(f"Applied market blend: alpha={alpha:.2f} ({has_market.sum()}/{len(preds)} games with odds)")

    preds["window_start"] = preds["game_date"].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")

    overall = evaluate(preds["label_home_win"], preds["pred_home_win_prob"].to_numpy())

    window_metrics = []
    for window in sorted(preds["window_start"].unique()):
        mask = preds["window_start"] == window
        wm = evaluate(preds.loc[mask, "label_home_win"], preds.loc[mask, "pred_home_win_prob"].to_numpy())
        wm["window_start"] = window
        wm["window_end"] = (pd.Timestamp(window) + pd.offsets.MonthEnd(0)).date().isoformat()
        window_metrics.append(wm)

    output_dir.mkdir(parents=True, exist_ok=True)

    params["alpha"] = alpha
    config = {
        "model": "elo",
        "backtest_start": args.backtest_start,
        "calibrated": not args.no_calibrate,
        **params,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2) + "\n")
    pd.DataFrame(window_metrics).to_csv(output_dir / "window_metrics.csv", index=False)

    out_preds = preds[
        ["game_id", "game_date", "home_team_id", "away_team_id", "label_home_win", "pred_home_win_prob", "window_start"]
    ].copy()
    out_preds["game_date"] = out_preds["game_date"].dt.strftime("%Y-%m-%d")
    out_preds.to_csv(output_dir / "predictions.csv", index=False)

    elo_history = all_df[["game_id", "home_team_id", "away_team_id", "home_elo", "away_elo"]].copy()
    elo_history.to_csv(output_dir / "elo_ratings.csv", index=False)

    print(f"\n=== Elo Backtest Results ===")
    print(json.dumps(overall, indent=2))
    print(f"Predictions: {len(preds)} games")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
