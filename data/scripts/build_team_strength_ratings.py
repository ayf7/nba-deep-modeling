#!/usr/bin/env python3
"""Build leakage-safe dynamic opponent-adjusted team-strength priors.

The ratings are an online pregame-only offense/defense points model.  Before
processing each completed game, the script emits the current ratings and the
implied scoring/margin prior for that game.  Only after the row is emitted are
ratings updated with the final score.

This artifact is intentionally simple and interpretable:

    home_pts_prior = league_pts + off_home - def_away + 0.5 * home_adv_margin
    away_pts_prior = league_pts + off_away - def_home - 0.5 * home_adv_margin

where a positive defensive rating means "suppresses opponent points".  Rating
updates are exponential/online residual corrections, producing opponent-
adjusted strengths without using any future game information.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "team_strength_ratings.sqlite"

RATING_TABLE = "game_team_strength_priors"


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)


def _read_games(core_db: Path) -> pd.DataFrame:
    if not core_db.exists():
        raise FileNotFoundError(f"Missing core DB: {core_db}")
    query = """
        SELECT game_id, season, game_date, home_team_id, away_team_id,
               home_score, away_score
        FROM games
        WHERE game_date IS NOT NULL
          AND home_team_id IS NOT NULL
          AND away_team_id IS NOT NULL
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY game_date, game_id
    """
    with sqlite3.connect(core_db) as conn:
        df = pd.read_sql_query(query, conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["season"].astype(int)
    df["home_score"] = df["home_score"].astype(float)
    df["away_score"] = df["away_score"].astype(float)
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def build_rating_rows(
    games: pd.DataFrame,
    *,
    init_league_points: float = 112.0,
    init_home_adv_margin: float = 2.5,
    score_lr: float = 0.045,
    league_lr: float = 0.020,
    home_adv_lr: float = 0.012,
    season_carryover: float = 0.65,
    residual_clip: float = 40.0,
    margin_scale: float = 12.0,
) -> pd.DataFrame:
    """Return one strictly pregame row per completed game."""
    off = defaultdict(float)
    deff = defaultdict(float)  # positive => suppresses opponent points
    season_games = defaultdict(int)
    lifetime_games = defaultdict(int)

    league_points = float(init_league_points)
    home_adv_margin = float(init_home_adv_margin)
    last_season: int | None = None
    rows: list[dict] = []

    for game in games.itertuples(index=False):
        season = int(game.season)
        if last_season is None:
            last_season = season
        elif season != last_season:
            # Carry signal across seasons but increase uncertainty.  The row
            # emitted for the first game of the new season still uses only
            # past-season knowledge shrunk toward league average.
            for team in list(off.keys()):
                off[team] *= season_carryover
            for team in list(deff.keys()):
                deff[team] *= season_carryover
            season_games = defaultdict(int)
            last_season = season

        home = str(game.home_team_id)
        away = str(game.away_team_id)
        off_h = float(off[home])
        def_h = float(deff[home])
        off_a = float(off[away])
        def_a = float(deff[away])

        home_attack = off_h - def_a
        away_attack = off_a - def_h
        pred_home_pts = league_points + home_attack + 0.5 * home_adv_margin
        pred_away_pts = league_points + away_attack - 0.5 * home_adv_margin
        pred_margin = pred_home_pts - pred_away_pts
        pred_home_win_prob = _sigmoid(pred_margin / max(margin_scale, 1e-6))

        home_net = off_h + def_h
        away_net = off_a + def_a
        rows.append({
            "game_id": str(game.game_id),
            "game_date": pd.Timestamp(game.game_date).strftime("%Y-%m-%d"),
            "season": season,
            "home_team_id": home,
            "away_team_id": away,
            "home_off_rating": off_h,
            "home_def_rating": def_h,
            "home_net_rating": home_net,
            "away_off_rating": off_a,
            "away_def_rating": def_a,
            "away_net_rating": away_net,
            "diff_net_rating": home_net - away_net,
            "home_attack_rating": home_attack,
            "away_attack_rating": away_attack,
            "diff_attack_rating": home_attack - away_attack,
            "rating_home_points_pred": pred_home_pts,
            "rating_away_points_pred": pred_away_pts,
            "rating_margin_pred": pred_margin,
            "rating_home_win_prob": pred_home_win_prob,
            "home_rating_games_before": float(season_games[home]),
            "away_rating_games_before": float(season_games[away]),
            "diff_rating_games_before": float(season_games[home] - season_games[away]),
            "home_lifetime_rating_games_before": float(lifetime_games[home]),
            "away_lifetime_rating_games_before": float(lifetime_games[away]),
            "league_points_baseline": league_points,
            "home_adv_margin_baseline": home_adv_margin,
        })

        # Update only after recording the pregame row.
        actual_home = float(game.home_score)
        actual_away = float(game.away_score)
        resid_home = _clip(actual_home - pred_home_pts, -residual_clip, residual_clip)
        resid_away = _clip(actual_away - pred_away_pts, -residual_clip, residual_clip)

        off[home] += score_lr * resid_home
        deff[away] -= score_lr * resid_home
        off[away] += score_lr * resid_away
        deff[home] -= score_lr * resid_away

        observed_mean_pts = 0.5 * (actual_home + actual_away)
        league_points += league_lr * (observed_mean_pts - league_points)

        # Home advantage update estimates residual home margin after the team
        # strength component, using the pregame offense/defense ratings.
        no_hca_margin = home_attack - away_attack
        target_hca = _clip((actual_home - actual_away) - no_hca_margin, -20.0, 20.0)
        home_adv_margin += home_adv_lr * (target_hca - home_adv_margin)
        home_adv_margin = _clip(home_adv_margin, -5.0, 8.0)

        season_games[home] += 1
        season_games[away] += 1
        lifetime_games[home] += 1
        lifetime_games[away] += 1

    return pd.DataFrame(rows)


def write_sqlite(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as conn:
        conn.executescript(f"DROP TABLE IF EXISTS {RATING_TABLE};")
        df.to_sql(RATING_TABLE, conn, if_exists="replace", index=False)
        conn.executescript(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_rating_game
              ON {RATING_TABLE}(game_id);
            CREATE INDEX IF NOT EXISTS idx_rating_date
              ON {RATING_TABLE}(game_date, game_id);
            CREATE INDEX IF NOT EXISTS idx_rating_season
              ON {RATING_TABLE}(season, game_date, game_id);
        """)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--init-league-points", type=float, default=112.0)
    p.add_argument("--init-home-adv-margin", type=float, default=2.5)
    p.add_argument("--score-lr", type=float, default=0.045)
    p.add_argument("--league-lr", type=float, default=0.020)
    p.add_argument("--home-adv-lr", type=float, default=0.012)
    p.add_argument("--season-carryover", type=float, default=0.65)
    p.add_argument("--residual-clip", type=float, default=40.0)
    p.add_argument("--margin-scale", type=float, default=12.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    games = _read_games(args.core_db)
    rows = build_rating_rows(
        games,
        init_league_points=args.init_league_points,
        init_home_adv_margin=args.init_home_adv_margin,
        score_lr=args.score_lr,
        league_lr=args.league_lr,
        home_adv_lr=args.home_adv_lr,
        season_carryover=args.season_carryover,
        residual_clip=args.residual_clip,
        margin_scale=args.margin_scale,
    )
    write_sqlite(rows, args.output)
    print(f"[ratings] games={len(rows)} output={args.output}")
    if not rows.empty:
        print("[ratings] date_range=", rows["game_date"].min(), "..", rows["game_date"].max())
        print("[ratings] mean_pred_margin=", round(float(rows["rating_margin_pred"].mean()), 4))
        print("[ratings] mean_home_win_prob=", round(float(rows["rating_home_win_prob"].mean()), 4))


if __name__ == "__main__":
    main()
