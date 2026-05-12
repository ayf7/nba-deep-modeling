#!/usr/bin/env python3
"""Build leakage-safe multi-timescale dynamic opponent-adjusted team priors.

Ratings-v2 showed that a single online offense/defense ratings prior was the
strongest non-market win predictor in the project.  Ratings-v3 upgrades the
prior itself by maintaining three pregame-only rating tracks:

    slow   = more season carryover, lower update speed
    medium = the v2-style general-purpose prior
    fast   = less carryover, higher update speed

Each track emits its current pregame margin prior *before* the focal game's
score is used to update ratings.  Downstream models can then learn a compact,
leakage-safe mixture of stable, balanced, and reactive team-strength signals.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "team_strength_ratings_v3.sqlite"
RATING_TABLE = "game_team_strength_priors_v3"
TRACK_NAMES: tuple[str, ...] = ("slow", "medium", "fast")


@dataclass(frozen=True)
class TrackConfig:
    name: str
    init_league_points: float
    init_home_adv_margin: float
    score_lr: float
    league_lr: float
    home_adv_lr: float
    season_carryover: float
    residual_clip: float
    residual_saturation_scale: float


DEFAULT_TRACKS: tuple[TrackConfig, ...] = (
    TrackConfig(
        name="slow",
        init_league_points=112.0,
        init_home_adv_margin=2.5,
        score_lr=0.028,
        league_lr=0.014,
        home_adv_lr=0.008,
        season_carryover=0.82,
        residual_clip=36.0,
        residual_saturation_scale=24.0,
    ),
    TrackConfig(
        name="medium",
        init_league_points=112.0,
        init_home_adv_margin=2.5,
        score_lr=0.045,
        league_lr=0.020,
        home_adv_lr=0.012,
        season_carryover=0.65,
        residual_clip=40.0,
        residual_saturation_scale=30.0,
    ),
    TrackConfig(
        name="fast",
        init_league_points=112.0,
        init_home_adv_margin=2.5,
        score_lr=0.072,
        league_lr=0.026,
        home_adv_lr=0.018,
        season_carryover=0.48,
        residual_clip=38.0,
        residual_saturation_scale=26.0,
    ),
)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)


def _saturate_residual(x: float, clip: float, scale: float) -> float:
    """Smoothly saturate large score residuals instead of hard-overreacting."""
    clipped = _clip(x, -clip, clip)
    if scale <= 0.0:
        return clipped
    return float(scale * math.tanh(clipped / scale))


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


@dataclass
class _TrackState:
    cfg: TrackConfig
    off: defaultdict[str, float]
    deff: defaultdict[str, float]
    league_points: float
    home_adv_margin: float

    @classmethod
    def create(cls, cfg: TrackConfig) -> "_TrackState":
        return cls(
            cfg=cfg,
            off=defaultdict(float),
            deff=defaultdict(float),
            league_points=float(cfg.init_league_points),
            home_adv_margin=float(cfg.init_home_adv_margin),
        )

    def season_transition(self) -> None:
        for team in list(self.off.keys()):
            self.off[team] *= float(self.cfg.season_carryover)
        for team in list(self.deff.keys()):
            self.deff[team] *= float(self.cfg.season_carryover)

    def pregame(self, home: str, away: str, *, margin_scale: float) -> dict[str, float]:
        off_h = float(self.off[home])
        def_h = float(self.deff[home])
        off_a = float(self.off[away])
        def_a = float(self.deff[away])
        home_attack = off_h - def_a
        away_attack = off_a - def_h
        pred_home_pts = self.league_points + home_attack + 0.5 * self.home_adv_margin
        pred_away_pts = self.league_points + away_attack - 0.5 * self.home_adv_margin
        pred_margin = pred_home_pts - pred_away_pts
        pred_logit = pred_margin / max(float(margin_scale), 1e-6)
        return {
            "home_off_rating": off_h,
            "home_def_rating": def_h,
            "home_net_rating": off_h + def_h,
            "away_off_rating": off_a,
            "away_def_rating": def_a,
            "away_net_rating": off_a + def_a,
            "diff_net_rating": (off_h + def_h) - (off_a + def_a),
            "home_attack_rating": home_attack,
            "away_attack_rating": away_attack,
            "diff_attack_rating": home_attack - away_attack,
            "rating_home_points_pred": pred_home_pts,
            "rating_away_points_pred": pred_away_pts,
            "rating_margin_pred": pred_margin,
            "rating_logit_prior": pred_logit,
            "rating_home_win_prob": _sigmoid(pred_logit),
            "league_points_baseline": self.league_points,
            "home_adv_margin_baseline": self.home_adv_margin,
        }

    def update(self, actual_home: float, actual_away: float, pre: dict[str, float], home: str, away: str) -> None:
        cfg = self.cfg
        resid_home = _saturate_residual(
            actual_home - pre["rating_home_points_pred"],
            cfg.residual_clip,
            cfg.residual_saturation_scale,
        )
        resid_away = _saturate_residual(
            actual_away - pre["rating_away_points_pred"],
            cfg.residual_clip,
            cfg.residual_saturation_scale,
        )

        self.off[home] += cfg.score_lr * resid_home
        self.deff[away] -= cfg.score_lr * resid_home
        self.off[away] += cfg.score_lr * resid_away
        self.deff[home] -= cfg.score_lr * resid_away

        observed_mean_pts = 0.5 * (actual_home + actual_away)
        self.league_points += cfg.league_lr * (observed_mean_pts - self.league_points)

        no_hca_margin = pre["home_attack_rating"] - pre["away_attack_rating"]
        target_hca = _clip((actual_home - actual_away) - no_hca_margin, -20.0, 20.0)
        self.home_adv_margin += cfg.home_adv_lr * (target_hca - self.home_adv_margin)
        self.home_adv_margin = _clip(self.home_adv_margin, -5.0, 8.0)


def build_rating_rows(
    games: pd.DataFrame,
    *,
    tracks: tuple[TrackConfig, ...] = DEFAULT_TRACKS,
    margin_scale: float = 12.0,
) -> pd.DataFrame:
    """Return one strictly pregame ratings-ensemble row per completed game."""
    if tuple(t.name for t in tracks) != TRACK_NAMES:
        raise ValueError(f"tracks must be ordered {TRACK_NAMES}, got {[t.name for t in tracks]}")

    states = {cfg.name: _TrackState.create(cfg) for cfg in tracks}
    season_games = defaultdict(int)
    lifetime_games = defaultdict(int)
    last_season: int | None = None
    rows: list[dict[str, float | str | int]] = []

    for game in games.itertuples(index=False):
        season = int(game.season)
        if last_season is None:
            last_season = season
        elif season != last_season:
            for state in states.values():
                state.season_transition()
            season_games = defaultdict(int)
            last_season = season

        home = str(game.home_team_id)
        away = str(game.away_team_id)
        row: dict[str, float | str | int] = {
            "game_id": str(game.game_id),
            "game_date": pd.Timestamp(game.game_date).strftime("%Y-%m-%d"),
            "season": season,
            "home_team_id": home,
            "away_team_id": away,
            "home_rating_games_before": float(season_games[home]),
            "away_rating_games_before": float(season_games[away]),
            "diff_rating_games_before": float(season_games[home] - season_games[away]),
            "home_lifetime_rating_games_before": float(lifetime_games[home]),
            "away_lifetime_rating_games_before": float(lifetime_games[away]),
        }

        pre_by_track: dict[str, dict[str, float]] = {}
        margins: list[float] = []
        logits: list[float] = []
        probs: list[float] = []
        for name in TRACK_NAMES:
            pre = states[name].pregame(home, away, margin_scale=margin_scale)
            pre_by_track[name] = pre
            margins.append(float(pre["rating_margin_pred"]))
            logits.append(float(pre["rating_logit_prior"]))
            probs.append(float(pre["rating_home_win_prob"]))
            for key, value in pre.items():
                row[f"{name}_{key}"] = float(value)

        margins_arr = np.asarray(margins, dtype="float64")
        logits_arr = np.asarray(logits, dtype="float64")
        probs_arr = np.asarray(probs, dtype="float64")
        row.update({
            "rating_margin_mean": float(margins_arr.mean()),
            "rating_margin_std": float(margins_arr.std()),
            "rating_margin_fast_minus_slow": float(margins_arr[2] - margins_arr[0]),
            "rating_logit_mean": float(logits_arr.mean()),
            "rating_logit_std": float(logits_arr.std()),
            "rating_logit_fast_minus_slow": float(logits_arr[2] - logits_arr[0]),
            "rating_prob_mean": float(probs_arr.mean()),
            "rating_prob_std": float(probs_arr.std()),
        })
        rows.append(row)

        actual_home = float(game.home_score)
        actual_away = float(game.away_score)
        for name in TRACK_NAMES:
            states[name].update(actual_home, actual_away, pre_by_track[name], home, away)

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
            CREATE UNIQUE INDEX IF NOT EXISTS idx_rating_v3_game
              ON {RATING_TABLE}(game_id);
            CREATE INDEX IF NOT EXISTS idx_rating_v3_date
              ON {RATING_TABLE}(game_date, game_id);
            CREATE INDEX IF NOT EXISTS idx_rating_v3_season
              ON {RATING_TABLE}(season, game_date, game_id);
        """)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--margin-scale", type=float, default=12.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    games = _read_games(args.core_db)
    rows = build_rating_rows(games, margin_scale=args.margin_scale)
    write_sqlite(rows, args.output)
    print(f"[ratings-v3] games={len(rows)} output={args.output}")
    if not rows.empty:
        print("[ratings-v3] date_range=", rows["game_date"].min(), "..", rows["game_date"].max())
        print("[ratings-v3] mean_margin_by_track=", {
            name: round(float(rows[f"{name}_rating_margin_pred"].mean()), 4)
            for name in TRACK_NAMES
        })
        print("[ratings-v3] mean_prob_by_track=", {
            name: round(float(rows[f"{name}_rating_home_win_prob"].mean()), 4)
            for name in TRACK_NAMES
        })


if __name__ == "__main__":
    main()
