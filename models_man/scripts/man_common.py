"""Shared utilities for MAN (Matchup Attention Network).

End-to-end attention model on game outcomes. Inputs are home/away rosters
(top-K by recent minute share, exp-decayed) plus clipped rest days; output is
P(home wins). No v1/v2 features, no warm-start.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = Path(__file__).resolve().parents[1]
DATA_ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_FEATURES_DB = DATA_ARTIFACT_ROOT / "features.sqlite"
DEFAULT_CORE_DB = DATA_ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_OUTPUT_ROOT = MODELS_ROOT / "artifacts"

LABEL_COLUMN = "label_home_win"
REST_DAYS_CLIP = 5.0
DEFAULT_ROTATION_GAMES = 10
DEFAULT_ROTATION_SIZE = 8
DEFAULT_ROTATION_DECAY = 0.85


@dataclass(frozen=True)
class TeamGameExposure:
    game_date: pd.Timestamp
    game_id: str
    player_seconds: dict[str, float]


@dataclass(frozen=True)
class Rotation:
    player_ids: tuple[str, ...]
    weights: tuple[float, ...]
    source_games: int

    @property
    def is_empty(self) -> bool:
        return all(w == 0 for w in self.weights)


def load_games(
    features_db: Path,
    table: str = "model_games",
    min_games_before: int = 10,
) -> pd.DataFrame:
    with sqlite3.connect(features_db) as conn:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    df = df[
        (df["home_games_played_before"] >= min_games_before)
        & (df["away_games_played_before"] >= min_games_before)
    ].reset_index(drop=True)
    return df


def load_team_exposures(core_db: Path) -> dict[str, list[TeamGameExposure]]:
    """Per-team chronological list of (game_date, game_id, player_seconds dict).

    Includes both offensive seconds (player on team T) and defensive seconds
    (player guarded an opponent of team T) so the rotation captures everyone
    who played for the team, not just on offense.
    """
    query = """
        WITH offensive_raw AS (
            SELECT
                game_id,
                team_id AS exposure_team_id,
                player_id AS exposure_player_id,
                SUM(COALESCE(matchup_seconds, 0.0)) AS exposure_seconds
            FROM player_matchups
            WHERE team_id IS NOT NULL
              AND player_id IS NOT NULL
              AND matchup_seconds IS NOT NULL
              AND matchup_seconds > 0
            GROUP BY game_id, team_id, player_id
        ),
        defensive_raw AS (
            SELECT
                game_id,
                CASE
                    WHEN team_id = home_team_id THEN away_team_id
                    WHEN team_id = away_team_id THEN home_team_id
                    ELSE NULL
                END AS exposure_team_id,
                defender_player_id AS exposure_player_id,
                SUM(COALESCE(matchup_seconds, 0.0)) AS exposure_seconds
            FROM player_matchups
            WHERE defender_player_id IS NOT NULL
              AND matchup_seconds IS NOT NULL
              AND matchup_seconds > 0
            GROUP BY
                game_id,
                CASE
                    WHEN team_id = home_team_id THEN away_team_id
                    WHEN team_id = away_team_id THEN home_team_id
                    ELSE NULL
                END,
                defender_player_id
        ),
        raw_exposures AS (
            SELECT * FROM offensive_raw
            UNION ALL
            SELECT * FROM defensive_raw
        )
        SELECT
            g.game_date,
            r.game_id,
            r.exposure_team_id AS team_id,
            r.exposure_player_id AS player_id,
            SUM(r.exposure_seconds) AS involvement_seconds
        FROM raw_exposures r
        JOIN games g ON g.game_id = r.game_id
        WHERE r.exposure_team_id IS NOT NULL
          AND r.exposure_player_id IS NOT NULL
          AND g.game_date IS NOT NULL
        GROUP BY g.game_date, r.game_id, r.exposure_team_id, r.exposure_player_id
        HAVING involvement_seconds > 0
        ORDER BY r.exposure_team_id, g.game_date, r.game_id
    """
    with sqlite3.connect(core_db) as conn:
        exposure_frame = pd.read_sql_query(query, conn)
    exposure_frame["game_date"] = pd.to_datetime(exposure_frame["game_date"])
    histories: dict[str, list[TeamGameExposure]] = {}
    for (team_id, game_date, game_id), group in exposure_frame.groupby(
        ["team_id", "game_date", "game_id"], sort=True
    ):
        player_seconds = {
            str(row.player_id): float(row.involvement_seconds)
            for row in group.itertuples(index=False)
        }
        histories.setdefault(str(team_id), []).append(
            TeamGameExposure(
                game_date=pd.Timestamp(game_date),
                game_id=str(game_id),
                player_seconds=player_seconds,
            )
        )
    return histories


def build_rotation(
    histories: dict[str, list[TeamGameExposure]],
    *,
    team_id: str,
    game_date: pd.Timestamp,
    rotation_games: int,
    rotation_size: int,
    decay: float,
) -> Rotation:
    prior_games = [
        item for item in histories.get(str(team_id), []) if item.game_date < game_date
    ][-rotation_games:]
    weighted_seconds: dict[str, float] = {}
    for recency_index, item in enumerate(reversed(prior_games)):
        recency_weight = decay**recency_index
        for player_id, seconds in item.player_seconds.items():
            weighted_seconds[player_id] = (
                weighted_seconds.get(player_id, 0.0) + recency_weight * seconds
            )
    ranked = sorted(weighted_seconds.items(), key=lambda x: x[1], reverse=True)
    top = [(p, w) for p, w in ranked[:rotation_size] if w > 0]
    total = sum(w for _, w in top)

    pids: list[str] = [p for p, _ in top]
    weights: list[float] = [w / total for _, w in top] if total > 0 else []
    while len(pids) < rotation_size:
        pids.append("")
        weights.append(0.0)
    return Rotation(
        player_ids=tuple(pids),
        weights=tuple(weights),
        source_games=len(prior_games),
    )


@dataclass(frozen=True)
class Vocab:
    player_to_idx: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.player_to_idx)

    def encode(self, pid: str) -> int:
        return self.player_to_idx.get(pid, 0) if pid else 0

    def encode_list(self, pids: Iterable[str]) -> list[int]:
        return [self.encode(p) for p in pids]


def build_vocab(rotations: Iterable[tuple[Rotation, Rotation]]) -> Vocab:
    """Build vocab from training rotations only; index 0 reserved for OOV/padding."""
    seen: set[str] = set()
    for home_rot, away_rot in rotations:
        for p in home_rot.player_ids + away_rot.player_ids:
            if p:
                seen.add(p)
    sorted_pids = sorted(seen)
    return Vocab(player_to_idx={p: i + 1 for i, p in enumerate(sorted_pids)})


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    game_date: pd.Timestamp
    label: int
    home_team_id: str
    away_team_id: str
    home_player_idx: tuple[int, ...]
    home_weights: tuple[float, ...]
    away_player_idx: tuple[int, ...]
    away_weights: tuple[float, ...]
    home_rest: float
    away_rest: float


def _clip_rest(value) -> float:
    if value is None or pd.isna(value):
        return REST_DAYS_CLIP
    v = float(value)
    if v < 0:
        return 0.0
    return min(v, REST_DAYS_CLIP)


def build_rotations_for_games(
    games: pd.DataFrame,
    histories: dict[str, list[TeamGameExposure]],
    *,
    rotation_games: int,
    rotation_size: int,
    decay: float,
) -> tuple[pd.DataFrame, list[tuple[Rotation, Rotation]]]:
    """For each game, build (home_rot, away_rot). Drop games where either side is empty.

    Returns parallel (kept_games_df, rotations_list) of equal length.
    """
    keep_mask: list[bool] = []
    rotations: list[tuple[Rotation, Rotation]] = []
    for game in games.itertuples(index=False):
        gd = pd.Timestamp(game.game_date)
        home_rot = build_rotation(
            histories,
            team_id=str(game.home_team_id),
            game_date=gd,
            rotation_games=rotation_games,
            rotation_size=rotation_size,
            decay=decay,
        )
        away_rot = build_rotation(
            histories,
            team_id=str(game.away_team_id),
            game_date=gd,
            rotation_games=rotation_games,
            rotation_size=rotation_size,
            decay=decay,
        )
        if home_rot.is_empty or away_rot.is_empty:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        rotations.append((home_rot, away_rot))
    kept = games.iloc[[i for i, k in enumerate(keep_mask) if k]].reset_index(drop=True)
    return kept, rotations


def make_records(
    games: pd.DataFrame,
    rotations: list[tuple[Rotation, Rotation]],
    vocab: Vocab,
) -> list[GameRecord]:
    records: list[GameRecord] = []
    for game, (home_rot, away_rot) in zip(games.itertuples(index=False), rotations):
        records.append(
            GameRecord(
                game_id=str(game.game_id),
                game_date=pd.Timestamp(game.game_date),
                label=int(getattr(game, LABEL_COLUMN)),
                home_team_id=str(game.home_team_id),
                away_team_id=str(game.away_team_id),
                home_player_idx=tuple(vocab.encode_list(home_rot.player_ids)),
                home_weights=home_rot.weights,
                away_player_idx=tuple(vocab.encode_list(away_rot.player_ids)),
                away_weights=away_rot.weights,
                home_rest=_clip_rest(game.home_rest_days),
                away_rest=_clip_rest(game.away_rest_days),
            )
        )
    return records


class GameDataset(Dataset):
    def __init__(self, records: list[GameRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "home_ids": torch.tensor(r.home_player_idx, dtype=torch.long),
            "home_w": torch.tensor(r.home_weights, dtype=torch.float32),
            "away_ids": torch.tensor(r.away_player_idx, dtype=torch.long),
            "away_w": torch.tensor(r.away_weights, dtype=torch.float32),
            "home_rest": torch.tensor(r.home_rest / REST_DAYS_CLIP, dtype=torch.float32),
            "away_rest": torch.tensor(r.away_rest / REST_DAYS_CLIP, dtype=torch.float32),
            "label": torch.tensor(r.label, dtype=torch.float32),
        }
