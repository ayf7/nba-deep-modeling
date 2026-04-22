"""Shared utilities for MAN v2 (Poisson + margin + BCE).

Extends `man_common` with per-game matchup-row loading and game scores.
v1 utilities are reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import pandas as pd
import torch
from torch.utils.data import Dataset

from man_common import (  # noqa: F401
    DATA_ARTIFACT_ROOT,
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_ROTATION_DECAY,
    DEFAULT_ROTATION_GAMES,
    DEFAULT_ROTATION_SIZE,
    LABEL_COLUMN,
    MODELS_ROOT,
    PROJECT_ROOT,
    REST_DAYS_CLIP,
    Rotation,
    Vocab,
    build_rotation,
    build_rotations_for_games,
    load_games,
    load_team_exposures,
)


DEFAULT_MATCHUP_DB = DATA_ARTIFACT_ROOT / "player_matchup_training.sqlite"


def _clip_rest(value) -> float:
    if value is None or pd.isna(value):
        return REST_DAYS_CLIP
    v = float(value)
    if v < 0:
        return 0.0
    return min(v, REST_DAYS_CLIP)


@dataclass(frozen=True)
class MatchupRow:
    off_idx: int
    def_idx: int
    exposure_possessions: float
    player_points: float


@dataclass(frozen=True)
class GameRecordV2:
    game_id: str
    game_date: pd.Timestamp
    label: int
    margin: float
    home_team_id: str
    away_team_id: str
    home_player_idx: tuple[int, ...]
    home_weights: tuple[float, ...]
    away_player_idx: tuple[int, ...]
    away_weights: tuple[float, ...]
    home_rest: float
    away_rest: float
    matchup_rows: tuple[MatchupRow, ...]


def load_game_scores(core_db: Path, game_ids: list[str]) -> dict[str, tuple[int, int]]:
    if not game_ids:
        return {}
    out: dict[str, tuple[int, int]] = {}
    chunk = 500
    with sqlite3.connect(core_db) as conn:
        for start in range(0, len(game_ids), chunk):
            ids = game_ids[start : start + chunk]
            placeholders = ",".join("?" * len(ids))
            query = (
                f"SELECT game_id, home_score, away_score FROM games "
                f"WHERE game_id IN ({placeholders})"
            )
            df = pd.read_sql_query(query, conn, params=ids)
            for row in df.itertuples(index=False):
                if pd.isna(row.home_score) or pd.isna(row.away_score):
                    continue
                out[str(row.game_id)] = (int(row.home_score), int(row.away_score))
    return out


def load_matchup_rows(
    matchup_db: Path,
    game_ids: list[str],
) -> dict[str, list[tuple[str, str, float, float]]]:
    """For each game_id, raw (off_player_id, def_player_id, exposure_possessions, player_points)."""
    if not game_ids:
        return {}
    out: dict[str, list[tuple[str, str, float, float]]] = {}
    chunk = 500
    with sqlite3.connect(matchup_db) as conn:
        for start in range(0, len(game_ids), chunk):
            ids = game_ids[start : start + chunk]
            placeholders = ",".join("?" * len(ids))
            query = (
                f"SELECT game_id, offensive_player_id, defender_player_id, "
                f"exposure_possessions, player_points "
                f"FROM matchup_training_rows "
                f"WHERE game_id IN ({placeholders}) AND exposure_possessions > 0"
            )
            df = pd.read_sql_query(query, conn, params=ids)
            for row in df.itertuples(index=False):
                gid = str(row.game_id)
                out.setdefault(gid, []).append(
                    (
                        str(row.offensive_player_id),
                        str(row.defender_player_id),
                        float(row.exposure_possessions),
                        float(row.player_points),
                    )
                )
    return out


def build_vocab_v2(
    rotations: list[tuple[Rotation, Rotation]],
    matchup_rows: dict[str, list[tuple[str, str, float, float]]],
) -> Vocab:
    """Vocab built from training rotations + training matchup rows. 0 reserved for OOV/padding."""
    seen: set[str] = set()
    for home_rot, away_rot in rotations:
        for p in home_rot.player_ids + away_rot.player_ids:
            if p:
                seen.add(p)
    for rows in matchup_rows.values():
        for off_pid, def_pid, _, _ in rows:
            seen.add(off_pid)
            seen.add(def_pid)
    sorted_pids = sorted(seen)
    return Vocab(player_to_idx={p: i + 1 for i, p in enumerate(sorted_pids)})


def make_records_v2(
    games: pd.DataFrame,
    rotations: list[tuple[Rotation, Rotation]],
    vocab: Vocab,
    *,
    matchup_rows: dict[str, list[tuple[str, str, float, float]]] | None,
    game_scores: dict[str, tuple[int, int]],
) -> list[GameRecordV2]:
    """Pass matchup_rows for the training split; pass None for val/test (no Poisson supervision needed)."""
    records: list[GameRecordV2] = []
    for game, (home_rot, away_rot) in zip(games.itertuples(index=False), rotations):
        gid = str(game.game_id)
        if gid not in game_scores:
            continue
        home_score, away_score = game_scores[gid]
        margin = float(home_score - away_score)
        rows: tuple[MatchupRow, ...] = ()
        if matchup_rows is not None:
            encoded: list[MatchupRow] = []
            for off_pid, def_pid, exposure, points in matchup_rows.get(gid, []):
                off_idx = vocab.encode(off_pid)
                def_idx = vocab.encode(def_pid)
                if off_idx == 0 or def_idx == 0:
                    continue
                encoded.append(MatchupRow(off_idx, def_idx, exposure, points))
            rows = tuple(encoded)
        records.append(
            GameRecordV2(
                game_id=gid,
                game_date=pd.Timestamp(game.game_date),
                label=int(getattr(game, LABEL_COLUMN)),
                margin=margin,
                home_team_id=str(game.home_team_id),
                away_team_id=str(game.away_team_id),
                home_player_idx=tuple(vocab.encode_list(home_rot.player_ids)),
                home_weights=home_rot.weights,
                away_player_idx=tuple(vocab.encode_list(away_rot.player_ids)),
                away_weights=away_rot.weights,
                home_rest=_clip_rest(game.home_rest_days),
                away_rest=_clip_rest(game.away_rest_days),
                matchup_rows=rows,
            )
        )
    return records


class GameDatasetV2(Dataset):
    def __init__(self, records: list[GameRecordV2]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        if r.matchup_rows:
            m_off = torch.tensor([row.off_idx for row in r.matchup_rows], dtype=torch.long)
            m_def = torch.tensor([row.def_idx for row in r.matchup_rows], dtype=torch.long)
            m_exp = torch.tensor(
                [row.exposure_possessions for row in r.matchup_rows], dtype=torch.float32
            )
            m_pts = torch.tensor(
                [row.player_points for row in r.matchup_rows], dtype=torch.float32
            )
        else:
            m_off = torch.zeros(0, dtype=torch.long)
            m_def = torch.zeros(0, dtype=torch.long)
            m_exp = torch.zeros(0, dtype=torch.float32)
            m_pts = torch.zeros(0, dtype=torch.float32)
        return {
            "home_ids": torch.tensor(r.home_player_idx, dtype=torch.long),
            "home_w": torch.tensor(r.home_weights, dtype=torch.float32),
            "away_ids": torch.tensor(r.away_player_idx, dtype=torch.long),
            "away_w": torch.tensor(r.away_weights, dtype=torch.float32),
            "home_rest": torch.tensor(r.home_rest / REST_DAYS_CLIP, dtype=torch.float32),
            "away_rest": torch.tensor(r.away_rest / REST_DAYS_CLIP, dtype=torch.float32),
            "label": torch.tensor(r.label, dtype=torch.float32),
            "margin": torch.tensor(r.margin, dtype=torch.float32),
            "m_off": m_off,
            "m_def": m_def,
            "m_exp": m_exp,
            "m_pts": m_pts,
        }


def collate_v2(batch: list[dict]) -> dict:
    fixed = ("home_ids", "home_w", "away_ids", "away_w", "home_rest", "away_rest", "label", "margin")
    out: dict = {k: torch.stack([b[k] for b in batch]) for k in fixed}
    out["m_off"] = torch.cat([b["m_off"] for b in batch])
    out["m_def"] = torch.cat([b["m_def"] for b in batch])
    out["m_exp"] = torch.cat([b["m_exp"] for b in batch])
    out["m_pts"] = torch.cat([b["m_pts"] for b in batch])
    return out
