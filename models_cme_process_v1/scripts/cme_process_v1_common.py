"""Process-CME v1 data layer.

This module deliberately reuses the proven CME-v4.2 game/roster/pair-label
builder and *wraps* each record with two process-specific payloads:

  * leakage-safe pregame process-history features, appended to the usual
    normalized tabular vector;
  * realized home/away process targets used only for auxiliary training.

The process artifact itself is created by:

    python data/scripts/build_game_process_features.py

The wrapper design keeps the v4.2 data path unchanged and makes this branch a
clean experiment on top of the current best architecture.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_PROCESS_DB = ARTIFACT_ROOT / "possession_process.sqlite"

V42_SCRIPTS = PROJECT_ROOT / "models_cme_v4_2" / "scripts"
sys.path.insert(0, str(V42_SCRIPTS))

from cme_v4_2_common import *  # noqa: F403,E402
from cme_v4_2_common import (  # noqa: E402
    GameDatasetV42,
    GameRecordV42,
    build_records_v42,
    collate_v42,
)

PROCESS_METRICS: tuple[str, ...] = (
    "possessions",
    "fg2a_rate",
    "fg3a_rate",
    "turnovers_rate",
    "offensive_rebounds_rate",
    "shooting_fouls_drawn_rate",
)
PROCESS_TARGET_COLUMNS: tuple[str, ...] = tuple(
    [f"home_{m}" for m in PROCESS_METRICS]
    + [f"away_{m}" for m in PROCESS_METRICS]
)
K_PROCESS = len(PROCESS_TARGET_COLUMNS)


def _process_feature_columns() -> tuple[str, ...]:
    cols: list[str] = [
        "home_process_games_before",
        "away_process_games_before",
        "diff_process_games_before",
        "home_allowed_process_games_before",
        "away_allowed_process_games_before",
        "diff_allowed_process_games_before",
    ]
    for metric in PROCESS_METRICS:
        cols.extend(
            [
                f"home_own_{metric}_last10",
                f"away_own_{metric}_last10",
                f"diff_own_{metric}_last10",
                f"home_allowed_{metric}_last10",
                f"away_allowed_{metric}_last10",
                f"diff_allowed_{metric}_last10",
                f"home_attack_gap_{metric}_last10",
                f"away_attack_gap_{metric}_last10",
                f"diff_attack_gap_{metric}_last10",
            ]
        )
    return tuple(cols)


PROCESS_PREGAME_FEATURE_COLUMNS = _process_feature_columns()
K_PROCESS_FEATURES = len(PROCESS_PREGAME_FEATURE_COLUMNS)


@dataclass(frozen=True)
class VectorStats:
    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def _fit_vector_stats(rows: np.ndarray) -> VectorStats:
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D, got shape={rows.shape}")
    if rows.shape[0] == 0:
        med = np.zeros(rows.shape[1], dtype="float64")
        mean = np.zeros(rows.shape[1], dtype="float64")
        std = np.ones(rows.shape[1], dtype="float64")
        return VectorStats(medians=med, means=mean, stds=std)
    X = rows.astype("float64", copy=True)
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    nan_rows, nan_cols = np.where(np.isnan(X))
    if len(nan_rows):
        X[nan_rows, nan_cols] = medians[nan_cols]
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    return VectorStats(medians=medians, means=means, stds=stds)


def _transform_vector(values: np.ndarray, stats: VectorStats) -> np.ndarray:
    row = values.astype("float64", copy=True)
    nans = np.isnan(row)
    if nans.any():
        row[nans] = stats.medians[nans]
    return ((row - stats.means) / stats.stds).astype("float32")


def load_process_pregame_features(
    process_db: Path,
    game_ids: list[str] | None = None,
) -> dict[str, tuple[float, ...]]:
    if not process_db.exists():
        raise FileNotFoundError(
            f"Missing process DB: {process_db}. Run data/scripts/build_game_process_features.py first."
        )
    cols = ", ".join(["game_id", *PROCESS_PREGAME_FEATURE_COLUMNS])
    query = f"SELECT {cols} FROM game_process_pregame_features"
    params: list[str] | None = None
    if game_ids:
        # For a few thousand game ids, chunking avoids SQLite's parameter cap.
        out: dict[str, tuple[float, ...]] = {}
        with sqlite3.connect(process_db) as conn:
            for start in range(0, len(game_ids), 500):
                ids = [str(g) for g in game_ids[start : start + 500]]
                ph = ",".join("?" for _ in ids)
                df = pd.read_sql_query(
                    f"{query} WHERE game_id IN ({ph})", conn, params=ids,
                )
                for row in df.itertuples(index=False):
                    gid = str(row.game_id)
                    vals = tuple(float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                                 for c in PROCESS_PREGAME_FEATURE_COLUMNS)
                    out[gid] = vals
        return out
    with sqlite3.connect(process_db) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    out: dict[str, tuple[float, ...]] = {}
    for row in df.itertuples(index=False):
        gid = str(row.game_id)
        vals = tuple(float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                     for c in PROCESS_PREGAME_FEATURE_COLUMNS)
        out[gid] = vals
    return out


def load_process_targets(
    process_db: Path,
    game_ids: list[str] | None = None,
) -> dict[str, tuple[tuple[float, ...], bool]]:
    if not process_db.exists():
        raise FileNotFoundError(
            f"Missing process DB: {process_db}. Run data/scripts/build_game_process_features.py first."
        )
    cols = ", ".join(["game_id", *PROCESS_TARGET_COLUMNS, "process_target_valid"])
    query = f"SELECT {cols} FROM game_process_targets"
    out: dict[str, tuple[tuple[float, ...], bool]] = {}
    with sqlite3.connect(process_db) as conn:
        if game_ids:
            for start in range(0, len(game_ids), 500):
                ids = [str(g) for g in game_ids[start : start + 500]]
                ph = ",".join("?" for _ in ids)
                df = pd.read_sql_query(f"{query} WHERE game_id IN ({ph})", conn, params=ids)
                for row in df.itertuples(index=False):
                    vals = tuple(float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                                 for c in PROCESS_TARGET_COLUMNS)
                    out[str(row.game_id)] = (vals, bool(int(row.process_target_valid or 0)))
        else:
            df = pd.read_sql_query(query, conn)
            for row in df.itertuples(index=False):
                vals = tuple(float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                             for c in PROCESS_TARGET_COLUMNS)
                out[str(row.game_id)] = (vals, bool(int(row.process_target_valid or 0)))
    return out


def fit_process_feature_stats(
    train_games: pd.DataFrame,
    feature_lookup: dict[str, tuple[float, ...]],
) -> VectorStats:
    rows = []
    for gid in train_games["game_id"].astype(str).tolist():
        rows.append(feature_lookup.get(gid, tuple(float("nan") for _ in PROCESS_PREGAME_FEATURE_COLUMNS)))
    X = np.asarray(rows, dtype="float64") if rows else np.empty((0, K_PROCESS_FEATURES), dtype="float64")
    return _fit_vector_stats(X)


def fit_process_target_stats(
    train_games: pd.DataFrame,
    target_lookup: dict[str, tuple[tuple[float, ...], bool]],
) -> VectorStats:
    rows = []
    for gid in train_games["game_id"].astype(str).tolist():
        vals, valid = target_lookup.get(gid, (tuple(float("nan") for _ in PROCESS_TARGET_COLUMNS), False))
        if valid:
            rows.append(vals)
    X = np.asarray(rows, dtype="float64") if rows else np.empty((0, K_PROCESS), dtype="float64")
    return _fit_vector_stats(X)


@dataclass(frozen=True)
class GameRecordProcessV1:
    base: GameRecordV42
    process_features: tuple[float, ...]
    process_target: tuple[float, ...]
    process_target_valid: bool

    @property
    def game_id(self) -> str:
        return self.base.game_id

    @property
    def game_date(self):
        return self.base.game_date

    def __getattr__(self, name: str):
        # Delegate the many existing v4.2 record attributes used by diagnostics
        # and backtest printing without copying the full record surface.
        return getattr(self.base, name)


def build_records_process_v1(
    games: pd.DataFrame,
    *args,
    process_feature_lookup: dict[str, tuple[float, ...]],
    process_target_lookup: dict[str, tuple[tuple[float, ...], bool]],
    process_feature_stats: VectorStats,
    process_target_stats: VectorStats,
    use_process_features: bool = True,
    **kwargs,
) -> list[GameRecordProcessV1]:
    base_records = build_records_v42(games, *args, **kwargs)
    out: list[GameRecordProcessV1] = []
    empty_feat_raw = np.full(K_PROCESS_FEATURES, np.nan, dtype="float64")
    empty_tgt_raw = np.full(K_PROCESS, np.nan, dtype="float64")
    for base in base_records:
        feat_raw = np.asarray(
            process_feature_lookup.get(base.game_id, tuple(empty_feat_raw.tolist())),
            dtype="float64",
        )
        feat_z = _transform_vector(feat_raw, process_feature_stats) if use_process_features else np.zeros(0, dtype="float32")
        tgt_raw_tuple, tgt_valid = process_target_lookup.get(
            base.game_id, (tuple(empty_tgt_raw.tolist()), False),
        )
        tgt_z = _transform_vector(np.asarray(tgt_raw_tuple, dtype="float64"), process_target_stats)
        out.append(
            GameRecordProcessV1(
                base=base,
                process_features=tuple(float(x) for x in feat_z.tolist()),
                process_target=tuple(float(x) for x in tgt_z.tolist()),
                process_target_valid=bool(tgt_valid),
            )
        )
    return out


class GameDatasetProcessV1(Dataset):
    def __init__(self, records: list[GameRecordProcessV1]) -> None:
        self.records = records
        self.base_dataset = GameDatasetV42([r.base for r in records])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        r = self.records[idx]
        process_features = torch.tensor(r.process_features, dtype=torch.float32)
        if process_features.numel():
            item["tabular"] = torch.cat([item["tabular"], process_features], dim=0)
        item["process_features"] = process_features
        item["process_target"] = torch.tensor(r.process_target, dtype=torch.float32)
        item["process_target_valid"] = torch.tensor(float(r.process_target_valid), dtype=torch.float32)
        return item


def collate_process_v1(batch: list[dict]) -> dict:
    out = collate_v42(batch)
    feat_dim = batch[0]["process_features"].numel()
    if feat_dim:
        out["process_features"] = torch.stack([b["process_features"] for b in batch])
    else:
        out["process_features"] = torch.zeros(len(batch), 0, dtype=torch.float32)
    out["process_target"] = torch.stack([b["process_target"] for b in batch])
    out["process_target_valid"] = torch.stack([b["process_target_valid"] for b in batch])
    return out


__all__ = [
    *[name for name in globals().keys() if name.isupper()],
    "DEFAULT_PROCESS_DB", "VectorStats",
    "load_process_pregame_features", "load_process_targets",
    "fit_process_feature_stats", "fit_process_target_stats",
    "GameRecordProcessV1", "build_records_process_v1",
    "GameDatasetProcessV1", "collate_process_v1",
]
