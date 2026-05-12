"""CME-v4.4 data layer: v4.2 + leakage-safe rolling process-style features.

This experiment deliberately avoids the Process-CME v1 auxiliary target losses.
Instead, it treats process history as *pregame input signal* and lets the
already-proven v4.2 winner model decide whether the features help.

The process artifact is built by::

    python data/scripts/build_game_process_features.py

It contains leakage-safe rolling features such as possessions, shot-mix rates,
turnover rates, offensive-rebound rates, foul-drawing rates, opponent-allowed
versions, and offense-vs-defense interaction gaps.  v4.4 z-scores those
features using the train split/window only and appends them to v4.2's existing
tabular vector.
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
    out: dict[str, tuple[float, ...]] = {}
    with sqlite3.connect(process_db) as conn:
        if game_ids:
            for start in range(0, len(game_ids), 500):
                ids = [str(g) for g in game_ids[start : start + 500]]
                ph = ",".join("?" for _ in ids)
                df = pd.read_sql_query(f"{query} WHERE game_id IN ({ph})", conn, params=ids)
                for row in df.itertuples(index=False):
                    gid = str(row.game_id)
                    vals = tuple(
                        float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                        for c in PROCESS_PREGAME_FEATURE_COLUMNS
                    )
                    out[gid] = vals
        else:
            df = pd.read_sql_query(query, conn)
            for row in df.itertuples(index=False):
                gid = str(row.game_id)
                vals = tuple(
                    float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                    for c in PROCESS_PREGAME_FEATURE_COLUMNS
                )
                out[gid] = vals
    return out


def fit_process_feature_stats(
    train_games: pd.DataFrame,
    feature_lookup: dict[str, tuple[float, ...]],
) -> VectorStats:
    rows = [
        feature_lookup.get(
            str(gid), tuple(float("nan") for _ in PROCESS_PREGAME_FEATURE_COLUMNS),
        )
        for gid in train_games["game_id"].astype(str).tolist()
    ]
    X = (
        np.asarray(rows, dtype="float64")
        if rows
        else np.empty((0, K_PROCESS_FEATURES), dtype="float64")
    )
    return _fit_vector_stats(X)


@dataclass(frozen=True)
class GameRecordV44:
    base: GameRecordV42
    process_features: tuple[float, ...]

    @property
    def game_id(self) -> str:
        return self.base.game_id

    @property
    def game_date(self):
        return self.base.game_date

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def build_records_v44(
    games: pd.DataFrame,
    *args,
    process_feature_lookup: dict[str, tuple[float, ...]],
    process_feature_stats: VectorStats,
    use_process_features: bool = True,
    **kwargs,
) -> list[GameRecordV44]:
    base_records = build_records_v42(games, *args, **kwargs)
    out: list[GameRecordV44] = []
    empty_feat_raw = np.full(K_PROCESS_FEATURES, np.nan, dtype="float64")
    for base in base_records:
        feat_raw = np.asarray(
            process_feature_lookup.get(base.game_id, tuple(empty_feat_raw.tolist())),
            dtype="float64",
        )
        feat_z = (
            _transform_vector(feat_raw, process_feature_stats)
            if use_process_features
            else np.zeros(0, dtype="float32")
        )
        out.append(
            GameRecordV44(
                base=base,
                process_features=tuple(float(x) for x in feat_z.tolist()),
            )
        )
    return out


class GameDatasetV44(Dataset):
    def __init__(
        self,
        records: list[GameRecordV44],
        *,
        include_base_tabular: bool = True,
    ) -> None:
        self.records = records
        self.base_dataset = GameDatasetV42([r.base for r in records])
        self.include_base_tabular = include_base_tabular

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        if not self.include_base_tabular:
            item["tabular"] = torch.zeros(0, dtype=torch.float32)
        r = self.records[idx]
        process_features = torch.tensor(r.process_features, dtype=torch.float32)
        if process_features.numel():
            item["tabular"] = torch.cat([item["tabular"], process_features], dim=0)
        item["process_features"] = process_features
        return item


def collate_v44(batch: list[dict]) -> dict:
    out = collate_v42(batch)
    feat_dim = batch[0]["process_features"].numel()
    if feat_dim:
        out["process_features"] = torch.stack([b["process_features"] for b in batch])
    else:
        out["process_features"] = torch.zeros(len(batch), 0, dtype=torch.float32)
    return out


__all__ = [
    *[name for name in globals().keys() if name.isupper()],
    "DEFAULT_PROCESS_DB",
    "VectorStats",
    "load_process_pregame_features",
    "fit_process_feature_stats",
    "GameRecordV44",
    "build_records_v44",
    "GameDatasetV44",
    "collate_v44",
]
