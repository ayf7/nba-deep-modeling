"""CME-Ratings-v1 data layer: v4.2 + dynamic opponent-adjusted team priors.

This branch keeps the proven CME-v4.2 record construction unchanged and wraps
it with a compact set of strictly pregame dynamic team-strength features built
by::

    python data/scripts/build_team_strength_ratings.py

The rating artifact is generated sequentially from completed games only.  For
any focal game, its ratings row was emitted before that game's score was used
to update offense/defense strengths.
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
DEFAULT_RATINGS_DB = ARTIFACT_ROOT / "team_strength_ratings.sqlite"
RATING_TABLE = "game_team_strength_priors"

V42_SCRIPTS = PROJECT_ROOT / "models_cme_v4_2" / "scripts"
sys.path.insert(0, str(V42_SCRIPTS))

from cme_v4_2_common import *  # noqa: F403,E402
from cme_v4_2_common import (  # noqa: E402
    GameDatasetV42,
    GameRecordV42,
    build_records_v42,
    collate_v42,
)

RATING_FEATURE_COLUMNS: tuple[str, ...] = (
    "home_off_rating",
    "home_def_rating",
    "home_net_rating",
    "away_off_rating",
    "away_def_rating",
    "away_net_rating",
    "diff_net_rating",
    "home_attack_rating",
    "away_attack_rating",
    "diff_attack_rating",
    "rating_home_points_pred",
    "rating_away_points_pred",
    "rating_margin_pred",
    "rating_home_win_prob",
    "home_rating_games_before",
    "away_rating_games_before",
    "diff_rating_games_before",
    "home_lifetime_rating_games_before",
    "away_lifetime_rating_games_before",
    "league_points_baseline",
    "home_adv_margin_baseline",
)
K_RATING_FEATURES = len(RATING_FEATURE_COLUMNS)
RATING_MARGIN_COLUMN = "rating_margin_pred"
RATING_HOME_WIN_PROB_COLUMN = "rating_home_win_prob"


@dataclass(frozen=True)
class VectorStats:
    medians: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def _fit_vector_stats(rows: np.ndarray) -> VectorStats:
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D, got shape={rows.shape}")
    if rows.shape[0] == 0:
        return VectorStats(
            medians=np.zeros(rows.shape[1], dtype="float64"),
            means=np.zeros(rows.shape[1], dtype="float64"),
            stds=np.ones(rows.shape[1], dtype="float64"),
        )
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


def load_rating_priors(
    ratings_db: Path,
    game_ids: list[str] | None = None,
) -> dict[str, tuple[float, ...]]:
    if not ratings_db.exists():
        raise FileNotFoundError(
            f"Missing ratings DB: {ratings_db}. Run data/scripts/build_team_strength_ratings.py first."
        )
    cols = ", ".join(["game_id", *RATING_FEATURE_COLUMNS])
    query = f"SELECT {cols} FROM {RATING_TABLE}"
    out: dict[str, tuple[float, ...]] = {}
    with sqlite3.connect(ratings_db) as conn:
        if game_ids:
            for start in range(0, len(game_ids), 500):
                ids = [str(g) for g in game_ids[start : start + 500]]
                ph = ",".join("?" for _ in ids)
                df = pd.read_sql_query(f"{query} WHERE game_id IN ({ph})", conn, params=ids)
                for row in df.itertuples(index=False):
                    vals = tuple(
                        float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                        for c in RATING_FEATURE_COLUMNS
                    )
                    out[str(row.game_id)] = vals
        else:
            df = pd.read_sql_query(query, conn)
            for row in df.itertuples(index=False):
                vals = tuple(
                    float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                    for c in RATING_FEATURE_COLUMNS
                )
                out[str(row.game_id)] = vals
    return out


def fit_rating_feature_stats(
    train_games: pd.DataFrame,
    rating_lookup: dict[str, tuple[float, ...]],
) -> VectorStats:
    rows = [
        rating_lookup.get(
            str(gid), tuple(float("nan") for _ in RATING_FEATURE_COLUMNS),
        )
        for gid in train_games["game_id"].astype(str).tolist()
    ]
    X = (
        np.asarray(rows, dtype="float64")
        if rows
        else np.empty((0, K_RATING_FEATURES), dtype="float64")
    )
    return _fit_vector_stats(X)


@dataclass(frozen=True)
class GameRecordRatingsV1:
    base: GameRecordV42
    rating_features: tuple[float, ...]
    rating_margin_prior: float
    rating_home_win_prob_prior: float
    has_rating_prior: bool

    @property
    def game_id(self) -> str:
        return self.base.game_id

    @property
    def game_date(self):
        return self.base.game_date

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def build_records_ratings_v1(
    games: pd.DataFrame,
    *args,
    rating_lookup: dict[str, tuple[float, ...]],
    rating_feature_stats: VectorStats,
    use_rating_features: bool = True,
    **kwargs,
) -> list[GameRecordRatingsV1]:
    base_records = build_records_v42(games, *args, **kwargs)
    out: list[GameRecordRatingsV1] = []
    empty = np.full(K_RATING_FEATURES, np.nan, dtype="float64")
    margin_idx = RATING_FEATURE_COLUMNS.index(RATING_MARGIN_COLUMN)
    prob_idx = RATING_FEATURE_COLUMNS.index(RATING_HOME_WIN_PROB_COLUMN)
    for base in base_records:
        vals = rating_lookup.get(base.game_id)
        has_rating = vals is not None
        raw = np.asarray(vals if vals is not None else tuple(empty.tolist()), dtype="float64")
        feat_z = _transform_vector(raw, rating_feature_stats) if use_rating_features else np.zeros(0, dtype="float32")
        rating_margin = float(raw[margin_idx]) if has_rating and np.isfinite(raw[margin_idx]) else 0.0
        rating_prob = float(raw[prob_idx]) if has_rating and np.isfinite(raw[prob_idx]) else 0.5
        out.append(
            GameRecordRatingsV1(
                base=base,
                rating_features=tuple(float(x) for x in feat_z.tolist()),
                rating_margin_prior=rating_margin,
                rating_home_win_prob_prior=rating_prob,
                has_rating_prior=bool(has_rating),
            )
        )
    return out


class GameDatasetRatingsV1(Dataset):
    def __init__(self, records: list[GameRecordRatingsV1]) -> None:
        self.records = records
        self.base_dataset = GameDatasetV42([r.base for r in records])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        r = self.records[idx]
        item["rating_features"] = torch.tensor(r.rating_features, dtype=torch.float32)
        item["rating_margin_prior"] = torch.tensor(r.rating_margin_prior, dtype=torch.float32)
        item["rating_home_win_prob_prior"] = torch.tensor(r.rating_home_win_prob_prior, dtype=torch.float32)
        item["has_rating_prior"] = torch.tensor(float(r.has_rating_prior), dtype=torch.float32)
        return item


def collate_ratings_v1(batch: list[dict]) -> dict:
    out = collate_v42(batch)
    feat_dim = batch[0]["rating_features"].numel()
    if feat_dim:
        out["rating_features"] = torch.stack([b["rating_features"] for b in batch])
    else:
        out["rating_features"] = torch.zeros(len(batch), 0, dtype=torch.float32)
    out["rating_margin_prior"] = torch.stack([b["rating_margin_prior"] for b in batch])
    out["rating_home_win_prob_prior"] = torch.stack([b["rating_home_win_prob_prior"] for b in batch])
    out["has_rating_prior"] = torch.stack([b["has_rating_prior"] for b in batch])
    return out


__all__ = [
    *[name for name in globals().keys() if name.isupper()],
    "DEFAULT_RATINGS_DB",
    "RATING_FEATURE_COLUMNS",
    "K_RATING_FEATURES",
    "VectorStats",
    "load_rating_priors",
    "fit_rating_feature_stats",
    "GameRecordRatingsV1",
    "build_records_ratings_v1",
    "GameDatasetRatingsV1",
    "collate_ratings_v1",
]
