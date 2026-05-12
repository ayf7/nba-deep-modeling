"""CME-Ratings-v3 data layer: v4.2 + multi-timescale ratings priors.

Ratings-v3 consumes the leakage-safe artifact built by::

    python data/scripts/build_team_strength_ratings_v3.py

That artifact emits three opponent-adjusted pregame rating tracks per game:
slow, medium, and fast.  The neural model learns a compact ratings-only gate
that mixes these tracks while preserving the ratings-first architecture that
worked in Ratings-v2.
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
DEFAULT_RATINGS_DB = ARTIFACT_ROOT / "team_strength_ratings_v3.sqlite"
RATING_TABLE = "game_team_strength_priors_v3"
RATING_TRACKS: tuple[str, ...] = ("slow", "medium", "fast")
K_RATING_TRACKS = len(RATING_TRACKS)

V42_SCRIPTS = PROJECT_ROOT / "models_cme_v4_2" / "scripts"
sys.path.insert(0, str(V42_SCRIPTS))

from cme_v4_2_common import *  # noqa: F403,E402
from cme_v4_2_common import (  # noqa: E402
    GameDatasetV42,
    GameRecordV42,
    build_records_v42,
    collate_v42,
)

TRACK_FEATURE_BASES: tuple[str, ...] = (
    "home_net_rating",
    "away_net_rating",
    "diff_net_rating",
    "home_attack_rating",
    "away_attack_rating",
    "diff_attack_rating",
    "rating_margin_pred",
    "rating_logit_prior",
    "rating_home_win_prob",
    "league_points_baseline",
    "home_adv_margin_baseline",
)
GLOBAL_RATING_FEATURES: tuple[str, ...] = (
    "home_rating_games_before",
    "away_rating_games_before",
    "diff_rating_games_before",
    "home_lifetime_rating_games_before",
    "away_lifetime_rating_games_before",
    "rating_margin_mean",
    "rating_margin_std",
    "rating_margin_fast_minus_slow",
    "rating_logit_mean",
    "rating_logit_std",
    "rating_logit_fast_minus_slow",
    "rating_prob_mean",
    "rating_prob_std",
)
RATING_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    [f"{track}_{base}" for track in RATING_TRACKS for base in TRACK_FEATURE_BASES]
    + list(GLOBAL_RATING_FEATURES)
)
K_RATING_FEATURES = len(RATING_FEATURE_COLUMNS)
RATING_TRACK_MARGIN_COLUMNS: tuple[str, ...] = tuple(
    f"{track}_rating_margin_pred" for track in RATING_TRACKS
)
RATING_TRACK_LOGIT_COLUMNS: tuple[str, ...] = tuple(
    f"{track}_rating_logit_prior" for track in RATING_TRACKS
)
RATING_TRACK_PROB_COLUMNS: tuple[str, ...] = tuple(
    f"{track}_rating_home_win_prob" for track in RATING_TRACKS
)


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
            f"Missing ratings DB: {ratings_db}. Run data/scripts/build_team_strength_ratings_v3.py first."
        )
    all_cols = [
        "game_id",
        *RATING_FEATURE_COLUMNS,
        *RATING_TRACK_MARGIN_COLUMNS,
        *RATING_TRACK_LOGIT_COLUMNS,
        *RATING_TRACK_PROB_COLUMNS,
    ]
    # Preserve first occurrence if a column is already in RATING_FEATURE_COLUMNS.
    dedup_cols: list[str] = []
    seen: set[str] = set()
    for c in all_cols:
        if c not in seen:
            dedup_cols.append(c)
            seen.add(c)
    query = f"SELECT {', '.join(dedup_cols)} FROM {RATING_TABLE}"
    out: dict[str, tuple[float, ...]] = {}
    feature_len = len(RATING_FEATURE_COLUMNS)
    with sqlite3.connect(ratings_db) as conn:
        if game_ids:
            dfs = []
            for start in range(0, len(game_ids), 500):
                ids = [str(g) for g in game_ids[start : start + 500]]
                ph = ",".join("?" for _ in ids)
                dfs.append(pd.read_sql_query(f"{query} WHERE game_id IN ({ph})", conn, params=ids))
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=dedup_cols)
        else:
            df = pd.read_sql_query(query, conn)
    for row in df.itertuples(index=False):
        feat = [float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan") for c in RATING_FEATURE_COLUMNS]
        margins = [float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan") for c in RATING_TRACK_MARGIN_COLUMNS]
        logits = [float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan") for c in RATING_TRACK_LOGIT_COLUMNS]
        probs = [float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan") for c in RATING_TRACK_PROB_COLUMNS]
        if len(feat) != feature_len:
            raise RuntimeError("rating feature length mismatch")
        out[str(row.game_id)] = tuple(feat + margins + logits + probs)
    return out


def _split_lookup_tuple(vals: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f = K_RATING_FEATURES
    t = K_RATING_TRACKS
    arr = np.asarray(vals, dtype="float64")
    if arr.shape[0] != f + 3 * t:
        raise ValueError(f"Unexpected rating tuple length={arr.shape[0]} expected={f + 3*t}")
    return arr[:f], arr[f:f+t], arr[f+t:f+2*t], arr[f+2*t:f+3*t]


def fit_rating_feature_stats(
    train_games: pd.DataFrame,
    rating_lookup: dict[str, tuple[float, ...]],
) -> VectorStats:
    rows: list[np.ndarray] = []
    for gid in train_games["game_id"].astype(str).tolist():
        vals = rating_lookup.get(str(gid))
        if vals is None:
            rows.append(np.full(K_RATING_FEATURES, np.nan, dtype="float64"))
        else:
            feat, _, _, _ = _split_lookup_tuple(vals)
            rows.append(feat)
    X = np.asarray(rows, dtype="float64") if rows else np.empty((0, K_RATING_FEATURES), dtype="float64")
    return _fit_vector_stats(X)


@dataclass(frozen=True)
class GameRecordRatingsV3:
    base: GameRecordV42
    rating_features: tuple[float, ...]
    rating_track_margins: tuple[float, ...]
    rating_track_logits: tuple[float, ...]
    rating_track_probs: tuple[float, ...]
    has_rating_prior: bool

    @property
    def game_id(self) -> str:
        return self.base.game_id

    @property
    def game_date(self):
        return self.base.game_date

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def build_records_ratings_v3(
    games: pd.DataFrame,
    *args,
    rating_lookup: dict[str, tuple[float, ...]],
    rating_feature_stats: VectorStats,
    use_rating_features: bool = True,
    **kwargs,
) -> list[GameRecordRatingsV3]:
    base_records = build_records_v42(games, *args, **kwargs)
    out: list[GameRecordRatingsV3] = []
    empty_feat = np.full(K_RATING_FEATURES, np.nan, dtype="float64")
    empty_tracks = np.full(K_RATING_TRACKS, np.nan, dtype="float64")
    for base in base_records:
        vals = rating_lookup.get(base.game_id)
        has_rating = vals is not None
        if vals is None:
            feat_raw, margins, logits, probs = empty_feat, empty_tracks, empty_tracks, empty_tracks
        else:
            feat_raw, margins, logits, probs = _split_lookup_tuple(vals)
        feat_z = _transform_vector(feat_raw, rating_feature_stats) if use_rating_features else np.zeros(0, dtype="float32")
        margins = np.where(np.isfinite(margins), margins, 0.0)
        logits = np.where(np.isfinite(logits), logits, 0.0)
        probs = np.where(np.isfinite(probs), probs, 0.5)
        out.append(
            GameRecordRatingsV3(
                base=base,
                rating_features=tuple(float(x) for x in feat_z.tolist()),
                rating_track_margins=tuple(float(x) for x in margins.tolist()),
                rating_track_logits=tuple(float(x) for x in logits.tolist()),
                rating_track_probs=tuple(float(x) for x in probs.tolist()),
                has_rating_prior=bool(has_rating),
            )
        )
    return out


class GameDatasetRatingsV3(Dataset):
    def __init__(self, records: list[GameRecordRatingsV3]) -> None:
        self.records = records
        self.base_dataset = GameDatasetV42([r.base for r in records])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        r = self.records[idx]
        item["rating_features"] = torch.tensor(r.rating_features, dtype=torch.float32)
        item["rating_track_margins"] = torch.tensor(r.rating_track_margins, dtype=torch.float32)
        item["rating_track_logits"] = torch.tensor(r.rating_track_logits, dtype=torch.float32)
        item["rating_track_probs"] = torch.tensor(r.rating_track_probs, dtype=torch.float32)
        item["has_rating_prior"] = torch.tensor(float(r.has_rating_prior), dtype=torch.float32)
        return item


def collate_ratings_v3(batch: list[dict]) -> dict:
    out = collate_v42(batch)
    feat_dim = batch[0]["rating_features"].numel()
    if feat_dim:
        out["rating_features"] = torch.stack([b["rating_features"] for b in batch])
    else:
        out["rating_features"] = torch.zeros(len(batch), 0, dtype=torch.float32)
    out["rating_track_margins"] = torch.stack([b["rating_track_margins"] for b in batch])
    out["rating_track_logits"] = torch.stack([b["rating_track_logits"] for b in batch])
    out["rating_track_probs"] = torch.stack([b["rating_track_probs"] for b in batch])
    out["has_rating_prior"] = torch.stack([b["has_rating_prior"] for b in batch])
    return out


__all__ = [
    *[name for name in globals().keys() if name.isupper()],
    "DEFAULT_RATINGS_DB",
    "RATING_TABLE",
    "RATING_TRACKS",
    "K_RATING_TRACKS",
    "RATING_FEATURE_COLUMNS",
    "RATING_TRACK_MARGIN_COLUMNS",
    "RATING_TRACK_LOGIT_COLUMNS",
    "RATING_TRACK_PROB_COLUMNS",
    "K_RATING_FEATURES",
    "VectorStats",
    "load_rating_priors",
    "fit_rating_feature_stats",
    "GameRecordRatingsV3",
    "build_records_ratings_v3",
    "GameDatasetRatingsV3",
    "collate_ratings_v3",
]
