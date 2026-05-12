"""CME-Ratings-v4 data layer: Ratings-v2 + pregame roster-shock features.

Ratings-v4 preserves the strong single-track dynamic ratings prior from
Ratings-v2, then attaches leakage-safe game-level roster-shock features built
by::

    python data/scripts/build_roster_shock_features.py

The shock artifact estimates how much recent rotation mass is listed as out,
questionable, or otherwise unavailable for each side before tip-off.  These
features are standardized using the train window only, while the signed
``roster_shock_advantage_signal`` is also provided in raw bounded form for a
small structured logit adjustment.
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
DEFAULT_ROSTER_SHOCK_DB = ARTIFACT_ROOT / "roster_shock_features.sqlite"
ROSTER_SHOCK_TABLE = "game_roster_shock_features"

RATINGS_V2_SCRIPTS = PROJECT_ROOT / "models_cme_ratings_v2" / "scripts"
sys.path.insert(0, str(RATINGS_V2_SCRIPTS))

from cme_ratings_v2_common import *  # noqa: F403,E402
from cme_ratings_v2_common import (  # noqa: E402
    BOX_INDEX,
    BOX_TARGETS,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB,
    DEFAULT_LINEUP_DECAY,
    DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB,
    DEFAULT_PLAYER_FORM_DECAY,
    DEFAULT_PLAYER_FORM_LOOKBACK,
    DEFAULT_RATINGS_DB,
    GameDatasetRatingsV2,
    GameRecordRatingsV2,
    K_BOX,
    K_PAIR,
    K_RATING_FEATURES,
    PAIR_TARGETS,
    PLAYER_FORM_DIM,
    RATING_FEATURE_COLUMNS,
    TABULAR_FEATURE_COLUMNS,
    VectorStats,
    _fit_vector_stats,
    _transform_vector,
    build_records_ratings_v2,
    build_team_vocab,
    build_vocab_from_records_v42,
    collate_ratings_v2,
    fit_player_form_stats,
    fit_rating_feature_stats,
    fit_tabular_stats,
    load_game_odds,
    load_game_player_status,
    load_game_scores,
    load_games,
    load_matchup_rows_v2,
    load_player_game_stats,
    load_player_histories,
    load_rating_priors,
    load_status_calibration,
    load_team_exposures,
)

ROSTER_SHOCK_FEATURE_COLUMNS: tuple[str, ...] = (
    "home_rotation_mass_seconds",
    "away_rotation_mass_seconds",
    "home_rotation_mass_points",
    "away_rotation_mass_points",
    "home_tracked_players",
    "away_tracked_players",
    "home_games_seen",
    "away_games_seen",
    "home_report_count",
    "away_report_count",
    "home_out_count",
    "away_out_count",
    "home_uncertain_count",
    "away_uncertain_count",
    "home_probable_count",
    "away_probable_count",
    "home_report_rotation_share",
    "away_report_rotation_share",
    "home_unavailable_seconds_share",
    "away_unavailable_seconds_share",
    "home_unavailable_points_share",
    "away_unavailable_points_share",
    "home_out_seconds_share",
    "away_out_seconds_share",
    "home_uncertain_seconds_share",
    "away_uncertain_seconds_share",
    "home_top1_unavailable_share",
    "away_top1_unavailable_share",
    "home_top3_unavailable_share",
    "away_top3_unavailable_share",
    "home_mean_minutes_to_tip",
    "away_mean_minutes_to_tip",
    "diff_unavailable_seconds_share",
    "diff_unavailable_points_share",
    "diff_out_seconds_share",
    "diff_uncertain_seconds_share",
    "diff_top1_unavailable_share",
    "diff_top3_unavailable_share",
    "has_roster_shock_report",
)
K_ROSTER_SHOCK_FEATURES = len(ROSTER_SHOCK_FEATURE_COLUMNS)
ROSTER_SHOCK_SIGNAL_COLUMN = "roster_shock_advantage_signal"


def load_roster_shock_features(
    shock_db: Path,
    game_ids: list[str] | None = None,
) -> dict[str, tuple[float, ...]]:
    if not shock_db.exists():
        raise FileNotFoundError(
            f"Missing roster-shock DB: {shock_db}. Run data/scripts/build_roster_shock_features.py first."
        )
    cols = ", ".join(["game_id", *ROSTER_SHOCK_FEATURE_COLUMNS, ROSTER_SHOCK_SIGNAL_COLUMN])
    query = f"SELECT {cols} FROM {ROSTER_SHOCK_TABLE}"
    out: dict[str, tuple[float, ...]] = {}
    with sqlite3.connect(shock_db) as conn:
        if game_ids:
            for start in range(0, len(game_ids), 500):
                ids = [str(g) for g in game_ids[start : start + 500]]
                ph = ",".join("?" for _ in ids)
                df = pd.read_sql_query(f"{query} WHERE game_id IN ({ph})", conn, params=ids)
                for row in df.itertuples(index=False):
                    vals = tuple(
                        float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                        for c in (*ROSTER_SHOCK_FEATURE_COLUMNS, ROSTER_SHOCK_SIGNAL_COLUMN)
                    )
                    out[str(row.game_id)] = vals
        else:
            df = pd.read_sql_query(query, conn)
            for row in df.itertuples(index=False):
                vals = tuple(
                    float(getattr(row, c)) if pd.notna(getattr(row, c)) else float("nan")
                    for c in (*ROSTER_SHOCK_FEATURE_COLUMNS, ROSTER_SHOCK_SIGNAL_COLUMN)
                )
                out[str(row.game_id)] = vals
    return out


def fit_roster_shock_feature_stats(
    train_games: pd.DataFrame,
    shock_lookup: dict[str, tuple[float, ...]],
) -> VectorStats:
    rows = [
        shock_lookup.get(str(gid), tuple(float("nan") for _ in range(K_ROSTER_SHOCK_FEATURES + 1)))[:K_ROSTER_SHOCK_FEATURES]
        for gid in train_games["game_id"].astype(str).tolist()
    ]
    X = np.asarray(rows, dtype="float64") if rows else np.empty((0, K_ROSTER_SHOCK_FEATURES), dtype="float64")
    return _fit_vector_stats(X)


@dataclass(frozen=True)
class GameRecordRatingsV4:
    base: GameRecordRatingsV2
    roster_shock_features: tuple[float, ...]
    roster_shock_advantage_signal: float
    has_roster_shock: bool

    @property
    def game_id(self) -> str:
        return self.base.game_id

    @property
    def game_date(self):
        return self.base.game_date

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def build_records_ratings_v4(
    games: pd.DataFrame,
    *args,
    roster_shock_lookup: dict[str, tuple[float, ...]],
    roster_shock_feature_stats: VectorStats,
    use_roster_shock_features: bool = True,
    **kwargs,
) -> list[GameRecordRatingsV4]:
    base_records = build_records_ratings_v2(games, *args, **kwargs)
    out: list[GameRecordRatingsV4] = []
    empty = np.full(K_ROSTER_SHOCK_FEATURES + 1, np.nan, dtype="float64")
    for base in base_records:
        vals = roster_shock_lookup.get(base.game_id)
        has_shock = vals is not None
        raw = np.asarray(vals if vals is not None else tuple(empty.tolist()), dtype="float64")
        raw_feat = raw[:K_ROSTER_SHOCK_FEATURES]
        feat_z = (
            _transform_vector(raw_feat, roster_shock_feature_stats)
            if use_roster_shock_features
            else np.zeros(0, dtype="float32")
        )
        signal = float(raw[K_ROSTER_SHOCK_FEATURES]) if has_shock and np.isfinite(raw[K_ROSTER_SHOCK_FEATURES]) else 0.0
        out.append(GameRecordRatingsV4(
            base=base,
            roster_shock_features=tuple(float(x) for x in feat_z.tolist()),
            roster_shock_advantage_signal=signal,
            has_roster_shock=bool(has_shock),
        ))
    return out


class GameDatasetRatingsV4(Dataset):
    def __init__(self, records: list[GameRecordRatingsV4]) -> None:
        self.records = records
        self.base_dataset = GameDatasetRatingsV2([r.base for r in records])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        r = self.records[idx]
        item["roster_shock_features"] = torch.tensor(r.roster_shock_features, dtype=torch.float32)
        item["roster_shock_advantage_signal"] = torch.tensor(r.roster_shock_advantage_signal, dtype=torch.float32)
        item["has_roster_shock"] = torch.tensor(float(r.has_roster_shock), dtype=torch.float32)
        return item


def collate_ratings_v4(batch: list[dict]) -> dict:
    out = collate_ratings_v2(batch)
    feat_dim = batch[0]["roster_shock_features"].numel()
    if feat_dim:
        out["roster_shock_features"] = torch.stack([b["roster_shock_features"] for b in batch])
    else:
        out["roster_shock_features"] = torch.zeros(len(batch), 0, dtype=torch.float32)
    out["roster_shock_advantage_signal"] = torch.stack([b["roster_shock_advantage_signal"] for b in batch])
    out["has_roster_shock"] = torch.stack([b["has_roster_shock"] for b in batch])
    return out


__all__ = [
    *[name for name in globals().keys() if name.isupper()],
    "DEFAULT_ROSTER_SHOCK_DB",
    "ROSTER_SHOCK_FEATURE_COLUMNS",
    "K_ROSTER_SHOCK_FEATURES",
    "ROSTER_SHOCK_SIGNAL_COLUMN",
    "load_roster_shock_features",
    "fit_roster_shock_feature_stats",
    "GameRecordRatingsV4",
    "build_records_ratings_v4",
    "GameDatasetRatingsV4",
    "collate_ratings_v4",
]
