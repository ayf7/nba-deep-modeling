"""CME-Availability-v1.1 data layer.

This branch keeps the proven CME-v4.2 game/player/matchup construction, then
augments each pregame roster slot with leakage-safe availability/rotation
features and targets:

* play target: did the player appear in matchup exposure for this game?
* log-seconds target: log1p(actual involvement seconds)
* role-share target: actual involvement seconds normalized within the modeled
  pregame candidate roster.

The features are intentionally pregame-only.  They use:

* the latest injury-report row already materialized by
  ``game_player_availability`` (status, reason, minutes_to_tip),
* the existing global ``P(plays | status)`` calibration as a stable prior, and
* strictly-before-game player/team exposure history from ``player_matchups``.

The model uses these features to learn a correction on top of the status-only
play prior and to learn a recent-role/minutes prior that is injected into the
CME involvement softmaxes.
"""
from __future__ import annotations

import bisect
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
V42_SCRIPTS = REPO_ROOT / "models_cme_v4_2" / "scripts"
V2_SCRIPTS = REPO_ROOT / "models_cme_v2" / "scripts"
if str(V42_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V42_SCRIPTS))
if str(V2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V2_SCRIPTS))

from cme_v4_2_common import (  # noqa: E402
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
    GameDatasetV42,
    GameRecordV42,
    K_BOX,
    K_PAIR,
    K_PLAYER,
    MIN_EXPOSURE_SECONDS,
    PAIR_TARGETS,
    PLAYER_FORM_DIM,
    PLAYER_FORM_FEATURE_NAMES,
    PLAYER_TARGETS,
    REST_DAYS_CLIP,
    SEASON_PHASE_DAYS,
    SEASON_PHASE_DIM,
    TABULAR_FEATURE_COLUMNS,
    TeamGameExposure,
    TeamVocab,
    Vocab,
    build_full_lineup,
    build_records_v42,
    build_team_vocab,
    build_vocab_from_records_v42,
    collate_v42,
    fit_player_form_stats,
    fit_tabular_stats,
    load_game_odds,
    load_game_player_status,
    load_game_scores,
    load_games,
    load_matchup_rows_v2,
    load_player_game_stats,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
    play_prob_for,
)


AVAILABILITY_FEATURE_NAMES: tuple[str, ...] = (
    "status_prior_play_prob",
    "is_listed",
    "status_available",
    "status_probable",
    "status_questionable",
    "status_doubtful",
    "status_out",
    "status_other_listed",
    "log1p_hours_to_tip",
    "reason_rest_or_management",
    "reason_illness",
    "reason_assignment_or_gleague",
    "reason_nonempty",
    "recent_weighted_appearance_rate",
    "recent_games_seen_frac",
    "log1p_recent_avg_involvement_seconds",
    "recent_role_share",
    "log1p_last_involvement_seconds",
    "games_since_last_appearance_scaled",
    "recent_involvement_cv",
)
AVAILABILITY_FEATURE_DIM = len(AVAILABILITY_FEATURE_NAMES)

# Target validity threshold for within-roster role-share supervision.
MIN_ROLE_TARGET_SECONDS = 60.0


@dataclass(frozen=True)
class PlayerStatusDetail:
    status: str = ""
    reason: str = ""
    minutes_to_tip: float | None = None

    @property
    def listed(self) -> bool:
        return bool(self.status)


@dataclass(frozen=True)
class AvailabilityFeatureStats:
    means: np.ndarray
    stds: np.ndarray


@dataclass(frozen=True)
class GameRecordAvailabilityV11:
    base: GameRecordV42
    home_avail_raw: tuple[tuple[float, ...], ...]
    away_avail_raw: tuple[tuple[float, ...], ...]
    home_avail_minute_log_prior: tuple[float, ...]
    away_avail_minute_log_prior: tuple[float, ...]
    home_avail_play_actual: tuple[float, ...]
    away_avail_play_actual: tuple[float, ...]
    home_avail_log_seconds_actual: tuple[float, ...]
    away_avail_log_seconds_actual: tuple[float, ...]
    home_avail_role_share_actual: tuple[float, ...]
    away_avail_role_share_actual: tuple[float, ...]
    home_avail_role_valid: bool
    away_avail_role_valid: bool


def _status_bucket(status: str) -> str:
    s = (status or "").strip().casefold()
    if not s:
        return "not_listed"
    if "available" in s:
        return "available"
    if "probable" in s:
        return "probable"
    if "question" in s:
        return "questionable"
    if "doubt" in s:
        return "doubtful"
    if s == "out" or " out" in f" {s}" or s.startswith("out"):
        return "out"
    return "other_listed"


def _reason_flags(reason: str) -> tuple[float, float, float, float]:
    r = (reason or "").casefold()
    nonempty = float(bool(r.strip()))
    rest = float(any(k in r for k in ("rest", "management", "load", "conditioning")))
    illness = float(any(k in r for k in ("illness", "sick", "flu", "virus", "health")))
    assignment = float(any(k in r for k in ("g league", "gleague", "assignment", "two-way", "two way")))
    return rest, illness, assignment, nonempty


def load_game_player_status_details(
    injury_db: Path, game_ids: Iterable[str]
) -> dict[tuple[str, str], PlayerStatusDetail]:
    """Return latest pretip injury-report detail rows keyed by (game_id, player_id)."""
    ids = list({str(g) for g in game_ids})
    if not ids:
        return {}
    out: dict[tuple[str, str], PlayerStatusDetail] = {}
    chunk = 500
    with sqlite3.connect(injury_db) as conn:
        for start in range(0, len(ids), chunk):
            sub = ids[start : start + chunk]
            placeholders = ",".join("?" * len(sub))
            df = pd.read_sql_query(
                f"SELECT game_id, player_id, status, reason, minutes_to_tip "
                f"FROM game_player_availability WHERE game_id IN ({placeholders})",
                conn,
                params=sub,
            )
            for row in df.itertuples(index=False):
                mts = None
                if pd.notna(row.minutes_to_tip):
                    mts = float(row.minutes_to_tip)
                out[(str(row.game_id), str(row.player_id))] = PlayerStatusDetail(
                    status=str(row.status) if pd.notna(row.status) and row.status else "",
                    reason=str(row.reason) if pd.notna(row.reason) and row.reason else "",
                    minutes_to_tip=mts,
                )
    return out


def build_team_game_player_seconds_index(
    histories: dict[str, list[TeamGameExposure]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Return {(game_id, team_id): {player_id: involvement_seconds}}."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for team_id, recs in histories.items():
        for rec in recs:
            out[(str(rec.game_id), str(team_id))] = {
                str(pid): float(seconds)
                for pid, seconds in rec.player_seconds.items()
                if float(seconds) > 0.0
            }
    return out


def _prior_team_games(
    histories: dict[str, list[TeamGameExposure]],
    *,
    team_id: str,
    game_date: pd.Timestamp,
    lookback_games: int,
) -> list[TeamGameExposure]:
    recs = histories.get(str(team_id), [])
    # Histories are chronologically ordered.  A simple list filter is fine at
    # this dataset size and mirrors build_full_lineup's leakage-safe logic.
    prior = [rec for rec in recs if rec.game_date < game_date]
    return prior[-lookback_games:]


def build_availability_raw_features(
    histories: dict[str, list[TeamGameExposure]],
    *,
    team_id: str,
    player_id: str,
    game_date: pd.Timestamp,
    lookback_games: int,
    decay: float,
    status_detail: PlayerStatusDetail,
    calibration: dict[str, float],
) -> tuple[np.ndarray, float]:
    """Return (raw feature vector, minute_log_prior) for one pregame player slot.

    ``minute_log_prior`` is a raw, interpretable baseline used by the model's
    minute head.  It is not standardized; the availability head predicts a
    bounded residual around this prior.
    """
    prior = _prior_team_games(
        histories,
        team_id=team_id,
        game_date=game_date,
        lookback_games=lookback_games,
    )
    n = len(prior)
    # recency_idx = 0 for most recent
    weights = decay ** np.arange(n - 1, -1, -1, dtype="float32") if n else np.zeros(0, dtype="float32")
    # We constructed weights oldest->newest above: most recent has 1.0.
    seconds = np.array(
        [float(rec.player_seconds.get(str(player_id), 0.0)) for rec in prior],
        dtype="float32",
    )
    team_totals = np.array(
        [float(sum(rec.player_seconds.values())) for rec in prior],
        dtype="float32",
    )
    if n:
        weight_sum = float(weights.sum())
        weighted_seconds = float((weights * seconds).sum())
        weighted_team_seconds = float((weights * team_totals).sum())
        weighted_appearance = float((weights * (seconds > 0.0).astype("float32")).sum()) / max(weight_sum, 1e-6)
        recent_avg_seconds = weighted_seconds / max(weight_sum, 1e-6)
        recent_role_share = weighted_seconds / max(weighted_team_seconds, 1e-6)
        last_seconds = float(seconds[-1])
        present_idxs = np.where(seconds > 0.0)[0]
        if present_idxs.size:
            games_since_last = float(n - 1 - int(present_idxs[-1]))
        else:
            games_since_last = float(lookback_games + 1)
        mean_secs = float(seconds.mean())
        std_secs = float(seconds.std())
        cv = std_secs / max(mean_secs, 1.0)
    else:
        weighted_appearance = 0.0
        recent_avg_seconds = 0.0
        recent_role_share = 0.0
        last_seconds = 0.0
        games_since_last = float(lookback_games + 1)
        cv = 0.0

    status_bucket = _status_bucket(status_detail.status)
    prior_play_prob = float(play_prob_for(status_detail.status, calibration))
    hours_to_tip = 0.0
    if status_detail.minutes_to_tip is not None and math.isfinite(status_detail.minutes_to_tip):
        hours_to_tip = max(float(status_detail.minutes_to_tip), 0.0) / 60.0
    reason_rest, reason_illness, reason_assignment, reason_nonempty = _reason_flags(status_detail.reason)
    status_flags = {
        "available": float(status_bucket == "available"),
        "probable": float(status_bucket == "probable"),
        "questionable": float(status_bucket == "questionable"),
        "doubtful": float(status_bucket == "doubtful"),
        "out": float(status_bucket == "out"),
        "other_listed": float(status_bucket == "other_listed"),
    }

    minute_log_prior = float(math.log1p(max(recent_avg_seconds, 0.0)))
    raw = np.array(
        [
            prior_play_prob,
            float(status_detail.listed),
            status_flags["available"],
            status_flags["probable"],
            status_flags["questionable"],
            status_flags["doubtful"],
            status_flags["out"],
            status_flags["other_listed"],
            math.log1p(hours_to_tip),
            reason_rest,
            reason_illness,
            reason_assignment,
            reason_nonempty,
            weighted_appearance,
            float(n) / max(float(lookback_games), 1.0),
            minute_log_prior,
            recent_role_share,
            math.log1p(max(last_seconds, 0.0)),
            min(games_since_last, float(lookback_games + 1)) / max(float(lookback_games + 1), 1.0),
            min(cv, 5.0),
        ],
        dtype="float32",
    )
    if raw.size != AVAILABILITY_FEATURE_DIM:
        raise RuntimeError(f"availability feature size mismatch: {raw.size} != {AVAILABILITY_FEATURE_DIM}")
    return raw, minute_log_prior


def _targets_for_roster(
    actual_seconds: dict[str, float],
    pids: list[str],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], bool]:
    seconds = np.array([float(actual_seconds.get(str(pid), 0.0)) for pid in pids], dtype="float32")
    played = tuple(float(v > 0.0) for v in seconds.tolist())
    log_seconds = tuple(float(math.log1p(max(float(v), 0.0))) for v in seconds.tolist())
    total = float(seconds.sum())
    if total >= MIN_ROLE_TARGET_SECONDS:
        role = tuple(float(v / total) for v in seconds.tolist())
        valid = True
    else:
        role = tuple(0.0 for _ in pids)
        valid = False
    return played, log_seconds, role, valid


def build_records_availability_v1_1(
    games: pd.DataFrame,
    histories: dict[str, list[TeamGameExposure]],
    *,
    vocab: Vocab,
    team_vocab: TeamVocab,
    status_lookup: dict[tuple[str, str], str],
    status_details: dict[tuple[str, str], PlayerStatusDetail],
    calibration: dict[str, float],
    game_scores: dict[str, tuple[int, int]],
    matchup_rows: dict[str, list[tuple[str, str, str, tuple[float, ...]]]] | None,
    player_game_stats: dict[tuple[str, str], tuple[float, ...]] | None,
    lookback_games: int,
    decay: float,
    tabular_stats,
    player_histories=None,
    player_form_stats=None,
    player_form_lookback: int = DEFAULT_PLAYER_FORM_LOOKBACK,
    player_form_decay: float = DEFAULT_PLAYER_FORM_DECAY,
    game_odds: dict[str, tuple[float, float]] | None = None,
    actual_seconds_index: dict[tuple[str, str], dict[str, float]] | None = None,
) -> list[GameRecordAvailabilityV11]:
    """Build v4.2 records, then attach availability features/targets."""
    base_records = build_records_v42(
        games,
        histories,
        vocab=vocab,
        team_vocab=team_vocab,
        status_lookup=status_lookup,
        calibration=calibration,
        game_scores=game_scores,
        matchup_rows=matchup_rows,
        player_game_stats=player_game_stats,
        lookback_games=lookback_games,
        decay=decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=player_form_lookback,
        player_form_decay=player_form_decay,
        game_odds=game_odds,
    )
    actual_seconds_index = actual_seconds_index or build_team_game_player_seconds_index(histories)
    out: list[GameRecordAvailabilityV11] = []
    for base in base_records:
        gd = pd.Timestamp(base.game_date)
        home_pids = build_full_lineup(
            histories,
            team_id=base.home_team_id,
            game_date=gd,
            lookback_games=lookback_games,
            decay=decay,
        )
        away_pids = build_full_lineup(
            histories,
            team_id=base.away_team_id,
            game_date=gd,
            lookback_games=lookback_games,
            decay=decay,
        )
        if len(home_pids) != len(base.home_player_idx) or len(away_pids) != len(base.away_player_idx):
            # Defensive guard: the v4.2 builder and this augmentation should be
            # using the same lineup logic.  If not, silently skipping would make
            # labels/features misaligned.
            raise RuntimeError(f"lineup reconstruction mismatch for game {base.game_id}")

        home_raw: list[tuple[float, ...]] = []
        away_raw: list[tuple[float, ...]] = []
        home_minute_prior: list[float] = []
        away_minute_prior: list[float] = []
        for pid in home_pids:
            raw, prior_log = build_availability_raw_features(
                histories,
                team_id=base.home_team_id,
                player_id=pid,
                game_date=gd,
                lookback_games=lookback_games,
                decay=decay,
                status_detail=status_details.get((base.game_id, pid), PlayerStatusDetail()),
                calibration=calibration,
            )
            home_raw.append(tuple(float(v) for v in raw.tolist()))
            home_minute_prior.append(float(prior_log))
        for pid in away_pids:
            raw, prior_log = build_availability_raw_features(
                histories,
                team_id=base.away_team_id,
                player_id=pid,
                game_date=gd,
                lookback_games=lookback_games,
                decay=decay,
                status_detail=status_details.get((base.game_id, pid), PlayerStatusDetail()),
                calibration=calibration,
            )
            away_raw.append(tuple(float(v) for v in raw.tolist()))
            away_minute_prior.append(float(prior_log))

        h_actual = actual_seconds_index.get((base.game_id, base.home_team_id), {})
        a_actual = actual_seconds_index.get((base.game_id, base.away_team_id), {})
        h_play, h_log_sec, h_role, h_valid = _targets_for_roster(h_actual, home_pids)
        a_play, a_log_sec, a_role, a_valid = _targets_for_roster(a_actual, away_pids)
        out.append(GameRecordAvailabilityV11(
            base=base,
            home_avail_raw=tuple(home_raw),
            away_avail_raw=tuple(away_raw),
            home_avail_minute_log_prior=tuple(home_minute_prior),
            away_avail_minute_log_prior=tuple(away_minute_prior),
            home_avail_play_actual=h_play,
            away_avail_play_actual=a_play,
            home_avail_log_seconds_actual=h_log_sec,
            away_avail_log_seconds_actual=a_log_sec,
            home_avail_role_share_actual=h_role,
            away_avail_role_share_actual=a_role,
            home_avail_role_valid=h_valid,
            away_avail_role_valid=a_valid,
        ))
    return out


def fit_availability_feature_stats(
    records: list[GameRecordAvailabilityV11],
) -> AvailabilityFeatureStats:
    vecs: list[np.ndarray] = []
    for r in records:
        if r.home_avail_raw:
            vecs.extend(np.asarray(r.home_avail_raw, dtype="float32"))
        if r.away_avail_raw:
            vecs.extend(np.asarray(r.away_avail_raw, dtype="float32"))
    if not vecs:
        return AvailabilityFeatureStats(
            means=np.zeros(AVAILABILITY_FEATURE_DIM, dtype="float32"),
            stds=np.ones(AVAILABILITY_FEATURE_DIM, dtype="float32"),
        )
    M = np.stack(vecs, axis=0).astype("float32")
    means = M.mean(axis=0).astype("float32")
    stds = M.std(axis=0).astype("float32")
    stds = np.where(stds < 1e-6, 1.0, stds).astype("float32")
    return AvailabilityFeatureStats(means=means, stds=stds)


def transform_availability_features(
    raw: tuple[tuple[float, ...], ...],
    stats: AvailabilityFeatureStats,
) -> torch.Tensor:
    if not raw:
        return torch.zeros(0, AVAILABILITY_FEATURE_DIM, dtype=torch.float32)
    arr = np.asarray(raw, dtype="float32")
    z = (arr - stats.means) / stats.stds
    return torch.tensor(z, dtype=torch.float32)


class GameDatasetAvailabilityV11(Dataset):
    def __init__(
        self,
        records: list[GameRecordAvailabilityV11],
        availability_stats: AvailabilityFeatureStats,
    ) -> None:
        self.records = records
        self.availability_stats = availability_stats
        self.base_dataset = GameDatasetV42([r.base for r in records])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.base_dataset[idx])
        r = self.records[idx]
        item.update({
            "home_avail_features": transform_availability_features(r.home_avail_raw, self.availability_stats),
            "away_avail_features": transform_availability_features(r.away_avail_raw, self.availability_stats),
            "home_avail_minute_log_prior": torch.tensor(r.home_avail_minute_log_prior, dtype=torch.float32),
            "away_avail_minute_log_prior": torch.tensor(r.away_avail_minute_log_prior, dtype=torch.float32),
            "home_avail_play_actual": torch.tensor(r.home_avail_play_actual, dtype=torch.float32),
            "away_avail_play_actual": torch.tensor(r.away_avail_play_actual, dtype=torch.float32),
            "home_avail_log_seconds_actual": torch.tensor(r.home_avail_log_seconds_actual, dtype=torch.float32),
            "away_avail_log_seconds_actual": torch.tensor(r.away_avail_log_seconds_actual, dtype=torch.float32),
            "home_avail_role_share_actual": torch.tensor(r.home_avail_role_share_actual, dtype=torch.float32),
            "away_avail_role_share_actual": torch.tensor(r.away_avail_role_share_actual, dtype=torch.float32),
            "home_avail_role_valid": torch.tensor(float(r.home_avail_role_valid), dtype=torch.float32),
            "away_avail_role_valid": torch.tensor(float(r.away_avail_role_valid), dtype=torch.float32),
        })
        return item


_EXTRA_SLOT_KEYS = (
    "home_avail_features", "away_avail_features",
    "home_avail_minute_log_prior", "away_avail_minute_log_prior",
    "home_avail_play_actual", "away_avail_play_actual",
    "home_avail_log_seconds_actual", "away_avail_log_seconds_actual",
    "home_avail_role_share_actual", "away_avail_role_share_actual",
)
_EXTRA_SCALAR_KEYS = ("home_avail_role_valid", "away_avail_role_valid")


def collate_availability_v1_1(batch: list[dict]) -> dict:
    base_batch = [
        {k: v for k, v in item.items() if k not in _EXTRA_SLOT_KEYS and k not in _EXTRA_SCALAR_KEYS}
        for item in batch
    ]
    out = collate_v42(base_batch)
    B = len(batch)
    L_h = out["home_idx"].size(1)
    L_a = out["away_idx"].size(1)
    home_avail_features = torch.zeros(B, L_h, AVAILABILITY_FEATURE_DIM, dtype=torch.float32)
    away_avail_features = torch.zeros(B, L_a, AVAILABILITY_FEATURE_DIM, dtype=torch.float32)
    slot_1d = {
        "home_avail_minute_log_prior": torch.zeros(B, L_h, dtype=torch.float32),
        "away_avail_minute_log_prior": torch.zeros(B, L_a, dtype=torch.float32),
        "home_avail_play_actual": torch.zeros(B, L_h, dtype=torch.float32),
        "away_avail_play_actual": torch.zeros(B, L_a, dtype=torch.float32),
        "home_avail_log_seconds_actual": torch.zeros(B, L_h, dtype=torch.float32),
        "away_avail_log_seconds_actual": torch.zeros(B, L_a, dtype=torch.float32),
        "home_avail_role_share_actual": torch.zeros(B, L_h, dtype=torch.float32),
        "away_avail_role_share_actual": torch.zeros(B, L_a, dtype=torch.float32),
    }
    for i, item in enumerate(batch):
        n_h = item["home_idx"].numel()
        n_a = item["away_idx"].numel()
        home_avail_features[i, :n_h] = item["home_avail_features"]
        away_avail_features[i, :n_a] = item["away_avail_features"]
        for k, dst in slot_1d.items():
            n = n_h if k.startswith("home_") else n_a
            dst[i, :n] = item[k]
    out["home_avail_features"] = home_avail_features
    out["away_avail_features"] = away_avail_features
    out.update(slot_1d)
    out["home_avail_role_valid"] = torch.stack([item["home_avail_role_valid"] for item in batch])
    out["away_avail_role_valid"] = torch.stack([item["away_avail_role_valid"] for item in batch])
    return out


# Vocab logic remains identical to v4.2.
def build_vocab_from_records_availability_v1_1(*args, **kwargs):
    return build_vocab_from_records_v42(*args, **kwargs)


__all__ = [
    # v4.2 constants/utilities re-exported for train/backtest scripts
    "BOX_INDEX", "BOX_TARGETS", "DEFAULT_CALIBRATION_PATH", "DEFAULT_CORE_DB",
    "DEFAULT_FEATURES_DB", "DEFAULT_INJURY_DB", "DEFAULT_LINEUP_DECAY",
    "DEFAULT_LINEUP_LOOKBACK_GAMES", "DEFAULT_MATCHUP_DB",
    "DEFAULT_PLAYER_FORM_DECAY", "DEFAULT_PLAYER_FORM_LOOKBACK", "K_BOX", "K_PAIR",
    "K_PLAYER", "MIN_EXPOSURE_SECONDS", "PAIR_TARGETS", "PLAYER_FORM_DIM",
    "PLAYER_FORM_FEATURE_NAMES", "PLAYER_TARGETS", "REST_DAYS_CLIP", "SEASON_PHASE_DIM",
    "SEASON_PHASE_DAYS", "TABULAR_FEATURE_COLUMNS", "TeamGameExposure", "TeamVocab", "Vocab",
    "build_team_vocab", "fit_player_form_stats", "fit_tabular_stats", "load_game_odds",
    "load_game_player_status", "load_game_scores", "load_games", "load_matchup_rows_v2",
    "load_player_game_stats", "load_player_histories", "load_status_calibration", "load_team_exposures",
    # availability branch
    "AVAILABILITY_FEATURE_NAMES", "AVAILABILITY_FEATURE_DIM", "MIN_ROLE_TARGET_SECONDS",
    "PlayerStatusDetail", "AvailabilityFeatureStats", "GameRecordAvailabilityV11",
    "build_team_game_player_seconds_index", "load_game_player_status_details",
    "build_availability_raw_features", "build_records_availability_v1_1",
    "fit_availability_feature_stats", "transform_availability_features",
    "GameDatasetAvailabilityV11", "collate_availability_v1_1",
    "build_vocab_from_records_availability_v1_1",
]
