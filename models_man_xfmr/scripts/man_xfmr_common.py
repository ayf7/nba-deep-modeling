"""Shared utilities for MAN-Xfmr (v4): cross-attention with Poisson prevalence.

End-to-end attention over the full lineup (no top-K cutoff). Each player carries
a soft availability mask m_i = P(plays | status_i) computed from the injury
report → ground-truth play-rate calibration. Out + Doubtful collapse to ~0.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = Path(__file__).resolve().parents[1]
DATA_ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_FEATURES_DB = DATA_ARTIFACT_ROOT / "features.sqlite"
DEFAULT_CORE_DB = DATA_ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_INJURY_DB = DATA_ARTIFACT_ROOT / "nba_injury_history.sqlite"
DEFAULT_MATCHUP_DB = DATA_ARTIFACT_ROOT / "player_matchup_training.sqlite"
DEFAULT_CALIBRATION_PATH = DATA_ARTIFACT_ROOT / "status_play_calibration.json"
DEFAULT_OUTPUT_ROOT = MODELS_ROOT / "artifacts"

LABEL_COLUMN = "label_home_win"
REST_DAYS_CLIP = 5.0
DEFAULT_LINEUP_LOOKBACK_GAMES = 10
DEFAULT_LINEUP_DECAY = 0.85
DEFAULT_PLAYER_FORM_LOOKBACK = 20
DEFAULT_PLAYER_FORM_DECAY = 0.9
NOT_LISTED_KEY = "NotListed"

# Per-player rolling form feature names (stable order — model relies on this).
PLAYER_FORM_FEATURE_NAMES: tuple[str, ...] = (
    "log_off_poss",            # workload proxy (offensive)
    "pts_per_off_poss",
    "fg_pct_off",
    "3p_pct_off",
    "ast_per_off_poss",
    "tov_per_off_poss",
    "log_def_poss",            # workload proxy (defensive)
    "pts_allowed_per_def_poss",
    "fg_pct_allowed",
    "blocks_per_def_poss",
)
PLAYER_FORM_DIM = len(PLAYER_FORM_FEATURE_NAMES)

CYCLIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "cyc_dow_sin1", "cyc_dow_cos1", "cyc_dow_sin2", "cyc_dow_cos2",
    "cyc_dos_sin1", "cyc_dos_cos1", "cyc_dos_sin2", "cyc_dos_cos2",
    "cyc_dos_sin3", "cyc_dos_cos3",
    "cyc_moy_sin1", "cyc_moy_cos1",
)

TABULAR_FEATURE_COLUMNS: tuple[str, ...] = (
    "home_games_played_before", "away_games_played_before",
    "home_win_pct_before", "away_win_pct_before", "diff_win_pct_before",
    "home_avg_point_diff_before", "away_avg_point_diff_before", "diff_avg_point_diff_before",
    "home_avg_points_for_before", "away_avg_points_for_before", "diff_avg_points_for_before",
    "home_avg_points_against_before", "away_avg_points_against_before", "diff_avg_points_against_before",
    "home_win_pct_last_5", "away_win_pct_last_5", "diff_win_pct_last_5",
    "home_avg_point_diff_last_5", "away_avg_point_diff_last_5", "diff_avg_point_diff_last_5",
    "home_avg_points_for_last_5", "away_avg_points_for_last_5", "diff_avg_points_for_last_5",
    "home_avg_points_against_last_5", "away_avg_points_against_last_5", "diff_avg_points_against_last_5",
    "home_win_pct_last_10", "away_win_pct_last_10", "diff_win_pct_last_10",
    "home_avg_point_diff_last_10", "away_avg_point_diff_last_10", "diff_avg_point_diff_last_10",
    "home_avg_points_for_last_10", "away_avg_points_for_last_10", "diff_avg_points_for_last_10",
    "home_avg_points_against_last_10", "away_avg_points_against_last_10", "diff_avg_points_against_last_10",
    "home_rest_days", "away_rest_days", "diff_rest_days",
) + CYCLIC_FEATURE_COLUMNS


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["game_date"])
    dow = ts.dt.dayofweek.to_numpy()
    for k in (1, 2):
        df[f"cyc_dow_sin{k}"] = np.sin(2.0 * np.pi * k * dow / 7.0)
        df[f"cyc_dow_cos{k}"] = np.cos(2.0 * np.pi * k * dow / 7.0)
    season_start = df.groupby("season")["game_date"].transform("min")
    dos = (ts - pd.to_datetime(season_start)).dt.days.to_numpy()
    for k in (1, 2, 3):
        df[f"cyc_dos_sin{k}"] = np.sin(2.0 * np.pi * k * dos / 200.0)
        df[f"cyc_dos_cos{k}"] = np.cos(2.0 * np.pi * k * dos / 200.0)
    moy = ts.dt.month.to_numpy()
    df["cyc_moy_sin1"] = np.sin(2.0 * np.pi * moy / 12.0)
    df["cyc_moy_cos1"] = np.cos(2.0 * np.pi * moy / 12.0)
    return df


@dataclass(frozen=True)
class TabularStats:
    medians: np.ndarray  # (D,)
    means: np.ndarray    # (D,) computed AFTER median imputation
    stds: np.ndarray     # (D,)


def fit_tabular_stats(train_df: pd.DataFrame) -> TabularStats:
    X = train_df[list(TABULAR_FEATURE_COLUMNS)].to_numpy(dtype="float64").copy()
    medians = np.nanmedian(X, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = medians[nan_cols]
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    return TabularStats(medians=medians, means=means, stds=stds)


def transform_tabular_row(values: np.ndarray, stats: TabularStats) -> np.ndarray:
    """1-D (D,) row: impute NaN with train medians, standardize. Returns float32."""
    row = values.astype("float64").copy()
    nans = np.isnan(row)
    if nans.any():
        row[nans] = stats.medians[nans]
    return ((row - stats.means) / stats.stds).astype("float32")


# ----------------------------- vocab -----------------------------


@dataclass(frozen=True)
class Vocab:
    player_to_idx: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.player_to_idx)

    def encode(self, pid: str) -> int:
        return self.player_to_idx.get(pid, 0) if pid else 0


@dataclass(frozen=True)
class TeamVocab:
    team_to_idx: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.team_to_idx)

    def encode(self, tid: str) -> int:
        return self.team_to_idx.get(tid, 0) if tid else 0


def build_team_vocab(games_train: pd.DataFrame) -> TeamVocab:
    """Team vocab from training games. Index 0 reserved for OOV; teams start at 1."""
    teams: set[str] = set()
    for game in games_train.itertuples(index=False):
        teams.add(str(game.home_team_id))
        teams.add(str(game.away_team_id))
    return TeamVocab(team_to_idx={t: i + 1 for i, t in enumerate(sorted(teams))})


# ----------------------------- exposures + lineups -----------------------------


@dataclass(frozen=True)
class TeamGameExposure:
    game_date: pd.Timestamp
    game_id: str
    player_seconds: dict[str, float]


def load_team_exposures(core_db: Path) -> dict[str, list[TeamGameExposure]]:
    """Per-team chronological list of (game_date, game_id, player_seconds dict)."""
    query = """
        WITH offensive_raw AS (
            SELECT game_id, team_id AS exposure_team_id,
                   player_id AS exposure_player_id,
                   SUM(COALESCE(matchup_seconds, 0.0)) AS exposure_seconds
            FROM player_matchups
            WHERE team_id IS NOT NULL AND player_id IS NOT NULL
              AND matchup_seconds > 0
            GROUP BY game_id, team_id, player_id
        ),
        defensive_raw AS (
            SELECT game_id,
                   CASE WHEN team_id = home_team_id THEN away_team_id
                        WHEN team_id = away_team_id THEN home_team_id
                        ELSE NULL END AS exposure_team_id,
                   defender_player_id AS exposure_player_id,
                   SUM(COALESCE(matchup_seconds, 0.0)) AS exposure_seconds
            FROM player_matchups
            WHERE defender_player_id IS NOT NULL AND matchup_seconds > 0
            GROUP BY game_id,
                     CASE WHEN team_id = home_team_id THEN away_team_id
                          WHEN team_id = away_team_id THEN home_team_id
                          ELSE NULL END,
                     defender_player_id
        ),
        raw AS (
            SELECT * FROM offensive_raw UNION ALL SELECT * FROM defensive_raw
        )
        SELECT g.game_date, r.game_id, r.exposure_team_id AS team_id,
               r.exposure_player_id AS player_id,
               SUM(r.exposure_seconds) AS involvement_seconds
        FROM raw r JOIN games g ON g.game_id = r.game_id
        WHERE r.exposure_team_id IS NOT NULL
          AND r.exposure_player_id IS NOT NULL
          AND g.game_date IS NOT NULL
        GROUP BY g.game_date, r.game_id, r.exposure_team_id, r.exposure_player_id
        HAVING involvement_seconds > 0
        ORDER BY r.exposure_team_id, g.game_date, r.game_id
    """
    with sqlite3.connect(core_db) as conn:
        df = pd.read_sql_query(query, conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    out: dict[str, list[TeamGameExposure]] = {}
    for (team_id, game_date, game_id), group in df.groupby(
        ["team_id", "game_date", "game_id"], sort=True
    ):
        seconds = {
            str(row.player_id): float(row.involvement_seconds)
            for row in group.itertuples(index=False)
        }
        out.setdefault(str(team_id), []).append(
            TeamGameExposure(
                game_date=pd.Timestamp(game_date),
                game_id=str(game_id),
                player_seconds=seconds,
            )
        )
    return out


def build_full_lineup(
    histories: dict[str, list[TeamGameExposure]],
    *,
    team_id: str,
    game_date: pd.Timestamp,
    lookback_games: int,
    decay: float,
) -> list[str]:
    """All players who appeared in this team's last `lookback_games` games,
    sorted by recency-decayed total seconds (descending). No top-K cutoff."""
    prior = [
        item for item in histories.get(str(team_id), []) if item.game_date < game_date
    ][-lookback_games:]
    weighted: dict[str, float] = {}
    for recency_idx, item in enumerate(reversed(prior)):
        w = decay**recency_idx
        for pid, secs in item.player_seconds.items():
            weighted[pid] = weighted.get(pid, 0.0) + w * secs
    return [p for p, _ in sorted(weighted.items(), key=lambda x: x[1], reverse=True) if _ > 0]


# ----------------------------- player rolling form -----------------------------


_PLAYER_HIST_FIELDS = (
    "off_poss", "off_pts", "off_fga", "off_fgm",
    "off_3pa", "off_3pm", "off_ast", "off_tov",
    "def_poss", "def_pts_allowed", "def_fga_allowed", "def_fgm_allowed",
    "def_blocks",
)


@dataclass(frozen=True)
class PlayerHistory:
    """Chronological per-game stat arrays for a single player.

    `dates` are int64 nanosecond timestamps (sorted ascending) so we can use
    bisect for fast lookup in `build_player_form`.
    """
    dates: np.ndarray  # (G,) int64
    raw: np.ndarray    # (G, len(_PLAYER_HIST_FIELDS)) float32


def load_player_histories(core_db: Path) -> dict[str, PlayerHistory]:
    """Per-player chronological per-game offensive + defensive aggregate stats."""
    with sqlite3.connect(core_db) as conn:
        off = pd.read_sql_query(
            """
            SELECT game_id, player_id,
                   SUM(COALESCE(partial_possessions, 0)) AS off_poss,
                   SUM(COALESCE(player_points, 0))       AS off_pts,
                   SUM(COALESCE(matchup_fga, 0))         AS off_fga,
                   SUM(COALESCE(matchup_fgm, 0))         AS off_fgm,
                   SUM(COALESCE(matchup_3pa, 0))         AS off_3pa,
                   SUM(COALESCE(matchup_3pm, 0))         AS off_3pm,
                   SUM(COALESCE(matchup_assists, 0))     AS off_ast,
                   SUM(COALESCE(matchup_turnovers, 0))   AS off_tov
            FROM player_matchups
            WHERE player_id IS NOT NULL
            GROUP BY game_id, player_id
            """,
            conn,
        )
        defn = pd.read_sql_query(
            """
            SELECT game_id, defender_player_id AS player_id,
                   SUM(COALESCE(partial_possessions, 0)) AS def_poss,
                   SUM(COALESCE(player_points, 0))       AS def_pts_allowed,
                   SUM(COALESCE(matchup_fga, 0))         AS def_fga_allowed,
                   SUM(COALESCE(matchup_fgm, 0))         AS def_fgm_allowed,
                   SUM(COALESCE(matchup_blocks, 0))      AS def_blocks
            FROM player_matchups
            WHERE defender_player_id IS NOT NULL
            GROUP BY game_id, defender_player_id
            """,
            conn,
        )
        dates = pd.read_sql_query(
            "SELECT game_id, game_date FROM games WHERE game_date IS NOT NULL", conn
        )
    df = off.merge(defn, on=["game_id", "player_id"], how="outer").fillna(0.0)
    df = df.merge(dates, on="game_id", how="inner")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["player_id"] = df["player_id"].astype(str)
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    out: dict[str, PlayerHistory] = {}
    field_cols = list(_PLAYER_HIST_FIELDS)
    for pid, group in df.groupby("player_id", sort=False):
        dates_arr = group["game_date"].to_numpy().astype("datetime64[ns]").view("int64")
        raw_arr = group[field_cols].to_numpy(dtype="float32")
        out[str(pid)] = PlayerHistory(dates=dates_arr, raw=raw_arr)
    return out


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den > 0 else default


def build_player_form(
    histories: dict[str, PlayerHistory],
    *,
    player_id: str,
    game_date: pd.Timestamp,
    lookback_games: int,
    decay: float,
) -> np.ndarray | None:
    """Return raw (un-standardized) form vector or None if no prior games."""
    h = histories.get(str(player_id))
    if h is None or h.dates.size == 0:
        return None
    cutoff = pd.Timestamp(game_date).value  # nanoseconds since epoch (matches h.dates)
    upper = bisect.bisect_left(h.dates, cutoff)  # strictly-before
    if upper <= 0:
        return None
    lo = max(0, upper - lookback_games)
    rows = h.raw[lo:upper]                         # (k, F) chronological asc
    k = rows.shape[0]
    weights = decay ** np.arange(k - 1, -1, -1, dtype="float32")  # most recent gets weight 1
    weighted = (weights[:, None] * rows).sum(axis=0)               # (F,)

    fields = dict(zip(_PLAYER_HIST_FIELDS, weighted.tolist()))
    op = fields["off_poss"]
    dp = fields["def_poss"]
    feat = np.array([
        np.log1p(op),
        _safe_div(fields["off_pts"], op),
        _safe_div(fields["off_fgm"], fields["off_fga"]),
        _safe_div(fields["off_3pm"], fields["off_3pa"]),
        _safe_div(fields["off_ast"], op),
        _safe_div(fields["off_tov"], op),
        np.log1p(dp),
        _safe_div(fields["def_pts_allowed"], dp),
        _safe_div(fields["def_fgm_allowed"], fields["def_fga_allowed"]),
        _safe_div(fields["def_blocks"], dp),
    ], dtype="float32")
    return feat


@dataclass(frozen=True)
class PlayerFormStats:
    means: np.ndarray  # (D,)
    stds: np.ndarray   # (D,)


def fit_player_form_stats(
    histories: dict[str, PlayerHistory],
    games_train: pd.DataFrame,
    *,
    lookback_games: int,
    decay: float,
) -> PlayerFormStats:
    """Standardization fit by walking each player's own history once and
    sampling the form vector at every game date that falls inside the train
    window (i.e. before train's last game date). Output: feature-wise mean/std.
    """
    if games_train.empty:
        D = PLAYER_FORM_DIM
        return PlayerFormStats(means=np.zeros(D, dtype="float32"),
                               stds=np.ones(D, dtype="float32"))
    train_end = pd.Timestamp(games_train["game_date"].max())
    cutoff_train = train_end.value  # nanoseconds since epoch
    vecs: list[np.ndarray] = []
    for pid, h in histories.items():
        upper_train = bisect.bisect_right(h.dates, cutoff_train)
        # For each j in [1, upper_train), form vector uses rows[0:j].
        for j in range(1, upper_train):
            lo = max(0, j - lookback_games)
            rows = h.raw[lo:j]
            k = rows.shape[0]
            weights = decay ** np.arange(k - 1, -1, -1, dtype="float32")
            w = (weights[:, None] * rows).sum(axis=0)
            op = float(w[_PLAYER_HIST_FIELDS.index("off_poss")])
            dp = float(w[_PLAYER_HIST_FIELDS.index("def_poss")])
            f = w  # alias
            v = np.array([
                np.log1p(op),
                _safe_div(float(f[_PLAYER_HIST_FIELDS.index("off_pts")]), op),
                _safe_div(float(f[_PLAYER_HIST_FIELDS.index("off_fgm")]),
                          float(f[_PLAYER_HIST_FIELDS.index("off_fga")])),
                _safe_div(float(f[_PLAYER_HIST_FIELDS.index("off_3pm")]),
                          float(f[_PLAYER_HIST_FIELDS.index("off_3pa")])),
                _safe_div(float(f[_PLAYER_HIST_FIELDS.index("off_ast")]), op),
                _safe_div(float(f[_PLAYER_HIST_FIELDS.index("off_tov")]), op),
                np.log1p(dp),
                _safe_div(float(f[_PLAYER_HIST_FIELDS.index("def_pts_allowed")]), dp),
                _safe_div(float(f[_PLAYER_HIST_FIELDS.index("def_fgm_allowed")]),
                          float(f[_PLAYER_HIST_FIELDS.index("def_fga_allowed")])),
                _safe_div(float(f[_PLAYER_HIST_FIELDS.index("def_blocks")]), dp),
            ], dtype="float32")
            vecs.append(v)
    if not vecs:
        D = PLAYER_FORM_DIM
        return PlayerFormStats(means=np.zeros(D, dtype="float32"),
                               stds=np.ones(D, dtype="float32"))
    M = np.stack(vecs, axis=0)
    means = M.mean(axis=0).astype("float32")
    stds = M.std(axis=0).astype("float32")
    stds = np.where(stds < 1e-6, 1.0, stds).astype("float32")
    return PlayerFormStats(means=means, stds=stds)


def transform_player_form(
    vec: np.ndarray | None, stats: PlayerFormStats
) -> np.ndarray:
    if vec is None:
        return np.zeros(PLAYER_FORM_DIM, dtype="float32")
    return ((vec - stats.means) / stats.stds).astype("float32")


# ----------------------------- games + scores -----------------------------


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
    df = add_cyclic_features(df)
    return df


def load_game_scores(core_db: Path, game_ids: list[str]) -> dict[str, tuple[int, int]]:
    if not game_ids:
        return {}
    out: dict[str, tuple[int, int]] = {}
    chunk = 500
    with sqlite3.connect(core_db) as conn:
        for start in range(0, len(game_ids), chunk):
            ids = game_ids[start : start + chunk]
            placeholders = ",".join("?" * len(ids))
            df = pd.read_sql_query(
                f"SELECT game_id, home_score, away_score FROM games "
                f"WHERE game_id IN ({placeholders})",
                conn,
                params=ids,
            )
            for row in df.itertuples(index=False):
                if pd.isna(row.home_score) or pd.isna(row.away_score):
                    continue
                out[str(row.game_id)] = (int(row.home_score), int(row.away_score))
    return out


def load_game_odds(
    core_db: Path, game_ids: list[str]
) -> dict[str, tuple[float, float]]:
    """{game_id: (home_dec_odds, away_dec_odds)} for games with moneyline odds."""
    if not game_ids:
        return {}
    out: dict[str, tuple[float, float]] = {}
    chunk = 500
    with sqlite3.connect(core_db) as conn:
        for start in range(0, len(game_ids), chunk):
            ids = game_ids[start : start + chunk]
            placeholders = ",".join("?" * len(ids))
            df = pd.read_sql_query(
                f"SELECT game_id, home_avg_decimal_odds, away_avg_decimal_odds "
                f"FROM game_moneyline_odds "
                f"WHERE has_moneyline_odds=1 AND game_id IN ({placeholders})",
                conn,
                params=ids,
            )
            for row in df.itertuples(index=False):
                if pd.isna(row.home_avg_decimal_odds) or pd.isna(row.away_avg_decimal_odds):
                    continue
                out[str(row.game_id)] = (
                    float(row.home_avg_decimal_odds),
                    float(row.away_avg_decimal_odds),
                )
    return out


# ----------------------------- availability + calibration -----------------------------


def load_status_calibration(path: Path) -> dict[str, float]:
    with open(path) as f:
        table = json.load(f)
    return {str(k): float(v) for k, v in table.items()}


def load_game_player_status(
    injury_db: Path, game_ids: Iterable[str]
) -> dict[tuple[str, str], str]:
    """{(game_id, player_id): status} from `game_player_availability`."""
    ids = list({str(g) for g in game_ids})
    if not ids:
        return {}
    out: dict[tuple[str, str], str] = {}
    chunk = 500
    with sqlite3.connect(injury_db) as conn:
        for start in range(0, len(ids), chunk):
            sub = ids[start : start + chunk]
            placeholders = ",".join("?" * len(sub))
            df = pd.read_sql_query(
                f"SELECT game_id, player_id, status "
                f"FROM game_player_availability WHERE game_id IN ({placeholders})",
                conn,
                params=sub,
            )
            for row in df.itertuples(index=False):
                status = str(row.status) if pd.notna(row.status) and row.status else ""
                out[(str(row.game_id), str(row.player_id))] = status
    return out


def play_prob_for(status: str, calibration: dict[str, float]) -> float:
    if not status:
        return calibration.get(NOT_LISTED_KEY, 1.0)
    return calibration.get(status, calibration.get(NOT_LISTED_KEY, 1.0))


# ----------------------------- matchup supervision -----------------------------


def load_matchup_rows(
    matchup_db: Path, game_ids: list[str]
) -> dict[str, list[tuple[str, str, str, float]]]:
    """{game_id: [(offensive_team_id, off_pid, def_pid, exposure_possessions)]}."""
    if not game_ids:
        return {}
    out: dict[str, list[tuple[str, str, str, float]]] = {}
    chunk = 500
    with sqlite3.connect(matchup_db) as conn:
        for start in range(0, len(game_ids), chunk):
            ids = game_ids[start : start + chunk]
            placeholders = ",".join("?" * len(ids))
            df = pd.read_sql_query(
                f"SELECT game_id, offensive_team_id, offensive_player_id, "
                f"defender_player_id, exposure_possessions "
                f"FROM matchup_training_rows "
                f"WHERE game_id IN ({placeholders}) AND exposure_possessions > 0",
                conn,
                params=ids,
            )
            for row in df.itertuples(index=False):
                out.setdefault(str(row.game_id), []).append(
                    (
                        str(row.offensive_team_id),
                        str(row.offensive_player_id),
                        str(row.defender_player_id),
                        float(row.exposure_possessions),
                    )
                )
    return out


# ----------------------------- records + dataset -----------------------------


@dataclass(frozen=True)
class MatchupSupervision:
    side: int  # 0 = home_off × away_def, 1 = away_off × home_def
    off_slot: int
    def_slot: int
    exposure: float


@dataclass(frozen=True)
class XfmrGameRecord:
    game_id: str
    game_date: pd.Timestamp
    label: int
    margin: float
    home_team_id: str
    away_team_id: str
    home_team_idx: int
    away_team_idx: int
    home_player_idx: tuple[int, ...]
    home_play_prob: tuple[float, ...]
    away_player_idx: tuple[int, ...]
    away_play_prob: tuple[float, ...]
    home_rest: float
    away_rest: float
    supervisions: tuple[MatchupSupervision, ...]
    tabular: tuple[float, ...]
    # Standardized per-player rolling form. Shape (L, D); D = PLAYER_FORM_DIM
    # (or 0 if disabled). Stored as nested tuples of float for hashable record.
    home_player_stats: tuple[tuple[float, ...], ...]
    away_player_stats: tuple[tuple[float, ...], ...]
    # Decimal odds + has-odds mask. Sentinel 1.0 when missing; loss masks them.
    home_dec_odds: float = 1.0
    away_dec_odds: float = 1.0
    has_odds: bool = False


def _clip_rest(value) -> float:
    if value is None or pd.isna(value):
        return REST_DAYS_CLIP
    v = float(value)
    if v < 0:
        return 0.0
    return min(v, REST_DAYS_CLIP)


def build_records(
    games: pd.DataFrame,
    histories: dict[str, list[TeamGameExposure]],
    *,
    vocab: Vocab,
    team_vocab: TeamVocab,
    status_lookup: dict[tuple[str, str], str],
    calibration: dict[str, float],
    game_scores: dict[str, tuple[int, int]],
    matchup_rows: dict[str, list[tuple[str, str, str, float]]] | None,
    lookback_games: int,
    decay: float,
    tabular_stats: TabularStats,
    player_histories: dict[str, PlayerHistory] | None = None,
    player_form_stats: PlayerFormStats | None = None,
    player_form_lookback: int = DEFAULT_PLAYER_FORM_LOOKBACK,
    player_form_decay: float = DEFAULT_PLAYER_FORM_DECAY,
    game_odds: dict[str, tuple[float, float]] | None = None,
) -> list[XfmrGameRecord]:
    """Build per-game records. Pass `matchup_rows` for training games (Poisson
    supervision); pass None for val/test (skips supervision rows)."""
    raw_tabular = games[list(TABULAR_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    out: list[XfmrGameRecord] = []
    for row_i, game in enumerate(games.itertuples(index=False)):
        gid = str(game.game_id)
        if gid not in game_scores:
            continue
        gd = pd.Timestamp(game.game_date)
        home_team = str(game.home_team_id)
        away_team = str(game.away_team_id)

        home_pids = build_full_lineup(
            histories, team_id=home_team, game_date=gd,
            lookback_games=lookback_games, decay=decay,
        )
        away_pids = build_full_lineup(
            histories, team_id=away_team, game_date=gd,
            lookback_games=lookback_games, decay=decay,
        )
        if not home_pids or not away_pids:
            continue

        home_idx = tuple(vocab.encode(p) for p in home_pids)
        away_idx = tuple(vocab.encode(p) for p in away_pids)
        home_prob = tuple(
            play_prob_for(status_lookup.get((gid, p), ""), calibration) for p in home_pids
        )
        away_prob = tuple(
            play_prob_for(status_lookup.get((gid, p), ""), calibration) for p in away_pids
        )

        home_pid_to_slot = {p: i for i, p in enumerate(home_pids)}
        away_pid_to_slot = {p: i for i, p in enumerate(away_pids)}

        sups: list[MatchupSupervision] = []
        if matchup_rows is not None:
            for off_team, off_pid, def_pid, exposure in matchup_rows.get(gid, []):
                if off_team == home_team:
                    if off_pid in home_pid_to_slot and def_pid in away_pid_to_slot:
                        sups.append(MatchupSupervision(
                            side=0,
                            off_slot=home_pid_to_slot[off_pid],
                            def_slot=away_pid_to_slot[def_pid],
                            exposure=exposure,
                        ))
                elif off_team == away_team:
                    if off_pid in away_pid_to_slot and def_pid in home_pid_to_slot:
                        sups.append(MatchupSupervision(
                            side=1,
                            off_slot=away_pid_to_slot[off_pid],
                            def_slot=home_pid_to_slot[def_pid],
                            exposure=exposure,
                        ))

        home_score, away_score = game_scores[gid]
        tab = transform_tabular_row(raw_tabular[row_i], tabular_stats)

        if player_histories is not None and player_form_stats is not None:
            home_stats = tuple(
                tuple(transform_player_form(
                    build_player_form(
                        player_histories, player_id=p, game_date=gd,
                        lookback_games=player_form_lookback, decay=player_form_decay,
                    ),
                    player_form_stats,
                ).tolist())
                for p in home_pids
            )
            away_stats = tuple(
                tuple(transform_player_form(
                    build_player_form(
                        player_histories, player_id=p, game_date=gd,
                        lookback_games=player_form_lookback, decay=player_form_decay,
                    ),
                    player_form_stats,
                ).tolist())
                for p in away_pids
            )
        else:
            home_stats = tuple(() for _ in home_pids)
            away_stats = tuple(() for _ in away_pids)

        h_dec, a_dec = (game_odds.get(gid) if game_odds else None) or (1.0, 1.0)
        has_odds = bool(game_odds and gid in game_odds)
        out.append(XfmrGameRecord(
            game_id=gid,
            game_date=gd,
            label=int(getattr(game, LABEL_COLUMN)),
            margin=float(home_score - away_score),
            home_team_id=home_team,
            away_team_id=away_team,
            home_team_idx=team_vocab.encode(home_team),
            away_team_idx=team_vocab.encode(away_team),
            home_player_idx=home_idx,
            home_play_prob=home_prob,
            away_player_idx=away_idx,
            away_play_prob=away_prob,
            home_rest=_clip_rest(game.home_rest_days),
            away_rest=_clip_rest(game.away_rest_days),
            supervisions=tuple(sups),
            tabular=tuple(tab.tolist()),
            home_player_stats=home_stats,
            away_player_stats=away_stats,
            home_dec_odds=h_dec,
            away_dec_odds=a_dec,
            has_odds=has_odds,
        ))
    return out


def build_vocab_from_records(
    games_train: pd.DataFrame,
    histories: dict[str, list[TeamGameExposure]],
    matchup_rows_train: dict[str, list[tuple[str, str, str, float]]],
    *,
    lookback_games: int,
    decay: float,
) -> Vocab:
    """Vocab from train rosters + train matchup rows. 0 reserved for OOV/padding."""
    seen: set[str] = set()
    for game in games_train.itertuples(index=False):
        gd = pd.Timestamp(game.game_date)
        for team in (str(game.home_team_id), str(game.away_team_id)):
            for p in build_full_lineup(
                histories, team_id=team, game_date=gd,
                lookback_games=lookback_games, decay=decay,
            ):
                seen.add(p)
    for rows in matchup_rows_train.values():
        for _, off_pid, def_pid, _ in rows:
            seen.add(off_pid)
            seen.add(def_pid)
    sorted_pids = sorted(seen)
    return Vocab(player_to_idx={p: i + 1 for i, p in enumerate(sorted_pids)})


class XfmrGameDataset(Dataset):
    def __init__(self, records: list[XfmrGameRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        # Per-player stat tensors. Shape (L, D); D=0 if disabled.
        L_h = len(r.home_player_idx)
        L_a = len(r.away_player_idx)
        D = len(r.home_player_stats[0]) if L_h and r.home_player_stats[0] else 0
        if D > 0:
            home_stats = torch.tensor(r.home_player_stats, dtype=torch.float32)
            away_stats = torch.tensor(r.away_player_stats, dtype=torch.float32)
        else:
            home_stats = torch.zeros(L_h, 0, dtype=torch.float32)
            away_stats = torch.zeros(L_a, 0, dtype=torch.float32)
        return {
            "home_idx": torch.tensor(r.home_player_idx, dtype=torch.long),
            "home_prob": torch.tensor(r.home_play_prob, dtype=torch.float32),
            "away_idx": torch.tensor(r.away_player_idx, dtype=torch.long),
            "away_prob": torch.tensor(r.away_play_prob, dtype=torch.float32),
            "home_stats": home_stats,
            "away_stats": away_stats,
            "home_team_idx": torch.tensor(r.home_team_idx, dtype=torch.long),
            "away_team_idx": torch.tensor(r.away_team_idx, dtype=torch.long),
            "home_rest": torch.tensor(r.home_rest / REST_DAYS_CLIP, dtype=torch.float32),
            "away_rest": torch.tensor(r.away_rest / REST_DAYS_CLIP, dtype=torch.float32),
            "label": torch.tensor(r.label, dtype=torch.float32),
            "margin": torch.tensor(r.margin, dtype=torch.float32),
            "tabular": torch.tensor(r.tabular, dtype=torch.float32),
            "home_dec_odds": torch.tensor(r.home_dec_odds, dtype=torch.float32),
            "away_dec_odds": torch.tensor(r.away_dec_odds, dtype=torch.float32),
            "has_odds": torch.tensor(float(r.has_odds), dtype=torch.float32),
            "sup_side": torch.tensor([s.side for s in r.supervisions], dtype=torch.long),
            "sup_off": torch.tensor([s.off_slot for s in r.supervisions], dtype=torch.long),
            "sup_def": torch.tensor([s.def_slot for s in r.supervisions], dtype=torch.long),
            "sup_exp": torch.tensor([s.exposure for s in r.supervisions], dtype=torch.float32),
        }


def collate_xfmr(batch: list[dict]) -> dict:
    """Pad variable-length rosters to batch max. Mask = (idx > 0)."""
    B = len(batch)
    L_h = max(b["home_idx"].numel() for b in batch)
    L_a = max(b["away_idx"].numel() for b in batch)
    D_stats = batch[0]["home_stats"].size(-1) if "home_stats" in batch[0] else 0

    home_idx = torch.zeros(B, L_h, dtype=torch.long)
    home_prob = torch.zeros(B, L_h, dtype=torch.float32)
    home_mask = torch.zeros(B, L_h, dtype=torch.bool)
    away_idx = torch.zeros(B, L_a, dtype=torch.long)
    away_prob = torch.zeros(B, L_a, dtype=torch.float32)
    away_mask = torch.zeros(B, L_a, dtype=torch.bool)
    home_stats = torch.zeros(B, L_h, D_stats, dtype=torch.float32)
    away_stats = torch.zeros(B, L_a, D_stats, dtype=torch.float32)

    for b, item in enumerate(batch):
        n_h = item["home_idx"].numel()
        n_a = item["away_idx"].numel()
        home_idx[b, :n_h] = item["home_idx"]
        home_prob[b, :n_h] = item["home_prob"]
        home_mask[b, :n_h] = True
        away_idx[b, :n_a] = item["away_idx"]
        away_prob[b, :n_a] = item["away_prob"]
        away_mask[b, :n_a] = True
        if D_stats > 0:
            home_stats[b, :n_h] = item["home_stats"]
            away_stats[b, :n_a] = item["away_stats"]

    out = {
        "home_idx": home_idx,
        "home_prob": home_prob,
        "home_mask": home_mask,
        "away_idx": away_idx,
        "away_prob": away_prob,
        "away_mask": away_mask,
        "home_stats": home_stats,
        "away_stats": away_stats,
        "home_team_idx": torch.stack([b["home_team_idx"] for b in batch]),
        "away_team_idx": torch.stack([b["away_team_idx"] for b in batch]),
        "home_rest": torch.stack([b["home_rest"] for b in batch]),
        "away_rest": torch.stack([b["away_rest"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "margin": torch.stack([b["margin"] for b in batch]),
        "tabular": torch.stack([b["tabular"] for b in batch]),
        "home_dec_odds": torch.stack([b["home_dec_odds"] for b in batch]),
        "away_dec_odds": torch.stack([b["away_dec_odds"] for b in batch]),
        "has_odds": torch.stack([b["has_odds"] for b in batch]),
    }

    sup_game = torch.cat([
        torch.full_like(b["sup_side"], i) for i, b in enumerate(batch)
    ]) if any(b["sup_side"].numel() for b in batch) else torch.zeros(0, dtype=torch.long)
    out["sup_game"] = sup_game
    out["sup_side"] = torch.cat([b["sup_side"] for b in batch])
    out["sup_off"] = torch.cat([b["sup_off"] for b in batch])
    out["sup_def"] = torch.cat([b["sup_def"] for b in batch])
    out["sup_exp"] = torch.cat([b["sup_exp"] for b in batch])
    return out
