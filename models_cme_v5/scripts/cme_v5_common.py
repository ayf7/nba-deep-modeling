"""CME-v5 data layer (v4 + per-minute rotation curve targets).

Self-contained — re-exports everything downstream code needs.  Pulls:
  - man_xfmr_common for base utilities (Vocab, TeamVocab, build_full_lineup,
    loaders, tabular/player-form helpers)
  - cme_v2_common for shared PAIR/PLAYER/BOX constants and the matchup/box
    label dataclasses (PairLabelV2, PlayerBoxLabel)
  - cme_v4_common for play-decision loading, box-minutes-and-pace loading,
    career-year helpers

v5 differs from v4 by:
  1. 48-dim per-minute on-court presence vectors from
     `data/artifacts/cme_v5_features.sqlite` as supervision targets for
     the rotation head.
  2. Regulation-end scores (also from cme_v5_features.sqlite) replace
     final scores in team_box point targets, giving a clean target for
     non-OT game state prediction.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

V5_MAN_SCRIPTS = Path(__file__).resolve().parents[2] / "models_man_xfmr" / "scripts"
sys.path.insert(0, str(V5_MAN_SCRIPTS))
V2_SCRIPTS = Path(__file__).resolve().parents[2] / "models_cme_v2" / "scripts"
sys.path.insert(0, str(V2_SCRIPTS))
V4_SCRIPTS = Path(__file__).resolve().parents[2] / "models_cme_v4" / "scripts"
sys.path.insert(0, str(V4_SCRIPTS))

from man_xfmr_common import (  # noqa: E402
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_LINEUP_DECAY,
    DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB,
    DEFAULT_PLAYER_FORM_DECAY,
    DEFAULT_PLAYER_FORM_LOOKBACK,
    LABEL_COLUMN,
    PLAYER_FORM_DIM,
    PLAYER_FORM_FEATURE_NAMES,
    REST_DAYS_CLIP,
    TABULAR_FEATURE_COLUMNS,
    PlayerFormStats,
    PlayerHistory,
    TabularStats,
    TeamGameExposure,
    TeamVocab,
    Vocab,
    build_full_lineup,
    build_player_form,
    build_team_vocab,
    fit_player_form_stats,
    fit_tabular_stats,
    load_game_odds,
    load_game_scores,
    load_games,
    load_player_histories,
    load_team_exposures,
    transform_player_form,
    transform_tabular_row,
)

from cme_v2_common import (  # noqa: E402
    BOX_INDEX,
    BOX_TARGETS,
    K_BOX,
    K_PAIR,
    K_PLAYER,
    PAIR_TARGETS,
    PLAYER_TARGETS,
    _PAIR_3PA_IDX,
    _PAIR_3PM_IDX,
    _PAIR_AST_IDX,
    _PAIR_BLK_IDX,
    _PAIR_FGA_IDX,
    _PAIR_FGM_IDX,
    _PAIR_PTS_IDX,
    _PAIR_TOV_IDX,
    _PLAYER_DREB_IDX,
    _PLAYER_FTA_IDX,
    _PLAYER_FTM_IDX,
    _PLAYER_OREB_IDX,
    _PLAYER_PF_IDX,
    _PLAYER_STL_IDX,
    PairLabelV2,
    PlayerBoxLabel,
    load_matchup_rows_v2,
    load_player_game_stats,
)

from cme_v4_common import (  # noqa: E402
    BoxMinutesPace,
    MAX_CAREER_YEAR,
    MIN_TEAM_SUPERVISION_MINUTES,
    TEAM_BASE_POSSESSIONS,
    TEAM_TOTAL_MINUTES,
    _compute_player_minutes_and_pace,
    career_year_to_idx,
    load_box_minutes_and_pace,
    load_play_decisions,
    load_player_debut,
    load_player_first_season,
)


# ----------------------------- constants ----------------------------- #

DEFAULT_PLAYER_GAME_STATS_DB = (
    Path(__file__).resolve().parents[2] / "data" / "artifacts" / "player_game_stats.sqlite"
)
DEFAULT_PLAYER_DECISIONS_DB = (
    Path(__file__).resolve().parents[2] / "data" / "artifacts" / "player_decisions.sqlite"
)
DEFAULT_PLAYER_DEBUT_DB = (
    Path(__file__).resolve().parents[2] / "data" / "artifacts" / "player_debut.sqlite"
)
DEFAULT_V5_FEATURES_DB = (
    Path(__file__).resolve().parents[2] / "data" / "artifacts" / "cme_v5_features.sqlite"
)

N_SEGMENTS = 48

_MINUTE_COLS = [f"m{i:02d}" for i in range(N_SEGMENTS)]


# ----------------------------- v5 data loaders ----------------------------- #


def load_minute_presence(
    v5_features_db: Path,
    game_ids: list[str] | None = None,
) -> dict[tuple[str, str], tuple[float, ...]]:
    """Load per-minute presence vectors.

    Returns {(game_id, player_id): tuple of 48 floats}
    """
    out: dict[tuple[str, str], tuple[float, ...]] = {}
    cols = ", ".join(_MINUTE_COLS)
    with sqlite3.connect(v5_features_db) as conn:
        if game_ids:
            chunk = 500
            q = (
                f"SELECT game_id, player_id, {cols} FROM player_minute_presence "
                "WHERE game_id IN ({ph})"
            )
            for start in range(0, len(game_ids), chunk):
                ids = game_ids[start : start + chunk]
                ph = ",".join("?" * len(ids))
                for row in conn.execute(q.format(ph=ph), ids):
                    gid = str(row[0])
                    pid = str(row[1])
                    out[(gid, pid)] = tuple(float(v) for v in row[2:])
        else:
            q = f"SELECT game_id, player_id, {cols} FROM player_minute_presence"
            for row in conn.execute(q):
                gid = str(row[0])
                pid = str(row[1])
                out[(gid, pid)] = tuple(float(v) for v in row[2:])
    return out


def load_regulation_scores(
    v5_features_db: Path,
    game_ids: list[str] | None = None,
) -> dict[str, tuple[int, int, bool]]:
    """Load regulation-end scores.

    Returns {game_id: (home_score_reg, away_score_reg, is_overtime)}
    """
    out: dict[str, tuple[int, int, bool]] = {}
    with sqlite3.connect(v5_features_db) as conn:
        if game_ids:
            chunk = 500
            q = (
                "SELECT game_id, home_score_regulation, away_score_regulation, is_overtime "
                "FROM game_regulation_info WHERE game_id IN ({ph})"
            )
            for start in range(0, len(game_ids), chunk):
                ids = game_ids[start : start + chunk]
                ph = ",".join("?" * len(ids))
                for gid, h_score, a_score, ot in conn.execute(q.format(ph=ph), ids):
                    out[str(gid)] = (int(h_score), int(a_score), bool(ot))
        else:
            for gid, h_score, a_score, ot in conn.execute(
                "SELECT game_id, home_score_regulation, away_score_regulation, is_overtime "
                "FROM game_regulation_info"
            ):
                out[str(gid)] = (int(h_score), int(a_score), bool(ot))
    return out


# ----------------------------- record + builder ----------------------------- #

_ZERO_PRESENCE = tuple(0.0 for _ in range(N_SEGMENTS))


def _clip_rest(value) -> float:
    if value is None or pd.isna(value):
        return REST_DAYS_CLIP
    v = float(value)
    if v < 0:
        return 0.0
    return min(v, REST_DAYS_CLIP)


def _build_box_label(
    matchup_rows_for_player: list[tuple[float, ...]],
    player_only_stats: tuple[float, float, float, float, float, float] | None,
) -> tuple[float, ...]:
    fgm = sum(r[_PAIR_FGM_IDX] for r in matchup_rows_for_player)
    fga = sum(r[_PAIR_FGA_IDX] for r in matchup_rows_for_player)
    p3m = sum(r[_PAIR_3PM_IDX] for r in matchup_rows_for_player)
    p3a = sum(r[_PAIR_3PA_IDX] for r in matchup_rows_for_player)
    ast = sum(r[_PAIR_AST_IDX] for r in matchup_rows_for_player)
    tov = sum(r[_PAIR_TOV_IDX] for r in matchup_rows_for_player)
    blk = sum(r[_PAIR_BLK_IDX] for r in matchup_rows_for_player)
    if player_only_stats is None:
        ftm = fta = oreb = dreb = stl = pf = 0.0
    else:
        ftm, fta, oreb, dreb, stl, pf = player_only_stats
    pts = 2.0 * fgm + p3m + ftm
    return (pts, fgm, fga, p3m, p3a, ftm, fta, ast, tov, blk, oreb, dreb, stl, pf)


@dataclass(frozen=True)
class GameRecordV5:
    """Per-game record. Lineups are pre-filtered to players who played.

    Extends v4 with per-player rotation curve targets (48-dim on-court
    presence vectors) and uses regulation scores for team_box points.
    """
    game_id: str
    game_date: pd.Timestamp
    label: int
    margin: float
    home_team_id: str
    away_team_id: str
    home_team_idx: int
    away_team_idx: int
    # Filtered to players with played=1 in game_player_decisions.
    home_player_idx: tuple[int, ...]
    away_player_idx: tuple[int, ...]
    # Per-(player, game) career-year embedding index (1-indexed, 0 = padding).
    home_career_year_idx: tuple[int, ...]
    away_career_year_idx: tuple[int, ...]
    home_rest: float
    away_rest: float
    pair_labels: tuple[PairLabelV2, ...]
    player_labels: tuple[PlayerBoxLabel, ...]
    team_box_home: tuple[float, ...]
    team_box_away: tuple[float, ...]
    tabular: tuple[float, ...]
    home_player_stats: tuple[tuple[float, ...], ...]
    away_player_stats: tuple[tuple[float, ...], ...]
    # v4: per-player minutes + per-team possessions targets.
    home_minutes_actual: tuple[float, ...]
    away_minutes_actual: tuple[float, ...]
    home_pace_actual: float
    away_pace_actual: float
    home_minutes_valid: bool
    away_minutes_valid: bool
    # v5: per-player 48-dim rotation curve targets.
    home_rotation_target: tuple[tuple[float, ...], ...]
    away_rotation_target: tuple[tuple[float, ...], ...]
    home_rotation_valid: bool
    away_rotation_valid: bool
    home_dec_odds: float = 1.0
    away_dec_odds: float = 1.0
    has_odds: bool = False


def build_records_v5(
    games: pd.DataFrame,
    histories: dict[str, list[TeamGameExposure]],
    *,
    vocab: Vocab,
    team_vocab: TeamVocab,
    play_decisions: dict[str, dict[str, int]],
    game_scores: dict[str, tuple[int, int]],
    matchup_rows: dict[str, list[tuple[str, str, str, tuple[float, ...]]]] | None,
    box_minutes_pace: dict[str, BoxMinutesPace] | None,
    player_game_stats: dict[tuple[str, str], tuple[float, ...]] | None,
    minute_presence: dict[tuple[str, str], tuple[float, ...]] | None,
    regulation_scores: dict[str, tuple[int, int, bool]] | None,
    lookback_games: int,
    decay: float,
    tabular_stats: TabularStats,
    player_first_season: dict[str, int] | None = None,
    player_histories: dict[str, PlayerHistory] | None = None,
    player_form_stats: PlayerFormStats | None = None,
    player_form_lookback: int = DEFAULT_PLAYER_FORM_LOOKBACK,
    player_form_decay: float = DEFAULT_PLAYER_FORM_DECAY,
    game_odds: dict[str, tuple[float, float]] | None = None,
) -> list[GameRecordV5]:
    """Build per-game v5 records.

    Same as v4's build_records_v4 but additionally:
      - Looks up per-player 48-dim presence vectors from minute_presence
      - Uses regulation scores for team_box point targets when available
    """
    raw_tabular = games[list(TABULAR_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    out: list[GameRecordV5] = []
    for row_i, game in enumerate(games.itertuples(index=False)):
        gid = str(game.game_id)
        if gid not in game_scores:
            continue
        decisions = play_decisions.get(gid)
        if not decisions:
            continue
        gd = pd.Timestamp(game.game_date)
        home_team = str(game.home_team_id)
        away_team = str(game.away_team_id)

        home_pids_full = build_full_lineup(
            histories, team_id=home_team, game_date=gd,
            lookback_games=lookback_games, decay=decay,
        )
        away_pids_full = build_full_lineup(
            histories, team_id=away_team, game_date=gd,
            lookback_games=lookback_games, decay=decay,
        )
        home_pids = tuple(p for p in home_pids_full if decisions.get(p, 0) == 1)
        away_pids = tuple(p for p in away_pids_full if decisions.get(p, 0) == 1)
        if not home_pids or not away_pids:
            continue

        home_idx = tuple(vocab.encode(p) for p in home_pids)
        away_idx = tuple(vocab.encode(p) for p in away_pids)
        home_pid_to_slot = {p: i for i, p in enumerate(home_pids)}
        away_pid_to_slot = {p: i for i, p in enumerate(away_pids)}

        # Per-(player, game) career-year embedding index.
        if player_first_season is not None:
            game_season = int(getattr(game, "season"))
            home_career_year_idx = tuple(
                career_year_to_idx(game_season - player_first_season[p])
                if p in player_first_season else 0
                for p in home_pids
            )
            away_career_year_idx = tuple(
                career_year_to_idx(game_season - player_first_season[p])
                if p in player_first_season else 0
                for p in away_pids
            )
        else:
            home_career_year_idx = tuple(0 for _ in home_pids)
            away_career_year_idx = tuple(0 for _ in away_pids)

        pair_labels: list[PairLabelV2] = []
        rows_for_home_off: dict[int, list[tuple[float, ...]]] = {i: [] for i in range(len(home_pids))}
        rows_for_away_off: dict[int, list[tuple[float, ...]]] = {i: [] for i in range(len(away_pids))}

        if matchup_rows is not None:
            for off_team, off_pid, def_pid, vals in matchup_rows.get(gid, []):
                if off_team == home_team:
                    if off_pid in home_pid_to_slot and def_pid in away_pid_to_slot:
                        slot = home_pid_to_slot[off_pid]
                        pair_labels.append(PairLabelV2(
                            side=0, off_slot=slot,
                            def_slot=away_pid_to_slot[def_pid], targets=vals,
                        ))
                        rows_for_home_off[slot].append(vals)
                elif off_team == away_team:
                    if off_pid in away_pid_to_slot and def_pid in home_pid_to_slot:
                        slot = away_pid_to_slot[off_pid]
                        pair_labels.append(PairLabelV2(
                            side=1, off_slot=slot,
                            def_slot=home_pid_to_slot[def_pid], targets=vals,
                        ))
                        rows_for_away_off[slot].append(vals)

        player_labels: list[PlayerBoxLabel] = []
        team_box_home = [0.0] * K_BOX
        team_box_away = [0.0] * K_BOX
        if matchup_rows is not None or player_game_stats is not None:
            for slot, pid in enumerate(home_pids):
                stats = player_game_stats.get((gid, pid)) if player_game_stats else None
                box = _build_box_label(rows_for_home_off.get(slot, []), stats)
                player_labels.append(PlayerBoxLabel(side=0, slot=slot, targets=box))
                for k in range(K_BOX):
                    team_box_home[k] += box[k]
            for slot, pid in enumerate(away_pids):
                stats = player_game_stats.get((gid, pid)) if player_game_stats else None
                box = _build_box_label(rows_for_away_off.get(slot, []), stats)
                player_labels.append(PlayerBoxLabel(side=1, slot=slot, targets=box))
                for k in range(K_BOX):
                    team_box_away[k] += box[k]

        # Team box point targets: prefer regulation scores when available.
        home_score, away_score = game_scores[gid]
        if regulation_scores is not None and gid in regulation_scores:
            reg_home, reg_away, _is_ot = regulation_scores[gid]
            team_box_home[BOX_INDEX["pts"]] = float(reg_home)
            team_box_away[BOX_INDEX["pts"]] = float(reg_away)
        else:
            team_box_home[BOX_INDEX["pts"]] = float(home_score)
            team_box_away[BOX_INDEX["pts"]] = float(away_score)

        tab = transform_tabular_row(raw_tabular[row_i], tabular_stats)

        box_mp = box_minutes_pace.get(gid) if box_minutes_pace else None
        (h_min, a_min, p_home, p_away, valid_h, valid_a) = _compute_player_minutes_and_pace(
            box_mp,
            home_team=home_team, away_team=away_team,
            home_pid_to_slot=home_pid_to_slot,
            away_pid_to_slot=away_pid_to_slot,
            n_home=len(home_pids), n_away=len(away_pids),
        )

        # v5: rotation curve targets.
        if minute_presence is not None:
            home_rot = tuple(
                minute_presence.get((gid, pid), _ZERO_PRESENCE) for pid in home_pids
            )
            away_rot = tuple(
                minute_presence.get((gid, pid), _ZERO_PRESENCE) for pid in away_pids
            )
            # Valid when at least one player on the team has non-zero presence data.
            home_rot_valid = any(
                (gid, pid) in minute_presence for pid in home_pids
            )
            away_rot_valid = any(
                (gid, pid) in minute_presence for pid in away_pids
            )
        else:
            home_rot = tuple(_ZERO_PRESENCE for _ in home_pids)
            away_rot = tuple(_ZERO_PRESENCE for _ in away_pids)
            home_rot_valid = False
            away_rot_valid = False

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
        out.append(GameRecordV5(
            game_id=gid,
            game_date=gd,
            label=int(getattr(game, LABEL_COLUMN)),
            margin=float(home_score - away_score),
            home_team_id=home_team,
            away_team_id=away_team,
            home_team_idx=team_vocab.encode(home_team),
            away_team_idx=team_vocab.encode(away_team),
            home_player_idx=home_idx,
            away_player_idx=away_idx,
            home_career_year_idx=home_career_year_idx,
            away_career_year_idx=away_career_year_idx,
            home_rest=_clip_rest(game.home_rest_days),
            away_rest=_clip_rest(game.away_rest_days),
            pair_labels=tuple(pair_labels),
            player_labels=tuple(player_labels),
            team_box_home=tuple(team_box_home),
            team_box_away=tuple(team_box_away),
            tabular=tuple(tab.tolist()),
            home_player_stats=home_stats,
            away_player_stats=away_stats,
            home_minutes_actual=h_min,
            away_minutes_actual=a_min,
            home_pace_actual=p_home,
            away_pace_actual=p_away,
            home_minutes_valid=valid_h,
            away_minutes_valid=valid_a,
            home_rotation_target=home_rot,
            away_rotation_target=away_rot,
            home_rotation_valid=home_rot_valid,
            away_rotation_valid=away_rot_valid,
            home_dec_odds=h_dec,
            away_dec_odds=a_dec,
            has_odds=has_odds,
        ))
    return out


def build_vocab_from_records_v5(
    games_train: pd.DataFrame,
    histories: dict[str, list[TeamGameExposure]],
    matchup_rows_train: dict[str, list[tuple[str, str, str, tuple[float, ...]]]],
    play_decisions: dict[str, dict[str, int]],
    *,
    lookback_games: int,
    decay: float,
) -> Vocab:
    """Vocab from train rosters (decision-filtered) + train matchup rows."""
    seen: set[str] = set()
    for game in games_train.itertuples(index=False):
        gid = str(game.game_id)
        decisions = play_decisions.get(gid)
        if not decisions:
            continue
        gd = pd.Timestamp(game.game_date)
        for team in (str(game.home_team_id), str(game.away_team_id)):
            for p in build_full_lineup(
                histories, team_id=team, game_date=gd,
                lookback_games=lookback_games, decay=decay,
            ):
                if decisions.get(p, 0) == 1:
                    seen.add(p)
    for rows in matchup_rows_train.values():
        for _, off_pid, def_pid, _vals in rows:
            seen.add(off_pid)
            seen.add(def_pid)
    sorted_pids = sorted(seen)
    return Vocab(player_to_idx={p: i + 1 for i, p in enumerate(sorted_pids)})


# ----------------------------- dataset + collate ----------------------------- #


class GameDatasetV5(Dataset):
    def __init__(self, records: list[GameRecordV5]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        L_h = len(r.home_player_idx)
        L_a = len(r.away_player_idx)
        D = len(r.home_player_stats[0]) if L_h and r.home_player_stats[0] else 0
        if D > 0:
            home_stats = torch.tensor(r.home_player_stats, dtype=torch.float32)
            away_stats = torch.tensor(r.away_player_stats, dtype=torch.float32)
        else:
            home_stats = torch.zeros(L_h, 0, dtype=torch.float32)
            away_stats = torch.zeros(L_a, 0, dtype=torch.float32)

        if r.pair_labels:
            sup_pair_side = torch.tensor([p.side for p in r.pair_labels], dtype=torch.long)
            sup_pair_off = torch.tensor([p.off_slot for p in r.pair_labels], dtype=torch.long)
            sup_pair_def = torch.tensor([p.def_slot for p in r.pair_labels], dtype=torch.long)
            sup_pair_y = torch.tensor([p.targets for p in r.pair_labels], dtype=torch.float32)
        else:
            sup_pair_side = torch.zeros(0, dtype=torch.long)
            sup_pair_off = torch.zeros(0, dtype=torch.long)
            sup_pair_def = torch.zeros(0, dtype=torch.long)
            sup_pair_y = torch.zeros(0, K_PAIR, dtype=torch.float32)

        if r.player_labels:
            sup_pl_side = torch.tensor([p.side for p in r.player_labels], dtype=torch.long)
            sup_pl_slot = torch.tensor([p.slot for p in r.player_labels], dtype=torch.long)
            sup_pl_y = torch.tensor([p.targets for p in r.player_labels], dtype=torch.float32)
        else:
            sup_pl_side = torch.zeros(0, dtype=torch.long)
            sup_pl_slot = torch.zeros(0, dtype=torch.long)
            sup_pl_y = torch.zeros(0, K_BOX, dtype=torch.float32)

        # v5: rotation curve targets.
        home_rotation_target = torch.tensor(r.home_rotation_target, dtype=torch.float32)
        away_rotation_target = torch.tensor(r.away_rotation_target, dtype=torch.float32)

        return {
            "home_idx": torch.tensor(r.home_player_idx, dtype=torch.long),
            "away_idx": torch.tensor(r.away_player_idx, dtype=torch.long),
            "home_career_year_idx": torch.tensor(r.home_career_year_idx, dtype=torch.long),
            "away_career_year_idx": torch.tensor(r.away_career_year_idx, dtype=torch.long),
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
            "team_box_home": torch.tensor(r.team_box_home, dtype=torch.float32),
            "team_box_away": torch.tensor(r.team_box_away, dtype=torch.float32),
            "sup_pair_side": sup_pair_side,
            "sup_pair_off": sup_pair_off,
            "sup_pair_def": sup_pair_def,
            "sup_pair_y": sup_pair_y,
            "sup_pl_side": sup_pl_side,
            "sup_pl_slot": sup_pl_slot,
            "sup_pl_y": sup_pl_y,
            "home_minutes_actual": torch.tensor(r.home_minutes_actual, dtype=torch.float32),
            "away_minutes_actual": torch.tensor(r.away_minutes_actual, dtype=torch.float32),
            "home_pace_actual": torch.tensor(r.home_pace_actual, dtype=torch.float32),
            "away_pace_actual": torch.tensor(r.away_pace_actual, dtype=torch.float32),
            "home_minutes_valid": torch.tensor(float(r.home_minutes_valid), dtype=torch.float32),
            "away_minutes_valid": torch.tensor(float(r.away_minutes_valid), dtype=torch.float32),
            "home_rotation_target": home_rotation_target,
            "away_rotation_target": away_rotation_target,
            "home_rotation_valid": torch.tensor(float(r.home_rotation_valid), dtype=torch.float32),
            "away_rotation_valid": torch.tensor(float(r.away_rotation_valid), dtype=torch.float32),
        }


def collate_v5(batch: list[dict]) -> dict:
    """Pad rosters, concat flat sup_* tensors, pad rotation targets."""
    B = len(batch)
    L_h = max(b["home_idx"].numel() for b in batch)
    L_a = max(b["away_idx"].numel() for b in batch)
    D_stats = batch[0]["home_stats"].size(-1)

    home_idx = torch.zeros(B, L_h, dtype=torch.long)
    home_mask = torch.zeros(B, L_h, dtype=torch.bool)
    away_idx = torch.zeros(B, L_a, dtype=torch.long)
    away_mask = torch.zeros(B, L_a, dtype=torch.bool)
    home_career_year_idx = torch.zeros(B, L_h, dtype=torch.long)
    away_career_year_idx = torch.zeros(B, L_a, dtype=torch.long)
    home_stats = torch.zeros(B, L_h, D_stats, dtype=torch.float32)
    away_stats = torch.zeros(B, L_a, D_stats, dtype=torch.float32)
    home_minutes = torch.zeros(B, L_h, dtype=torch.float32)
    away_minutes = torch.zeros(B, L_a, dtype=torch.float32)
    home_rotation = torch.zeros(B, L_h, N_SEGMENTS, dtype=torch.float32)
    away_rotation = torch.zeros(B, L_a, N_SEGMENTS, dtype=torch.float32)

    for b, item in enumerate(batch):
        n_h = item["home_idx"].numel()
        n_a = item["away_idx"].numel()
        home_idx[b, :n_h] = item["home_idx"]
        home_mask[b, :n_h] = True
        away_idx[b, :n_a] = item["away_idx"]
        away_mask[b, :n_a] = True
        home_career_year_idx[b, :n_h] = item["home_career_year_idx"]
        away_career_year_idx[b, :n_a] = item["away_career_year_idx"]
        if D_stats > 0:
            home_stats[b, :n_h] = item["home_stats"]
            away_stats[b, :n_a] = item["away_stats"]
        home_minutes[b, :n_h] = item["home_minutes_actual"]
        away_minutes[b, :n_a] = item["away_minutes_actual"]
        home_rotation[b, :n_h] = item["home_rotation_target"]
        away_rotation[b, :n_a] = item["away_rotation_target"]

    out = {
        "home_idx": home_idx,
        "home_mask": home_mask,
        "away_idx": away_idx,
        "away_mask": away_mask,
        "home_career_year_idx": home_career_year_idx,
        "away_career_year_idx": away_career_year_idx,
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
        "team_box_home": torch.stack([b["team_box_home"] for b in batch]),
        "team_box_away": torch.stack([b["team_box_away"] for b in batch]),
        "home_minutes_actual": home_minutes,
        "away_minutes_actual": away_minutes,
        "home_pace_actual": torch.stack([b["home_pace_actual"] for b in batch]),
        "away_pace_actual": torch.stack([b["away_pace_actual"] for b in batch]),
        "home_minutes_valid": torch.stack([b["home_minutes_valid"] for b in batch]),
        "away_minutes_valid": torch.stack([b["away_minutes_valid"] for b in batch]),
        "home_rotation_target": home_rotation,
        "away_rotation_target": away_rotation,
        "home_rotation_valid": torch.stack([b["home_rotation_valid"] for b in batch]),
        "away_rotation_valid": torch.stack([b["away_rotation_valid"] for b in batch]),
    }

    pair_game = torch.cat([
        torch.full_like(b["sup_pair_side"], i) for i, b in enumerate(batch)
    ]) if any(b["sup_pair_side"].numel() for b in batch) else torch.zeros(0, dtype=torch.long)
    out["sup_pair_game"] = pair_game
    out["sup_pair_side"] = torch.cat([b["sup_pair_side"] for b in batch])
    out["sup_pair_off"] = torch.cat([b["sup_pair_off"] for b in batch])
    out["sup_pair_def"] = torch.cat([b["sup_pair_def"] for b in batch])
    out["sup_pair_y"] = torch.cat([b["sup_pair_y"] for b in batch], dim=0) \
        if any(b["sup_pair_y"].numel() for b in batch) \
        else torch.zeros(0, K_PAIR, dtype=torch.float32)

    pl_game = torch.cat([
        torch.full_like(b["sup_pl_side"], i) for i, b in enumerate(batch)
    ]) if any(b["sup_pl_side"].numel() for b in batch) else torch.zeros(0, dtype=torch.long)
    out["sup_pl_game"] = pl_game
    out["sup_pl_side"] = torch.cat([b["sup_pl_side"] for b in batch])
    out["sup_pl_slot"] = torch.cat([b["sup_pl_slot"] for b in batch])
    out["sup_pl_y"] = torch.cat([b["sup_pl_y"] for b in batch], dim=0) \
        if any(b["sup_pl_y"].numel() for b in batch) \
        else torch.zeros(0, K_BOX, dtype=torch.float32)

    return out


__all__ = [
    # constants
    "BOX_TARGETS", "K_BOX", "BOX_INDEX",
    "PAIR_TARGETS", "K_PAIR", "PLAYER_TARGETS", "K_PLAYER",
    "TEAM_TOTAL_MINUTES", "TEAM_BASE_POSSESSIONS",
    "MIN_TEAM_SUPERVISION_MINUTES",
    "MAX_CAREER_YEAR", "career_year_to_idx", "load_player_first_season",
    "load_player_debut",
    "N_SEGMENTS",
    # pair indices
    "_PAIR_PTS_IDX", "_PAIR_FGM_IDX", "_PAIR_FGA_IDX", "_PAIR_3PM_IDX",
    "_PAIR_3PA_IDX", "_PAIR_AST_IDX", "_PAIR_TOV_IDX", "_PAIR_BLK_IDX",
    "_PLAYER_FTM_IDX", "_PLAYER_FTA_IDX", "_PLAYER_OREB_IDX",
    "_PLAYER_DREB_IDX", "_PLAYER_STL_IDX", "_PLAYER_PF_IDX",
    # v5 record / dataset
    "GameRecordV5", "build_records_v5", "build_vocab_from_records_v5",
    "GameDatasetV5", "collate_v5",
    # v5 data loaders
    "load_minute_presence", "load_regulation_scores",
    # v4 minutes / pace / decisions (re-exported)
    "BoxMinutesPace", "load_box_minutes_and_pace", "load_play_decisions",
    # default DB paths
    "DEFAULT_CORE_DB", "DEFAULT_FEATURES_DB", "DEFAULT_MATCHUP_DB",
    "DEFAULT_PLAYER_GAME_STATS_DB", "DEFAULT_PLAYER_DECISIONS_DB",
    "DEFAULT_PLAYER_DEBUT_DB", "DEFAULT_V5_FEATURES_DB",
    # defaults
    "DEFAULT_LINEUP_DECAY", "DEFAULT_LINEUP_LOOKBACK_GAMES",
    "DEFAULT_PLAYER_FORM_DECAY", "DEFAULT_PLAYER_FORM_LOOKBACK",
    "PLAYER_FORM_DIM", "PLAYER_FORM_FEATURE_NAMES",
    "TABULAR_FEATURE_COLUMNS", "REST_DAYS_CLIP", "LABEL_COLUMN",
    # base dataclasses (re-exported from v2 / man_xfmr)
    "PairLabelV2", "PlayerBoxLabel",
    "PlayerHistory", "PlayerFormStats", "TabularStats",
    "TeamGameExposure", "TeamVocab", "Vocab",
    # base utilities (re-exported)
    "build_full_lineup", "build_player_form",
    "build_team_vocab", "fit_player_form_stats", "fit_tabular_stats",
    "load_game_odds", "load_game_scores", "load_games",
    "load_matchup_rows_v2", "load_player_game_stats",
    "load_player_histories", "load_team_exposures",
    "transform_player_form", "transform_tabular_row",
]
