"""CME-v4 data layer (bottom-up: per-player minutes + pace + pair stats).

Self-contained — does NOT import from cme_v3_common. Pulls only:
  - man_xfmr_common for base utilities (Vocab, TeamVocab, build_full_lineup,
    loaders, tabular/player-form helpers)
  - cme_v2_common for shared PAIR/PLAYER/BOX constants and the matchup/box
    label dataclasses (PairLabelV2, PlayerBoxLabel)

v4 differs from v3 by:
  1. Hard play/no-play decisions from data/artifacts/player_decisions.sqlite
     replace the soft injury-report calibrated probability. Players who did
     not play are DROPPED from the input sequence entirely — no positional
     embedding, so the input is a set and removing a slot doesn't disturb
     others' representations. Attention runs over actives only; the padding
     mask IS the actives mask.
  2. Per-player minutes target supervises observed zeros directly, fixing
     alpha-leak (finding #4).
  3. Per-team possessions target supervises the pace head.

Both v4 targets come from `data/artifacts/player_game_stats.sqlite` — exact
PBP-derived per-game per-player stats. Per-team poss uses the NBA standard:
`FGA + 0.44·FTA - OREB + TOV`.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

V5_SCRIPTS = Path(__file__).resolve().parents[1] / "models_man_xfmr" / "scripts"
sys.path.insert(0, str(V5_SCRIPTS))
V2_SCRIPTS = Path(__file__).resolve().parents[1] / "models_cme_v2" / "scripts"
sys.path.insert(0, str(V2_SCRIPTS))

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


DEFAULT_PLAYER_GAME_STATS_DB = (
    Path(__file__).resolve().parents[2] / "data" / "artifacts" / "player_game_stats.sqlite"
)
DEFAULT_PLAYER_DECISIONS_DB = (
    Path(__file__).resolve().parents[2] / "data" / "artifacts" / "player_decisions.sqlite"
)
DEFAULT_PLAYER_DEBUT_DB = (
    Path(__file__).resolve().parents[2] / "data" / "artifacts" / "player_debut.sqlite"
)

# 5 players × 48 min = 240 minutes per team per regulation. Used as the scale
# in the minutes loss (and as a sanity threshold for supervision validity).
TEAM_TOTAL_MINUTES = 240.0
TEAM_BASE_POSSESSIONS = 100.0
MIN_TEAM_SUPERVISION_MINUTES = 200.0

# Max number of career-year buckets the time-position embedding supports.
# career_year = game_season - nba_debut_season (clamped to [0, MAX_CAREER_YEAR-1]).
# With true NBA debut years (via player_debut.sqlite), LeBron is at year ~22
# today; sized to 25 to cover that with headroom. Embedding table has size
# MAX_CAREER_YEAR + 1 with idx 0 reserved for padding; real career years
# 0..MAX_CAREER_YEAR-1 map to idx 1..MAX_CAREER_YEAR.
MAX_CAREER_YEAR = 25


def career_year_to_idx(year: int) -> int:
    """0-indexed career year → embedding index. 0 reserved for padding."""
    if year < 0:
        year = 0
    if year > MAX_CAREER_YEAR - 1:
        year = MAX_CAREER_YEAR - 1
    return year + 1


def load_player_first_season(player_game_stats_db: Path) -> dict[str, int]:
    """{player_id: min(season)} across all rows in player_game_stats.

    DB-relative "first season" — equal to the player's NBA debut only when our
    data window covers their debut. Prefer `load_player_debut` for new runs;
    this is kept as the fallback path and for back-compat with older configs.
    """
    out: dict[str, int] = {}
    with sqlite3.connect(player_game_stats_db) as conn:
        for pid, first in conn.execute(
            "SELECT player_id, MIN(season) FROM player_game_stats GROUP BY player_id"
        ):
            out[str(pid)] = int(first)
    return out


def load_player_debut(
    player_debut_db: Path | None,
    fallback_player_game_stats_db: Path,
) -> dict[str, int]:
    """{player_id: true NBA debut season} via `player_debut.sqlite`, with
    `MIN(season) FROM player_game_stats` as a fallback for pids missing from
    the debut table (e.g., very recent two-way contracts / G-League call-ups
    not yet pulled).

    The returned map is invariant to which subset of seasons we train on, so
    for a fixed real (player, game) the resulting `career_year` is stable
    across data-window choices.
    """
    out: dict[str, int] = {}
    if player_debut_db is not None and Path(player_debut_db).exists():
        with sqlite3.connect(player_debut_db) as conn:
            for pid, from_year in conn.execute(
                "SELECT player_id, from_year FROM player_debut WHERE from_year IS NOT NULL"
            ):
                out[str(pid)] = int(from_year)
    fallback = load_player_first_season(fallback_player_game_stats_db)
    for pid, first in fallback.items():
        out.setdefault(pid, first)
    return out


# ----------------------------- decisions ----------------------------- #


def load_play_decisions(
    decisions_db: Path,
    game_ids: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """{game_id: {player_id: played ∈ {0,1}}} from hoopR's player_box.

    Built upstream by data/scripts/build_player_decisions.py. `played=0`
    covers both DNP-CD and Inactive — anyone whose ESPN row had
    did_not_play=1. Used to filter the lineup so only players who actually
    appeared get tokens.
    """
    out: dict[str, dict[str, int]] = {}
    with sqlite3.connect(decisions_db) as conn:
        if game_ids:
            chunk = 500
            q = (
                "SELECT game_id, player_id, played FROM game_player_decisions "
                "WHERE game_id IN ({ph})"
            )
            for start in range(0, len(game_ids), chunk):
                ids = game_ids[start:start + chunk]
                ph = ",".join("?" * len(ids))
                for gid, pid, played in conn.execute(q.format(ph=ph), ids):
                    out.setdefault(str(gid), {})[str(pid)] = int(played)
        else:
            for gid, pid, played in conn.execute(
                "SELECT game_id, player_id, played FROM game_player_decisions"
            ):
                out.setdefault(str(gid), {})[str(pid)] = int(played)
    return out


# ----------------------------- minutes / pace ----------------------------- #


@dataclass(frozen=True)
class BoxMinutesPace:
    """Per-game per-player minutes + per-team possessions, from PBP box scores."""
    player_seconds: dict[tuple[str, str], float]  # (team_id, player_id) -> seconds
    team_possessions: dict[str, float]            # team_id -> possessions


def load_box_minutes_and_pace(
    player_game_stats_db: Path, game_ids: list[str]
) -> dict[str, BoxMinutesPace]:
    """Read exact box-score minutes + team possessions from `player_game_stats.sqlite`.

    Team possessions use the standard NBA formula
    `FGA + 0.44·FTA - OREB + TOV` summed across the team's players.
    """
    if not game_ids:
        return {}

    per_gid_secs: dict[str, dict[tuple[str, str], float]] = {}
    per_gid_team_acc: dict[str, dict[str, list[float]]] = {}
    chunk = 500
    query_template = """
        SELECT game_id, team_id, player_id, seconds_played, fga, fta, oreb, tov
        FROM player_game_stats
        WHERE game_id IN ({placeholders})
    """
    with sqlite3.connect(player_game_stats_db) as conn:
        for start in range(0, len(game_ids), chunk):
            ids = game_ids[start : start + chunk]
            placeholders = ",".join("?" * len(ids))
            for row in conn.execute(query_template.format(placeholders=placeholders), ids):
                gid, team, pid, secs, fga, fta, oreb, tov = row
                gid, team, pid = str(gid), str(team), str(pid)
                per_gid_secs.setdefault(gid, {})[(team, pid)] = float(secs or 0.0)
                acc = per_gid_team_acc.setdefault(gid, {}).setdefault(team, [0.0, 0.0, 0.0, 0.0])
                acc[0] += float(fga or 0)
                acc[1] += float(fta or 0)
                acc[2] += float(oreb or 0)
                acc[3] += float(tov or 0)

    out: dict[str, BoxMinutesPace] = {}
    for gid, team_acc in per_gid_team_acc.items():
        team_poss = {
            t: max(0.0, fga + 0.44 * fta - oreb + tov)
            for t, (fga, fta, oreb, tov) in team_acc.items()
        }
        out[gid] = BoxMinutesPace(
            player_seconds=per_gid_secs.get(gid, {}),
            team_possessions=team_poss,
        )
    return out


def _compute_player_minutes_and_pace(
    box: BoxMinutesPace | None,
    *,
    home_team: str,
    away_team: str,
    home_pid_to_slot: dict[str, int],
    away_pid_to_slot: dict[str, int],
    n_home: int,
    n_away: int,
) -> tuple[
    tuple[float, ...], tuple[float, ...],
    float, float,
    bool, bool,
]:
    home_min = [0.0] * n_home
    away_min = [0.0] * n_away
    if box is not None:
        for (team, pid), sec in box.player_seconds.items():
            if team == home_team:
                slot = home_pid_to_slot.get(pid)
                if slot is not None:
                    home_min[slot] = sec / 60.0
            elif team == away_team:
                slot = away_pid_to_slot.get(pid)
                if slot is not None:
                    away_min[slot] = sec / 60.0
    pace_home = (box.team_possessions.get(home_team, TEAM_BASE_POSSESSIONS)
                 if box is not None else TEAM_BASE_POSSESSIONS)
    pace_away = (box.team_possessions.get(away_team, TEAM_BASE_POSSESSIONS)
                 if box is not None else TEAM_BASE_POSSESSIONS)
    valid_h = sum(home_min) >= MIN_TEAM_SUPERVISION_MINUTES
    valid_a = sum(away_min) >= MIN_TEAM_SUPERVISION_MINUTES
    return tuple(home_min), tuple(away_min), pace_home, pace_away, valid_h, valid_a


# ----------------------------- record + builder ----------------------------- #


@dataclass(frozen=True)
class GameRecordV4:
    """Per-game record. Lineups are pre-filtered to players who played.

    No `play_prob` fields — the injury-report soft probability has been
    retired in v4. The DNP signal enters via lineup filtering only.
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


def build_records_v4(
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
    lookback_games: int,
    decay: float,
    tabular_stats: TabularStats,
    player_first_season: dict[str, int] | None = None,
    player_histories: dict[str, PlayerHistory] | None = None,
    player_form_stats: PlayerFormStats | None = None,
    player_form_lookback: int = DEFAULT_PLAYER_FORM_LOOKBACK,
    player_form_decay: float = DEFAULT_PLAYER_FORM_DECAY,
    game_odds: dict[str, tuple[float, float]] | None = None,
) -> list[GameRecordV4]:
    """Build per-game v4 records.

    Lineup for each team = `build_full_lineup(...)` (top-N by recent exposure)
    filtered to player_ids with played=1 in `play_decisions[game_id]`. Games
    missing from `play_decisions` are skipped. Players in the full lineup
    whose decision is missing are treated as DNP (no row in player_box →
    likely not on the active roster) and dropped.
    """
    raw_tabular = games[list(TABULAR_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    out: list[GameRecordV4] = []
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

        # Per-(player, game) career-year embedding index. Falls back to
        # idx=0 (padding bucket) if first_season is unknown for the player —
        # the model treats this as "no time-position info" via padding_idx.
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

        home_score, away_score = game_scores[gid]
        team_box_home[BOX_INDEX["pts"]] = float(home_score)
        team_box_away[BOX_INDEX["pts"]] = float(away_score)
        tab = transform_tabular_row(raw_tabular[row_i], tabular_stats)

        box = box_minutes_pace.get(gid) if box_minutes_pace else None
        (h_min, a_min, p_home, p_away, valid_h, valid_a) = _compute_player_minutes_and_pace(
            box,
            home_team=home_team, away_team=away_team,
            home_pid_to_slot=home_pid_to_slot,
            away_pid_to_slot=away_pid_to_slot,
            n_home=len(home_pids), n_away=len(away_pids),
        )

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
        out.append(GameRecordV4(
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
            home_dec_odds=h_dec,
            away_dec_odds=a_dec,
            has_odds=has_odds,
        ))
    return out


def build_vocab_from_records_v4(
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


class GameDatasetV4(Dataset):
    def __init__(self, records: list[GameRecordV4]) -> None:
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
        }


def collate_v4(batch: list[dict]) -> dict:
    """Pad rosters, concat flat sup_* tensors. No prob fields."""
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
    # pair indices
    "_PAIR_PTS_IDX", "_PAIR_FGM_IDX", "_PAIR_FGA_IDX", "_PAIR_3PM_IDX",
    "_PAIR_3PA_IDX", "_PAIR_AST_IDX", "_PAIR_TOV_IDX", "_PAIR_BLK_IDX",
    "_PLAYER_FTM_IDX", "_PLAYER_FTA_IDX", "_PLAYER_OREB_IDX",
    "_PLAYER_DREB_IDX", "_PLAYER_STL_IDX", "_PLAYER_PF_IDX",
    # v4 record / dataset
    "GameRecordV4", "build_records_v4", "build_vocab_from_records_v4",
    "GameDatasetV4", "collate_v4",
    # v4 minutes / pace / decisions
    "BoxMinutesPace", "load_box_minutes_and_pace", "load_play_decisions",
    # default DB paths
    "DEFAULT_CORE_DB", "DEFAULT_FEATURES_DB", "DEFAULT_MATCHUP_DB",
    "DEFAULT_PLAYER_GAME_STATS_DB", "DEFAULT_PLAYER_DECISIONS_DB",
    "DEFAULT_PLAYER_DEBUT_DB",
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
