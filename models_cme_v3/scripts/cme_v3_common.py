"""CME-v3 data layer (hybrid: v2 channels + per-player involvement shares).

v3 builds on v2's K_PAIR=9 / K_PLAYER=6 / K_BOX=14 supervision and adds
per-player **involvement shares** as a direct supervision target:

    alpha_off_actual[i] = (sum_j exposure_seconds[i, j]) / sum_i,j exposure_seconds[i, j]
    alpha_def_actual[j] = (sum_i exposure_seconds[i, j]) / sum_i,j exposure_seconds[i, j]

These per-side shares sum to 1 over the unmasked lineup and anchor the
model's softmax involvement heads to observed minutes/possession share.
For a player who didn't play (or played 0 matchup seconds), the share is
0; the supervision mask flags games whose recorded exposure totals are
non-trivial (avoids spurious supervision on games with no matchup rows).

Everything else is reused from cme_v2_common: matchup row loader, pair
targets, per-player BOX label construction, team-box totals.
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

V2_SCRIPTS = Path(__file__).resolve().parents[2] / "models_cme_v2" / "scripts"
sys.path.insert(0, str(V2_SCRIPTS))

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
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB,
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
    PairLabelV2,
    PlayerBoxLabel,
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
    load_game_player_status,
    load_game_scores,
    load_games,
    load_matchup_rows_v2,
    load_player_game_stats,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
    play_prob_for,
    transform_player_form,
    transform_tabular_row,
)


# Minimum total exposure_seconds (per side) for the involvement supervision
# to count. Avoids dividing by zero when matchup rows are missing/sparse,
# and dampens supervision in cases where only a partial slice of pairs was
# recorded.
MIN_EXPOSURE_SECONDS = 100.0


# ----------------------------- exposure shares ----------------------------- #


def load_pair_exposure_seconds(
    matchup_db: Path, game_ids: list[str]
) -> dict[str, list[tuple[str, str, str, float]]]:
    """{game_id: [(off_team_id, off_pid, def_pid, exposure_seconds), ...]}.

    Pull JUST exposure_seconds so we can compute actual involvement shares
    independently from the full per-target pair rows. This duplicates a
    subset of load_matchup_rows_v2 but is cheaper when we don't need the
    full target tuple, and makes the computation explicit.
    """
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
                f"defender_player_id, exposure_seconds "
                f"FROM matchup_training_rows "
                f"WHERE game_id IN ({placeholders}) AND exposure_seconds > 0",
                conn,
                params=ids,
            )
            for row in df.itertuples(index=False):
                out.setdefault(str(row.game_id), []).append(
                    (
                        str(row.offensive_team_id),
                        str(row.offensive_player_id),
                        str(row.defender_player_id),
                        float(row.exposure_seconds),
                    )
                )
    return out


def _compute_actual_shares(
    matchup_rows_for_game: list[tuple[str, str, str, tuple[float, ...]]] | None,
    *,
    home_team: str,
    away_team: str,
    home_pid_to_slot: dict[str, int],
    away_pid_to_slot: dict[str, int],
    n_home: int,
    n_away: int,
    exposure_idx: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], float, float]:
    """Per-side actual involvement shares from raw matchup rows.

    Returns (home_alpha_off, home_alpha_def, away_alpha_off, away_alpha_def,
             total_home_off_seconds, total_away_off_seconds).

    A side's "off" share is from rows where THAT team is on offense (and
    sums to 1 if total exposure > 0). "def" share is from rows where the
    OPPOSING team is on offense (i.e., this team is defending).
    """
    home_off = [0.0] * n_home  # home players on offense
    away_def = [0.0] * n_away  # away players on defense (opposing same offense)
    away_off = [0.0] * n_away
    home_def = [0.0] * n_home

    total_home_off = 0.0
    total_away_off = 0.0

    if matchup_rows_for_game is not None:
        for off_team, off_pid, def_pid, vals in matchup_rows_for_game:
            exposure = float(vals[exposure_idx])
            if exposure <= 0.0:
                continue
            if off_team == home_team:
                if off_pid in home_pid_to_slot:
                    home_off[home_pid_to_slot[off_pid]] += exposure
                if def_pid in away_pid_to_slot:
                    away_def[away_pid_to_slot[def_pid]] += exposure
                total_home_off += exposure
            elif off_team == away_team:
                if off_pid in away_pid_to_slot:
                    away_off[away_pid_to_slot[off_pid]] += exposure
                if def_pid in home_pid_to_slot:
                    home_def[home_pid_to_slot[def_pid]] += exposure
                total_away_off += exposure

    def _normalize(seconds: list[float], total: float) -> tuple[float, ...]:
        if total < MIN_EXPOSURE_SECONDS:
            return tuple(0.0 for _ in seconds)
        return tuple(s / total for s in seconds)

    return (
        _normalize(home_off, total_home_off),
        _normalize(home_def, total_away_off),
        _normalize(away_off, total_away_off),
        _normalize(away_def, total_home_off),
        total_home_off,
        total_away_off,
    )


# ----------------------------- records + dataset ----------------------------- #


@dataclass(frozen=True)
class GameRecordV3:
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
    pair_labels: tuple[PairLabelV2, ...]
    player_labels: tuple[PlayerBoxLabel, ...]
    team_box_home: tuple[float, ...]
    team_box_away: tuple[float, ...]
    tabular: tuple[float, ...]
    home_player_stats: tuple[tuple[float, ...], ...]
    away_player_stats: tuple[tuple[float, ...], ...]
    # v3 additions: per-player actual involvement shares (sum to 1 on each side)
    home_alpha_off_actual: tuple[float, ...]  # length len(home_player_idx)
    home_alpha_def_actual: tuple[float, ...]
    away_alpha_off_actual: tuple[float, ...]  # length len(away_player_idx)
    away_alpha_def_actual: tuple[float, ...]
    # Mask: whether the home-offense (=away-defense) exposure total was non-trivial
    home_off_exposure_valid: bool
    away_off_exposure_valid: bool
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


_EXPOSURE_IDX = PAIR_TARGETS.index("exposure_possessions")


def build_records_v3(
    games: pd.DataFrame,
    histories: dict[str, list[TeamGameExposure]],
    *,
    vocab: Vocab,
    team_vocab: TeamVocab,
    status_lookup: dict[tuple[str, str], str],
    calibration: dict[str, float],
    game_scores: dict[str, tuple[int, int]],
    matchup_rows: dict[str, list[tuple[str, str, str, tuple[float, ...]]]] | None,
    player_game_stats: dict[tuple[str, str], tuple[float, ...]] | None,
    lookback_games: int,
    decay: float,
    tabular_stats: TabularStats,
    player_histories: dict[str, PlayerHistory] | None = None,
    player_form_stats: PlayerFormStats | None = None,
    player_form_lookback: int = DEFAULT_PLAYER_FORM_LOOKBACK,
    player_form_decay: float = DEFAULT_PLAYER_FORM_DECAY,
    game_odds: dict[str, tuple[float, float]] | None = None,
) -> list[GameRecordV3]:
    """Build per-game v3 records. Identical to v2 but additionally computes
    per-player involvement shares (alpha_off_actual / alpha_def_actual) from
    `matchup_rows[<gid>][...][exposure_possessions]` aggregated by side."""
    raw_tabular = games[list(TABULAR_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    out: list[GameRecordV3] = []
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

        # --- pair labels + accumulators for per-player BOX assembly ---
        pair_labels: list[PairLabelV2] = []
        rows_for_home_off: dict[int, list[tuple[float, ...]]] = {i: [] for i in range(len(home_pids))}
        rows_for_away_off: dict[int, list[tuple[float, ...]]] = {i: [] for i in range(len(away_pids))}

        if matchup_rows is not None:
            for off_team, off_pid, def_pid, vals in matchup_rows.get(gid, []):
                if off_team == home_team:
                    if off_pid in home_pid_to_slot and def_pid in away_pid_to_slot:
                        slot = home_pid_to_slot[off_pid]
                        pair_labels.append(PairLabelV2(
                            side=0,
                            off_slot=slot,
                            def_slot=away_pid_to_slot[def_pid],
                            targets=vals,
                        ))
                        rows_for_home_off[slot].append(vals)
                elif off_team == away_team:
                    if off_pid in away_pid_to_slot and def_pid in home_pid_to_slot:
                        slot = away_pid_to_slot[off_pid]
                        pair_labels.append(PairLabelV2(
                            side=1,
                            off_slot=slot,
                            def_slot=home_pid_to_slot[def_pid],
                            targets=vals,
                        ))
                        rows_for_away_off[slot].append(vals)

        # --- per-player BOX labels ---
        player_labels: list[PlayerBoxLabel] = []
        team_box_home = [0.0] * K_BOX
        team_box_away = [0.0] * K_BOX
        if matchup_rows is not None or player_game_stats is not None:
            for slot, pid in enumerate(home_pids):
                stats = player_game_stats.get((gid, pid)) if player_game_stats else None
                box = _build_box_label(rows_for_home_off.get(slot, []), stats)
                if any(v != 0.0 for v in box):
                    player_labels.append(PlayerBoxLabel(side=0, slot=slot, targets=box))
                for k in range(K_BOX):
                    team_box_home[k] += box[k]
            for slot, pid in enumerate(away_pids):
                stats = player_game_stats.get((gid, pid)) if player_game_stats else None
                box = _build_box_label(rows_for_away_off.get(slot, []), stats)
                if any(v != 0.0 for v in box):
                    player_labels.append(PlayerBoxLabel(side=1, slot=slot, targets=box))
                for k in range(K_BOX):
                    team_box_away[k] += box[k]

        home_score, away_score = game_scores[gid]
        team_box_home[BOX_INDEX["pts"]] = float(home_score)
        team_box_away[BOX_INDEX["pts"]] = float(away_score)
        tab = transform_tabular_row(raw_tabular[row_i], tabular_stats)

        # --- v3 addition: per-player involvement shares ---
        (h_off_share, h_def_share, a_off_share, a_def_share,
         tot_h_off, tot_a_off) = _compute_actual_shares(
            matchup_rows.get(gid) if matchup_rows is not None else None,
            home_team=home_team, away_team=away_team,
            home_pid_to_slot=home_pid_to_slot,
            away_pid_to_slot=away_pid_to_slot,
            n_home=len(home_pids), n_away=len(away_pids),
            exposure_idx=_EXPOSURE_IDX,
        )
        home_off_valid = tot_h_off >= MIN_EXPOSURE_SECONDS
        away_off_valid = tot_a_off >= MIN_EXPOSURE_SECONDS

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
        out.append(GameRecordV3(
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
            pair_labels=tuple(pair_labels),
            player_labels=tuple(player_labels),
            team_box_home=tuple(team_box_home),
            team_box_away=tuple(team_box_away),
            tabular=tuple(tab.tolist()),
            home_player_stats=home_stats,
            away_player_stats=away_stats,
            home_alpha_off_actual=h_off_share,
            home_alpha_def_actual=h_def_share,
            away_alpha_off_actual=a_off_share,
            away_alpha_def_actual=a_def_share,
            home_off_exposure_valid=home_off_valid,
            away_off_exposure_valid=away_off_valid,
            home_dec_odds=h_dec,
            away_dec_odds=a_dec,
            has_odds=has_odds,
        ))
    return out


def build_vocab_from_records_v3(
    games_train: pd.DataFrame,
    histories: dict[str, list[TeamGameExposure]],
    matchup_rows_train: dict[str, list[tuple[str, str, str, tuple[float, ...]]]],
    *,
    lookback_games: int,
    decay: float,
) -> Vocab:
    """Vocab from train rosters + train matchup rows. Identical to v2."""
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
        for _, off_pid, def_pid, _vals in rows:
            seen.add(off_pid)
            seen.add(def_pid)
    sorted_pids = sorted(seen)
    return Vocab(player_to_idx={p: i + 1 for i, p in enumerate(sorted_pids)})


class GameDatasetV3(Dataset):
    def __init__(self, records: list[GameRecordV3]) -> None:
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
            "team_box_home": torch.tensor(r.team_box_home, dtype=torch.float32),
            "team_box_away": torch.tensor(r.team_box_away, dtype=torch.float32),
            "sup_pair_side": sup_pair_side,
            "sup_pair_off": sup_pair_off,
            "sup_pair_def": sup_pair_def,
            "sup_pair_y": sup_pair_y,
            "sup_pl_side": sup_pl_side,
            "sup_pl_slot": sup_pl_slot,
            "sup_pl_y": sup_pl_y,
            # v3: per-player involvement-share targets (sum-to-1 on each side)
            "home_alpha_off_actual": torch.tensor(r.home_alpha_off_actual, dtype=torch.float32),
            "home_alpha_def_actual": torch.tensor(r.home_alpha_def_actual, dtype=torch.float32),
            "away_alpha_off_actual": torch.tensor(r.away_alpha_off_actual, dtype=torch.float32),
            "away_alpha_def_actual": torch.tensor(r.away_alpha_def_actual, dtype=torch.float32),
            "home_off_exposure_valid": torch.tensor(float(r.home_off_exposure_valid), dtype=torch.float32),
            "away_off_exposure_valid": torch.tensor(float(r.away_off_exposure_valid), dtype=torch.float32),
        }


def collate_v3(batch: list[dict]) -> dict:
    """Pad rosters; concat per-game flattened sup_* tensors. Adds v3 fields."""
    B = len(batch)
    L_h = max(b["home_idx"].numel() for b in batch)
    L_a = max(b["away_idx"].numel() for b in batch)
    D_stats = batch[0]["home_stats"].size(-1)

    home_idx = torch.zeros(B, L_h, dtype=torch.long)
    home_prob = torch.zeros(B, L_h, dtype=torch.float32)
    home_mask = torch.zeros(B, L_h, dtype=torch.bool)
    away_idx = torch.zeros(B, L_a, dtype=torch.long)
    away_prob = torch.zeros(B, L_a, dtype=torch.float32)
    away_mask = torch.zeros(B, L_a, dtype=torch.bool)
    home_stats = torch.zeros(B, L_h, D_stats, dtype=torch.float32)
    away_stats = torch.zeros(B, L_a, D_stats, dtype=torch.float32)

    # v3 actual-share targets (padded with 0; mask handled by sum check at loss time)
    home_alpha_off_actual = torch.zeros(B, L_h, dtype=torch.float32)
    home_alpha_def_actual = torch.zeros(B, L_h, dtype=torch.float32)
    away_alpha_off_actual = torch.zeros(B, L_a, dtype=torch.float32)
    away_alpha_def_actual = torch.zeros(B, L_a, dtype=torch.float32)

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
        home_alpha_off_actual[b, :n_h] = item["home_alpha_off_actual"]
        home_alpha_def_actual[b, :n_h] = item["home_alpha_def_actual"]
        away_alpha_off_actual[b, :n_a] = item["away_alpha_off_actual"]
        away_alpha_def_actual[b, :n_a] = item["away_alpha_def_actual"]

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
        "team_box_home": torch.stack([b["team_box_home"] for b in batch]),
        "team_box_away": torch.stack([b["team_box_away"] for b in batch]),
        "home_alpha_off_actual": home_alpha_off_actual,
        "home_alpha_def_actual": home_alpha_def_actual,
        "away_alpha_off_actual": away_alpha_off_actual,
        "away_alpha_def_actual": away_alpha_def_actual,
        "home_off_exposure_valid": torch.stack([b["home_off_exposure_valid"] for b in batch]),
        "away_off_exposure_valid": torch.stack([b["away_off_exposure_valid"] for b in batch]),
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
    "PAIR_TARGETS", "K_PAIR", "PLAYER_TARGETS", "K_PLAYER",
    "BOX_TARGETS", "K_BOX", "BOX_INDEX",
    # pair indices
    "_PAIR_PTS_IDX", "_PAIR_FGM_IDX", "_PAIR_FGA_IDX", "_PAIR_3PM_IDX",
    "_PAIR_3PA_IDX", "_PAIR_AST_IDX", "_PAIR_TOV_IDX", "_PAIR_BLK_IDX",
    "_PLAYER_FTM_IDX", "_PLAYER_FTA_IDX", "_PLAYER_OREB_IDX",
    "_PLAYER_DREB_IDX", "_PLAYER_STL_IDX", "_PLAYER_PF_IDX",
    "MIN_EXPOSURE_SECONDS",
    # reused utilities
    "DEFAULT_CALIBRATION_PATH", "DEFAULT_CORE_DB", "DEFAULT_FEATURES_DB",
    "DEFAULT_INJURY_DB", "DEFAULT_LINEUP_DECAY", "DEFAULT_LINEUP_LOOKBACK_GAMES",
    "DEFAULT_MATCHUP_DB", "DEFAULT_PLAYER_FORM_DECAY", "DEFAULT_PLAYER_FORM_LOOKBACK",
    "PLAYER_FORM_DIM", "TABULAR_FEATURE_COLUMNS", "REST_DAYS_CLIP",
    "PairLabelV2", "PlayerBoxLabel",
    "PlayerHistory", "PlayerFormStats", "TabularStats",
    "TeamGameExposure", "TeamVocab", "Vocab",
    "build_team_vocab", "fit_player_form_stats", "fit_tabular_stats",
    "load_game_odds", "load_game_player_status", "load_game_scores",
    "load_games", "load_matchup_rows_v2", "load_player_game_stats",
    "load_player_histories", "load_status_calibration", "load_team_exposures",
    # v3 additions
    "GameRecordV3", "build_records_v3", "build_vocab_from_records_v3",
    "GameDatasetV3", "collate_v3", "load_pair_exposure_seconds",
]
