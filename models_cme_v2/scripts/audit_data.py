#!/usr/bin/env python3
"""Audit the CME-v2 training data for cleanliness.

Four sections:
  1. Tabular feature distributions (mean/std/min/max/NaN% across 41 features)
  2. Label correctness sweep (reconstructed team_box vs unrestricted ground truth)
  3. Lineup audit (do top-N lineups pick real rotation players)
  4. Single-game spot check (dump one full record)

Output goes to stdout (and the file specified by --report).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_SCRIPTS))

from cme_v2_common import (  # noqa: E402
    BOX_INDEX, BOX_TARGETS, DEFAULT_CALIBRATION_PATH, DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB, DEFAULT_INJURY_DB, DEFAULT_LINEUP_DECAY,
    DEFAULT_LINEUP_LOOKBACK_GAMES, DEFAULT_MATCHUP_DB, K_PAIR, PAIR_TARGETS,
    Vocab, build_full_lineup, build_records_v2, build_team_vocab,
    fit_tabular_stats, load_game_player_status, load_game_scores, load_games,
    load_matchup_rows_v2, load_player_game_stats, load_status_calibration,
    load_team_exposures, play_prob_for,
)
from man_xfmr_common import TABULAR_FEATURE_COLUMNS  # noqa: E402

PAIR_TO_BOX = {
    "matchup_fgm": "fgm",
    "matchup_fga": "fga",
    "matchup_3pm": "3pm",
    "matchup_3pa": "3pa",
    "matchup_assists": "ast",
    "matchup_turnovers": "tov",
    "matchup_blocks": "blk",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--game-id", type=str, default=None,
                   help="Specific game_id for the spot check (default: latest train game)")
    p.add_argument("--report", type=Path,
                   default=REPO_ROOT / "models_cme_v2" / "artifacts" / "data_audit.txt")
    return p.parse_args()


def chrono_split(games, val_frac, test_frac):
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(games)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return (games.iloc[:n_train].reset_index(drop=True),
            games.iloc[n_train: n_train + n_val].reset_index(drop=True),
            games.iloc[n_train + n_val:].reset_index(drop=True))


def load_player_names(core_db: Path) -> dict[str, str]:
    with sqlite3.connect(core_db) as c:
        df = pd.read_sql_query("SELECT player_id, player_name_i FROM players", c)
    return dict(zip(df["player_id"].astype(str), df["player_name_i"].fillna("?")))


def load_team_abbrs(core_db: Path) -> dict[str, str]:
    with sqlite3.connect(core_db) as c:
        df = pd.read_sql_query(
            "SELECT DISTINCT home_team_id AS team_id, home_team_abbr AS abbr FROM games "
            "UNION SELECT DISTINCT away_team_id, away_team_abbr FROM games", c,
        )
    return dict(zip(df["team_id"].astype(str), df["abbr"]))


# ============== 1. Tabular feature distributions ==============
def audit_tabular(train_df: pd.DataFrame) -> str:
    out = ["", "=" * 90, "1. TABULAR FEATURE DISTRIBUTIONS", "=" * 90,
           f"train games: {len(train_df)},  features: {len(TABULAR_FEATURE_COLUMNS)}",
           "",
           f"  {'feature':38s}  {'mean':>9s} {'std':>8s} {'min':>9s} {'max':>9s} "
           f"{'%NaN':>6s} {'%zero':>6s}  flags",
           "  " + "-" * 100]
    flags_summary = []
    for col in TABULAR_FEATURE_COLUMNS:
        x = train_df[col].to_numpy(dtype="float64")
        n_nan = int(np.isnan(x).sum())
        xc = x[~np.isnan(x)]
        if xc.size == 0:
            out.append(f"  {col:38s}  ALL NaN")
            flags_summary.append((col, "ALL NaN"))
            continue
        n_zero = int((xc == 0).sum())
        mean, std, mn, mx = float(xc.mean()), float(xc.std()), float(xc.min()), float(xc.max())
        flag = []
        if n_nan / len(x) > 0.05:
            flag.append("HIGH-NaN")
        if std < 1e-6:
            flag.append("ZERO-STD")
        if mn == mx:
            flag.append("CONSTANT")
        if abs(mean) > 1000 or abs(std) > 1000:
            flag.append("LARGE-SCALE")
        out.append(
            f"  {col:38s}  {mean:>9.3f} {std:>8.3f} {mn:>9.3f} {mx:>9.3f} "
            f"{100*n_nan/len(x):>5.1f}% {100*n_zero/len(x):>5.1f}%  "
            + (",".join(flag) if flag else "")
        )
        if flag:
            flags_summary.append((col, ",".join(flag)))
    out.append("")
    if flags_summary:
        out.append("  FLAGS:")
        for c, f in flags_summary:
            out.append(f"    {c}: {f}")
    else:
        out.append("  No anomalous features.")
    return "\n".join(out)


# ============== 2. Label correctness sweep ==============
def audit_labels(train_df: pd.DataFrame, matchup_rows, player_game_stats,
                 game_scores, histories, vocab, lookback_games, decay,
                 statuses, calibration, tabular_stats) -> str:
    """Compare reconstructed team_box (filtered to top-N lineup) vs unrestricted
    sums from raw matchup rows + player_game_stats.

    Two sanity checks per game:
      A. team_box[pts] vs official game_scores  (this is what we already overwrite)
      B. team_box[fgm/fga/3pm/3pa/ast/tov/blk] vs unrestricted matchup sum per team
    """
    # Build records WITHOUT the pts overwrite to see the raw reconstruction error.
    # build_records_v2 already does the overwrite; we'll recompute the raw value here.
    print("  building train records ...")
    records = build_records_v2(
        train_df, matchup_rows=matchup_rows, player_game_stats=player_game_stats,
        histories=histories, vocab=vocab, team_vocab=build_team_vocab(train_df),
        status_lookup=statuses, calibration=calibration, game_scores=game_scores,
        lookback_games=lookback_games, decay=decay, tabular_stats=tabular_stats,
        player_histories=None, player_form_stats=None, game_odds=None,
    )

    # ---- A. pts reconstruction error (pre-overwrite) ----
    # The team_box[pts] in records is already overwritten. We need to redo the
    # pre-overwrite calculation by summing per-player labels.
    pts_recon_h = np.zeros(len(records))
    pts_recon_a = np.zeros(len(records))
    pts_off_h = np.zeros(len(records))
    pts_off_a = np.zeros(len(records))
    for i, r in enumerate(records):
        # Reconstruct what team_box[pts] WOULD be before overwrite: sum of
        # per-player label pts (each was built from filtered pair rows + filtered
        # player_only_stats).
        h_pts = sum(pl.targets[BOX_INDEX["pts"]] for pl in r.player_labels if pl.side == 0)
        a_pts = sum(pl.targets[BOX_INDEX["pts"]] for pl in r.player_labels if pl.side == 1)
        pts_recon_h[i] = h_pts
        pts_recon_a[i] = a_pts
        pts_off_h[i] = game_scores[r.game_id][0]
        pts_off_a[i] = game_scores[r.game_id][1]

    def _stats(name, recon, truth):
        diff = recon - truth
        return (f"{name:>20s}: MAE={np.mean(np.abs(diff)):6.2f}  bias={np.mean(diff):+7.2f}  "
                f"r={np.corrcoef(recon, truth)[0,1]:.3f}  "
                f"min={diff.min():+6.1f}  max={diff.max():+6.1f}  "
                f"|>5pts|={100*(np.abs(diff) > 5).mean():.1f}%")

    # ---- B. matchup-derived team stats ----
    # Compute unrestricted team totals by summing matchup_rows grouped by off_team.
    box_results = []
    for pair_col, box_name in PAIR_TO_BOX.items():
        pair_idx = PAIR_TARGETS.index(pair_col)
        box_idx = BOX_INDEX[box_name]
        recon_h = np.zeros(len(records))
        recon_a = np.zeros(len(records))
        unrest_h = np.zeros(len(records))
        unrest_a = np.zeros(len(records))
        for i, r in enumerate(records):
            recon_h[i] = r.team_box_home[box_idx]
            recon_a[i] = r.team_box_away[box_idx]
            # unrestricted: sum across ALL matchup rows for this game, grouped by off_team
            for off_team, _off_pid, _def_pid, vals in matchup_rows.get(r.game_id, []):
                if off_team == r.home_team_id:
                    unrest_h[i] += vals[pair_idx]
                elif off_team == r.away_team_id:
                    unrest_a[i] += vals[pair_idx]
        box_results.append((box_name, recon_h, recon_a, unrest_h, unrest_a))

    out = ["", "=" * 90, "2. LABEL CORRECTNESS SWEEP", "=" * 90,
           f"records: {len(records)}  (out of {len(train_df)} train games)",
           "",
           "  PTS: reconstructed-from-per-player-labels vs official game_scores",
           "    (this is what the team_box[pts] would be WITHOUT the overwrite fix at common.py:402)",
           "  " + _stats("home pts", pts_recon_h, pts_off_h),
           "  " + _stats("away pts", pts_recon_a, pts_off_a),
           "",
           "  Matchup-derived stats: reconstructed (top-N filtered) vs unrestricted DB sums",
           "    diff < 0 means reconstructed missed real plays (top-N filter dropped them)",
           "",
           f"  {'stat':>6s} | {'home: MAE / bias / drop%':<40s} | {'away: MAE / bias / drop%'}"]
    for box_name, rh, ra, uh, ua in box_results:
        d_h = rh - uh
        d_a = ra - ua
        drop_h = 100 * (d_h < 0).mean()
        drop_a = 100 * (d_a < 0).mean()
        out.append(f"  {box_name:>6s} | MAE={np.mean(np.abs(d_h)):5.2f} "
                   f"bias={np.mean(d_h):+6.2f} drop={drop_h:5.1f}%  "
                   f"| MAE={np.mean(np.abs(d_a)):5.2f} bias={np.mean(d_a):+6.2f} "
                   f"drop={drop_a:5.1f}%")
    return "\n".join(out), records


# ============== 3. Lineup audit ==============
def audit_lineups(records, vocab, name_map, abbr_map, n_samples=5) -> str:
    """For a sample of train games, show the top-N home lineup with names + play_probs."""
    out = ["", "=" * 90, "3. LINEUP AUDIT (sample of games)", "=" * 90,
           f"  lineup size: {len(records[0].home_player_idx)}",
           ""]
    inv_vocab = {v: k for k, v in vocab.player_to_idx.items()}
    rng = np.random.default_rng(42)
    sample_idxs = sorted(rng.choice(len(records), size=min(n_samples, len(records)),
                                    replace=False))
    for si in sample_idxs:
        r = records[si]
        h_abbr = abbr_map.get(r.home_team_id, r.home_team_id)
        a_abbr = abbr_map.get(r.away_team_id, r.away_team_id)
        out.append(f"  [{r.game_date.strftime('%Y-%m-%d')}] {h_abbr} vs {a_abbr}  "
                   f"(gid={r.game_id})")
        out.append(f"    HOME ({h_abbr}):")
        for slot, (pidx, prob) in enumerate(zip(r.home_player_idx, r.home_play_prob)):
            pid = inv_vocab.get(pidx, f"<oov={pidx}>")
            name = name_map.get(pid, "?")
            out.append(f"      slot {slot:2d}  prob={prob:.2f}  pid={pid:>7s}  {name}")
        out.append(f"    AWAY ({a_abbr}):")
        for slot, (pidx, prob) in enumerate(zip(r.away_player_idx, r.away_play_prob)):
            pid = inv_vocab.get(pidx, f"<oov={pidx}>")
            name = name_map.get(pid, "?")
            out.append(f"      slot {slot:2d}  prob={prob:.2f}  pid={pid:>7s}  {name}")
        out.append("")
    return "\n".join(out)


# ============== 4. Single-game spot check ==============
def audit_single_game(records, train_df, raw_features_row, name_map, abbr_map,
                      game_scores, matchup_rows, vocab) -> str:
    r = records[-1]  # latest train game by default
    raw = raw_features_row  # ALREADY a 1-d numpy of length 41
    inv_vocab = {v: k for k, v in vocab.player_to_idx.items()}

    h_abbr = abbr_map.get(r.home_team_id, r.home_team_id)
    a_abbr = abbr_map.get(r.away_team_id, r.away_team_id)

    out = ["", "=" * 90, "4. SINGLE-GAME SPOT CHECK", "=" * 90,
           f"  game_id={r.game_id}   date={r.game_date.strftime('%Y-%m-%d')}",
           f"  {h_abbr} (home) vs {a_abbr} (away)",
           f"  official score: {h_abbr} {game_scores[r.game_id][0]} - "
           f"{game_scores[r.game_id][1]} {a_abbr}   (home_win={r.label}, "
           f"margin={r.margin:+.0f})",
           "", "  -- TABULAR FEATURES (raw, pre-normalization) --"]
    for col, val in zip(TABULAR_FEATURE_COLUMNS, raw):
        out.append(f"    {col:38s}  {val:>10.3f}" + ("   NaN" if np.isnan(val) else ""))

    out.append("")
    out.append(f"  -- HOME LINEUP ({h_abbr}) --")
    for slot, (pidx, prob) in enumerate(zip(r.home_player_idx, r.home_play_prob)):
        pid = inv_vocab.get(pidx, f"<oov={pidx}>")
        name = name_map.get(pid, "?")
        out.append(f"    slot {slot:2d}  prob={prob:.2f}  pid={pid:>7s}  {name}")
    out.append(f"  -- AWAY LINEUP ({a_abbr}) --")
    for slot, (pidx, prob) in enumerate(zip(r.away_player_idx, r.away_play_prob)):
        pid = inv_vocab.get(pidx, f"<oov={pidx}>")
        name = name_map.get(pid, "?")
        out.append(f"    slot {slot:2d}  prob={prob:.2f}  pid={pid:>7s}  {name}")

    out.append("")
    out.append("  -- TEAM BOX LABELS (reconstructed; pts is overwritten) --")
    out.append(f"    {'stat':>6s}   {'home':>8s}   {'away':>8s}")
    for stat in BOX_TARGETS:
        out.append(f"    {stat:>6s}   {r.team_box_home[BOX_INDEX[stat]]:>8.1f}   "
                   f"{r.team_box_away[BOX_INDEX[stat]]:>8.1f}")

    # Top-3 scorers per side from per-player labels
    home_pl = sorted([pl for pl in r.player_labels if pl.side == 0],
                     key=lambda x: -x.targets[BOX_INDEX["pts"]])[:3]
    away_pl = sorted([pl for pl in r.player_labels if pl.side == 1],
                     key=lambda x: -x.targets[BOX_INDEX["pts"]])[:3]
    out.append("")
    out.append("  -- TOP 3 SCORERS (from per-player labels) --")
    out.append(f"    {h_abbr}:")
    for pl in home_pl:
        pid = inv_vocab.get(r.home_player_idx[pl.slot], "?")
        name = name_map.get(pid, "?")
        out.append(f"      slot {pl.slot:2d}  {name:20s}  pts={pl.targets[BOX_INDEX['pts']]:.0f} "
                   f"fgm/a={pl.targets[BOX_INDEX['fgm']]:.0f}/{pl.targets[BOX_INDEX['fga']]:.0f} "
                   f"3pm/a={pl.targets[BOX_INDEX['3pm']]:.0f}/{pl.targets[BOX_INDEX['3pa']]:.0f} "
                   f"ast={pl.targets[BOX_INDEX['ast']]:.0f}")
    out.append(f"    {a_abbr}:")
    for pl in away_pl:
        pid = inv_vocab.get(r.away_player_idx[pl.slot], "?")
        name = name_map.get(pid, "?")
        out.append(f"      slot {pl.slot:2d}  {name:20s}  pts={pl.targets[BOX_INDEX['pts']]:.0f} "
                   f"fgm/a={pl.targets[BOX_INDEX['fgm']]:.0f}/{pl.targets[BOX_INDEX['fga']]:.0f} "
                   f"3pm/a={pl.targets[BOX_INDEX['3pm']]:.0f}/{pl.targets[BOX_INDEX['3pa']]:.0f} "
                   f"ast={pl.targets[BOX_INDEX['ast']]:.0f}")

    out.append("")
    out.append(f"  -- PAIR LABEL COUNTS --")
    n_h = sum(1 for p in r.pair_labels if p.side == 0)
    n_a = sum(1 for p in r.pair_labels if p.side == 1)
    out.append(f"    home_off×away_def: {n_h} rows")
    out.append(f"    away_off×home_def: {n_a} rows")
    out.append(f"    total raw matchup rows (no top-N filter): "
               f"{len(matchup_rows.get(r.game_id, []))}")
    return "\n".join(out)


def main() -> None:
    args = parse_args()

    print("[load] games, scores, statuses, calibration")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)

    train_df, _val_df, _test_df = chrono_split(games_all, args.val_frac, args.test_frac)
    print(f"[split] train={len(train_df)}")

    train_gids = [str(g) for g in train_df["game_id"].tolist()]
    print("[load] matchup_rows + player_game_stats for train")
    matchup_rows = load_matchup_rows_v2(args.matchup_db, train_gids)
    player_game_stats = load_player_game_stats(args.core_db, train_gids)

    print("[load] player + team name maps")
    name_map = load_player_names(args.core_db)
    abbr_map = load_team_abbrs(args.core_db)

    print("[fit] tabular stats from train")
    tabular_stats = fit_tabular_stats(train_df)

    print("[build] vocab")
    pids_seen = set()
    for gid_rows in matchup_rows.values():
        for _, off_pid, def_pid, _ in gid_rows:
            pids_seen.add(off_pid)
            pids_seen.add(def_pid)
    vocab = Vocab(player_to_idx={pid: i + 1 for i, pid in enumerate(sorted(pids_seen))})

    sections = []

    print("[audit 1] tabular feature distributions")
    sections.append(audit_tabular(train_df))

    print("[audit 2] label correctness sweep (this is the slow one)")
    sec, records = audit_labels(
        train_df, matchup_rows, player_game_stats, scores, histories, vocab,
        args.lookback_games, args.decay, statuses, calibration, tabular_stats,
    )
    sections.append(sec)

    print("[audit 3] lineup audit")
    sections.append(audit_lineups(records, vocab, name_map, abbr_map, n_samples=5))

    print("[audit 4] single-game spot check")
    # raw tabular for the LAST train game
    raw_tabular = train_df[list(TABULAR_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    sections.append(audit_single_game(
        records, train_df, raw_tabular[-1], name_map, abbr_map,
        scores, matchup_rows, vocab,
    ))

    report = "\n".join(sections) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(f"\n[done] report written: {args.report}")
    print(report)


if __name__ == "__main__":
    main()
