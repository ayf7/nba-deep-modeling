#!/usr/bin/env python3
"""Pre-compute v5 features into a single SQLite database.

Runs the full build_records_v5 pipeline once and writes all per-game tensors
into SQLite tables. Subsequent training runs load from this DB directly,
skipping the ~8 min Python preprocessing.

Usage:
    python models_cme_v5/scripts/build_features_db.py [--output features_precomputed.db]

The output DB contains:
    - meta: build parameters (lookback, decay, etc.)
    - vocab: player_id -> token_idx mapping
    - team_vocab: team_id -> token_idx mapping
    - games: per-game scalars (label, margin, odds, rest, tabular, team_box, pace)
    - players: per-(game, side, slot) features (player_idx, career_year, form_stats,
               box_targets, rotation_targets, minutes)
    - pairs: per-(game, side, off_slot, def_slot) matchup targets
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
V5_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_SCRIPTS))

from cme_v5_common import (
    BOX_TARGETS, K_BOX, K_PAIR, PAIR_TARGETS,
    DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    DEFAULT_PLAYER_DEBUT_DB,
    DEFAULT_PLAYER_DECISIONS_DB, DEFAULT_PLAYER_GAME_STATS_DB,
    DEFAULT_V5_FEATURES_DB,
    N_SEGMENTS, PLAYER_FORM_DIM, TABULAR_FEATURE_COLUMNS,
    GameRecordV5, build_records_v5, build_team_vocab,
    build_vocab_from_records_v5, fit_player_form_stats, fit_tabular_stats,
    load_box_minutes_and_pace, load_game_odds, load_game_scores, load_games,
    load_matchup_rows_v2, load_minute_presence, load_play_decisions,
    load_player_debut, load_player_first_season, load_player_game_stats,
    load_player_histories, load_regulation_scores, load_team_exposures,
)

REST_DAYS_CLIP = 7.0


def _pack_floats(vals: tuple | list) -> bytes:
    return struct.pack(f"{len(vals)}f", *vals)


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS vocab (
            player_id TEXT PRIMARY KEY,
            token_idx INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS team_vocab (
            team_id TEXT PRIMARY KEY,
            token_idx INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT NOT NULL,
            split TEXT NOT NULL,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL,
            label INTEGER NOT NULL,
            margin REAL NOT NULL,
            home_team_idx INTEGER NOT NULL,
            away_team_idx INTEGER NOT NULL,
            home_rest REAL NOT NULL,
            away_rest REAL NOT NULL,
            home_dec_odds REAL NOT NULL,
            away_dec_odds REAL NOT NULL,
            has_odds INTEGER NOT NULL,
            tabular BLOB NOT NULL,
            team_box_home BLOB NOT NULL,
            team_box_away BLOB NOT NULL,
            home_pace_actual REAL NOT NULL,
            away_pace_actual REAL NOT NULL,
            home_minutes_valid INTEGER NOT NULL,
            away_minutes_valid INTEGER NOT NULL,
            home_rotation_valid INTEGER NOT NULL,
            away_rotation_valid INTEGER NOT NULL,
            PRIMARY KEY (game_id, window_start)
        );
        CREATE TABLE IF NOT EXISTS players (
            game_id TEXT NOT NULL,
            window_start TEXT NOT NULL,
            side INTEGER NOT NULL,
            slot INTEGER NOT NULL,
            player_idx INTEGER NOT NULL,
            career_year_idx INTEGER NOT NULL,
            form_stats BLOB NOT NULL,
            minutes_actual REAL NOT NULL,
            rotation_target BLOB NOT NULL,
            PRIMARY KEY (game_id, window_start, side, slot)
        );
        CREATE TABLE IF NOT EXISTS player_labels (
            game_id TEXT NOT NULL,
            window_start TEXT NOT NULL,
            side INTEGER NOT NULL,
            slot INTEGER NOT NULL,
            targets BLOB NOT NULL,
            PRIMARY KEY (game_id, window_start, side, slot)
        );
        CREATE TABLE IF NOT EXISTS pairs (
            game_id TEXT NOT NULL,
            window_start TEXT NOT NULL,
            side INTEGER NOT NULL,
            off_slot INTEGER NOT NULL,
            def_slot INTEGER NOT NULL,
            targets BLOB NOT NULL,
            PRIMARY KEY (game_id, window_start, side, off_slot, def_slot)
        );
        CREATE INDEX IF NOT EXISTS idx_games_window ON games(window_start, split);
        CREATE INDEX IF NOT EXISTS idx_players_game ON players(game_id, window_start);
        CREATE INDEX IF NOT EXISTS idx_pairs_game ON pairs(game_id, window_start);
    """)


def _write_records(
    conn: sqlite3.Connection,
    records: list[GameRecordV5],
    split: str,
    window_start: str,
    window_end: str,
) -> None:
    game_rows = []
    player_rows = []
    player_label_rows = []
    pair_rows = []

    for r in records:
        gid = r.game_id
        game_rows.append((
            gid, split, window_start, window_end,
            r.label, r.margin,
            r.home_team_idx, r.away_team_idx,
            r.home_rest / REST_DAYS_CLIP, r.away_rest / REST_DAYS_CLIP,
            r.home_dec_odds, r.away_dec_odds, int(r.has_odds),
            _pack_floats(r.tabular),
            _pack_floats(r.team_box_home),
            _pack_floats(r.team_box_away),
            r.home_pace_actual, r.away_pace_actual,
            int(r.home_minutes_valid), int(r.away_minutes_valid),
            int(r.home_rotation_valid), int(r.away_rotation_valid),
        ))

        for side, (idxs, cyears, stats, mins, rots) in enumerate([
            (r.home_player_idx, r.home_career_year_idx,
             r.home_player_stats, r.home_minutes_actual, r.home_rotation_target),
            (r.away_player_idx, r.away_career_year_idx,
             r.away_player_stats, r.away_minutes_actual, r.away_rotation_target),
        ]):
            for slot, (pidx, cy, st, m, rot) in enumerate(zip(idxs, cyears, stats, mins, rots)):
                player_rows.append((
                    gid, window_start, side, slot,
                    pidx, cy,
                    _pack_floats(st) if st else b"",
                    m,
                    _pack_floats(rot),
                ))

        for pl in r.player_labels:
            player_label_rows.append((
                gid, window_start, pl.side, pl.slot,
                _pack_floats(pl.targets),
            ))

        for p in r.pair_labels:
            pair_rows.append((
                gid, window_start, p.side, p.off_slot, p.def_slot,
                _pack_floats(p.targets),
            ))

    conn.executemany(
        "INSERT OR REPLACE INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        game_rows,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?,?,?,?)",
        player_rows,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO player_labels VALUES (?,?,?,?,?)",
        player_label_rows,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO pairs VALUES (?,?,?,?,?,?)",
        pair_rows,
    )
    conn.commit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute v5 features DB")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "data" / "features_v5_precomputed.db")
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--player-game-stats-db", type=Path, default=DEFAULT_PLAYER_GAME_STATS_DB)
    p.add_argument("--player-debut-db", type=Path, default=DEFAULT_PLAYER_DEBUT_DB)
    p.add_argument("--decisions-db", type=Path, default=DEFAULT_PLAYER_DECISIONS_DB)
    p.add_argument("--v5-features-db", type=Path, default=DEFAULT_V5_FEATURES_DB)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--initial-train-end", type=str, default="2023-12-31")
    p.add_argument("--train-start", type=str, default=None)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--min-games-before", type=int, default=50)
    p.add_argument("--use-player-form", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[output] {args.output}")

    # Global data loading (same as backtest_cme_v5.py)
    t0 = time.time()
    print("[load] games + exposures + scores + decisions + v5_features ...")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    all_gids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, all_gids)
    play_decisions = load_play_decisions(args.decisions_db, all_gids)
    game_odds = load_game_odds(args.core_db, all_gids)
    player_first_season = load_player_debut(args.player_debut_db, args.player_game_stats_db)
    minute_presence = load_minute_presence(args.v5_features_db)
    regulation_scores = load_regulation_scores(args.v5_features_db)
    player_histories = load_player_histories(args.core_db) if args.use_player_form else None
    print(f"[load] done in {time.time() - t0:.1f}s — "
          f"n_games={len(games_all)} play_decisions={len(play_decisions)}")

    if args.train_start:
        train_start = pd.Timestamp(args.train_start)
        games_all = games_all[games_all["game_date"] >= train_start].copy()
        print(f"[train-start] {args.train_start}: {len(games_all)} games")

    # Generate monthly windows (same logic as backtest)
    initial_train_end = pd.Timestamp(args.initial_train_end)
    games_all = games_all.sort_values("game_date").reset_index(drop=True)
    test_games = games_all[games_all["game_date"] > initial_train_end]
    months = sorted(test_games["game_date"].dt.to_period("M").unique())
    if args.max_windows:
        months = months[:args.max_windows]

    conn = sqlite3.connect(str(args.output))
    _create_tables(conn)

    # Build ONE global vocab from ALL games — stable token_idx across all windows.
    print("[vocab] building global vocab from all games ...")
    t_vocab = time.time()
    all_matchup = load_matchup_rows_v2(args.matchup_db, all_gids)
    vocab = build_vocab_from_records_v5(
        games_all, histories, all_matchup, play_decisions,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(games_all)
    print(f"[vocab] {vocab.size} players, {team_vocab.size} teams ({time.time()-t_vocab:.1f}s)")

    # Write global vocab to DB once
    conn.execute("DELETE FROM vocab")
    for pid, idx in vocab.player_to_idx.items():
        conn.execute("INSERT INTO vocab VALUES (?, ?)", (pid, idx))
    conn.execute("DELETE FROM team_vocab")
    for tid, idx in team_vocab.team_to_idx.items():
        conn.execute("INSERT INTO team_vocab VALUES (?, ?)", (tid, idx))
    conn.commit()

    print(f"[windows] {len(months)} monthly windows to build")

    for wi, month in enumerate(months):
        window_start = str(month.start_time.date())
        window_end = str(month.end_time.date())
        t1 = time.time()

        # Split: train = before window, val = last val_frac of train, test = window
        train_all = games_all[games_all["game_date"] <= initial_train_end].copy()
        # Expanding window: include months between initial_train_end and current window
        expanding = games_all[
            (games_all["game_date"] > initial_train_end) &
            (games_all["game_date"] < month.start_time)
        ]
        train_all = pd.concat([train_all, expanding], ignore_index=True)

        n_val = max(1, int(len(train_all) * args.val_frac))
        train_fit = train_all.iloc[:-n_val]
        val = train_all.iloc[-n_val:]
        test = games_all[
            (games_all["game_date"] >= month.start_time) &
            (games_all["game_date"] <= month.end_time)
        ]

        print(f"  [window {wi+1}/{len(months)}] {window_start} → {window_end} | "
              f"train={len(train_fit)} val={len(val)} test={len(test)}")

        # Load per-window matchup/box data
        train_gids = [str(g) for g in train_fit["game_id"].tolist()]
        val_gids = [str(g) for g in val["game_id"].tolist()]
        test_gids = [str(g) for g in test["game_id"].tolist()]

        train_matchup = {g: all_matchup[g] for g in train_gids if g in all_matchup}
        val_matchup = {g: all_matchup[g] for g in val_gids if g in all_matchup}
        test_matchup = {g: all_matchup[g] for g in test_gids if g in all_matchup}
        train_box = load_box_minutes_and_pace(args.player_game_stats_db, train_gids)
        val_box = load_box_minutes_and_pace(args.player_game_stats_db, val_gids)
        test_box = load_box_minutes_and_pace(args.player_game_stats_db, test_gids)
        train_pl = load_player_game_stats(args.core_db, train_gids)
        val_pl = load_player_game_stats(args.core_db, val_gids)
        test_pl = load_player_game_stats(args.core_db, test_gids)

        tabular_stats = fit_tabular_stats(train_fit)

        player_form_stats = None
        if player_histories is not None:
            player_form_stats = fit_player_form_stats(
                player_histories, train_fit,
                lookback_games=DEFAULT_PLAYER_FORM_LOOKBACK,
                decay=DEFAULT_PLAYER_FORM_DECAY,
            )

        common = dict(
            histories=histories, vocab=vocab, team_vocab=team_vocab,
            play_decisions=play_decisions, game_scores=scores,
            lookback_games=args.lookback_games, decay=args.decay,
            tabular_stats=tabular_stats,
            player_first_season=player_first_season,
            player_histories=player_histories, player_form_stats=player_form_stats,
            player_form_lookback=DEFAULT_PLAYER_FORM_LOOKBACK,
            player_form_decay=DEFAULT_PLAYER_FORM_DECAY,
            game_odds=game_odds,
            minute_presence=minute_presence,
            regulation_scores=regulation_scores,
        )

        train_recs = build_records_v5(
            train_fit, matchup_rows=train_matchup,
            box_minutes_pace=train_box, player_game_stats=train_pl, **common,
        )
        val_recs = build_records_v5(
            val, matchup_rows=val_matchup,
            box_minutes_pace=val_box, player_game_stats=val_pl, **common,
        )
        test_recs = build_records_v5(
            test, matchup_rows=test_matchup,
            box_minutes_pace=test_box, player_game_stats=test_pl, **common,
        )

        _write_records(conn, train_recs, "train", window_start, window_end)
        _write_records(conn, val_recs, "val", window_start, window_end)
        _write_records(conn, test_recs, "test", window_start, window_end)

        conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                     (f"vocab_size_{window_start}", str(vocab.size)))
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                     (f"team_vocab_size_{window_start}", str(team_vocab.size)))
        conn.commit()

        dt = time.time() - t1
        print(f"    wrote {len(train_recs)}+{len(val_recs)}+{len(test_recs)} records ({dt:.1f}s)")

    # Save global meta
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("lookback_games", str(args.lookback_games)))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("decay", str(args.decay)))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("val_frac", str(args.val_frac)))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("initial_train_end", args.initial_train_end))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("use_player_form", str(args.use_player_form)))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("n_windows", str(len(months))))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("player_form_dim", str(PLAYER_FORM_DIM)))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("n_segments", str(N_SEGMENTS)))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("k_box", str(K_BOX)))
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                 ("k_pair", str(K_PAIR)))
    conn.commit()
    conn.close()

    print(f"\n[done] {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
