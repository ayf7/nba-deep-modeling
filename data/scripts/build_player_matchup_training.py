#!/usr/bin/env python3
"""Build cleaned player-matchup rows for embedding models."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"

DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "player_matchup_training.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build grouped player-matchup training rows from nba_core.sqlite."
    )
    parser.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-exposure-possessions",
        type=float,
        default=0.5,
        help="Minimum summed partial possessions retained per game/player/defender row.",
    )
    parser.add_argument(
        "--min-matchup-seconds",
        type=float,
        default=5.0,
        help="Minimum summed matchup seconds retained per game/player/defender row.",
    )
    return parser.parse_args()


def require_core_tables(conn: sqlite3.Connection) -> None:
    required = {"games", "player_matchups"}
    existing = {
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM core.sqlite_master
            WHERE type = 'table'
            """
        )
    }
    missing = sorted(required - existing)
    if missing:
        raise RuntimeError(f"Missing core tables: {', '.join(missing)}")


def build_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS matchup_training_rows;
        DROP TABLE IF EXISTS build_metadata;

        CREATE TABLE matchup_training_rows (
            game_id TEXT NOT NULL,
            game_date TEXT NOT NULL,
            season INTEGER NOT NULL,
            offensive_team_id TEXT,
            offensive_team_tricode TEXT,
            offensive_player_id TEXT NOT NULL,
            offensive_player_name TEXT,
            defender_player_id TEXT NOT NULL,
            defender_player_name TEXT,
            exposure_possessions REAL NOT NULL,
            exposure_seconds REAL NOT NULL,
            player_points REAL NOT NULL,
            team_points REAL NOT NULL,
            matchup_assists REAL NOT NULL,
            matchup_potential_assists REAL NOT NULL,
            matchup_turnovers REAL NOT NULL,
            matchup_blocks REAL NOT NULL,
            matchup_fgm REAL NOT NULL,
            matchup_fga REAL NOT NULL,
            matchup_3pm REAL NOT NULL,
            matchup_3pa REAL NOT NULL,
            matchup_ftm REAL NOT NULL,
            matchup_fta REAL NOT NULL,
            shooting_fouls REAL NOT NULL,
            PRIMARY KEY (
                game_id,
                offensive_player_id,
                defender_player_id,
                offensive_team_id
            )
        );

        CREATE TABLE build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def insert_training_rows(
    conn: sqlite3.Connection,
    min_exposure_possessions: float,
    min_matchup_seconds: float,
) -> None:
    conn.execute(
        """
        INSERT INTO matchup_training_rows
        SELECT
            pm.game_id,
            g.game_date,
            g.season,
            pm.team_id AS offensive_team_id,
            pm.team_tricode AS offensive_team_tricode,
            pm.player_id AS offensive_player_id,
            MAX(pm.player_name_i) AS offensive_player_name,
            pm.defender_player_id,
            MAX(pm.defender_player_name_i) AS defender_player_name,
            SUM(COALESCE(pm.partial_possessions, 0.0)) AS exposure_possessions,
            SUM(COALESCE(pm.matchup_seconds, 0.0)) AS exposure_seconds,
            SUM(COALESCE(pm.player_points, 0)) AS player_points,
            SUM(COALESCE(pm.team_points, 0)) AS team_points,
            SUM(COALESCE(pm.matchup_assists, 0)) AS matchup_assists,
            SUM(COALESCE(pm.matchup_potential_assists, 0)) AS matchup_potential_assists,
            SUM(COALESCE(pm.matchup_turnovers, 0)) AS matchup_turnovers,
            SUM(COALESCE(pm.matchup_blocks, 0)) AS matchup_blocks,
            SUM(COALESCE(pm.matchup_fgm, 0)) AS matchup_fgm,
            SUM(COALESCE(pm.matchup_fga, 0)) AS matchup_fga,
            SUM(COALESCE(pm.matchup_3pm, 0)) AS matchup_3pm,
            SUM(COALESCE(pm.matchup_3pa, 0)) AS matchup_3pa,
            SUM(COALESCE(pm.matchup_ftm, 0)) AS matchup_ftm,
            SUM(COALESCE(pm.matchup_fta, 0)) AS matchup_fta,
            SUM(COALESCE(pm.shooting_fouls, 0)) AS shooting_fouls
        FROM core.player_matchups pm
        JOIN core.games g
          ON g.game_id = pm.game_id
        WHERE g.game_date IS NOT NULL
          AND g.season IS NOT NULL
          AND pm.player_id IS NOT NULL
          AND pm.player_id <> ''
          AND pm.defender_player_id IS NOT NULL
          AND pm.defender_player_id <> ''
          AND pm.partial_possessions IS NOT NULL
          AND pm.matchup_seconds IS NOT NULL
          AND pm.player_points IS NOT NULL
        GROUP BY
            pm.game_id,
            g.game_date,
            g.season,
            pm.team_id,
            pm.team_tricode,
            pm.player_id,
            pm.defender_player_id
        HAVING exposure_possessions >= ?
           AND exposure_seconds >= ?
           AND player_points >= 0
        """,
        (min_exposure_possessions, min_matchup_seconds),
    )


def create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_matchup_training_date
        ON matchup_training_rows (game_date, game_id);

        CREATE INDEX idx_matchup_training_players
        ON matchup_training_rows (offensive_player_id, defender_player_id);

        CREATE INDEX idx_matchup_training_season
        ON matchup_training_rows (season);
        """
    )


def write_metadata(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> None:
    metadata = {
        "core_db": str(args.core_db),
        "output": str(args.output),
        "min_exposure_possessions": args.min_exposure_possessions,
        "min_matchup_seconds": args.min_matchup_seconds,
        **summary,
    }
    conn.executemany(
        """
        INSERT INTO build_metadata (key, value)
        VALUES (?, ?)
        """,
        [(key, json.dumps(value)) for key, value in metadata.items()],
    )


def summarize(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT game_id) AS games,
            COUNT(DISTINCT offensive_player_id) AS offensive_players,
            COUNT(DISTINCT defender_player_id) AS defenders,
            MIN(game_date) AS min_game_date,
            MAX(game_date) AS max_game_date,
            SUM(exposure_possessions) AS total_exposure_possessions,
            SUM(player_points) AS total_player_points
        FROM matchup_training_rows
        """
    ).fetchone()
    keys = [
        "rows",
        "games",
        "offensive_players",
        "defenders",
        "min_game_date",
        "max_game_date",
        "total_exposure_possessions",
        "total_player_points",
    ]
    return dict(zip(keys, row))


def main() -> None:
    args = parse_args()
    if not args.core_db.exists():
        raise FileNotFoundError(f"Core database not found: {args.core_db}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.output) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("ATTACH DATABASE ? AS core", (str(args.core_db),))
        require_core_tables(conn)
        build_schema(conn)
        insert_training_rows(
            conn,
            min_exposure_possessions=args.min_exposure_possessions,
            min_matchup_seconds=args.min_matchup_seconds,
        )
        create_indexes(conn)
        summary = summarize(conn)
        write_metadata(conn, args, summary)
        conn.commit()

    print(json.dumps(summary, indent=2))
    print(f"Player matchup training SQLite database: {args.output}")


if __name__ == "__main__":
    main()
