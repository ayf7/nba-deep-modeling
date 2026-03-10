#!/usr/bin/env python3
"""Build model-facing feature tables from nba_core.sqlite.

The key rule for every feature in this file: a game's feature row may only use
games that happened before that game's date/order.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"

DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "features.sqlite"


def execute_script(conn: sqlite3.Connection, name: str, sql: str) -> None:
    conn.executescript(sql)
    conn.execute(
        "INSERT INTO feature_builds (build_name, notes) VALUES (?, ?)",
        (name, "completed"),
    )


def build_metadata(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS feature_builds;

        CREATE TABLE feature_builds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            build_name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        );
        """
    )


def build_labels(conn: sqlite3.Connection) -> None:
    execute_script(
        conn,
        "game_labels",
        """
        DROP TABLE IF EXISTS game_labels;

        CREATE TABLE game_labels (
            game_id TEXT PRIMARY KEY,
            season INTEGER NOT NULL,
            game_date TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            home_team_abbr TEXT,
            away_team_abbr TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_win INTEGER NOT NULL
        );

        INSERT INTO game_labels
        SELECT
            game_id,
            season,
            game_date,
            home_team_id,
            away_team_id,
            home_team_abbr,
            away_team_abbr,
            home_score,
            away_score,
            home_win
        FROM core.games
        WHERE home_win IS NOT NULL;

        CREATE INDEX idx_game_labels_date ON game_labels (game_date);
        """,
    )


def build_team_game_results(conn: sqlite3.Connection) -> None:
    execute_script(
        conn,
        "team_game_results",
        """
        DROP TABLE IF EXISTS team_game_results;

        CREATE TABLE team_game_results (
            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            team_id TEXT NOT NULL,
            opponent_team_id TEXT NOT NULL,
            is_home INTEGER NOT NULL,
            team_score INTEGER NOT NULL,
            opponent_score INTEGER NOT NULL,
            win INTEGER NOT NULL,
            point_diff INTEGER NOT NULL,
            game_order INTEGER NOT NULL,
            PRIMARY KEY (game_id, team_id)
        );

        INSERT INTO team_game_results
        SELECT
            game_id,
            season,
            game_date,
            home_team_id AS team_id,
            away_team_id AS opponent_team_id,
            1 AS is_home,
            home_score AS team_score,
            away_score AS opponent_score,
            home_win AS win,
            home_score - away_score AS point_diff,
            ROW_NUMBER() OVER (
                PARTITION BY home_team_id
                ORDER BY game_date, game_id
            ) AS game_order
        FROM game_labels
        UNION ALL
        SELECT
            game_id,
            season,
            game_date,
            away_team_id AS team_id,
            home_team_id AS opponent_team_id,
            0 AS is_home,
            away_score AS team_score,
            home_score AS opponent_score,
            1 - home_win AS win,
            away_score - home_score AS point_diff,
            ROW_NUMBER() OVER (
                PARTITION BY away_team_id
                ORDER BY game_date, game_id
            ) AS game_order
        FROM game_labels;

        CREATE INDEX idx_team_game_results_team_order
        ON team_game_results (team_id, game_order);

        CREATE INDEX idx_team_game_results_game
        ON team_game_results (game_id);
        """,
    )


def build_team_pregame_features(conn: sqlite3.Connection) -> None:
    execute_script(
        conn,
        "team_pregame_features",
        """
        DROP TABLE IF EXISTS team_pregame_features;

        CREATE TABLE team_pregame_features (
            game_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            game_date TEXT NOT NULL,
            is_home INTEGER NOT NULL,
            games_played_before INTEGER NOT NULL,
            win_pct_before REAL,
            avg_point_diff_before REAL,
            avg_points_for_before REAL,
            avg_points_against_before REAL,
            win_pct_last_5 REAL,
            avg_point_diff_last_5 REAL,
            avg_points_for_last_5 REAL,
            avg_points_against_last_5 REAL,
            win_pct_last_10 REAL,
            avg_point_diff_last_10 REAL,
            avg_points_for_last_10 REAL,
            avg_points_against_last_10 REAL,
            rest_days INTEGER,
            PRIMARY KEY (game_id, team_id)
        );

        INSERT INTO team_pregame_features
        SELECT
            game_id,
            team_id,
            game_date,
            is_home,
            COUNT(*) OVER previous_all AS games_played_before,
            AVG(win) OVER previous_all AS win_pct_before,
            AVG(point_diff) OVER previous_all AS avg_point_diff_before,
            AVG(team_score) OVER previous_all AS avg_points_for_before,
            AVG(opponent_score) OVER previous_all AS avg_points_against_before,
            AVG(win) OVER previous_5 AS win_pct_last_5,
            AVG(point_diff) OVER previous_5 AS avg_point_diff_last_5,
            AVG(team_score) OVER previous_5 AS avg_points_for_last_5,
            AVG(opponent_score) OVER previous_5 AS avg_points_against_last_5,
            AVG(win) OVER previous_10 AS win_pct_last_10,
            AVG(point_diff) OVER previous_10 AS avg_point_diff_last_10,
            AVG(team_score) OVER previous_10 AS avg_points_for_last_10,
            AVG(opponent_score) OVER previous_10 AS avg_points_against_last_10,
            CAST(
                julianday(game_date) -
                julianday(LAG(game_date) OVER team_order)
                AS INTEGER
            ) AS rest_days
        FROM team_game_results
        WINDOW
            team_order AS (
                PARTITION BY team_id
                ORDER BY game_date, game_id
            ),
            previous_all AS (
                PARTITION BY team_id
                ORDER BY game_date, game_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            previous_5 AS (
                PARTITION BY team_id
                ORDER BY game_date, game_id
                ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
            ),
            previous_10 AS (
                PARTITION BY team_id
                ORDER BY game_date, game_id
                ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
            );

        CREATE INDEX idx_team_pregame_features_game
        ON team_pregame_features (game_id);

        CREATE INDEX idx_team_pregame_features_team
        ON team_pregame_features (team_id);
        """,
    )


def build_model_games(conn: sqlite3.Connection) -> None:
    execute_script(
        conn,
        "model_games",
        """
        DROP TABLE IF EXISTS model_games;

        CREATE TABLE model_games (
            game_id TEXT PRIMARY KEY,
            season INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            label_home_win INTEGER NOT NULL,
            home_games_played_before INTEGER NOT NULL,
            away_games_played_before INTEGER NOT NULL,
            home_win_pct_before REAL,
            away_win_pct_before REAL,
            diff_win_pct_before REAL,
            home_avg_point_diff_before REAL,
            away_avg_point_diff_before REAL,
            diff_avg_point_diff_before REAL,
            home_avg_points_for_before REAL,
            away_avg_points_for_before REAL,
            diff_avg_points_for_before REAL,
            home_avg_points_against_before REAL,
            away_avg_points_against_before REAL,
            diff_avg_points_against_before REAL,
            home_win_pct_last_5 REAL,
            away_win_pct_last_5 REAL,
            diff_win_pct_last_5 REAL,
            home_avg_point_diff_last_5 REAL,
            away_avg_point_diff_last_5 REAL,
            diff_avg_point_diff_last_5 REAL,
            home_avg_points_for_last_5 REAL,
            away_avg_points_for_last_5 REAL,
            diff_avg_points_for_last_5 REAL,
            home_avg_points_against_last_5 REAL,
            away_avg_points_against_last_5 REAL,
            diff_avg_points_against_last_5 REAL,
            home_win_pct_last_10 REAL,
            away_win_pct_last_10 REAL,
            diff_win_pct_last_10 REAL,
            home_avg_point_diff_last_10 REAL,
            away_avg_point_diff_last_10 REAL,
            diff_avg_point_diff_last_10 REAL,
            home_avg_points_for_last_10 REAL,
            away_avg_points_for_last_10 REAL,
            diff_avg_points_for_last_10 REAL,
            home_avg_points_against_last_10 REAL,
            away_avg_points_against_last_10 REAL,
            diff_avg_points_against_last_10 REAL,
            home_rest_days INTEGER,
            away_rest_days INTEGER,
            diff_rest_days INTEGER
        );

        INSERT INTO model_games
        SELECT
            gl.game_id,
            gl.season,
            gl.game_date,
            gl.home_team_id,
            gl.away_team_id,
            gl.home_win AS label_home_win,
            home.games_played_before AS home_games_played_before,
            away.games_played_before AS away_games_played_before,
            home.win_pct_before AS home_win_pct_before,
            away.win_pct_before AS away_win_pct_before,
            home.win_pct_before - away.win_pct_before AS diff_win_pct_before,
            home.avg_point_diff_before AS home_avg_point_diff_before,
            away.avg_point_diff_before AS away_avg_point_diff_before,
            home.avg_point_diff_before - away.avg_point_diff_before AS diff_avg_point_diff_before,
            home.avg_points_for_before AS home_avg_points_for_before,
            away.avg_points_for_before AS away_avg_points_for_before,
            home.avg_points_for_before - away.avg_points_for_before AS diff_avg_points_for_before,
            home.avg_points_against_before AS home_avg_points_against_before,
            away.avg_points_against_before AS away_avg_points_against_before,
            home.avg_points_against_before - away.avg_points_against_before AS diff_avg_points_against_before,
            home.win_pct_last_5 AS home_win_pct_last_5,
            away.win_pct_last_5 AS away_win_pct_last_5,
            home.win_pct_last_5 - away.win_pct_last_5 AS diff_win_pct_last_5,
            home.avg_point_diff_last_5 AS home_avg_point_diff_last_5,
            away.avg_point_diff_last_5 AS away_avg_point_diff_last_5,
            home.avg_point_diff_last_5 - away.avg_point_diff_last_5 AS diff_avg_point_diff_last_5,
            home.avg_points_for_last_5 AS home_avg_points_for_last_5,
            away.avg_points_for_last_5 AS away_avg_points_for_last_5,
            home.avg_points_for_last_5 - away.avg_points_for_last_5 AS diff_avg_points_for_last_5,
            home.avg_points_against_last_5 AS home_avg_points_against_last_5,
            away.avg_points_against_last_5 AS away_avg_points_against_last_5,
            home.avg_points_against_last_5 - away.avg_points_against_last_5 AS diff_avg_points_against_last_5,
            home.win_pct_last_10 AS home_win_pct_last_10,
            away.win_pct_last_10 AS away_win_pct_last_10,
            home.win_pct_last_10 - away.win_pct_last_10 AS diff_win_pct_last_10,
            home.avg_point_diff_last_10 AS home_avg_point_diff_last_10,
            away.avg_point_diff_last_10 AS away_avg_point_diff_last_10,
            home.avg_point_diff_last_10 - away.avg_point_diff_last_10 AS diff_avg_point_diff_last_10,
            home.avg_points_for_last_10 AS home_avg_points_for_last_10,
            away.avg_points_for_last_10 AS away_avg_points_for_last_10,
            home.avg_points_for_last_10 - away.avg_points_for_last_10 AS diff_avg_points_for_last_10,
            home.avg_points_against_last_10 AS home_avg_points_against_last_10,
            away.avg_points_against_last_10 AS away_avg_points_against_last_10,
            home.avg_points_against_last_10 - away.avg_points_against_last_10 AS diff_avg_points_against_last_10,
            home.rest_days AS home_rest_days,
            away.rest_days AS away_rest_days,
            home.rest_days - away.rest_days AS diff_rest_days
        FROM game_labels gl
        JOIN team_pregame_features home
          ON home.game_id = gl.game_id
         AND home.team_id = gl.home_team_id
        JOIN team_pregame_features away
          ON away.game_id = gl.game_id
         AND away.team_id = gl.away_team_id;
        """,
    )


def build_features(conn: sqlite3.Connection) -> None:
    build_metadata(conn)
    build_labels(conn)
    build_team_game_results(conn)
    build_team_pregame_features(conn)
    build_model_games(conn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-facing feature tables.")
    parser.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(args.output) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("ATTACH DATABASE ? AS core", (str(args.core_db),))
        build_features(conn)
        conn.commit()

    print(f"Features SQLite database: {args.output}")


if __name__ == "__main__":
    main()
