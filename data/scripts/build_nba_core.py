#!/usr/bin/env python3
"""Build typed core NBA tables from raw nba_data imports."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"

DEFAULT_RAW_DB = ARTIFACT_ROOT / "nba_raw.sqlite"
DEFAULT_ODDS_DB = ARTIFACT_ROOT / "oddsportal_moneyline.sqlite"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "nba_core.sqlite"


ODDSPORTAL_TEAM_TRICODES = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def source_table(source: str, season: int) -> str:
    return f"{source}_{season}"


def qname(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


def require_tables(conn: sqlite3.Connection, tables: list[str]) -> None:
    existing = {
        row[0]
        for row in conn.execute(
            "select name from raw.sqlite_master where type = 'table'"
        ).fetchall()
    }
    missing = sorted(set(tables) - existing)
    if missing:
        raise RuntimeError(f"Missing raw tables in attached raw DB: {', '.join(missing)}")


def require_odds_tables(conn: sqlite3.Connection) -> None:
    exists = conn.execute(
        """
        SELECT 1
        FROM odds.sqlite_master
        WHERE type = 'table'
          AND name = 'moneyline_odds'
        """
    ).fetchone()
    if exists is None:
        raise RuntimeError("Missing moneyline_odds table in attached odds DB.")


def create_core_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS games;
        DROP TABLE IF EXISTS teams;
        DROP TABLE IF EXISTS players;
        DROP TABLE IF EXISTS game_events;
        DROP TABLE IF EXISTS shots;
        DROP TABLE IF EXISTS player_matchups;
        DROP TABLE IF EXISTS game_moneyline_odds;

        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            season INTEGER NOT NULL,
            game_date TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            home_team_abbr TEXT,
            away_team_abbr TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_win INTEGER
        );

        CREATE TABLE teams (
            team_id TEXT PRIMARY KEY,
            team_tricode TEXT,
            team_city TEXT,
            team_name TEXT,
            team_slug TEXT
        );

        CREATE TABLE players (
            player_id TEXT PRIMARY KEY,
            player_name TEXT,
            player_name_i TEXT
        );

        CREATE TABLE game_events (
            game_id TEXT NOT NULL,
            action_number INTEGER NOT NULL,
            clock TEXT,
            time_actual TEXT,
            period INTEGER,
            period_type TEXT,
            action_type TEXT,
            sub_type TEXT,
            team_id TEXT,
            team_tricode TEXT,
            player_id TEXT,
            player_name TEXT,
            possession_team_id TEXT,
            score_home INTEGER,
            score_away INTEGER,
            is_field_goal INTEGER,
            x REAL,
            y REAL,
            x_legacy INTEGER,
            y_legacy INTEGER,
            shot_distance REAL,
            shot_result TEXT,
            description TEXT
        );

        CREATE TABLE shots (
            game_id TEXT NOT NULL,
            event_number INTEGER NOT NULL,
            player_id TEXT,
            player_name TEXT,
            team_id TEXT,
            team_name TEXT,
            period INTEGER,
            minutes_remaining INTEGER,
            seconds_remaining INTEGER,
            period_seconds_remaining INTEGER,
            event_type TEXT,
            action_type TEXT,
            shot_type TEXT,
            shot_zone_basic TEXT,
            shot_zone_area TEXT,
            shot_zone_range TEXT,
            shot_distance INTEGER,
            loc_x INTEGER,
            loc_y INTEGER,
            shot_attempted INTEGER,
            shot_made INTEGER,
            game_date TEXT,
            home_team_abbr TEXT,
            away_team_abbr TEXT
        );

        CREATE TABLE player_matchups (
            game_id TEXT NOT NULL,
            away_team_id TEXT,
            home_team_id TEXT,
            team_id TEXT,
            team_tricode TEXT,
            player_id TEXT,
            player_name_i TEXT,
            position TEXT,
            defender_player_id TEXT,
            defender_player_name_i TEXT,
            matchup_minutes TEXT,
            matchup_seconds REAL,
            partial_possessions REAL,
            pct_defender_total_time REAL,
            pct_offensive_total_time REAL,
            pct_total_time_both_on REAL,
            switches_on INTEGER,
            player_points INTEGER,
            team_points INTEGER,
            matchup_assists INTEGER,
            matchup_potential_assists INTEGER,
            matchup_turnovers INTEGER,
            matchup_blocks INTEGER,
            matchup_fgm INTEGER,
            matchup_fga INTEGER,
            matchup_fg_pct REAL,
            matchup_3pm INTEGER,
            matchup_3pa INTEGER,
            matchup_3p_pct REAL,
            matchup_ftm INTEGER,
            matchup_fta INTEGER,
            shooting_fouls INTEGER
        );

        CREATE TABLE game_moneyline_odds (
            game_id TEXT PRIMARY KEY,
            oddsportal_event_id TEXT NOT NULL,
            oddsportal_encoded_event_id TEXT,
            home_avg_decimal_odds REAL,
            away_avg_decimal_odds REAL,
            home_max_decimal_odds REAL,
            away_max_decimal_odds REAL,
            home_avg_american_odds INTEGER,
            away_avg_american_odds INTEGER,
            home_implied_prob_raw REAL,
            away_implied_prob_raw REAL,
            market_overround REAL,
            home_implied_prob_normalized REAL,
            away_implied_prob_normalized REAL,
            home_max_provider_id INTEGER,
            away_max_provider_id INTEGER,
            has_moneyline_odds INTEGER NOT NULL,
            source_event_url TEXT,
            FOREIGN KEY (game_id) REFERENCES games (game_id)
        );
        """
    )


def insert_season(conn: sqlite3.Connection, season: int) -> None:
    cdn = qname("raw", source_table("cdnnba", season))
    shots_raw = qname("raw", source_table("shotdetail", season))
    matchups_raw = qname("raw", source_table("matchups", season))

    conn.execute(
        f"""
        INSERT INTO games
        WITH shot_games AS (
            SELECT
                GAME_ID AS game_id,
                MIN(GAME_DATE) AS game_date_raw,
                MIN(HTM) AS home_team_abbr,
                MIN(VTM) AS away_team_abbr
            FROM {shots_raw}
            GROUP BY GAME_ID
        ),
        matchup_games AS (
            SELECT
                game_id,
                MIN(home_team_id) AS home_team_id,
                MIN(away_team_id) AS away_team_id
            FROM {matchups_raw}
            GROUP BY game_id
        ),
        team_lookup AS (
            SELECT
                team_tricode,
                MIN(team_id) AS team_id
            FROM {matchups_raw}
            WHERE team_tricode <> ''
              AND team_id <> ''
            GROUP BY team_tricode
        ),
        final_scores AS (
            SELECT
                gameId AS game_id,
                MAX(CAST(scoreHome AS INTEGER)) AS home_score,
                MAX(CAST(scoreAway AS INTEGER)) AS away_score
            FROM {cdn}
            WHERE actionType = 'game'
              AND description = 'Game End'
            GROUP BY gameId
        )
        SELECT
            sg.game_id,
            {season} AS season,
            substr(sg.game_date_raw, 1, 4) || '-' ||
                substr(sg.game_date_raw, 5, 2) || '-' ||
                substr(sg.game_date_raw, 7, 2) AS game_date,
            COALESCE(mg.home_team_id, home_lookup.team_id) AS home_team_id,
            COALESCE(mg.away_team_id, away_lookup.team_id) AS away_team_id,
            sg.home_team_abbr,
            sg.away_team_abbr,
            fs.home_score,
            fs.away_score,
            CASE
                WHEN fs.home_score IS NULL OR fs.away_score IS NULL THEN NULL
                WHEN fs.home_score > fs.away_score THEN 1
                ELSE 0
            END AS home_win
        FROM shot_games sg
        LEFT JOIN matchup_games mg ON mg.game_id = sg.game_id
        LEFT JOIN team_lookup home_lookup ON home_lookup.team_tricode = sg.home_team_abbr
        LEFT JOIN team_lookup away_lookup ON away_lookup.team_tricode = sg.away_team_abbr
        LEFT JOIN final_scores fs ON fs.game_id = sg.game_id
        """
    )

    conn.execute(
        f"""
        INSERT INTO game_events
        SELECT
            gameId AS game_id,
            CAST(actionNumber AS INTEGER) AS action_number,
            CASE WHEN clock = '' THEN NULL ELSE clock END AS clock,
            CASE WHEN timeActual = '' THEN NULL ELSE timeActual END AS time_actual,
            CAST(period AS INTEGER) AS period,
            CASE WHEN periodType = '' THEN NULL ELSE periodType END AS period_type,
            CASE WHEN actionType = '' THEN NULL ELSE actionType END AS action_type,
            CASE WHEN subType = '' THEN NULL ELSE subType END AS sub_type,
            CASE WHEN teamId = '' THEN NULL ELSE teamId END AS team_id,
            CASE WHEN teamTricode = '' THEN NULL ELSE teamTricode END AS team_tricode,
            CASE WHEN personId = '' OR personId = '0' THEN NULL ELSE personId END AS player_id,
            CASE WHEN playerName = '' THEN NULL ELSE playerName END AS player_name,
            CASE WHEN possession = '' OR possession = '0' THEN NULL ELSE possession END AS possession_team_id,
            CASE WHEN scoreHome = '' THEN NULL ELSE CAST(scoreHome AS INTEGER) END AS score_home,
            CASE WHEN scoreAway = '' THEN NULL ELSE CAST(scoreAway AS INTEGER) END AS score_away,
            CASE WHEN isFieldGoal = '' THEN NULL ELSE CAST(isFieldGoal AS INTEGER) END AS is_field_goal,
            CASE WHEN x = '' THEN NULL ELSE CAST(x AS REAL) END AS x,
            CASE WHEN y = '' THEN NULL ELSE CAST(y AS REAL) END AS y,
            CASE WHEN xLegacy = '' THEN NULL ELSE CAST(xLegacy AS INTEGER) END AS x_legacy,
            CASE WHEN yLegacy = '' THEN NULL ELSE CAST(yLegacy AS INTEGER) END AS y_legacy,
            CASE WHEN shotDistance = '' THEN NULL ELSE CAST(shotDistance AS REAL) END AS shot_distance,
            CASE WHEN shotResult = '' THEN NULL ELSE shotResult END AS shot_result,
            CASE WHEN description = '' THEN NULL ELSE description END AS description
        FROM {cdn}
        """
    )

    conn.execute(
        f"""
        INSERT INTO shots
        SELECT
            GAME_ID AS game_id,
            CAST(GAME_EVENT_ID AS INTEGER) AS event_number,
            PLAYER_ID AS player_id,
            PLAYER_NAME AS player_name,
            TEAM_ID AS team_id,
            TEAM_NAME AS team_name,
            CAST(PERIOD AS INTEGER) AS period,
            CAST(MINUTES_REMAINING AS INTEGER) AS minutes_remaining,
            CAST(SECONDS_REMAINING AS INTEGER) AS seconds_remaining,
            CAST(MINUTES_REMAINING AS INTEGER) * 60 +
                CAST(SECONDS_REMAINING AS INTEGER) AS period_seconds_remaining,
            EVENT_TYPE AS event_type,
            ACTION_TYPE AS action_type,
            SHOT_TYPE AS shot_type,
            SHOT_ZONE_BASIC AS shot_zone_basic,
            SHOT_ZONE_AREA AS shot_zone_area,
            SHOT_ZONE_RANGE AS shot_zone_range,
            CAST(SHOT_DISTANCE AS INTEGER) AS shot_distance,
            CAST(LOC_X AS INTEGER) AS loc_x,
            CAST(LOC_Y AS INTEGER) AS loc_y,
            CAST(SHOT_ATTEMPTED_FLAG AS INTEGER) AS shot_attempted,
            CAST(SHOT_MADE_FLAG AS INTEGER) AS shot_made,
            substr(GAME_DATE, 1, 4) || '-' ||
                substr(GAME_DATE, 5, 2) || '-' ||
                substr(GAME_DATE, 7, 2) AS game_date,
            HTM AS home_team_abbr,
            VTM AS away_team_abbr
        FROM {shots_raw}
        """
    )

    conn.execute(
        f"""
        INSERT INTO player_matchups
        SELECT
            game_id,
            away_team_id,
            home_team_id,
            team_id,
            team_tricode,
            person_id AS player_id,
            name_i AS player_name_i,
            position,
            matchups_person_id AS defender_player_id,
            matchups_name_i AS defender_player_name_i,
            matchup_minutes,
            CAST(matchup_minutes_sort AS REAL) AS matchup_seconds,
            CAST(partial_possessions AS REAL) AS partial_possessions,
            CAST(percentage_defender_total_time AS REAL) AS pct_defender_total_time,
            CAST(percentage_offensive_total_time AS REAL) AS pct_offensive_total_time,
            CAST(percentage_total_time_both_on AS REAL) AS pct_total_time_both_on,
            CAST(switches_on AS INTEGER) AS switches_on,
            CAST(player_points AS INTEGER) AS player_points,
            CAST(team_points AS INTEGER) AS team_points,
            CAST(matchup_assists AS INTEGER) AS matchup_assists,
            CAST(matchup_potential_assists AS INTEGER) AS matchup_potential_assists,
            CAST(matchup_turnovers AS INTEGER) AS matchup_turnovers,
            CAST(matchup_blocks AS INTEGER) AS matchup_blocks,
            CAST(matchup_field_goals_made AS INTEGER) AS matchup_fgm,
            CAST(matchup_field_goals_attempted AS INTEGER) AS matchup_fga,
            CAST(matchup_field_goals_percentage AS REAL) AS matchup_fg_pct,
            CAST(matchup_three_pointers_made AS INTEGER) AS matchup_3pm,
            CAST(matchup_three_pointers_attempted AS INTEGER) AS matchup_3pa,
            CAST(matchup_three_pointers_percentage AS REAL) AS matchup_3p_pct,
            CAST(matchup_free_throws_made AS INTEGER) AS matchup_ftm,
            CAST(matchup_free_throws_attempted AS INTEGER) AS matchup_fta,
            CAST(shooting_fouls AS INTEGER) AS shooting_fouls
        FROM {matchups_raw}
        """
    )


def insert_dimensions(conn: sqlite3.Connection, seasons: list[int]) -> None:
    matchup_unions = "\nUNION ALL\n".join(
        f"""
        SELECT
            team_id,
            team_tricode,
            team_city,
            team_name,
            team_slug,
            person_id,
            name_i,
            first_name,
            family_name,
            matchups_person_id,
            matchups_name_i,
            matchups_first_name,
            matchups_family_name
        FROM {qname('raw', source_table('matchups', season))}
        """
        for season in seasons
    )
    cdn_unions = "\nUNION ALL\n".join(
        f"""
        SELECT
            personId,
            playerNameI,
            playerName
        FROM {qname('raw', source_table('cdnnba', season))}
        """
        for season in seasons
    )
    shot_unions = "\nUNION ALL\n".join(
        f"""
        SELECT
            PLAYER_ID,
            PLAYER_NAME
        FROM {qname('raw', source_table('shotdetail', season))}
        """
        for season in seasons
    )

    conn.execute(
        f"""
        INSERT INTO teams
        WITH all_matchups AS (
            {matchup_unions}
        )
        SELECT
            team_id,
            MIN(team_tricode) AS team_tricode,
            MIN(team_city) AS team_city,
            MIN(team_name) AS team_name,
            MIN(team_slug) AS team_slug
        FROM all_matchups
        WHERE team_id <> ''
        GROUP BY team_id
        """
    )

    conn.execute(
        f"""
        INSERT INTO players
        WITH all_matchups AS (
            {matchup_unions}
        ),
        all_cdn AS (
            {cdn_unions}
        ),
        all_shots AS (
            {shot_unions}
        ),
        raw_players AS (
            SELECT person_id AS player_id, name_i AS player_name_i,
                   first_name || ' ' || family_name AS player_name
            FROM all_matchups
            WHERE person_id <> ''
            UNION
            SELECT matchups_person_id AS player_id, matchups_name_i AS player_name_i,
                   matchups_first_name || ' ' || matchups_family_name AS player_name
            FROM all_matchups
            WHERE matchups_person_id <> ''
            UNION
            SELECT personId AS player_id, playerNameI AS player_name_i, playerName AS player_name
            FROM all_cdn
            WHERE personId <> '' AND personId <> '0'
            UNION
            SELECT PLAYER_ID AS player_id, NULL AS player_name_i, PLAYER_NAME AS player_name
            FROM all_shots
            WHERE PLAYER_ID <> ''
        )
        SELECT
            player_id,
            MAX(player_name) AS player_name,
            MAX(player_name_i) AS player_name_i
        FROM raw_players
        GROUP BY player_id
        """
    )


def insert_moneyline_odds(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS temp.oddsportal_team_map")
    conn.execute(
        """
        CREATE TEMP TABLE oddsportal_team_map (
            oddsportal_name TEXT PRIMARY KEY,
            team_tricode TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO oddsportal_team_map (oddsportal_name, team_tricode)
        VALUES (?, ?)
        """,
        sorted(ODDSPORTAL_TEAM_TRICODES.items()),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO game_moneyline_odds
        SELECT
            g.game_id,
            o.event_id AS oddsportal_event_id,
            o.encoded_event_id AS oddsportal_encoded_event_id,
            o.home_avg_decimal_odds,
            o.away_avg_decimal_odds,
            o.home_max_decimal_odds,
            o.away_max_decimal_odds,
            o.home_avg_american_odds,
            o.away_avg_american_odds,
            o.home_implied_prob_raw,
            o.away_implied_prob_raw,
            o.market_overround,
            o.home_implied_prob_normalized,
            o.away_implied_prob_normalized,
            o.home_max_provider_id,
            o.away_max_provider_id,
            o.has_moneyline_odds,
            o.event_url AS source_event_url
        FROM games g
        JOIN odds.moneyline_odds o
          ON o.season_start_year = g.season
         AND o.game_date_et = g.game_date
         AND o.home_result = g.home_score
         AND o.away_result = g.away_score
        JOIN oddsportal_team_map home_map
          ON home_map.oddsportal_name = o.home_name
         AND home_map.team_tricode = g.home_team_abbr
        JOIN oddsportal_team_map away_map
          ON away_map.oddsportal_name = o.away_name
         AND away_map.team_tricode = g.away_team_abbr
        WHERE o.status_id IN (3, 10)
        """
    )


def create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_games_date ON games (game_date);
        CREATE INDEX idx_teams_tricode ON teams (team_tricode);
        CREATE UNIQUE INDEX idx_game_events_game_action ON game_events (game_id, action_number);
        CREATE INDEX idx_game_events_player ON game_events (player_id);
        CREATE INDEX idx_game_events_team ON game_events (team_id);
        CREATE INDEX idx_shots_game_event ON shots (game_id, event_number);
        CREATE INDEX idx_shots_player ON shots (player_id);
        CREATE INDEX idx_shots_team ON shots (team_id);
        CREATE INDEX idx_player_matchups_game ON player_matchups (game_id);
        CREATE INDEX idx_player_matchups_player ON player_matchups (player_id);
        CREATE INDEX idx_player_matchups_defender ON player_matchups (defender_player_id);
        CREATE INDEX idx_game_moneyline_has_odds ON game_moneyline_odds (has_moneyline_odds);
        """
    )


def build_core(conn: sqlite3.Connection, seasons: list[int], include_odds: bool = False) -> None:
    create_core_tables(conn)
    for season in seasons:
        insert_season(conn, season)
    insert_dimensions(conn, seasons)
    if include_odds:
        insert_moneyline_odds(conn)
    create_indexes(conn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build core NBA tables from nba_raw.sqlite.")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Season start years, e.g. 2020 2021 2022 2023 2024 2025.",
    )
    parser.add_argument("--raw-db", type=Path, default=DEFAULT_RAW_DB)
    parser.add_argument("--odds-db", type=Path, default=DEFAULT_ODDS_DB)
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Do not import OddsPortal moneyline odds into the core database.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = sorted(set(args.seasons))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    required = []
    for season in seasons:
        required.extend(
            [
                source_table("cdnnba", season),
                source_table("shotdetail", season),
                source_table("matchups", season),
            ]
        )

    with sqlite3.connect(args.output) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("ATTACH DATABASE ? AS raw", (str(args.raw_db),))
        require_tables(conn, required)
        include_odds = False
        if not args.skip_odds:
            if args.odds_db.exists():
                conn.execute("ATTACH DATABASE ? AS odds", (str(args.odds_db),))
                require_odds_tables(conn)
                include_odds = True
            else:
                print(f"Odds SQLite database not found, skipping odds import: {args.odds_db}")
        build_core(conn, seasons, include_odds=include_odds)
        conn.commit()

    print(f"Core SQLite database: {args.output}")


if __name__ == "__main__":
    main()
