#!/usr/bin/env python3
"""Materialize `game_player_availability`: one row per (game_id, player_id) with
the player's status as of the latest injury snapshot whose timestamp is at or
before tip-off.

For each game G:
  1. Find injury_snapshots_resolved rows with game_date == G.date and
     matchup == "{away}@{home}".
  2. Compute tip_off_dt from game_date + game_time. Times are 12-hour, all PM:
     "07:30 (ET)" -> 19:30, "12:30 (ET)" -> 12:30.
  3. Keep snapshots with snapshot_ts <= tip_off_dt.
  4. Pick the latest such snapshot. Output its rows for both teams.

Players not on the injury report are implicitly Available — the table only
contains explicit Out / Questionable / Doubtful / Probable / Available rows.

Output table is in nba_injury_history.sqlite alongside the resolved snapshots.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_INJURY_DB = ARTIFACT_ROOT / "nba_injury_history.sqlite"
DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"

GAME_TIME_RE = re.compile(r"^(\d{2}):(\d{2})\s*\(ET\)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    return p.parse_args()


def parse_tipoff(game_date: str, game_time: str) -> pd.Timestamp | None:
    """game_date 'MM/DD/YYYY' + game_time 'HH:MM (ET)' -> timestamp.

    All times are PM (12 stays as noon, 1-11 -> 13-23). NBA tip-offs span
    noon to 11pm ET; an AM time would be invalid data.
    """
    if not game_time:
        return None
    m = GAME_TIME_RE.match(game_time.strip())
    if not m:
        return None
    h, mn = int(m.group(1)), int(m.group(2))
    if not (1 <= h <= 12) or not (0 <= mn <= 59):
        return None
    if h != 12:
        h += 12
    try:
        dt = datetime.strptime(game_date, "%m/%d/%Y").replace(hour=h, minute=mn)
    except ValueError:
        return None
    return pd.Timestamp(dt)


def main() -> None:
    args = parse_args()

    with sqlite3.connect(args.core_db) as conn:
        games = pd.read_sql_query(
            "SELECT game_id, game_date, home_team_abbr, away_team_abbr "
            "FROM games WHERE game_date IS NOT NULL",
            conn,
        )
    games["matchup"] = games["away_team_abbr"] + "@" + games["home_team_abbr"]
    games["game_date_iso"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")

    with sqlite3.connect(args.injury_db) as conn:
        inj = pd.read_sql_query(
            "SELECT snapshot_ts, game_date, game_time, matchup, team_id, "
            "player_id, status, reason "
            "FROM injury_snapshots_resolved "
            "WHERE player_id IS NOT NULL AND matchup IS NOT NULL AND game_time IS NOT NULL",
            conn,
        )

    inj["game_date_iso"] = pd.to_datetime(
        inj["game_date"], format="%m/%d/%Y", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    inj["tip_off"] = inj.apply(
        lambda r: parse_tipoff(r["game_date"], r["game_time"]), axis=1
    )
    inj["snapshot_dt"] = pd.to_datetime(inj["snapshot_ts"], errors="coerce")
    n_inj_input = len(inj)
    inj = inj.dropna(subset=["game_date_iso", "tip_off", "snapshot_dt"])
    n_inj_after_parse = len(inj)

    merged = inj.merge(
        games[["game_id", "game_date_iso", "matchup"]],
        on=["game_date_iso", "matchup"],
        how="inner",
    )
    n_merged = len(merged)

    valid = merged[merged["snapshot_dt"] <= merged["tip_off"]].copy()
    n_valid = len(valid)

    latest = valid.groupby("game_id")["snapshot_ts"].transform("max")
    chosen = valid[valid["snapshot_ts"] == latest].copy()
    chosen["minutes_to_tip"] = (
        (chosen["tip_off"] - chosen["snapshot_dt"]).dt.total_seconds() / 60.0
    )

    out = chosen[
        [
            "game_id",
            "team_id",
            "player_id",
            "status",
            "reason",
            "snapshot_ts",
            "minutes_to_tip",
        ]
    ].drop_duplicates(subset=["game_id", "player_id"])

    with sqlite3.connect(args.injury_db) as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS game_player_availability;
            CREATE TABLE game_player_availability (
                game_id TEXT NOT NULL,
                team_id TEXT NOT NULL,
                player_id TEXT NOT NULL,
                status TEXT,
                reason TEXT,
                snapshot_ts TEXT NOT NULL,
                minutes_to_tip REAL,
                PRIMARY KEY (game_id, player_id)
            );
            CREATE INDEX idx_avail_game ON game_player_availability(game_id);
            CREATE INDEX idx_avail_player ON game_player_availability(player_id);
            CREATE INDEX idx_avail_team ON game_player_availability(team_id);
            """
        )
        out.to_sql(
            "game_player_availability", conn, if_exists="append", index=False
        )

        n_games_total = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM games WHERE game_date IS NOT NULL",
            sqlite3.connect(args.core_db),
        )["n"].iloc[0]

        covered = pd.read_sql_query(
            "SELECT COUNT(DISTINCT game_id) AS n FROM game_player_availability",
            conn,
        )["n"].iloc[0]

        status_counts = pd.read_sql_query(
            "SELECT status, COUNT(*) AS n FROM game_player_availability "
            "GROUP BY status ORDER BY n DESC",
            conn,
        )

    print(f"[in]      injury_rows={n_inj_input} after_parse={n_inj_after_parse}")
    print(f"[merge]   matched_with_games={n_merged}")
    print(f"[filter]  valid_pre_tip={n_valid}")
    print(f"[out]     rows={len(out)}")
    print(f"[cover]   distinct_games={covered}/{n_games_total}")
    print("[status]")
    print(status_counts.to_string(index=False))


if __name__ == "__main__":
    main()
