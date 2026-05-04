"""Pull hoopR-nba-data player_box parquets (one per season) and write a sqlite.

Source: https://github.com/sportsdataverse/hoopR-nba-data
Path:   nba/player_box/parquet/player_box_<season>.parquet
        nba/schedules/parquet/nba_schedule_<season>.parquet

Why we want this:
  hoopR's per-game player rows include DNP-CD vs injury distinction via
  `did_not_play` + `reason`, which the NBA Stats API box-score does not
  expose without per-game scraping. ESPN ids are different from NBA Stats
  ids; we keep the raw rows here and do id-mapping in a downstream step.

Output: data/artifacts/hoopr_player_box.sqlite with two tables:
    player_box(season, espn_game_id, espn_athlete_id, ...)
    schedules(season, espn_game_id, game_date, home_abbr, away_abbr, ...)
"""
from __future__ import annotations

import argparse
import io
import sqlite3
from pathlib import Path

import pandas as pd
import requests

REPO_RAW = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"

PLAYER_BOX_KEEP = [
    "game_id", "season", "season_type", "game_date", "game_date_time",
    "athlete_id", "athlete_display_name", "athlete_short_name",
    "team_id", "team_abbreviation", "team_name", "team_location",
    "minutes", "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "rebounds",
    "assists", "steals", "blocks", "turnovers", "fouls",
    "plus_minus", "points",
    "starter", "ejected", "did_not_play", "reason", "active",
    "home_away",
    "opponent_team_id", "opponent_team_abbreviation",
]

SCHEDULE_KEEP = [
    "game_id", "season", "season_type", "game_date", "game_date_time",
    "home_id", "home_abbreviation", "home_name", "home_score",
    "away_id", "away_abbreviation", "away_name", "away_score",
    "status_type_completed", "status_type_description",
    "neutral_site", "venue_full_name",
]


def fetch_parquet(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2020)
    p.add_argument("--end-season", type=int, default=2026)
    p.add_argument("--output", type=Path,
                   default=Path("data/artifacts/hoopr_player_box.sqlite"))
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    pb_frames: list[pd.DataFrame] = []
    sch_frames: list[pd.DataFrame] = []

    for season in range(args.start_season, args.end_season + 1):
        pb_url = f"{REPO_RAW}/player_box/parquet/player_box_{season}.parquet"
        sch_url = f"{REPO_RAW}/schedules/parquet/nba_schedule_{season}.parquet"
        print(f"[fetch] {season} player_box ...", flush=True)
        pb = fetch_parquet(pb_url)
        pb = pb[[c for c in PLAYER_BOX_KEEP if c in pb.columns]].copy()
        pb["season"] = season
        pb_frames.append(pb)
        print(f"        rows={len(pb)}")
        print(f"[fetch] {season} schedule ...", flush=True)
        sch = fetch_parquet(sch_url)
        sch = sch[[c for c in SCHEDULE_KEEP if c in sch.columns]].copy()
        sch["season"] = season
        sch_frames.append(sch)
        print(f"        rows={len(sch)}")

    player_box = pd.concat(pb_frames, ignore_index=True)
    schedules = pd.concat(sch_frames, ignore_index=True)

    player_box = player_box.rename(columns={
        "game_id": "espn_game_id",
        "athlete_id": "espn_athlete_id",
        "team_id": "espn_team_id",
        "opponent_team_id": "espn_opponent_team_id",
    })
    schedules = schedules.rename(columns={
        "game_id": "espn_game_id",
        "home_id": "espn_home_id",
        "away_id": "espn_away_id",
    })

    print(f"[write] {args.output}")
    with sqlite3.connect(args.output) as conn:
        player_box.to_sql("player_box", conn, index=False)
        schedules.to_sql("schedules", conn, index=False)
        conn.execute("CREATE INDEX idx_pb_game ON player_box(espn_game_id)")
        conn.execute("CREATE INDEX idx_pb_athlete ON player_box(espn_athlete_id)")
        conn.execute("CREATE INDEX idx_pb_date ON player_box(game_date)")
        conn.execute("CREATE INDEX idx_sch_date ON schedules(game_date)")
        conn.execute("CREATE INDEX idx_sch_game ON schedules(espn_game_id)")
    print(f"[done] player_box={len(player_box)} schedule={len(schedules)}")


if __name__ == "__main__":
    main()
