#!/usr/bin/env python3
"""Resolve `injury_snapshots.player_name` -> `players.player_id`.

Approach
--------
- Drop rows with empty player_name or team='Non-NBA Team'.
- Map team string -> team_id by exact match on `team_city || ' ' || team_name`.
- Build a *normalized* match key for each player: (first_initial, last_part)
  with diacritics stripped, casefolded, and trailing suffix (Jr./Sr./II/III/...)
  optionally removed. This handles:
    * "Čančar" / "Cancar" diacritic differences
    * Stephen Curry's `St. Curry` 2-letter initial (we take only the first char)
    * "Dowtin, Jeff" injury vs `Jeff Dowtin Jr.` player table
- Scope match by (team_id, season). Season derived from snapshot_ts:
  month >= 10 -> year, else year-1.
- Roster per (team, season): join `player_matchups` to `games`, then look up
  the canonical row in `players` by player_id.
- Disambiguation:
    1. If exactly 1 candidate at (team_id, season) for the key -> match.
    2. If multiple candidates, narrow by full first-name match against
       `players.player_name` (e.g. Stephen Curry vs Seth Curry on GSW).
    3. Fallback: global (across all players) with the same disambiguation.
    4. If still ambiguous -> emit NULL with `match_source = 'collision'`.

Output
------
Table `injury_snapshots_resolved` in nba_injury_history.sqlite. Columns:
original snapshot fields + team_id, season, player_id, match_source.
match_source values: 'team_season' / 'global' / NULL.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_INJURY_DB = ARTIFACT_ROOT / "nba_injury_history.sqlite"
DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"

SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$")
NAME_I_RE = re.compile(r"^([A-Za-z]+)\.?\s*(.*)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    return p.parse_args()


def strip_diacritics(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_last(s: str, *, strip_suffix: bool = False) -> str:
    s = strip_diacritics(s).casefold().strip()
    if strip_suffix:
        s = SUFFIX_RE.sub("", s)
    return s


def parse_injury_name(raw: str) -> tuple[str, str] | None:
    """`"Bagley III, Marvin"` -> ('M', 'Bagley III'). None on malformed."""
    if not raw or "," not in raw:
        return None
    last_part, _, first_part = raw.partition(", ")
    last_part = last_part.strip()
    first_part = first_part.strip()
    if not last_part or not first_part:
        return None
    return first_part[0].upper(), last_part


def parse_name_i(name_i: str) -> tuple[str, str] | None:
    """`'St. Curry'` -> ('S', 'Curry'). `'Yang'` -> None (no period)."""
    if not name_i:
        return None
    m = NAME_I_RE.match(name_i.strip())
    if not m:
        return None
    initial_raw, last = m.group(1), m.group(2).strip()
    if not initial_raw or not last:
        return None
    return initial_raw[0].upper(), last


def first_token_norm(player_name: str) -> str | None:
    """`'Stephen Curry'` -> 'stephen'. None if no space."""
    if not player_name or " " not in player_name:
        return None
    return strip_diacritics(player_name.split(" ", 1)[0]).casefold()


def season_for_iso_ts(ts: str) -> int | None:
    if not ts:
        return None
    try:
        year = int(ts[:4])
        month = int(ts[5:7])
    except ValueError:
        return None
    return year if month >= 10 else year - 1


def load_team_string_to_id(core_db: Path) -> dict[str, str]:
    with sqlite3.connect(core_db) as conn:
        df = pd.read_sql_query(
            "SELECT team_id, team_city || ' ' || team_name AS full_name FROM teams",
            conn,
        )
    return {row.full_name: str(row.team_id) for row in df.itertuples(index=False)}


def load_players(core_db: Path) -> dict[str, tuple[str | None, str | None]]:
    """Return `{player_id: (player_name, player_name_i)}`."""
    with sqlite3.connect(core_db) as conn:
        df = pd.read_sql_query(
            "SELECT player_id, player_name, player_name_i FROM players", conn
        )
    out: dict[str, tuple[str | None, str | None]] = {}
    for row in df.itertuples(index=False):
        out[str(row.player_id)] = (
            row.player_name if pd.notna(row.player_name) else None,
            row.player_name_i if pd.notna(row.player_name_i) else None,
        )
    return out


def load_team_season_player_ids(core_db: Path) -> dict[tuple[str, int], set[str]]:
    """Return `{(team_id, season): {player_id, ...}}` from player_matchups+games."""
    query = """
        SELECT DISTINCT pm.team_id, g.season, pm.player_id
        FROM player_matchups pm
        JOIN games g ON g.game_id = pm.game_id
        WHERE pm.team_id IS NOT NULL AND pm.player_id IS NOT NULL
    """
    with sqlite3.connect(core_db) as conn:
        df = pd.read_sql_query(query, conn)
    out: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in df.itertuples(index=False):
        out[(str(row.team_id), int(row.season))].add(str(row.player_id))
    return out


def build_player_keys(
    players: dict[str, tuple[str | None, str | None]],
) -> tuple[
    dict[tuple[str, str], list[str]],
    dict[tuple[str, str], list[str]],
    dict[str, str | None],
]:
    """Return three indexes:
    - by_key_strict: (initial, last_norm) -> [player_id, ...]
    - by_key_loose:  (initial, last_norm_no_suffix) -> [player_id, ...]
    - first_token:   player_id -> normalized first-name token (or None)
    """
    by_key_strict: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_key_loose: dict[tuple[str, str], list[str]] = defaultdict(list)
    first_token: dict[str, str | None] = {}
    for pid, (player_name, name_i) in players.items():
        first_token[pid] = first_token_norm(player_name) if player_name else None
        if not name_i:
            continue
        parsed = parse_name_i(name_i)
        if not parsed:
            continue
        initial, last = parsed
        by_key_strict[(initial, normalize_last(last))].append(pid)
        by_key_loose[(initial, normalize_last(last, strip_suffix=True))].append(pid)
    return dict(by_key_strict), dict(by_key_loose), first_token


def disambiguate(
    candidates: list[str],
    full_first_norm: str,
    first_token: dict[str, str | None],
) -> str | None:
    """Reduce candidate list to 1 by full first-name match. None if not unique."""
    if len(candidates) == 1:
        return candidates[0]
    refined = [pid for pid in candidates if first_token.get(pid) == full_first_norm]
    if len(refined) == 1:
        return refined[0]
    return None


def main() -> None:
    args = parse_args()

    team_to_id = load_team_string_to_id(args.core_db)
    players = load_players(args.core_db)
    team_season_pids = load_team_season_player_ids(args.core_db)
    by_strict, by_loose, first_token = build_player_keys(players)

    with sqlite3.connect(args.injury_db) as inj_conn:
        snapshots = pd.read_sql_query("SELECT * FROM injury_snapshots", inj_conn)

    n_input = len(snapshots)
    snapshots = snapshots[
        snapshots["player_name"].notna()
        & (snapshots["player_name"] != "")
        & (snapshots["team"] != "Non-NBA Team")
    ].reset_index(drop=True)
    n_after_drop = len(snapshots)

    snapshots["team_id"] = snapshots["team"].map(team_to_id)
    snapshots["season"] = snapshots["snapshot_ts"].map(season_for_iso_ts)

    player_ids: list[str | None] = []
    sources: list[str | None] = []
    n_team = 0
    n_global = 0
    n_collision_team = 0
    n_collision_global = 0
    n_no_match = 0
    unmatched: list[tuple[str, str, int | None]] = []

    for row in snapshots.itertuples(index=False):
        team_id = row.team_id
        season = row.season
        if team_id is None or pd.isna(team_id) or season is None or pd.isna(season):
            player_ids.append(None)
            sources.append(None)
            continue
        parsed = parse_injury_name(row.player_name)
        if not parsed:
            player_ids.append(None)
            sources.append(None)
            continue
        initial, last = parsed
        full_first = strip_diacritics(row.player_name.split(", ", 1)[1]).casefold().strip()

        # Build key candidates: strict, then loose (suffix-stripped)
        key_strict = (initial, normalize_last(last))
        key_loose = (initial, normalize_last(last, strip_suffix=True))

        team_pids = team_season_pids.get((str(team_id), int(season)), set())

        def filter_to_team(pids: list[str]) -> list[str]:
            return [p for p in pids if p in team_pids]

        # 1. team-season strict
        team_candidates = filter_to_team(by_strict.get(key_strict, []))
        # 2. team-season loose (only if strict empty)
        if not team_candidates:
            team_candidates = filter_to_team(by_loose.get(key_loose, []))

        if team_candidates:
            chosen = disambiguate(team_candidates, full_first, first_token)
            if chosen:
                player_ids.append(chosen)
                sources.append("team_season")
                n_team += 1
                continue
            n_collision_team += 1

        # 3. global strict
        global_candidates = by_strict.get(key_strict, [])
        if not global_candidates:
            global_candidates = by_loose.get(key_loose, [])

        if global_candidates:
            chosen = disambiguate(global_candidates, full_first, first_token)
            if chosen:
                player_ids.append(chosen)
                sources.append("global")
                n_global += 1
                continue
            n_collision_global += 1
            player_ids.append(None)
            sources.append(None)
            continue

        n_no_match += 1
        if len(unmatched) < 15:
            unmatched.append(
                (row.player_name, row.team, int(season) if pd.notna(season) else None)
            )
        player_ids.append(None)
        sources.append(None)

    snapshots["player_id"] = player_ids
    snapshots["match_source"] = sources

    out = snapshots[
        [
            "snapshot_ts",
            "game_date",
            "game_time",
            "matchup",
            "team",
            "team_id",
            "player_name",
            "player_id",
            "match_source",
            "season",
            "status",
            "reason",
        ]
    ].copy()

    with sqlite3.connect(args.injury_db) as inj_conn:
        inj_conn.executescript(
            """
            DROP TABLE IF EXISTS injury_snapshots_resolved;
            CREATE TABLE injury_snapshots_resolved (
                snapshot_ts TEXT NOT NULL,
                game_date TEXT,
                game_time TEXT,
                matchup TEXT,
                team TEXT,
                team_id TEXT,
                player_name TEXT NOT NULL,
                player_id TEXT,
                match_source TEXT,
                season INTEGER,
                status TEXT,
                reason TEXT
            );
            CREATE INDEX idx_resolved_player ON injury_snapshots_resolved(player_id);
            CREATE INDEX idx_resolved_team_season ON injury_snapshots_resolved(team_id, season);
            CREATE INDEX idx_resolved_game_date ON injury_snapshots_resolved(game_date);
            """
        )
        out.to_sql(
            "injury_snapshots_resolved", inj_conn, if_exists="append", index=False
        )

    n_matched = n_team + n_global
    pct = (n_matched / n_after_drop * 100.0) if n_after_drop else 0.0
    print(f"[in]    rows={n_input}")
    print(f"[clean] rows_after_drop={n_after_drop} (dropped empty/non-nba)")
    print(
        f"[match] team_season={n_team} global={n_global} total={n_matched} "
        f"({pct:.2f}% of clean)"
    )
    print(
        f"[miss]  team_collisions={n_collision_team} global_collisions={n_collision_global} "
        f"no_match={n_no_match}"
    )
    if unmatched:
        print("[examples] first 15 still-unmatched (player_name, team, season):")
        for ex in unmatched:
            print(f"  - {ex}")


if __name__ == "__main__":
    main()
