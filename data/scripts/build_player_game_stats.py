#!/usr/bin/env python3
"""Build per-(game, player) box-score-style stats from PBP events.

Derives minutes from substitution events + period boundaries; counts other
stats from action types in `game_events`. Assists are not currently emitted
(would require parsing event descriptions or pulling `assistPersonId` from
the raw `cdnnba_*` archives).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"

DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "player_game_stats.sqlite"

REGULATION_PERIOD_SEC = 720
OT_PERIOD_SEC = 300

# ISO 8601 duration: PT MM M SS.SS S
CLOCK_RE = re.compile(r"^PT(\d+)M([\d.]+)S$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--limit-games", type=int, default=None,
                   help="If set, only process this many games (for testing).")
    p.add_argument("--strict", action="store_true",
                   help="Raise on any per-period minutes sanity-check failure.")
    return p.parse_args()


def parse_clock_to_remaining_sec(clock: str | None) -> float | None:
    """Returns seconds remaining in the period, or None if unparseable."""
    if not clock:
        return None
    m = CLOCK_RE.match(clock)
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = float(m.group(2))
    return minutes * 60.0 + seconds


def period_length_sec(period: int) -> int:
    return REGULATION_PERIOD_SEC if period <= 4 else OT_PERIOD_SEC


def derive_minutes_for_game(
    events: list[tuple],
) -> dict[tuple[str, str], float]:
    """Walk substitutions per period and return {(player_id, team_id): seconds}.

    `events` is the list of all rows for this game, sorted by (period, action_number),
    each row: (period, action_number, action_type, sub_type, player_id, team_id, clock).
    """
    by_period: dict[int, list[tuple]] = defaultdict(list)
    for e in events:
        by_period[e[0]].append(e)

    total_seconds: dict[tuple[str, str], float] = defaultdict(float)
    previous_on_floor: dict[str, str] = {}  # pid -> team_id at end of prev period

    # Pre-pass: P1 starters who had ZERO stat events in P1 (e.g. a screener who
    # came out early without rebounding/shooting/fouling) are invisible to the
    # first-action heuristic. Recover them by scanning the entire game: any
    # player whose first substitution event (chronologically across all
    # periods) is sub=out must have been on the floor at game start.
    sub_events_chrono: list[tuple[int, float, int, tuple]] = []
    for period, period_events in by_period.items():
        plen = period_length_sec(period)
        for e in period_events:
            if e[2] != "substitution":
                continue
            rem = parse_clock_to_remaining_sec(e[6])
            if rem is None:
                continue
            elapsed = float(plen) - rem
            sub_events_chrono.append((period, elapsed, e[1], e))
    sub_events_chrono.sort(key=lambda x: (x[0], x[1], x[2]))

    starter_pre_pass: dict[str, str] = {}  # pid -> team_id (extra P1 starters)
    seen_pid: set[str] = set()
    for _period, _elapsed, _an, e in sub_events_chrono:
        pid, team_id, sub_type = e[4], e[5], e[3]
        if not pid or not team_id or pid == team_id or team_id == "0":
            continue
        if pid in seen_pid:
            continue
        seen_pid.add(pid)
        if sub_type == "out" and _period == 1:
            starter_pre_pass[pid] = team_id

    for period in sorted(by_period.keys()):
        period_events = by_period[period]
        plen = period_length_sec(period)

        # NBA PBP back-dates substitutions to the stoppage clock but emits them
        # at the action_number of when they're processed (often after a timeout
        # or amended event), so action_number is NOT chronological w.r.t. clock.
        # Re-sort by (elapsed_time, action_number) to get true chronological order.
        events_with_elapsed: list[tuple[float, int, tuple]] = []
        for e in period_events:
            rem = parse_clock_to_remaining_sec(e[6])
            if rem is None:
                continue
            elapsed = float(plen) - rem
            events_with_elapsed.append((elapsed, e[1], e))
        events_with_elapsed.sort(key=lambda x: (x[0], x[1]))

        on_floor: dict[str, tuple[str, float]] = {}

        if previous_on_floor:
            # Lineup carries forward from end of previous period. Inter-period
            # subs (typically logged at clock=12:00 / elapsed=0 of new period)
            # are processed naturally below.
            for pid, team_id in previous_on_floor.items():
                on_floor[pid] = (team_id, 0.0)
        else:
            # P1 (or no prior data): detect starters from first stat actions.
            # A player whose first floor-presence event is NOT
            # `substitution sub=in` was on the floor at period start. We skip
            # `foul technical` events when picking that first event: techs can
            # be assessed against bench players / coaches and the API attributes
            # them to a single player_id, which would otherwise put a non-
            # starter on the floor for the entire game. Cap at 5 per team as a
            # final safety net.
            first_action: dict[str, tuple[float, tuple]] = {}
            for elapsed, _an, e in events_with_elapsed:
                pid = e[4]
                if not pid or pid == e[5] or e[5] == "0":
                    continue
                if e[2] == "foul" and e[3] == "technical":
                    continue
                if pid not in first_action:
                    first_action[pid] = (elapsed, e)

            candidates_by_team: dict[str, list[tuple[float, str]]] = defaultdict(list)
            for pid, (elapsed, e) in first_action.items():
                action_type, sub_type, team_id = e[2], e[3], e[5]
                if action_type == "substitution" and sub_type == "in":
                    continue
                if not team_id:
                    continue
                candidates_by_team[team_id].append((elapsed, pid))

            for team_id, cands in candidates_by_team.items():
                cands.sort()  # earliest elapsed first
                for _elapsed, pid in cands[:5]:
                    on_floor[pid] = (team_id, 0.0)

            # Recover any pre-pass starters whose first-game sub is sub=out and
            # who weren't already added above (they had zero P1 stat events).
            for pid, team_id in starter_pre_pass.items():
                if pid not in on_floor:
                    on_floor[pid] = (team_id, 0.0)

        # Reconcile carry-over with sub evidence. nbastatsv3 (2017-2019)
        # omits period-start subs, so:
        #  - first sub "out" + not on floor → was subbed in at period boundary
        #  - first sub "in" + on floor → was subbed out at period boundary
        first_sub: dict[str, tuple[str, str]] = {}
        for elapsed, _an, e in events_with_elapsed:
            if e[2] != "substitution":
                continue
            pid, team_id, sub_type = e[4], e[5], e[3]
            if not pid or not team_id or pid == team_id or team_id == "0":
                continue
            if pid not in first_sub:
                first_sub[pid] = (sub_type, team_id)

        for pid, (sub_type, team_id) in first_sub.items():
            if sub_type == "out" and pid not in on_floor:
                on_floor[pid] = (team_id, 0.0)
            elif sub_type == "in" and pid in on_floor:
                entered_team, entered_at = on_floor.pop(pid)
                total_seconds[(pid, entered_team)] += max(0.0, 0.0 - entered_at)

        # Collect which players have any activity (non-sub, non-tech-foul)
        # in this period — needed to add missing players and evict phantoms.
        period_active_by_team: dict[str, set[str]] = defaultdict(set)
        period_has_subs: set[str] = set()
        for _elapsed, _an, e in events_with_elapsed:
            pid, team_id = e[4], e[5]
            if not pid or not team_id or team_id == "0":
                continue
            if e[2] == "substitution":
                period_has_subs.add(pid)
            elif not (e[2] == "foul" and e[3] == "technical"):
                period_active_by_team[team_id].add(pid)

        # Players with activity but no subs who aren't on floor were
        # subbed in at the period boundary (missing event).
        for team_id, active_pids in period_active_by_team.items():
            for pid in active_pids:
                if pid not in on_floor and pid not in period_has_subs:
                    on_floor[pid] = (team_id, 0.0)

        # If a team has >5 on floor, evict carry-over players with no
        # activity and no subs this period — they were silently replaced.
        all_active = set()
        for pids in period_active_by_team.values():
            all_active |= pids

        by_team: dict[str, list[str]] = defaultdict(list)
        for pid, (team_id, _) in on_floor.items():
            by_team[team_id].append(pid)
        for team_id, pids in by_team.items():
            if len(pids) <= 5:
                continue
            removable = [
                pid for pid in pids
                if pid not in all_active
                and pid not in period_has_subs
                and pid in previous_on_floor
            ]
            for pid in removable:
                if len(by_team[team_id]) <= 5:
                    break
                entered_team, entered_at = on_floor.pop(pid)
                by_team[team_id].remove(pid)

        # Walk substitution events in chronological (elapsed) order.
        for elapsed, _an, e in events_with_elapsed:
            action_type, sub_type = e[2], e[3]
            pid, team_id = e[4], e[5]
            if action_type != "substitution":
                continue
            if not pid or not team_id or pid == team_id or team_id == "0":
                continue
            if sub_type == "in":
                if pid not in on_floor:
                    on_floor[pid] = (team_id, elapsed)
            elif sub_type == "out":
                if pid in on_floor:
                    entered_team, entered_at = on_floor.pop(pid)
                    total_seconds[(pid, entered_team)] += max(0.0, elapsed - entered_at)

        # Flush anyone still on floor at period end.
        for pid, (team_id, entered_at) in on_floor.items():
            total_seconds[(pid, team_id)] += max(0.0, plen - entered_at)

        # Record lineup state for next period's carry-forward.
        previous_on_floor = {pid: team_id for pid, (team_id, _) in on_floor.items()}

    return total_seconds


def derive_box_stats_for_game(
    events: list[tuple],
) -> dict[tuple[str, str], dict[str, int]]:
    """Count traditional box-score counts from PBP events for one game.

    Returns {(player_id, team_id): {stat_name: int, ...}}.
    Each event row: (period, action_number, action_type, sub_type, player_id,
                     team_id, clock, shot_result, is_field_goal).
    """
    stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: dict(fgm=0, fga=0, fg3m=0, fg3a=0, ftm=0, fta=0,
                     points=0, oreb=0, dreb=0, stl=0, blk=0, tov=0, pf=0)
    )

    for e in events:
        action_type, sub_type = e[2], e[3]
        pid, team_id = e[4], e[5]
        shot_result = e[7]
        if not pid or not team_id or pid == team_id or team_id == "0":
            continue
        key = (pid, team_id)
        made = shot_result == "Made"

        if action_type == "2pt":
            stats[key]["fga"] += 1
            if made:
                stats[key]["fgm"] += 1
                stats[key]["points"] += 2
        elif action_type == "3pt":
            stats[key]["fga"] += 1
            stats[key]["fg3a"] += 1
            if made:
                stats[key]["fgm"] += 1
                stats[key]["fg3m"] += 1
                stats[key]["points"] += 3
        elif action_type == "freethrow":
            stats[key]["fta"] += 1
            if made:
                stats[key]["ftm"] += 1
                stats[key]["points"] += 1
        elif action_type == "rebound":
            if sub_type == "offensive":
                stats[key]["oreb"] += 1
            elif sub_type == "defensive":
                stats[key]["dreb"] += 1
        elif action_type == "steal":
            stats[key]["stl"] += 1
        elif action_type == "block":
            stats[key]["blk"] += 1
        elif action_type == "turnover":
            stats[key]["tov"] += 1
        elif action_type == "foul":
            # NBA `foulsPersonal` includes personal + offensive fouls but
            # excludes technicals (which are tracked separately as
            # `foulsTechnical`). Mirror that convention.
            if sub_type != "technical":
                stats[key]["pf"] += 1

    return stats


def fetch_game_events(conn: sqlite3.Connection, game_id: str) -> list[tuple]:
    rows = conn.execute(
        """
        SELECT period, action_number, action_type, sub_type, player_id,
               team_id, clock, shot_result, is_field_goal
        FROM core.game_events
        WHERE game_id = ?
        ORDER BY period, action_number
        """,
        (game_id,),
    ).fetchall()
    return rows


def fetch_game_meta(conn: sqlite3.Connection) -> dict[str, tuple[str, int]]:
    """{game_id: (game_date, season)}"""
    return {
        gid: (date, season)
        for gid, date, season in conn.execute(
            "SELECT game_id, game_date, season FROM core.games WHERE game_date IS NOT NULL"
        )
    }


def fetch_team_tricodes(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    """{(game_id, team_id): tricode} from game_events (one row per game/team)."""
    rows = conn.execute(
        """
        SELECT game_id, team_id, team_tricode
        FROM core.game_events
        WHERE team_id IS NOT NULL AND team_tricode IS NOT NULL
        GROUP BY game_id, team_id
        """
    ).fetchall()
    return {(gid, tid): tri for gid, tid, tri in rows}


def fetch_player_names(conn: sqlite3.Connection) -> dict[str, str]:
    return {pid: name for pid, name in conn.execute(
        "SELECT player_id, COALESCE(player_name, player_name_i) FROM core.players"
    )}


def build_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS player_game_stats;
        DROP TABLE IF EXISTS build_metadata;

        CREATE TABLE player_game_stats (
            game_id TEXT NOT NULL,
            game_date TEXT NOT NULL,
            season INTEGER NOT NULL,
            team_id TEXT NOT NULL,
            team_tricode TEXT,
            player_id TEXT NOT NULL,
            player_name TEXT,
            seconds_played REAL NOT NULL,
            minutes_played REAL NOT NULL,
            fgm INTEGER NOT NULL,
            fga INTEGER NOT NULL,
            fg3m INTEGER NOT NULL,
            fg3a INTEGER NOT NULL,
            ftm INTEGER NOT NULL,
            fta INTEGER NOT NULL,
            points INTEGER NOT NULL,
            oreb INTEGER NOT NULL,
            dreb INTEGER NOT NULL,
            reb INTEGER NOT NULL,
            stl INTEGER NOT NULL,
            blk INTEGER NOT NULL,
            tov INTEGER NOT NULL,
            pf INTEGER NOT NULL,
            PRIMARY KEY (game_id, player_id)
        );

        CREATE TABLE build_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_pgs_date ON player_game_stats (game_date, game_id);
        CREATE INDEX idx_pgs_player ON player_game_stats (player_id, game_date);
        CREATE INDEX idx_pgs_team_game ON player_game_stats (game_id, team_id);
        CREATE INDEX idx_pgs_season ON player_game_stats (season);
        """
    )


def validate_team_minutes(
    minutes_by_pid: dict[tuple[str, str], float],
    events: list[tuple],
    game_id: str,
    strict: bool,
) -> list[str]:
    """Sanity-check: team total seconds == period_length × 5 × n_periods."""
    warnings: list[str] = []
    max_period = max((e[0] for e in events), default=0)
    expected_per_team = sum(
        period_length_sec(p) * 5 for p in range(1, max_period + 1)
    )

    by_team: dict[str, float] = defaultdict(float)
    for (_pid, team_id), sec in minutes_by_pid.items():
        by_team[team_id] += sec

    for team_id, secs in by_team.items():
        diff = secs - expected_per_team
        # NBA play tolerates small under-counts (e.g. injury-driven 4-man lineups
        # are rare but exist briefly). Warn if >2% off.
        if abs(diff) > 0.02 * expected_per_team:
            msg = (
                f"game={game_id} team={team_id} "
                f"sum_seconds={secs:.0f} expected={expected_per_team} "
                f"diff={diff:+.0f}"
            )
            warnings.append(msg)
            if strict:
                raise RuntimeError(f"minutes validation failed: {msg}")
    return warnings


def process_games(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
) -> dict[str, Any]:
    game_meta = fetch_game_meta(conn)
    tricodes = fetch_team_tricodes(conn)
    player_names = fetch_player_names(conn)

    game_ids = sorted(game_meta.keys())
    if args.limit_games:
        game_ids = game_ids[:args.limit_games]

    rows_to_insert: list[tuple] = []
    n_games_ok = 0
    n_games_warn = 0
    warnings_sample: list[str] = []

    for i, gid in enumerate(game_ids):
        events = fetch_game_events(conn, gid)
        if not events:
            continue
        minutes = derive_minutes_for_game(events)
        # box stats needs (period, action_number, action_type, sub_type, pid,
        # team_id, clock, shot_result, is_field_goal) — same shape as fetched.
        box = derive_box_stats_for_game(events)

        warnings = validate_team_minutes(minutes, events, gid, args.strict)
        if warnings:
            n_games_warn += 1
            if len(warnings_sample) < 20:
                warnings_sample.extend(warnings[:20 - len(warnings_sample)])
        else:
            n_games_ok += 1

        game_date, season = game_meta[gid]
        # Union of keys (player, team) seen in either minutes or box.
        all_keys = set(minutes.keys()) | set(box.keys())
        for (pid, team_id) in all_keys:
            sec = minutes.get((pid, team_id), 0.0)
            b = box.get((pid, team_id), {})
            oreb = b.get("oreb", 0)
            dreb = b.get("dreb", 0)
            rows_to_insert.append((
                gid, game_date, season, team_id,
                tricodes.get((gid, team_id)),
                pid, player_names.get(pid),
                sec, sec / 60.0,
                b.get("fgm", 0), b.get("fga", 0),
                b.get("fg3m", 0), b.get("fg3a", 0),
                b.get("ftm", 0), b.get("fta", 0),
                b.get("points", 0),
                oreb, dreb, oreb + dreb,
                b.get("stl", 0), b.get("blk", 0),
                b.get("tov", 0), b.get("pf", 0),
            ))

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(game_ids)} games...")

    print(f"  inserting {len(rows_to_insert)} rows...")
    conn.executemany(
        """
        INSERT INTO player_game_stats VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows_to_insert,
    )

    return {
        "n_games": len(game_ids),
        "n_games_ok": n_games_ok,
        "n_games_warn": n_games_warn,
        "n_rows": len(rows_to_insert),
        "warnings_sample": warnings_sample,
    }


def write_metadata(conn: sqlite3.Connection, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    metadata = {
        "core_db": str(args.core_db),
        "output": str(args.output),
        "strict": args.strict,
        **summary,
    }
    conn.executemany(
        "INSERT INTO build_metadata (key, value) VALUES (?, ?)",
        [(key, json.dumps(value)) for key, value in metadata.items()],
    )


def main() -> None:
    args = parse_args()
    if not args.core_db.exists():
        raise FileNotFoundError(f"Core database not found: {args.core_db}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.output) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("ATTACH DATABASE ? AS core", (str(args.core_db),))
        build_schema(conn)
        summary = process_games(conn, args)
        create_indexes(conn)
        write_metadata(conn, args, summary)
        conn.commit()

    short = {k: v for k, v in summary.items() if k != "warnings_sample"}
    print(json.dumps(short, indent=2))
    print(f"\nplayer game stats SQLite database: {args.output}")
    if summary["warnings_sample"]:
        print(f"\nFirst {len(summary['warnings_sample'])} validation warnings:")
        for w in summary["warnings_sample"]:
            print(f"  {w}")


if __name__ == "__main__":
    main()
