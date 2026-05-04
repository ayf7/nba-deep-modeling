#!/usr/bin/env python3
"""Build per-minute on-court presence vectors from PBP substitution data.

Reads game_events from nba_core.sqlite and produces:
  data/artifacts/cme_v5_features.sqlite

Tables
------
player_minute_presence
    game_id, team_id, player_id, is_starter,
    m00..m47  (fractional on-court presence per regulation minute),
    total_seconds_regulation

game_regulation_info
    game_id, home_team_id, away_team_id,
    home_score_regulation, away_score_regulation,
    home_score_final, away_score_final,
    is_overtime, n_periods
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "cme_v5_features.sqlite"

REGULATION_PERIODS = 4
MINUTES_PER_PERIOD = 12
REGULATION_MINUTES = REGULATION_PERIODS * MINUTES_PER_PERIOD  # 48
SECONDS_PER_MINUTE = 60
REGULATION_SECONDS = REGULATION_MINUTES * SECONDS_PER_MINUTE  # 2880

MINUTE_COLS = [f"m{i:02d}" for i in range(REGULATION_MINUTES)]


# ------------------------------------------------------------------ #
#  Clock parsing
# ------------------------------------------------------------------ #

def parse_clock_remaining(clock: str) -> float:
    """Parse 'PT12M00.00S' → remaining seconds in period."""
    if not clock or not clock.startswith("PT"):
        return 0.0
    s = clock[2:]
    if s.endswith("S"):
        s = s[:-1]
    parts = s.split("M")
    if len(parts) != 2:
        return 0.0
    return float(parts[0]) * 60.0 + (float(parts[1]) if parts[1] else 0.0)


def clock_to_game_seconds(clock: str, period: int) -> float:
    """Clock + period → elapsed game-seconds from tip-off (0–2880 for regulation)."""
    remaining = parse_clock_remaining(clock)
    period_start = (period - 1) * MINUTES_PER_PERIOD * SECONDS_PER_MINUTE
    elapsed_in_period = MINUTES_PER_PERIOD * SECONDS_PER_MINUTE - remaining
    return period_start + max(0.0, elapsed_in_period)


# ------------------------------------------------------------------ #
#  Per-game presence computation (per-quarter approach)
# ------------------------------------------------------------------ #

QUARTER_SECONDS = MINUTES_PER_PERIOD * SECONDS_PER_MINUTE  # 720


def _compute_game(
    events: list[dict],
    home_team: str,
    away_team: str,
) -> tuple[
    dict[tuple[str, str], tuple[list[float], float]],  # (team, pid) → (vec48, total_sec)
    dict[str, set[str]],                                # starters (period-1)
    tuple[int, int],                                    # regulation score (home, away)
    tuple[int, int],                                    # final score (home, away)
    bool,                                               # is_overtime
    int,                                                # n_periods
]:
    team_ids = {home_team, away_team}

    # ---- score tracking ----
    reg_score_home = 0
    reg_score_away = 0
    final_score_home = 0
    final_score_away = 0
    max_period = 0

    for ev in events:
        period = ev["period"]
        max_period = max(max_period, period)
        sh, sa = ev["score_home"], ev["score_away"]
        if sh is not None:
            final_score_home = int(sh)
        if sa is not None:
            final_score_away = int(sa)
        if period <= REGULATION_PERIODS:
            if sh is not None:
                reg_score_home = int(sh)
            if sa is not None:
                reg_score_away = int(sa)

    # ---- per-quarter presence ----
    intervals: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    starters: dict[str, set[str]] = {t: set() for t in team_ids}

    on_court_end: set[tuple[str, str]] = set()

    for period in range(1, REGULATION_PERIODS + 1):
        q_start = (period - 1) * QUARTER_SECONDS
        q_end = period * QUARTER_SECONDS

        period_events = [ev for ev in events if ev["period"] == period]

        player_subs: dict[tuple[str, str], list[tuple[float, int, str]]] = defaultdict(list)
        player_active: set[tuple[str, str]] = set()

        for ev in period_events:
            tid, pid = ev["team_id"], ev["player_id"]
            if not tid or not pid or tid not in team_ids:
                continue
            if ev["action_type"] == "substitution":
                game_sec = clock_to_game_seconds(ev["clock"], period)
                player_subs[(tid, pid)].append((game_sec, ev["action_number"], ev["sub_type"]))
            elif not (ev["action_type"] == "foul" and ev["sub_type"] == "technical"):
                player_active.add((tid, pid))

        for key in player_subs:
            player_subs[key].sort()

        all_players = set(player_subs.keys()) | player_active

        # Classify who's on court at period start: sub-confirmed vs activity-only.
        # Sub-confirmed = first sub is "out" (strong evidence).
        # Activity-only = has events but no subs (weaker; could be bench player
        # who entered via unrecorded sub-in).
        sub_confirmed: dict[str, list[tuple[str, str]]] = {t: [] for t in team_ids}
        activity_only: dict[str, set[tuple[str, str]]] = {t: set() for t in team_ids}

        for key in all_players:
            tid, pid = key
            subs = player_subs.get(key, [])
            if not subs:
                if key in player_active:
                    activity_only[tid].add(key)
            elif subs[0][2] != "in":
                sub_confirmed[tid].append(key)

        # Enforce max-5 on court at period start per team.
        # Demote activity-only first; if still >5, demote sub-confirmed
        # with earliest first sub-out (shortest on-court = likely entered
        # via missing sub-in).
        demoted: set[tuple[str, str]] = set()
        for tid in team_ids:
            n_on = len(sub_confirmed[tid]) + len(activity_only[tid])
            if n_on <= 5:
                continue
            demoted |= activity_only[tid]
            if len(sub_confirmed[tid]) > 5:
                by_first_out = sorted(
                    sub_confirmed[tid],
                    key=lambda k: player_subs[k][0][0],
                )
                for k in by_first_out[: len(sub_confirmed[tid]) - 5]:
                    demoted.add(k)

        new_on_court_end: set[tuple[str, str]] = set()

        for key in all_players:
            tid, pid = key
            subs = player_subs.get(key, [])
            has_activity = key in player_active

            if not subs:
                if has_activity and key not in demoted:
                    intervals[key].append((q_start, q_end))
                    new_on_court_end.add(key)
                    if period == 1:
                        starters[tid].add(pid)
                continue

            # First sub "out" → on court at period start.
            # First sub "in" → off court at period start, EVEN if
            # carried over (missing period-boundary sub-out).
            on_at_start = subs[0][2] != "in"
            if on_at_start and period == 1:
                starters[tid].add(pid)

            cur_start: float | None = q_start if on_at_start else None

            for game_sec, _an, sub_type in subs:
                if sub_type == "out":
                    if cur_start is not None:
                        intervals[key].append((cur_start, game_sec))
                        cur_start = None
                elif sub_type == "in":
                    if cur_start is not None:
                        intervals[key].append((cur_start, game_sec))
                    cur_start = game_sec

            if cur_start is not None:
                intervals[key].append((cur_start, q_end))
                new_on_court_end.add(key)

        on_court_end = new_on_court_end

    # ---- build 48-dim vectors ----
    presence: dict[tuple[str, str], tuple[list[float], float]] = {}
    for (tid, pid), ivs in intervals.items():
        vec = [0.0] * REGULATION_MINUTES
        total = 0.0
        for start, end in ivs:
            if start >= end:
                continue
            total += end - start
            for m in range(REGULATION_MINUTES):
                bstart = m * SECONDS_PER_MINUTE
                bend = bstart + SECONDS_PER_MINUTE
                overlap = max(0.0, min(end, bend) - max(start, bstart))
                vec[m] += overlap / SECONDS_PER_MINUTE
        vec = [max(0.0, min(1.0, v)) for v in vec]
        presence[(tid, pid)] = (vec, total)

    is_ot = max_period > REGULATION_PERIODS
    return (
        presence, starters,
        (reg_score_home, reg_score_away),
        (final_score_home, final_score_away),
        is_ot, max_period,
    )


# ------------------------------------------------------------------ #
#  Bulk load + process
# ------------------------------------------------------------------ #

def load_game_teams(core_db: Path) -> dict[str, tuple[str, str]]:
    """game_id → (home_team_id, away_team_id)."""
    out: dict[str, tuple[str, str]] = {}
    with sqlite3.connect(core_db) as conn:
        for gid, h, a in conn.execute(
            "SELECT game_id, home_team_id, away_team_id FROM games"
        ):
            out[str(gid)] = (str(h), str(a))
    return out


def load_events_for_game(conn: sqlite3.Connection, game_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT action_number, clock, period, action_type, sub_type,
               team_id, player_id, score_home, score_away
        FROM game_events
        WHERE game_id = ?
        ORDER BY period, action_number
        """,
        (game_id,),
    ).fetchall()
    return [
        {
            "action_number": r[0],
            "clock": r[1],
            "period": int(r[2]) if r[2] is not None else 0,
            "action_type": r[3] or "",
            "sub_type": r[4] or "",
            "team_id": str(r[5]) if r[5] else "",
            "player_id": str(r[6]) if r[6] else "",
            "score_home": r[7],
            "score_away": r[8],
        }
        for r in rows
    ]


# ------------------------------------------------------------------ #
#  Write output
# ------------------------------------------------------------------ #

def create_tables(conn: sqlite3.Connection) -> None:
    minute_cols_def = ",\n        ".join(f"{c} REAL NOT NULL" for c in MINUTE_COLS)
    conn.executescript(f"""
        DROP TABLE IF EXISTS player_minute_presence;
        DROP TABLE IF EXISTS game_regulation_info;

        CREATE TABLE player_minute_presence (
            game_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            is_starter INTEGER NOT NULL,
            {minute_cols_def},
            total_seconds_regulation REAL NOT NULL,
            PRIMARY KEY (game_id, player_id)
        );
        CREATE INDEX idx_pmp_game ON player_minute_presence (game_id);
        CREATE INDEX idx_pmp_player ON player_minute_presence (player_id);

        CREATE TABLE game_regulation_info (
            game_id TEXT NOT NULL PRIMARY KEY,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            home_score_regulation INTEGER NOT NULL,
            away_score_regulation INTEGER NOT NULL,
            home_score_final INTEGER NOT NULL,
            away_score_final INTEGER NOT NULL,
            is_overtime INTEGER NOT NULL,
            n_periods INTEGER NOT NULL
        );
    """)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-minute presence vectors")
    parser.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Reading from {args.core_db}")
    game_teams = load_game_teams(args.core_db)
    print(f"Found {len(game_teams)} games")

    presence_rows: list[tuple] = []
    reg_info_rows: list[tuple] = []
    n_skipped = 0
    n_starter_issues = 0

    core_conn = sqlite3.connect(args.core_db)
    try:
        for i, (gid, (home, away)) in enumerate(sorted(game_teams.items())):
            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1}/{len(game_teams)} games "
                      f"({n_skipped} skipped, {n_starter_issues} starter issues)")

            events = load_events_for_game(core_conn, gid)
            if not events:
                n_skipped += 1
                continue

            presence, starters, reg_score, final_score, is_ot, n_periods = (
                _compute_game(events, home, away)
            )

            for tid in (home, away):
                if len(starters.get(tid, set())) != 5:
                    n_starter_issues += 1

            reg_info_rows.append((
                gid, home, away,
                reg_score[0], reg_score[1],
                final_score[0], final_score[1],
                int(is_ot), n_periods,
            ))

            starter_set = starters.get(home, set()) | starters.get(away, set())
            for (tid, pid), (vec, total_sec) in presence.items():
                row = (gid, tid, pid, int(pid in starter_set), *vec, total_sec)
                presence_rows.append(row)
    finally:
        core_conn.close()

    print(f"\nProcessed {len(game_teams)} games")
    print(f"  {n_skipped} skipped (no events)")
    print(f"  {n_starter_issues} team-games with != 5 identified starters")
    print(f"  {len(presence_rows)} player-minute-presence rows")
    print(f"  {len(reg_info_rows)} game-regulation-info rows")
    print(f"  {sum(1 for r in reg_info_rows if r[7])} overtime games")

    print(f"\nWriting to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.output) as out_conn:
        create_tables(out_conn)

        minute_placeholders = ",".join("?" * REGULATION_MINUTES)
        out_conn.executemany(
            f"""INSERT INTO player_minute_presence
                (game_id, team_id, player_id, is_starter,
                 {','.join(MINUTE_COLS)}, total_seconds_regulation)
                VALUES (?,?,?,?, {minute_placeholders}, ?)""",
            presence_rows,
        )

        out_conn.executemany(
            """INSERT INTO game_regulation_info
               (game_id, home_team_id, away_team_id,
                home_score_regulation, away_score_regulation,
                home_score_final, away_score_final,
                is_overtime, n_periods)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            reg_info_rows,
        )
        out_conn.commit()

    # Validation: compare total_seconds against player_game_stats for regulation games
    _validate(args.output, args.core_db)
    print("\nDone.")


def _validate(output_db: Path, core_db: Path) -> None:
    pgs_db = ARTIFACT_ROOT / "player_game_stats.sqlite"
    if not pgs_db.exists():
        print("  (skipping validation — player_game_stats.sqlite not found)")
        return

    print("\nValidation against player_game_stats (regulation games only):")
    with sqlite3.connect(output_db) as conn:
        conn.execute(f"ATTACH DATABASE '{pgs_db}' AS pgs")
        conn.execute(f"ATTACH DATABASE '{core_db}' AS core")
        rows = conn.execute("""
            SELECT p.game_id, p.player_id,
                   p.total_seconds_regulation, pgs.seconds_played,
                   g.is_overtime
            FROM player_minute_presence p
            JOIN game_regulation_info g ON p.game_id = g.game_id
            JOIN pgs.player_game_stats pgs
                 ON p.game_id = pgs.game_id AND p.player_id = pgs.player_id
            WHERE g.is_overtime = 0
            LIMIT 50000
        """).fetchall()

    if not rows:
        print("  (no matching rows for validation)")
        return

    diffs = [abs(r[2] - r[3]) for r in rows]
    mean_diff = sum(diffs) / len(diffs)
    max_diff = max(diffs)
    within_5 = sum(1 for d in diffs if d <= 5.0)
    print(f"  {len(rows)} player-games compared")
    print(f"  mean |diff| = {mean_diff:.1f}s, max |diff| = {max_diff:.1f}s")
    print(f"  {within_5}/{len(rows)} ({100*within_5/len(rows):.1f}%) within 5s")


if __name__ == "__main__":
    main()
