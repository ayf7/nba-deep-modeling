#!/usr/bin/env python3
"""Pull NBA injury report snapshots from the `nbainjuries` package and cache to sqlite.

Coverage of the upstream feed: 2021-22 NBA season onwards (15-minute snapshots
on the NBA's server). One snapshot is pulled per game date at --target-hour
(default 17:00, before most evening tip-offs). Output is one row per
(snapshot_ts, team, player_name).

Usage
-----
# 1) Verify the package returns what we expect on a single date:
#    python build_injury_history.py --probe --probe-date 2024-01-15
#
# 2) Full pull (resumable; skips dates already in the db):
#    python build_injury_history.py
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path


def _silence_pdfbox_warnings() -> None:
    """Suppress PDFBox font-fallback noise emitted via java.util.logging.

    PDFBox is invoked indirectly by nbainjuries (PDF parsing of the official
    NBA injury report PDF). Must run BEFORE any import that may start the JVM.
    """
    if "JAVA_TOOL_OPTIONS" in os.environ:
        return
    fd, path = tempfile.mkstemp(suffix=".properties", prefix="jul_silence_pdfbox_")
    with os.fdopen(fd, "w") as f:
        f.write(
            ".level = SEVERE\n"
            "handlers = java.util.logging.ConsoleHandler\n"
            "java.util.logging.ConsoleHandler.level = SEVERE\n"
        )
    os.environ["JAVA_TOOL_OPTIONS"] = f"-Djava.util.logging.config.file={path}"


_silence_pdfbox_warnings()

import pandas as pd  # noqa: E402  (after JVM-options setup above)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "nba_injury_history.sqlite"
DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--start-date",
        default="2021-10-19",
        help="First game date (inclusive). Default = first day of 2021-22 regular season.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last game date (inclusive). Default = today.",
    )
    parser.add_argument(
        "--target-hour",
        type=int,
        default=17,
        help="Hour-of-day for the daily snapshot (24h). 17 = 5pm, before most tip-offs.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Fetch one snapshot, print schema/sample/status counts, and exit. No DB writes.",
    )
    parser.add_argument(
        "--probe-date",
        default="2024-01-15",
        help="Date to probe in --probe mode.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds between requests. Default 0 (NBA's CDN doesn't rate-limit).",
    )
    parser.add_argument(
        "--core-db",
        type=Path,
        default=DEFAULT_CORE_DB,
        help="Path to nba_core.sqlite. If present, only pull dates that have NBA games.",
    )
    parser.add_argument(
        "--all-days",
        action="store_true",
        help="Iterate every calendar day instead of only NBA-game dates from --core-db.",
    )
    return parser.parse_args()


def fetch_snapshot(when: datetime) -> pd.DataFrame:
    """Local import so --probe failure (missing package) is the first error a user sees clearly."""
    from nbainjuries import injury

    return injury.get_reportdata(when, return_df=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    return df.rename(columns=rename_map)


def build_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS injury_snapshots (
            snapshot_ts TEXT NOT NULL,
            game_date TEXT,
            game_time TEXT,
            matchup TEXT,
            team TEXT,
            player_name TEXT,
            status TEXT,
            reason TEXT,
            PRIMARY KEY (snapshot_ts, team, player_name)
        );
        CREATE INDEX IF NOT EXISTS idx_injury_game_date ON injury_snapshots(game_date);
        CREATE INDEX IF NOT EXISTS idx_injury_player ON injury_snapshots(player_name);
        CREATE INDEX IF NOT EXISTS idx_injury_team ON injury_snapshots(team);
        """
    )


def probe(args: argparse.Namespace) -> None:
    when = datetime.fromisoformat(args.probe_date).replace(hour=args.target_hour, minute=0)
    print(f"[probe] fetching snapshot for {when.isoformat()}")
    df = fetch_snapshot(when)
    print(f"[probe] returned {len(df)} rows")
    print(f"[probe] raw columns: {list(df.columns)}")
    norm = normalize_columns(df)
    print(f"[probe] normalized columns: {list(norm.columns)}")
    print("[probe] head(10):")
    print(norm.head(10).to_string())
    status_col = next(
        (c for c in ("current_status", "status") if c in norm.columns), None
    )
    if status_col:
        print(f"[probe] status counts ({status_col}):")
        print(norm[status_col].value_counts())
    print("[probe] done. No DB writes.")


def insert_snapshot(conn: sqlite3.Connection, snapshot_ts: str, df: pd.DataFrame) -> int:
    df = normalize_columns(df)
    status_col = next(
        (c for c in ("current_status", "status") if c in df.columns), None
    )
    n = 0
    for row in df.itertuples(index=False):
        rd = row._asdict()
        conn.execute(
            """
            INSERT OR REPLACE INTO injury_snapshots
            (snapshot_ts, game_date, game_time, matchup, team, player_name, status, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_ts,
                rd.get("game_date"),
                rd.get("game_time"),
                rd.get("matchup"),
                rd.get("team"),
                rd.get("player_name"),
                rd.get(status_col) if status_col else None,
                rd.get("reason"),
            ),
        )
        n += 1
    return n


def existing_snapshot_dates(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT substr(snapshot_ts, 1, 10) FROM injury_snapshots"
        )
        if row[0]
    }


def game_dates_in_window(
    core_db: Path, start: pd.Timestamp, end: pd.Timestamp
) -> list[pd.Timestamp]:
    with sqlite3.connect(core_db) as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT date(game_date) AS d FROM games WHERE game_date IS NOT NULL",
            conn,
        )
    dates = pd.to_datetime(df["d"]).sort_values().reset_index(drop=True)
    mask = (dates >= start) & (dates <= end)
    return list(dates[mask])


def main() -> None:
    args = parse_args()

    if args.probe:
        probe(args)
        return

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.today().normalize()

    if args.all_days or not args.core_db.exists():
        if not args.core_db.exists() and not args.all_days:
            print(f"[warn] {args.core_db} not found; iterating all calendar days")
        n_calendar = (end_date - start_date).days + 1
        target_dates = [start_date + pd.Timedelta(days=i) for i in range(n_calendar)]
    else:
        target_dates = game_dates_in_window(args.core_db, start_date, end_date)
        print(f"[run] using {len(target_dates)} NBA game dates from {args.core_db}")

    print(
        f"[run] {start_date.date()} -> {end_date.date()}, target_hour={args.target_hour}, "
        f"sleep={args.sleep}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.output)
    try:
        build_schema(conn)
        already_have = existing_snapshot_dates(conn)
        print(f"[resume] {len(already_have)} dates already in db")

        n_pulled = 0
        n_skipped = 0
        n_failed = 0
        n_rows = 0
        for cur in target_dates:
            iso = cur.strftime("%Y-%m-%d")
            if iso in already_have:
                n_skipped += 1
                continue

            when = cur.to_pydatetime().replace(hour=args.target_hour, minute=0)
            try:
                df = fetch_snapshot(when)
            except Exception as e:
                msg = str(e)
                is_no_report = (
                    "403" in msg
                    or "Forbidden" in msg
                    or "Failed validation" in msg
                    or "Failed to access" in msg
                )
                if not is_no_report:
                    print(f"[fail] {iso}: {type(e).__name__}: {e}")
                n_failed += 1
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            if df is None or len(df) == 0:
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            row_count = insert_snapshot(conn, when.isoformat(), df)
            conn.commit()
            n_pulled += 1
            n_rows += row_count
            if n_pulled % 25 == 0:
                print(f"[progress] {iso}: pulled={n_pulled}/{len(target_dates) - n_skipped} rows_total={n_rows}")
            if args.sleep:
                time.sleep(args.sleep)

        print(
            f"[done] pulled={n_pulled} skipped_existing={n_skipped} "
            f"failed={n_failed} rows_inserted={n_rows}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
