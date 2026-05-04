"""Pull `FROM_YEAR` / `DRAFT_YEAR` for every player_id appearing in
player_game_stats.sqlite via nba_api.commonplayerinfo. Writes
data/artifacts/player_debut.sqlite. Resumable: skips pids already stored.

Rate-limited at ~1 request/sec to be polite to stats.nba.com.
"""
from __future__ import annotations
import argparse
import datetime as dt
import sqlite3
import sys
import time
from pathlib import Path

from nba_api.stats.endpoints import commonplayerinfo

ROOT = Path(__file__).resolve().parents[2]
PGS_DB = ROOT / "data" / "artifacts" / "player_game_stats.sqlite"
OUT_DB = ROOT / "data" / "artifacts" / "player_debut.sqlite"


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_debut (
            player_id TEXT PRIMARY KEY,
            from_year INTEGER,
            to_year INTEGER,
            draft_year INTEGER,
            season_exp INTEGER,
            display_name TEXT,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_debut_errors (
            player_id TEXT PRIMARY KEY,
            error TEXT,
            attempted_at TEXT NOT NULL
        )
    """)
    conn.commit()


def already_fetched(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT player_id FROM player_debut").fetchall()
    return {r[0] for r in rows}


def all_target_pids(pgs_db: Path) -> list[str]:
    with sqlite3.connect(pgs_db) as conn:
        return [
            r[0] for r in conn.execute(
                "SELECT DISTINCT player_id FROM player_game_stats ORDER BY player_id"
            ).fetchall()
        ]


def fetch_one(pid: str, timeout: int = 30) -> dict | None:
    info = commonplayerinfo.CommonPlayerInfo(player_id=int(pid), timeout=timeout)
    df = info.get_data_frames()[0]
    if df.empty:
        return None
    row = df.iloc[0]
    def _to_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    return {
        "from_year": _to_int(row.get("FROM_YEAR")),
        "to_year": _to_int(row.get("TO_YEAR")),
        "draft_year": _to_int(row.get("DRAFT_YEAR")),
        "season_exp": _to_int(row.get("SEASON_EXP")),
        "display_name": str(row.get("DISPLAY_FIRST_LAST", "") or ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.6, help="seconds between requests")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=5.0)
    args = ap.parse_args()

    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    pids = all_target_pids(PGS_DB)
    print(f"[pull_player_debut] {len(pids):,} pids in player_game_stats", flush=True)

    with sqlite3.connect(OUT_DB) as conn:
        ensure_schema(conn)
        done = already_fetched(conn)
        todo = [p for p in pids if p not in done]
        print(f"[pull_player_debut] {len(done):,} already fetched, {len(todo):,} to do",
              flush=True)

        n_ok = 0
        n_fail = 0
        for i, pid in enumerate(todo, 1):
            err = None
            result = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    result = fetch_one(pid)
                    break
                except Exception as e:  # network / parse / rate-limit
                    err = repr(e)
                    if attempt < args.max_retries:
                        time.sleep(args.retry_sleep)
            now = dt.datetime.utcnow().isoformat(timespec="seconds")
            if result is not None:
                conn.execute("""
                    INSERT OR REPLACE INTO player_debut
                      (player_id, from_year, to_year, draft_year, season_exp,
                       display_name, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (pid, result["from_year"], result["to_year"],
                      result["draft_year"], result["season_exp"],
                      result["display_name"], now))
                conn.execute("DELETE FROM player_debut_errors WHERE player_id=?", (pid,))
                n_ok += 1
            else:
                conn.execute("""
                    INSERT OR REPLACE INTO player_debut_errors
                      (player_id, error, attempted_at) VALUES (?, ?, ?)
                """, (pid, err or "empty result", now))
                n_fail += 1
            if i % 25 == 0 or i == len(todo):
                conn.commit()
                print(f"[pull_player_debut] {i}/{len(todo)} pid={pid} ok={n_ok} fail={n_fail}",
                      flush=True)
            time.sleep(args.sleep)
        conn.commit()
    print(f"[pull_player_debut] done. ok={n_ok} fail={n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
