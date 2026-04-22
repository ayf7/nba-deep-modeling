#!/usr/bin/env python3
"""Compute P(plays | status) from injury reports vs `player_matchups` ground truth.

Output: data/artifacts/status_play_calibration.json

Used as a soft mask in MAN-Xfmr: each player gets m_i = P(plays | status_i),
multiplied into pairwise prevalence so Out/Doubtful effectively zero out and
Questionable scales by ~0.5.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_INJURY_DB = ARTIFACT_ROOT / "nba_injury_history.sqlite"
DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_OUT = ARTIFACT_ROOT / "status_play_calibration.json"

NOT_LISTED_KEY = "NotListed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with sqlite3.connect(args.injury_db) as conn:
        conn.execute(f"ATTACH DATABASE '{args.core_db}' AS c")
        df = pd.read_sql_query(
            """
            SELECT
                gpa.status,
                COUNT(*) AS n_listed,
                SUM(CASE WHEN p.played IS NOT NULL THEN 1 ELSE 0 END) AS n_played
            FROM game_player_availability gpa
            LEFT JOIN (
                SELECT DISTINCT game_id, player_id, 1 AS played FROM c.player_matchups
            ) p
              ON p.game_id = gpa.game_id AND p.player_id = gpa.player_id
            GROUP BY gpa.status
            """,
            conn,
        )

    df["pct_played"] = df["n_played"] / df["n_listed"]
    table = {
        str(row.status if row.status else ""): float(row.pct_played)
        for row in df.itertuples(index=False)
    }

    # Players not on the injury report: assume ~certain to play (default mask).
    # Empirically near-perfect; we fix at 1.0 to keep healthy starters unaffected.
    table[NOT_LISTED_KEY] = 1.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)

    print(f"[out] {args.output}")
    for k in sorted(table, key=lambda x: -table[x]):
        print(f"  {k:>14}: {table[k]:.4f}")


if __name__ == "__main__":
    main()
