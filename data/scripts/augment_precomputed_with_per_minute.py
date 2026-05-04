"""Add per-minute box stats to the precomputed features DB.

Reads per_minute_box.sqlite and maps (game_id, player_id, minute_idx)
rows to the precomputed DB's (game_id, window_start, side, slot) via
the vocab table. Stores a packed blob of 48*14 = 672 floats per player.

Adds table: player_minute_box
  (game_id, window_start, side, slot, stats_blob)
  stats_blob = struct.pack('672f', ...) — 48 minutes × 14 box stats,
  row-major: [m0_pts, m0_fgm, ..., m0_pf, m1_pts, ..., m47_pf]
"""

import sqlite3
import struct
from pathlib import Path

PRECOMPUTED_DB = Path("data/features_v5_precomputed.db")
PER_MINUTE_DB = Path("data/artifacts/per_minute_box.sqlite")

N_MINUTES = 48
N_STATS = 14
BLOB_FLOATS = N_MINUTES * N_STATS

BOX_COLS = (
    "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "ast", "tov", "blk", "oreb", "dreb", "stl", "pf",
)


def build():
    conn_pre = sqlite3.connect(str(PRECOMPUTED_DB))
    conn_pm = sqlite3.connect(str(PER_MINUTE_DB))

    # Build reverse vocab: token_idx -> player_id
    vocab_rows = conn_pre.execute("SELECT player_id, token_idx FROM vocab").fetchall()
    idx_to_pid = {r[1]: r[0] for r in vocab_rows}
    print(f"Vocab: {len(idx_to_pid)} entries")

    # Get all (game_id, window_start, side, slot, player_idx) from precomputed
    player_rows = conn_pre.execute(
        "SELECT game_id, window_start, side, slot, player_idx FROM players"
    ).fetchall()
    print(f"Player slots: {len(player_rows):,}")

    # Load ALL per-minute stats into memory keyed by (game_id, player_id)
    # Each value: dict[minute_idx] -> tuple of 14 stats
    print("Loading per-minute box stats...")
    pm_rows = conn_pm.execute(
        f"SELECT game_id, player_id, minute_idx, {','.join(BOX_COLS)} FROM per_minute_box"
    ).fetchall()
    print(f"Per-minute rows: {len(pm_rows):,}")

    pm_data: dict[tuple[str, str], dict[int, tuple]] = {}
    for row in pm_rows:
        key = (row[0], row[1])
        if key not in pm_data:
            pm_data[key] = {}
        pm_data[key][row[2]] = tuple(row[3:])

    # Create table
    conn_pre.execute("DROP TABLE IF EXISTS player_minute_box")
    conn_pre.execute(
        """CREATE TABLE player_minute_box (
            game_id TEXT,
            window_start TEXT,
            side INTEGER,
            slot INTEGER,
            stats_blob BLOB,
            PRIMARY KEY (game_id, window_start, side, slot)
        )"""
    )

    # Load supervised player labels to cross-check
    label_pts: dict[tuple[str, str, int, int], float] = {}
    label_rows = conn_pre.execute(
        "SELECT game_id, window_start, side, slot, targets FROM player_labels"
    ).fetchall()
    for lr in label_rows:
        tgts = struct.unpack(f"{N_STATS}f", lr[4])
        label_pts[(lr[0], lr[1], lr[2], lr[3])] = tgts[0]  # PTS

    # Load player minutes for cross-check
    minutes_data: dict[tuple[str, str, int, int], float] = {}
    for r in conn_pre.execute(
        "SELECT game_id, window_start, side, slot, minutes_actual FROM players"
    ).fetchall():
        minutes_data[(r[0], r[1], r[2], r[3])] = r[4]

    # Process each player slot
    batch = []
    matched = 0
    unmatched = 0
    padded = 0
    errors = []

    for game_id, window_start, side, slot, player_idx in player_rows:
        if player_idx == 0:
            blob = struct.pack(f"{BLOB_FLOATS}f", *([0.0] * BLOB_FLOATS))
            batch.append((game_id, window_start, side, slot, blob))
            padded += 1
            continue

        player_id = idx_to_pid.get(player_idx)
        if not player_id:
            blob = struct.pack(f"{BLOB_FLOATS}f", *([0.0] * BLOB_FLOATS))
            batch.append((game_id, window_start, side, slot, blob))
            unmatched += 1
            continue

        minute_stats = pm_data.get((game_id, player_id), {})

        if not minute_stats:
            sup_pts = label_pts.get((game_id, window_start, side, slot), 0)
            mins = minutes_data.get((game_id, window_start, side, slot), 0)
            if sup_pts > 0 and mins > 1.0:
                errors.append(
                    f"  game={game_id} side={side} slot={slot} "
                    f"player_id={player_id} mins={mins:.1f} sup_pts={sup_pts:.0f} "
                    f"-- no per_minute_box data"
                )

        flat = []
        for m in range(N_MINUTES):
            stats = minute_stats.get(m, (0.0,) * N_STATS)
            flat.extend(stats)

        blob = struct.pack(f"{BLOB_FLOATS}f", *flat)
        batch.append((game_id, window_start, side, slot, blob))
        if minute_stats:
            matched += 1
        else:
            unmatched += 1

        if len(batch) >= 10000:
            conn_pre.executemany(
                "INSERT INTO player_minute_box VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            batch.clear()

    if batch:
        conn_pre.executemany(
            "INSERT INTO player_minute_box VALUES (?, ?, ?, ?, ?)",
            batch,
        )
    conn_pre.commit()

    total = conn_pre.execute("SELECT COUNT(*) FROM player_minute_box").fetchone()[0]
    print(f"\nDone. {total:,} rows added to player_minute_box")
    print(f"  matched: {matched:,}, padded(idx=0): {padded:,}, unmatched: {unmatched:,}")

    n_unique_errors = len({(e.split("game=")[1].split()[0], e.split("player_id=")[1].split()[0]) for e in errors})
    if errors:
        print(f"\n*** WARNING: {len(errors):,} rows ({n_unique_errors} unique player-games) "
              f"played and scored but have no per_minute_box data ***")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20:,} more")
        if n_unique_errors > 10:
            raise RuntimeError(
                f"{n_unique_errors} unique player-games with minutes+PTS have no "
                f"per_minute_box match. The precomputed DB roster likely needs rebuilding."
            )

    # Validation: check Reaves in OKC game
    print("\n--- Validation: Reaves vs OKC (22300884) ---")
    row = conn_pre.execute(
        """SELECT stats_blob FROM player_minute_box
           WHERE game_id='22300884' AND window_start='2024-03-01'
           AND side=0 AND slot=1""",
    ).fetchone()
    if row:
        vals = struct.unpack(f"{BLOB_FLOATS}f", row[0])
        total_pts = sum(vals[m * N_STATS + 0] for m in range(N_MINUTES))
        total_ast = sum(vals[m * N_STATS + 7] for m in range(N_MINUTES))
        print(f"  PTS={total_pts:.0f} AST={total_ast:.0f}")
        # Show per-minute breakdown for minutes with events
        for m in range(N_MINUTES):
            pts = vals[m * N_STATS + 0]
            if pts > 0:
                stats = vals[m * N_STATS : (m + 1) * N_STATS]
                print(f"  min {m:2d}: pts={stats[0]:.0f} fgm={stats[1]:.0f} fga={stats[2]:.0f} ast={stats[7]:.0f}")

    conn_pre.close()
    conn_pm.close()


if __name__ == "__main__":
    build()
