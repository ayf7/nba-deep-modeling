"""Build per-player per-minute box stats from PBP events.

Reads game_events from nba_core.sqlite, buckets each stat event into
one of 48 regulation minutes, and writes a SQLite DB with one row per
(game_id, player_id, minute_idx) containing all 14 box-stat columns.

Assists are parsed from the made-shot description string (format:
"(A. Reaves 1 AST)") and mapped to player_id via the players table's
player_name_i field, scoped to players active in that game.

Output: data/artifacts/per_minute_box.sqlite
"""

import re
import sqlite3
import unicodedata
from pathlib import Path

CORE_DB = Path(__file__).resolve().parents[1] / "artifacts" / "nba_core.sqlite"
OUTPUT_DB = Path(__file__).resolve().parents[1] / "artifacts" / "per_minute_box.sqlite"

BOX_COLS = (
    "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "ast", "tov", "blk", "oreb", "dreb", "stl", "pf",
)

AST_PATTERN = re.compile(r"\(([^)]+?)\s+\d+\s+AST\)")
SUFFIX_RE = re.compile(r"\s+(?:Jr\.|Sr\.|III|IV|II)$")


def _strip_accents(s: str) -> str:
    """Normalize unicode to ASCII: Jokić -> Jokic, Schröder -> Schroder."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clock_to_minute_idx(clock: str, period: int) -> int | None:
    if period > 4:
        return None
    m = re.match(r"PT(\d+)M([\d.]+)S", clock)
    if not m:
        return None
    mins_left = int(m.group(1))
    minute_in_period = min(11, 11 - mins_left)
    return (period - 1) * 12 + minute_in_period


def _get_or_create(stats, key):
    if key not in stats:
        stats[key] = {c: 0.0 for c in BOX_COLS}
    return stats[key]


def process_game(
    conn_core: sqlite3.Connection,
    game_id: str,
    name_i_to_pids: dict[str, list[str]],
) -> list[tuple]:
    rows = conn_core.execute(
        """SELECT clock, period, action_type, sub_type, shot_result,
                  player_id, team_id, description
           FROM game_events
           WHERE game_id = ? AND period <= 4""",
        (game_id,),
    ).fetchall()

    # Build per-game player_id -> team_id from events
    pid_to_team: dict[str, str] = {}
    for row in rows:
        pid = row["player_id"]
        tid = row["team_id"]
        if pid and tid:
            pid_to_team[pid] = tid

    # Build per-game name_i -> player_id scoped to this game's players.
    # For collision names (e.g. "T. Young"), only the player actually
    # in this game gets mapped. If both colliders are in the same game,
    # setdefault keeps the first and the second is lost (rare).
    game_pids = set(pid_to_team.keys())
    game_name_map: dict[str, str] = {}

    def _register(name: str, pid: str):
        variants = [name]
        no_suffix = SUFFIX_RE.sub("", name)
        if no_suffix != name:
            variants.append(no_suffix)
        for v in list(variants):
            a = _strip_accents(v)
            if a != v:
                variants.append(a)
        for v in list(variants):
            normalized = v.replace(".", ". ").replace("  ", " ").strip()
            parts = normalized.split(". ", 1)
            if len(parts) == 2:
                last = parts[1]
                variants.append(last)
                la = _strip_accents(last)
                if la != last:
                    variants.append(la)
                if len(parts[0]) > 1:
                    short = parts[0][0] + ". " + last
                    variants.append(short)
                    sa = _strip_accents(short)
                    if sa != short:
                        variants.append(sa)
        for v in variants:
            game_name_map.setdefault(v, pid)

    for name_i, pids in name_i_to_pids.items():
        for pid in pids:
            if pid in game_pids:
                _register(name_i, pid)

    stats: dict[tuple[str, str, int], dict[str, float]] = {}

    for row in rows:
        clock, period = row["clock"], row["period"]
        action_type = (row["action_type"] or "").lower()
        shot_result = row["shot_result"] or ""
        player_id = row["player_id"]
        team_id = row["team_id"] or ""
        description = row["description"] or ""

        if not player_id or not clock:
            continue
        midx = clock_to_minute_idx(clock, period)
        if midx is None:
            continue

        s = _get_or_create(stats, (player_id, team_id, midx))
        made = shot_result == "Made"

        if action_type == "2pt":
            s["fga"] += 1
            if made:
                s["fgm"] += 1
                s["pts"] += 2
                _credit_assist(stats, description, game_name_map, pid_to_team, midx)

        elif action_type == "3pt":
            s["fga"] += 1
            s["fg3a"] += 1
            if made:
                s["fgm"] += 1
                s["fg3m"] += 1
                s["pts"] += 3
                _credit_assist(stats, description, game_name_map, pid_to_team, midx)

        elif action_type == "freethrow":
            s["fta"] += 1
            if made:
                s["ftm"] += 1
                s["pts"] += 1

        elif action_type == "rebound":
            sub = (row["sub_type"] or "").lower()
            if "offensive" in sub:
                s["oreb"] += 1
            else:
                s["dreb"] += 1

        elif action_type in ("turnover",):
            s["tov"] += 1

        elif action_type == "steal":
            s["stl"] += 1

        elif action_type == "block":
            s["blk"] += 1

        elif action_type == "foul":
            sub = (row["sub_type"] or "").lower()
            # All fouls count as PF except pure technicals (technical, delay-technical, etc.)
            # Offensive fouls and double-technicals DO count as personal fouls
            if sub not in ("technical", "delay-technical", "hanging-technical",
                           "taunting-technical", "non-unsportsmanlike-technical",
                           "too-many-players-technical", "excess-timeout-technical"):
                s["pf"] += 1

    result = []
    for (pid, tid, midx), s in stats.items():
        result.append(
            (game_id, pid, tid, midx)
            + tuple(s[c] for c in BOX_COLS)
        )
    return result


def _credit_assist(stats, description, game_name_map, pid_to_team, midx):
    ast_match = AST_PATTERN.search(description)
    if not ast_match:
        return
    assister_name_i = ast_match.group(1).strip()
    assister_pid = game_name_map.get(assister_name_i)
    if not assister_pid:
        return
    a_tid = pid_to_team.get(assister_pid, "")
    sa = _get_or_create(stats, (assister_pid, a_tid, midx))
    sa["ast"] += 1


def build():
    conn_core = sqlite3.connect(str(CORE_DB))
    conn_core.row_factory = sqlite3.Row

    # Build global name_i -> [player_ids] mapping (list to handle collisions)
    name_i_to_pids: dict[str, list[str]] = {}
    for row in conn_core.execute(
        "SELECT player_id, player_name_i FROM players WHERE player_name_i != ''"
    ).fetchall():
        name_i = row["player_name_i"]
        pid = row["player_id"]
        name_i_to_pids.setdefault(name_i, [])
        if pid not in name_i_to_pids[name_i]:
            name_i_to_pids[name_i].append(pid)
    collisions = {k for k, v in name_i_to_pids.items() if len(v) > 1}
    print(f"Player name_i entries: {len(name_i_to_pids)}, collisions: {len(collisions)}")
    if collisions:
        print(f"  (colliding names resolved by per-game scoping)")

    game_ids = [
        r[0]
        for r in conn_core.execute(
            "SELECT DISTINCT game_id FROM game_events ORDER BY game_id"
        ).fetchall()
    ]
    print(f"Processing {len(game_ids)} games...")

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn_out = sqlite3.connect(str(OUTPUT_DB))
    conn_out.execute("DROP TABLE IF EXISTS per_minute_box")
    cols_def = ", ".join(f"{c} REAL DEFAULT 0" for c in BOX_COLS)
    conn_out.execute(
        f"""CREATE TABLE per_minute_box (
            game_id TEXT,
            player_id TEXT,
            team_id TEXT,
            minute_idx INTEGER,
            {cols_def},
            PRIMARY KEY (game_id, player_id, minute_idx)
        )"""
    )
    conn_out.execute(
        "CREATE INDEX idx_pmb_game ON per_minute_box(game_id)"
    )

    placeholders = ", ".join("?" * (4 + len(BOX_COLS)))
    batch = []
    for i, gid in enumerate(game_ids):
        rows = process_game(conn_core, gid, name_i_to_pids)
        batch.extend(rows)
        if len(batch) >= 50000:
            conn_out.executemany(
                f"INSERT OR REPLACE INTO per_minute_box VALUES ({placeholders})",
                batch,
            )
            batch.clear()
        if (i + 1) % 1000 == 0:
            conn_out.commit()
            print(f"  {i+1}/{len(game_ids)} games processed")

    if batch:
        conn_out.executemany(
            f"INSERT OR REPLACE INTO per_minute_box VALUES ({placeholders})",
            batch,
        )
    conn_out.commit()

    total = conn_out.execute("SELECT COUNT(*) FROM per_minute_box").fetchone()[0]
    games = conn_out.execute(
        "SELECT COUNT(DISTINCT game_id) FROM per_minute_box"
    ).fetchone()[0]
    print(f"\nDone. {total:,} rows across {games:,} games -> {OUTPUT_DB}")

    # Validation
    print("\n--- Validation: Reaves vs OKC (22300884) ---")
    rows = conn_out.execute(
        """SELECT minute_idx, pts, fgm, fga, fg3m, fg3a, ftm, fta, ast, tov, blk, oreb, dreb, stl, pf
           FROM per_minute_box
           WHERE game_id='22300884' AND player_id='1630559'
           ORDER BY minute_idx""",
    ).fetchall()
    totals = {c: 0.0 for c in BOX_COLS}
    for r in rows:
        for j, c in enumerate(BOX_COLS):
            totals[c] += r[j + 1]
    print(f"  Computed: " + " ".join(f"{c}={totals[c]:.0f}" for c in BOX_COLS))
    print(f"  Actual:   pts=16 fgm=6 fga=8 fg3m=4 fg3a=5 ftm=0 fta=0 ast=7 tov=5 blk=0 oreb=1 dreb=5 stl=1 pf=1")

    conn_out.close()
    conn_core.close()


if __name__ == "__main__":
    build()
