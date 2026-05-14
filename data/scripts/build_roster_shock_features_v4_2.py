#!/usr/bin/env python3
"""Build strictly pregame roster-shock features for Ratings-v4.

The Ratings-v2 prior is intentionally team-level and evolves from completed
scores only.  That signal can become stale on games where the current roster is
very different from the roster that generated the rating.  This builder adds a
pregame-only roster-shock artifact by combining:

* the latest game-specific injury-report availability rows already materialized
  in ``nba_injury_history.sqlite.game_player_availability``;
* recent per-player rotation mass from completed historical matchup rows in
  ``player_matchup_training.sqlite.matchup_training_rows``;
* a pregame player-impact proxy derived from prior scoring + playmaking
  involvement, so the shock artifact can distinguish star absences from lower-
  leverage rotation losses.

For each game, a row is emitted **before** current-game exposure rows are used to
update the player rotation EMAs.  This keeps the shock features leakage-safe.
Positive ``roster_shock_advantage_signal`` means the away team appears more
rotation-shocked than the home team, i.e. the signal should generally help the
home team.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_MATCHUP_DB = ARTIFACT_ROOT / "player_matchup_training.sqlite"
DEFAULT_INJURY_DB = ARTIFACT_ROOT / "nba_injury_history.sqlite"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "roster_shock_features_v4_2.sqlite"
SHOCK_TABLE = "game_roster_shock_features"


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if abs(float(den)) > 1e-9 else 0.0


def _status_unavailable_weight(status: str | None) -> float:
    s = "" if status is None else str(status).strip().lower()
    if not s:
        return 0.0
    if any(tok in s for tok in (
        "out", "inactive", "suspended", "not with team", "not available",
        "ineligible", "health and safety", "g league", "two-way not with",
    )):
        return 1.0
    if "doubtful" in s:
        return 0.85
    if "questionable" in s or "game time" in s or "gtd" in s:
        return 0.50
    if "probable" in s:
        return 0.15
    if "available" in s or "active" in s:
        return 0.0
    # Unknown statuses are weak evidence of disruption rather than ignored.
    return 0.35


def _status_bucket(status: str | None) -> str:
    w = _status_unavailable_weight(status)
    if w >= 0.95:
        return "out"
    if w >= 0.40:
        return "uncertain"
    if w > 0.0:
        return "probable"
    return "available"


def _read_games(core_db: Path) -> pd.DataFrame:
    if not core_db.exists():
        raise FileNotFoundError(f"Missing core DB: {core_db}")
    query = """
        SELECT game_id, season, game_date, home_team_id, away_team_id
        FROM games
        WHERE game_date IS NOT NULL
          AND home_team_id IS NOT NULL
          AND away_team_id IS NOT NULL
        ORDER BY game_date, game_id
    """
    with sqlite3.connect(core_db) as conn:
        df = pd.read_sql_query(query, conn)
    df["game_id"] = df["game_id"].astype(str)
    df["home_team_id"] = df["home_team_id"].astype(str)
    df["away_team_id"] = df["away_team_id"].astype(str)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def _impact_units(points: float, assists: float, turnovers: float) -> float:
    """Crude, leakage-safe offensive impact proxy in point-like units.

    This intentionally stays simple and local to fields already available in
    ``matchup_training_rows``.  It is *not* a RAPM estimate; it is a pregame
    player-value proxy that lets the roster-shock artifact treat a high-usage
    creator differently from a low-leverage rotation player with similar recent
    exposure seconds.
    """
    return max(float(points) + 0.70 * float(assists) - 1.00 * float(turnovers), 0.0)


def _read_exposures(matchup_db: Path) -> dict[str, dict[str, dict[str, tuple[float, float, float]]]]:
    if not matchup_db.exists():
        raise FileNotFoundError(f"Missing matchup DB: {matchup_db}")
    query = """
        SELECT game_id,
               CAST(offensive_team_id AS TEXT) AS team_id,
               CAST(offensive_player_id AS TEXT) AS player_id,
               SUM(COALESCE(exposure_seconds, 0.0)) AS exposure_seconds,
               SUM(COALESCE(player_points, 0.0)) AS player_points,
               SUM(COALESCE(matchup_assists, 0.0)) AS matchup_assists,
               SUM(COALESCE(matchup_turnovers, 0.0)) AS matchup_turnovers
        FROM matchup_training_rows
        GROUP BY game_id, offensive_team_id, offensive_player_id
    """
    with sqlite3.connect(matchup_db) as conn:
        df = pd.read_sql_query(query, conn)
    out: dict[str, dict[str, dict[str, tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in df.itertuples(index=False):
        points = float(row.player_points or 0.0)
        assists = float(row.matchup_assists or 0.0)
        turnovers = float(row.matchup_turnovers or 0.0)
        out[str(row.game_id)][str(row.team_id)][str(row.player_id)] = (
            float(row.exposure_seconds or 0.0),
            points,
            _impact_units(points, assists, turnovers),
        )
    return out


def _read_availability(injury_db: Path, *, require: bool) -> dict[str, dict[str, list[tuple[str, str, float | None]]]]:
    if not injury_db.exists():
        if require:
            raise FileNotFoundError(f"Missing injury DB: {injury_db}")
        print(f"[shock] warning: missing injury DB {injury_db}; emitting zero-shock rows")
        return defaultdict(lambda: defaultdict(list))
    query = """
        SELECT game_id, CAST(team_id AS TEXT) AS team_id, CAST(player_id AS TEXT) AS player_id,
               status, minutes_to_tip
        FROM game_player_availability
    """
    try:
        with sqlite3.connect(injury_db) as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as exc:
        if require:
            raise RuntimeError(f"Could not read game_player_availability from {injury_db}: {exc}") from exc
        print(f"[shock] warning: availability table unavailable ({exc}); emitting zero-shock rows")
        return defaultdict(lambda: defaultdict(list))
    out: dict[str, dict[str, list[tuple[str, str, float | None]]]] = defaultdict(lambda: defaultdict(list))
    for row in df.itertuples(index=False):
        minutes = None if pd.isna(row.minutes_to_tip) else float(row.minutes_to_tip)
        out[str(row.game_id)][str(row.team_id)].append((str(row.player_id), str(row.status or ""), minutes))
    return out


def _summarize_team(
    team_id: str,
    status_rows: list[tuple[str, str, float | None]],
    seconds_state: dict[str, dict[str, float]],
    points_state: dict[str, dict[str, float]],
    impact_state: dict[str, dict[str, float]],
    games_seen: dict[str, int],
) -> dict[str, float]:
    sec_map = seconds_state.get(team_id, {})
    pts_map = points_state.get(team_id, {})
    imp_map = impact_state.get(team_id, {})
    rotation_mass = float(sum(max(v, 0.0) for v in sec_map.values()))
    points_mass = float(sum(max(v, 0.0) for v in pts_map.values()))
    impact_mass = float(sum(max(v, 0.0) for v in imp_map.values()))

    report_mass = 0.0
    report_points_mass = 0.0
    report_impact_mass = 0.0
    unavailable_mass = 0.0
    unavailable_points_mass = 0.0
    unavailable_impact_mass = 0.0
    out_mass = 0.0
    uncertain_mass = 0.0
    probable_mass = 0.0
    top_weighted_seconds: list[float] = []
    top_weighted_impact: list[float] = []
    out_count = uncertain_count = probable_count = available_count = 0
    minutes_to_tip_vals: list[float] = []

    for pid, status, minutes_to_tip in status_rows:
        sec = float(max(sec_map.get(pid, 0.0), 0.0))
        pts = float(max(pts_map.get(pid, 0.0), 0.0))
        imp = float(max(imp_map.get(pid, 0.0), 0.0))
        w = _status_unavailable_weight(status)
        bucket = _status_bucket(status)
        report_mass += sec
        report_points_mass += pts
        report_impact_mass += imp
        unavailable_mass += w * sec
        unavailable_points_mass += w * pts
        unavailable_impact_mass += w * imp
        top_weighted_seconds.append(w * sec)
        top_weighted_impact.append(w * imp)
        if bucket == "out":
            out_count += 1
            out_mass += sec
        elif bucket == "uncertain":
            uncertain_count += 1
            uncertain_mass += sec
        elif bucket == "probable":
            probable_count += 1
            probable_mass += sec
        else:
            available_count += 1
        if minutes_to_tip is not None and math.isfinite(minutes_to_tip):
            minutes_to_tip_vals.append(minutes_to_tip)

    top_weighted_seconds.sort(reverse=True)
    top_weighted_impact.sort(reverse=True)
    top1_unavailable = top_weighted_seconds[0] if top_weighted_seconds else 0.0
    top3_unavailable = float(sum(top_weighted_seconds[:3]))
    top1_unavailable_impact = top_weighted_impact[0] if top_weighted_impact else 0.0
    top3_unavailable_impact = float(sum(top_weighted_impact[:3]))
    report_count = float(len(status_rows))
    tracked_players = float(len(sec_map))
    mean_minutes_to_tip = float(sum(minutes_to_tip_vals) / len(minutes_to_tip_vals)) if minutes_to_tip_vals else 0.0
    min_minutes_to_tip = float(min(minutes_to_tip_vals)) if minutes_to_tip_vals else 0.0

    return {
        "rotation_mass_seconds": rotation_mass,
        "rotation_mass_points": points_mass,
        "impact_mass_units": impact_mass,
        "tracked_players": tracked_players,
        "games_seen": float(games_seen.get(team_id, 0)),
        "report_count": report_count,
        "out_count": float(out_count),
        "uncertain_count": float(uncertain_count),
        "probable_count": float(probable_count),
        "available_count": float(available_count),
        "report_rotation_share": _safe_div(report_mass, rotation_mass),
        "report_points_share": _safe_div(report_points_mass, points_mass),
        "report_impact_share": _safe_div(report_impact_mass, impact_mass),
        "unavailable_seconds": unavailable_mass,
        "unavailable_points": unavailable_points_mass,
        "unavailable_impact_units": unavailable_impact_mass,
        "unavailable_seconds_share": _safe_div(unavailable_mass, rotation_mass),
        "unavailable_points_share": _safe_div(unavailable_points_mass, points_mass),
        "unavailable_impact_share": _safe_div(unavailable_impact_mass, impact_mass),
        "out_seconds_share": _safe_div(out_mass, rotation_mass),
        "uncertain_seconds_share": _safe_div(uncertain_mass, rotation_mass),
        "probable_seconds_share": _safe_div(probable_mass, rotation_mass),
        "top1_unavailable_share": _safe_div(top1_unavailable, rotation_mass),
        "top3_unavailable_share": _safe_div(top3_unavailable, rotation_mass),
        "top1_unavailable_impact_share": _safe_div(top1_unavailable_impact, impact_mass),
        "top3_unavailable_impact_share": _safe_div(top3_unavailable_impact, impact_mass),
        "mean_minutes_to_tip": mean_minutes_to_tip,
        "min_minutes_to_tip": min_minutes_to_tip,
        "has_report": 1.0 if status_rows else 0.0,
    }


def _update_team_state(
    team_id: str,
    exposures: dict[str, tuple[float, float, float]],
    seconds_state: dict[str, dict[str, float]],
    points_state: dict[str, dict[str, float]],
    impact_state: dict[str, dict[str, float]],
    games_seen: dict[str, int],
    *,
    decay: float,
) -> None:
    sec_map = seconds_state.setdefault(team_id, {})
    pts_map = points_state.setdefault(team_id, {})
    imp_map = impact_state.setdefault(team_id, {})
    all_pids = set(sec_map) | set(pts_map) | set(imp_map) | set(exposures)
    keep_sec: dict[str, float] = {}
    keep_pts: dict[str, float] = {}
    keep_imp: dict[str, float] = {}
    alpha = 1.0 - float(decay)
    for pid in all_pids:
        cur_sec, cur_pts, cur_imp = exposures.get(pid, (0.0, 0.0, 0.0))
        sec_val = float(decay) * float(sec_map.get(pid, 0.0)) + alpha * float(cur_sec)
        pts_val = float(decay) * float(pts_map.get(pid, 0.0)) + alpha * float(cur_pts)
        imp_val = float(decay) * float(imp_map.get(pid, 0.0)) + alpha * float(cur_imp)
        if sec_val > 1e-6:
            keep_sec[pid] = sec_val
        if pts_val > 1e-6:
            keep_pts[pid] = pts_val
        if imp_val > 1e-6:
            keep_imp[pid] = imp_val
    seconds_state[team_id] = keep_sec
    points_state[team_id] = keep_pts
    impact_state[team_id] = keep_imp
    games_seen[team_id] += 1


def build_rows(
    games: pd.DataFrame,
    exposures: dict[str, dict[str, dict[str, tuple[float, float, float]]]],
    availability: dict[str, dict[str, list[tuple[str, str, float | None]]]],
    *,
    ema_decay: float = 0.80,
) -> pd.DataFrame:
    seconds_state: dict[str, dict[str, float]] = {}
    points_state: dict[str, dict[str, float]] = {}
    impact_state: dict[str, dict[str, float]] = {}
    games_seen: dict[str, int] = defaultdict(int)
    rows: list[dict[str, float | str | int]] = []

    for game in games.itertuples(index=False):
        gid = str(game.game_id)
        home = str(game.home_team_id)
        away = str(game.away_team_id)
        avail_game = availability.get(gid, {})
        h = _summarize_team(home, avail_game.get(home, []), seconds_state, points_state, impact_state, games_seen)
        a = _summarize_team(away, avail_game.get(away, []), seconds_state, points_state, impact_state, games_seen)

        diff_unavailable = a["unavailable_seconds_share"] - h["unavailable_seconds_share"]
        diff_points = a["unavailable_points_share"] - h["unavailable_points_share"]
        diff_impact = a["unavailable_impact_share"] - h["unavailable_impact_share"]
        diff_out = a["out_seconds_share"] - h["out_seconds_share"]
        diff_uncertain = a["uncertain_seconds_share"] - h["uncertain_seconds_share"]
        diff_top1 = a["top1_unavailable_share"] - h["top1_unavailable_share"]
        diff_top3 = a["top3_unavailable_share"] - h["top3_unavailable_share"]
        diff_top1_impact = a["top1_unavailable_impact_share"] - h["top1_unavailable_impact_share"]
        diff_top3_impact = a["top3_unavailable_impact_share"] - h["top3_unavailable_impact_share"]

        # Keep the pre-impact signal for backtest/debug comparisons.
        legacy_advantage_signal = (
            0.55 * diff_unavailable
            + 0.20 * diff_points
            + 0.15 * diff_top1
            + 0.10 * diff_out
        )
        legacy_advantage_signal = _clip(legacy_advantage_signal, -2.0, 2.0)

        # Impact-aware signed scalar for the structured v4/v4.1 linear shock
        # path.  Positive => away more roster-shocked than home => home team
        # should improve.  The impact share is additive rather than replacing
        # the proven rotation/points channels, which preserves continuity with
        # the prior roster-shock feature family.
        advantage_signal = (
            0.30 * diff_unavailable
            + 0.15 * diff_points
            + 0.30 * diff_impact
            + 0.10 * diff_top1_impact
            + 0.05 * diff_top3_impact
            + 0.10 * diff_out
        )
        advantage_signal = _clip(advantage_signal, -2.0, 2.0)

        row: dict[str, float | str | int] = {
            "game_id": gid,
            "game_date": pd.Timestamp(game.game_date).strftime("%Y-%m-%d"),
            "season": int(game.season),
            "home_team_id": home,
            "away_team_id": away,
            "diff_unavailable_seconds_share": diff_unavailable,
            "diff_unavailable_points_share": diff_points,
            "diff_unavailable_impact_share": diff_impact,
            "diff_out_seconds_share": diff_out,
            "diff_uncertain_seconds_share": diff_uncertain,
            "diff_top1_unavailable_share": diff_top1,
            "diff_top3_unavailable_share": diff_top3,
            "diff_top1_unavailable_impact_share": diff_top1_impact,
            "diff_top3_unavailable_impact_share": diff_top3_impact,
            "roster_shock_advantage_signal_legacy": legacy_advantage_signal,
            "roster_shock_advantage_signal": advantage_signal,
            "has_roster_shock_report": 1.0 if (h["has_report"] or a["has_report"]) else 0.0,
        }
        for prefix, summary in (("home", h), ("away", a)):
            for key, val in summary.items():
                row[f"{prefix}_{key}"] = float(val)
        rows.append(row)

        exposure_game = exposures.get(gid, {})
        _update_team_state(home, exposure_game.get(home, {}), seconds_state, points_state, impact_state, games_seen, decay=ema_decay)
        _update_team_state(away, exposure_game.get(away, {}), seconds_state, points_state, impact_state, games_seen, decay=ema_decay)

    return pd.DataFrame(rows)


def write_sqlite(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as conn:
        conn.executescript(f"DROP TABLE IF EXISTS {SHOCK_TABLE};")
        df.to_sql(SHOCK_TABLE, conn, if_exists="replace", index=False)
        conn.executescript(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_roster_shock_game
              ON {SHOCK_TABLE}(game_id);
            CREATE INDEX IF NOT EXISTS idx_roster_shock_date
              ON {SHOCK_TABLE}(game_date, game_id);
            CREATE INDEX IF NOT EXISTS idx_roster_shock_season
              ON {SHOCK_TABLE}(season, game_date, game_id);
        """)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--ema-decay", type=float, default=0.80,
                   help="Game-level EMA decay for recent player exposure/point mass. Higher = slower.")
    p.add_argument("--require-injury-db", action="store_true",
                   help="Fail instead of emitting zero-shock rows if the availability artifact is missing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    games = _read_games(args.core_db)
    exposures = _read_exposures(args.matchup_db)
    availability = _read_availability(args.injury_db, require=args.require_injury_db)
    rows = build_rows(games, exposures, availability, ema_decay=args.ema_decay)
    write_sqlite(rows, args.output)
    print(f"[roster-shock-v4.2] games={len(rows)} output={args.output}")
    if not rows.empty:
        print("[roster-shock-v4.2] date_range=", rows["game_date"].min(), "..", rows["game_date"].max())
        print("[roster-shock-v4.2] report_coverage=", round(float(rows["has_roster_shock_report"].mean()), 4))
        print("[roster-shock-v4.2] mean_advantage_signal=", round(float(rows["roster_shock_advantage_signal"].mean()), 6))
        if "diff_unavailable_impact_share" in rows:
            print("[roster-shock-v4.2] mean_diff_unavailable_impact_share=",
                  round(float(rows["diff_unavailable_impact_share"].mean()), 6))


if __name__ == "__main__":
    main()
