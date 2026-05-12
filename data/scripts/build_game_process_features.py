#!/usr/bin/env python3
"""Build leakage-safe pregame game-process features and process targets.

This artifact is for Process-CME experiments.  It mines *possession/process*
information from the optional `pbpstats_<season>` raw imports when available,
and can fall back to a conservative proxy built from the already-materialized
`nba_core.sqlite` shots/events tables for games without pbpstats coverage.

The output SQLite file contains:

  * team_game_process_actual
      One row per game/team with realized process counts and per-possession
      rates.  `process_source` is either `pbpstats` or `core_proxy`.

  * game_process_targets
      Home/away process targets aligned to model games.

  * game_process_pregame_features
      Leakage-safe rolling team process features built only from games strictly
      prior to the focal game's date/order.  These become extra pregame inputs
      to the Process-CME model.

The key modeling idea is to let a pregame model predict *how a game will be
played* (pace/shot mix/turnovers/rebounding/foul pressure) as an auxiliary task,
then use its predicted process state when estimating win probability.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_CORE_DB = ARTIFACT_ROOT / "nba_core.sqlite"
DEFAULT_RAW_DB = ARTIFACT_ROOT / "nba_raw.sqlite"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "possession_process.sqlite"

PROCESS_METRICS: tuple[str, ...] = (
    "possessions",
    "fg2a_rate",
    "fg3a_rate",
    "turnovers_rate",
    "offensive_rebounds_rate",
    "shooting_fouls_drawn_rate",
)
COUNT_COLUMNS: tuple[str, ...] = (
    "fg2a",
    "fg3a",
    "turnovers",
    "offensive_rebounds",
    "shooting_fouls_drawn",
)
PROCESS_TARGET_COLUMNS: tuple[str, ...] = tuple(
    [f"home_{m}" for m in PROCESS_METRICS]
    + [f"away_{m}" for m in PROCESS_METRICS]
)


def process_feature_columns() -> tuple[str, ...]:
    cols: list[str] = [
        "home_process_games_before",
        "away_process_games_before",
        "diff_process_games_before",
        "home_allowed_process_games_before",
        "away_allowed_process_games_before",
        "diff_allowed_process_games_before",
    ]
    for metric in PROCESS_METRICS:
        cols.extend(
            [
                f"home_own_{metric}_last10",
                f"away_own_{metric}_last10",
                f"diff_own_{metric}_last10",
                f"home_allowed_{metric}_last10",
                f"away_allowed_{metric}_last10",
                f"diff_allowed_{metric}_last10",
                f"home_attack_gap_{metric}_last10",
                f"away_attack_gap_{metric}_last10",
                f"diff_attack_gap_{metric}_last10",
            ]
        )
    return tuple(cols)


PROCESS_PREGAME_FEATURE_COLUMNS = process_feature_columns()


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def list_raw_tables(raw_db: Path, prefix: str = "pbpstats_") -> list[str]:
    if not raw_db.exists():
        return []
    with sqlite3.connect(raw_db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name",
            (f"{prefix}%",),
        ).fetchall()
    # Exclude playoff tables by convention.  This experiment targets the same
    # regular-season game universe as CME backtests.
    return [str(r[0]) for r in rows if "_po_" not in str(r[0])]


def load_core_games(core_db: Path) -> pd.DataFrame:
    if not core_db.exists():
        raise FileNotFoundError(f"Missing core DB: {core_db}")
    with sqlite3.connect(core_db) as conn:
        games = pd.read_sql_query(
            """
            SELECT game_id, season, game_date,
                   home_team_id, away_team_id,
                   home_team_abbr, away_team_abbr
            FROM games
            WHERE game_date IS NOT NULL
              AND home_team_id IS NOT NULL
              AND away_team_id IS NOT NULL
            ORDER BY game_date, game_id
            """,
            conn,
        )
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games


def read_pbpstats_table(raw_db: Path, table: str) -> pd.DataFrame:
    # Possession rows are repeated once per description/event inside a possession
    # in the source archive.  The SELECT DISTINCT collapses them to one row per
    # possession-level process record.
    sql = f"""
        WITH poss AS (
            SELECT DISTINCT
                GAMEID AS game_id,
                OPPONENT AS opponent_abbr,
                PERIOD AS period,
                STARTTIME AS start_time,
                ENDTIME AS end_time,
                STARTTYPE AS start_type,
                STARTSCOREDIFFERENTIAL AS start_score_differential,
                CAST(COALESCE(NULLIF(FG2A, ''), '0') AS REAL) AS fg2a,
                CAST(COALESCE(NULLIF(FG3A, ''), '0') AS REAL) AS fg3a,
                CAST(COALESCE(NULLIF(TURNOVERS, ''), '0') AS REAL) AS turnovers,
                CAST(COALESCE(NULLIF(OFFENSIVEREBOUNDS, ''), '0') AS REAL) AS offensive_rebounds,
                CAST(COALESCE(NULLIF(SHOOTINGFOULSDRAWN, ''), '0') AS REAL) AS shooting_fouls_drawn
            FROM {quote_ident(table)}
            WHERE GAMEID IS NOT NULL AND GAMEID <> ''
              AND OPPONENT IS NOT NULL AND OPPONENT <> ''
        )
        SELECT
            game_id,
            opponent_abbr,
            COUNT(*) AS possessions,
            SUM(fg2a) AS fg2a,
            SUM(fg3a) AS fg3a,
            SUM(turnovers) AS turnovers,
            SUM(offensive_rebounds) AS offensive_rebounds,
            SUM(shooting_fouls_drawn) AS shooting_fouls_drawn
        FROM poss
        GROUP BY game_id, opponent_abbr
    """
    with sqlite3.connect(raw_db) as conn:
        df = pd.read_sql_query(sql, conn)
    df["game_id"] = df["game_id"].astype(str)
    return df


def pbpstats_team_actuals(raw_db: Path, core_games: pd.DataFrame) -> pd.DataFrame:
    tables = list_raw_tables(raw_db)
    if not tables:
        return pd.DataFrame()
    chunks = [read_pbpstats_table(raw_db, table) for table in tables]
    poss = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if poss.empty:
        return poss

    games = core_games[
        [
            "game_id", "season", "game_date", "home_team_id", "away_team_id",
            "home_team_abbr", "away_team_abbr",
        ]
    ].copy()
    merged = poss.merge(games, on="game_id", how="inner")
    # In pbpstats, OPPONENT is the defensive team.  Thus if OPPONENT equals the
    # home tricode, the possession belonged to the away team, and vice versa.
    home_def = merged["opponent_abbr"].astype(str) == merged["home_team_abbr"].astype(str)
    away_def = merged["opponent_abbr"].astype(str) == merged["away_team_abbr"].astype(str)
    valid = home_def | away_def
    merged = merged.loc[valid].copy()
    home_def = home_def.loc[valid]
    away_def = away_def.loc[valid]
    merged["team_id"] = np.where(home_def, merged["away_team_id"], merged["home_team_id"])
    merged["opponent_team_id"] = np.where(home_def, merged["home_team_id"], merged["away_team_id"])
    merged["is_home"] = np.where(home_def, 0, 1).astype(int)
    merged["process_source"] = "pbpstats"
    keep = [
        "game_id", "season", "game_date", "team_id", "opponent_team_id", "is_home",
        "process_source", "possessions", *COUNT_COLUMNS,
    ]
    return merged[keep]


def _read_core_proxy_shots(core_db: Path) -> pd.DataFrame:
    with sqlite3.connect(core_db) as conn:
        shots = pd.read_sql_query(
            """
            SELECT
                game_id,
                team_id,
                SUM(CASE WHEN shot_attempted = 1 AND UPPER(COALESCE(shot_type, '')) LIKE '2PT%' THEN 1 ELSE 0 END) AS fg2a,
                SUM(CASE WHEN shot_attempted = 1 AND UPPER(COALESCE(shot_type, '')) LIKE '3PT%' THEN 1 ELSE 0 END) AS fg3a
            FROM shots
            WHERE team_id IS NOT NULL
            GROUP BY game_id, team_id
            """,
            conn,
        )
    shots["game_id"] = shots["game_id"].astype(str)
    return shots


def _read_core_proxy_events(core_db: Path) -> pd.DataFrame:
    # cdnnba action/sub-type vocabularies can vary over seasons, so use a few
    # conservative string matches.  These are proxies, not replacements for
    # pbpstats, and the output marks them as `core_proxy`.
    with sqlite3.connect(core_db) as conn:
        events = pd.read_sql_query(
            """
            SELECT
                game_id,
                team_id,
                SUM(CASE WHEN LOWER(COALESCE(action_type, '')) LIKE '%turnover%' THEN 1 ELSE 0 END) AS turnovers,
                SUM(CASE
                    WHEN LOWER(COALESCE(action_type, '')) LIKE '%rebound%'
                     AND (
                        LOWER(COALESCE(sub_type, '')) LIKE '%off%'
                        OR LOWER(COALESCE(description, '')) LIKE '%offensive rebound%'
                     )
                    THEN 1 ELSE 0 END) AS offensive_rebounds,
                SUM(CASE
                    WHEN LOWER(COALESCE(action_type, '')) LIKE '%foul%'
                     AND (
                        LOWER(COALESCE(sub_type, '')) LIKE '%shoot%'
                        OR LOWER(COALESCE(description, '')) LIKE '%shooting foul%'
                     )
                    THEN 1 ELSE 0 END) AS shooting_fouls_committed,
                SUM(CASE
                    WHEN LOWER(COALESCE(action_type, '')) LIKE '%freethrow%'
                      OR LOWER(COALESCE(action_type, '')) LIKE '%free throw%'
                      OR LOWER(COALESCE(description, '')) LIKE '%free throw%'
                    THEN 1 ELSE 0 END) AS fta_proxy
            FROM game_events
            WHERE team_id IS NOT NULL
            GROUP BY game_id, team_id
            """,
            conn,
        )
    events["game_id"] = events["game_id"].astype(str)
    return events


def core_proxy_team_actuals(core_db: Path, core_games: pd.DataFrame) -> pd.DataFrame:
    shots = _read_core_proxy_shots(core_db)
    events = _read_core_proxy_events(core_db)
    raw = shots.merge(events, on=["game_id", "team_id"], how="outer")
    if raw.empty:
        return raw
    for c in ["fg2a", "fg3a", "turnovers", "offensive_rebounds", "shooting_fouls_committed", "fta_proxy"]:
        if c not in raw:
            raw[c] = 0.0
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
    # Approximate team possessions via the standard box-score proxy.
    raw["possessions"] = (
        raw["fg2a"] + raw["fg3a"] + 0.44 * raw["fta_proxy"]
        - raw["offensive_rebounds"] + raw["turnovers"]
    ).clip(lower=1.0)

    games = core_games[
        ["game_id", "season", "game_date", "home_team_id", "away_team_id"]
    ].copy()
    committed = raw[["game_id", "team_id", "shooting_fouls_committed"]].rename(
        columns={"team_id": "opponent_team_id", "shooting_fouls_committed": "shooting_fouls_drawn"}
    )
    home = raw.merge(games, left_on=["game_id", "team_id"], right_on=["game_id", "home_team_id"], how="inner")
    home["opponent_team_id"] = home["away_team_id"]
    home["is_home"] = 1
    home = home.merge(committed, on=["game_id", "opponent_team_id"], how="left")
    away = raw.merge(games, left_on=["game_id", "team_id"], right_on=["game_id", "away_team_id"], how="inner")
    away["opponent_team_id"] = away["home_team_id"]
    away["is_home"] = 0
    away = away.merge(committed, on=["game_id", "opponent_team_id"], how="left")
    merged = pd.concat([home, away], ignore_index=True)
    merged["shooting_fouls_drawn"] = pd.to_numeric(merged["shooting_fouls_drawn"], errors="coerce").fillna(0.0)
    merged["process_source"] = "core_proxy"
    keep = [
        "game_id", "season", "game_date", "team_id", "opponent_team_id", "is_home",
        "process_source", "possessions", *COUNT_COLUMNS,
    ]
    return merged[keep]


def finalize_actuals(actuals: pd.DataFrame, min_possessions: float) -> pd.DataFrame:
    if actuals.empty:
        return actuals
    out = actuals.copy()
    out["game_id"] = out["game_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    out["opponent_team_id"] = out["opponent_team_id"].astype(str)
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.strftime("%Y-%m-%d")
    for c in ["possessions", *COUNT_COLUMNS]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    denom = out["possessions"].clip(lower=1.0)
    out["fg2a_rate"] = out["fg2a"] / denom
    out["fg3a_rate"] = out["fg3a"] / denom
    out["turnovers_rate"] = out["turnovers"] / denom
    out["offensive_rebounds_rate"] = out["offensive_rebounds"] / denom
    out["shooting_fouls_drawn_rate"] = out["shooting_fouls_drawn"] / denom
    out["has_process_actual"] = (out["possessions"] >= float(min_possessions)).astype(int)
    ordered = [
        "game_id", "season", "game_date", "team_id", "opponent_team_id", "is_home",
        "process_source", "has_process_actual", "possessions", *COUNT_COLUMNS,
        "fg2a_rate", "fg3a_rate", "turnovers_rate", "offensive_rebounds_rate",
        "shooting_fouls_drawn_rate",
    ]
    out = out[ordered]
    out = out.sort_values(["game_date", "game_id", "is_home"], ascending=[True, True, False])
    out = out.drop_duplicates(["game_id", "team_id"], keep="first").reset_index(drop=True)
    return out


def combine_actuals(
    pbpstats_actuals: pd.DataFrame,
    core_proxy_actuals: pd.DataFrame,
    *,
    include_core_proxy: bool,
    min_possessions: float,
) -> pd.DataFrame:
    exact = finalize_actuals(pbpstats_actuals, min_possessions)
    if not include_core_proxy:
        return exact
    proxy = finalize_actuals(core_proxy_actuals, min_possessions)
    if exact.empty:
        return proxy
    if proxy.empty:
        return exact
    exact_keys = set(zip(exact["game_id"].astype(str), exact["team_id"].astype(str)))
    keep_proxy = [
        (str(g), str(t)) not in exact_keys
        for g, t in zip(proxy["game_id"], proxy["team_id"])
    ]
    proxy = proxy.loc[keep_proxy].copy()
    return pd.concat([exact, proxy], ignore_index=True).sort_values(
        ["game_date", "game_id", "is_home"], ascending=[True, True, False]
    ).reset_index(drop=True)


def build_game_targets(actuals: pd.DataFrame, core_games: pd.DataFrame) -> pd.DataFrame:
    if actuals.empty:
        return pd.DataFrame(columns=["game_id", *PROCESS_TARGET_COLUMNS, "process_target_valid"])
    metrics = list(PROCESS_METRICS)
    home = actuals.loc[actuals["is_home"] == 1, ["game_id", "has_process_actual", *metrics]].copy()
    away = actuals.loc[actuals["is_home"] == 0, ["game_id", "has_process_actual", *metrics]].copy()
    home = home.rename(columns={**{m: f"home_{m}" for m in metrics}, "has_process_actual": "home_has_process_actual"})
    away = away.rename(columns={**{m: f"away_{m}" for m in metrics}, "has_process_actual": "away_has_process_actual"})
    targets = core_games[["game_id"]].merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
    targets["process_target_valid"] = (
        targets["home_has_process_actual"].fillna(0).astype(int)
        * targets["away_has_process_actual"].fillna(0).astype(int)
    )
    for c in PROCESS_TARGET_COLUMNS:
        if c not in targets:
            targets[c] = np.nan
    keep = ["game_id", *PROCESS_TARGET_COLUMNS, "process_target_valid"]
    return targets[keep].copy()


def _rolling_last10(group: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    group = group.sort_values(["game_date", "game_id"]).copy()
    group["process_games_before"] = np.arange(len(group), dtype=float)
    for metric in metrics:
        group[f"{metric}_last10"] = (
            group[metric]
            .shift(1)
            .rolling(window=10, min_periods=1)
            .mean()
        )
    return group


def build_team_history_table(actuals: pd.DataFrame, *, allowed: bool) -> pd.DataFrame:
    metrics = list(PROCESS_METRICS)
    if actuals.empty:
        cols = ["game_id", "team_id", "process_games_before", *[f"{m}_last10" for m in metrics]]
        return pd.DataFrame(columns=cols)
    if allowed:
        df = actuals[
            ["game_id", "game_date", "opponent_team_id", *metrics]
        ].rename(columns={"opponent_team_id": "team_id"})
    else:
        df = actuals[["game_id", "game_date", "team_id", *metrics]].copy()
    parts = [_rolling_last10(g, metrics) for _, g in df.groupby("team_id", sort=False)]
    hist = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    keep = ["game_id", "team_id", "process_games_before", *[f"{m}_last10" for m in metrics]]
    return hist[keep].copy()


def build_pregame_features(actuals: pd.DataFrame, core_games: pd.DataFrame) -> pd.DataFrame:
    own = build_team_history_table(actuals, allowed=False)
    allowed = build_team_history_table(actuals, allowed=True)
    metrics = list(PROCESS_METRICS)

    games = core_games[["game_id", "home_team_id", "away_team_id"]].copy()
    h_own = own.rename(columns={
        "team_id": "home_team_id",
        "process_games_before": "home_process_games_before",
        **{f"{m}_last10": f"home_own_{m}_last10" for m in metrics},
    })
    a_own = own.rename(columns={
        "team_id": "away_team_id",
        "process_games_before": "away_process_games_before",
        **{f"{m}_last10": f"away_own_{m}_last10" for m in metrics},
    })
    h_allowed = allowed.rename(columns={
        "team_id": "home_team_id",
        "process_games_before": "home_allowed_process_games_before",
        **{f"{m}_last10": f"home_allowed_{m}_last10" for m in metrics},
    })
    a_allowed = allowed.rename(columns={
        "team_id": "away_team_id",
        "process_games_before": "away_allowed_process_games_before",
        **{f"{m}_last10": f"away_allowed_{m}_last10" for m in metrics},
    })

    out = games.merge(h_own, on=["game_id", "home_team_id"], how="left")
    out = out.merge(a_own, on=["game_id", "away_team_id"], how="left")
    out = out.merge(h_allowed, on=["game_id", "home_team_id"], how="left")
    out = out.merge(a_allowed, on=["game_id", "away_team_id"], how="left")

    out["diff_process_games_before"] = out["home_process_games_before"] - out["away_process_games_before"]
    out["diff_allowed_process_games_before"] = (
        out["home_allowed_process_games_before"] - out["away_allowed_process_games_before"]
    )
    for m in metrics:
        out[f"diff_own_{m}_last10"] = out[f"home_own_{m}_last10"] - out[f"away_own_{m}_last10"]
        out[f"diff_allowed_{m}_last10"] = out[f"home_allowed_{m}_last10"] - out[f"away_allowed_{m}_last10"]
        out[f"home_attack_gap_{m}_last10"] = out[f"home_own_{m}_last10"] - out[f"away_allowed_{m}_last10"]
        out[f"away_attack_gap_{m}_last10"] = out[f"away_own_{m}_last10"] - out[f"home_allowed_{m}_last10"]
        out[f"diff_attack_gap_{m}_last10"] = (
            out[f"home_attack_gap_{m}_last10"] - out[f"away_attack_gap_{m}_last10"]
        )
    keep = ["game_id", *PROCESS_PREGAME_FEATURE_COLUMNS]
    for c in keep:
        if c not in out:
            out[c] = np.nan
    return out[keep].copy()


def write_artifact(
    output: Path,
    actuals: pd.DataFrame,
    targets: pd.DataFrame,
    pregame: pd.DataFrame,
    *,
    raw_tables: list[str],
    include_core_proxy: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS process_builds;
            DROP TABLE IF EXISTS team_game_process_actual;
            DROP TABLE IF EXISTS game_process_targets;
            DROP TABLE IF EXISTS game_process_pregame_features;

            CREATE TABLE process_builds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                raw_pbpstats_tables TEXT,
                include_core_proxy INTEGER NOT NULL,
                notes TEXT
            );
            """
        )
        actuals.to_sql("team_game_process_actual", conn, index=False, if_exists="replace")
        targets.to_sql("game_process_targets", conn, index=False, if_exists="replace")
        pregame.to_sql("game_process_pregame_features", conn, index=False, if_exists="replace")
        conn.execute(
            "INSERT INTO process_builds (raw_pbpstats_tables, include_core_proxy, notes) VALUES (?, ?, ?)",
            (
                ",".join(raw_tables),
                int(include_core_proxy),
                "Process-CME possession/process artifact",
            ),
        )
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_team_game_process_actual_game
                ON team_game_process_actual (game_id);
            CREATE INDEX IF NOT EXISTS idx_team_game_process_actual_team_date
                ON team_game_process_actual (team_id, game_date);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_game_process_targets_game
                ON game_process_targets (game_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_game_process_pregame_game
                ON game_process_pregame_features (game_id);
            """
        )
        conn.commit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--raw-db", type=Path, default=DEFAULT_RAW_DB)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--min-possessions", type=float, default=40.0,
                   help="Minimum realized possessions per team for process-target supervision.")
    p.add_argument("--no-core-proxy", action="store_true",
                   help="Use only imported pbpstats tables. By default, missing games are filled with a conservative core-table proxy.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    games = load_core_games(args.core_db)
    raw_tables = list_raw_tables(args.raw_db)
    print(f"[load] core games={len(games)} raw_pbpstats_tables={len(raw_tables)}")
    pbpstats = pbpstats_team_actuals(args.raw_db, games)
    print(f"[pbpstats] team-game rows={len(pbpstats)}")
    proxy = pd.DataFrame()
    if not args.no_core_proxy:
        proxy = core_proxy_team_actuals(args.core_db, games)
        print(f"[core-proxy] team-game rows={len(proxy)}")
    actuals = combine_actuals(
        pbpstats, proxy,
        include_core_proxy=not args.no_core_proxy,
        min_possessions=args.min_possessions,
    )
    if actuals.empty:
        raise RuntimeError(
            "No process actuals could be built. Import pbpstats_<season> archives into nba_raw.sqlite "
            "or rerun without --no-core-proxy."
        )
    targets = build_game_targets(actuals, games)
    pregame = build_pregame_features(actuals, games)
    write_artifact(
        args.output, actuals, targets, pregame,
        raw_tables=raw_tables, include_core_proxy=not args.no_core_proxy,
    )
    valid_targets = int(targets["process_target_valid"].fillna(0).sum()) if not targets.empty else 0
    exact_rows = int((actuals["process_source"] == "pbpstats").sum())
    proxy_rows = int((actuals["process_source"] == "core_proxy").sum())
    print(f"[done] {args.output}")
    print(f"[done] team_game_process_actual rows={len(actuals)} exact_rows={exact_rows} proxy_rows={proxy_rows}")
    print(f"[done] game_process_targets rows={len(targets)} valid_games={valid_targets}")
    print(f"[done] game_process_pregame_features rows={len(pregame)} features={len(PROCESS_PREGAME_FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
