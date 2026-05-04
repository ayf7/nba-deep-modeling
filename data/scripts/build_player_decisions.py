"""Build ESPN<->NBA-Stats id crosswalks and a `game_player_decisions` table.

Sources:
    data/artifacts/hoopr_player_box.sqlite   (raw ESPN/hoopR rows)
    data/artifacts/nba_core.sqlite           (NBA Stats games + players)
    data/artifacts/player_game_stats.sqlite  (per-game played rows, full names)

Output:
    data/artifacts/player_decisions.sqlite
        game_id_crosswalk(espn_game_id, nba_game_id, game_date)
        player_id_crosswalk(espn_athlete_id, nba_player_id, n_matches)
        game_player_decisions(
            game_id,         -- NBA Stats game_id
            player_id,       -- NBA Stats player_id
            team_id, team_abbr, season,
            dressed,         -- always 1 (row exists in hoopR player_box)
            played,          -- 1 if did_not_play=0
            did_not_play,    -- 0/1
            reason,          -- "COACH'S DECISION", "LEFT ANKLE SPRAIN", etc.
            starter, ejected, minutes
        )

Matching strategy:
    1. Game match: (game_date, home_abbr, away_abbr) - unique within season.
    2. Player match: ASCII-normalized (athlete_display_name, team_abbr, game_date)
       joined to player_game_stats (game_id, team_tricode, normalized_name).
       Aggregate by modal NBA player_id per ESPN athlete_id.
"""
from __future__ import annotations

import argparse
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd


TEAM_ABBR_MAP = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def normalize_team(abbr: str) -> str:
    if abbr is None:
        return ""
    return TEAM_ABBR_MAP.get(abbr, abbr)


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    # German romanization: ESPN uses "Poeltl", NBA Stats keeps "Pöltl" (→ "poltl").
    s = s.replace("oe", "o").replace("ue", "u").replace("ae", "a")
    parts = [p for p in s.split() if p not in _NAME_SUFFIXES]
    return " ".join(parts)


# Manual ESPN-name -> NBA-Stats pid overrides. Used for cases where the ESPN
# display name and the NBA Stats canonical name diverge in ways that neither
# full-name nor last-name matching can reconcile (true renames, mononyms).
# Lookup runs BEFORE pass-1 matching; a hit binds the row directly to the pid
# and skips both pass-1 and pass-2 for that row. Keys are normalize_name()'d.
_NAME_OVERRIDES: dict[str, str] = {
    # 2021 rename: ESPN kept "Enes Freedom" (or "Enes Kanter Freedom"),
    # NBA Stats kept "Kanter".
    normalize_name("Enes Freedom"): "202683",
    normalize_name("Enes Kanter Freedom"): "202683",
    # Mononym handling differs: ESPN uses "Nene", NBA Stats uses "Nene Hilario".
    normalize_name("Nene"): "2403",
}


def initial_form(name_norm: str) -> str:
    """Build the "F. Last" initial-form key from a normalized full name.

    Matches the convention used in nba_core.players.player_name_i (e.g.
    "T. Young", "E. Mobley"). Returns "" if the name has fewer than 2 tokens.
    The result is lower-cased to match how we'll compare it (we lowercase the
    NBA Stats column at lookup time too).
    """
    parts = name_norm.split()
    if len(parts) < 2:
        return ""
    return f"{parts[0][0]}. {parts[-1]}"


def normalize_initial(s: str) -> str:
    """Normalize an NBA Stats player_name_i string (e.g. "G. Trent Jr.") into
    the same form initial_form() emits: lowercase, accent-stripped, suffixes
    removed, first-token reduced to "X.". This is what makes
    "G. Trent Jr." compare equal to initial_form("Gary Trent Jr.").
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    s = s.replace("oe", "o").replace("ue", "u").replace("ae", "a")
    parts = [p for p in s.split() if p not in _NAME_SUFFIXES]
    if len(parts) < 2:
        return ""
    return f"{parts[0][0]}. {parts[-1]}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hoopr-db", type=Path,
                   default=Path("data/artifacts/hoopr_player_box.sqlite"))
    p.add_argument("--core-db", type=Path,
                   default=Path("data/artifacts/nba_core.sqlite"))
    p.add_argument("--pgs-db", type=Path,
                   default=Path("data/artifacts/player_game_stats.sqlite"))
    p.add_argument("--output", type=Path,
                   default=Path("data/artifacts/player_decisions.sqlite"))
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    print("[load] hoopR player_box + schedules")
    with sqlite3.connect(args.hoopr_db) as conn:
        hp = pd.read_sql("SELECT * FROM player_box", conn)
        hs = pd.read_sql("SELECT * FROM schedules", conn)
    print(f"        player_box rows={len(hp)}   schedules rows={len(hs)}")

    print("[load] nba_core games + players")
    with sqlite3.connect(args.core_db) as conn:
        nba_games = pd.read_sql(
            "SELECT game_id, game_date, home_team_abbr, away_team_abbr, season "
            "FROM games", conn)
        nba_players = pd.read_sql(
            "SELECT player_id, player_name, player_name_i FROM players", conn)
    print(f"        games rows={len(nba_games)}   players rows={len(nba_players)}")

    print("[load] player_game_stats")
    with sqlite3.connect(args.pgs_db) as conn:
        pgs = pd.read_sql(
            "SELECT game_id, game_date, team_id, team_tricode, player_id, "
            "player_name, minutes_played FROM player_game_stats", conn)
    print(f"        player_game_stats rows={len(pgs)}")

    # ---- Game crosswalk: (game_date, home_abbr, away_abbr) -----
    print("\n[crosswalk] games")
    hs_norm = hs[["espn_game_id", "game_date", "home_abbreviation",
                  "away_abbreviation", "season"]].copy()
    hs_norm["game_date"] = pd.to_datetime(hs_norm["game_date"]).dt.strftime("%Y-%m-%d")
    hs_norm["home_abbreviation"] = hs_norm["home_abbreviation"].map(normalize_team)
    hs_norm["away_abbreviation"] = hs_norm["away_abbreviation"].map(normalize_team)
    nba_games["game_date"] = pd.to_datetime(nba_games["game_date"]).dt.strftime("%Y-%m-%d")
    gx = hs_norm.merge(
        nba_games[["game_id", "game_date", "home_team_abbr", "away_team_abbr"]],
        left_on=["game_date", "home_abbreviation", "away_abbreviation"],
        right_on=["game_date", "home_team_abbr", "away_team_abbr"],
        how="left",
    )
    matched = gx["game_id"].notna().sum()
    print(f"        matched {matched}/{len(gx)} games  "
          f"({100*matched/len(gx):.1f}%)")
    gx_out = gx[["espn_game_id", "game_id", "game_date"]].dropna(
        subset=["game_id"]).rename(columns={"game_id": "nba_game_id"})
    gx_out["espn_game_id"] = gx_out["espn_game_id"].astype("int64")
    print(f"        crosswalk size: {len(gx_out)}")

    # ---- Player crosswalk via (game, team, name) ----
    print("\n[crosswalk] players")
    espn_to_nba_gid = dict(zip(gx_out["espn_game_id"], gx_out["nba_game_id"]))

    hp_match = hp[["espn_game_id", "espn_athlete_id", "athlete_display_name",
                   "team_abbreviation", "season"]].copy()
    hp_match["team_abbreviation"] = hp_match["team_abbreviation"].map(normalize_team)
    hp_match["espn_game_id"] = hp_match["espn_game_id"].astype("int64")
    hp_match["nba_game_id"] = hp_match["espn_game_id"].map(espn_to_nba_gid)
    hp_match = hp_match.dropna(subset=["nba_game_id"])
    hp_match["nba_game_id"] = hp_match["nba_game_id"].astype(str)
    hp_match["name_norm"] = hp_match["athlete_display_name"].map(normalize_name)
    hp_match["last_norm"] = hp_match["name_norm"].str.split().str[-1]
    hp_match["init_form"] = hp_match["name_norm"].map(initial_form)

    pgs_match = pgs[["game_id", "team_tricode", "player_id", "player_name"]].copy()
    pgs_match["game_id"] = pgs_match["game_id"].astype(str)
    pgs_match["name_norm"] = pgs_match["player_name"].map(normalize_name)
    pgs_match["last_norm"] = pgs_match["name_norm"].str.split().str[-1]

    # Initial-form lookup table built from nba_core.players.player_name_i
    # (e.g. "T. Young", "E. Mobley"). Normalized with normalize_initial() so
    # accents and suffixes get stripped the same way they are on the hp side
    # (e.g. "G. Trent Jr." -> "g. trent"). Used to disambiguate last-name
    # collisions in pass 1.5 below.
    np_init = nba_players[["player_id", "player_name_i"]].copy()
    np_init["player_id"] = np_init["player_id"].astype(str)
    np_init["init_form_norm"] = np_init["player_name_i"].map(normalize_initial)
    np_init = np_init[np_init["init_form_norm"] != ""]

    # Pass 0: manual override for true renames and mononym mismatches.
    # Binds (espn_athlete_id, game) directly to a pid from _NAME_OVERRIDES,
    # skipping pass-1/1.5/2 for the matched rows entirely.
    override_mask = hp_match["name_norm"].isin(_NAME_OVERRIDES)
    pm_override = hp_match[override_mask].copy()
    pm_override["player_id"] = pm_override["name_norm"].map(_NAME_OVERRIDES)
    pm_override["game_id"] = pm_override["nba_game_id"]
    pm_override["team_tricode"] = pm_override["team_abbreviation"]
    hp_match_rem = hp_match[~override_mask]

    # Pass 1: full normalized name match
    pm_full = hp_match_rem.merge(
        pgs_match[["game_id", "team_tricode", "name_norm", "player_id"]],
        left_on=["nba_game_id", "team_abbreviation", "name_norm"],
        right_on=["game_id", "team_tricode", "name_norm"],
        how="inner",
    )

    # Pass 1.5: initial-form ("F. Last") match against
    # nba_core.players.player_name_i. Disambiguates last-name collisions like
    # "Trae Young" vs "James Young", "Evan Mobley" vs "Isaiah Mobley". An
    # initial-form key is unique only when exactly one pid in nba_players
    # carries that "F. Last" string; we drop ambiguous keys to keep the match
    # safe. Runs only on rows not bound by pass 0 or pass 1.
    init_unique = (np_init.groupby("init_form_norm")["player_id"]
                          .nunique().reset_index(name="n_pids"))
    init_unique = init_unique[init_unique["n_pids"] == 1]["init_form_norm"]
    np_init_unique = np_init[np_init["init_form_norm"].isin(init_unique)]

    matched_athletes_pass1 = set(pm_full["espn_athlete_id"].unique())
    leftover_for_init = hp_match_rem[
        (~hp_match_rem["espn_athlete_id"].isin(matched_athletes_pass1))
        & (hp_match_rem["init_form"] != "")
    ]
    pm_init = leftover_for_init.merge(
        np_init_unique[["init_form_norm", "player_id"]],
        left_on="init_form",
        right_on="init_form_norm",
        how="inner",
    )
    # Mirror the column layout of pm_full/pm_last so concat works cleanly.
    pm_init["game_id"] = pm_init["nba_game_id"]
    pm_init["team_tricode"] = pm_init["team_abbreviation"]

    # Pass 2: last-word-only match within (game, team), used for athletes
    # still unmatched after passes 0, 1, and 1.5.
    matched_athletes_pass15 = (matched_athletes_pass1
                               | set(pm_init["espn_athlete_id"].unique()))
    leftover = hp_match_rem[
        ~hp_match_rem["espn_athlete_id"].isin(matched_athletes_pass15)
    ]
    pm_last = leftover.merge(
        pgs_match[["game_id", "team_tricode", "last_norm", "player_id"]],
        left_on=["nba_game_id", "team_abbreviation", "last_norm"],
        right_on=["game_id", "team_tricode", "last_norm"],
        how="inner",
    )

    # Exclude (game, pid) cells already claimed by Pass 0/1/1.5. Prevents
    # Bronny ("Bronny James" in ESPN, "James" in NBA Stats) from latching onto
    # LeBron's pid in same-game father/son rows, where pass-2 last-name match
    # would otherwise tie LeBron's count.
    claimed = (set(zip(pm_full["game_id"].astype(str),
                       pm_full["player_id"].astype(str)))
               | set(zip(pm_init["game_id"].astype(str),
                         pm_init["player_id"].astype(str)))
               | set(zip(pm_override["game_id"].astype(str),
                         pm_override["player_id"].astype(str))))
    p2_cells = list(zip(pm_last["game_id"].astype(str),
                         pm_last["player_id"].astype(str)))
    p2_keep = [c not in claimed for c in p2_cells]
    n_dropped = len(pm_last) - sum(p2_keep)
    pm_last = pm_last[p2_keep].reset_index(drop=True)

    pm = pd.concat([pm_override, pm_full, pm_init, pm_last], ignore_index=True)
    matched_rows = len(pm)
    total_rows = len(hp_match)
    print(f"        pass0 (override) : {len(pm_override)} matched rows")
    print(f"        pass1 (full name): {len(pm_full)} matched rows")
    print(f"        pass1.5 (init.)  : {len(pm_init)} matched rows")
    print(f"        pass2 (last name): {len(pm_last)} matched rows "
          f"(after dropping {n_dropped} earlier-claimed (game,pid) cells)")
    print(f"        total matched {matched_rows}/{total_rows}  "
          f"({100*matched_rows/total_rows:.1f}%)")

    # Aggregate to (espn_athlete_id -> modal nba_player_id)
    px = (pm.groupby(["espn_athlete_id", "player_id"])
            .size().reset_index(name="n"))
    px = px.sort_values(["espn_athlete_id", "n"], ascending=[True, False])
    px_best = px.drop_duplicates("espn_athlete_id", keep="first").rename(
        columns={"player_id": "nba_player_id", "n": "n_matches"})

    # Enforce reverse uniqueness: each NBA pid should be claimed by at most one
    # ESPN id. When several ESPN ids claim the same NBA pid (e.g. "Carlik Jones"
    # and "Derrick Jones Jr." both land on NBA "Jones Jr." via pass-2 last-name
    # match, or a player with two ESPN ids over their career), keep only the
    # highest-count claimant. The loser becomes unresolved.
    px_best = px_best.sort_values("n_matches", ascending=False)
    n_before_rev = len(px_best)
    px_best = px_best.drop_duplicates("nba_player_id", keep="first")
    n_rev_dropped = n_before_rev - len(px_best)
    print(f"        crosswalk: {len(px_best)} unique espn_athlete_ids "
          f"resolved to nba player_id  "
          f"(dropped {n_rev_dropped} secondary claims on shared NBA pid)")

    n_unresolved = hp["espn_athlete_id"].nunique() - len(px_best)
    print(f"        unresolved espn_athlete_ids: {n_unresolved}")

    # ---- game_player_decisions enriched table ----
    print("\n[build] game_player_decisions")
    espn_to_nba_pid = dict(zip(px_best["espn_athlete_id"],
                               px_best["nba_player_id"]))
    teams_lookup = dict(
        pgs[["team_tricode", "team_id"]].drop_duplicates().itertuples(
            index=False, name=None))

    out = hp.copy()
    out["team_abbreviation"] = out["team_abbreviation"].map(normalize_team)
    out["espn_game_id"] = out["espn_game_id"].astype("int64")
    out["game_id"] = out["espn_game_id"].map(espn_to_nba_gid)
    out["player_id"] = out["espn_athlete_id"].map(espn_to_nba_pid)
    out["team_id"] = out["team_abbreviation"].map(teams_lookup)
    out = out.dropna(subset=["game_id", "player_id"]).copy()
    out["played"] = (out["did_not_play"].astype(int) == 0).astype(int)
    out["did_not_play"] = out["did_not_play"].astype(int)
    out["dressed"] = 1
    out["starter"] = out["starter"].fillna(False).astype(int)
    out["ejected"] = out["ejected"].fillna(False).astype(int)
    out["reason"] = out["reason"].where(out["did_not_play"] == 1, None)
    out["minutes"] = out["minutes"].fillna(0.0).astype(float)

    keep = ["game_id", "player_id", "team_id", "team_abbreviation", "season",
            "dressed", "played", "did_not_play", "reason", "starter",
            "ejected", "minutes", "espn_game_id", "espn_athlete_id"]
    out = out[keep].rename(columns={"team_abbreviation": "team_abbr"})
    out["game_id"] = out["game_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    cd = "COACH'S DECISION"
    n_played = int(out["played"].sum())
    n_dnpcd = int(((out["did_not_play"] == 1) & (out["reason"] == cd)).sum())
    n_inactive = int(((out["did_not_play"] == 1) & (out["reason"] != cd)).sum())
    print(f"        rows     : {len(out)}")
    print(f"        played   : {n_played}")
    print(f"        DNP-CD   : {n_dnpcd}")
    print(f"        Inactive : {n_inactive}")

    print(f"\n[write] {args.output}")
    with sqlite3.connect(args.output) as conn:
        gx_out.to_sql("game_id_crosswalk", conn, index=False)
        px_best.to_sql("player_id_crosswalk", conn, index=False)
        out.to_sql("game_player_decisions", conn, index=False)
        conn.execute("CREATE INDEX idx_gx_espn ON game_id_crosswalk(espn_game_id)")
        conn.execute("CREATE INDEX idx_gx_nba  ON game_id_crosswalk(nba_game_id)")
        conn.execute("CREATE INDEX idx_px_espn ON player_id_crosswalk(espn_athlete_id)")
        conn.execute("CREATE INDEX idx_px_nba  ON player_id_crosswalk(nba_player_id)")
        conn.execute("CREATE INDEX idx_gpd_game ON game_player_decisions(game_id)")
        conn.execute("CREATE INDEX idx_gpd_pid  ON game_player_decisions(game_id, player_id)")
    print("[done]")


if __name__ == "__main__":
    main()
