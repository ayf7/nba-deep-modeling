#!/usr/bin/env python3
"""Counterfactual analysis with CME-v6.

Loads a checkpoint, runs inference on test games with modified lineups:
  - Remove a player (injury scenario)
  - Swap a player between teams (trade scenario)
  - Build a custom roster (super team)

Compares win probabilities before/after the modification.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cme_v5_common import (
    PrecomputedDatasetV5, collate_v5,
    load_precomputed_vocab_size,
)
from model import CmeV6, CmeV6Config

DEFAULT_PRECOMPUTED_DB = REPO_ROOT / "data" / "features_v5_precomputed.db"
DEFAULT_CKPT_DIR = REPO_ROOT / "nba_transformer" / "artifacts" / "backtest_dec" / "checkpoints"
CORE_DB = REPO_ROOT / "data" / "artifacts" / "nba_core.sqlite"

STARS = {
    "giannis": ("203507", 1268, "MIL"),
    "lebron": ("2544", 1354, "LAL"),
    "luka": ("1629029", 320, "LAL"),
    "curry": ("201939", 1119, "GSW"),
    "durant": ("201142", 1072, "PHX"),
    "jokic": ("203999", 1331, "DEN"),
    "tatum": ("1628369", 164, "BOS"),
    "embiid": ("203954", 1320, "PHI"),
    "sga": ("1628983", 281, "OKC"),
    "ant": ("1630162", 496, "MIN"),
    "brown": ("1627759", 101, "BOS"),
    "wemby": ("1641705", 816, "SAS"),
}

TEAM_IDS = {
    "ATL": 1, "BOS": 2, "CLE": 3, "NOP": 4, "CHI": 5, "DAL": 6, "DEN": 7,
    "GSW": 8, "HOU": 9, "LAC": 10, "LAL": 11, "MIA": 12, "MIL": 13,
    "MIN": 14, "BKN": 15, "NYK": 16, "ORL": 17, "IND": 18, "PHI": 19,
    "PHX": 20, "POR": 21, "SAC": 22, "SAS": 23, "OKC": 24, "TOR": 25,
    "UTA": 26, "MEM": 27, "WAS": 28, "DET": 29, "CHA": 30,
}
TEAM_NAMES = {v: k for k, v in TEAM_IDS.items()}


def load_model(ckpt_path, db_path, window_start, device):
    vocab_size, team_vocab_size = load_precomputed_vocab_size(db_path, window_start)
    train_ds = PrecomputedDatasetV5(db_path, window_start, "train")
    sample = train_ds[0]
    stats_dim = sample["home_stats"].size(-1) if sample["home_stats"].numel() > 0 else 0

    cfg = CmeV6Config(
        vocab_size=vocab_size, num_teams=team_vocab_size,
        tabular_dim=sample["tabular"].numel(),
        d=32, n_heads=4, n_enc=2, n_dec=2,
        dropout=0.0, player_stats_dim=stats_dim,
    )
    model = CmeV6(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def get_win_prob(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(batch, teacher_force=False, autoregressive=True)
    return torch.sigmoid(out["win_logit"]).cpu().numpy()


def get_game_info(db_path, window_start):
    """Return game_id → (home_team_idx, away_team_idx, label) for test games."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT game_id, home_team_idx, away_team_idx, label FROM games "
        "WHERE window_start=? AND split='test'", (window_start,)).fetchall()
    con.close()
    return {r["game_id"]: dict(r) for r in rows}


def get_game_players(db_path, window_start):
    """Return game_id → list of (side, slot, player_idx)."""
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "SELECT game_id, side, slot, player_idx FROM players "
        "WHERE window_start=?", (window_start,)).fetchall()
    con.close()
    result = {}
    for gid, side, slot, pidx in rows:
        result.setdefault(gid, []).append((side, slot, pidx))
    return result


def find_player_games(game_players, game_info, player_idx, team_idx=None):
    """Find test games where player appears, optionally filter by team."""
    games = []
    for gid, players in game_players.items():
        if gid not in game_info:
            continue
        for side, slot, pidx in players:
            if pidx == player_idx:
                info = game_info[gid]
                if team_idx is not None:
                    player_team = info["home_team_idx"] if side == 0 else info["away_team_idx"]
                    if player_team != team_idx:
                        continue
                games.append((gid, side, slot))
                break
    return games


def scenario_remove_player(model, ds, device, player_idx, player_name, team_abbr):
    """Remove a player from all their games, replace with UNK (idx=0)."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {player_name} is OUT (removed from {team_abbr} lineup)")
    print(f"{'='*70}")

    results = []
    for idx in range(len(ds)):
        gid = ds._game_ids[idx]
        players = ds._players.get(gid, [])
        player_in_game = any(p["player_idx"] == player_idx for p in players)
        if not player_in_game:
            continue

        player_side = next(p["side"] for p in players if p["player_idx"] == player_idx)
        g = ds._games[gid]
        team = TEAM_NAMES.get(g["home_team_idx"] if player_side == 0 else g["away_team_idx"], "???")
        opp = TEAM_NAMES.get(g["away_team_idx"] if player_side == 0 else g["home_team_idx"], "???")
        label = int(g["label"])

        # Baseline
        item = ds[idx]
        batch_base = collate_v5([item])
        p_base = get_win_prob(model, batch_base, device)[0]

        # Modified: replace player_idx with 0 (UNK), zero their stats
        item_mod = ds[idx]
        side_key = "home_idx" if player_side == 0 else "away_idx"
        stats_key = "home_stats" if player_side == 0 else "away_stats"
        cy_key = "home_career_year_idx" if player_side == 0 else "away_career_year_idx"
        idx_tensor = item_mod[side_key]
        mask = idx_tensor == player_idx
        item_mod[side_key] = idx_tensor.clone()
        item_mod[side_key][mask] = 0
        item_mod[stats_key] = item_mod[stats_key].clone()
        item_mod[stats_key][mask] = 0
        item_mod[cy_key] = item_mod[cy_key].clone()
        item_mod[cy_key][mask] = 0
        batch_mod = collate_v5([item_mod])
        p_mod = get_win_prob(model, batch_mod, device)[0]

        delta = p_mod - p_base
        side_str = "home" if player_side == 0 else "away"
        actual = "W" if (label == 1 and player_side == 0) or (label == 0 and player_side == 1) else "L"

        results.append({
            "game_id": gid, "team": team, "opp": opp, "side": side_str,
            "p_base": p_base, "p_mod": p_mod, "delta": delta,
            "actual": actual, "label": label,
        })

    if not results:
        print(f"  No games found for {player_name}")
        return results

    print(f"\n{'game_id':>10s} {'matchup':>15s} {'side':>5s} {'base':>6s} {'w/o':>6s} {'delta':>7s} {'actual':>6s}")
    for r in results:
        matchup = f"{r['team']}v{r['opp']}" if r['side'] == 'home' else f"{r['opp']}v{r['team']}"
        color = "\033[31m" if r['delta'] < -0.01 else ("\033[32m" if r['delta'] > 0.01 else "")
        print(f"{r['game_id']:>10s} {matchup:>15s} {r['side']:>5s} "
              f"{r['p_base']:>5.1%} {r['p_mod']:>5.1%} {color}{r['delta']:>+6.1%}\033[0m {r['actual']:>6s}")

    deltas = [r['delta'] for r in results]
    team_wp_base = np.mean([r['p_base'] if r['side'] == 'home' else 1 - r['p_base'] for r in results])
    team_wp_mod = np.mean([r['p_mod'] if r['side'] == 'home' else 1 - r['p_mod'] for r in results])
    print(f"\n  {player_name}'s team avg win prob: {team_wp_base:.1%} → {team_wp_mod:.1%} "
          f"(Δ = {team_wp_mod - team_wp_base:+.1%})")
    print(f"  Mean home_p delta: {np.mean(deltas):+.3f}  (n={len(results)} games)")
    return results


def scenario_trade_player(model, ds, device, player_idx, player_name,
                          from_team_idx, to_team_idx, replacement_idx=0):
    """Move player from one team to another in games where both teams play."""
    from_team = TEAM_NAMES.get(from_team_idx, "???")
    to_team = TEAM_NAMES.get(to_team_idx, "???")

    print(f"\n{'='*70}")
    print(f"SCENARIO: {player_name} traded from {from_team} to {to_team}")
    print(f"{'='*70}")

    results = []
    for idx in range(len(ds)):
        gid = ds._game_ids[idx]
        g = ds._games[gid]
        h_team = g["home_team_idx"]
        a_team = g["away_team_idx"]

        # Only process games involving the destination team
        if to_team_idx not in (h_team, a_team):
            continue

        players = ds._players.get(gid, [])
        player_in_game = any(p["player_idx"] == player_idx for p in players)

        # Baseline
        item = ds[idx]
        batch_base = collate_v5([item])
        p_base = get_win_prob(model, batch_base, device)[0]

        # Modified: add player to destination team's lineup
        item_mod = ds[idx]
        dest_side = 0 if h_team == to_team_idx else 1
        side_key = "home_idx" if dest_side == 0 else "away_idx"

        # Replace the last roster slot with the traded player
        idx_tensor = item_mod[side_key].clone()
        idx_tensor[-1] = player_idx
        item_mod[side_key] = idx_tensor

        # If player was on the other team in this game, remove them
        if player_in_game:
            src_side = 0 if h_team == from_team_idx else 1
            if src_side != dest_side:
                src_key = "home_idx" if src_side == 0 else "away_idx"
                src_tensor = item_mod[src_key].clone()
                mask = src_tensor == player_idx
                src_tensor[mask] = replacement_idx
                item_mod[src_key] = src_tensor

        batch_mod = collate_v5([item_mod])
        p_mod = get_win_prob(model, batch_mod, device)[0]

        delta = p_mod - p_base
        label = int(g["label"])
        home = TEAM_NAMES.get(h_team, "???")
        away = TEAM_NAMES.get(a_team, "???")

        results.append({
            "game_id": gid, "home": home, "away": away,
            "p_base": p_base, "p_mod": p_mod, "delta": delta,
            "label": label, "dest_side": "home" if dest_side == 0 else "away",
        })

    if not results:
        print(f"  No games found for {to_team}")
        return results

    print(f"\n{'game_id':>10s} {'matchup':>12s} {player_name+' on':>12s} {'base':>6s} {'after':>6s} {'delta':>7s}")
    for r in results:
        matchup = f"{r['home']}v{r['away']}"
        color = "\033[32m" if abs(r['delta']) > 0.01 else ""
        print(f"{r['game_id']:>10s} {matchup:>12s} {r['dest_side']:>12s} "
              f"{r['p_base']:>5.1%} {r['p_mod']:>5.1%} {color}{r['delta']:>+6.1%}\033[0m")

    dest_wp_base = np.mean([r['p_base'] if r['dest_side'] == 'home' else 1 - r['p_base'] for r in results])
    dest_wp_mod = np.mean([r['p_mod'] if r['dest_side'] == 'home' else 1 - r['p_mod'] for r in results])
    print(f"\n  {to_team} avg win prob: {dest_wp_base:.1%} → {dest_wp_mod:.1%} "
          f"(Δ = {dest_wp_mod - dest_wp_base:+.1%})")
    print(f"  n={len(results)} games")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--precomputed-db", type=Path, default=DEFAULT_PRECOMPUTED_DB)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--window", type=str, default="2026-03-01")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    db_path = args.precomputed_db
    window = args.window
    ckpt_path = args.ckpt_dir / f"model_{window}.pt"

    print(f"Loading model for window {window}...")
    model = load_model(ckpt_path, db_path, window, args.device)
    test_ds = PrecomputedDatasetV5(db_path, window, "test")
    print(f"Test set: {len(test_ds)} games")

    # ---- Injury scenarios ----
    for name in ["giannis", "luka", "lebron", "sga", "jokic", "curry", "tatum", "ant", "brown", "wemby"]:
        pid, token_idx, team = STARS[name]
        scenario_remove_player(model, test_ds, args.device, token_idx,
                               name.capitalize(), team)

    # ---- Trade scenarios ----
    # Giannis to OKC
    _, giannis_idx, _ = STARS["giannis"]
    scenario_trade_player(model, test_ds, args.device,
                          giannis_idx, "Giannis",
                          from_team_idx=TEAM_IDS["MIL"],
                          to_team_idx=TEAM_IDS["OKC"])

    # Giannis to GSW
    scenario_trade_player(model, test_ds, args.device,
                          giannis_idx, "Giannis",
                          from_team_idx=TEAM_IDS["MIL"],
                          to_team_idx=TEAM_IDS["GSW"])

    # LeBron to NYK
    _, lebron_idx, _ = STARS["lebron"]
    scenario_trade_player(model, test_ds, args.device,
                          lebron_idx, "LeBron",
                          from_team_idx=TEAM_IDS["LAL"],
                          to_team_idx=TEAM_IDS["NYK"])


if __name__ == "__main__":
    main()
