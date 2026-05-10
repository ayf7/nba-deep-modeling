#!/usr/bin/env python3
"""Remove top ~25 players one at a time from all 2025-26 season test games, measure win prob impact."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import PrecomputedDataset, collate, load_vocab_size
from model import NBATransformer, NBATransformerConfig

DB = REPO_ROOT / "data" / "features_v5_precomputed.db"
CKPT_DIR = REPO_ROOT / "nba_transformer" / "artifacts" / "backtest_dec" / "checkpoints"
DEVICE = "cuda"

WINDOWS = [
    "2025-10-01", "2025-11-01", "2025-12-01",
    "2026-01-01", "2026-02-01", "2026-03-01",
]

PLAYERS = [
    ("Luka Doncic", 320, "LAL"),
    ("SGA", 281, "OKC"),
    ("Jokic", 1331, "DEN"),
    ("Giannis", 1268, "MIL"),
    ("Durant", 1072, "HOU"),
    ("Curry", 1119, "GSW"),
    ("LeBron", 1354, "LAL"),
    ("Donovan Mitchell", 170, "CLE"),
    ("Kawhi Leonard", 1178, "LAC"),
    ("Tyrese Maxey", 512, "PHI"),
    ("Paolo Banchero", 715, "ORL"),
    ("Devin Booker", 33, "PHX"),
    ("James Harden", 1116, "LAC"),
    ("Wembanyama", 816, "SAS"),
    ("Evan Mobley", 659, "CLE"),
    ("Anthony Edwards", 496, "MIN"),
    ("Trae Young", 318, "WAS"),
    ("Jayson Tatum", 164, "BOS"),
    ("Alperen Sengun", 644, "HOU"),
    ("Cade Cunningham", 658, "DET"),
    ("Bam Adebayo", 181, "MIA"),
    ("Jaylen Brown", 101, "BOS"),
    ("Khris Middleton", 1226, "MIL"),
]

EXCLUDE = {"Jayson Tatum", "Donovan Mitchell", "Paolo Banchero", "Tyrese Maxey"}


def load_model(window):
    ckpt_path = CKPT_DIR / f"model_{window}.pt"
    vocab_size, team_vocab_size = load_vocab_size(DB, window)
    train_ds = PrecomputedDataset(DB, window, "train")
    sample = train_ds[0]
    stats_dim = sample["home_stats"].size(-1) if sample["home_stats"].numel() > 0 else 0
    cfg = NBATransformerConfig(
        vocab_size=vocab_size, num_teams=team_vocab_size,
        tabular_dim=sample["tabular"].numel(),
        d=32, n_heads=4, n_enc=2, n_dec=2,
        dropout=0.0, player_stats_dim=stats_dim,
    )
    model = NBATransformer(cfg).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def get_win_prob(model, batch):
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    out = model(batch, teacher_force=False, autoregressive=True)
    return torch.sigmoid(out["win_logit"]).cpu().numpy()[0]


def measure_player_impact(model, ds, player_idx, bad_games):
    """Remove player from all their test games, return list of deltas."""
    deltas = []
    for idx in range(len(ds)):
        gid = ds._game_ids[idx]
        if gid in bad_games:
            continue
        players = ds._players.get(gid, [])
        player_in_game = any(p["player_idx"] == player_idx for p in players)
        if not player_in_game:
            continue

        player_side = next(p["side"] for p in players if p["player_idx"] == player_idx)

        item = ds[idx]
        p_base = get_win_prob(model, collate([item]))

        item_mod = ds[idx]
        side_key = "home_idx" if player_side == 0 else "away_idx"
        stats_key = "home_stats" if player_side == 0 else "away_stats"
        cy_key = "home_career_year_idx" if player_side == 0 else "away_career_year_idx"

        idx_tensor = item_mod[side_key].clone()
        mask = idx_tensor == player_idx
        idx_tensor[mask] = 0
        item_mod[side_key] = idx_tensor
        item_mod[stats_key] = item_mod[stats_key].clone()
        item_mod[stats_key][mask] = 0
        item_mod[cy_key] = item_mod[cy_key].clone()
        item_mod[cy_key][mask] = 0

        p_mod = get_win_prob(model, collate([item_mod]))

        if player_side == 0:
            delta = p_mod - p_base
        else:
            delta = (1 - p_mod) - (1 - p_base)
        deltas.append(delta)

    return deltas


def main():
    all_deltas = {name: [] for name, _, _ in PLAYERS}

    for window in WINDOWS:
        ckpt_path = CKPT_DIR / f"model_{window}.pt"
        if not ckpt_path.exists():
            print(f"  Skipping {window} — no checkpoint")
            continue

        print(f"\nWindow {window}...")
        model = load_model(window)
        test_ds = PrecomputedDataset(DB, window, "test")
        print(f"  Test set: {len(test_ds)} games")

        # Find games with < 5 players on either side
        bad_games = set()
        for gid, plist in test_ds._players.items():
            n_home = sum(1 for p in plist if p["side"] == 0)
            n_away = sum(1 for p in plist if p["side"] == 1)
            if n_home < 5 or n_away < 5:
                bad_games.add(gid)
        if bad_games:
            print(f"  Skipping {len(bad_games)} games with broken rosters")

        for name, token_idx, team in PLAYERS:
            deltas = measure_player_impact(model, test_ds, token_idx, bad_games)
            all_deltas[name].extend(deltas)
            if deltas:
                print(f"    {name:20s}: {len(deltas)} games this window")

    # Aggregate
    results = []
    for name, token_idx, team in PLAYERS:
        d = all_deltas[name]
        if not d:
            continue
        results.append((name, team, np.mean(d), len(d)))

    results.sort(key=lambda x: x[2])
    results = [(n, t, d, ng) for n, t, d, ng in results if n not in EXCLUDE]

    print(f"\n{'='*60}")
    print("PLAYER IMPACT RANKING — FULL 2025-26 SEASON")
    print(f"{'='*60}")
    for i, (name, team, delta, n) in enumerate(results, 1):
        print(f"  {i:2d}. {name:20s} ({team}): {delta:+.1%}  ({n} games)")

    # Plot top 10
    top = results[:10]
    names = [f"{r[0]}\n({r[1]})" for r in top]
    deltas_pct = [r[2] * 100 for r in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds(np.linspace(0.9, 0.4, len(top)))
    bars = ax.barh(range(len(top)), deltas_pct, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Team Win Probability Change (%)", fontsize=12)
    ax.set_title("Most Impactful Players — Win Prob Drop When Removed\n(2025-26 season)", fontsize=13)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()

    for bar, val in zip(bars, deltas_pct):
        ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", ha='right', va='center', fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()
    out_path = REPO_ROOT / "nba_transformer" / "artifacts" / "player_impact_top25.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
