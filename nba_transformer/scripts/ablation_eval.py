#!/usr/bin/env python3
"""Ablation study: disable components of v6 model and measure accuracy drop.

Ablations:
1. Full model (baseline) — AR decoder inference
2. No RoPE — zero out rope tables
3. No player embeddings — zero out E_player
4. Encoder only — freeze encoder, train a small MLP win head on top
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cme_v5_common import PrecomputedDatasetV5, collate_v5, load_precomputed_vocab_size
from model import CmeV6, CmeV6Config

DB = REPO_ROOT / "data" / "features_v5_precomputed.db"
CKPT_DIR = REPO_ROOT / "nba_transformer" / "artifacts" / "backtest_dec" / "checkpoints"
DEVICE = "cuda"

WINDOWS = ["2025-12-01", "2026-01-01", "2026-03-01"]


class EncoderWinHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d * 4, d), nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, home_team, away_team, home_pool, away_pool):
        x = torch.cat([home_team, away_team, home_pool, away_pool], dim=-1)
        return self.head(x).squeeze(-1)


def load_model(window):
    vocab_size, team_vocab_size = load_precomputed_vocab_size(DB, window)
    train_ds = PrecomputedDatasetV5(DB, window, "train")
    sample = train_ds[0]
    stats_dim = sample["home_stats"].size(-1) if sample["home_stats"].numel() > 0 else 0
    cfg = CmeV6Config(
        vocab_size=vocab_size, num_teams=team_vocab_size,
        tabular_dim=sample["tabular"].numel(),
        d=32, n_heads=4, n_enc=2, n_dec=2,
        dropout=0.0, player_stats_dim=stats_dim,
    )
    model = CmeV6(cfg).to(DEVICE)
    ckpt = CKPT_DIR / f"model_{window}.pt"
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def encode_to_reps(model, batch):
    """Run encoder, return (home_team, away_team, home_pool, away_pool)."""
    P, P_mask, home_team, away_team = model._encode(batch)
    Lh = batch["home_mask"].size(1)
    home_p = P[:, :Lh]
    away_p = P[:, Lh:]
    hm = batch["home_mask"].float().unsqueeze(-1)
    am = batch["away_mask"].float().unsqueeze(-1)
    home_pool = (home_p * hm).sum(dim=1) / hm.sum(dim=1).clamp(min=1)
    away_pool = (away_p * am).sum(dim=1) / am.sum(dim=1).clamp(min=1)
    return home_team, away_team, home_pool, away_pool


def train_encoder_win_head(model, train_loader, d):
    """Freeze encoder, train MLP win head for 15 epochs."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    head = EncoderWinHead(d).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    for epoch in range(15):
        head.train()
        total_loss = 0
        n = 0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                ht, at, hp, ap = encode_to_reps(model, batch)
            logit = head(ht, at, hp, ap)
            loss = F.binary_cross_entropy_with_logits(logit, batch["label"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch["label"])
            n += len(batch["label"])

    head.eval()
    return head


@torch.no_grad()
def eval_encoder_head(model, head, loader):
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        ht, at, hp, ap = encode_to_reps(model, batch)
        logit = head(ht, at, hp, ap)
        pred = (logit > 0).long()
        correct += (pred == batch["label"].long()).sum().item()
        total += len(batch["label"])
    return correct / total


@torch.no_grad()
def eval_accuracy(model, loader, mode="full"):
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(batch, teacher_force=False, autoregressive=True)
        pred = (out["win_logit"] > 0).long()
        label = batch["label"].long()
        correct += (pred == label).sum().item()
        total += label.size(0)
    return correct / total if total > 0 else 0.0


def ablate_rope(model):
    model.rope_cos.zero_()
    model.rope_cos.add_(1.0)
    model.rope_sin.zero_()


def ablate_player_embeddings(model):
    model.E_player.weight.data.zero_()


def main():
    ablation_names = ["Full Model", "No RoPE", "No Player Emb", "Encoder Only"]
    results = {name: [] for name in ablation_names}

    for window in WINDOWS:
        print(f"\nWindow {window}")
        train_ds = PrecomputedDatasetV5(DB, window, "train")
        test_ds = PrecomputedDatasetV5(DB, window, "test")
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_v5)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_v5)
        print(f"  Train: {len(train_ds)}, Test: {len(test_ds)} games")

        # Full model
        model, cfg = load_model(window)
        acc = eval_accuracy(model, test_loader)
        results["Full Model"].append(acc)
        print(f"  {'Full Model':25s}: {acc:.1%}")
        del model; torch.cuda.empty_cache()

        # No RoPE
        model, cfg = load_model(window)
        ablate_rope(model)
        acc = eval_accuracy(model, test_loader)
        results["No RoPE"].append(acc)
        print(f"  {'No RoPE':25s}: {acc:.1%}")
        del model; torch.cuda.empty_cache()

        # No Player Emb
        model, cfg = load_model(window)
        ablate_player_embeddings(model)
        acc = eval_accuracy(model, test_loader)
        results["No Player Emb"].append(acc)
        print(f"  {'No Player Emb':25s}: {acc:.1%}")
        del model; torch.cuda.empty_cache()

        # Encoder Only — train a win head on frozen encoder
        model, cfg = load_model(window)
        print(f"  Training encoder win head...")
        head = train_encoder_win_head(model, train_loader, cfg.d)
        acc = eval_encoder_head(model, head, test_loader)
        results["Encoder Only"].append(acc)
        print(f"  {'Encoder Only':25s}: {acc:.1%}")
        del model, head; torch.cuda.empty_cache()

    ablations = {n: None for n in ablation_names}

    # Aggregate
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY (mean accuracy across windows)")
    print(f"{'='*60}")
    for name in ablations:
        accs = results[name]
        mean = np.mean(accs)
        print(f"  {name:25s}: {mean:.1%}  ({', '.join(f'{a:.1%}' for a in accs)})")

    # Plot
    names = list(ablations.keys())
    means = [np.mean(results[n]) * 100 for n in names]
    per_window = {n: [a * 100 for a in results[n]] for n in names}

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(names))
    width = 0.2
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, window in enumerate(WINDOWS):
        vals = [per_window[n][i] for n in names]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=window, color=colors[i], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Ablation Study — Component Contribution to Win Prediction", fontsize=13)
    ax.legend(title="Window", fontsize=9)
    ax.set_ylim(45, 75)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="Coin flip")

    for i, window in enumerate(WINDOWS):
        vals = [per_window[n][i] for n in names]
        for j, v in enumerate(vals):
            ax.text(j + (i - 1) * width, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)

    plt.tight_layout()
    out = REPO_ROOT / "nba_transformer" / "artifacts" / "ablation_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
