#!/usr/bin/env python3
"""Train MAN v2 — Matchup Attention Network with Poisson + margin + BCE losses.

Per-matchup log-lambda is shared between two paths:
  (a) Poisson NLL on observed matchup rows (offense, defense, exposure, points).
  (b) Game-level forward over the rotation grid: expected_points_ij summed to
      team totals, margin = home - away (+ context), P(home_win) = sigmoid(margin / sigma).

Training combines the three losses with fixed CLI weights. Validation uses
log_loss for early stopping (matches the betting target).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from man_common import (
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_ROTATION_DECAY,
    DEFAULT_ROTATION_GAMES,
    DEFAULT_ROTATION_SIZE,
    LABEL_COLUMN,
    build_rotations_for_games,
    load_games,
    load_team_exposures,
)
from man_v2_common import (
    DEFAULT_MATCHUP_DB,
    GameDatasetV2,
    build_vocab_v2,
    collate_v2,
    load_game_scores,
    load_matchup_rows,
    make_records_v2,
)


MODEL_VERSION = "man_v2"
EMBEDDING_PARAM_NAMES = (
    "off_emb.weight",
    "def_emb.weight",
    "off_bias.weight",
    "def_bias.weight",
)


class MANv2(nn.Module):
    def __init__(
        self,
        num_players: int,
        embedding_dim: int = 8,
        attn_dim: int = 8,
        team_possessions: float = 100.0,
        sigma_init: float = 13.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attn_dim = attn_dim
        self.attn_scale = math.sqrt(attn_dim)
        self.team_possessions = team_possessions
        self.off_emb = nn.Embedding(num_players + 1, embedding_dim, padding_idx=0)
        self.def_emb = nn.Embedding(num_players + 1, embedding_dim, padding_idx=0)
        self.off_bias = nn.Embedding(num_players + 1, 1, padding_idx=0)
        self.def_bias = nn.Embedding(num_players + 1, 1, padding_idx=0)
        self.W_q = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init), dtype=torch.float32))
        self.context = nn.Linear(2, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.off_emb.weight[1:], std=0.05)
        nn.init.normal_(self.def_emb.weight[1:], std=0.05)
        nn.init.zeros_(self.off_bias.weight)
        nn.init.zeros_(self.def_bias.weight)
        with torch.no_grad():
            self.off_emb.weight[0].fill_(0.0)
            self.def_emb.weight[0].fill_(0.0)
            self.off_bias.weight[0].fill_(0.0)
            self.def_bias.weight[0].fill_(0.0)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)

    def matchup_log_lambda(
        self, off_ids: torch.Tensor, def_ids: torch.Tensor
    ) -> torch.Tensor:
        off = self.off_emb(off_ids)
        def_ = self.def_emb(def_ids)
        score = (self.W_q(off) * self.W_k(def_)).sum(dim=-1) / self.attn_scale
        return score + self.off_bias(off_ids).squeeze(-1) + self.def_bias(def_ids).squeeze(-1)

    def pairwise_log_lambda(
        self,
        off_emb: torch.Tensor,
        def_emb: torch.Tensor,
        off_ids: torch.Tensor,
        def_ids: torch.Tensor,
    ) -> torch.Tensor:
        Q = self.W_q(off_emb)
        K = self.W_k(def_emb)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.attn_scale
        bias_off = self.off_bias(off_ids).squeeze(-1).unsqueeze(2)
        bias_def = self.def_bias(def_ids).squeeze(-1).unsqueeze(1)
        return scores + bias_off + bias_def

    def forward_game(self, batch: dict) -> dict:
        h_off = self.off_emb(batch["home_ids"])
        h_def = self.def_emb(batch["home_ids"])
        a_off = self.off_emb(batch["away_ids"])
        a_def = self.def_emb(batch["away_ids"])

        home_off_ll = self.pairwise_log_lambda(h_off, a_def, batch["home_ids"], batch["away_ids"])
        away_off_ll = self.pairwise_log_lambda(a_off, h_def, batch["away_ids"], batch["home_ids"])

        home_w = batch["home_w"]
        away_w = batch["away_w"]
        home_off_mask = (home_w > 0).unsqueeze(2) & (away_w > 0).unsqueeze(1)
        away_off_mask = (away_w > 0).unsqueeze(2) & (home_w > 0).unsqueeze(1)

        home_exp = self.team_possessions * home_w.unsqueeze(2) * away_w.unsqueeze(1)
        away_exp = self.team_possessions * away_w.unsqueeze(2) * home_w.unsqueeze(1)

        home_pts = (home_exp * torch.exp(home_off_ll) * home_off_mask).sum(dim=(1, 2))
        away_pts = (away_exp * torch.exp(away_off_ll) * away_off_mask).sum(dim=(1, 2))

        rest = torch.stack([batch["home_rest"], batch["away_rest"]], dim=-1)
        offset = self.context(rest).squeeze(-1)
        margin = home_pts - away_pts + offset
        sigma = torch.exp(self.log_sigma)
        return {
            "home_pts": home_pts,
            "away_pts": away_pts,
            "margin": margin,
            "margin_logit": margin / sigma,
            "sigma": sigma,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    metrics = {
        "n": int(len(labels)),
        "log_loss": float(log_loss(labels, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probs)),
        "accuracy": float(accuracy_score(labels, (probs >= 0.5).astype(int))),
    }
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    return metrics


def evaluate(
    model: MANv2, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_all, labels_all, margins_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.forward_game(batch)
            probs_all.append(torch.sigmoid(out["margin_logit"]).cpu().numpy())
            labels_all.append(batch["label"].cpu().numpy())
            margins_all.append(out["margin"].cpu().numpy())
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    margins = np.concatenate(margins_all)
    return compute_metrics(probs, labels), probs, labels, margins


def train_one_epoch(
    model: MANv2,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    poisson_weight: float,
    margin_weight: float,
    bce_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    sums = {"poisson": 0.0, "margin": 0.0, "bce": 0.0, "total": 0.0}
    n_batches = 0
    n_matchup_rows = 0
    n_games = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.forward_game(batch)
        bce = F.binary_cross_entropy_with_logits(out["margin_logit"], batch["label"])
        mse = F.mse_loss(out["margin"], batch["margin"])

        if batch["m_off"].numel() > 0:
            log_lambda = model.matchup_log_lambda(batch["m_off"], batch["m_def"])
            log_rate = log_lambda + torch.log(batch["m_exp"].clamp(min=1e-8))
            poisson = F.poisson_nll_loss(
                log_rate, batch["m_pts"], log_input=True, full=False, reduction="mean"
            )
        else:
            poisson = torch.zeros((), device=device)

        loss = poisson_weight * poisson + margin_weight * mse + bce_weight * bce

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        n_batches += 1
        n_matchup_rows += batch["m_off"].numel()
        n_games += batch["label"].numel()
        sums["poisson"] += float(poisson.item())
        sums["margin"] += float(mse.item())
        sums["bce"] += float(bce.item())
        sums["total"] += float(loss.item())

    return {
        "poisson": sums["poisson"] / max(n_batches, 1),
        "margin": sums["margin"] / max(n_batches, 1),
        "bce": sums["bce"] / max(n_batches, 1),
        "total": sums["total"] / max(n_batches, 1),
        "matchup_rows": n_matchup_rows,
        "games": n_games,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    parser.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    parser.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    parser.add_argument("--table", default="model_games")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-end-date", default="2024-01-01")
    parser.add_argument("--val-end-date", default="2024-04-01")
    parser.add_argument("--min-games-before", type=int, default=10)
    parser.add_argument("--rotation-games", type=int, default=DEFAULT_ROTATION_GAMES)
    parser.add_argument("--rotation-size", type=int, default=DEFAULT_ROTATION_SIZE)
    parser.add_argument("--rotation-decay", type=float, default=DEFAULT_ROTATION_DECAY)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--attn-dim", type=int, default=8)
    parser.add_argument("--team-possessions", type=float, default=100.0)
    parser.add_argument("--sigma-init", type=float, default=13.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--embedding-weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--poisson-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.01)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] features_db={args.features_db}")
    games = load_games(args.features_db, table=args.table, min_games_before=args.min_games_before)
    train_end = pd.Timestamp(args.train_end_date)
    val_end = pd.Timestamp(args.val_end_date)
    train_games = games[games["game_date"] < train_end].reset_index(drop=True)
    val_games = games[
        (games["game_date"] >= train_end) & (games["game_date"] < val_end)
    ].reset_index(drop=True)
    print(f"[load] train={len(train_games)} val={len(val_games)}")

    print(f"[load] core_db={args.core_db} (team exposures)")
    histories = load_team_exposures(args.core_db)

    train_games_kept, train_rotations = build_rotations_for_games(
        train_games,
        histories,
        rotation_games=args.rotation_games,
        rotation_size=args.rotation_size,
        decay=args.rotation_decay,
    )
    val_games_kept, val_rotations = build_rotations_for_games(
        val_games,
        histories,
        rotation_games=args.rotation_games,
        rotation_size=args.rotation_size,
        decay=args.rotation_decay,
    )
    print(f"[rotations] train_kept={len(train_games_kept)} val_kept={len(val_games_kept)}")

    print(f"[load] matchup_db={args.matchup_db} (training matchup rows)")
    train_matchup_rows = load_matchup_rows(
        args.matchup_db, [str(g) for g in train_games_kept["game_id"].tolist()]
    )
    print(f"[load] games_with_matchup_rows={len(train_matchup_rows)}")

    print(f"[load] core_db={args.core_db} (game scores)")
    train_scores = load_game_scores(args.core_db, [str(g) for g in train_games_kept["game_id"].tolist()])
    val_scores = load_game_scores(args.core_db, [str(g) for g in val_games_kept["game_id"].tolist()])
    print(f"[load] train_scores={len(train_scores)} val_scores={len(val_scores)}")

    vocab = build_vocab_v2(train_rotations, train_matchup_rows)
    print(f"[vocab] num_players={vocab.size}")

    train_records = make_records_v2(
        train_games_kept,
        train_rotations,
        vocab,
        matchup_rows=train_matchup_rows,
        game_scores=train_scores,
    )
    val_records = make_records_v2(
        val_games_kept,
        val_rotations,
        vocab,
        matchup_rows=None,
        game_scores=val_scores,
    )
    print(f"[records] train={len(train_records)} val={len(val_records)}")

    train_loader = DataLoader(
        GameDatasetV2(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_v2,
    )
    val_loader = DataLoader(
        GameDatasetV2(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_v2,
    )

    model = MANv2(
        num_players=vocab.size,
        embedding_dim=args.embedding_dim,
        attn_dim=args.attn_dim,
        team_possessions=args.team_possessions,
        sigma_init=args.sigma_init,
    ).to(device)

    embedding_params = [p for n, p in model.named_parameters() if n in EMBEDDING_PARAM_NAMES]
    other_params = [p for n, p in model.named_parameters() if n not in EMBEDDING_PARAM_NAMES]
    optimizer = torch.optim.Adam(
        [
            {"params": embedding_params, "weight_decay": args.embedding_weight_decay},
            {"params": other_params, "weight_decay": args.weight_decay},
        ],
        lr=args.learning_rate,
    )

    log_path = args.output_dir / "training_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_total",
                "train_poisson",
                "train_margin",
                "train_bce",
                "train_matchup_rows",
                "val_log_loss",
                "val_brier",
                "val_accuracy",
                "val_roc_auc",
                "sigma",
                "elapsed_s",
            ]
        )

    best_val_log_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            poisson_weight=args.poisson_weight,
            margin_weight=args.margin_weight,
            bce_weight=args.bce_weight,
            grad_clip=args.grad_clip,
        )
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        sigma_val = float(torch.exp(model.log_sigma).item())

        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_stats['total']:.6f}",
                    f"{train_stats['poisson']:.6f}",
                    f"{train_stats['margin']:.6f}",
                    f"{train_stats['bce']:.6f}",
                    train_stats["matchup_rows"],
                    f"{val_metrics['log_loss']:.6f}",
                    f"{val_metrics['brier']:.6f}",
                    f"{val_metrics['accuracy']:.6f}",
                    f"{val_metrics.get('roc_auc', float('nan')):.6f}",
                    f"{sigma_val:.4f}",
                    f"{elapsed:.2f}",
                ]
            )

        if epoch % args.log_every == 0:
            print(
                f"[epoch {epoch:03d}] train_total={train_stats['total']:.4f} "
                f"poi={train_stats['poisson']:.4f} mar={train_stats['margin']:.2f} "
                f"bce={train_stats['bce']:.4f} | "
                f"val log_loss={val_metrics['log_loss']:.4f} "
                f"auc={val_metrics.get('roc_auc', float('nan')):.4f} "
                f"acc={val_metrics['accuracy']:.4f} sigma={sigma_val:.2f} "
                f"({elapsed:.1f}s)"
            )

        if val_metrics["log_loss"] < best_val_log_loss - 1e-6:
            best_val_log_loss = val_metrics["log_loss"]
            best_epoch = epoch
            epochs_since_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"[early-stop] no improvement for {args.patience} epochs (best={best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, val_probs, val_labels, val_margins = evaluate(model, val_loader, device)
    print("[final-val]", json.dumps(final_metrics, indent=2))

    config = {
        "model_version": MODEL_VERSION,
        "features_db": str(args.features_db),
        "core_db": str(args.core_db),
        "matchup_db": str(args.matchup_db),
        "table": args.table,
        "train_end_date": args.train_end_date,
        "val_end_date": args.val_end_date,
        "min_games_before": args.min_games_before,
        "rotation_games": args.rotation_games,
        "rotation_size": args.rotation_size,
        "rotation_decay": args.rotation_decay,
        "embedding_dim": args.embedding_dim,
        "attn_dim": args.attn_dim,
        "team_possessions": args.team_possessions,
        "sigma_init": args.sigma_init,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_weight_decay": args.embedding_weight_decay,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
        "poisson_weight": args.poisson_weight,
        "margin_weight": args.margin_weight,
        "bce_weight": args.bce_weight,
        "random_state": args.random_state,
        "device": args.device,
        "num_players": vocab.size,
        "n_train_records": len(train_records),
        "n_val_records": len(val_records),
        "best_epoch": best_epoch,
        "best_val_log_loss": best_val_log_loss,
        "final_sigma": float(torch.exp(model.log_sigma).item()),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": {"player_to_idx": vocab.player_to_idx},
            "config": config,
        },
        args.output_dir / "model.pt",
    )

    pred_df = pd.DataFrame(
        {
            "game_id": [r.game_id for r in val_records],
            "game_date": [r.game_date.date().isoformat() for r in val_records],
            "home_team_id": [r.home_team_id for r in val_records],
            "away_team_id": [r.away_team_id for r in val_records],
            LABEL_COLUMN: val_labels.astype(int),
            "actual_margin": [r.margin for r in val_records],
            "pred_margin": val_margins,
            "pred_home_win_prob": val_probs,
        }
    )
    pred_df.to_csv(args.output_dir / "predictions.csv", index=False)
    print(f"[saved] {args.output_dir}")


if __name__ == "__main__":
    main()
