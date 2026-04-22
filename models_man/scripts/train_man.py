#!/usr/bin/env python3
"""Train a single MAN snapshot end-to-end on game win/loss.

Cross-attention over home/away rosters: each offensive player attends to all
opposing defenders, weighted by minute share. Pure BCE loss on home_win;
no auxiliary targets, no warm-start, no v1/v2 features.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from man_common import (
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_ROTATION_DECAY,
    DEFAULT_ROTATION_GAMES,
    DEFAULT_ROTATION_SIZE,
    GameDataset,
    build_rotations_for_games,
    build_vocab,
    load_games,
    load_team_exposures,
    make_records,
)


MODEL_VERSION = "man_v1"
EMBEDDING_PARAM_NAMES = ("off_emb.weight", "def_emb.weight")


class MAN(nn.Module):
    def __init__(
        self,
        num_players: int,
        embedding_dim: int = 8,
        attn_dim: int = 8,
        head_hidden: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attn_dim = attn_dim
        self.off_emb = nn.Embedding(num_players + 1, embedding_dim, padding_idx=0)
        self.def_emb = nn.Embedding(num_players + 1, embedding_dim, padding_idx=0)
        self.W_q = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, attn_dim, bias=False)
        head_input = 2 * attn_dim + 2 * embedding_dim + 2
        self.head = nn.Sequential(
            nn.Linear(head_input, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.off_emb.weight[1:], std=0.05)
        nn.init.normal_(self.def_emb.weight[1:], std=0.05)
        with torch.no_grad():
            self.off_emb.weight[0].fill_(0.0)
            self.def_emb.weight[0].fill_(0.0)

    def cross_attention(
        self,
        off: torch.Tensor,
        def_: torch.Tensor,
        off_w: torch.Tensor,
        def_w: torch.Tensor,
    ) -> torch.Tensor:
        scale = math.sqrt(self.attn_dim)
        Q = self.W_q(off)
        K = self.W_k(def_)
        V = self.W_v(def_)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / scale
        mask = (def_w > 0).unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        per_offender = torch.matmul(attn, V)
        off_w_norm = off_w / off_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return (per_offender * off_w_norm.unsqueeze(-1)).sum(dim=1)

    def pool_def(self, def_: torch.Tensor, def_w: torch.Tensor) -> torch.Tensor:
        norm = def_w / def_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return (def_ * norm.unsqueeze(-1)).sum(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        h_off = self.off_emb(batch["home_ids"])
        h_def = self.def_emb(batch["home_ids"])
        a_off = self.off_emb(batch["away_ids"])
        a_def = self.def_emb(batch["away_ids"])
        home_attack = self.cross_attention(h_off, a_def, batch["home_w"], batch["away_w"])
        away_attack = self.cross_attention(a_off, h_def, batch["away_w"], batch["home_w"])
        home_def_pool = self.pool_def(h_def, batch["home_w"])
        away_def_pool = self.pool_def(a_def, batch["away_w"])
        rest = torch.stack([batch["home_rest"], batch["away_rest"]], dim=-1)
        feats = torch.cat(
            [home_attack, away_attack, home_def_pool, away_def_pool, rest], dim=-1
        )
        return self.head(feats).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def evaluate_loader(
    model: MAN, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return labels, probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAN end-to-end on game outcomes.")
    parser.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    parser.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train-end-date",
        default="2024-01-01",
        help="Train on games with date < this. Validate on [this, --val-end-date).",
    )
    parser.add_argument("--val-end-date", default="2024-04-01")
    parser.add_argument("--min-games-before", type=int, default=10)
    parser.add_argument("--rotation-games", type=int, default=DEFAULT_ROTATION_GAMES)
    parser.add_argument("--rotation-size", type=int, default=DEFAULT_ROTATION_SIZE)
    parser.add_argument("--rotation-decay", type=float, default=DEFAULT_ROTATION_DECAY)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--attn-dim", type=int, default=8)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--embedding-weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def split_train_val(
    games: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = games[games["game_date"] < train_end].reset_index(drop=True)
    val_df = games[
        (games["game_date"] >= train_end) & (games["game_date"] < val_end)
    ].reset_index(drop=True)
    return train_df, val_df


def main() -> None:
    args = parse_args()
    if not args.features_db.exists():
        raise FileNotFoundError(f"Features DB not found: {args.features_db}")
    if not args.core_db.exists():
        raise FileNotFoundError(f"Core DB not found: {args.core_db}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    set_seed(args.random_state)
    device = torch.device(args.device)

    print("Loading games and exposures...", flush=True)
    games = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)

    train_end = pd.Timestamp(args.train_end_date)
    val_end = pd.Timestamp(args.val_end_date)
    train_games_raw, val_games_raw = split_train_val(games, train_end, val_end)
    print(
        f"Pre-rotation counts: train={len(train_games_raw)}, val={len(val_games_raw)}",
        flush=True,
    )

    print("Building rotations...", flush=True)
    train_games, train_rotations = build_rotations_for_games(
        train_games_raw,
        histories,
        rotation_games=args.rotation_games,
        rotation_size=args.rotation_size,
        decay=args.rotation_decay,
    )
    val_games, val_rotations = build_rotations_for_games(
        val_games_raw,
        histories,
        rotation_games=args.rotation_games,
        rotation_size=args.rotation_size,
        decay=args.rotation_decay,
    )
    print(
        f"Post-rotation counts: train={len(train_games)}, val={len(val_games)}",
        flush=True,
    )
    if len(train_games) == 0 or len(val_games) == 0:
        raise ValueError("Empty train or validation set after rotation build.")

    vocab = build_vocab(train_rotations)
    print(f"Vocab size (training players): {vocab.size}", flush=True)

    train_records = make_records(train_games, train_rotations, vocab)
    val_records = make_records(val_games, val_rotations, vocab)
    train_loader = DataLoader(GameDataset(train_records), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(GameDataset(val_records), batch_size=args.batch_size, shuffle=False)

    model = MAN(
        num_players=vocab.size,
        embedding_dim=args.embedding_dim,
        attn_dim=args.attn_dim,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
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
    bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    epochs_since_best = 0
    log_rows: list[dict] = []
    started_at = time.time()
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = bce(logits, batch["label"])
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n = len(batch["label"])
            train_loss_sum += float(loss.detach()) * n
            train_n += n
        train_loss = train_loss_sum / max(train_n, 1)

        val_labels, val_probs = evaluate_loader(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_probs)
        val_loss = val_metrics["log_loss"]

        elapsed = time.time() - started_at
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "elapsed_seconds": elapsed,
            }
        )
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0):
            auc = val_metrics.get("roc_auc", float("nan"))
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_auc={auc:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}",
                flush=True,
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_labels, final_probs = evaluate_loader(model, val_loader, device)
    final_metrics = compute_metrics(final_labels, final_probs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_version": MODEL_VERSION,
        "features_db": str(args.features_db),
        "core_db": str(args.core_db),
        "train_end_date": args.train_end_date,
        "val_end_date": args.val_end_date,
        "min_games_before": args.min_games_before,
        "rotation_games": args.rotation_games,
        "rotation_size": args.rotation_size,
        "rotation_decay": args.rotation_decay,
        "embedding_dim": args.embedding_dim,
        "attn_dim": args.attn_dim,
        "head_hidden": args.head_hidden,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_weight_decay": args.embedding_weight_decay,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
        "random_state": args.random_state,
        "device": args.device,
        "vocab_size": vocab.size,
        "train_n": len(train_records),
        "val_n": len(val_records),
        "train_date_min": train_games["game_date"].min().date().isoformat(),
        "train_date_max": train_games["game_date"].max().date().isoformat(),
        "val_date_min": val_games["game_date"].min().date().isoformat(),
        "val_date_max": val_games["game_date"].max().date().isoformat(),
        "best_epoch": best_epoch,
        "epochs_completed": last_epoch,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    torch.save(
        {
            "model_version": MODEL_VERSION,
            "model_state_dict": model.state_dict(),
            "vocab": vocab.player_to_idx,
            "config": config,
        },
        args.output_dir / "model.pt",
    )
    pd.DataFrame(log_rows).to_csv(args.output_dir / "training_log.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")

    predictions = pd.DataFrame(
        {
            "game_id": [r.game_id for r in val_records],
            "game_date": [r.game_date.date().isoformat() for r in val_records],
            "home_team_id": [r.home_team_id for r in val_records],
            "away_team_id": [r.away_team_id for r in val_records],
            "label_home_win": final_labels.astype(int),
            "pred_home_win_prob": final_probs.astype(float),
        }
    )
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)

    print(json.dumps(final_metrics, indent=2))
    print(f"Saved artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
