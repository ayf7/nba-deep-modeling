#!/usr/bin/env python3
"""Expanding-window monthly backtest for the NBA Transformer.

Uses precomputed features DB. Supports --window-shards / --shard-idx for parallelism.
Picks best val (AR) epoch by accuracy, reports test accuracy at that epoch.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import PrecomputedDataset, collate, load_window_info, load_vocab_size
from model import NBATransformer, NBATransformerConfig, rotation_loss, box_cum_loss, win_hinge


def run_epoch_train(model, loader, optim, device, ss_ratio, w_delta):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, teacher_force=True, ss_ratio=ss_ratio)
        L_rot = rotation_loss(out, batch)
        L_delta = box_cum_loss(out, batch)
        L_win = win_hinge(out["win_logit"], batch["label"])
        loss = 1.0 * L_rot + w_delta * L_delta + 1.0 * L_win
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        bs = batch["label"].size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / total_n


def eval_accuracy(model, loader, device, autoregressive=True):
    model.eval()
    correct = 0
    total_n = 0
    bce_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if autoregressive:
                out = model(batch, teacher_force=False, autoregressive=True)
            else:
                out = model(batch, teacher_force=True)
            logit = out["win_logit"]
            label = batch["label"]
            pred = (torch.sigmoid(logit) > 0.5).float()
            correct += int((pred == label).sum())
            bce_sum += float(F.binary_cross_entropy_with_logits(logit, label)) * label.size(0)
            total_n += label.size(0)
    return correct / total_n, bce_sum / total_n


def train_one_window(args, window_start, db_path):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = PrecomputedDataset(db_path, window_start, "train")
    val_ds = PrecomputedDataset(db_path, window_start, "val")
    test_ds = PrecomputedDataset(db_path, window_start, "test")

    if len(test_ds) < 5:
        return None

    vocab_size, team_vocab_size = load_vocab_size(db_path, window_start)
    sample = train_ds[0]
    stats_dim = sample["home_stats"].size(-1) if sample["home_stats"].numel() > 0 else 0

    cfg = NBATransformerConfig(
        vocab_size=vocab_size, num_teams=team_vocab_size,
        tabular_dim=sample["tabular"].numel(),
        d=args.d, n_heads=args.n_heads,
        n_enc=args.n_enc, n_dec=args.n_dec,
        dropout=args.dropout,
        player_stats_dim=stats_dim,
        use_player_embeddings=not args.no_player_emb,
    )
    model = NBATransformer(cfg).to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    t0 = time.time()
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        ss_ratio = min(epoch / max(args.ss_warmup, 1), 1.0) * args.ss_max
        t_ep = time.time()
        train_loss = run_epoch_train(model, train_loader, optim, args.device, ss_ratio, args.w_delta)

        val_acc, val_bce = eval_accuracy(model, val_loader, args.device, autoregressive=True)
        dt_ep = time.time() - t_ep

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  ep{epoch:3d} val_acc={val_acc:.3f} (early stop)", flush=True)
                break

        if epoch <= 3 or epoch % 5 == 0 or marker:
            print(f"  ep{epoch:3d} ({dt_ep:.0f}s) loss={train_loss:.4f} val_acc={val_acc:.3f} bce={val_bce:.4f}{marker}", flush=True)

    train_time = time.time() - t0

    # Restore best and eval test
    model.load_state_dict(best_state)
    test_acc, test_bce = eval_accuracy(model, test_loader, args.device, autoregressive=True)

    # Save checkpoint
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / f"model_{window_start}.pt")

    return {
        "window_start": window_start,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "test_n": len(test_ds),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_bce": test_bce,
        "secs": train_time,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--precomputed-db", type=Path,
                   default=REPO_ROOT / "data" / "features_v5_precomputed.db")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "nba_transformer" / "artifacts" / "backtest_dec")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--n-enc", type=int, default=2)
    p.add_argument("--n-dec", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-player-emb", action="store_true")
    # training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--w-delta", type=float, default=1.0)
    p.add_argument("--ss-warmup", type=int, default=5)
    p.add_argument("--ss-max", type=float, default=0.5)
    # sharding
    p.add_argument("--window-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.precomputed_db
    windows = load_window_info(db_path)

    # Filter to windows with test data
    valid_windows = []
    for ws, we in windows:
        ds = PrecomputedDataset(db_path, ws, "test")
        if len(ds) >= 5:
            valid_windows.append(ws)
    print(f"Total valid windows: {len(valid_windows)}")

    # Shard selection
    if args.window_shards > 1:
        shard_windows = [valid_windows[i] for i in range(len(valid_windows))
                         if i % args.window_shards == args.shard_idx]
    else:
        shard_windows = valid_windows

    if args.aggregate_only:
        aggregate_results(args, valid_windows)
        return

    print(f"Shard {args.shard_idx}/{args.window_shards}: {len(shard_windows)} windows")
    print(f"Config: d={args.d} n_enc={args.n_enc} n_dec={args.n_dec} emb={not args.no_player_emb}")

    results = []
    for ws in shard_windows:
        print(f"\n{'='*60}")
        print(f"Window: {ws}")
        print(f"{'='*60}")
        result = train_one_window(args, ws, db_path)
        if result is None:
            print(f"  Skipped (not enough test data)")
            continue
        results.append(result)
        print(f"  best_epoch={result['best_epoch']} val_acc={result['best_val_acc']:.3f} "
              f"test_acc={result['test_acc']:.3f} test_bce={result['test_bce']:.4f} "
              f"({result['secs']:.0f}s)")

    # Save shard results
    shard_file = args.output_dir / f"shard_{args.shard_idx}.csv"
    if results:
        with open(shard_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nShard results saved to {shard_file}")

        accs = [r["test_acc"] for r in results]
        print(f"Shard mean test_acc: {sum(accs)/len(accs):.3f} ({len(accs)} windows)")


def aggregate_results(args, valid_windows):
    """Combine shard CSVs into final metrics."""
    all_results = []
    for shard_file in sorted(args.output_dir.glob("shard_*.csv")):
        with open(shard_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["test_acc"] = float(row["test_acc"])
                row["test_bce"] = float(row["test_bce"])
                row["best_val_acc"] = float(row["best_val_acc"])
                row["best_epoch"] = int(row["best_epoch"])
                row["train_n"] = int(row["train_n"])
                row["test_n"] = int(row["test_n"])
                all_results.append(row)

    if not all_results:
        print("No shard results found!")
        return

    # Save combined
    combined_file = args.output_dir / "window_metrics.csv"
    with open(combined_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(sorted(all_results, key=lambda r: r["window_start"]))

    accs = [r["test_acc"] for r in all_results]
    print(f"\nCombined results ({len(all_results)} windows):")
    print(f"  Mean test_acc: {sum(accs)/len(accs):.3f}")
    print(f"  Min: {min(accs):.3f}  Max: {max(accs):.3f}")
    print(f"  Saved to {combined_file}")


if __name__ == "__main__":
    main()
