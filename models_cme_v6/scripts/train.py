#!/usr/bin/env python3
"""Expanding-window backtest for CME-v6 cumulative game-state decoder.

Reuses the v5 precomputed-DB loader (data layer is shared). Trains
per-minute rotation + cumulative box-score prediction + win logit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cme_v5_common import (  # noqa: E402
    PrecomputedDatasetV5, collate_v5,
    load_precomputed_window_info, load_precomputed_vocab_size,
)
from model import (  # noqa: E402
    CmeV6, CmeV6Config, total_loss_cumstate,
    rotation_loss, win_bce,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v6" / "artifacts"
DEFAULT_PRECOMPUTED_DB = REPO_ROOT / "data" / "features_v5_precomputed.db"


def _iou_top5(rot_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    """IoU of argtop-5 over (rotation_logits) vs argtop-5 over rotation_target.

    rot_logits, target: (B, L, S). mask: (B, L). Returns (sum_iou, count) for streaming.
    """
    B, L, S = rot_logits.shape
    if L < 5:
        return 0.0, 0
    rot_logits = rot_logits.masked_fill(~mask.unsqueeze(-1), -1e9)
    pred_top = rot_logits.topk(5, dim=1).indices
    tgt_top = target.topk(5, dim=1).indices
    pred_set = torch.zeros(B, L, S, dtype=torch.bool, device=rot_logits.device)
    tgt_set = torch.zeros(B, L, S, dtype=torch.bool, device=rot_logits.device)
    pred_set.scatter_(1, pred_top, True)
    tgt_set.scatter_(1, tgt_top, True)
    inter = (pred_set & tgt_set).sum(dim=1).float()
    union = (pred_set | tgt_set).sum(dim=1).clamp_min(1).float()
    iou = (inter / union).mean(dim=1)
    return float(iou.sum().item()), int(B)


def run_epoch(model, loader, *, device, optim, loss_w, grad_clip: float = 5.0) -> dict:
    is_train = optim is not None
    model.train(is_train)

    total_n = 0
    sums: dict[str, float] = {"loss": 0.0, "L_rot": 0.0, "L_win": 0.0,
                              "L_delta": 0.0, "bce": 0.0}
    correct = 0
    grad_norm_sum = 0.0
    grad_norm_n = 0
    iou_sum = 0.0
    iou_n = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            if is_train:
                out = model(batch, teacher_force=True)
            else:
                out = model(batch, teacher_force=False, autoregressive=True)
            loss, diag = total_loss_cumstate(out, batch, **loss_w)

            if is_train:
                optim.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_norm_sum += float(gn)
                grad_norm_n += 1
                optim.step()

            with torch.no_grad():
                bce = F.binary_cross_entropy_with_logits(
                    out["win_logit"], batch["label"])
                pred = (torch.sigmoid(out["win_logit"]) > 0.5).float()
                correct += (pred == batch["label"]).sum().item()
                if not is_train:
                    s_h, n_h = _iou_top5(out["home_rotation_logits"],
                                         batch["home_rotation_target"],
                                         batch["home_mask"])
                    s_a, n_a = _iou_top5(out["away_rotation_logits"],
                                         batch["away_rotation_target"],
                                         batch["away_mask"])
                    iou_sum += (s_h + s_a) / 2.0
                    iou_n += (n_h + n_a) // 2

        bs = batch["label"].size(0)
        total_n += bs
        sums["loss"] += loss.item() * bs
        for k in ("L_rot", "L_win", "L_delta"):
            if k in diag:
                sums[k] += diag[k].item() * bs
        sums["bce"] += bce.item() * bs

    return {
        "n": total_n,
        "loss": sums["loss"] / total_n,
        "L_rot": sums["L_rot"] / total_n,
        "L_win": sums["L_win"] / total_n,
        "L_delta": sums["L_delta"] / total_n,
        "bce": sums["bce"] / total_n,
        "acc": correct / total_n,
        "iou_top5": (iou_sum / iou_n) if iou_n else 0.0,
        "mean_grad_norm": (grad_norm_sum / grad_norm_n) if grad_norm_n else 0.0,
    }


def _build_lr_scheduler(optim, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        denom = max(1, total_epochs - warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precomputed-db", type=Path, default=DEFAULT_PRECOMPUTED_DB)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="backtest_v6")
    # arch
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-enc", type=int, default=4)
    p.add_argument("--n-dec", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--rot-only-epochs", type=int, default=0,
                   help="If >0, first N epochs train only on L_rot (w_delta=w_win=0)")
    p.add_argument("--use-player-stats", action="store_true",
                   help="Inject per-player form features into encoder.")
    p.add_argument("--side-on-team-only", action="store_true",
                   help="Add E_side only to team tokens.")
    # loss weights
    p.add_argument("--w-rot", type=float, default=1.0)
    p.add_argument("--w-delta", type=float, default=1.0)
    p.add_argument("--w-win", type=float, default=1.0)
    p.add_argument("--track-metric", type=str, default="bce",
                   choices=["bce", "loss", "acc"])
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    # parallel sharding
    p.add_argument("--window-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--regular-season-only", action="store_true",
                   help="Exclude playoff games.")
    return p.parse_args()


def _aggregate_shards(out_dir: Path) -> None:
    shards_dir = out_dir / "_shards"
    files = sorted(shards_dir.glob("window_metrics.shard*.csv"))
    if not files:
        raise FileNotFoundError(f"No shard files in {shards_dir}")
    wm = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    wm = wm.sort_values("window_start").reset_index(drop=True)
    wm.to_csv(out_dir / "window_metrics.csv", index=False)

    overall = {
        "n_windows": len(wm),
        "avg_bce": float(wm["test_bce"].mean()),
        "avg_acc": float(wm["test_acc"].mean()),
    }
    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(f"[done] bce={overall['avg_bce']:.4f} acc={overall['avg_acc']:.3f} "
          f"over {len(wm)} windows")


def _run(args, out_dir):
    db_path = args.precomputed_db
    windows = load_precomputed_window_info(db_path)
    if args.max_windows is not None:
        windows = windows[:args.max_windows]
    print(f"[precomputed] {db_path} — {len(windows)} windows available")

    if args.window_shards > 1:
        windows = windows[args.shard_idx::args.window_shards]
        shards_dir = out_dir / "_shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        tag = f".shard{args.shard_idx}"
        wm_out = shards_dir / f"window_metrics{tag}.csv"
        print(f"[shard {args.shard_idx}/{args.window_shards}] {len(windows)} windows")
    else:
        wm_out = out_dir / "window_metrics.csv"

    rows = []
    training_log_path = out_dir / "training_log.csv"
    training_log_rows: list[dict] = []

    for wi, (window_start, window_end) in enumerate(windows):
        t_window = time.time()
        t_load = time.time()
        train_ds = PrecomputedDatasetV5(db_path, window_start, "train",
                                        regular_season_only=args.regular_season_only)
        val_ds = PrecomputedDatasetV5(db_path, window_start, "val",
                                      regular_season_only=args.regular_season_only)
        test_ds = PrecomputedDatasetV5(db_path, window_start, "test",
                                       regular_season_only=args.regular_season_only)
        print(f"\n[window {wi+1}/{len(windows)}] {window_start} → {window_end} | "
              f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
              f"(loaded in {time.time()-t_load:.1f}s)")

        vocab_size, team_vocab_size = load_precomputed_vocab_size(db_path, window_start)
        sample = train_ds[0]
        tabular_dim = sample["tabular"].numel()
        cfg = CmeV6Config(
            vocab_size=vocab_size, num_teams=team_vocab_size,
            tabular_dim=tabular_dim,
            d=args.d, n_heads=args.n_heads,
            n_enc=args.n_enc, n_dec=args.n_dec,
            dropout=args.dropout,
            side_on_team_only=args.side_on_team_only,
            player_stats_dim=sample["home_stats"].size(-1) if args.use_player_stats else 0,
        )
        model = CmeV6(cfg).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params: {n_params:,}")

        loss_w = {
            "w_rot": args.w_rot,
            "w_delta": args.w_delta,
            "w_win": args.w_win,
        }
        print(f"  loss weights: w_rot={args.w_rot} w_delta={args.w_delta} w_win={args.w_win}")

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
        scheduler = _build_lr_scheduler(optim, args.warmup_epochs, args.epochs)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_v5)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_v5)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_v5)

        metric_higher_better = args.track_metric == "acc"
        best_val = float("-inf") if metric_higher_better else float("inf")
        best_state = None
        best_epoch = -1
        epochs_since_best = 0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            if epoch <= args.rot_only_epochs:
                loss_w_ep = dict(loss_w)
                loss_w_ep["w_delta"] = 0.0
                loss_w_ep["w_win"] = 0.0
            else:
                loss_w_ep = loss_w

            tr = run_epoch(model, train_loader, device=args.device,
                           optim=optim, loss_w=loss_w_ep, grad_clip=args.grad_clip)
            va = run_epoch(model, val_loader, device=args.device,
                           optim=None, loss_w=loss_w_ep, grad_clip=args.grad_clip)
            scheduler.step()
            dt = time.time() - t0

            print(f"  [ep{epoch:3d}] tr_loss={tr['loss']:.3f} va_loss={va['loss']:.3f} "
                  f"bce={va['bce']:.4f} acc={va['acc']:.3f} iou5={va['iou_top5']:.3f} "
                  f"L_rot={va['L_rot']:.4f} L_delta={va['L_delta']:.4f} "
                  f"gn={tr['mean_grad_norm']:.2f} ({dt:.0f}s)")

            log_row = {
                "window_start": window_start, "epoch": epoch, "secs": dt,
                "tr_loss": tr["loss"], "va_loss": va["loss"],
                "tr_bce": tr["bce"], "va_bce": va["bce"],
                "tr_acc": tr["acc"], "va_acc": va["acc"],
                "tr_L_rot": tr["L_rot"], "va_L_rot": va["L_rot"],
                "tr_L_delta": tr["L_delta"], "va_L_delta": va["L_delta"],
                "tr_L_win": tr["L_win"], "va_L_win": va["L_win"],
                "va_iou_top5": va["iou_top5"],
                "mean_grad_norm": tr["mean_grad_norm"],
            }
            training_log_rows.append(log_row)
            pd.DataFrame(training_log_rows).to_csv(training_log_path, index=False)

            metric = va[args.track_metric]
            improved = (metric > best_val + 1e-5) if metric_higher_better else (metric < best_val - 1e-5)
            if improved:
                best_val = metric
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                if epochs_since_best >= args.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            ckpt_path = out_dir / f"ckpt_{window_start}.pt"
            torch.save({
                "state_dict": best_state,
                "cfg": cfg,
                "vocab_size": vocab_size,
                "team_vocab_size": team_vocab_size,
                "tabular_dim": tabular_dim,
                "window_start": window_start,
                "window_end": window_end,
                "best_epoch": best_epoch,
                "best_val": best_val,
            }, ckpt_path)
            print(f"  saved checkpoint → {ckpt_path.name}")
        te = run_epoch(model, test_loader, device=args.device, optim=None, loss_w=loss_w)
        secs = time.time() - t_window
        print(f"[window {wi+1}] best_ep={best_epoch} "
              f"test bce={te['bce']:.4f} acc={te['acc']:.3f} ({secs:.0f}s)")

        row = {
            "window_start": window_start, "window_end": window_end,
            "train_n": len(train_ds), "val_n": len(val_ds), "test_n": len(test_ds),
            "best_epoch": best_epoch, "epochs_run": epoch,
            "test_bce": te["bce"], "test_acc": te["acc"],
            "test_L_rot": te["L_rot"], "test_L_delta": te["L_delta"],
            "test_iou_top5": te["iou_top5"],
            "secs": secs,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(wm_out, index=False)

    if args.window_shards <= 1:
        avg_acc = float(np.mean([r["test_acc"] for r in rows]))
        avg_bce = float(np.mean([r["test_bce"] for r in rows]))
        print(f"\n[summary] bce={avg_bce:.4f} acc={avg_acc:.3f} over {len(rows)} windows")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[device] {args.device}")
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    if args.aggregate_only:
        _aggregate_shards(out_dir)
        return

    _run(args, out_dir)


if __name__ == "__main__":
    main()
