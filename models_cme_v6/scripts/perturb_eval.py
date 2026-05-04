"""Diagnostic eval that perturbs inputs to a trained CME-v6 checkpoint.

Modes:
  normal_autoreg     — autoregressive argtop-5 rollout (the honest eval)
  normal_tf          — teacher-forced eval (ceiling reference)
  random_player_ids  — random home_idx/away_idx within vocab (mask-respecting)
  random_team_ids    — random home_team_idx/away_team_idx within team vocab
  random_tabular     — shuffle tabular features across the batch
  random_all_enc     — randomize player IDs + team IDs + tabular + rest jointly
  random_lineup_tf   — teacher-force a uniformly-random 5-per-side lineup as truth
  random_label_check — compute BCE on random labels (sanity prior)

For each mode it reports: bce, acc, L_rot, L_pts, L_win, iou_top5.

Usage:
  python models_cme_v6/scripts/perturb_eval.py \
      --ckpt models_cme_v6/artifacts/v6_ckpt_baseline/ckpt_2024-01-01.pt \
      --db data/features_v5_precomputed.db \
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "models_cme_v5" / "scripts"))
sys.path.insert(0, str(REPO / "models_cme_v6" / "scripts"))

from cme_v5_common import (  # noqa: E402
    PrecomputedDatasetV5, collate_v5, load_precomputed_vocab_size,
)
from model import CmeV6, CmeV6Config  # noqa: E402
from model import (  # noqa: E402
    pl_rotation_loss, rotation_loss, minute_emit_loss, win_bce, total_loss,
)


def _iou_top5(rot_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    B, L, S = target.shape
    valid = mask.any(dim=1)
    if valid.sum() == 0:
        return 0.0, 0
    top_idx = rot_logits.topk(5, dim=1).indices  # (B, 5, S)
    pred = torch.zeros_like(rot_logits)
    pred.scatter_(1, top_idx, 1.0)
    pred = pred * mask.unsqueeze(-1).float()
    tgt = target * mask.unsqueeze(-1).float()
    inter = (pred * tgt).sum(dim=(1, 2))
    union = (pred + tgt - pred * tgt).sum(dim=(1, 2)).clamp(min=1.0)
    iou = (inter / union)[valid]
    return float(iou.sum()), int(valid.sum())


MODES = (
    "normal_autoreg",
    "normal_tf",
    "random_player_ids",
    "random_team_ids",
    "random_tabular",
    "random_all_enc",
    "random_lineup_tf",
    "random_label_check",
)


def _random_lineup_target(rot_target: torch.Tensor, mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Generate random 5-per-side one-hot rotation (B, L_team, S) respecting mask.

    For each (b, s), pick 5 valid players uniformly at random from mask[b].
    """
    B, L, S = rot_target.shape
    dev = rot_target.device
    out = torch.zeros_like(rot_target)
    for b in range(B):
        valid = torch.where(mask[b])[0]
        if valid.numel() < 5:
            out[b] = rot_target[b]
            continue
        for s in range(S):
            perm = torch.randperm(valid.numel(), generator=gen, device="cpu")[:5]
            chosen = valid[perm.to(dev)]
            out[b, chosen, s] = 1.0
    return out


def perturb_batch(batch: dict, mode: str, vocab_size: int, team_vocab_size: int,
                  gen: torch.Generator) -> dict:
    """Return a perturbed shallow-copy of the batch."""
    b = {k: v for k, v in batch.items()}
    dev = b["home_mask"].device
    B = b["home_mask"].size(0)

    if mode in ("random_player_ids", "random_all_enc"):
        for k in ("home_idx", "away_idx"):
            shape = b[k].shape
            rand = torch.randint(0, vocab_size, shape, device="cpu", generator=gen).to(dev)
            b[k] = rand
    if mode in ("random_team_ids", "random_all_enc"):
        for k in ("home_team_idx", "away_team_idx"):
            shape = b[k].shape
            rand = torch.randint(0, team_vocab_size, shape, device="cpu", generator=gen).to(dev)
            b[k] = rand
    if mode in ("random_tabular", "random_all_enc"):
        tab = b["tabular"]
        perm = torch.randperm(tab.size(0), generator=gen, device="cpu").to(dev)
        b["tabular"] = tab[perm]
    if mode == "random_all_enc":
        for k in ("home_rest", "away_rest"):
            perm = torch.randperm(B, generator=gen, device="cpu").to(dev)
            b[k] = b[k][perm]
    if mode == "random_lineup_tf":
        b["home_rotation_target"] = _random_lineup_target(b["home_rotation_target"], b["home_mask"], gen)
        b["away_rotation_target"] = _random_lineup_target(b["away_rotation_target"], b["away_mask"], gen)
    return b


@torch.no_grad()
def eval_mode(model: CmeV6, loader: DataLoader, mode: str, *, device: str,
              vocab_size: int, team_vocab_size: int, loss_w: dict, seed: int = 0) -> dict:
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sums = {"loss": 0.0, "L_rot": 0.0, "L_pts": 0.0, "L_win": 0.0,
            "L_emit_pts": 0.0, "L_emit_aux": 0.0, "bce": 0.0}
    total_n = 0
    correct = 0
    iou_sum = 0.0
    iou_n = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = perturb_batch(batch, mode, vocab_size, team_vocab_size, gen)

        if mode == "normal_autoreg":
            out = model(batch, teacher_force=False, autoregressive=True)
        elif mode == "normal_tf":
            out = model(batch, teacher_force=True)
        elif mode == "random_lineup_tf":
            out = model(batch, teacher_force=True)
        else:
            # All encoder-perturb modes use the honest autoreg eval path.
            out = model(batch, teacher_force=False, autoregressive=True)

        if mode == "random_label_check":
            target = torch.bernoulli(torch.full_like(batch["label"], 0.5), generator=None)
        else:
            target = batch["label"]

        loss, diag = total_loss(out, batch, **loss_w)
        bce = F.binary_cross_entropy_with_logits(out["win_logit"], target)
        pred = (torch.sigmoid(out["win_logit"]) > 0.5).float()
        correct += (pred == target).sum().item()

        bs = batch["label"].size(0)
        total_n += bs
        sums["bce"] += bce.item() * bs
        sums["loss"] += loss.item() * bs
        for k in ("L_rot", "L_pts", "L_win", "L_emit_pts", "L_emit_aux"):
            if k in diag:
                sums[k] += diag[k].item() * bs

        s_h, n_h = _iou_top5(out["home_rotation_logits"],
                             batch["home_rotation_target"],
                             batch["home_mask"])
        s_a, n_a = _iou_top5(out["away_rotation_logits"],
                             batch["away_rotation_target"],
                             batch["away_mask"])
        iou_sum += (s_h + s_a) / 2.0
        iou_n += (n_h + n_a) // 2

    return {
        "mode": mode,
        "n": total_n,
        "loss": sums["loss"] / total_n,
        "bce": sums["bce"] / total_n,
        "acc": correct / total_n,
        "L_rot": sums["L_rot"] / total_n,
        "L_pts": sums["L_pts"] / total_n,
        "L_win": sums["L_win"] / total_n,
        "iou_top5": (iou_sum / iou_n) if iou_n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--regular-season-only", action="store_true", default=True)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    print(f"[load] ckpt={args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: CmeV6Config = ck["cfg"]
    vocab_size = ck["vocab_size"]
    team_vocab_size = ck["team_vocab_size"]
    emit_stds = ck["emit_stds"]
    window_start = ck["window_start"]
    window_end = ck["window_end"]
    print(f"  window={window_start} → {window_end}  best_ep={ck['best_epoch']}  best_val={ck['best_val']:.4f}")
    print(f"  vocab={vocab_size} teams={team_vocab_size} emit_stds={emit_stds}")

    model = CmeV6(cfg).to(args.device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ds = PrecomputedDatasetV5(args.db, window_start, args.split,
                              regular_season_only=args.regular_season_only)
    print(f"  {args.split} split: n={len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v5)

    loss_w = {
        "w_rot": 1.0,
        "w_emit_pts": 0.02,
        "w_emit_aux": 0.01,
        "w_win": 1.0,
        "emit_channels": cfg.emit_channels,
        "emit_stds": emit_stds,
        "pl_rotation": cfg.pl_rotation,
    }

    results = []
    for mode in args.modes:
        print(f"\n[eval] mode={mode}")
        r = eval_mode(model, loader, mode,
                      device=args.device,
                      vocab_size=vocab_size, team_vocab_size=team_vocab_size,
                      loss_w=loss_w, seed=args.seed)
        results.append(r)
        print(f"  bce={r['bce']:.4f}  acc={r['acc']:.4f}  L_rot={r['L_rot']:.3f}  "
              f"L_win={r['L_win']:.3f}  iou5={r['iou_top5']:.3f}  n={r['n']}")

    # Pretty table.
    print("\n" + "=" * 92)
    print(f"{'mode':<22} {'bce':>8} {'acc':>7} {'L_rot':>8} {'L_win':>8} {'iou5':>7} {'n':>6}")
    print("-" * 92)
    for r in results:
        print(f"{r['mode']:<22} {r['bce']:>8.4f} {r['acc']:>7.4f} {r['L_rot']:>8.3f} "
              f"{r['L_win']:>8.3f} {r['iou_top5']:>7.3f} {r['n']:>6}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "ckpt": str(args.ckpt), "window_start": window_start,
            "best_epoch": ck["best_epoch"], "best_val": ck["best_val"],
            "results": results,
        }, indent=2) + "\n")
        print(f"\n[wrote] {args.out_json}")


if __name__ == "__main__":
    main()
