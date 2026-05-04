#!/usr/bin/env python3
"""Overfitting profiler: detailed per-epoch diagnostics on a single window.

Captures per-component loss curves (train vs val), prediction distribution
stats, weight norms, and teacher-forced vs AR eval gap to pinpoint what's
overfitting and why.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cme_v5_common import PrecomputedDatasetV5, collate_v5, load_precomputed_window_info, load_precomputed_vocab_size
from model import CmeV6, CmeV6Config, rotation_loss, box_cum_loss, win_bce, win_hinge


def prediction_stats(out, batch, Lh):
    """Capture distribution stats of model outputs."""
    stats = {}

    # Rotation logit stats
    h_rl = out["home_rotation_logits"]
    a_rl = out["away_rotation_logits"]
    h_mask = batch["home_mask"].unsqueeze(-1)
    a_mask = batch["away_mask"].unsqueeze(-1)
    h_valid = h_rl[h_mask.expand_as(h_rl)]
    a_valid = a_rl[a_mask.expand_as(a_rl)]
    all_rot = torch.cat([h_valid, a_valid])
    stats["rot_logit_mean"] = float(all_rot.mean())
    stats["rot_logit_std"] = float(all_rot.std())
    stats["rot_logit_min"] = float(all_rot.min())
    stats["rot_logit_max"] = float(all_rot.max())
    stats["rot_sigmoid_mean"] = float(torch.sigmoid(all_rot).mean())

    # Box prediction stats
    box = out["box_cum"]  # (B, S, L, 14)
    P_mask = torch.cat([batch["home_mask"], batch["away_mask"]], dim=1)
    mask = P_mask.unsqueeze(1).unsqueeze(-1)
    box_valid = box[mask.expand_as(box)]
    stats["box_mean"] = float(box_valid.mean())
    stats["box_std"] = float(box_valid.std())
    stats["box_max"] = float(box_valid.max())

    # Per-channel final-minute cumulative stats (= predicted game totals)
    final_box = box[:, -1, :, :]  # (B, L, 14) — minute 48
    ch_names = ["pts", "fgm", "fga", "3pm", "3pa", "ftm", "fta",
                "ast", "tov", "blk", "oreb", "dreb", "stl", "pf"]
    for i, ch in enumerate(ch_names):
        vals = final_box[:, :, i][P_mask]
        stats[f"box48_{ch}_mean"] = float(vals.mean())
        stats[f"box48_{ch}_max"] = float(vals.max())

    # Win logit stats
    wl = out["win_logit"]
    stats["win_logit_mean"] = float(wl.mean())
    stats["win_logit_std"] = float(wl.std())
    stats["win_logit_absmax"] = float(wl.abs().max())

    # Home/away points
    stats["home_pts_mean"] = float(out["home_points"].mean())
    stats["away_pts_mean"] = float(out["away_points"].mean())

    return stats


def weight_norms(model):
    """Per-module parameter norms."""
    norms = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top not in norms:
            norms[top] = 0.0
        norms[top] += float(param.detach().pow(2).sum())
    return {k: v ** 0.5 for k, v in norms.items()}


def _forward_tf_rot_only(model, batch):
    """Teacher-force rotations but zero out cumstats — isolates rotation signal."""
    home_mask = batch["home_mask"]
    Lh = home_mask.size(1)
    P, P_mask, home_team_out, away_team_out = model._encode(batch)

    # Build GT rotation mask (same as full TF)
    h_rot = batch["home_rotation_target"]
    a_rot = batch["away_rotation_target"]
    rot = torch.cat([h_rot, a_rot], dim=1)
    rot_shifted = torch.cat([rot[:, :, :1], rot[:, :, :-1]], dim=2)
    rot_mask = rot_shifted.permute(0, 2, 1)
    rot_mask = rot_mask * P_mask.unsqueeze(1).float()
    ca_mask = rot_mask > 0.5

    # Zero cumstats — model gets no box score history
    B, L = P_mask.shape
    S = model.cfg.n_segments
    cumstats_input = torch.zeros(B, S, L, model.cfg.n_box_channels, device=P.device)

    D, P_aug = model._decode_cumstate(P, P_mask, cumstats_input, ca_mask)
    out = model._cumstate_heads(D, P, P_aug, P_mask, Lh)
    out["home_team_out"] = home_team_out
    out["away_team_out"] = away_team_out
    return out


def eval_split(model, loader, device, mode="ar", win_loss="hinge"):
    """Eval a full split. mode='tf'/'ar'/'tf_rot' (rotation-only TF, zero cumstats)."""
    model.eval()
    total_n = 0
    sums = {"L_rot": 0.0, "L_delta": 0.0, "L_win": 0.0, "bce": 0.0}
    correct = 0
    all_stats = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            Lh = batch["home_mask"].size(1)
            if mode == "tf":
                out = model(batch, teacher_force=True)
            elif mode == "tf_rot":
                out = _forward_tf_rot_only(model, batch)
            else:
                out = model(batch, teacher_force=False, autoregressive=True)

            L_rot = rotation_loss(out, batch)
            L_delta = box_cum_loss(out, batch)
            if win_loss == "hinge":
                L_win = win_hinge(out["win_logit"], batch["label"])
            else:
                L_win = win_bce(out["win_logit"], batch["label"])
            bce = F.binary_cross_entropy_with_logits(out["win_logit"], batch["label"])
            pred = (torch.sigmoid(out["win_logit"]) > 0.5).float()

            bs = batch["label"].size(0)
            total_n += bs
            sums["L_rot"] += float(L_rot) * bs
            sums["L_delta"] += float(L_delta) * bs
            sums["L_win"] += float(L_win) * bs
            sums["bce"] += float(bce) * bs
            correct += int((pred == batch["label"]).sum())
            all_stats.append(prediction_stats(out, batch, Lh))

    avg = {k: v / total_n for k, v in sums.items()}
    avg["acc"] = correct / total_n

    # Average prediction stats across batches
    pred_avg = {}
    for k in all_stats[0]:
        pred_avg[k] = sum(s[k] for s in all_stats) / len(all_stats)

    return avg, pred_avg


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w-delta", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n-enc", type=int, default=4)
    p.add_argument("--n-dec", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ss-warmup", type=int, default=5,
                   help="Epochs to linearly ramp ss_ratio from 0 to ss-max")
    p.add_argument("--ss-max", type=float, default=0.5,
                   help="Max scheduled sampling ratio (0=pure TF, 1=pure AR)")
    p.add_argument("--win-loss", default="hinge", choices=["hinge", "bce"],
                   help="Loss function for win prediction")
    p.add_argument("--no-player-emb", action="store_true",
                   help="Drop learned player embeddings, use stats + career year only")
    args = p.parse_args()

    db_path = REPO_ROOT / "data" / "features_v5_precomputed.db"
    windows = load_precomputed_window_info(db_path)
    window_start = windows[0][0]

    train_ds = PrecomputedDatasetV5(db_path, window_start, "train")
    val_ds = PrecomputedDatasetV5(db_path, window_start, "val")
    test_ds = PrecomputedDatasetV5(db_path, window_start, "test")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    vocab_size, team_vocab_size = load_precomputed_vocab_size(db_path, window_start)
    sample = train_ds[0]
    stats_dim = sample["home_stats"].size(-1) if sample["home_stats"].numel() > 0 else 0
    cfg = CmeV6Config(
        vocab_size=vocab_size, num_teams=team_vocab_size,
        tabular_dim=sample["tabular"].numel(),
        d=args.d, n_heads=args.n_heads,
        n_enc=args.n_enc, n_dec=args.n_dec,
        dropout=args.dropout,
        player_stats_dim=stats_dim,
        use_player_embeddings=not args.no_player_emb,
    )
    model = CmeV6(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train_samples = len(train_ds)
    print(f"params: {n_params:,} | train samples: {n_train_samples} | ratio: {n_params/n_train_samples:.1f} params/sample")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_v5)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v5)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v5)

    # Epoch 0: untrained baseline
    print("\n--- Epoch 0 (untrained) ---")
    tr_m, tr_ps = eval_split(model, train_loader, args.device, mode="tf", win_loss=args.win_loss)
    va_tf, va_tf_ps = eval_split(model, val_loader, args.device, mode="tf", win_loss=args.win_loss)
    va_rot, va_rot_ps = eval_split(model, val_loader, args.device, mode="tf_rot", win_loss=args.win_loss)
    va_ar, va_ar_ps = eval_split(model, val_loader, args.device, mode="ar", win_loss=args.win_loss)
    wn = weight_norms(model)

    print(f"  TRAIN(tf)  L_rot={tr_m['L_rot']:.4f} L_delta={tr_m['L_delta']:.4f} L_win={tr_m['L_win']:.4f} bce={tr_m['bce']:.4f} acc={tr_m['acc']:.3f}")
    print(f"  VAL(tf)    L_rot={va_tf['L_rot']:.4f} L_delta={va_tf['L_delta']:.4f} L_win={va_tf['L_win']:.4f} bce={va_tf['bce']:.4f} acc={va_tf['acc']:.3f}")
    print(f"  VAL(rot)   L_rot={va_rot['L_rot']:.4f} L_delta={va_rot['L_delta']:.4f} L_win={va_rot['L_win']:.4f} bce={va_rot['bce']:.4f} acc={va_rot['acc']:.3f}")
    print(f"  VAL(ar)    L_rot={va_ar['L_rot']:.4f} L_delta={va_ar['L_delta']:.4f} L_win={va_ar['L_win']:.4f} bce={va_ar['bce']:.4f} acc={va_ar['acc']:.3f}")
    print(f"  pred(tf):   rot_logit μ={va_tf_ps['rot_logit_mean']:.3f} σ={va_tf_ps['rot_logit_std']:.3f} | box48_pts μ={va_tf_ps['box48_pts_mean']:.1f} max={va_tf_ps['box48_pts_max']:.1f}")
    print(f"  pred(rot):  rot_logit μ={va_rot_ps['rot_logit_mean']:.3f} σ={va_rot_ps['rot_logit_std']:.3f} | box48_pts μ={va_rot_ps['box48_pts_mean']:.1f} max={va_rot_ps['box48_pts_max']:.1f}")
    print(f"  pred(ar):   rot_logit μ={va_ar_ps['rot_logit_mean']:.3f} σ={va_ar_ps['rot_logit_std']:.3f} | box48_pts μ={va_ar_ps['box48_pts_mean']:.1f} max={va_ar_ps['box48_pts_max']:.1f}")
    print(f"  pred(rot):  home_pts={va_rot_ps['home_pts_mean']:.1f} away_pts={va_rot_ps['away_pts_mean']:.1f} win_logit σ={va_rot_ps['win_logit_std']:.3f}")
    print(f"  pred(ar):   home_pts={va_ar_ps['home_pts_mean']:.1f} away_pts={va_ar_ps['away_pts_mean']:.1f} win_logit σ={va_ar_ps['win_logit_std']:.3f}")
    print(f"  weights:    {' '.join(f'{k}={v:.2f}' for k,v in sorted(wn.items()))}")
    te_ar, _ = eval_split(model, test_loader, args.device, mode="ar", win_loss=args.win_loss)
    print(f"  TEST(ar)   acc={te_ar['acc']:.3f} bce={te_ar['bce']:.4f}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        grad_norm_sum = 0.0

        ss_ratio = min(epoch / max(args.ss_warmup, 1), 1.0) * args.ss_max

        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch, teacher_force=True, ss_ratio=ss_ratio)
            L_rot = rotation_loss(out, batch)
            L_delta = box_cum_loss(out, batch)
            if args.win_loss == "hinge":
                L_win = win_hinge(out["win_logit"], batch["label"])
            else:
                L_win = win_bce(out["win_logit"], batch["label"])
            loss = 1.0 * L_rot + args.w_delta * L_delta + 1.0 * L_win
            optim.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            grad_norm_sum += float(gn)
            optim.step()
            train_loss_sum += float(loss) * batch["label"].size(0)
            train_n += batch["label"].size(0)

        dt = time.time() - t0

        print(f"\n--- Epoch {epoch} ({dt:.0f}s, gn={grad_norm_sum/len(train_loader):.2f}, ss={ss_ratio:.2f}) ---")

        tr_m, tr_ps = eval_split(model, train_loader, args.device, mode="tf", win_loss=args.win_loss)
        va_tf, va_tf_ps = eval_split(model, val_loader, args.device, mode="tf", win_loss=args.win_loss)
        va_rot, va_rot_ps = eval_split(model, val_loader, args.device, mode="tf_rot", win_loss=args.win_loss)
        va_ar, va_ar_ps = eval_split(model, val_loader, args.device, mode="ar", win_loss=args.win_loss)
        wn = weight_norms(model)

        print(f"  TRAIN(tf)  L_rot={tr_m['L_rot']:.4f} L_delta={tr_m['L_delta']:.4f} L_win={tr_m['L_win']:.4f} bce={tr_m['bce']:.4f} acc={tr_m['acc']:.3f}")
        print(f"  VAL(tf)    L_rot={va_tf['L_rot']:.4f} L_delta={va_tf['L_delta']:.4f} L_win={va_tf['L_win']:.4f} bce={va_tf['bce']:.4f} acc={va_tf['acc']:.3f}")
        print(f"  VAL(rot)   L_rot={va_rot['L_rot']:.4f} L_delta={va_rot['L_delta']:.4f} L_win={va_rot['L_win']:.4f} bce={va_rot['bce']:.4f} acc={va_rot['acc']:.3f}")
        print(f"  VAL(ar)    L_rot={va_ar['L_rot']:.4f} L_delta={va_ar['L_delta']:.4f} L_win={va_ar['L_win']:.4f} bce={va_ar['bce']:.4f} acc={va_ar['acc']:.3f}")

        # Gaps
        print(f"  GAPS:      rot={va_tf['L_rot']-tr_m['L_rot']:+.4f} delta={va_tf['L_delta']-tr_m['L_delta']:+.4f} win={va_tf['L_win']-tr_m['L_win']:+.4f} bce={va_tf['bce']-tr_m['bce']:+.4f}")
        print(f"  TF-AR gap: bce={va_ar['bce']-va_tf['bce']:+.4f} acc={va_ar['acc']-va_tf['acc']:+.3f} L_delta={va_ar['L_delta']-va_tf['L_delta']:+.4f}")

        print(f"  pred(tf):   rot_logit μ={va_tf_ps['rot_logit_mean']:.3f} σ={va_tf_ps['rot_logit_std']:.3f} [{va_tf_ps['rot_logit_min']:.2f},{va_tf_ps['rot_logit_max']:.2f}]")
        print(f"  pred(rot):  rot_logit μ={va_rot_ps['rot_logit_mean']:.3f} σ={va_rot_ps['rot_logit_std']:.3f} [{va_rot_ps['rot_logit_min']:.2f},{va_rot_ps['rot_logit_max']:.2f}]")
        print(f"  pred(ar):   rot_logit μ={va_ar_ps['rot_logit_mean']:.3f} σ={va_ar_ps['rot_logit_std']:.3f} [{va_ar_ps['rot_logit_min']:.2f},{va_ar_ps['rot_logit_max']:.2f}]")
        print(f"  pred(tf):   box48_pts μ={va_tf_ps['box48_pts_mean']:.1f} max={va_tf_ps['box48_pts_max']:.1f} | box μ={va_tf_ps['box_mean']:.2f} σ={va_tf_ps['box_std']:.2f} max={va_tf_ps['box_max']:.1f}")
        print(f"  pred(rot):  box48_pts μ={va_rot_ps['box48_pts_mean']:.1f} max={va_rot_ps['box48_pts_max']:.1f} | box μ={va_rot_ps['box_mean']:.2f} σ={va_rot_ps['box_std']:.2f} max={va_rot_ps['box_max']:.1f}")
        print(f"  pred(ar):   box48_pts μ={va_ar_ps['box48_pts_mean']:.1f} max={va_ar_ps['box48_pts_max']:.1f} | box μ={va_ar_ps['box_mean']:.2f} σ={va_ar_ps['box_std']:.2f} max={va_ar_ps['box_max']:.1f}")
        print(f"  pred(rot):  home_pts={va_rot_ps['home_pts_mean']:.1f} away_pts={va_rot_ps['away_pts_mean']:.1f} win_logit σ={va_rot_ps['win_logit_std']:.3f}")
        print(f"  pred(ar):   home_pts={va_ar_ps['home_pts_mean']:.1f} away_pts={va_ar_ps['away_pts_mean']:.1f} win_logit σ={va_ar_ps['win_logit_std']:.3f}")
        print(f"  weights:    {' '.join(f'{k}={v:.2f}' for k,v in sorted(wn.items()))}")
        te_ar, _ = eval_split(model, test_loader, args.device, mode="ar", win_loss=args.win_loss)
        print(f"  TEST(ar)   acc={te_ar['acc']:.3f} bce={te_ar['bce']:.4f}")


if __name__ == "__main__":
    main()
