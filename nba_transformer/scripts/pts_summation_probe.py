"""Verify the 'summation washes out lineup' hypothesis.

For each test game, compute (home_pts, away_pts, win_logit) under:
  - normal_tf            : teacher-forced w/ true rotation
  - random_lineup_tf     : teacher-forced w/ uniformly-random 5-per-side rotation
  - normal_autoreg       : honest argtop-5 rollout

If the win head is decoupled from the lineup pathway, paired
(home_pts, away_pts) totals will be ~identical between normal_tf and
random_lineup_tf despite the gate being completely different per minute.

Reports per-mode: mean home_pts/away_pts/diff, std, paired-mode correlation,
and paired mean absolute difference vs normal_tf.

Usage:
  python nba_transformer/scripts/pts_summation_probe.py \
      --ckpt nba_transformer/artifacts/v6_ckpt_baseline/ckpt_2024-01-01.pt \
      --db data/features_v5_precomputed.db --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "nba_transformer"))

from cme_v5_common import (  # noqa: E402
    PrecomputedDatasetV5, collate_v5,
)
from model import CmeV6, CmeV6Config  # noqa: E402
from perturb_eval import _random_lineup_target  # noqa: E402


@torch.no_grad()
def collect(model: CmeV6, loader: DataLoader, mode: str, *, device: str,
            gen: torch.Generator) -> dict:
    model.eval()
    home_list, away_list, logit_list = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        b = {k: v for k, v in batch.items()}
        if mode == "random_lineup_tf":
            b["home_rotation_target"] = _random_lineup_target(b["home_rotation_target"], b["home_mask"], gen)
            b["away_rotation_target"] = _random_lineup_target(b["away_rotation_target"], b["away_mask"], gen)
        if mode in ("normal_tf", "random_lineup_tf"):
            out = model(b, teacher_force=True)
        elif mode == "normal_autoreg":
            out = model(b, teacher_force=False, autoregressive=True)
        else:
            raise ValueError(mode)
        home_list.append(out["home_points"].cpu())
        away_list.append(out["away_points"].cpu())
        logit_list.append(out["win_logit"].cpu())
    return {
        "home_pts": torch.cat(home_list),
        "away_pts": torch.cat(away_list),
        "win_logit": torch.cat(logit_list),
    }


def summarize(name: str, d: dict) -> dict:
    h, a, w = d["home_pts"], d["away_pts"], d["win_logit"]
    diff = h - a
    return {
        "mode": name,
        "home_mean": float(h.mean()), "home_std": float(h.std()),
        "away_mean": float(a.mean()), "away_std": float(a.std()),
        "diff_mean": float(diff.mean()), "diff_std": float(diff.std()),
        "diff_abs_mean": float(diff.abs().mean()),
        "logit_mean": float(w.mean()), "logit_std": float(w.std()),
    }


def paired_compare(a: dict, b: dict, name_a: str, name_b: str) -> dict:
    """Paired stats: a vs b (must be same games in same order)."""
    dh = a["home_pts"] - b["home_pts"]
    da = a["away_pts"] - b["away_pts"]
    ddiff = (a["home_pts"] - a["away_pts"]) - (b["home_pts"] - b["away_pts"])
    dwin = a["win_logit"] - b["win_logit"]
    def corr(x, y):
        xm, ym = x - x.mean(), y - y.mean()
        denom = (xm.pow(2).sum() * ym.pow(2).sum()).sqrt().clamp(min=1e-9)
        return float((xm * ym).sum() / denom)
    return {
        "pair": f"{name_a}_vs_{name_b}",
        "home_pts_corr": corr(a["home_pts"], b["home_pts"]),
        "away_pts_corr": corr(a["away_pts"], b["away_pts"]),
        "win_logit_corr": corr(a["win_logit"], b["win_logit"]),
        "home_pts_mad": float(dh.abs().mean()),
        "away_pts_mad": float(da.abs().mean()),
        "diff_mad": float(ddiff.abs().mean()),  # |Δ(home-away)|
        "win_logit_mad": float(dwin.abs().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--regular-season-only", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg: CmeV6Config = ck["cfg"]
    window_start = ck["window_start"]
    model = CmeV6(cfg).to(args.device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ds = PrecomputedDatasetV5(args.db, window_start, args.split,
                              regular_season_only=args.regular_season_only)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v5)
    print(f"[ckpt] {args.ckpt}  window={window_start}  best_ep={ck['best_epoch']}")
    print(f"[data] split={args.split} n={len(ds)}\n")

    modes = ["normal_tf", "random_lineup_tf", "normal_autoreg"]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    results = {}
    for m in modes:
        results[m] = collect(model, loader, m, device=args.device, gen=gen)

    print("Per-mode totals (home_pts, away_pts, win_logit):")
    print("-" * 92)
    print(f"{'mode':<22} {'home_mean':>10} {'home_std':>9} {'away_mean':>10} {'away_std':>9} "
          f"{'diff_mean':>10} {'logit_std':>9}")
    summaries = []
    for m in modes:
        s = summarize(m, results[m])
        summaries.append(s)
        print(f"{m:<22} {s['home_mean']:>10.2f} {s['home_std']:>9.2f} "
              f"{s['away_mean']:>10.2f} {s['away_std']:>9.2f} "
              f"{s['diff_mean']:>10.3f} {s['logit_std']:>9.3f}")

    print("\nPaired comparisons (per-game absolute deltas vs normal_tf):")
    print("-" * 92)
    print(f"{'pair':<32} {'h_corr':>7} {'a_corr':>7} {'w_corr':>7} "
          f"{'h_mad':>7} {'a_mad':>7} {'diff_mad':>9} {'w_mad':>7}")
    pairs = []
    for m in ["random_lineup_tf", "normal_autoreg"]:
        p = paired_compare(results[m], results["normal_tf"], m, "normal_tf")
        pairs.append(p)
        print(f"{p['pair']:<32} {p['home_pts_corr']:>7.3f} {p['away_pts_corr']:>7.3f} "
              f"{p['win_logit_corr']:>7.3f} {p['home_pts_mad']:>7.2f} {p['away_pts_mad']:>7.2f} "
              f"{p['diff_mad']:>9.3f} {p['win_logit_mad']:>7.3f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "ckpt": str(args.ckpt), "split": args.split,
            "summaries": summaries, "pairs": pairs,
        }, indent=2) + "\n")
        print(f"\n[wrote] {args.out_json}")


if __name__ == "__main__":
    main()
