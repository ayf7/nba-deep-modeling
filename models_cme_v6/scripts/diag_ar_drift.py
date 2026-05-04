"""Diagnose autoregressive rotation drift vs ground truth.

For each test game, runs the model in AR mode and compares predicted
top-5 lineup per minute against GT rotation. Reports:
  - Per-minute accuracy (fraction of 5 predicted players matching GT)
  - Drift over time (does accuracy degrade as minutes advance?)
  - How wrong predictions affect downstream pts computation
"""
import argparse, sys, pathlib, torch, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from model import CmeV6, CmeV6Config


def compute_ar_drift(model, loader, device):
    model.eval()
    S = model.cfg.n_segments  # 48

    # per-minute stats
    correct_per_min = np.zeros(S)
    total_per_min = np.zeros(S)
    # per-minute rotation BCE (sigmoid output vs GT)
    rot_bce_per_min = np.zeros(S)
    rot_bce_count = np.zeros(S)
    # per-minute overlap: |predicted_top5 ∩ gt_top5| / 5
    overlap_per_min = np.zeros(S)
    overlap_count = np.zeros(S)

    # Also track teacher-forced rotation accuracy for comparison
    tf_correct_per_min = np.zeros(S)
    tf_total_per_min = np.zeros(S)

    n_games = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # AR forward
            out_ar = model(batch, teacher_force=False, autoregressive=True)
            # TF forward
            out_tf = model(batch, teacher_force=True)

            home_rot_gt = batch["home_rotation_target"]  # (B, L_h, S)
            away_rot_gt = batch["away_rotation_target"]
            home_mask = batch["home_mask"]  # (B, L_h)
            away_mask = batch["away_mask"]

            B = home_rot_gt.size(0)
            Lh = home_rot_gt.size(1)
            n_games += B

            for side, rot_gt, mask, offset in [
                ("home", home_rot_gt, home_mask, 0),
                ("away", away_rot_gt, away_mask, Lh),
            ]:
                # AR rotation sigmoid output: (B, L, S) from logits
                ar_logits = out_ar[f"{side}_rotation_logits"]  # (B, L_side, S)
                ar_sig = torch.sigmoid(ar_logits)

                tf_logits = out_tf[f"{side}_rotation_logits"]
                tf_sig = torch.sigmoid(tf_logits)

                for t in range(S):
                    gt_t = rot_gt[:, :, t]  # (B, L_side)
                    ar_t = ar_sig[:, :, t]
                    tf_t = tf_sig[:, :, t]

                    for b in range(B):
                        valid = mask[b]  # (L_side,)
                        n_valid = valid.sum().item()
                        if n_valid < 5:
                            continue

                        gt_top5 = set(gt_t[b].topk(5).indices.cpu().tolist())
                        ar_top5 = set(ar_t[b].topk(5).indices.cpu().tolist())
                        tf_top5 = set(tf_t[b].topk(5).indices.cpu().tolist())

                        overlap = len(gt_top5 & ar_top5)
                        overlap_per_min[t] += overlap
                        overlap_count[t] += 1

                        # Per-cell binary accuracy (threshold 0.5)
                        ar_binary = (ar_t[b] > 0.5).float()
                        gt_binary = gt_t[b]
                        match = ((ar_binary == gt_binary) & valid).sum().item()
                        correct_per_min[t] += match
                        total_per_min[t] += valid.sum().item()

                        tf_binary = (tf_t[b] > 0.5).float()
                        tf_match = ((tf_binary == gt_binary) & valid).sum().item()
                        tf_correct_per_min[t] += tf_match
                        tf_total_per_min[t] += valid.sum().item()

    return {
        "ar_cell_acc": correct_per_min / np.maximum(total_per_min, 1),
        "tf_cell_acc": tf_correct_per_min / np.maximum(tf_total_per_min, 1),
        "ar_overlap_5": overlap_per_min / np.maximum(overlap_count, 1),
        "n_games": n_games,
        "S": S,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--precomputed-db", type=str, default="data/features_v5_precomputed.db")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--window", type=str, default="2024-01-01",
                   help="window_start date to evaluate")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    model = CmeV6(cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    from models_cme_v5.scripts.cme_v5_common import PrecomputedDatasetV5, collate_v5
    from torch.utils.data import DataLoader

    ds_test = PrecomputedDatasetV5(args.precomputed_db, args.window, split="test")
    test_loader = DataLoader(ds_test, batch_size=32, shuffle=False,
                             collate_fn=collate_v5, num_workers=0)

    print(f"Window: {args.window}")
    print(f"Test games: {len(ds_test)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    results = compute_ar_drift(model, test_loader, args.device)
    S = results["S"]

    # Print per-minute stats in blocks of 8
    print(f"{'min':>4} | {'AR cell_acc':>10} | {'TF cell_acc':>10} | {'AR top5_overlap':>14}")
    print("-" * 50)
    for t in range(S):
        print(f"{t:4d} | {results['ar_cell_acc'][t]:10.3f} | {results['tf_cell_acc'][t]:10.3f} | {results['ar_overlap_5'][t]:10.1f}/5")

    # Summary by quarter
    print("\n=== By quarter ===")
    for q, (s, e) in enumerate([(0, 12), (12, 24), (24, 36), (36, 48)]):
        ar_acc = results["ar_cell_acc"][s:e].mean()
        tf_acc = results["tf_cell_acc"][s:e].mean()
        ar_ol = results["ar_overlap_5"][s:e].mean()
        print(f"Q{q+1} (min {s:2d}-{e-1:2d}): AR_cell={ar_acc:.3f}  TF_cell={tf_acc:.3f}  AR_overlap={ar_ol:.1f}/5")

    print(f"\nOverall AR cell_acc: {results['ar_cell_acc'].mean():.3f}")
    print(f"Overall TF cell_acc: {results['tf_cell_acc'].mean():.3f}")
    print(f"Overall AR overlap:  {results['ar_overlap_5'].mean():.1f}/5")


if __name__ == "__main__":
    main()
