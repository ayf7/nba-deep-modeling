"""Evaluate saved checkpoints on test with teacher forcing."""
import argparse, sys, pathlib, glob, torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from model import CmeV6
from cme_v5_common import PrecomputedDatasetV5, collate_v5
from torch.utils.data import DataLoader


def eval_test(model, loader, device):
    model.eval()
    total_bce, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch, teacher_force=True)
            logit = out["win_logit"]
            label = batch["label"]
            bce = F.binary_cross_entropy_with_logits(logit, label, reduction="sum")
            total_bce += bce.item()
            correct += ((logit > 0).float() == label).sum().item()
            n += label.size(0)
    return total_bce / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", type=str, required=True)
    p.add_argument("--precomputed-db", type=str, default="data/features_v5_precomputed.db")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    ckpts = sorted(glob.glob(f"{args.artifact_dir}/ckpt_*.pt"))
    print(f"Found {len(ckpts)} checkpoints\n")
    print(f"{'window':>12} | {'n':>4} | {'TF bce':>8} | {'TF acc':>8} | {'ep':>3}")
    print("-" * 50)

    all_bce, all_acc, all_n = [], [], []
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        cfg = ckpt["cfg"]
        window = ckpt["window_start"]

        model = CmeV6(cfg).to(args.device)
        model.load_state_dict(ckpt["state_dict"])

        ds = PrecomputedDatasetV5(args.precomputed_db, window, split="test")
        if len(ds) < 5:
            print(f"{window:>12} | {len(ds):>4} | {'skip':>8} | {'skip':>8} | {ckpt['best_epoch']:>3}")
            continue
        loader = DataLoader(ds, batch_size=64, shuffle=False,
                            collate_fn=collate_v5, num_workers=0)

        bce, acc = eval_test(model, loader, args.device)
        print(f"{window:>12} | {len(ds):>4} | {bce:8.4f} | {acc*100:7.1f}% | {ckpt['best_epoch']:>3}")
        all_bce.append(bce)
        all_acc.append(acc)
        all_n.append(len(ds))

    print("-" * 50)
    avg_bce = sum(all_bce) / len(all_bce)
    avg_acc = sum(all_acc) / len(all_acc)
    print(f"{'avg':>12} | {sum(all_n):>4} | {avg_bce:8.4f} | {avg_acc*100:7.1f}%")


if __name__ == "__main__":
    main()
