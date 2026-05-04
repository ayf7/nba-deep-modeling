"""Profile encoder-only model: just roster → win prediction."""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cme_v5_common import PrecomputedDatasetV5, collate_v5, load_precomputed_window_info, load_precomputed_vocab_size
from encoder_only import EncoderOnlyModel, EncoderOnlyConfig, win_hinge, win_bce


def eval_split(model, loader, device, win_loss="hinge"):
    model.eval()
    total_n = 0
    correct = 0
    loss_sum = 0.0
    bce_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            logit = out["win_logit"]
            label = batch["label"]

            if win_loss == "hinge":
                loss = win_hinge(logit, label)
            else:
                loss = win_bce(logit, label)
            bce = F.binary_cross_entropy_with_logits(logit, label)

            bs = label.size(0)
            total_n += bs
            loss_sum += float(loss) * bs
            bce_sum += float(bce) * bs
            pred = (torch.sigmoid(logit) > 0.5).float()
            correct += int((pred == label).sum())

    return {
        "loss": loss_sum / total_n,
        "bce": bce_sum / total_n,
        "acc": correct / total_n,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--n-enc", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--win-loss", default="hinge", choices=["hinge", "bce"])
    p.add_argument("--no-player-emb", action="store_true")
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

    cfg = EncoderOnlyConfig(
        vocab_size=vocab_size, num_teams=team_vocab_size,
        tabular_dim=sample["tabular"].numel(),
        d=args.d, n_heads=args.n_heads, n_enc=args.n_enc,
        dropout=args.dropout,
        player_stats_dim=stats_dim,
        use_player_embeddings=not args.no_player_emb,
    )
    model = EncoderOnlyModel(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,} | emb={not args.no_player_emb} stats_dim={stats_dim}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_v5)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v5)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_v5)

    # Epoch 0
    tr = eval_split(model, train_loader, args.device, args.win_loss)
    va = eval_split(model, val_loader, args.device, args.win_loss)
    te = eval_split(model, test_loader, args.device, args.win_loss)
    print(f"\nEp  0 | train: loss={tr['loss']:.4f} bce={tr['bce']:.4f} acc={tr['acc']:.3f} | val: loss={va['loss']:.4f} bce={va['bce']:.4f} acc={va['acc']:.3f} | test: acc={te['acc']:.3f}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_n = 0

        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch)
            if args.win_loss == "hinge":
                loss = win_hinge(out["win_logit"], batch["label"])
            else:
                loss = win_bce(out["win_logit"], batch["label"])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            train_loss += loss.item() * batch["label"].size(0)
            train_n += batch["label"].size(0)

        dt = time.time() - t0
        tr = eval_split(model, train_loader, args.device, args.win_loss)
        va = eval_split(model, val_loader, args.device, args.win_loss)
        te = eval_split(model, test_loader, args.device, args.win_loss)
        gap = va['acc'] - tr['acc']
        print(f"Ep {epoch:2d} ({dt:.0f}s) | train: loss={tr['loss']:.4f} bce={tr['bce']:.4f} acc={tr['acc']:.3f} | val: loss={va['loss']:.4f} bce={va['bce']:.4f} acc={va['acc']:.3f} | test: acc={te['acc']:.3f} | gap={gap:+.3f}")


if __name__ == "__main__":
    main()
