"""Self-consistency eval: N stochastic AR rollouts per game, average win logits."""
import argparse, sys, pathlib, torch, math
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from model import CmeV6
from cme_v5_common import PrecomputedDatasetV5, collate_v5
from torch.utils.data import DataLoader


def stochastic_rollout(model, P, P_mask, batch, Lh, tau=1.0):
    """Like _rollout_lineup but samples from rotation scores instead of topk."""
    B, L, _ = P.shape
    S = model.cfg.n_segments
    k = model.cfg.score_k
    dev = P.device
    home_mask = batch["home_mask"]
    away_mask = batch["away_mask"]

    rot_in = torch.zeros(B, S, L, device=dev, dtype=P.dtype)
    idx_h = torch.zeros(B, S, 5, dtype=torch.long, device=dev)
    idx_a = torch.zeros(B, S, 5, dtype=torch.long, device=dev)

    rot_h = batch["home_rotation_target"]
    rot_a = batch["away_rotation_target"]
    start_h = rot_h[:, :, 0]
    start_a = rot_a[:, :, 0]
    rot_in[:, 0, :Lh] = start_h.to(rot_in.dtype)
    rot_in[:, 0, Lh:] = start_a.to(rot_in.dtype)
    idx_h[:, 0, :] = start_h.topk(5, dim=-1).indices
    idx_a[:, 0, :] = start_a.topk(5, dim=-1).indices

    Kr = model.W_k_r(P)

    for t in range(S):
        slot_h = idx_h if model.cfg.slot_aware_input else None
        slot_a = idx_a if model.cfg.slot_aware_input else None
        D = model._decode(P, P_mask, rot_in, Lh=Lh,
                          lineup_idx_home=slot_h, lineup_idx_away=slot_a)
        Qr_t = model.W_q_r(D[:, t:t + 1, :])
        logits_t = torch.bmm(Qr_t, Kr.transpose(1, 2)).squeeze(1) / (k ** 0.5)

        # Sample top-5 per side using Gumbel-topk
        scores_h = logits_t[:, :Lh] / tau
        scores_a = logits_t[:, Lh:] / tau
        scores_h = scores_h.masked_fill(~home_mask, float("-inf"))
        scores_a = scores_a.masked_fill(~away_mask, float("-inf"))

        # Gumbel noise for sampling
        gumbel_h = -torch.log(-torch.log(torch.rand_like(scores_h).clamp(1e-8, 1 - 1e-8)))
        gumbel_a = -torch.log(-torch.log(torch.rand_like(scores_a).clamp(1e-8, 1 - 1e-8)))
        noisy_h = scores_h + gumbel_h
        noisy_a = scores_a + gumbel_a
        noisy_h = noisy_h.masked_fill(~home_mask, float("-inf"))
        noisy_a = noisy_a.masked_fill(~away_mask, float("-inf"))

        top_h = noisy_h.topk(5, dim=-1).indices
        top_a = noisy_a.topk(5, dim=-1).indices

        if t + 1 < S:
            one_hot = torch.zeros(B, L, device=dev, dtype=rot_in.dtype)
            one_hot.scatter_(1, top_h, 1.0)
            one_hot.scatter_(1, Lh + top_a, 1.0)
            rot_in[:, t + 1, :] = one_hot
            idx_h[:, t + 1, :] = top_h
            idx_a[:, t + 1, :] = top_a

    return rot_in, idx_h, idx_a


def forward_with_rollout(model, batch, rot_in):
    """Run model forward using a precomputed rotation input."""
    home_mask = batch["home_mask"]
    away_mask = batch["away_mask"]
    Lh = home_mask.size(1)
    P, P_mask, home_team_out, away_team_out = model._encode(batch)
    B, L, d = P.shape
    S = model.cfg.n_segments
    k = model.cfg.score_k

    D = model._decode(P, P_mask, rot_in, Lh=Lh)

    Qr = model.W_q_r(D)
    Kr = model.W_k_r(P)
    rot_logits = torch.bmm(Qr, Kr.transpose(1, 2)) / (k ** 0.5)
    mask_exp = P_mask.unsqueeze(1).to(rot_logits.dtype)
    rotation = torch.sigmoid(rot_logits) * mask_exp

    rate_pts = None
    for ch in model.cfg.emit_channels:
        Ql = model.W_q_lam[ch](D)
        Kl = model.W_k_lam[ch](P)
        rate_logits_ch = torch.bmm(Ql, Kl.transpose(1, 2)) / (k ** 0.5)
        if model.cfg.rate_clamp > 0.0:
            rate_logits_ch = rate_logits_ch.clamp(-3.0, model.cfg.rate_clamp)
        r = F.softplus(rate_logits_ch) * mask_exp
        if ch == "pts":
            rate_pts = r

    minute_pts = rotation * rate_pts
    home_pts = minute_pts[:, :, :Lh].sum(dim=(1, 2))
    away_pts = minute_pts[:, :, Lh:].sum(dim=(1, 2))

    gamma = torch.exp(model.log_gamma)
    pts_logit = (home_pts - away_pts) / gamma

    if model.cfg.direct_win == "blend":
        direct_logit = model.win_head(
            torch.cat([home_team_out, away_team_out], dim=-1)
        ).squeeze(-1)
        alpha = torch.sigmoid(model.log_alpha)
        win_logit = alpha * pts_logit + (1 - alpha) * direct_logit
    elif model.cfg.direct_win == "only":
        win_logit = model.win_head(
            torch.cat([home_team_out, away_team_out], dim=-1)
        ).squeeze(-1)
    else:
        win_logit = pts_logit

    return win_logit


def eval_self_consistency(model, loader, device, n_samples, tau):
    model.eval()
    total_bce_sc, total_bce_det, correct_sc, correct_det, n = 0., 0., 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            label = batch["label"]
            B = label.size(0)
            Lh = batch["home_mask"].size(1)

            P, P_mask, _, _ = model._encode(batch)

            # Deterministic AR (existing behavior)
            out_det = model(batch, teacher_force=False, autoregressive=True)
            det_logit = out_det["win_logit"]

            # N stochastic rollouts
            logit_sum = torch.zeros(B, device=device)
            for _ in range(n_samples):
                rot_in, _, _ = stochastic_rollout(model, P, P_mask, batch, Lh, tau=tau)
                logit = forward_with_rollout(model, batch, rot_in)
                logit_sum += logit
            sc_logit = logit_sum / n_samples

            # Metrics
            total_bce_sc += F.binary_cross_entropy_with_logits(sc_logit, label, reduction="sum").item()
            total_bce_det += F.binary_cross_entropy_with_logits(det_logit, label, reduction="sum").item()
            correct_sc += ((sc_logit > 0).float() == label).sum().item()
            correct_det += ((det_logit > 0).float() == label).sum().item()
            n += B

    return {
        "sc_bce": total_bce_sc / n, "sc_acc": correct_sc / n,
        "det_bce": total_bce_det / n, "det_acc": correct_det / n,
        "n": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--precomputed-db", type=str, default="data/features_v5_precomputed.db")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--window", type=str, default="2024-01-01")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--tau", type=float, default=1.0)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    model = CmeV6(cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"])

    ds = PrecomputedDatasetV5(args.precomputed_db, args.window, split="test")
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_v5, num_workers=0)

    print(f"Window: {args.window} | n_test={len(ds)} | n_samples={args.n_samples} | tau={args.tau}")

    results = eval_self_consistency(model, loader, args.device, args.n_samples, args.tau)
    print(f"\nDeterministic AR:  bce={results['det_bce']:.4f}  acc={results['det_acc']*100:.1f}%")
    print(f"Self-consistency:  bce={results['sc_bce']:.4f}  acc={results['sc_acc']*100:.1f}%")


if __name__ == "__main__":
    main()
