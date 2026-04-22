"""Log-space Sinkhorn matrix scaling for masked rectangular pair matrices.

Given pair scores `log_K[i, j]` plus row marginals `row_mass[i]` and column
marginals `col_mass[j]`, return a non-negative matrix `P[i, j]` such that
    sum_j P[i, j] ≈ row_mass[i]   for unmasked rows
    sum_i P[i, j] ≈ col_mass[j]   for unmasked cols
"""
from __future__ import annotations

import torch


NEG_INF_FILL = -1e9


def sinkhorn_transport(
    log_scores: torch.Tensor,    # (B, Lo, Ld)
    row_mass: torch.Tensor,      # (B, Lo)
    col_mass: torch.Tensor,      # (B, Ld)
    mask: torch.Tensor,          # (B, Lo, Ld) bool
    n_iter: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    log_K = log_scores.masked_fill(~mask, NEG_INF_FILL)
    log_r = torch.log(row_mass.clamp_min(eps))
    log_c = torch.log(col_mass.clamp_min(eps))

    row_mask = mask.any(dim=2)
    col_mask = mask.any(dim=1)

    u = torch.zeros_like(log_r)
    v = torch.zeros_like(log_c)
    for _ in range(n_iter):
        u = log_r - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)
        u = u.masked_fill(~row_mask, NEG_INF_FILL)
        v = log_c - torch.logsumexp(log_K + u.unsqueeze(2), dim=1)
        v = v.masked_fill(~col_mask, NEG_INF_FILL)

    log_P = log_K + u.unsqueeze(2) + v.unsqueeze(1)
    P = torch.exp(log_P) * mask.to(log_P.dtype)
    return P


def _self_test() -> None:
    torch.manual_seed(0)
    B, Lo, Ld = 4, 12, 14
    log_scores = torch.randn(B, Lo, Ld) * 0.5

    row_mass = torch.rand(B, Lo) * 5.0 + 1.0
    col_mass = row_mass.sum(dim=1, keepdim=True) * (
        torch.rand(B, Ld) / torch.rand(B, Ld).sum(dim=1, keepdim=True)
    )

    mask_o = torch.rand(B, Lo) > 0.1
    mask_d = torch.rand(B, Ld) > 0.1
    mask = mask_o.unsqueeze(2) & mask_d.unsqueeze(1)

    row_mass = row_mass * mask_o.float()
    col_mass = col_mass * mask_d.float()
    total_r = row_mass.sum(dim=1)
    total_c = col_mass.sum(dim=1)
    col_mass = col_mass * (total_r / total_c.clamp_min(1e-8)).unsqueeze(1)

    P = sinkhorn_transport(log_scores, row_mass, col_mass, mask, n_iter=20)

    row_err = (P.sum(dim=2) - row_mass).abs().max().item()
    col_err = (P.sum(dim=1) - col_mass).abs().max().item()
    masked_max = P[~mask].abs().max().item() if (~mask).any() else 0.0
    nonneg = (P >= -1e-9).all().item()

    print(f"row marginal max err = {row_err:.2e}")
    print(f"col marginal max err = {col_err:.2e}")
    print(f"masked entries max abs = {masked_max:.2e}")
    print(f"all non-negative: {nonneg}")
    assert row_err < 1e-3, "row marginal not satisfied"
    assert col_err < 1e-3, "col marginal not satisfied"
    assert masked_max == 0.0, "leakage into masked entries"
    assert nonneg, "negative values in P"

    Lo_unmasked = mask_o[0].sum().item()
    Ld_unmasked = mask_d[0].sum().item()
    rank1 = row_mass[0:1].unsqueeze(2) * col_mass[0:1].unsqueeze(1) / total_r[0:1].view(1, 1, 1).clamp_min(1e-8)
    rank1_fit = (rank1 - P[0:1]).abs().mean().item()
    print(
        f"\nbatch 0: Lo={Lo_unmasked} Ld={Ld_unmasked} rank-1 vs Sinkhorn mean abs diff = {rank1_fit:.4f}"
    )
    print("(non-zero diff = Sinkhorn used the score signal, not just outer product)")


if __name__ == "__main__":
    _self_test()
