"""CME-v1.5: cme_v1 encoder + learned bet-selector head.

Wraps a CmeV1 encoder. Adds a 3-way selector head whose action space is
{bet_home, bet_away, no_bet}. Selector input concatenates:
  - cme_v1 win_logit, p_model, |p - mkt_p|
  - market features: mkt_p, h_dec, a_dec
  - rest features: home_rest, away_rest
  - margin_mu (raw)
  - encoder taps: pooled home_emb (d), away_emb (d), ctx (ctx_dim)

Trained jointly with cme_v1's pair_poisson + margin_huber losses, so the
encoder is shaped both for win prediction AND for bet-selectability. The
selector loss is the negated *expected* realized profit on each game's two
possible bets:
  L_sel = -E_{a~pi}[ R(a, y, odds) ]
        = -[ pi_home * R_home + pi_away * R_away ]
A small coverage penalty keeps the policy from collapsing to no_bet.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
CME_V1_SCRIPTS = REPO_ROOT / "models_cme_v1" / "scripts"
sys.path.insert(0, str(CME_V1_SCRIPTS))

from cme_v1_model import CmeV1, CmeV1Config  # noqa: E402


@dataclass
class CmeV15Config:
    cme: CmeV1Config
    sel_hidden: int = 96
    sel_dropout: float = 0.20
    coverage_target: float = 0.10
    coverage_weight: float = 5.0
    sel_loss_weight: float = 0.30
    nobet_bias_init: float = 1.0


def _zero_init_last_linear(seq: nn.Sequential) -> None:
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return


class CmeV15(nn.Module):
    def __init__(self, cfg: CmeV15Config) -> None:
        super().__init__()
        # Force bet-token probe in the encoder so home_emb / away_emb are
        # CLS-style residual stream taps rather than a destructive mean-pool.
        cfg.cme.use_bet_token = True
        self.cfg = cfg
        self.cme = CmeV1(cfg.cme)

        d = cfg.cme.d
        ctx_dim = cfg.cme.tabular_dim + 2 + 2 * cfg.cme.team_emb_dim

        # selector input:
        #   scalars: win_logit, p_model, mkt_p, p-mkt, h_dec, a_dec,
        #            home_rest, away_rest, margin_mu, N_home, N_away  -> 11
        #   taps:    home_emb (d), away_emb (d), ctx (ctx_dim)
        self.scalar_in = 11
        sel_in = self.scalar_in + 2 * d + ctx_dim

        self.selector = nn.Sequential(
            nn.Linear(sel_in, cfg.sel_hidden), nn.GELU(), nn.Dropout(cfg.sel_dropout),
            nn.Linear(cfg.sel_hidden, cfg.sel_hidden), nn.GELU(), nn.Dropout(cfg.sel_dropout),
            nn.Linear(cfg.sel_hidden, 3),  # [bet_home, bet_away, no_bet]
        )
        # Small init on the last layer — NOT zero, since zero-weight kills gradient
        # flow back into the encoder (the whole point of joint training).
        with torch.no_grad():
            nn.init.normal_(self.selector[-1].weight, std=0.02)
            self.selector[-1].bias.zero_()
            self.selector[-1].bias[2] = cfg.nobet_bias_init

    def forward(self, batch: dict) -> dict:
        out = self.cme(batch)

        win_logit = out["win_logit"]
        p_model = torch.sigmoid(win_logit)
        mkt_p = self._market_implied_home(batch)

        scalars = torch.stack([
            win_logit,
            p_model,
            mkt_p,
            (p_model - mkt_p).abs(),
            batch["home_dec_odds"],
            batch["away_dec_odds"],
            batch["home_rest"],
            batch["away_rest"],
            out["margin_mu"],
            out["N_home"] / 491.0,
            out["N_away"] / 491.0,
        ], dim=-1)  # (B, 11)

        feats = torch.cat([
            scalars, out["home_emb"], out["away_emb"], out["ctx"],
        ], dim=-1)
        sel_logits = self.selector(feats)
        sel_probs = F.softmax(sel_logits, dim=-1)

        out["sel_logits"] = sel_logits
        out["sel_probs"] = sel_probs
        out["mkt_p_home"] = mkt_p
        return out

    @staticmethod
    def _market_implied_home(batch: dict) -> torch.Tensor:
        """Vig-removed home implied prob from decimal odds; falls back to 0.5
        when has_odds=0 to avoid NaN gradients (those rows are masked out of
        the selector loss anyway via has_odds)."""
        h = batch["home_dec_odds"].clamp_min(1.001)
        a = batch["away_dec_odds"].clamp_min(1.001)
        h_imp = 1.0 / h
        a_imp = 1.0 / a
        denom = (h_imp + a_imp).clamp_min(1e-6)
        mkt = h_imp / denom
        has = batch["has_odds"]
        return torch.where(has > 0, mkt, torch.full_like(mkt, 0.5))


def selector_loss(
    sel_probs: torch.Tensor,    # (B, 3)
    label: torch.Tensor,        # (B,) 1 = home_won
    h_dec: torch.Tensor,        # (B,)
    a_dec: torch.Tensor,        # (B,)
    has_odds: torch.Tensor,     # (B,) 1 if odds present
    coverage_target: float,
    coverage_weight: float,
) -> tuple[torch.Tensor, dict]:
    """Returns total loss and a diagnostics dict."""
    label = label.to(sel_probs.dtype)
    R_home = torch.where(label > 0.5, h_dec - 1.0, -torch.ones_like(h_dec))
    R_away = torch.where(label < 0.5, a_dec - 1.0, -torch.ones_like(a_dec))

    expected_ev = sel_probs[:, 0] * R_home + sel_probs[:, 1] * R_away
    mask = has_odds > 0
    if mask.sum() == 0:
        zero = sel_probs.new_zeros(())
        return zero, {"sel_ev": 0.0, "coverage": 0.0, "p_nobet_mean": 1.0}

    ev_loss = -expected_ev[mask].mean()
    coverage = (1.0 - sel_probs[mask, 2]).mean()
    cov_penalty = F.relu(torch.tensor(coverage_target, device=sel_probs.device) - coverage)
    total = ev_loss + coverage_weight * cov_penalty

    return total, {
        "sel_ev": float((-ev_loss).detach().item()),
        "coverage": float(coverage.detach().item()),
        "p_nobet_mean": float(sel_probs[mask, 2].mean().detach().item()),
        "cov_penalty": float(cov_penalty.detach().item()),
    }


def realized_profit_from_actions(
    actions: torch.Tensor,      # (B,) 0=home, 1=away, 2=no_bet
    label: torch.Tensor,        # (B,) 1 = home_won
    h_dec: torch.Tensor,
    a_dec: torch.Tensor,
) -> torch.Tensor:
    profit = torch.zeros_like(h_dec)
    home_bet = actions == 0
    away_bet = actions == 1
    if home_bet.any():
        won = (label[home_bet] == 1)
        profit[home_bet] = torch.where(won, h_dec[home_bet] - 1.0, -torch.ones_like(h_dec[home_bet]))
    if away_bet.any():
        won = (label[away_bet] == 0)
        profit[away_bet] = torch.where(won, a_dec[away_bet] - 1.0, -torch.ones_like(a_dec[away_bet]))
    return profit


def _smoke() -> None:
    torch.manual_seed(0)
    cme_cfg = CmeV1Config(
        vocab_size=50, num_teams=10, d=16, tabular_dim=4, team_emb_dim=8,
        player_stat_dim=6, eff_hidden=16,
    )
    cfg = CmeV15Config(cme=cme_cfg)
    model = CmeV15(cfg)

    B, Lo, Ld = 4, 12, 14
    batch = {
        "home_idx": torch.randint(1, cme_cfg.vocab_size + 1, (B, Lo)),
        "away_idx": torch.randint(1, cme_cfg.vocab_size + 1, (B, Ld)),
        "home_prob": torch.rand(B, Lo),
        "away_prob": torch.rand(B, Ld),
        "home_mask": torch.ones(B, Lo, dtype=torch.bool),
        "away_mask": torch.ones(B, Ld, dtype=torch.bool),
        "home_stats": torch.randn(B, Lo, cme_cfg.player_stat_dim),
        "away_stats": torch.randn(B, Ld, cme_cfg.player_stat_dim),
        "home_team_idx": torch.randint(1, cme_cfg.num_teams + 1, (B,)),
        "away_team_idx": torch.randint(1, cme_cfg.num_teams + 1, (B,)),
        "home_rest": torch.randn(B),
        "away_rest": torch.randn(B),
        "tabular": torch.randn(B, cme_cfg.tabular_dim),
        "home_dec_odds": torch.tensor([1.85, 2.10, 1.50, 3.20]),
        "away_dec_odds": torch.tensor([2.05, 1.80, 2.70, 1.40]),
        "has_odds": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "label": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        "margin": torch.tensor([5.0, -3.0, 12.0, 8.0]),
    }

    out = model(batch)
    print("=== cme_v1.5 smoke ===")
    print(f"sel_logits {tuple(out['sel_logits'].shape)} sel_probs sum-per-row "
          f"{out['sel_probs'].sum(dim=-1).tolist()}")
    print(f"sel_probs[0] = {out['sel_probs'][0].tolist()}")
    print(f"mkt_p_home   = {out['mkt_p_home'].tolist()}")

    loss, diag = selector_loss(
        out["sel_probs"], batch["label"], batch["home_dec_odds"], batch["away_dec_odds"],
        batch["has_odds"], cfg.coverage_target, cfg.coverage_weight,
    )
    print(f"selector loss: {loss.item():+.4f}  diag={diag}")

    loss.backward()
    n_grad = sum(p.grad is not None and p.grad.abs().sum().item() > 0
                 for p in model.parameters())
    n_total = sum(1 for _ in model.parameters())
    print(f"backward ok: {n_grad}/{n_total} parameters got nonzero gradients")


if __name__ == "__main__":
    _smoke()
