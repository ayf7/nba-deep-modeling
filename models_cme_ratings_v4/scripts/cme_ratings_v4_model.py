"""CME-Ratings-v4: ratings-first anchor + pregame roster-shock correction.

Ratings-v2 established that the calibrated dynamic team-strength prior is the
main outcome predictor.  Ratings-v4 keeps that exact hierarchy and inserts one
new signal before the tiny CME residual: a bounded, strictly pregame roster-
shock adjustment.  The adjustment is driven by recent rotation mass listed as
out/questionable/etc. in the latest pre-tip availability artifact.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RATINGS_V2_SCRIPTS = PROJECT_ROOT / "models_cme_ratings_v2" / "scripts"
sys.path.insert(0, str(RATINGS_V2_SCRIPTS))

from cme_ratings_v2_model import (  # noqa: E402
    CmeRatingsV2,
    CmeRatingsV2Config,
    total_loss as total_loss_v2,
)

THIS_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_SCRIPTS))
from cme_ratings_v4_common import K_ROSTER_SHOCK_FEATURES  # noqa: E402


def _zero_last_linear(seq: nn.Sequential) -> None:
    for module in reversed(list(seq.modules())):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return


def _inverse_tanh_init(value: float, max_abs: float) -> float:
    if max_abs <= 0.0:
        return 0.0
    ratio = max(min(float(value) / float(max_abs), 0.999), -0.999)
    return float(math.atanh(ratio))


@dataclass
class CmeRatingsV4Config(CmeRatingsV2Config):
    roster_shock_feature_dim: int = K_ROSTER_SHOCK_FEATURES
    roster_shock_hidden: int = 32
    roster_shock_dropout: float = 0.05
    max_roster_shock_signal_weight: float = 1.25
    init_roster_shock_signal_weight: float = 0.00
    max_roster_shock_feature_residual: float = 0.45


class CmeRatingsV4(CmeRatingsV2):
    def __init__(self, cfg: CmeRatingsV4Config) -> None:
        super().__init__(cfg)
        self.roster_cfg = cfg
        self.raw_roster_shock_signal_weight = nn.Parameter(
            torch.tensor(
                _inverse_tanh_init(cfg.init_roster_shock_signal_weight, cfg.max_roster_shock_signal_weight),
                dtype=torch.float32,
            )
        )
        # shock features + raw signed signal + rating anchor + abs anchor.
        in_dim = int(cfg.roster_shock_feature_dim) + 3
        self.roster_shock_head = nn.Sequential(
            nn.Linear(in_dim, cfg.roster_shock_hidden),
            nn.GELU(),
            nn.Dropout(cfg.roster_shock_dropout),
            nn.Linear(cfg.roster_shock_hidden, cfg.roster_shock_hidden),
            nn.GELU(),
            nn.Dropout(cfg.roster_shock_dropout),
            nn.Linear(cfg.roster_shock_hidden, 1),
        )
        _zero_last_linear(self.roster_shock_head)

    def forward(self, batch: dict) -> dict:
        # Run the full ratings-v2 path first so we keep the proven world-model
        # and calibrated rating anchor machinery, then recompute the last part
        # around a roster-shock-adjusted ratings anchor.
        out = super().forward(batch)
        v42_logit = out["win_logit_v42"]
        rating_logit_calibrated = out["rating_logit_calibrated"]
        B = v42_logit.shape[0]
        dtype = v42_logit.dtype
        device = v42_logit.device

        roster_features = batch.get("roster_shock_features")
        if roster_features is None:
            roster_features = torch.zeros(B, int(self.roster_cfg.roster_shock_feature_dim), device=device, dtype=dtype)
        elif roster_features.size(-1) == 0 and self.roster_cfg.roster_shock_feature_dim > 0:
            roster_features = torch.zeros(B, int(self.roster_cfg.roster_shock_feature_dim), device=device, dtype=dtype)
        else:
            roster_features = roster_features.to(dtype)
        roster_signal = batch.get("roster_shock_advantage_signal")
        if roster_signal is None:
            roster_signal = torch.zeros_like(v42_logit)
        else:
            roster_signal = roster_signal.to(dtype)
        has_roster_shock = batch.get("has_roster_shock")
        if has_roster_shock is None:
            has_roster_shock = torch.zeros_like(v42_logit)
        else:
            has_roster_shock = has_roster_shock.to(dtype)
        has_rating = out["has_rating_prior"].to(dtype)
        active_shock = has_roster_shock * has_rating

        if self.roster_cfg.max_roster_shock_signal_weight > 0.0:
            roster_shock_signal_weight = (
                float(self.roster_cfg.max_roster_shock_signal_weight)
                * torch.tanh(self.raw_roster_shock_signal_weight)
            )
        else:
            roster_shock_signal_weight = torch.zeros((), device=device, dtype=dtype)
        roster_shock_linear_adjustment = active_shock * roster_shock_signal_weight * roster_signal

        shock_head_features = torch.cat([
            roster_features,
            roster_signal.unsqueeze(-1),
            rating_logit_calibrated.unsqueeze(-1),
            rating_logit_calibrated.abs().unsqueeze(-1),
        ], dim=-1)
        raw_roster_shock_feature_adjustment = self.roster_shock_head(shock_head_features).squeeze(-1)
        if self.roster_cfg.max_roster_shock_feature_residual > 0.0:
            roster_shock_feature_adjustment = (
                float(self.roster_cfg.max_roster_shock_feature_residual)
                * torch.tanh(raw_roster_shock_feature_adjustment)
                * active_shock
            )
        else:
            roster_shock_feature_adjustment = torch.zeros_like(v42_logit)
        roster_shock_logit_adjustment = roster_shock_linear_adjustment + roster_shock_feature_adjustment
        rating_logit_roster_adjusted = rating_logit_calibrated + roster_shock_logit_adjustment
        rating_prob_roster_adjusted = torch.sigmoid(rating_logit_roster_adjusted)

        # Recompute the small CME residual around the roster-adjusted rating anchor.
        cme_gap = v42_logit - rating_logit_roster_adjusted
        abs_cme_gap = cme_gap.abs()
        abs_rating_logit = rating_logit_roster_adjusted.abs()
        gate_logit = (
            float(self.rating_cfg.disagreement_gate_center)
            - float(self.rating_cfg.disagreement_gate_rating_confidence_w) * abs_rating_logit
            - float(self.rating_cfg.disagreement_gate_gap_w) * abs_cme_gap
        )
        disagreement_gate = torch.sigmoid(gate_logit) * has_rating
        cme_gap_weight = out["cme_gap_weight"]
        cme_gap_linear_residual = disagreement_gate * cme_gap_weight * cme_gap
        head_features = torch.cat([
            batch.get("rating_features", torch.zeros(B, int(self.rating_cfg.rating_feature_dim), device=device, dtype=dtype)).to(dtype),
            rating_logit_roster_adjusted.unsqueeze(-1),
            v42_logit.unsqueeze(-1),
            cme_gap.unsqueeze(-1),
            abs_cme_gap.unsqueeze(-1),
            disagreement_gate.unsqueeze(-1),
        ], dim=-1)
        raw_cme_feature_residual = self.cme_residual_head(head_features).squeeze(-1)
        if self.rating_cfg.max_cme_feature_residual > 0.0:
            cme_feature_residual = (
                float(self.rating_cfg.max_cme_feature_residual)
                * torch.tanh(raw_cme_feature_residual)
                * disagreement_gate
            )
        else:
            cme_feature_residual = torch.zeros_like(v42_logit)
        ratings_anchored_logit = rating_logit_roster_adjusted + cme_gap_linear_residual + cme_feature_residual
        final_logit = has_rating * ratings_anchored_logit + (1.0 - has_rating) * v42_logit

        out.update({
            "win_logit": final_logit,
            "rating_logit_roster_adjusted": rating_logit_roster_adjusted,
            "rating_prob_roster_adjusted": rating_prob_roster_adjusted,
            "roster_shock_features": roster_features,
            "roster_shock_advantage_signal": roster_signal,
            "has_roster_shock": has_roster_shock,
            "roster_shock_signal_weight": roster_shock_signal_weight,
            "raw_roster_shock_signal_weight": self.raw_roster_shock_signal_weight,
            "roster_shock_linear_adjustment": roster_shock_linear_adjustment,
            "raw_roster_shock_feature_adjustment": raw_roster_shock_feature_adjustment,
            "roster_shock_feature_adjustment": roster_shock_feature_adjustment,
            "roster_shock_logit_adjustment": roster_shock_logit_adjustment,
            "cme_gap": cme_gap,
            "abs_cme_gap": abs_cme_gap,
            "disagreement_gate": disagreement_gate,
            "cme_gap_linear_residual": cme_gap_linear_residual,
            "raw_cme_feature_residual": raw_cme_feature_residual,
            "cme_feature_residual": cme_feature_residual,
            "ratings_anchored_logit": ratings_anchored_logit,
        })
        return out


def roster_shock_feature_l2_loss(out: dict) -> torch.Tensor:
    return out["roster_shock_feature_adjustment"].pow(2).mean()


def roster_shock_signal_weight_l2_loss(out: dict) -> torch.Tensor:
    return out["roster_shock_signal_weight"].pow(2)


def total_loss(
    out: dict,
    batch: dict,
    *,
    roster_shock_feature_reg_w: float = 0.02,
    roster_shock_signal_weight_reg_w: float = 0.002,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    base_total, diag = total_loss_v2(out, batch, **kwargs)
    L_shock_feature_reg = roster_shock_feature_l2_loss(out)
    L_shock_signal_weight_reg = roster_shock_signal_weight_l2_loss(out)
    total = (
        base_total
        + roster_shock_feature_reg_w * L_shock_feature_reg
        + roster_shock_signal_weight_reg_w * L_shock_signal_weight_reg
    )
    diag = dict(diag)
    diag.update({
        "L_roster_shock_feature_reg": L_shock_feature_reg.detach(),
        "L_roster_shock_signal_weight_reg": L_shock_signal_weight_reg.detach(),
    })
    return total, diag
