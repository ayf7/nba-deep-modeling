"""CME-Ratings-v4.1: roster-adjusted ratings promoted to the final path.

Ratings-v4 showed that the strongest non-market predictor was not the final
ratings+shock+CME blend, but the *roster-adjusted ratings anchor itself*.
Ratings-v4.1 therefore removes CME residuals from the winner-probability path,
keeps them as diagnostics, and optionally learns a tiny post-shock affine
calibration:

    post_shock_logit = slope * roster_adjusted_rating_logit + bias

The final output is the post-shock calibrated ratings anchor whenever a ratings
prior is available; otherwise it falls back to the legacy v4.2/CME logit.
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
from cme_ratings_v4_1_common import K_ROSTER_SHOCK_FEATURES  # noqa: E402


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


def _safe_log_positive(value: float) -> float:
    return float(math.log(max(float(value), 1e-4)))


@dataclass
class CmeRatingsV41Config(CmeRatingsV2Config):
    roster_shock_feature_dim: int = K_ROSTER_SHOCK_FEATURES
    roster_shock_hidden: int = 32
    roster_shock_dropout: float = 0.05
    max_roster_shock_signal_weight: float = 1.25
    init_roster_shock_signal_weight: float = 0.00
    max_roster_shock_feature_residual: float = 0.45

    # v4.1 promotion: the roster-adjusted ratings anchor is final, with an
    # optional tiny post-shock affine calibration layer.
    use_post_shock_calibration: bool = True
    init_post_shock_logit_bias: float = 0.0
    init_post_shock_logit_slope: float = 1.0


class CmeRatingsV41(CmeRatingsV2):
    def __init__(self, cfg: CmeRatingsV41Config) -> None:
        super().__init__(cfg)
        self.roster_cfg = cfg
        self.raw_roster_shock_signal_weight = nn.Parameter(
            torch.tensor(
                _inverse_tanh_init(
                    cfg.init_roster_shock_signal_weight,
                    cfg.max_roster_shock_signal_weight,
                ),
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

        self.post_shock_logit_bias = nn.Parameter(
            torch.tensor(float(cfg.init_post_shock_logit_bias), dtype=torch.float32),
            requires_grad=bool(cfg.use_post_shock_calibration),
        )
        self.log_post_shock_logit_slope = nn.Parameter(
            torch.tensor(_safe_log_positive(cfg.init_post_shock_logit_slope), dtype=torch.float32),
            requires_grad=bool(cfg.use_post_shock_calibration),
        )

    def forward(self, batch: dict) -> dict:
        # Run the ratings-v2 path first: it computes the calibrated ratings
        # anchor, the legacy CME logit, and the v2 residual diagnostics.
        out = super().forward(batch)
        v42_logit = out["win_logit_v42"]
        rating_logit_calibrated = out["rating_logit_calibrated"]
        B = v42_logit.shape[0]
        dtype = v42_logit.dtype
        device = v42_logit.device

        roster_features = batch.get("roster_shock_features")
        if roster_features is None:
            roster_features = torch.zeros(
                B,
                int(self.roster_cfg.roster_shock_feature_dim),
                device=device,
                dtype=dtype,
            )
        elif roster_features.size(-1) == 0 and self.roster_cfg.roster_shock_feature_dim > 0:
            roster_features = torch.zeros(
                B,
                int(self.roster_cfg.roster_shock_feature_dim),
                device=device,
                dtype=dtype,
            )
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

        # Keep the CME disagreement diagnostics from v4, but *do not* use them
        # in the winner-probability path.  This preserves comparability while
        # promoting the empirically winning roster-adjusted ratings stage.
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
            batch.get(
                "rating_features",
                torch.zeros(B, int(self.rating_cfg.rating_feature_dim), device=device, dtype=dtype),
            ).to(dtype),
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
        cme_diagnostic_logit = rating_logit_roster_adjusted + cme_gap_linear_residual + cme_feature_residual

        # v4.1 final path: optional post-shock affine calibration.  If disabled,
        # the slope/bias are frozen at their configured values, defaulting to
        # identity (1.0, 0.0).
        post_shock_logit_slope = torch.exp(self.log_post_shock_logit_slope)
        rating_logit_post_shock_calibrated = (
            post_shock_logit_slope * rating_logit_roster_adjusted
            + self.post_shock_logit_bias
        )
        rating_prob_post_shock_calibrated = torch.sigmoid(rating_logit_post_shock_calibrated)
        final_logit = (
            has_rating * rating_logit_post_shock_calibrated
            + (1.0 - has_rating) * v42_logit
        )

        out.update({
            "win_logit": final_logit,
            "rating_logit_roster_adjusted": rating_logit_roster_adjusted,
            "rating_prob_roster_adjusted": rating_prob_roster_adjusted,
            "rating_logit_post_shock_calibrated": rating_logit_post_shock_calibrated,
            "rating_prob_post_shock_calibrated": rating_prob_post_shock_calibrated,
            "post_shock_logit_bias": self.post_shock_logit_bias,
            "log_post_shock_logit_slope": self.log_post_shock_logit_slope,
            "post_shock_logit_slope": post_shock_logit_slope,
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
            "cme_diagnostic_logit": cme_diagnostic_logit,
        })
        return out


def roster_shock_feature_l2_loss(out: dict) -> torch.Tensor:
    return out["roster_shock_feature_adjustment"].pow(2).mean()


def roster_shock_signal_weight_l2_loss(out: dict) -> torch.Tensor:
    return out["roster_shock_signal_weight"].pow(2)


def post_shock_bias_l2_loss(out: dict) -> torch.Tensor:
    return out["post_shock_logit_bias"].pow(2)


def post_shock_log_slope_l2_loss(out: dict) -> torch.Tensor:
    return out["log_post_shock_logit_slope"].pow(2)


def total_loss(
    out: dict,
    batch: dict,
    *,
    roster_shock_feature_reg_w: float = 0.02,
    roster_shock_signal_weight_reg_w: float = 0.002,
    post_shock_bias_reg_w: float = 0.0002,
    post_shock_slope_reg_w: float = 0.001,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    base_total, diag = total_loss_v2(out, batch, **kwargs)
    L_shock_feature_reg = roster_shock_feature_l2_loss(out)
    L_shock_signal_weight_reg = roster_shock_signal_weight_l2_loss(out)
    L_post_shock_bias_reg = post_shock_bias_l2_loss(out)
    L_post_shock_slope_reg = post_shock_log_slope_l2_loss(out)
    total = (
        base_total
        + roster_shock_feature_reg_w * L_shock_feature_reg
        + roster_shock_signal_weight_reg_w * L_shock_signal_weight_reg
        + post_shock_bias_reg_w * L_post_shock_bias_reg
        + post_shock_slope_reg_w * L_post_shock_slope_reg
    )
    diag = dict(diag)
    diag.update({
        "L_roster_shock_feature_reg": L_shock_feature_reg.detach(),
        "L_roster_shock_signal_weight_reg": L_shock_signal_weight_reg.detach(),
        "L_post_shock_bias_reg": L_post_shock_bias_reg.detach(),
        "L_post_shock_slope_reg": L_post_shock_slope_reg.detach(),
    })
    return total, diag
