"""CME-Ratings-v2: ratings-first outcome head with CME residual correction.

Ratings-v1 demonstrated that a leakage-safe dynamic opponent-adjusted team
strength prior is extremely informative.  In the backtest analyzed after v1,
the raw rating prior outperformed the hybrid model itself.  v2 therefore
inverts the winner-prediction architecture:

    raw_rating_logit = rating_margin_prior / rating_margin_scale
    anchor_logit     = rating_slope * raw_rating_logit + rating_bias
    final_logit      = anchor_logit
                     + disagreement_gate * (small CME-gap correction)
                     + disagreement_gate * bounded feature residual

The full player-aware CME-v4.2 world model is still trained and still produces
its margin / box-score / matchup outputs.  The difference is that, for the
binary game winner, the dynamic rating prior is the anchor, and CME is only a
regularized residual specialist.  When a rating prior is absent, the model
falls back to v4.2's winner logit.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V42_SCRIPTS = PROJECT_ROOT / "models_cme_v4_2" / "scripts"
sys.path.insert(0, str(V42_SCRIPTS))

from cme_v4_2_model import (  # noqa: E402
    CmeV42,
    CmeV42Config,
    total_loss as total_loss_v42,
)

THIS_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_SCRIPTS))
from cme_ratings_v2_common import K_RATING_FEATURES  # noqa: E402


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
class CmeRatingsV2Config(CmeV42Config):
    rating_feature_dim: int = K_RATING_FEATURES
    cme_residual_hidden: int = 32
    cme_residual_dropout: float = 0.05
    rating_margin_scale: float = 12.0
    init_rating_logit_bias: float = 0.04
    init_rating_logit_slope: float = 1.50
    max_cme_gap_weight: float = 0.35
    init_cme_gap_weight: float = 0.00
    max_cme_feature_residual: float = 0.15
    disagreement_gate_center: float = 0.80
    disagreement_gate_rating_confidence_w: float = 1.00
    disagreement_gate_gap_w: float = 0.75


class CmeRatingsV2(CmeV42):
    def __init__(self, cfg: CmeRatingsV2Config) -> None:
        super().__init__(cfg)
        self.rating_cfg = cfg
        if cfg.init_rating_logit_slope <= 0.0:
            raise ValueError("init_rating_logit_slope must be > 0")
        self.rating_logit_bias = nn.Parameter(
            torch.tensor(float(cfg.init_rating_logit_bias), dtype=torch.float32)
        )
        self.log_rating_logit_slope = nn.Parameter(
            torch.tensor(math.log(float(cfg.init_rating_logit_slope)), dtype=torch.float32)
        )
        self.raw_cme_gap_weight = nn.Parameter(
            torch.tensor(
                _inverse_tanh_init(cfg.init_cme_gap_weight, cfg.max_cme_gap_weight),
                dtype=torch.float32,
            )
        )
        # Rating features + anchor logit + CME logit + signed gap + abs gap + gate.
        in_dim = int(cfg.rating_feature_dim) + 5
        self.cme_residual_head = nn.Sequential(
            nn.Linear(in_dim, cfg.cme_residual_hidden),
            nn.GELU(),
            nn.Dropout(cfg.cme_residual_dropout),
            nn.Linear(cfg.cme_residual_hidden, cfg.cme_residual_hidden),
            nn.GELU(),
            nn.Dropout(cfg.cme_residual_dropout),
            nn.Linear(cfg.cme_residual_hidden, 1),
        )
        _zero_last_linear(self.cme_residual_head)

    def forward(self, batch: dict) -> dict:
        out = super().forward(batch)
        v42_logit = out["win_logit"]
        B = v42_logit.shape[0]
        dtype = v42_logit.dtype
        device = v42_logit.device

        rating_margin_prior = batch.get("rating_margin_prior")
        if rating_margin_prior is None:
            rating_margin_prior = torch.zeros(B, device=device, dtype=dtype)
        else:
            rating_margin_prior = rating_margin_prior.to(dtype)
        rating_logit_prior = rating_margin_prior / max(float(self.rating_cfg.rating_margin_scale), 1e-6)

        rating_features = batch.get("rating_features")
        if rating_features is None:
            rating_features = torch.zeros(
                B, int(self.rating_cfg.rating_feature_dim), device=device, dtype=dtype,
            )
        elif rating_features.size(-1) == 0 and self.rating_cfg.rating_feature_dim > 0:
            rating_features = torch.zeros(
                B, int(self.rating_cfg.rating_feature_dim), device=device, dtype=dtype,
            )
        else:
            rating_features = rating_features.to(dtype)

        has_rating = batch.get("has_rating_prior")
        if has_rating is None:
            has_rating = torch.ones_like(v42_logit)
        else:
            has_rating = has_rating.to(dtype)

        rating_logit_slope = self.log_rating_logit_slope.exp().clamp(min=0.25, max=4.0)
        rating_logit_calibrated = rating_logit_slope * rating_logit_prior + self.rating_logit_bias
        rating_prob_raw = torch.sigmoid(rating_logit_prior)
        rating_prob_calibrated = torch.sigmoid(rating_logit_calibrated)

        cme_gap = v42_logit - rating_logit_calibrated
        abs_cme_gap = cme_gap.abs()
        abs_rating_logit = rating_logit_calibrated.abs()
        gate_logit = (
            float(self.rating_cfg.disagreement_gate_center)
            - float(self.rating_cfg.disagreement_gate_rating_confidence_w) * abs_rating_logit
            - float(self.rating_cfg.disagreement_gate_gap_w) * abs_cme_gap
        )
        disagreement_gate = torch.sigmoid(gate_logit)
        disagreement_gate = disagreement_gate * has_rating

        if self.rating_cfg.max_cme_gap_weight > 0.0:
            cme_gap_weight = (
                float(self.rating_cfg.max_cme_gap_weight) * torch.tanh(self.raw_cme_gap_weight)
            )
        else:
            cme_gap_weight = torch.zeros((), device=device, dtype=dtype)
        cme_gap_linear_residual = disagreement_gate * cme_gap_weight * cme_gap

        head_features = torch.cat([
            rating_features,
            rating_logit_calibrated.unsqueeze(-1),
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

        ratings_anchored_logit = (
            rating_logit_calibrated + cme_gap_linear_residual + cme_feature_residual
        )
        final_logit = has_rating * ratings_anchored_logit + (1.0 - has_rating) * v42_logit

        out.update({
            "win_logit_v42": v42_logit,
            "win_logit": final_logit,
            "rating_margin_prior": rating_margin_prior,
            "rating_logit_prior": rating_logit_prior,
            "rating_prob_raw": rating_prob_raw,
            "rating_logit_calibrated": rating_logit_calibrated,
            "rating_prob_calibrated": rating_prob_calibrated,
            "rating_logit_bias": self.rating_logit_bias,
            "rating_logit_slope": rating_logit_slope,
            "log_rating_logit_slope": self.log_rating_logit_slope,
            "cme_gap": cme_gap,
            "abs_cme_gap": abs_cme_gap,
            "disagreement_gate": disagreement_gate,
            "cme_gap_weight": cme_gap_weight,
            "raw_cme_gap_weight": self.raw_cme_gap_weight,
            "cme_gap_linear_residual": cme_gap_linear_residual,
            "raw_cme_feature_residual": raw_cme_feature_residual,
            "cme_feature_residual": cme_feature_residual,
            "ratings_anchored_logit": ratings_anchored_logit,
            "has_rating_prior": has_rating,
        })
        return out


def cme_feature_residual_l2_loss(out: dict) -> torch.Tensor:
    return out["cme_feature_residual"].pow(2).mean()


def cme_gap_weight_l2_loss(out: dict) -> torch.Tensor:
    return out["cme_gap_weight"].pow(2)


def rating_calibration_shift_l2_loss(out: dict) -> torch.Tensor:
    # Do not regularize the slope toward 1.0: the v1 CSV indicated that the
    # raw rating prior benefits from a slope well above 1.  We only discourage
    # a very large learned intercept drift.
    return out["rating_logit_bias"].pow(2)


def total_loss(
    out: dict,
    batch: dict,
    *,
    cme_feature_reg_w: float = 0.02,
    cme_gap_weight_reg_w: float = 0.002,
    rating_bias_reg_w: float = 0.0002,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    base_total, diag = total_loss_v42(out, batch, **kwargs)
    L_cme_feature_reg = cme_feature_residual_l2_loss(out)
    L_cme_gap_weight_reg = cme_gap_weight_l2_loss(out)
    L_rating_bias_reg = rating_calibration_shift_l2_loss(out)
    total = (
        base_total
        + cme_feature_reg_w * L_cme_feature_reg
        + cme_gap_weight_reg_w * L_cme_gap_weight_reg
        + rating_bias_reg_w * L_rating_bias_reg
    )
    diag = dict(diag)
    diag.update({
        "L_cme_feature_reg": L_cme_feature_reg.detach(),
        "L_cme_gap_weight_reg": L_cme_gap_weight_reg.detach(),
        "L_rating_bias_reg": L_rating_bias_reg.detach(),
    })
    return total, diag


def _smoke() -> None:
    torch.manual_seed(0)
    cfg = CmeRatingsV2Config(
        vocab_size=64,
        num_teams=30,
        d=32,
        n_heads=4,
        n_self_layers=1,
        n_cross_layers=1,
        tabular_dim=4,
        team_emb_dim=8,
        player_stat_dim=0,
        rating_feature_dim=K_RATING_FEATURES,
    )
    model = CmeRatingsV2(cfg)
    B, Lh, La = 3, 8, 9
    batch = {
        "home_idx": torch.randint(1, cfg.vocab_size + 1, (B, Lh)),
        "away_idx": torch.randint(1, cfg.vocab_size + 1, (B, La)),
        "home_prob": torch.rand(B, Lh),
        "away_prob": torch.rand(B, La),
        "home_mask": torch.ones(B, Lh, dtype=torch.bool),
        "away_mask": torch.ones(B, La, dtype=torch.bool),
        "home_stats": torch.zeros(B, Lh, 0),
        "away_stats": torch.zeros(B, La, 0),
        "home_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "away_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "home_rest": torch.rand(B),
        "away_rest": torch.rand(B),
        "tabular": torch.randn(B, cfg.tabular_dim),
        "season_phase": torch.randn(B, 3).clamp(-1.0, 1.0),
        "label": torch.randint(0, 2, (B,), dtype=torch.float32),
        "margin": torch.randn(B),
        "team_box_home": torch.rand(B, 14),
        "team_box_away": torch.rand(B, 14),
        "sup_pair_game": torch.zeros(0, dtype=torch.long),
        "sup_pair_side": torch.zeros(0, dtype=torch.long),
        "sup_pair_off": torch.zeros(0, dtype=torch.long),
        "sup_pair_def": torch.zeros(0, dtype=torch.long),
        "sup_pair_y": torch.zeros(0, 9),
        "sup_pl_game": torch.zeros(0, dtype=torch.long),
        "sup_pl_side": torch.zeros(0, dtype=torch.long),
        "sup_pl_slot": torch.zeros(0, dtype=torch.long),
        "sup_pl_y": torch.zeros(0, 14),
        "home_alpha_off_actual": torch.softmax(torch.randn(B, Lh), dim=-1),
        "home_alpha_def_actual": torch.softmax(torch.randn(B, Lh), dim=-1),
        "away_alpha_off_actual": torch.softmax(torch.randn(B, La), dim=-1),
        "away_alpha_def_actual": torch.softmax(torch.randn(B, La), dim=-1),
        "home_off_exposure_valid": torch.ones(B),
        "away_off_exposure_valid": torch.ones(B),
        "rating_features": torch.randn(B, K_RATING_FEATURES),
        "rating_margin_prior": torch.randn(B) * 6.0,
        "has_rating_prior": torch.ones(B),
    }
    out = model(batch)
    loss, diag = total_loss(
        out,
        batch,
        box_weights=torch.ones(14),
        pair_weights=torch.ones(9),
    )
    loss.backward()
    print("[smoke] win_logit", out["win_logit"].shape)
    print("[smoke] rating_calibrated", out["rating_logit_calibrated"].shape)
    print("[smoke] gate", out["disagreement_gate"].shape)
    print("[smoke] loss", float(loss.item()))
    print("[smoke] L_cme_feature_reg", float(diag["L_cme_feature_reg"].item()))


if __name__ == "__main__":
    _smoke()
