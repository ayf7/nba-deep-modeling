"""CME-Ratings-v3: multi-timescale ratings anchor + gated CME residual.

Ratings-v2 established that a calibrated dynamic opponent-adjusted ratings
prior is stronger than a CME-first hybrid.  Ratings-v3 improves the *ratings*
side of that architecture by maintaining three strictly pregame rating tracks:

    slow   = stable carryover / conservative updates
    medium = balanced v2-like updates
    fast   = reactive recent-form updates

The model learns a compact ratings-only mixture of these tracks, calibrates the
result, and then permits only a small disagreement-gated CME correction.
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
from cme_ratings_v3_common import K_RATING_FEATURES, K_RATING_TRACKS  # noqa: E402


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


def _logits_from_probs(probs: tuple[float, ...]) -> list[float]:
    import math as _math

    out = []
    for p in probs:
        p = min(max(float(p), 1e-4), 1.0 - 1e-4)
        out.append(_math.log(p / (1.0 - p)))
    return out


@dataclass
class CmeRatingsV3Config(CmeV42Config):
    rating_feature_dim: int = K_RATING_FEATURES
    n_rating_tracks: int = K_RATING_TRACKS
    rating_mix_hidden: int = 32
    rating_mix_dropout: float = 0.05
    max_rating_mix_logit_delta: float = 0.75
    init_rating_mix_probs: tuple[float, ...] = (0.25, 0.50, 0.25)
    init_rating_logit_bias: float = 0.04
    init_rating_track_slopes: tuple[float, ...] = (1.45, 1.50, 1.55)
    cme_residual_hidden: int = 32
    cme_residual_dropout: float = 0.05
    max_cme_gap_weight: float = 0.35
    init_cme_gap_weight: float = 0.00
    max_cme_feature_residual: float = 0.15
    disagreement_gate_center: float = 0.80
    disagreement_gate_rating_confidence_w: float = 1.00
    disagreement_gate_gap_w: float = 0.75


class CmeRatingsV3(CmeV42):
    def __init__(self, cfg: CmeRatingsV3Config) -> None:
        super().__init__(cfg)
        self.rating_cfg = cfg
        if cfg.n_rating_tracks != K_RATING_TRACKS:
            raise ValueError(f"n_rating_tracks must be {K_RATING_TRACKS}")
        if len(cfg.init_rating_track_slopes) != cfg.n_rating_tracks:
            raise ValueError("init_rating_track_slopes length mismatch")
        if len(cfg.init_rating_mix_probs) != cfg.n_rating_tracks:
            raise ValueError("init_rating_mix_probs length mismatch")
        if any(s <= 0.0 for s in cfg.init_rating_track_slopes):
            raise ValueError("all init_rating_track_slopes must be > 0")

        self.rating_logit_bias = nn.Parameter(
            torch.tensor(float(cfg.init_rating_logit_bias), dtype=torch.float32)
        )
        self.log_rating_track_slopes = nn.Parameter(
            torch.tensor([math.log(float(s)) for s in cfg.init_rating_track_slopes], dtype=torch.float32)
        )
        self.rating_mix_base_logits = nn.Parameter(
            torch.tensor(_logits_from_probs(tuple(cfg.init_rating_mix_probs)), dtype=torch.float32)
        )
        self.rating_mix_head = nn.Sequential(
            nn.Linear(int(cfg.rating_feature_dim), cfg.rating_mix_hidden),
            nn.GELU(),
            nn.Dropout(cfg.rating_mix_dropout),
            nn.Linear(cfg.rating_mix_hidden, cfg.rating_mix_hidden),
            nn.GELU(),
            nn.Dropout(cfg.rating_mix_dropout),
            nn.Linear(cfg.rating_mix_hidden, cfg.n_rating_tracks),
        )
        _zero_last_linear(self.rating_mix_head)

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

        rating_features = batch.get("rating_features")
        if rating_features is None:
            rating_features = torch.zeros(B, int(self.rating_cfg.rating_feature_dim), device=device, dtype=dtype)
        elif rating_features.size(-1) == 0 and self.rating_cfg.rating_feature_dim > 0:
            rating_features = torch.zeros(B, int(self.rating_cfg.rating_feature_dim), device=device, dtype=dtype)
        else:
            rating_features = rating_features.to(dtype)

        track_logits_raw = batch.get("rating_track_logits")
        if track_logits_raw is None:
            track_logits_raw = torch.zeros(B, self.rating_cfg.n_rating_tracks, device=device, dtype=dtype)
        else:
            track_logits_raw = track_logits_raw.to(dtype)
        track_margins = batch.get("rating_track_margins")
        if track_margins is None:
            track_margins = torch.zeros(B, self.rating_cfg.n_rating_tracks, device=device, dtype=dtype)
        else:
            track_margins = track_margins.to(dtype)

        has_rating = batch.get("has_rating_prior")
        if has_rating is None:
            has_rating = torch.ones_like(v42_logit)
        else:
            has_rating = has_rating.to(dtype)

        if int(self.rating_cfg.rating_feature_dim) > 0:
            raw_mix_delta = self.rating_mix_head(rating_features)
        else:
            raw_mix_delta = torch.zeros(B, self.rating_cfg.n_rating_tracks, device=device, dtype=dtype)
        if self.rating_cfg.max_rating_mix_logit_delta > 0.0:
            mix_delta = float(self.rating_cfg.max_rating_mix_logit_delta) * torch.tanh(raw_mix_delta)
        else:
            mix_delta = torch.zeros_like(raw_mix_delta)
        mix_logits = self.rating_mix_base_logits.to(dtype).unsqueeze(0) + mix_delta
        track_weights = torch.softmax(mix_logits, dim=-1)

        track_slopes = self.log_rating_track_slopes.exp().clamp(min=0.25, max=4.0).to(dtype)
        track_logits_calibrated = track_logits_raw * track_slopes.unsqueeze(0) + self.rating_logit_bias
        track_probs_raw = torch.sigmoid(track_logits_raw)
        track_probs_calibrated = torch.sigmoid(track_logits_calibrated)

        rating_logit_prior = (track_weights * track_logits_raw).sum(dim=-1)
        rating_margin_prior = (track_weights * track_margins).sum(dim=-1)
        rating_prob_raw = torch.sigmoid(rating_logit_prior)
        rating_logit_calibrated = (track_weights * track_logits_calibrated).sum(dim=-1)
        rating_prob_calibrated = torch.sigmoid(rating_logit_calibrated)

        cme_gap = v42_logit - rating_logit_calibrated
        abs_cme_gap = cme_gap.abs()
        abs_rating_logit = rating_logit_calibrated.abs()
        gate_logit = (
            float(self.rating_cfg.disagreement_gate_center)
            - float(self.rating_cfg.disagreement_gate_rating_confidence_w) * abs_rating_logit
            - float(self.rating_cfg.disagreement_gate_gap_w) * abs_cme_gap
        )
        disagreement_gate = torch.sigmoid(gate_logit) * has_rating

        if self.rating_cfg.max_cme_gap_weight > 0.0:
            cme_gap_weight = float(self.rating_cfg.max_cme_gap_weight) * torch.tanh(self.raw_cme_gap_weight)
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

        ratings_anchored_logit = rating_logit_calibrated + cme_gap_linear_residual + cme_feature_residual
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
            "rating_track_slopes": track_slopes,
            "rating_logit_slope": track_slopes.mean(),
            "log_rating_track_slopes": self.log_rating_track_slopes,
            "rating_track_logits_raw": track_logits_raw,
            "rating_track_logits_calibrated": track_logits_calibrated,
            "rating_track_probs_raw": track_probs_raw,
            "rating_track_probs_calibrated": track_probs_calibrated,
            "rating_track_margins": track_margins,
            "rating_track_weights": track_weights,
            "rating_mix_base_logits": self.rating_mix_base_logits,
            "raw_rating_mix_delta": raw_mix_delta,
            "rating_mix_delta": mix_delta,
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
    return out["rating_logit_bias"].pow(2)


def rating_mix_delta_l2_loss(out: dict) -> torch.Tensor:
    return out["rating_mix_delta"].pow(2).mean()


def rating_track_slope_spread_l2_loss(out: dict) -> torch.Tensor:
    logs = out["log_rating_track_slopes"]
    return (logs - logs.mean()).pow(2).mean()


def total_loss(
    out: dict,
    batch: dict,
    *,
    cme_feature_reg_w: float = 0.02,
    cme_gap_weight_reg_w: float = 0.002,
    rating_bias_reg_w: float = 0.0002,
    rating_mix_delta_reg_w: float = 0.001,
    rating_track_slope_spread_reg_w: float = 0.0005,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    base_total, diag = total_loss_v42(out, batch, **kwargs)
    L_cme_feature_reg = cme_feature_residual_l2_loss(out)
    L_cme_gap_weight_reg = cme_gap_weight_l2_loss(out)
    L_rating_bias_reg = rating_calibration_shift_l2_loss(out)
    L_rating_mix_delta_reg = rating_mix_delta_l2_loss(out)
    L_rating_track_slope_spread_reg = rating_track_slope_spread_l2_loss(out)
    total = (
        base_total
        + cme_feature_reg_w * L_cme_feature_reg
        + cme_gap_weight_reg_w * L_cme_gap_weight_reg
        + rating_bias_reg_w * L_rating_bias_reg
        + rating_mix_delta_reg_w * L_rating_mix_delta_reg
        + rating_track_slope_spread_reg_w * L_rating_track_slope_spread_reg
    )
    diag = dict(diag)
    diag.update({
        "L_cme_feature_reg": L_cme_feature_reg.detach(),
        "L_cme_gap_weight_reg": L_cme_gap_weight_reg.detach(),
        "L_rating_bias_reg": L_rating_bias_reg.detach(),
        "L_rating_mix_delta_reg": L_rating_mix_delta_reg.detach(),
        "L_rating_track_slope_spread_reg": L_rating_track_slope_spread_reg.detach(),
    })
    return total, diag


def _smoke() -> None:
    torch.manual_seed(0)
    from cme_v4_2_model import K_BOX, K_PAIR, SEASON_PHASE_DIM
    cfg = CmeRatingsV3Config(
        vocab_size=200,
        num_teams=30,
        d=64,
        tabular_dim=4,
        team_emb_dim=8,
        player_stat_dim=10,
        rating_feature_dim=K_RATING_FEATURES,
    )
    model = CmeRatingsV3(cfg)
    B, Lh, La = 4, 14, 16
    home_idx = torch.randint(1, cfg.vocab_size + 1, (B, Lh))
    away_idx = torch.randint(1, cfg.vocab_size + 1, (B, La))
    home_mask = torch.ones(B, Lh, dtype=torch.bool); home_mask[0, 12:] = False
    away_mask = torch.ones(B, La, dtype=torch.bool); away_mask[1, 14:] = False
    home_prob = torch.rand(B, Lh) * home_mask.to(torch.float32)
    away_prob = torch.rand(B, La) * away_mask.to(torch.float32)

    def _rand_share(mask):
        x = torch.rand_like(mask, dtype=torch.float32) * mask.to(torch.float32)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    batch = {
        "home_idx": home_idx, "away_idx": away_idx,
        "home_prob": home_prob, "away_prob": away_prob,
        "home_mask": home_mask, "away_mask": away_mask,
        "home_stats": torch.randn(B, Lh, cfg.player_stat_dim),
        "away_stats": torch.randn(B, La, cfg.player_stat_dim),
        "home_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "away_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "home_rest": torch.rand(B), "away_rest": torch.rand(B),
        "tabular": torch.randn(B, cfg.tabular_dim),
        "season_phase": torch.randn(B, SEASON_PHASE_DIM).clamp(-1.0, 1.0),
        "label": torch.randint(0, 2, (B,), dtype=torch.float32),
        "margin": torch.randn(B) * 10,
        "team_box_home": torch.rand(B, K_BOX) * 30,
        "team_box_away": torch.rand(B, K_BOX) * 30,
        "sup_pair_game": torch.tensor([0, 0, 1, 2, 3]),
        "sup_pair_side": torch.tensor([0, 1, 0, 0, 1]),
        "sup_pair_off": torch.tensor([0, 0, 1, 2, 3]),
        "sup_pair_def": torch.tensor([0, 1, 2, 3, 4]),
        "sup_pair_y": torch.rand(5, K_PAIR) * 3.0,
        "sup_pl_game": torch.tensor([0, 0, 1, 2, 3, 3]),
        "sup_pl_side": torch.tensor([0, 1, 0, 1, 0, 1]),
        "sup_pl_slot": torch.tensor([0, 1, 2, 3, 4, 5]),
        "sup_pl_y": torch.rand(6, K_BOX) * 5.0,
        "home_alpha_off_actual": _rand_share(home_mask),
        "home_alpha_def_actual": _rand_share(home_mask),
        "away_alpha_off_actual": _rand_share(away_mask),
        "away_alpha_def_actual": _rand_share(away_mask),
        "home_off_exposure_valid": torch.ones(B, dtype=torch.float32),
        "away_off_exposure_valid": torch.ones(B, dtype=torch.float32),
        "rating_features": torch.randn(B, K_RATING_FEATURES),
        "rating_track_margins": torch.randn(B, K_RATING_TRACKS) * 6.0,
        "rating_track_logits": torch.randn(B, K_RATING_TRACKS) * 0.5,
        "has_rating_prior": torch.ones(B),
    }
    out = model(batch)
    total, _ = total_loss(
        out, batch,
        box_weights=torch.ones(K_BOX), pair_weights=torch.ones(K_PAIR),
        win_w=1.0, team_w=0.0, player_w=0.0, pair_w=0.0, inv_w=0.0,
    )
    total.backward()
    print("[smoke] ok", float(total.detach()))

if __name__ == "__main__":
    _smoke()
