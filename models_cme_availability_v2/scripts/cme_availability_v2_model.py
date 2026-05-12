"""CME-Availability-v2.

This branch keeps CME-v4.2's structured player/matchup world model and makes a
more careful distinction than Availability-v1/v1.1:

* ``prob_play`` remains the original status-derived **medical availability**
  prior used by v4.2.  The learned branch never overwrites this player-token
  gate.
* A separate rotation/impact branch predicts:
    - a meaningful-rotation probability,
    - expected log involvement-seconds, and
    - an implied within-roster role share.
* Only the rotation/impact score enters CME, as a tightly bounded prior inside
  the involvement softmaxes used by the Sinkhorn exposure allocation.

The experimental question is therefore clean: can pregame rotation/minutes
priors improve who CME assigns exposure to, without confusing medical
availability with coach/rotation appearance decisions?
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
V42_SCRIPTS = REPO_ROOT / "models_cme_v4_2" / "scripts"
HERE = Path(__file__).resolve().parent
if str(V42_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(V42_SCRIPTS))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cme_v4_2_model import (  # noqa: E402
    CmeV42,
    CmeV42Config,
    _zero_last_linear,
    masked_softmax,
    total_loss as total_loss_v42,
)
from cme_v4_2_common import BOX_INDEX, K_BOX, K_PAIR, K_PLAYER, SEASON_PHASE_DIM  # noqa: E402
from cme_availability_v2_common import AVAILABILITY_FEATURE_DIM, MIN_ROTATION_TARGET_SECONDS  # noqa: E402


@dataclass
class CmeAvailabilityV2Config(CmeV42Config):
    availability_feature_dim: int = AVAILABILITY_FEATURE_DIM
    availability_hidden: int = 64
    availability_dropout: float = 0.05
    # First head output is a bounded meaningful-rotation logit; the second is
    # a residual around a leakage-safe recent player-seconds prior.
    max_rotation_logit: float = 4.0
    max_minute_log_delta: float = 1.5
    # Rotation confidence contributes gently to the role score before the
    # centered role prior enters the involvement softmax.
    rotation_log_prob_weight: float = 0.50
    # Bounded role prior in (0, max_role_prior_strength), initialized small so
    # this branch can improve allocation without disrupting the proven v4.2
    # backbone on day one.
    max_role_prior_strength: float = 0.20
    init_role_prior_strength: float = 0.02
    # Deprecated compatibility fields retained so older train/backtest command
    # lines from Availability-v1.1 do not fail when copied into v2.  They are
    # intentionally unused: v2 never replaces CME's medical play prior.
    init_play_logit_bias: float = 0.0
    init_play_logit_slope: float = 1.0
    max_play_prior_mix_strength: float = 0.0
    init_play_prior_mix_strength: float = 0.0


class CmeAvailabilityV2(CmeV42):
    def __init__(self, cfg: CmeAvailabilityV2Config) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        d = cfg.d
        if cfg.availability_feature_dim < 0:
            raise ValueError("availability_feature_dim must be >= 0")
        avail_in = d + d + cfg.availability_feature_dim
        self.rotation_head = nn.Sequential(
            nn.Linear(avail_in, cfg.availability_hidden),
            nn.GELU(),
            nn.Dropout(cfg.availability_dropout),
            nn.Linear(cfg.availability_hidden, 2),  # rotation logit, minute-log delta
        )
        _zero_last_linear(self.rotation_head)
        max_role = max(float(cfg.max_role_prior_strength), 1e-6)
        init_role = min(max(float(cfg.init_role_prior_strength), 1e-6), max_role - 1e-6)
        init_role_frac = init_role / max_role
        self.availability_role_logit_strength = nn.Parameter(
            torch.tensor(math.log(init_role_frac / (1.0 - init_role_frac)), dtype=torch.float32)
        )

    # ------------------------- availability -------------------------

    def _base_player_repr(
        self,
        idx: torch.Tensor,
        stats: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        e = self.E_player(idx)
        if self.feat_mlp is not None and stats.size(-1) > 0:
            e = e + self.feat_mlp(stats)
        return e * mask.unsqueeze(-1).to(e.dtype)

    def _rotation_predictions(
        self,
        base_repr: torch.Tensor,
        ctx: torch.Tensor,
        avail_features: torch.Tensor,
        minute_log_prior: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, L, d = base_repr.shape
        if avail_features.size(-1) != self.cfg.availability_feature_dim:
            raise ValueError(
                f"availability feature dim mismatch: got {avail_features.size(-1)}, "
                f"expected {self.cfg.availability_feature_dim}"
            )
        ctx_e = ctx.unsqueeze(1).expand(-1, L, -1)
        x = torch.cat([base_repr, ctx_e, avail_features], dim=-1)
        raw = self.rotation_head(x)
        raw_rotation = raw[..., 0]
        raw_minute_delta = raw[..., 1]
        if self.cfg.max_rotation_logit > 0.0:
            rotation_logit = self.cfg.max_rotation_logit * torch.tanh(raw_rotation)
        else:
            rotation_logit = torch.zeros_like(raw_rotation)
        if self.cfg.max_minute_log_delta > 0.0:
            minute_log_delta = self.cfg.max_minute_log_delta * torch.tanh(raw_minute_delta)
        else:
            minute_log_delta = torch.zeros_like(raw_minute_delta)
        rotation_prob = torch.sigmoid(rotation_logit)
        pred_log_seconds = (minute_log_prior + minute_log_delta).clamp(min=0.0, max=12.0)
        role_score = pred_log_seconds + float(self.cfg.rotation_log_prob_weight) * torch.log(rotation_prob.clamp_min(1e-6))
        role_share = masked_softmax(role_score, mask, dim=-1)
        rotation_prob = rotation_prob * mask.to(rotation_prob.dtype)
        pred_log_seconds = pred_log_seconds * mask.to(pred_log_seconds.dtype)
        role_score = role_score * mask.to(role_score.dtype)
        return {
            "rotation_logit": rotation_logit,
            "rotation_prob": rotation_prob,
            "minute_log_delta": minute_log_delta,
            "pred_log_seconds": pred_log_seconds,
            "role_score": role_score,
            "role_share": role_share,
        }

    def _tokenize_from_base(
        self,
        base_repr: torch.Tensor,
        play_prob: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        e = base_repr * play_prob.unsqueeze(-1)
        return e * mask.unsqueeze(-1).to(e.dtype)

    def _involvement_with_rotation_prior(
        self,
        h: torch.Tensor,
        ctx: torch.Tensor,
        medical_prob: torch.Tensor,
        role_score: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, d = h.shape
        ctx_e = ctx.unsqueeze(1).expand(-1, L, -1)
        h_in = torch.cat([h, ctx_e], dim=-1)
        logit_off = self.inv_off_head(h_in).squeeze(-1)
        logit_def = self.inv_def_head(h_in).squeeze(-1)
        max_role = max(float(self.cfg.max_role_prior_strength), 0.0)
        if max_role > 0.0:
            role_strength = max_role * torch.sigmoid(self.availability_role_logit_strength)
        else:
            role_strength = torch.zeros((), device=h.device, dtype=h.dtype)
        mask_f = mask.to(role_score.dtype)
        role_mean = (role_score * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        centered_role = (role_score - role_mean).clamp(min=-4.0, max=4.0)
        # Critical v2 distinction: the medical/status play prior remains the
        # v4.2 prior.  Rotation/minutes only nudge involvement allocation.
        log_prior = torch.log(medical_prob.clamp_min(1e-6)) + role_strength * centered_role
        alpha_off = masked_softmax(logit_off + log_prior, mask, dim=-1)
        alpha_def = masked_softmax(logit_def + log_prior, mask, dim=-1)
        return alpha_off, alpha_def, role_strength

    # ------------------------- forward -------------------------

    def forward(self, batch: dict) -> dict:
        home_mask = batch["home_mask"]
        away_mask = batch["away_mask"]
        tabular = batch.get("tabular")

        # Team context is available pregame and is used by the availability
        # branch.  This reverses the v4.2 forward order, but not the values.
        home_ctx = self._team_ctx(batch["home_team_idx"], tabular, batch["home_rest"])
        away_ctx = self._team_ctx(batch["away_team_idx"], tabular, batch["away_rest"])

        home_base = self._base_player_repr(batch["home_idx"], batch["home_stats"], home_mask)
        away_base = self._base_player_repr(batch["away_idx"], batch["away_stats"], away_mask)
        home_av = self._rotation_predictions(
            home_base,
            home_ctx,
            batch["home_avail_features"],
            batch["home_avail_minute_log_prior"],
            home_mask,
        )
        away_av = self._rotation_predictions(
            away_base,
            away_ctx,
            batch["away_avail_features"],
            batch["away_avail_minute_log_prior"],
            away_mask,
        )

        # Medical availability remains exactly the original v4.2 play mask.
        home_tok = self._tokenize_from_base(home_base, batch["home_prob"], home_mask)
        away_tok = self._tokenize_from_base(away_base, batch["away_prob"], away_mask)
        home_tok = home_tok + home_ctx.unsqueeze(1) * home_mask.unsqueeze(-1).to(home_tok.dtype)
        away_tok = away_tok + away_ctx.unsqueeze(1) * away_mask.unsqueeze(-1).to(away_tok.dtype)

        home_self = home_tok
        away_self = away_tok
        for layer in self.self_layers:
            home_self = layer(home_self, home_mask)
            away_self = layer(away_self, away_mask)

        alpha_home_off, alpha_home_def, role_strength = self._involvement_with_rotation_prior(
            home_self, home_ctx, batch["home_prob"], home_av["role_score"], home_mask,
        )
        alpha_away_off, alpha_away_def, _ = self._involvement_with_rotation_prior(
            away_self, away_ctx, batch["away_prob"], away_av["role_score"], away_mask,
        )

        N = F.softplus(self.poss_head(torch.cat([home_ctx, away_ctx], dim=-1)).squeeze(-1))
        N = N + self.cfg.base_possessions_per_team

        home_off = self.off_proj(home_self) * home_mask.unsqueeze(-1).to(home_self.dtype)
        home_def = self.def_proj(home_self) * home_mask.unsqueeze(-1).to(home_self.dtype)
        away_off = self.off_proj(away_self) * away_mask.unsqueeze(-1).to(away_self.dtype)
        away_def = self.def_proj(away_self) * away_mask.unsqueeze(-1).to(away_self.dtype)
        for layer in self.cross_layers:
            home_off = layer(home_off, away_def, q_mask=home_mask, kv_mask=away_mask)
            away_off = layer(away_off, home_def, q_mask=away_mask, kv_mask=home_mask)

        dir_A = self._direction(
            home_off, away_def, alpha_home_off, alpha_away_def,
            home_mask, away_mask, N,
        )
        dir_B = self._direction(
            away_off, home_def, alpha_away_off, alpha_home_def,
            away_mask, home_mask, N,
        )

        home_pair_marg = dir_A["lam"].sum(dim=2)
        away_pair_marg = dir_B["lam"].sum(dim=2)
        home_player_raw = self._player_rates(home_self, home_mask)
        away_player_raw = self._player_rates(away_self, away_mask)
        home_player_rates = self._scale_player_rates(home_player_raw, alpha_home_off, alpha_home_def)
        away_player_rates = self._scale_player_rates(away_player_raw, alpha_away_off, alpha_away_def)
        home_box = self._assemble_player_box(home_pair_marg, home_player_rates)
        away_box = self._assemble_player_box(away_pair_marg, away_player_rates)
        home_box = home_box * home_mask.unsqueeze(-1).to(home_box.dtype)
        away_box = away_box * away_mask.unsqueeze(-1).to(away_box.dtype)
        home_team_box = home_box.sum(dim=1)
        away_team_box = away_box.sum(dim=1)

        pts_idx = BOX_INDEX["pts"]
        home_points = home_team_box[..., pts_idx]
        away_points = away_team_box[..., pts_idx]
        margin_mu = home_points - away_points
        scale = self.global_scale.exp().clamp_min(1.0)
        win_logit_base = margin_mu / scale
        win_logit_slope = self.log_win_logit_slope.exp().clamp(min=0.25, max=4.0)
        win_logit_affine = win_logit_slope * win_logit_base + self.home_logit_bias

        season_phase = batch.get("season_phase")
        if season_phase is None:
            season_phase = torch.zeros(
                win_logit_affine.size(0),
                SEASON_PHASE_DIM,
                device=win_logit_affine.device,
                dtype=win_logit_affine.dtype,
            )
        raw_season_delta = self.season_calibration_head(season_phase).squeeze(-1)
        if self.cfg.max_season_logit_adjustment > 0.0:
            season_logit_adjustment = self.cfg.max_season_logit_adjustment * torch.tanh(raw_season_delta)
        else:
            season_logit_adjustment = torch.zeros_like(win_logit_affine)
        win_logit_temporal = win_logit_affine + season_logit_adjustment

        calibration_features = torch.cat([
            home_ctx,
            away_ctx,
            home_ctx - away_ctx,
            home_ctx * away_ctx,
            win_logit_temporal.unsqueeze(-1),
        ], dim=-1)
        raw_win_residual = self.win_calibration_head(calibration_features).squeeze(-1)
        if self.cfg.max_calibration_residual > 0.0:
            bounded_win_residual = self.cfg.max_calibration_residual * torch.tanh(raw_win_residual)
        else:
            bounded_win_residual = torch.zeros_like(win_logit_temporal)
        if self.cfg.tail_gate_sharpness > 0.0:
            residual_tail_gate = torch.sigmoid(
                (self.cfg.tail_gate_center - win_logit_temporal.abs()) * self.cfg.tail_gate_sharpness
            )
        else:
            residual_tail_gate = torch.ones_like(win_logit_temporal)
        win_residual = residual_tail_gate * bounded_win_residual
        win_logit = win_logit_temporal + win_residual
        margin_log_sigma = self.margin_log_sigma_head(torch.cat([home_ctx, away_ctx], dim=-1)).squeeze(-1)

        return {
            # v4.2 outputs
            "pair_dir_A": dir_A["lam"],
            "pair_dir_B": dir_B["lam"],
            "P_A": dir_A["P"],
            "P_B": dir_B["P"],
            "rate_A": dir_A["rate"],
            "rate_B": dir_B["rate"],
            "pair_mask_A": home_mask.unsqueeze(-1) & away_mask.unsqueeze(1),
            "pair_mask_B": away_mask.unsqueeze(-1) & home_mask.unsqueeze(1),
            "alpha_home_off": alpha_home_off,
            "alpha_home_def": alpha_home_def,
            "alpha_away_off": alpha_away_off,
            "alpha_away_def": alpha_away_def,
            "N": N,
            "home_player_rates": home_player_rates,
            "away_player_rates": away_player_rates,
            "home_pair_marg": home_pair_marg,
            "away_pair_marg": away_pair_marg,
            "home_box": home_box,
            "away_box": away_box,
            "home_team_box": home_team_box,
            "away_team_box": away_team_box,
            "home_points": home_points,
            "away_points": away_points,
            "margin_mu": margin_mu,
            "margin_log_sigma": margin_log_sigma,
            "win_logit": win_logit,
            "win_logit_base": win_logit_base,
            "win_logit_affine": win_logit_affine,
            "win_logit_temporal": win_logit_temporal,
            "win_logit_slope": win_logit_slope,
            "log_win_logit_slope": self.log_win_logit_slope,
            "season_phase": season_phase,
            "season_logit_adjustment": season_logit_adjustment,
            "residual_tail_gate": residual_tail_gate,
            "bounded_win_residual": bounded_win_residual,
            "win_residual": win_residual,
            "home_logit_bias": self.home_logit_bias,
            "global_scale": scale,
            "home_ctx": home_ctx,
            "away_ctx": away_ctx,
            # rotation/impact branch diagnostics / supervised outputs
            "home_rotation_logit": home_av["rotation_logit"],
            "away_rotation_logit": away_av["rotation_logit"],
            "home_rotation_prob": home_av["rotation_prob"],
            "away_rotation_prob": away_av["rotation_prob"],
            "home_avail_minute_log_delta": home_av["minute_log_delta"],
            "away_avail_minute_log_delta": away_av["minute_log_delta"],
            "home_avail_log_seconds_pred": home_av["pred_log_seconds"],
            "away_avail_log_seconds_pred": away_av["pred_log_seconds"],
            "home_avail_role_score": home_av["role_score"],
            "away_avail_role_score": away_av["role_score"],
            "home_avail_role_share_pred": home_av["role_share"],
            "away_avail_role_share_pred": away_av["role_share"],
            # Backward-compatible aliases used by the v1.1 train/backtest
            # logging code.  In Availability-v2 these report **rotation**
            # probabilities, not medical appearance probabilities.
            "home_avail_play_logit": home_av["rotation_logit"],
            "away_avail_play_logit": away_av["rotation_logit"],
            "home_avail_play_prob": home_av["rotation_prob"],
            "away_avail_play_prob": away_av["rotation_prob"],
            "home_avail_play_prob_uncalibrated": home_av["rotation_prob"],
            "away_avail_play_prob_uncalibrated": away_av["rotation_prob"],
            "home_avail_cme_play_prob": batch["home_prob"],
            "away_avail_cme_play_prob": batch["away_prob"],
            "availability_play_prior_mix_strength": torch.zeros((), device=win_logit.device, dtype=win_logit.dtype),
            "availability_play_logit_bias": torch.zeros((), device=win_logit.device, dtype=win_logit.dtype),
            "availability_play_logit_slope": torch.ones((), device=win_logit.device, dtype=win_logit.dtype),
            "availability_role_prior_strength": role_strength,
        }


# ----------------------------- availability losses -----------------------------


def _masked_slot_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.to(value.dtype).sum().clamp_min(1.0)
    return (value * mask.to(value.dtype)).sum() / denom


def availability_rotation_bce_loss(out: dict, batch: dict) -> torch.Tensor:
    threshold = math.log1p(MIN_ROTATION_TARGET_SECONDS)
    home_target = (batch["home_avail_log_seconds_actual"] >= threshold).to(out["home_rotation_logit"].dtype)
    away_target = (batch["away_avail_log_seconds_actual"] >= threshold).to(out["away_rotation_logit"].dtype)
    home = F.binary_cross_entropy_with_logits(
        out["home_rotation_logit"],
        home_target,
        reduction="none",
    )
    away = F.binary_cross_entropy_with_logits(
        out["away_rotation_logit"],
        away_target,
        reduction="none",
    )
    h = _masked_slot_mean(home, batch["home_mask"])
    a = _masked_slot_mean(away, batch["away_mask"])
    return 0.5 * (h + a)


def availability_minutes_huber_loss(out: dict, batch: dict) -> torch.Tensor:
    home = F.smooth_l1_loss(
        out["home_avail_log_seconds_pred"],
        batch["home_avail_log_seconds_actual"],
        reduction="none",
        beta=1.0,
    )
    away = F.smooth_l1_loss(
        out["away_avail_log_seconds_pred"],
        batch["away_avail_log_seconds_actual"],
        reduction="none",
        beta=1.0,
    )
    h = _masked_slot_mean(home, batch["home_mask"])
    a = _masked_slot_mean(away, batch["away_mask"])
    return 0.5 * (h + a)


def availability_role_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    home_err = (out["home_avail_role_share_pred"] - batch["home_avail_role_share_actual"]) ** 2
    away_err = (out["away_avail_role_share_pred"] - batch["away_avail_role_share_actual"]) ** 2
    home_per_game = (home_err * batch["home_mask"].to(home_err.dtype)).sum(dim=1) / batch["home_mask"].sum(dim=1).clamp_min(1).to(home_err.dtype)
    away_per_game = (away_err * batch["away_mask"].to(away_err.dtype)).sum(dim=1) / batch["away_mask"].sum(dim=1).clamp_min(1).to(away_err.dtype)
    h_valid = batch["home_avail_role_valid"].to(home_per_game.dtype)
    a_valid = batch["away_avail_role_valid"].to(away_per_game.dtype)
    h = (home_per_game * h_valid).sum() / h_valid.sum().clamp_min(1.0)
    a = (away_per_game * a_valid).sum() / a_valid.sum().clamp_min(1.0)
    return 0.5 * (h + a)



def availability_delta_reg_loss(out: dict, batch: dict) -> torch.Tensor:
    # Keep the new branch local: minute residuals should not become a large
    # hidden replacement for the existing involvement heads, and the rotation
    # logits should not grow extreme without evidence.
    home = 0.10 * out["home_rotation_logit"].pow(2) + 0.25 * out["home_avail_minute_log_delta"].pow(2)
    away = 0.10 * out["away_rotation_logit"].pow(2) + 0.25 * out["away_avail_minute_log_delta"].pow(2)
    h = _masked_slot_mean(home, batch["home_mask"])
    a = _masked_slot_mean(away, batch["away_mask"])
    return 0.5 * (h + a)



def total_loss_availability_v2(
    out: dict,
    batch: dict,
    *,
    box_weights: torch.Tensor,
    pair_weights: torch.Tensor,
    team_w: float = 1.0,
    player_w: float = 0.01,
    pair_w: float = 0.001,
    inv_w: float = 5.0,
    win_w: float = 10.0,
    margin_nll_w: float = 0.0,
    calibration_reg_w: float = 0.075,
    season_calibration_reg_w: float = 0.02,
    calibration_slope_reg_w: float = 0.001,
    availability_rotation_w: float = 0.10,
    availability_minutes_w: float = 0.02,
    availability_role_w: float = 0.50,
    availability_delta_reg_w: float = 0.01,
    availability_play_slope_reg_w_REMOVED: float = 0.0,
    **_ignored,
) -> tuple[torch.Tensor, dict]:
    base_total, diag = total_loss_v42(
        out,
        batch,
        box_weights=box_weights,
        pair_weights=pair_weights,
        team_w=team_w,
        player_w=player_w,
        pair_w=pair_w,
        inv_w=inv_w,
        win_w=win_w,
        margin_nll_w=margin_nll_w,
        calibration_reg_w=calibration_reg_w,
        season_calibration_reg_w=season_calibration_reg_w,
        calibration_slope_reg_w=calibration_slope_reg_w,
    )
    L_rotation = availability_rotation_bce_loss(out, batch)
    L_minutes = availability_minutes_huber_loss(out, batch)
    L_role = availability_role_mse_loss(out, batch)
    L_delta_reg = availability_delta_reg_loss(out, batch)
    total = (
        base_total
        + availability_rotation_w * L_rotation
        + availability_minutes_w * L_minutes
        + availability_role_w * L_role
        + availability_delta_reg_w * L_delta_reg
    )
    diag = dict(diag)
    diag.update({
        "L_availability_rotation": L_rotation.detach(),
        # Compatibility aliases for reused v1.1 logging code.
        "L_availability_play": L_rotation.detach(),
        "L_availability_play_slope_reg": torch.zeros((), device=L_rotation.device, dtype=L_rotation.dtype),
        "L_availability_minutes": L_minutes.detach(),
        "L_availability_role": L_role.detach(),
        "L_availability_delta_reg": L_delta_reg.detach(),
    })
    return total, diag


# ----------------------------- smoke -----------------------------


def _smoke() -> None:
    torch.manual_seed(0)
    cfg = CmeAvailabilityV2Config(
        vocab_size=200,
        num_teams=30,
        d=64,
        tabular_dim=4,
        team_emb_dim=8,
        player_stat_dim=10,
        availability_feature_dim=AVAILABILITY_FEATURE_DIM,
    )
    model = CmeAvailabilityV2(cfg)
    B, Lh, La = 3, 12, 13
    home_mask = torch.ones(B, Lh, dtype=torch.bool)
    away_mask = torch.ones(B, La, dtype=torch.bool)
    home_mask[0, -2:] = False
    away_mask[1, -1:] = False

    def rand_share(mask: torch.Tensor) -> torch.Tensor:
        x = torch.rand_like(mask, dtype=torch.float32) * mask.to(torch.float32)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    batch = {
        "home_idx": torch.randint(1, cfg.vocab_size + 1, (B, Lh)),
        "away_idx": torch.randint(1, cfg.vocab_size + 1, (B, La)),
        "home_mask": home_mask,
        "away_mask": away_mask,
        "home_prob": torch.full((B, Lh), 0.95) * home_mask.to(torch.float32),
        "away_prob": torch.full((B, La), 0.95) * away_mask.to(torch.float32),
        "home_stats": torch.randn(B, Lh, 10),
        "away_stats": torch.randn(B, La, 10),
        "home_team_idx": torch.randint(1, 31, (B,)),
        "away_team_idx": torch.randint(1, 31, (B,)),
        "home_rest": torch.rand(B),
        "away_rest": torch.rand(B),
        "tabular": torch.randn(B, 4),
        "season_phase": torch.randn(B, SEASON_PHASE_DIM),
        "label": torch.randint(0, 2, (B,), dtype=torch.float32),
        "margin": torch.randn(B) * 10.0,
        "team_box_home": torch.rand(B, K_BOX) * 100.0,
        "team_box_away": torch.rand(B, K_BOX) * 100.0,
        "sup_pair_game": torch.zeros(0, dtype=torch.long),
        "sup_pair_side": torch.zeros(0, dtype=torch.long),
        "sup_pair_off": torch.zeros(0, dtype=torch.long),
        "sup_pair_def": torch.zeros(0, dtype=torch.long),
        "sup_pair_y": torch.zeros(0, K_PAIR),
        "sup_pl_game": torch.zeros(0, dtype=torch.long),
        "sup_pl_side": torch.zeros(0, dtype=torch.long),
        "sup_pl_slot": torch.zeros(0, dtype=torch.long),
        "sup_pl_y": torch.zeros(0, K_BOX),
        "home_alpha_off_actual": rand_share(home_mask),
        "home_alpha_def_actual": rand_share(home_mask),
        "away_alpha_off_actual": rand_share(away_mask),
        "away_alpha_def_actual": rand_share(away_mask),
        "home_off_exposure_valid": torch.ones(B),
        "away_off_exposure_valid": torch.ones(B),
        "home_avail_features": torch.randn(B, Lh, AVAILABILITY_FEATURE_DIM),
        "away_avail_features": torch.randn(B, La, AVAILABILITY_FEATURE_DIM),
        "home_avail_minute_log_prior": torch.rand(B, Lh) * 8.0,
        "away_avail_minute_log_prior": torch.rand(B, La) * 8.0,
        "home_avail_play_actual": torch.randint(0, 2, (B, Lh), dtype=torch.float32) * home_mask.to(torch.float32),
        "away_avail_play_actual": torch.randint(0, 2, (B, La), dtype=torch.float32) * away_mask.to(torch.float32),
        "home_avail_log_seconds_actual": torch.rand(B, Lh) * 8.0 * home_mask.to(torch.float32),
        "away_avail_log_seconds_actual": torch.rand(B, La) * 8.0 * away_mask.to(torch.float32),
        "home_avail_role_share_actual": rand_share(home_mask),
        "away_avail_role_share_actual": rand_share(away_mask),
        "home_avail_role_valid": torch.ones(B),
        "away_avail_role_valid": torch.ones(B),
    }
    out = model(batch)
    box_weights = torch.ones(K_BOX)
    pair_weights = torch.ones(K_PAIR)
    loss, diag = total_loss_availability_v2(out, batch, box_weights=box_weights, pair_weights=pair_weights)
    loss.backward()
    assert torch.isfinite(loss)
    print("[smoke] ok", float(loss.detach()))


if __name__ == "__main__":
    _smoke()
