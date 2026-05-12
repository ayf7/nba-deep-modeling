"""Process-CME v1 model.

Process-CME v1 is a deliberately radical-but-contained extension of CME-v4.2.
It preserves the best current structured matchup architecture and calibration
stack, then adds a *process pathway*:

  * predict standardized pregame game-process targets for both teams;
  * use the learned process state as an additional tail-gated win-logit
    residual.

Process targets are derived from pbpstats possession rows when available and
optionally approximated from core event/shot tables where pbpstats coverage is
missing.  They represent game style rather than final points:

  possessions, 2PA/poss, 3PA/poss, TOV/poss, OREB/poss, shooting fouls/poss

for the home and away side.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V42_SCRIPTS = PROJECT_ROOT / "models_cme_v4_2" / "scripts"
PROCESS_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V42_SCRIPTS))
sys.path.insert(0, str(PROCESS_SCRIPTS))

from cme_v4_2_model import (  # noqa: E402
    CmeV42,
    CmeV42Config,
    _zero_last_linear,
    total_loss as total_loss_v42,
)
from cme_process_v1_common import K_PROCESS, SEASON_PHASE_DIM  # noqa: E402


@dataclass
class CmeProcessV1Config(CmeV42Config):
    process_target_dim: int = K_PROCESS
    process_hidden: int = 96
    process_dropout: float = 0.05
    max_process_logit_residual: float = 0.25


class CmeProcessV1(CmeV42):
    def __init__(self, cfg: CmeProcessV1Config) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        d = cfg.d
        process_in = 4 * d + 1
        self.process_state_head = nn.Sequential(
            nn.Linear(process_in, cfg.process_hidden),
            nn.GELU(),
            nn.Dropout(cfg.process_dropout),
            nn.Linear(cfg.process_hidden, cfg.process_hidden),
            nn.GELU(),
        )
        self.process_target_head = nn.Linear(cfg.process_hidden, cfg.process_target_dim)
        _zero_last_linear(self.process_target_head)  # neutral z-score predictions at init
        process_residual_in = cfg.process_hidden + cfg.process_target_dim + 1
        self.process_win_head = nn.Sequential(
            nn.Linear(process_residual_in, cfg.process_hidden),
            nn.GELU(),
            nn.Dropout(cfg.process_dropout),
            nn.Linear(cfg.process_hidden, 1),
        )
        _zero_last_linear(self.process_win_head)

    def forward(self, batch: dict) -> dict:
        out = super().forward(batch)
        process_inputs = torch.cat(
            [
                out["home_ctx"],
                out["away_ctx"],
                out["home_ctx"] - out["away_ctx"],
                out["home_ctx"] * out["away_ctx"],
                out["win_logit_temporal"].unsqueeze(-1),
            ],
            dim=-1,
        )
        process_state = self.process_state_head(process_inputs)
        process_pred_z = self.process_target_head(process_state)
        process_residual_inputs = torch.cat(
            [process_state, process_pred_z, out["win_logit_temporal"].unsqueeze(-1)],
            dim=-1,
        )
        raw_process_logit_residual = self.process_win_head(process_residual_inputs).squeeze(-1)
        if self.cfg.max_process_logit_residual > 0.0:
            bounded_process_logit_residual = (
                self.cfg.max_process_logit_residual * torch.tanh(raw_process_logit_residual)
            )
        else:
            bounded_process_logit_residual = torch.zeros_like(raw_process_logit_residual)
        process_logit_residual = out["residual_tail_gate"] * bounded_process_logit_residual
        out["win_logit_v42"] = out["win_logit"]
        out["win_logit"] = out["win_logit"] + process_logit_residual
        out["process_state"] = process_state
        out["process_pred_z"] = process_pred_z
        out["raw_process_logit_residual"] = raw_process_logit_residual
        out["bounded_process_logit_residual"] = bounded_process_logit_residual
        out["process_logit_residual"] = process_logit_residual
        return out


def process_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    pred = out["process_pred_z"]
    target = batch["process_target"].to(pred)
    valid = batch["process_target_valid"].to(pred).view(-1, 1)
    if float(valid.sum().detach().cpu().item()) <= 0.0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    sq = (pred - target).pow(2)
    # Mean over target dimensions, then mean over valid games.
    per_game = sq.mean(dim=-1, keepdim=True)
    return (per_game * valid).sum() / valid.sum().clamp_min(1.0)


def process_residual_l2_loss(out: dict) -> torch.Tensor:
    return out["process_logit_residual"].pow(2).mean()


def total_loss_process_v1(
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
    process_w: float = 0.25,
    process_residual_reg_w: float = 0.05,
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
    L_process = process_mse_loss(out, batch)
    L_process_residual_reg = process_residual_l2_loss(out)
    total = base_total + process_w * L_process + process_residual_reg_w * L_process_residual_reg
    diag = dict(diag)
    diag["L_process"] = L_process.detach()
    diag["L_process_residual_reg"] = L_process_residual_reg.detach()
    return total, diag


def _smoke() -> None:
    torch.manual_seed(0)
    cfg = CmeProcessV1Config(
        vocab_size=200,
        num_teams=30,
        d=64,
        tabular_dim=4,
        team_emb_dim=8,
        player_stat_dim=10,
    )
    model = CmeProcessV1(cfg)
    B, Lh, La = 4, 14, 16
    home_mask = torch.ones(B, Lh, dtype=torch.bool)
    away_mask = torch.ones(B, La, dtype=torch.bool)
    home_mask[0, 12:] = False
    away_mask[1, 14:] = False

    def _rand_share(mask: torch.Tensor) -> torch.Tensor:
        x = torch.rand_like(mask, dtype=torch.float32) * mask.to(torch.float32)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    batch = {
        "home_idx": torch.randint(1, cfg.vocab_size + 1, (B, Lh)),
        "away_idx": torch.randint(1, cfg.vocab_size + 1, (B, La)),
        "home_prob": torch.rand(B, Lh) * home_mask.to(torch.float32),
        "away_prob": torch.rand(B, La) * away_mask.to(torch.float32),
        "home_mask": home_mask,
        "away_mask": away_mask,
        "home_stats": torch.randn(B, Lh, cfg.player_stat_dim),
        "away_stats": torch.randn(B, La, cfg.player_stat_dim),
        "home_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "away_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "home_rest": torch.rand(B),
        "away_rest": torch.rand(B),
        "tabular": torch.randn(B, cfg.tabular_dim),
        "season_phase": torch.randn(B, SEASON_PHASE_DIM).clamp(-1.0, 1.0),
        "label": torch.randint(0, 2, (B,), dtype=torch.float32),
        "margin": torch.randn(B) * 10,
        "team_box_home": torch.rand(B, K_PROCESS + 2)[:, :14] if False else torch.rand(B, 14) * 30,
        "team_box_away": torch.rand(B, 14) * 30,
        "sup_pair_game": torch.tensor([0, 0, 1, 2, 3]),
        "sup_pair_side": torch.tensor([0, 1, 0, 0, 1]),
        "sup_pair_off": torch.tensor([0, 0, 1, 2, 3]),
        "sup_pair_def": torch.tensor([0, 1, 2, 3, 4]),
        "sup_pair_y": torch.rand(5, 9) * 3.0,
        "sup_pl_game": torch.tensor([0, 0, 1, 2, 3, 3]),
        "sup_pl_side": torch.tensor([0, 1, 0, 1, 0, 1]),
        "sup_pl_slot": torch.tensor([0, 1, 2, 3, 4, 5]),
        "sup_pl_y": torch.rand(6, 14) * 5.0,
        "home_alpha_off_actual": _rand_share(home_mask),
        "home_alpha_def_actual": _rand_share(home_mask),
        "away_alpha_off_actual": _rand_share(away_mask),
        "away_alpha_def_actual": _rand_share(away_mask),
        "home_off_exposure_valid": torch.ones(B, dtype=torch.float32),
        "away_off_exposure_valid": torch.ones(B, dtype=torch.float32),
        "process_target": torch.randn(B, K_PROCESS),
        "process_target_valid": torch.tensor([1, 1, 0, 1], dtype=torch.float32),
    }
    out = model(batch)
    box_w = torch.ones(14)
    pair_w = torch.ones(9)
    loss, diag = total_loss_process_v1(
        out, batch, box_weights=box_w, pair_weights=pair_w, margin_nll_w=0.1,
    )
    loss.backward()
    print("loss", float(loss.detach()))
    print("process_pred_z", tuple(out["process_pred_z"].shape))
    print("L_process", float(diag["L_process"]))
    print("backward ok")


if __name__ == "__main__":
    _smoke()
