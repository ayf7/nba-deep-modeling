"""MAN-Xfmr (v4) model: cross-attention with Poisson-supervised prevalence.

For each game, compute pairwise prevalence λ(h, a) over the home offense ×
away defense grid (and the symmetric grid), gated by soft-mask play-probs.
Aggregate per-pair vector v_pair(h, a) weighted by λ to get a per-team
per-possession advantage vector. Concatenate both team vectors + rest days
into a head MLP for the win logit. λ is supervised against actual matchup
exposure_possessions via Poisson NLL on the training games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class XfmrConfig:
    vocab_size: int
    num_teams: int = 0
    d: int = 32
    pair_hidden: int = 64
    head_hidden: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    pair_dropout: float | None = None  # falls back to dropout if None
    player_dropout: float = 0.0        # train-time per-slot mask drop probability
    margin_head: bool = False
    tabular_dim: int = 0
    team_emb_dim: int = 0              # learned per-team identity vector; 0 = off
    player_stat_dim: int = 0           # per-player rolling form features; 0 = off
    two_stream: bool = False


class ManXfmr(nn.Module):
    def __init__(self, cfg: XfmrConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d

        if cfg.num_teams <= 0:
            raise ValueError("XfmrConfig.num_teams must be > 0 (set from team_vocab.size)")

        # Single player identity embedding. Index 0 reserved for OOV/padding.
        self.E_player = nn.Embedding(cfg.vocab_size + 1, d, padding_idx=0)
        self.b_off = nn.Embedding(cfg.vocab_size + 1, 1, padding_idx=0)
        self.b_def = nn.Embedding(cfg.vocab_size + 1, 1, padding_idx=0)

        # Team-conditioned Q/K projections. Index 0 reserved for OOV (won't normally be hit).
        self.W_q_team = nn.Embedding(cfg.num_teams + 1, d * d)
        self.W_k_team = nn.Embedding(cfg.num_teams + 1, d * d)

        # Optional learned team identity embedding fed into the head.
        if cfg.team_emb_dim > 0:
            self.E_team = nn.Embedding(cfg.num_teams + 1, cfg.team_emb_dim, padding_idx=0)

        # Optional per-player rolling form projection. Stats are added to the
        # player embedding so they participate in Q/K and pair_mlp paths.
        if cfg.player_stat_dim > 0:
            self.proj_player_stats = nn.Linear(cfg.player_stat_dim, d, bias=False)

        pair_drop = cfg.pair_dropout if cfg.pair_dropout is not None else cfg.dropout
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d, cfg.pair_hidden),
            nn.GELU(),
            nn.Dropout(pair_drop),
            nn.Linear(cfg.pair_hidden, d),
        )

        def _make_head(in_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = in_dim
            for h in cfg.head_hidden:
                layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(cfg.dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        team_extra = 2 * cfg.team_emb_dim  # home + away
        if cfg.two_stream:
            self.head_lineup = _make_head(2 * d + 2 + team_extra)
            if cfg.tabular_dim > 0:
                self.head_tab = _make_head(cfg.tabular_dim)
            else:
                self.head_tab = None  # type: ignore[assignment]
        else:
            head_in = 2 * d + 2 + cfg.tabular_dim + team_extra
            self.head = _make_head(head_in)

        if cfg.margin_head:
            self.margin_head = nn.Linear(d, 1, bias=True)

        # Init
        nn.init.normal_(self.E_player.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.E_player.weight[0].zero_()
        for emb in (self.b_off, self.b_def):
            nn.init.zeros_(emb.weight)
        if cfg.team_emb_dim > 0:
            nn.init.normal_(self.E_team.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.E_team.weight[0].zero_()
        if cfg.player_stat_dim > 0:
            # Zero-init so the model starts identical to the no-stats baseline,
            # then learns a residual from the stats.
            nn.init.zeros_(self.proj_player_stats.weight)
        # Init each team's Q/K projection to identity so the model starts equivalent
        # to a shared projection, then learns per-team deviations from there.
        with torch.no_grad():
            identity_flat = torch.eye(d).flatten()
            self.W_q_team.weight.copy_(
                identity_flat.unsqueeze(0).expand(cfg.num_teams + 1, -1)
            )
            self.W_k_team.weight.copy_(
                identity_flat.unsqueeze(0).expand(cfg.num_teams + 1, -1)
            )

    def _pairwise_lambda(
        self,
        idx_off: torch.Tensor,    # (B, L_o) offensive players' player_ids
        idx_def: torch.Tensor,    # (B, L_d) defensive players' player_ids
        mask_off: torch.Tensor,   # (B, L_o) bool
        mask_def: torch.Tensor,   # (B, L_d) bool
        prob_off: torch.Tensor,   # (B, L_o)
        prob_def: torch.Tensor,   # (B, L_d)
        team_off: torch.Tensor,   # (B,) team_idx of the offensive side
        team_def: torch.Tensor,   # (B,) team_idx of the defensive side
        stats_off: torch.Tensor | None = None,  # (B, L_o, S) standardized stats
        stats_def: torch.Tensor | None = None,  # (B, L_d, S)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lam, v_pair). lam: (B, L_o, L_d), v_pair: (B, L_o, L_d, d)."""
        d = self.cfg.d
        e_off = self.E_player(idx_off)           # (B, L_o, d)
        e_def = self.E_player(idx_def)           # (B, L_d, d)
        if self.cfg.player_stat_dim > 0 and stats_off is not None:
            e_off = e_off + self.proj_player_stats(stats_off)
            e_def = e_def + self.proj_player_stats(stats_def)

        W_q = self.W_q_team(team_off).view(-1, d, d)  # (B, d, d)
        W_k = self.W_k_team(team_def).view(-1, d, d)  # (B, d, d)
        q = torch.einsum("bld,bde->ble", e_off, W_q)  # (B, L_o, d)
        k = torch.einsum("bld,bde->ble", e_def, W_k)  # (B, L_d, d)

        scale = 1.0 / math.sqrt(d)
        score = torch.einsum("bid,bjd->bij", q, k) * scale  # (B, L_o, L_d)
        score = score + self.b_off(idx_off).squeeze(-1).unsqueeze(2)
        score = score + self.b_def(idx_def).squeeze(-1).unsqueeze(1)

        mask_outer = (mask_off.unsqueeze(2) & mask_def.unsqueeze(1)).float()
        prob_outer = prob_off.unsqueeze(2) * prob_def.unsqueeze(1)

        # Clamp for numerical safety; mask is the dominant zero-out mechanism.
        score = torch.clamp(score, min=-12.0, max=12.0)
        lam = torch.exp(score) * prob_outer * mask_outer

        # Pair value vector
        L_o, L_d = idx_off.size(1), idx_def.size(1)
        e_off_exp = e_off.unsqueeze(2).expand(-1, -1, L_d, -1)
        e_def_exp = e_def.unsqueeze(1).expand(-1, L_o, -1, -1)
        pair_in = torch.cat([e_off_exp, e_def_exp], dim=-1)
        v_pair = self.pair_mlp(pair_in)  # (B, L_o, L_d, d)

        return lam, v_pair

    def forward(
        self,
        batch: dict,
        *,
        zero_lineup: bool = False,
        zero_tabular: bool = False,
    ) -> dict:
        home_mask = batch["home_mask"]
        away_mask = batch["away_mask"]
        if self.training and self.cfg.player_dropout > 0.0:
            p = self.cfg.player_dropout
            keep_h = torch.rand(home_mask.shape, device=home_mask.device) > p
            keep_a = torch.rand(away_mask.shape, device=away_mask.device) > p
            home_mask = home_mask & keep_h
            away_mask = away_mask & keep_a

        home_team = batch["home_team_idx"]
        away_team = batch["away_team_idx"]
        home_stats = batch.get("home_stats")
        away_stats = batch.get("away_stats")

        # Direction 1: home offense × away defense
        lam_h, v_h = self._pairwise_lambda(
            batch["home_idx"], batch["away_idx"],
            home_mask, away_mask,
            batch["home_prob"], batch["away_prob"],
            team_off=home_team, team_def=away_team,
            stats_off=home_stats, stats_def=away_stats,
        )
        # Direction 2: away offense × home defense
        lam_a, v_a = self._pairwise_lambda(
            batch["away_idx"], batch["home_idx"],
            away_mask, home_mask,
            batch["away_prob"], batch["home_prob"],
            team_off=away_team, team_def=home_team,
            stats_off=away_stats, stats_def=home_stats,
        )

        # Aggregate to team vectors
        eps = 1e-6
        z_home_off = (lam_h.unsqueeze(-1) * v_h).sum(dim=(1, 2))
        z_home_off = z_home_off / (lam_h.sum(dim=(1, 2)).unsqueeze(-1) + eps)
        z_away_off = (lam_a.unsqueeze(-1) * v_a).sum(dim=(1, 2))
        z_away_off = z_away_off / (lam_a.sum(dim=(1, 2)).unsqueeze(-1) + eps)

        if zero_lineup:
            z_home_off = torch.zeros_like(z_home_off)
            z_away_off = torch.zeros_like(z_away_off)

        rest_h = batch["home_rest"].unsqueeze(-1)
        rest_a = batch["away_rest"].unsqueeze(-1)
        if self.cfg.team_emb_dim > 0:
            e_team_h = self.E_team(home_team)
            e_team_a = self.E_team(away_team)
        if self.cfg.two_stream:
            lin_parts = [z_home_off, z_away_off, rest_h, rest_a]
            if self.cfg.team_emb_dim > 0:
                lin_parts += [e_team_h, e_team_a]
            lin_in = torch.cat(lin_parts, dim=-1)
            logit_lineup = self.head_lineup(lin_in).squeeze(-1)
            if self.cfg.tabular_dim > 0 and self.head_tab is not None:
                tab = batch["tabular"]
                if zero_tabular:
                    tab = torch.zeros_like(tab)
                logit_tab = self.head_tab(tab).squeeze(-1)
            else:
                logit_tab = torch.zeros_like(logit_lineup)
            win_logit = logit_lineup + logit_tab
        else:
            parts = [z_home_off, z_away_off, rest_h, rest_a]
            if self.cfg.tabular_dim > 0:
                tab = batch["tabular"]
                if zero_tabular:
                    tab = torch.zeros_like(tab)
                parts.append(tab)
            if self.cfg.team_emb_dim > 0:
                parts += [e_team_h, e_team_a]
            feat = torch.cat(parts, dim=-1)
            win_logit = self.head(feat).squeeze(-1)

        out = {
            "win_logit": win_logit,
            "lam_h": lam_h,    # (B, L_h, L_a)
            "lam_a": lam_a,    # (B, L_a, L_h)
            "z_home_off": z_home_off,
            "z_away_off": z_away_off,
        }
        if self.cfg.margin_head:
            out["margin_pred"] = self.margin_head(z_home_off - z_away_off).squeeze(-1)
        return out


def gather_lambda_for_supervision(
    lam_h: torch.Tensor,     # (B, L_h, L_a)
    lam_a: torch.Tensor,     # (B, L_a, L_h)
    sup_game: torch.Tensor,  # (N,)
    sup_side: torch.Tensor,  # (N,) 0=home_off, 1=away_off
    sup_off: torch.Tensor,   # (N,) slot index in offense roster
    sup_def: torch.Tensor,   # (N,) slot index in defense roster
) -> torch.Tensor:
    """Gather predicted λ for each supervision row. Returns (N,)."""
    if sup_game.numel() == 0:
        return torch.zeros(0, device=lam_h.device, dtype=lam_h.dtype)
    is_home = sup_side == 0
    out = torch.empty(sup_game.numel(), device=lam_h.device, dtype=lam_h.dtype)
    if is_home.any():
        idx = is_home.nonzero(as_tuple=True)[0]
        out[idx] = lam_h[sup_game[idx], sup_off[idx], sup_def[idx]]
    if (~is_home).any():
        idx = (~is_home).nonzero(as_tuple=True)[0]
        out[idx] = lam_a[sup_game[idx], sup_off[idx], sup_def[idx]]
    return out


def poisson_nll(lam_pred: torch.Tensor, exposure: torch.Tensor) -> torch.Tensor:
    """λ - y log λ (constant log y! dropped). Mean over rows."""
    if lam_pred.numel() == 0:
        return torch.zeros((), device=lam_pred.device, dtype=lam_pred.dtype)
    eps = 1e-6
    return (lam_pred - exposure * torch.log(lam_pred + eps)).mean()
