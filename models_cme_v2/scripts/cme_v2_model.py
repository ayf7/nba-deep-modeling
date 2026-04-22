"""CME-v2 model (hierarchical-marginal version).

Pipeline (per game):
  1. Player tokenizer: e_id + form-features MLP, gated by play probability.
  2. Team + tabular context infusion: add (team_emb + tabular + rest) per player.
  3. Per-team self-attention (params shared across teams).
  4. Role projections: off_emb / def_emb = OffProj(self), DefProj(self).
  5. Directional cross-attention (params shared across directions). Direction A:
     home.off attends to away.def. Direction B: away.off attends to home.def.
  6. Head A — per-pair (K_PAIR=9 channels): for each (i,j) in each direction,
       score_ij = (W_q · off_i) · (W_k · def_j) / sqrt(d)
       λ_k_ij = exp(MLP_k([off_i; def_j; score_ij]) + bias_k)
  7. Head B — per-player (K_PLAYER=6 channels): non-pair stats (FTM/FTA/OREB/
     DREB/STL/PF) where there's no real defender attribution.

The natural output is a 14-d per-player box-score vector built by combining
the pair-grid row sums with the player head outputs. Team totals = sum over
players. The model is supervised on three levels of marginals:

    L_total = L_team + 0.01 · L_player + 0.001 · L_pair + win_w · L_win

where:
  * L_team:   per-stat MSE on team totals (home + away averaged).
  * L_player: per-stat MSE on the 14-d per-player BOX vector.
  * L_pair:   per-pair Poisson NLL on the 9 pair channels (kept here because
              pair counts are 0–3 — the regime Poisson is actually designed
              for).
  * L_win:    BCE on win_logit = (home_pts - away_pts) / global_scale.
              Provides direct gradient signal on the margin, which the
              team/player MSE losses do not — they treat home and away
              independently, so the model can satisfy them by predicting
              league-average pace without ever differentiating teams.

Why MSE on team / player and not Poisson? Poisson NLL has curvature 1/μ,
so at team-points scale (μ ≈ 110) the gradient signal is ~100× weaker than
at pair scale (μ ≈ 1). MSE is fully quadratic, so it provides strong
pull-back when predictions wander away from reality at any scale; the
exp() parameterization on the pair grid is what was previously letting
predictions explode under Huber's bounded gradient.

Win/margin emerge at inference: margin = home_team_pts - away_team_pts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cme_v2_common import (
    BOX_INDEX,
    BOX_TARGETS,
    K_BOX,
    K_PAIR,
    K_PLAYER,
    PAIR_TARGETS,
    PLAYER_TARGETS,
    _PAIR_3PA_IDX,
    _PAIR_3PM_IDX,
    _PAIR_AST_IDX,
    _PAIR_BLK_IDX,
    _PAIR_FGA_IDX,
    _PAIR_FGM_IDX,
    _PAIR_PTS_IDX,
    _PAIR_TOV_IDX,
    _PLAYER_DREB_IDX,
    _PLAYER_FTA_IDX,
    _PLAYER_FTM_IDX,
    _PLAYER_OREB_IDX,
    _PLAYER_PF_IDX,
    _PLAYER_STL_IDX,
)


# Mapping: BOX target index -> ("pair", pair_channel_idx) or ("player", player_channel_idx).
# Used to assemble the per-player BOX prediction from the two heads.
# PTS is special: handled separately as 2*FGM_pair + 3PM_pair + FTM_player by
# the helper below — but we also let the pair grid track FG points directly via
# `player_points`. We use the pair grid's player_points marginal directly + FTM
# from the player head, since that's what we have per-pair labels for.
_BOX_SOURCE: dict[str, tuple[str, int]] = {
    "fgm":  ("pair", _PAIR_FGM_IDX),
    "fga":  ("pair", _PAIR_FGA_IDX),
    "3pm":  ("pair", _PAIR_3PM_IDX),
    "3pa":  ("pair", _PAIR_3PA_IDX),
    "ast":  ("pair", _PAIR_AST_IDX),
    "tov":  ("pair", _PAIR_TOV_IDX),
    "blk":  ("pair", _PAIR_BLK_IDX),
    "ftm":  ("player", _PLAYER_FTM_IDX),
    "fta":  ("player", _PLAYER_FTA_IDX),
    "oreb": ("player", _PLAYER_OREB_IDX),
    "dreb": ("player", _PLAYER_DREB_IDX),
    "stl":  ("player", _PLAYER_STL_IDX),
    "pf":   ("player", _PLAYER_PF_IDX),
}


@dataclass
class CmeV2Config:
    vocab_size: int
    num_teams: int
    d: int = 64
    n_heads: int = 4
    n_self_layers: int = 2
    n_cross_layers: int = 2
    pair_hidden: int = 96
    player_hidden: int = 64
    dropout: float = 0.1
    pair_dropout: float = 0.2
    player_dropout: float = 0.0
    tabular_dim: int = 0
    team_emb_dim: int = 16
    player_stat_dim: int = 0
    init_scale: float = 12.0
    score_clamp: float = 12.0
    use_direct_win_head: bool = False
    use_team_token: bool = False
    use_decision_head: bool = False


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: (B, L, d), mask: (B, L) bool. Returns (B, d) masked mean over L."""
    m = mask.unsqueeze(-1).to(x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)  # (B, 1)
    return (x * m).sum(dim=1) / denom


def _zero_last_linear(seq: nn.Sequential) -> None:
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return


class SelfBlock(nn.Module):
    """Pre-LN MHA + FFN, padding-masked, in-place residual."""
    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d),
        )
        _zero_last_linear(self.ffn)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


class CrossBlock(nn.Module):
    """Pre-LN cross-attention + FFN. Q is updated; K/V comes from the other team."""
    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d),
        )
        _zero_last_linear(self.ffn)

    def forward(
        self, q: torch.Tensor, kv: torch.Tensor,
        q_mask: torch.Tensor, kv_mask: torch.Tensor,
    ) -> torch.Tensor:
        h_q = self.ln_q(q)
        h_kv = self.ln_kv(kv)
        attn_out, _ = self.attn(h_q, h_kv, h_kv, key_padding_mask=~kv_mask, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.ln2(q))
        return q * q_mask.unsqueeze(-1).to(q.dtype)


class CmeV2(nn.Module):
    def __init__(self, cfg: CmeV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d

        self.E_player = nn.Embedding(cfg.vocab_size + 1, d, padding_idx=0)
        nn.init.normal_(self.E_player.weight, std=0.02)
        with torch.no_grad():
            self.E_player.weight[0].zero_()

        self.E_team = nn.Embedding(cfg.num_teams + 1, cfg.team_emb_dim, padding_idx=0)
        nn.init.normal_(self.E_team.weight, std=0.02)
        with torch.no_grad():
            self.E_team.weight[0].zero_()

        if cfg.player_stat_dim > 0:
            self.feat_mlp = nn.Sequential(
                nn.Linear(cfg.player_stat_dim, d), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(d, d),
            )
            _zero_last_linear(self.feat_mlp)
        else:
            self.feat_mlp = None

        ctx_in = cfg.team_emb_dim + cfg.tabular_dim + 1
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_in, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
        )
        _zero_last_linear(self.ctx_proj)

        self.self_layers = nn.ModuleList([
            SelfBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_self_layers)
        ])

        self.off_proj = nn.Linear(d, d)
        self.def_proj = nn.Linear(d, d)
        self.cross_layers = nn.ModuleList([
            CrossBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_cross_layers)
        ])

        # Head A: per-pair (K_PAIR=9). Score head + MLP on [off; def; score].
        self.score_q = nn.Linear(d, d)
        self.score_k = nn.Linear(d, d)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d + 1, cfg.pair_hidden), nn.GELU(), nn.Dropout(cfg.pair_dropout),
            nn.Linear(cfg.pair_hidden, K_PAIR),
        )
        _zero_last_linear(self.pair_mlp)
        self.pair_bias = nn.Parameter(torch.zeros(K_PAIR))

        # Head B: per-player (K_PLAYER=6).
        self.player_mlp = nn.Sequential(
            nn.Linear(d, cfg.player_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.player_hidden, K_PLAYER),
        )
        _zero_last_linear(self.player_mlp)
        self.player_bias = nn.Parameter(torch.zeros(K_PLAYER))

        # Margin → win logit scale. Frozen so the win-BCE term has to reduce
        # loss by pushing margins apart, not by shrinking scale.
        self.global_scale = nn.Parameter(
            torch.tensor(math.log(cfg.init_scale)), requires_grad=False,
        )

        # Heteroscedastic margin std head. Inputs: [home_ctx, away_ctx] (2d).
        # Output: log σ for Gaussian NLL on margin. Bias init at log(13) (NBA prior).
        self.margin_log_sigma_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )
        with torch.no_grad():
            self.margin_log_sigma_head[-1].weight.zero_()
            self.margin_log_sigma_head[-1].bias.fill_(math.log(13.0))

        # Direct win head — backbone probe. Inputs are
        # [pool(home_off), pool(away_off), home_ctx, away_ctx] (4d). Bypasses
        # the box/margin chain; used when cfg.use_direct_win_head.
        self.direct_win_head = nn.Sequential(
            nn.Linear(4 * d, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )
        _zero_last_linear(self.direct_win_head)

        if cfg.use_team_token:
            self.team_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        else:
            self.team_token = None

        if cfg.use_decision_head:
            if not cfg.use_team_token:
                raise ValueError("use_decision_head requires use_team_token=True")
            self.decision_head = nn.Sequential(
                nn.Linear(2 * d + 3, d), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(d, 1),
            )
            _zero_last_linear(self.decision_head)
        else:
            self.decision_head = None

    def _tokenize(self, idx: torch.Tensor, stats: torch.Tensor, prob: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        e = self.E_player(idx)
        if self.feat_mlp is not None and stats.size(-1) > 0:
            e = e + self.feat_mlp(stats)
        e = e * prob.unsqueeze(-1)
        e = e * mask.unsqueeze(-1).to(e.dtype)
        return e

    def _team_ctx(self, team_idx: torch.Tensor, tabular: torch.Tensor | None,
                  rest: torch.Tensor) -> torch.Tensor:
        parts = [self.E_team(team_idx)]
        if tabular is not None and tabular.size(-1) > 0:
            parts.append(tabular)
        parts.append(rest.unsqueeze(-1))
        x = torch.cat(parts, dim=-1)
        return self.ctx_proj(x)

    def _pair_rates(self, off: torch.Tensor, def_: torch.Tensor,
                    off_mask: torch.Tensor, def_mask: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (rates [B,Lo,Ld,K_PAIR], pair_mask [B,Lo,Ld] bool)."""
        B, Lo, d = off.shape
        Ld = def_.size(1)

        Q = self.score_q(off)
        K = self.score_k(def_)
        score = torch.einsum("bld,bmd->blm", Q, K) / math.sqrt(d)
        score = torch.clamp(score, min=-self.cfg.score_clamp, max=self.cfg.score_clamp)

        Qe = off.unsqueeze(2).expand(-1, -1, Ld, -1)
        Ke = def_.unsqueeze(1).expand(-1, Lo, -1, -1)
        pair_in = torch.cat([Qe, Ke, score.unsqueeze(-1)], dim=-1)
        raw = self.pair_mlp(pair_in) + self.pair_bias
        raw = torch.clamp(raw, min=-12.0, max=8.0)
        rates = torch.exp(raw)

        pair_mask = off_mask.unsqueeze(-1) & def_mask.unsqueeze(1)
        rates = rates * pair_mask.unsqueeze(-1).to(rates.dtype)
        return rates, pair_mask

    def _player_rates(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = self.player_mlp(emb) + self.player_bias
        raw = torch.clamp(raw, min=-12.0, max=8.0)
        rates = torch.exp(raw)
        rates = rates * mask.unsqueeze(-1).to(rates.dtype)
        return rates

    def forward(self, batch: dict) -> dict:
        home_mask = batch["home_mask"]
        away_mask = batch["away_mask"]
        if self.training and self.cfg.player_dropout > 0.0:
            p = self.cfg.player_dropout
            home_mask = home_mask & (torch.rand_like(home_mask, dtype=torch.float32) > p)
            away_mask = away_mask & (torch.rand_like(away_mask, dtype=torch.float32) > p)

        tabular = batch.get("tabular")

        home_tok = self._tokenize(batch["home_idx"], batch["home_stats"],
                                  batch["home_prob"], home_mask)
        away_tok = self._tokenize(batch["away_idx"], batch["away_stats"],
                                  batch["away_prob"], away_mask)

        home_ctx = self._team_ctx(batch["home_team_idx"], tabular, batch["home_rest"])
        away_ctx = self._team_ctx(batch["away_team_idx"], tabular, batch["away_rest"])
        home_tok = home_tok + home_ctx.unsqueeze(1) * home_mask.unsqueeze(-1).to(home_tok.dtype)
        away_tok = away_tok + away_ctx.unsqueeze(1) * away_mask.unsqueeze(-1).to(away_tok.dtype)

        # Optionally prepend a learnable team token (CLS-style). It rides
        # through self- and cross-attention alongside players and is later
        # extracted as a per-team summary for the decision head.
        if self.team_token is not None:
            B_ = home_tok.size(0)
            home_tt = self.team_token.expand(B_, 1, -1).to(home_tok.dtype)
            away_tt = self.team_token.expand(B_, 1, -1).to(away_tok.dtype)
            home_tok = torch.cat([home_tt, home_tok], dim=1)
            away_tok = torch.cat([away_tt, away_tok], dim=1)
            ones_h = torch.ones(B_, 1, dtype=torch.bool, device=home_mask.device)
            ones_a = torch.ones(B_, 1, dtype=torch.bool, device=away_mask.device)
            home_seq_mask = torch.cat([ones_h, home_mask], dim=1)
            away_seq_mask = torch.cat([ones_a, away_mask], dim=1)
        else:
            home_seq_mask = home_mask
            away_seq_mask = away_mask

        home_self = home_tok
        away_self = away_tok
        for layer in self.self_layers:
            home_self = layer(home_self, home_seq_mask)
            away_self = layer(away_self, away_seq_mask)

        home_off = self.off_proj(home_self) * home_seq_mask.unsqueeze(-1).to(home_self.dtype)
        home_def = self.def_proj(home_self) * home_seq_mask.unsqueeze(-1).to(home_self.dtype)
        away_off = self.off_proj(away_self) * away_seq_mask.unsqueeze(-1).to(away_self.dtype)
        away_def = self.def_proj(away_self) * away_seq_mask.unsqueeze(-1).to(away_self.dtype)

        for layer in self.cross_layers:
            home_off = layer(home_off, away_def, q_mask=home_seq_mask, kv_mask=away_seq_mask)
            away_off = layer(away_off, home_def, q_mask=away_seq_mask, kv_mask=home_seq_mask)

        # Split off team-token slot (position 0) from player slots for the
        # per-player / per-pair heads.
        if self.team_token is not None:
            home_team_emb = home_off[:, 0]
            away_team_emb = away_off[:, 0]
            home_self_p = home_self[:, 1:]
            away_self_p = away_self[:, 1:]
            home_off = home_off[:, 1:]
            home_def = home_def[:, 1:]
            away_off = away_off[:, 1:]
            away_def = away_def[:, 1:]
        else:
            home_team_emb = None
            away_team_emb = None
            home_self_p = home_self
            away_self_p = away_self

        # Head A — per-pair, both directions
        rates_A, mask_A = self._pair_rates(home_off, away_def, home_mask, away_mask)
        rates_B, mask_B = self._pair_rates(away_off, home_def, away_mask, home_mask)

        # Head B — per-player rates
        home_player_rates = self._player_rates(home_self_p, home_mask)
        away_player_rates = self._player_rates(away_self_p, away_mask)

        # ---- per-player BOX (14-d) prediction by row-sum + player head ----
        # Pair direction A: home offense, axis (B, Lh, La, K_PAIR). Row-sum over La
        # gives per-home-player offensive marginals. Direction B: away offense.
        home_pair_marginal = rates_A.sum(dim=2)                  # (B, Lh, K_PAIR)
        away_pair_marginal = rates_B.sum(dim=2)                  # (B, La, K_PAIR)

        home_box = self._assemble_player_box(home_pair_marginal, home_player_rates)
        away_box = self._assemble_player_box(away_pair_marginal, away_player_rates)
        # Mask padding slots so team-sum is correct
        home_box = home_box * home_mask.unsqueeze(-1).to(home_box.dtype)
        away_box = away_box * away_mask.unsqueeze(-1).to(away_box.dtype)

        # ---- team totals (B, K_BOX) ----
        home_team_box = home_box.sum(dim=1)
        away_team_box = away_box.sum(dim=1)

        # ---- inference: margin / win logit using total points ----
        pts_idx = BOX_INDEX["pts"]
        home_points = home_team_box[..., pts_idx]
        away_points = away_team_box[..., pts_idx]
        margin_mu = home_points - away_points
        scale = self.global_scale.exp().clamp_min(1.0)
        win_logit = margin_mu / scale

        game_repr = torch.cat([home_ctx, away_ctx], dim=-1)  # (B, 2d)
        margin_log_sigma = self.margin_log_sigma_head(game_repr).squeeze(-1)  # (B,)

        if self.cfg.use_direct_win_head:
            if self.team_token is not None:
                home_summary = home_team_emb
                away_summary = away_team_emb
            else:
                home_summary = _masked_mean(home_off, home_mask)
                away_summary = _masked_mean(away_off, away_mask)
            direct_in = torch.cat([home_summary, away_summary, home_ctx, away_ctx], dim=-1)  # (B, 4d)
            win_logit = self.direct_win_head(direct_in).squeeze(-1)  # (B,)

        decision_logit = None
        if self.decision_head is not None:
            bet_home_flag = (win_logit.detach() >= 0.0).to(home_team_emb.dtype)
            h_dec = batch["home_dec_odds"].to(home_team_emb.dtype)
            a_dec = batch["away_dec_odds"].to(home_team_emb.dtype)
            dec_in = torch.cat([
                home_team_emb, away_team_emb,
                h_dec.unsqueeze(-1), a_dec.unsqueeze(-1),
                bet_home_flag.unsqueeze(-1),
            ], dim=-1)
            decision_logit = self.decision_head(dec_in).squeeze(-1)

        return {
            "pair_dir_A": rates_A,
            "pair_dir_B": rates_B,
            "pair_mask_A": mask_A,
            "pair_mask_B": mask_B,
            "home_player_rates": home_player_rates,
            "away_player_rates": away_player_rates,
            "home_pair_marginal": home_pair_marginal,
            "away_pair_marginal": away_pair_marginal,
            "home_box": home_box,                     # (B, Lh, K_BOX)
            "away_box": away_box,                     # (B, La, K_BOX)
            "home_team_box": home_team_box,           # (B, K_BOX)
            "away_team_box": away_team_box,           # (B, K_BOX)
            "home_points": home_points,
            "away_points": away_points,
            "margin_mu": margin_mu,
            "margin_log_sigma": margin_log_sigma,
            "win_logit": win_logit,
            "global_scale": scale,
            "home_team_emb": home_team_emb,
            "away_team_emb": away_team_emb,
            "decision_logit": decision_logit,
        }

    def _assemble_player_box(
        self, pair_marginal: torch.Tensor, player_rates: torch.Tensor,
    ) -> torch.Tensor:
        """Build (B, L, K_BOX) per-player predicted box-score vector.

          * fgm/fga/3pm/3pa/ast/tov/blk: row-sum of pair grid
          * ftm/fta/oreb/dreb/stl/pf:    direct from player head
          * pts: 2*fgm + 3pm + ftm  (matches NBA scoring formula)
        """
        B, L, _ = pair_marginal.shape
        device = pair_marginal.device
        dtype = pair_marginal.dtype
        out = torch.zeros(B, L, K_BOX, device=device, dtype=dtype)
        for box_name, (src, ch) in _BOX_SOURCE.items():
            i = BOX_INDEX[box_name]
            if src == "pair":
                out[..., i] = pair_marginal[..., ch]
            else:
                out[..., i] = player_rates[..., ch]
        # PTS = 2 * FGM + 3PM + FTM (NBA scoring)
        out[..., BOX_INDEX["pts"]] = (
            2.0 * pair_marginal[..., _PAIR_FGM_IDX]
            + pair_marginal[..., _PAIR_3PM_IDX]
            + player_rates[..., _PLAYER_FTM_IDX]
        )
        return out


# ---------------------------- losses ----------------------------


def gather_pair_pred(out: dict, sup_game: torch.Tensor, sup_side: torch.Tensor,
                     sup_off: torch.Tensor, sup_def: torch.Tensor) -> torch.Tensor:
    """Returns (M, K_PAIR) predicted rates at the supervised (game, side, off, def) indices."""
    A = out["pair_dir_A"]
    B = out["pair_dir_B"]
    M = sup_game.size(0)
    K = A.size(-1)
    pred = torch.zeros(M, K, device=A.device, dtype=A.dtype)
    if M == 0:
        return pred
    h = sup_side == 0
    a = sup_side == 1
    if h.any():
        idx = h.nonzero(as_tuple=False).squeeze(-1)
        pred[idx] = A[sup_game[idx], sup_off[idx], sup_def[idx]]
    if a.any():
        idx = a.nonzero(as_tuple=False).squeeze(-1)
        pred[idx] = B[sup_game[idx], sup_off[idx], sup_def[idx]]
    return pred


def gather_player_box(out: dict, sup_game: torch.Tensor, sup_side: torch.Tensor,
                      sup_slot: torch.Tensor) -> torch.Tensor:
    """Returns (M, K_BOX) predicted per-player BOX vector at the supervised indices."""
    H = out["home_box"]
    A = out["away_box"]
    M = sup_game.size(0)
    K = H.size(-1)
    pred = torch.zeros(M, K, device=H.device, dtype=H.dtype)
    if M == 0:
        return pred
    h = sup_side == 0
    a = sup_side == 1
    if h.any():
        idx = h.nonzero(as_tuple=False).squeeze(-1)
        pred[idx] = H[sup_game[idx], sup_slot[idx]]
    if a.any():
        idx = a.nonzero(as_tuple=False).squeeze(-1)
        pred[idx] = A[sup_game[idx], sup_slot[idx]]
    return pred


def pair_poisson_loss(out: dict, batch: dict, eps: float = 1e-6
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair dense Poisson NLL (closed-form): per-target (K_PAIR,) and n_pairs scalar."""
    A = out["pair_dir_A"]
    B = out["pair_dir_B"]
    mask_A = out["pair_mask_A"].unsqueeze(-1).to(A.dtype)
    mask_B = out["pair_mask_B"].unsqueeze(-1).to(B.dtype)
    sum_lam = (A * mask_A).sum(dim=(0, 1, 2)) + (B * mask_B).sum(dim=(0, 1, 2))

    pred = gather_pair_pred(out, batch["sup_pair_game"], batch["sup_pair_side"],
                            batch["sup_pair_off"], batch["sup_pair_def"])
    y = batch["sup_pair_y"]
    sum_y_log_lam = (y * torch.log(pred.clamp_min(eps))).sum(dim=0)

    n_pairs = out["pair_mask_A"].sum() + out["pair_mask_B"].sum()
    n_pairs = n_pairs.clamp_min(1).to(A.dtype)
    nll = (sum_lam - sum_y_log_lam) / n_pairs
    return nll, n_pairs


def player_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    """Per-stat MSE on the 14 BOX targets at supervised player slots,
    averaged over players. Returns (K_BOX,)."""
    pred = gather_player_box(out, batch["sup_pl_game"], batch["sup_pl_side"],
                             batch["sup_pl_slot"])
    y = batch["sup_pl_y"]
    if pred.size(0) == 0:
        return torch.zeros(K_BOX, device=pred.device, dtype=pred.dtype)
    return F.mse_loss(pred, y, reduction="none").mean(dim=0)


def team_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    """Per-stat MSE on team BOX totals (home + away averaged), per batch.
    Returns (K_BOX,)."""
    home_pred = out["home_team_box"]
    away_pred = out["away_team_box"]
    home_y = batch["team_box_home"]
    away_y = batch["team_box_away"]
    h_loss = F.mse_loss(home_pred, home_y, reduction="none").mean(dim=0)
    a_loss = F.mse_loss(away_pred, away_y, reduction="none").mean(dim=0)
    return 0.5 * (h_loss + a_loss)


def margin_nll_loss(out: dict, batch: dict) -> torch.Tensor:
    """Gaussian NLL on margin: 0.5 * [(y-μ)²/σ² + log σ²].

    Dense per-game gradient on margin, with learned heteroscedastic σ. The
    log σ is clamped to [0, 5] (σ ∈ [1, 148]) for numerical safety; real
    NBA margin std ≈ 14 sits well inside that range.
    """
    mu = out["margin_mu"]
    log_sigma = out["margin_log_sigma"].clamp(min=0.0, max=5.0)
    sq_err = (batch["margin"] - mu) ** 2
    inv_var = (-2.0 * log_sigma).exp()
    return 0.5 * (sq_err * inv_var + 2.0 * log_sigma).mean()


def win_bce_loss(out: dict, batch: dict) -> torch.Tensor:
    """BCE on win_logit (= margin_mu / global_scale) vs the home-win label.
    Returns a scalar (mean over batch). This is the only term that gives a
    direct gradient on margin, so it's what breaks the home/away symmetry
    that the box-score MSE losses can satisfy by predicting league-average
    pace alone."""
    return F.binary_cross_entropy_with_logits(
        out["win_logit"], batch["label"], reduction="mean"
    )


def decision_bce_loss(out: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE on decision_logit vs the win-of-chosen-side indicator.

    The decision head learns P(bet wins | game features, odds, chosen side),
    where the chosen side is the win head's argmax. Loss is masked to
    games with usable odds (has_odds == 1). Returns (loss, n_used).
    """
    logit = out["decision_logit"]
    label = batch["label"]
    if logit is None:
        zero = torch.zeros((), device=label.device)
        return zero, zero
    bet_home = (out["win_logit"].detach() >= 0.0).to(label.dtype)
    bet_won = (bet_home == label).to(label.dtype)
    has_odds = batch["has_odds"].to(label.dtype)
    raw = F.binary_cross_entropy_with_logits(logit, bet_won, reduction="none")
    n = has_odds.sum().clamp_min(1.0)
    return (raw * has_odds).sum() / n, has_odds.sum()


def total_loss(
    out: dict, batch: dict,
    *,
    box_weights: torch.Tensor,    # (K_BOX,) — per-stat weight, shared by team+player levels
    pair_weights: torch.Tensor,   # (K_PAIR,)
    team_w: float = 1.0,
    player_w: float = 0.01,
    pair_w: float = 0.001,
    win_w: float = 10.0,
    margin_nll_w: float = 0.0,
    decision_w: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Hierarchical loss with direct win-supervision.

        L = team_w · L_team_mse  +  player_w · L_player_mse
          + pair_w · L_pair_poisson  +  win_w · L_win
          + margin_nll_w · L_margin_nll

    The first three levels reduce over their native dimensions (players /
    pairs / sides) and aggregate K stats with `box_weights` (or
    `pair_weights` for L_pair). The win term is BCE on the composed
    margin → win_logit, and it's what makes the model actually learn the
    differential signal. global_scale becomes a trained parameter via this
    term (previously it had no gradient path). L_margin_nll is the
    Gaussian NLL on margin with learned heteroscedastic σ — dense
    per-game gradient instead of the one-bit BCE signal.
    """
    pair_nll, n_pairs = pair_poisson_loss(out, batch)
    player_mse = player_mse_loss(out, batch)
    team_mse = team_mse_loss(out, batch)
    win_bce = win_bce_loss(out, batch)

    box_w = box_weights.to(player_mse)
    pair_w_t = pair_weights.to(pair_nll)

    L_team = (box_w * team_mse).sum()
    L_player = (box_w * player_mse).sum() if player_mse.numel() > 0 else torch.zeros((), device=team_mse.device)
    L_pair = (pair_w_t * pair_nll).sum()
    L_win = win_bce
    L_margin_nll = margin_nll_loss(out, batch) if margin_nll_w > 0 else torch.zeros((), device=team_mse.device)
    if decision_w > 0 and out.get("decision_logit") is not None:
        L_decision, n_decision = decision_bce_loss(out, batch)
    else:
        L_decision = torch.zeros((), device=team_mse.device)
        n_decision = torch.zeros((), device=team_mse.device)

    total = (team_w * L_team + player_w * L_player + pair_w * L_pair
             + win_w * L_win + margin_nll_w * L_margin_nll
             + decision_w * L_decision)
    return total, {
        "team_mse_per_target": team_mse.detach(),
        "player_mse_per_target": player_mse.detach(),
        "pair_nll_per_target": pair_nll.detach(),
        "L_team": L_team.detach(),
        "L_player": L_player.detach(),
        "L_pair": L_pair.detach(),
        "L_win": L_win.detach(),
        "L_margin_nll": L_margin_nll.detach(),
        "L_decision": L_decision.detach(),
        "n_decision": n_decision.detach(),
        "n_pairs_valid": n_pairs.detach(),
    }


# ---------------------------- smoke ----------------------------


def _smoke() -> None:
    torch.manual_seed(0)
    cfg = CmeV2Config(
        vocab_size=200, num_teams=30, d=64, tabular_dim=4,
        team_emb_dim=8, player_stat_dim=10,
    )
    model = CmeV2(cfg)
    B, Lh, La = 4, 14, 16
    home_idx = torch.randint(1, cfg.vocab_size + 1, (B, Lh))
    away_idx = torch.randint(1, cfg.vocab_size + 1, (B, La))
    home_mask = torch.ones(B, Lh, dtype=torch.bool); home_mask[0, 12:] = False
    away_mask = torch.ones(B, La, dtype=torch.bool); away_mask[1, 14:] = False

    sup_pair_game = torch.tensor([0, 0, 1, 2, 3])
    sup_pair_side = torch.tensor([0, 1, 0, 0, 1])
    sup_pair_off = torch.tensor([0, 0, 1, 2, 3])
    sup_pair_def = torch.tensor([0, 1, 2, 3, 4])
    sup_pair_y = torch.rand(5, K_PAIR) * 3.0

    sup_pl_game = torch.tensor([0, 0, 1, 2, 3, 3])
    sup_pl_side = torch.tensor([0, 1, 0, 1, 0, 1])
    sup_pl_slot = torch.tensor([0, 1, 2, 3, 4, 5])
    sup_pl_y = torch.rand(6, K_BOX) * 5.0

    team_box_home = torch.rand(B, K_BOX) * 30
    team_box_away = torch.rand(B, K_BOX) * 30

    batch = {
        "home_idx": home_idx, "away_idx": away_idx,
        "home_prob": torch.rand(B, Lh), "away_prob": torch.rand(B, La),
        "home_mask": home_mask, "away_mask": away_mask,
        "home_stats": torch.randn(B, Lh, cfg.player_stat_dim),
        "away_stats": torch.randn(B, La, cfg.player_stat_dim),
        "home_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "away_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "home_rest": torch.rand(B), "away_rest": torch.rand(B),
        "tabular": torch.randn(B, cfg.tabular_dim),
        "label": torch.randint(0, 2, (B,), dtype=torch.float32),
        "margin": torch.randn(B) * 10,
        "team_box_home": team_box_home,
        "team_box_away": team_box_away,
        "sup_pair_game": sup_pair_game, "sup_pair_side": sup_pair_side,
        "sup_pair_off": sup_pair_off, "sup_pair_def": sup_pair_def,
        "sup_pair_y": sup_pair_y,
        "sup_pl_game": sup_pl_game, "sup_pl_side": sup_pl_side,
        "sup_pl_slot": sup_pl_slot, "sup_pl_y": sup_pl_y,
    }

    out = model(batch)
    print("=== shapes ===")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:20s} {tuple(v.shape)}")
        else:
            print(f"{k:20s} {type(v).__name__}")

    print(f"home_points        {out['home_points'].tolist()}")
    print(f"away_points        {out['away_points'].tolist()}")
    print(f"margin_mu          {out['margin_mu'].tolist()}")
    print(f"global_scale       {out['global_scale'].item():.3f}")

    box_w = torch.ones(K_BOX); box_w[BOX_INDEX["pts"]] = 3.0
    pair_w = torch.ones(K_PAIR)
    loss, diag = total_loss(out, batch, box_weights=box_w, pair_weights=pair_w)
    print(f"\nloss = {loss.item():.4f}")
    print(f"  L_team   = {diag['L_team'].item():.4f}")
    print(f"  L_player = {diag['L_player'].item():.4f}")
    print(f"  L_pair   = {diag['L_pair'].item():.4f}")
    print(f"  L_win    = {diag['L_win'].item():.4f}")
    print(f"  team_mse per stat   = {diag['team_mse_per_target'].tolist()}")
    print(f"  player_mse per stat = {diag['player_mse_per_target'].tolist()}")
    print(f"  pair_nll  per chan  = {diag['pair_nll_per_target'].tolist()}")
    print(f"  n_pairs_valid       = {diag['n_pairs_valid'].item():.0f}")

    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    n_total = sum(1 for _ in model.parameters())
    print(f"\nbackward ok: {n_grad}/{n_total} parameters got gradients")

    has_global_grad = model.global_scale.grad is not None
    print(f"global_scale gets grad: {has_global_grad}")


if __name__ == "__main__":
    _smoke()
