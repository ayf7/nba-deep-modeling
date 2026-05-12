"""CME-v5: hybrid world model (Sinkhorn budget + K-channel play-style).

Architecture combines the strengths of v1 (constrained pair exposure via
involvement softmax + Sinkhorn) with v2 (cross-attended K-channel rates and
hierarchical box-score supervision). The multiplicative decomposition

    lambda_ij^(k) = P_ij * r_ij^(k)
                  = (exposure: how often does pair (i,j) actually play)
                    x (play-style: per-exposure rate in stat channel k)

separates two quantities that v2 conflates into a single set of unconstrained
rates. The exposure factor P_ij is budget-constrained (Sinkhorn against
involvement shares alpha_i^off, alpha_j^def, both of which sum to 1 over the
lineup and are anchored by a log(prob_play) prior plus a direct MSE supervision
against the observed exposure_seconds shares). The play-style factor r_ij^(k)
is unconstrained, so per-pair counting stats can have arbitrary dynamic range
without breaking budget conservation.

Forward pipeline per game:

  1. Tokenize players: E_player(idx) + feat_mlp(stats), gated by play prob.
  2. Add per-team context: (team_emb + tabular + rest) -> ctx -> infuse into
     every player slot.
  3. Per-team self-attention (params shared across teams).
  4. Off / def role projections per player.
  5. Cross-attention: home.off <- away.def, away.off <- home.def (shared).
  6. Involvement heads (per team, from post-self-attn h_i):
       alpha_i^off = softmax(MLP_off(h_i) + log p_i^play)
       alpha_j^def = softmax(MLP_def(h_j) + log p_j^play)
       N = softplus(MLP_N(ctx)) + N_base
  7. Sinkhorn per direction:
       s_ij = (Q(off_i).K(def_j))/sqrt(d) + MLP_score([off; def])
       P    = Sinkhorn(s_ij, row=N*alpha_i^off, col=N*alpha_j^def, mask)
  8. Per-channel pair rates (K_PAIR-1 channels; exposure rides as P itself):
       r_ij^(k) = exp(MLP_k([off; def; s_ij]) + bias_k)
       lambda_ij^(0) = P_ij                          (exposure channel)
       lambda_ij^(k) = P_ij * r_ij^(k)  for k > 0
  9. Per-player non-pair head (K_PLAYER channels), scaled by alpha:
       y_hat_i^(k) = alpha_i^{off|def} * exp(MLP(h_i))
 10. Assemble per-player BOX (14-d) -> team BOX -> margin -> calibrated win_logit.

Eleven-level loss:
    L = w_inv * L_inv_mse           (alpha vs actual exposure shares)
      + w_pair * L_pair_poisson     (K_PAIR pair rates)
      + w_player * L_player_mse     (14-d per-player BOX)
      + w_team * L_team_mse         (14-d team total BOX)
      + w_win * L_win_bce           (binary outcome, direct margin gradient)
      + w_margin * L_margin_nll     (heteroscedastic Gaussian on margin)
      + w_calib * L_calibration_reg  (bounded win residual regularizer)
      + w_slope * L_calibration_slope_reg (log-slope prior around 1.0)

Counterfactual: setting prob_play[i]=0 sends log-prior to -inf, softmax
redistributes involvement to remaining players, Sinkhorn re-routes exposure
across the K-channel rates, and team totals shift coherently with the budget.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

V1_SCRIPTS = Path(__file__).resolve().parents[2] / "models_cme_v1" / "scripts"
sys.path.insert(0, str(V1_SCRIPTS))

from sinkhorn import sinkhorn_transport  # noqa: E402

V5_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_SCRIPTS))

from cme_v5_common import (  # noqa: E402
    BOX_INDEX,
    K_BOX,
    K_PAIR,
    K_PLAYER,
    SEASON_PHASE_DIM,
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


# BOX position -> source. Pair-attributed stats come from pair-grid row sums;
# the rest from the player head. PTS uses 2*FGM_pair + 3PM_pair + FTM_player.
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

# Whether each non-pair player channel is "offensive" (uses alpha^off) or
# "defensive" (uses alpha^def). Reasoning: FTs, ORebs are scored on offensive
# possessions; DRebs, steals, defensive fouls are accrued on defense.
_PLAYER_CHANNEL_IS_OFF: tuple[bool, ...] = (
    True,   # ftm
    True,   # fta
    True,   # oreb
    False,  # dreb
    False,  # stl
    False,  # pf
)
assert len(_PLAYER_CHANNEL_IS_OFF) == K_PLAYER


@dataclass
class CmeV5Config:
    vocab_size: int
    num_teams: int
    d: int = 64
    n_heads: int = 4
    n_self_layers: int = 2
    n_cross_layers: int = 2
    pair_hidden: int = 96
    player_hidden: int = 64
    inv_hidden: int = 64
    dropout: float = 0.1
    pair_dropout: float = 0.2
    player_dropout: float = 0.0
    tabular_dim: int = 0
    team_emb_dim: int = 16
    player_stat_dim: int = 0
    score_clamp: float = 12.0
    sinkhorn_iters: int = 8
    # Total player-possessions per game per offensive direction. Sum over the
    # pair grid in one direction equals N, since alpha^off sums to 1 and the
    # row marginals are N*alpha. Matches the prior used by v1 (491).
    base_possessions_per_team: float = 491.0
    rate_clamp: float = 8.0
    init_global_scale: float = 12.0
    # v5 win head.  v4.2 established that affine margin calibration,
    # season-phase conditioning, and tail-gated residuals improve the backtest.
    # The remaining weakness is ranking/robustness: the final correction only
    # saw coarse team context, while high-confidence misses likely depend on
    # roster/matchup states and uncertainty in the predicted margin.
    #
    # v5 therefore keeps the v4.2 path and adds two structured pieces:
    #   (a) a bounded lineup/matchup evidence residual pooled from player states,
    #   (b) a non-negative uncertainty temperature that damps logits when
    #       learned margin uncertainty or player-availability uncertainty is high.
    #
    # pretemp_logit = slope * (margin/scale) + home_bias
    #                + season_delta(phase)
    #                + tail_gate * context_residual(ctx)
    #                + tail_gate * matchup_residual(player pools)
    # final_logit   = pretemp_logit / (1 + uncertainty_temperature_delta)
    calibration_hidden: int = 64
    calibration_dropout: float = 0.05
    max_calibration_residual: float = 0.30
    season_calibration_hidden: int = 16
    max_season_logit_adjustment: float = 0.25
    tail_gate_center: float = 1.25
    tail_gate_sharpness: float = 2.0
    matchup_evidence_hidden: int = 96
    max_matchup_evidence_residual: float = 0.35
    uncertainty_hidden: int = 16
    max_uncertainty_temperature_delta: float = 0.75
    uncertainty_temperature_init_logit: float = -4.0
    init_margin_sigma: float = 13.0
    init_home_logit_bias: float = 0.15
    init_win_logit_slope: float = 1.0
    trainable_global_scale: bool = False


def _zero_last_linear(seq: nn.Sequential) -> None:
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
    return (x * m).sum(dim=1) / denom


def _masked_scalar_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    denom = mask.sum(dim=1).clamp_min(1).to(x.dtype)
    return (x * m).sum(dim=1) / denom


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(min=1e-6, max=1.0 - 1e-6)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


class SelfBlock(nn.Module):
    """Pre-LN MHA + FFN, padding-masked."""
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
    """Pre-LN cross-attention + FFN. Q is updated."""
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


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(logits, dim=dim)


class CmeV5(nn.Module):
    def __init__(self, cfg: CmeV5Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d

        # -- token embeddings --
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

        ctx_in = cfg.team_emb_dim + cfg.tabular_dim + 1   # team_emb + tabular + rest
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_in, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
        )
        _zero_last_linear(self.ctx_proj)

        # -- attention stack --
        self.self_layers = nn.ModuleList([
            SelfBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_self_layers)
        ])
        self.off_proj = nn.Linear(d, d)
        self.def_proj = nn.Linear(d, d)
        self.cross_layers = nn.ModuleList([
            CrossBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_cross_layers)
        ])

        # -- involvement heads (operate on post-self-attn h_i; per-team ctx
        #    is concatenated so the head can downweight bench players using
        #    team / matchup signals as well as play prob) --
        self.inv_off_head = nn.Sequential(
            nn.Linear(2 * d, cfg.inv_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.inv_hidden, 1),
        )
        self.inv_def_head = nn.Sequential(
            nn.Linear(2 * d, cfg.inv_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.inv_hidden, 1),
        )
        # Possession volume N from game ctx (home_ctx + away_ctx concat).
        self.poss_head = nn.Sequential(
            nn.Linear(2 * d, cfg.inv_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.inv_hidden, 1),
        )
        _zero_last_linear(self.poss_head)  # initialize N at base_possessions_per_team

        # -- pair score head (Sinkhorn input) --
        self.score_q = nn.Linear(d, d)
        self.score_k = nn.Linear(d, d)
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * d, cfg.pair_hidden), nn.GELU(), nn.Dropout(cfg.pair_dropout),
            nn.Linear(cfg.pair_hidden, 1),
        )
        _zero_last_linear(self.score_mlp)

        # -- per-channel pair rate head (K_PAIR - 1 channels; exposure rides as P) --
        # We still produce K_PAIR rates and just override channel 0 with 1.0
        # at composition time, so the "rate head" matches v2's layout. Bias is
        # initialized so that at zero MLP output, exp(bias) is the league-mean
        # per-possession rate for each channel (so the model starts producing
        # team totals near reality and learning is just a perturbation around
        # that prior). Channel ordering must match PAIR_TARGETS.
        self.pair_rate_mlp = nn.Sequential(
            nn.Linear(2 * d + 1, cfg.pair_hidden), nn.GELU(), nn.Dropout(cfg.pair_dropout),
            nn.Linear(cfg.pair_hidden, K_PAIR),
        )
        _zero_last_linear(self.pair_rate_mlp)
        # League-mean per-possession rates (approx 2024 NBA team averages /
        # base_possessions_per_team). Channel 0 is exposure and overridden
        # to 1.0 at composition time, so init value is irrelevant.
        _pair_rate_priors = torch.tensor([
            1.0,    # exposure_possessions (forced to 1 in composition)
            0.225,  # player_points (~110 FG-points / 491 player-poss)
            0.082,  # matchup_fgm  (~40 / 491)
            0.176,  # matchup_fga  (~86 / 491)
            0.027,  # matchup_3pm  (~13 / 491)
            0.075,  # matchup_3pa  (~37 / 491)
            0.051,  # matchup_assists (~25 / 491)
            0.029,  # matchup_turnovers (~14 / 491)
            0.010,  # matchup_blocks (~5 / 491)
        ])
        assert _pair_rate_priors.numel() == K_PAIR
        self.pair_rate_bias = nn.Parameter(_pair_rate_priors.log())

        # -- per-player non-pair head (operates on post-self-attn) --
        # Same bias-prior trick as pair head: with alpha summing to 1 over a
        # team, sum_i alpha_i * exp(bias) = exp(bias) at init, so the bias
        # equals log(team-target) per channel.
        self.player_mlp = nn.Sequential(
            nn.Linear(d, cfg.player_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.player_hidden, K_PLAYER),
        )
        _zero_last_linear(self.player_mlp)
        _player_target_priors = torch.tensor([
            17.0,   # ftm
            22.0,   # fta
            10.0,   # oreb
            36.0,   # dreb
            7.0,    # stl
            20.0,   # pf
        ])
        assert _player_target_priors.numel() == K_PLAYER
        self.player_bias = nn.Parameter(_player_target_priors.log())

        # -- margin -> win logit, with v5 affine calibration --
        # v3 used win_logit = margin / fixed_scale exactly.  v4 added a home
        # intercept and a broad context residual, but the full backtest showed
        # that the residual learned most of the home-side correction while the
        # global bias remained too close to zero.  v5 elevates the low-risk
        # correction path: a positive affine slope and a warm-started home
        # intercept are applied *before* a deliberately bounded residual.
        self.global_scale = nn.Parameter(
            torch.tensor(math.log(cfg.init_global_scale)),
            requires_grad=cfg.trainable_global_scale,
        )
        if cfg.init_win_logit_slope <= 0.0:
            raise ValueError("init_win_logit_slope must be > 0")
        self.log_win_logit_slope = nn.Parameter(torch.tensor(math.log(cfg.init_win_logit_slope)))
        self.home_logit_bias = nn.Parameter(torch.tensor(cfg.init_home_logit_bias))
        self.season_calibration_head = nn.Sequential(
            nn.Linear(SEASON_PHASE_DIM, cfg.season_calibration_hidden),
            nn.GELU(),
            nn.Dropout(cfg.calibration_dropout),
            nn.Linear(cfg.season_calibration_hidden, 1),
        )
        _zero_last_linear(self.season_calibration_head)

        calibration_in = 4 * d + 1
        self.win_calibration_head = nn.Sequential(
            nn.Linear(calibration_in, cfg.calibration_hidden),
            nn.GELU(),
            nn.Dropout(cfg.calibration_dropout),
            nn.Linear(cfg.calibration_hidden, 1),
        )
        _zero_last_linear(self.win_calibration_head)

        matchup_evidence_in = 5 * d + 1
        self.matchup_evidence_head = nn.Sequential(
            nn.Linear(matchup_evidence_in, cfg.matchup_evidence_hidden),
            nn.GELU(),
            nn.Dropout(cfg.calibration_dropout),
            nn.Linear(cfg.matchup_evidence_hidden, 1),
        )
        _zero_last_linear(self.matchup_evidence_head)

        # Learned heteroscedastic margin uncertainty.  v4.2 already had this
        # head but its default margin-NLL weight was zero; v5 trains it and uses
        # the result only as one input to a gently regularized temperature head.
        self.margin_log_sigma_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )
        if cfg.init_margin_sigma <= 0.0:
            raise ValueError("init_margin_sigma must be > 0")
        with torch.no_grad():
            self.margin_log_sigma_head[-1].weight.zero_()
            self.margin_log_sigma_head[-1].bias.fill_(math.log(cfg.init_margin_sigma))

        uncertainty_in = 7
        self.uncertainty_temperature_head = nn.Sequential(
            nn.Linear(uncertainty_in, cfg.uncertainty_hidden),
            nn.GELU(),
            nn.Dropout(cfg.calibration_dropout),
            nn.Linear(cfg.uncertainty_hidden, 1),
        )
        _zero_last_linear(self.uncertainty_temperature_head)
        with torch.no_grad():
            self.uncertainty_temperature_head[-1].bias.fill_(cfg.uncertainty_temperature_init_logit)

        # Buffer for which non-pair channels are scaled by alpha^off vs alpha^def
        self.register_buffer(
            "_player_is_off",
            torch.tensor(_PLAYER_CHANNEL_IS_OFF, dtype=torch.bool),
        )

    # ------------------------- tokenization -------------------------

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

    # ------------------------- involvement -------------------------

    def _involvement(
        self,
        h: torch.Tensor,        # (B, L, d) post-self-attn
        ctx: torch.Tensor,      # (B, d)
        prob: torch.Tensor,     # (B, L)
        mask: torch.Tensor,     # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (alpha_off, alpha_def), each (B, L), summing to 1 over mask."""
        B, L, d = h.shape
        ctx_e = ctx.unsqueeze(1).expand(-1, L, -1)
        h_in = torch.cat([h, ctx_e], dim=-1)
        logit_off = self.inv_off_head(h_in).squeeze(-1)
        logit_def = self.inv_def_head(h_in).squeeze(-1)
        log_p = torch.log(prob.clamp_min(1e-6))
        alpha_off = masked_softmax(logit_off + log_p, mask, dim=-1)
        alpha_def = masked_softmax(logit_def + log_p, mask, dim=-1)
        return alpha_off, alpha_def

    # ------------------------- pair direction -------------------------

    def _direction(
        self,
        off: torch.Tensor,             # (B, Lo, d) post-cross off rep
        def_: torch.Tensor,            # (B, Ld, d) post-cross def rep
        alpha_off: torch.Tensor,       # (B, Lo) involvement shares of offense
        alpha_def: torch.Tensor,       # (B, Ld) involvement shares of defense
        mask_off: torch.Tensor,        # (B, Lo)
        mask_def: torch.Tensor,        # (B, Ld)
        N: torch.Tensor,               # (B,)
    ) -> dict:
        B, Lo, d = off.shape
        Ld = def_.size(1)

        # -- score s_ij for Sinkhorn (and as a feature into the rate head) --
        Q = self.score_q(off)
        K = self.score_k(def_)
        s_dot = torch.einsum("bld,bmd->blm", Q, K) / math.sqrt(d)

        off_e = off.unsqueeze(2).expand(-1, -1, Ld, -1)
        def_e = def_.unsqueeze(1).expand(-1, Lo, -1, -1)
        pair_feat = torch.cat([off_e, def_e], dim=-1)  # (B, Lo, Ld, 2d)
        s_mlp = self.score_mlp(pair_feat).squeeze(-1)   # (B, Lo, Ld)
        score = s_dot + s_mlp
        score = torch.clamp(score, min=-self.cfg.score_clamp, max=self.cfg.score_clamp)

        mask_outer = mask_off.unsqueeze(2) & mask_def.unsqueeze(1)
        row_mass = N.unsqueeze(-1) * alpha_off
        col_mass = N.unsqueeze(-1) * alpha_def

        P = sinkhorn_transport(
            log_scores=score,
            row_mass=row_mass,
            col_mass=col_mass,
            mask=mask_outer,
            n_iter=self.cfg.sinkhorn_iters,
        )

        # -- per-channel pair rates --
        rate_in = torch.cat([off_e, def_e, score.unsqueeze(-1)], dim=-1)  # (B, Lo, Ld, 2d+1)
        rate_raw = self.pair_rate_mlp(rate_in) + self.pair_rate_bias       # (B, Lo, Ld, K_PAIR)
        rate_raw = torch.clamp(rate_raw, min=-12.0, max=self.cfg.rate_clamp)
        rate = torch.exp(rate_raw) * mask_outer.unsqueeze(-1).to(rate_raw.dtype)

        # Channel 0 is exposure: lambda^(0) = P_ij directly. Force rate[0]=1 so
        # composition lambda = P * rate gives P at channel 0.
        rate_filled = rate.clone()
        rate_filled[..., 0] = 1.0
        rate_filled = rate_filled * mask_outer.unsqueeze(-1).to(rate_filled.dtype)

        # Final pair rates: P * rate. For exposure channel (0), this = P.
        lam = P.unsqueeze(-1) * rate_filled  # (B, Lo, Ld, K_PAIR)

        return {
            "P": P,
            "rate": rate_filled,
            "lam": lam,
            "row_mass": row_mass,
            "col_mass": col_mass,
        }

    # ------------------------- player head -------------------------

    def _player_rates(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Returns (B, L, K_PLAYER) per-player raw non-pair rates exp(MLP(h))."""
        raw = self.player_mlp(h) + self.player_bias
        raw = torch.clamp(raw, min=-12.0, max=self.cfg.rate_clamp)
        rates = torch.exp(raw)
        rates = rates * mask.unsqueeze(-1).to(rates.dtype)
        return rates

    def _scale_player_rates(
        self,
        rates: torch.Tensor,         # (B, L, K_PLAYER)
        alpha_off: torch.Tensor,     # (B, L)
        alpha_def: torch.Tensor,     # (B, L)
    ) -> torch.Tensor:
        """Apply involvement scaling per channel: alpha_off for offensive
        non-pair stats, alpha_def for defensive. Returns same shape."""
        # Broadcasting: (B, L, K) * (B, L, 1) with mask per channel.
        is_off = self._player_is_off.view(1, 1, -1).to(rates.dtype)
        alpha = alpha_off.unsqueeze(-1) * is_off + alpha_def.unsqueeze(-1) * (1.0 - is_off)
        return rates * alpha

    # ------------------------- forward -------------------------

    def forward(self, batch: dict) -> dict:
        home_mask = batch["home_mask"]
        away_mask = batch["away_mask"]
        if self.training and self.cfg.player_dropout > 0.0:
            p = self.cfg.player_dropout
            home_mask = home_mask & (torch.rand_like(home_mask, dtype=torch.float32) > p)
            away_mask = away_mask & (torch.rand_like(away_mask, dtype=torch.float32) > p)

        tabular = batch.get("tabular")

        # -- token + ctx infusion --
        home_tok = self._tokenize(batch["home_idx"], batch["home_stats"],
                                  batch["home_prob"], home_mask)
        away_tok = self._tokenize(batch["away_idx"], batch["away_stats"],
                                  batch["away_prob"], away_mask)
        home_ctx = self._team_ctx(batch["home_team_idx"], tabular, batch["home_rest"])
        away_ctx = self._team_ctx(batch["away_team_idx"], tabular, batch["away_rest"])
        home_tok = home_tok + home_ctx.unsqueeze(1) * home_mask.unsqueeze(-1).to(home_tok.dtype)
        away_tok = away_tok + away_ctx.unsqueeze(1) * away_mask.unsqueeze(-1).to(away_tok.dtype)

        # -- per-team self-attn (shared params) --
        home_self = home_tok
        away_self = away_tok
        for layer in self.self_layers:
            home_self = layer(home_self, home_mask)
            away_self = layer(away_self, away_mask)

        # -- involvement heads operate on post-self-attn h_i --
        alpha_home_off, alpha_home_def = self._involvement(
            home_self, home_ctx, batch["home_prob"], home_mask,
        )
        alpha_away_off, alpha_away_def = self._involvement(
            away_self, away_ctx, batch["away_prob"], away_mask,
        )

        # Possession volume N from concatenated game ctx
        N = F.softplus(self.poss_head(torch.cat([home_ctx, away_ctx], dim=-1)).squeeze(-1))
        N = N + self.cfg.base_possessions_per_team

        # -- role projections + cross-attn (off is updated, def is static after proj) --
        home_off = self.off_proj(home_self) * home_mask.unsqueeze(-1).to(home_self.dtype)
        home_def = self.def_proj(home_self) * home_mask.unsqueeze(-1).to(home_self.dtype)
        away_off = self.off_proj(away_self) * away_mask.unsqueeze(-1).to(away_self.dtype)
        away_def = self.def_proj(away_self) * away_mask.unsqueeze(-1).to(away_self.dtype)
        for layer in self.cross_layers:
            home_off = layer(home_off, away_def, q_mask=home_mask, kv_mask=away_mask)
            away_off = layer(away_off, home_def, q_mask=away_mask, kv_mask=home_mask)

        # -- direction A: home offense vs away defense --
        dir_A = self._direction(
            home_off, away_def, alpha_home_off, alpha_away_def,
            home_mask, away_mask, N,
        )
        # -- direction B: away offense vs home defense --
        dir_B = self._direction(
            away_off, home_def, alpha_away_off, alpha_home_def,
            away_mask, home_mask, N,
        )

        # -- per-player BOX assembly --
        # Pair-attributed: row-sum over defender axis gives per-offensive-player
        # K_PAIR marginal.
        home_pair_marg = dir_A["lam"].sum(dim=2)   # (B, Lh, K_PAIR)
        away_pair_marg = dir_B["lam"].sum(dim=2)   # (B, La, K_PAIR)

        home_player_raw = self._player_rates(home_self, home_mask)
        away_player_raw = self._player_rates(away_self, away_mask)
        home_player_rates = self._scale_player_rates(
            home_player_raw, alpha_home_off, alpha_home_def,
        )
        away_player_rates = self._scale_player_rates(
            away_player_raw, alpha_away_off, alpha_away_def,
        )

        home_box = self._assemble_player_box(home_pair_marg, home_player_rates)
        away_box = self._assemble_player_box(away_pair_marg, away_player_rates)
        home_box = home_box * home_mask.unsqueeze(-1).to(home_box.dtype)
        away_box = away_box * away_mask.unsqueeze(-1).to(away_box.dtype)

        home_team_box = home_box.sum(dim=1)
        away_team_box = away_box.sum(dim=1)

        # -- margin / win logit --
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
            # Defensive fallback for ad-hoc synthetic callers; real v5
            # datasets always provide deterministic pregame season features.
            season_phase = torch.zeros(
                win_logit_affine.size(0), SEASON_PHASE_DIM,
                device=win_logit_affine.device, dtype=win_logit_affine.dtype,
            )
        raw_season_delta = self.season_calibration_head(season_phase).squeeze(-1)
        if self.cfg.max_season_logit_adjustment > 0.0:
            season_logit_adjustment = (
                self.cfg.max_season_logit_adjustment * torch.tanh(raw_season_delta)
            )
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

        # Residuals were helpful in v4.1, but the far probability tails became
        # slightly too sharp.  Suppress context residuals once the calibrated
        # margin logit is already large in absolute value.  This keeps the
        # residual useful around close/moderate games while preserving the
        # cleaner affine/season calibration in extreme cases.
        if self.cfg.tail_gate_sharpness > 0.0:
            residual_tail_gate = torch.sigmoid(
                (self.cfg.tail_gate_center - win_logit_temporal.abs())
                * self.cfg.tail_gate_sharpness
            )
        else:
            residual_tail_gate = torch.ones_like(win_logit_temporal)
        win_residual = residual_tail_gate * bounded_win_residual

        # Rich lineup/matchup evidence residual.  v4.2's residual only saw
        # coarse team/game context; this head can respond to the learned player
        # states and cross-attended offensive representations that already drive
        # the structured world model.
        home_lineup_pool = _masked_mean(home_self, home_mask)
        away_lineup_pool = _masked_mean(away_self, away_mask)
        home_off_pool = _masked_mean(home_off, home_mask)
        away_off_pool = _masked_mean(away_off, away_mask)
        matchup_evidence_features = torch.cat([
            home_lineup_pool,
            away_lineup_pool,
            home_lineup_pool - away_lineup_pool,
            home_lineup_pool * away_lineup_pool,
            home_off_pool - away_off_pool,
            win_logit_temporal.unsqueeze(-1),
        ], dim=-1)
        raw_matchup_evidence = self.matchup_evidence_head(matchup_evidence_features).squeeze(-1)
        if self.cfg.max_matchup_evidence_residual > 0.0:
            bounded_matchup_evidence = (
                self.cfg.max_matchup_evidence_residual * torch.tanh(raw_matchup_evidence)
            )
        else:
            bounded_matchup_evidence = torch.zeros_like(win_logit_temporal)
        matchup_evidence_residual = residual_tail_gate * bounded_matchup_evidence

        margin_log_sigma = self.margin_log_sigma_head(
            torch.cat([home_ctx, away_ctx], dim=-1)
        ).squeeze(-1)
        margin_sigma = margin_log_sigma.exp().clamp_min(1e-4)

        # Pregame uncertainty features.  The status probabilities already enter
        # tokenization and involvement priors; v5 additionally summarizes their
        # entropy so the win probability can become more conservative when the
        # expected rotation is genuinely uncertain.
        home_play_entropy = _masked_scalar_mean(_binary_entropy(batch["home_prob"]), home_mask)
        away_play_entropy = _masked_scalar_mean(_binary_entropy(batch["away_prob"]), away_mask)
        home_unavailable_prob = _masked_scalar_mean(1.0 - batch["home_prob"], home_mask)
        away_unavailable_prob = _masked_scalar_mean(1.0 - batch["away_prob"], away_mask)
        log_sigma_centered = margin_log_sigma - math.log(self.cfg.init_margin_sigma)
        uncertainty_features = torch.stack([
            log_sigma_centered,
            home_play_entropy,
            away_play_entropy,
            (home_play_entropy - away_play_entropy).abs(),
            home_unavailable_prob,
            away_unavailable_prob,
            (home_unavailable_prob - away_unavailable_prob).abs(),
        ], dim=-1)
        raw_uncertainty_temperature = self.uncertainty_temperature_head(uncertainty_features).squeeze(-1)
        if self.cfg.max_uncertainty_temperature_delta > 0.0:
            uncertainty_temperature_delta = (
                self.cfg.max_uncertainty_temperature_delta * torch.sigmoid(raw_uncertainty_temperature)
            )
        else:
            uncertainty_temperature_delta = torch.zeros_like(win_logit_temporal)
        uncertainty_temperature = 1.0 + uncertainty_temperature_delta

        win_logit_pretemp = win_logit_temporal + win_residual + matchup_evidence_residual
        win_logit = win_logit_pretemp / uncertainty_temperature

        return {
            # pair rates / exposure
            "pair_dir_A": dir_A["lam"],          # (B, Lh, La, K_PAIR)
            "pair_dir_B": dir_B["lam"],
            "P_A": dir_A["P"],                   # exposure matrices (B, Lh, La)
            "P_B": dir_B["P"],
            "rate_A": dir_A["rate"],
            "rate_B": dir_B["rate"],
            "pair_mask_A": home_mask.unsqueeze(-1) & away_mask.unsqueeze(1),
            "pair_mask_B": away_mask.unsqueeze(-1) & home_mask.unsqueeze(1),
            # involvement
            "alpha_home_off": alpha_home_off,
            "alpha_home_def": alpha_home_def,
            "alpha_away_off": alpha_away_off,
            "alpha_away_def": alpha_away_def,
            "N": N,
            # player / box / team
            "home_player_rates": home_player_rates,
            "away_player_rates": away_player_rates,
            "home_pair_marg": home_pair_marg,
            "away_pair_marg": away_pair_marg,
            "home_box": home_box,                # (B, Lh, K_BOX)
            "away_box": away_box,                # (B, La, K_BOX)
            "home_team_box": home_team_box,      # (B, K_BOX)
            "away_team_box": away_team_box,      # (B, K_BOX)
            # margin / win
            "home_points": home_points,
            "away_points": away_points,
            "margin_mu": margin_mu,
            "margin_log_sigma": margin_log_sigma,
            "margin_sigma": margin_sigma,
            "win_logit": win_logit,
            "win_logit_base": win_logit_base,
            "win_logit_affine": win_logit_affine,
            "win_logit_temporal": win_logit_temporal,
            "win_logit_pretemp": win_logit_pretemp,
            "win_logit_slope": win_logit_slope,
            "log_win_logit_slope": self.log_win_logit_slope,
            "season_phase": season_phase,
            "season_logit_adjustment": season_logit_adjustment,
            "residual_tail_gate": residual_tail_gate,
            "bounded_win_residual": bounded_win_residual,
            "win_residual": win_residual,
            "bounded_matchup_evidence": bounded_matchup_evidence,
            "matchup_evidence_residual": matchup_evidence_residual,
            "uncertainty_temperature": uncertainty_temperature,
            "uncertainty_temperature_delta": uncertainty_temperature_delta,
            "home_play_entropy": home_play_entropy,
            "away_play_entropy": away_play_entropy,
            "home_unavailable_prob": home_unavailable_prob,
            "away_unavailable_prob": away_unavailable_prob,
            "home_logit_bias": self.home_logit_bias,
            "global_scale": scale,
            "home_ctx": home_ctx,
            "away_ctx": away_ctx,
        }

    def _assemble_player_box(
        self, pair_marginal: torch.Tensor, player_rates: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = pair_marginal.shape
        out = torch.zeros(B, L, K_BOX, device=pair_marginal.device, dtype=pair_marginal.dtype)
        for box_name, (src, ch) in _BOX_SOURCE.items():
            i = BOX_INDEX[box_name]
            if src == "pair":
                out[..., i] = pair_marginal[..., ch]
            else:
                out[..., i] = player_rates[..., ch]
        # PTS = 2*FGM_pair + 3PM_pair + FTM_player (NBA scoring)
        out[..., BOX_INDEX["pts"]] = (
            2.0 * pair_marginal[..., _PAIR_FGM_IDX]
            + pair_marginal[..., _PAIR_3PM_IDX]
            + player_rates[..., _PLAYER_FTM_IDX]
        )
        return out


# ----------------------------- losses ----------------------------- #


def gather_pair_pred(out: dict, sup_game: torch.Tensor, sup_side: torch.Tensor,
                     sup_off: torch.Tensor, sup_def: torch.Tensor) -> torch.Tensor:
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
    pred = gather_player_box(out, batch["sup_pl_game"], batch["sup_pl_side"],
                             batch["sup_pl_slot"])
    y = batch["sup_pl_y"]
    if pred.size(0) == 0:
        return torch.zeros(K_BOX, device=pred.device, dtype=pred.dtype)
    return F.mse_loss(pred, y, reduction="none").mean(dim=0)


def team_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    home_pred = out["home_team_box"]
    away_pred = out["away_team_box"]
    home_y = batch["team_box_home"]
    away_y = batch["team_box_away"]
    h_loss = F.mse_loss(home_pred, home_y, reduction="none").mean(dim=0)
    a_loss = F.mse_loss(away_pred, away_y, reduction="none").mean(dim=0)
    return 0.5 * (h_loss + a_loss)


def involvement_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    """MSE between predicted and observed involvement shares, per side and
    per role. Each side's loss is masked by its `*_off_exposure_valid` flag
    so games with missing/sparse matchup rows don't contribute spurious
    supervision. Returns a single scalar.

    Direction-to-target mapping:
      - alpha_home_off  <- home_alpha_off_actual
      - alpha_home_def  <- home_alpha_def_actual
      - alpha_away_off  <- away_alpha_off_actual
      - alpha_away_def  <- away_alpha_def_actual
    """
    home_mask = batch["home_mask"].to(out["alpha_home_off"].dtype)
    away_mask = batch["away_mask"].to(out["alpha_away_off"].dtype)
    h_valid_off = batch["home_off_exposure_valid"].unsqueeze(-1)
    a_valid_off = batch["away_off_exposure_valid"].unsqueeze(-1)

    err_home_off = (out["alpha_home_off"] - batch["home_alpha_off_actual"]) ** 2
    err_home_def = (out["alpha_home_def"] - batch["home_alpha_def_actual"]) ** 2
    err_away_off = (out["alpha_away_off"] - batch["away_alpha_off_actual"]) ** 2
    err_away_def = (out["alpha_away_def"] - batch["away_alpha_def_actual"]) ** 2

    # home_off and away_def share the same "home offense" exposure validity
    # (they come from the same matchup-row partition). away_off and home_def
    # share the "away offense" validity.
    weights = []
    losses = []
    weights.append(home_mask * h_valid_off)
    losses.append(err_home_off * home_mask * h_valid_off)
    weights.append(home_mask * a_valid_off)
    losses.append(err_home_def * home_mask * a_valid_off)
    weights.append(away_mask * a_valid_off)
    losses.append(err_away_off * away_mask * a_valid_off)
    weights.append(away_mask * h_valid_off)
    losses.append(err_away_def * away_mask * h_valid_off)

    total_loss = sum(loss.sum() for loss in losses)
    total_w = sum(w.sum() for w in weights).clamp_min(1.0)
    return total_loss / total_w


def margin_nll_loss(out: dict, batch: dict) -> torch.Tensor:
    mu = out["margin_mu"]
    log_sigma = out["margin_log_sigma"].clamp(min=0.0, max=5.0)
    sq_err = (batch["margin"] - mu) ** 2
    inv_var = (-2.0 * log_sigma).exp()
    return 0.5 * (sq_err * inv_var + 2.0 * log_sigma).mean()


def win_bce_loss(out: dict, batch: dict) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        out["win_logit"], batch["label"], reduction="mean"
    )


def calibration_residual_l2_loss(out: dict) -> torch.Tensor:
    """Keep the post-tail-gate context residual small."""
    return out["win_residual"].pow(2).mean()


def matchup_evidence_l2_loss(out: dict) -> torch.Tensor:
    """Keep the lineup/matchup evidence residual small and interpretable."""
    return out["matchup_evidence_residual"].pow(2).mean()


def uncertainty_temperature_l2_loss(out: dict) -> torch.Tensor:
    """Avoid gratuitous logit damping unless the data pays for it."""
    return out["uncertainty_temperature_delta"].pow(2).mean()


def season_calibration_l2_loss(out: dict) -> torch.Tensor:
    """Keep the phase-only logit adjustment gentle and data-driven."""
    return out["season_logit_adjustment"].pow(2).mean()


def calibration_slope_l2_loss(out: dict) -> torch.Tensor:
    """Soft prior around slope=1.0 in log space.

    This keeps the calibrated-margin slope trainable without letting it become
    an arbitrary confidence re-scaling knob on small windows.
    """
    return out["log_win_logit_slope"].pow(2)


def total_loss(
    out: dict, batch: dict,
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
    matchup_evidence_reg_w: float = 0.05,
    uncertainty_temp_reg_w: float = 0.02,
    season_calibration_reg_w: float = 0.02,
    calibration_slope_reg_w: float = 0.001,
) -> tuple[torch.Tensor, dict]:
    """Eleven-level v5 loss.

        L = team_w * L_team + player_w * L_player + pair_w * L_pair
          + inv_w * L_invol_mse + win_w * L_win + margin_nll_w * L_margin_nll
          + calibration_reg_w * L_calibration_reg
          + matchup_evidence_reg_w * L_matchup_evidence_reg
          + uncertainty_temp_reg_w * L_uncertainty_temp_reg
          + season_calibration_reg_w * L_season_calibration_reg
          + calibration_slope_reg_w * L_calibration_slope_reg

    The new term vs. v2 is L_invol_mse, which anchors the involvement
    softmax outputs to observed per-player exposure-share. Without it,
    the softmax shares are only constrained indirectly through downstream
    box-score gradients, which is the channel CME-v2 uses (and which
    leaves shares free to drift away from real minutes allocation).
    """
    pair_nll, n_pairs = pair_poisson_loss(out, batch)
    player_mse = player_mse_loss(out, batch)
    team_mse = team_mse_loss(out, batch)
    win_bce = win_bce_loss(out, batch)
    inv_mse = involvement_mse_loss(out, batch)

    box_w = box_weights.to(player_mse)
    pair_w_t = pair_weights.to(pair_nll)

    L_team = (box_w * team_mse).sum()
    L_player = (box_w * player_mse).sum() if player_mse.numel() > 0 else torch.zeros((), device=team_mse.device)
    L_pair = (pair_w_t * pair_nll).sum()
    L_win = win_bce
    L_inv = inv_mse
    L_margin_nll = margin_nll_loss(out, batch) if margin_nll_w > 0 else torch.zeros((), device=team_mse.device)
    L_calibration_reg = calibration_residual_l2_loss(out)
    L_matchup_evidence_reg = matchup_evidence_l2_loss(out)
    L_uncertainty_temp_reg = uncertainty_temperature_l2_loss(out)
    L_season_calibration_reg = season_calibration_l2_loss(out)
    L_calibration_slope_reg = calibration_slope_l2_loss(out)

    total = (team_w * L_team + player_w * L_player + pair_w * L_pair
             + inv_w * L_inv + win_w * L_win + margin_nll_w * L_margin_nll
             + calibration_reg_w * L_calibration_reg
             + matchup_evidence_reg_w * L_matchup_evidence_reg
             + uncertainty_temp_reg_w * L_uncertainty_temp_reg
             + season_calibration_reg_w * L_season_calibration_reg
             + calibration_slope_reg_w * L_calibration_slope_reg)
    return total, {
        "team_mse_per_target": team_mse.detach(),
        "player_mse_per_target": player_mse.detach(),
        "pair_nll_per_target": pair_nll.detach(),
        "L_team": L_team.detach(),
        "L_player": L_player.detach(),
        "L_pair": L_pair.detach(),
        "L_inv": L_inv.detach(),
        "L_win": L_win.detach(),
        "L_margin_nll": L_margin_nll.detach(),
        "L_calibration_reg": L_calibration_reg.detach(),
        "L_matchup_evidence_reg": L_matchup_evidence_reg.detach(),
        "L_uncertainty_temp_reg": L_uncertainty_temp_reg.detach(),
        "L_season_calibration_reg": L_season_calibration_reg.detach(),
        "L_calibration_slope_reg": L_calibration_slope_reg.detach(),
        "n_pairs_valid": n_pairs.detach(),
    }


# ----------------------------- smoke ----------------------------- #


def _smoke() -> None:
    torch.manual_seed(0)
    cfg = CmeV5Config(
        vocab_size=200, num_teams=30, d=64, tabular_dim=4,
        team_emb_dim=8, player_stat_dim=10,
    )
    model = CmeV5(cfg)
    B, Lh, La = 4, 14, 16
    home_idx = torch.randint(1, cfg.vocab_size + 1, (B, Lh))
    away_idx = torch.randint(1, cfg.vocab_size + 1, (B, La))
    home_mask = torch.ones(B, Lh, dtype=torch.bool); home_mask[0, 12:] = False
    away_mask = torch.ones(B, La, dtype=torch.bool); away_mask[1, 14:] = False

    home_prob = torch.rand(B, Lh) * home_mask.to(torch.float32)
    away_prob = torch.rand(B, La) * away_mask.to(torch.float32)

    # synthetic actual shares: a random Dirichlet over masked slots
    def _rand_share(mask):
        x = torch.rand_like(mask, dtype=torch.float32) * mask.to(torch.float32)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    home_alpha_off_actual = _rand_share(home_mask)
    home_alpha_def_actual = _rand_share(home_mask)
    away_alpha_off_actual = _rand_share(away_mask)
    away_alpha_def_actual = _rand_share(away_mask)

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
        "team_box_home": team_box_home,
        "team_box_away": team_box_away,
        "sup_pair_game": sup_pair_game, "sup_pair_side": sup_pair_side,
        "sup_pair_off": sup_pair_off, "sup_pair_def": sup_pair_def,
        "sup_pair_y": sup_pair_y,
        "sup_pl_game": sup_pl_game, "sup_pl_side": sup_pl_side,
        "sup_pl_slot": sup_pl_slot, "sup_pl_y": sup_pl_y,
        "home_alpha_off_actual": home_alpha_off_actual,
        "home_alpha_def_actual": home_alpha_def_actual,
        "away_alpha_off_actual": away_alpha_off_actual,
        "away_alpha_def_actual": away_alpha_def_actual,
        "home_off_exposure_valid": torch.ones(B, dtype=torch.float32),
        "away_off_exposure_valid": torch.ones(B, dtype=torch.float32),
    }

    out = model(batch)
    print("=== shapes ===")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:24s} {tuple(v.shape)}")

    print(f"\nhome_points  {[f'{x:.2f}' for x in out['home_points'].tolist()]}")
    print(f"away_points  {[f'{x:.2f}' for x in out['away_points'].tolist()]}")
    print(f"margin_mu    {[f'{x:.2f}' for x in out['margin_mu'].tolist()]}")
    print(f"N            {[f'{x:.2f}' for x in out['N'].tolist()]}")
    print(f"global_scale {out['global_scale'].item():.3f}")

    # Sinkhorn marginal check
    P_A = out["P_A"]
    row_actual = P_A[0].sum(dim=1)  # batch 0
    row_target = out["N"][0] * out["alpha_home_off"][0]
    masked_diff = (row_actual - row_target)[home_mask[0]].abs().max().item()
    print(f"\nSinkhorn batch 0 row marginal max err = {masked_diff:.3e}")

    # alpha sums-to-1 check (over mask)
    alpha_sum = (out["alpha_home_off"] * home_mask.to(out["alpha_home_off"].dtype)).sum(dim=-1)
    print(f"alpha_home_off sums (should be 1) = {alpha_sum.tolist()}")

    box_w = torch.ones(K_BOX); box_w[BOX_INDEX["pts"]] = 3.0
    pair_w = torch.ones(K_PAIR)
    loss, diag = total_loss(out, batch, box_weights=box_w, pair_weights=pair_w, margin_nll_w=0.1)
    print(f"\nloss = {loss.item():.4f}")
    for k in ("L_team", "L_player", "L_pair", "L_inv", "L_win", "L_margin_nll", "L_calibration_reg", "L_season_calibration_reg", "L_calibration_slope_reg"):
        print(f"  {k:15s} = {diag[k].item():.4f}")
    print(f"  {'win_slope':15s} = {out['win_logit_slope'].item():.4f}")

    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    n_total = sum(1 for _ in model.parameters())
    print(f"\nbackward ok: {n_grad}/{n_total} parameters got gradients")


if __name__ == "__main__":
    _smoke()
