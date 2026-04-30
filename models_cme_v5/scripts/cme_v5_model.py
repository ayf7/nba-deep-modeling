"""CME-v5 -- cascaded world model with deeply supervised attention stages.

Three attention stages form a causal chain, each with its own supervised head
whose output is injected back into the token stream before the next stage:

  Tokenize [Th, Ph1..Phn, Ta, Pa1..Pam] + E_side

  Stage 1 — rotation (who plays when?)
    self-attention → rotation_head (B,L,48) σ per-minute on-court presence
    L_rotation MSE, derived: effective_min = Σ_s rotation[i,s]
    inject rotation embeddings → token stream

  Stage 2 — tempo + roles (how fast, what roles?)
    self-attention → pace_head (B,) per-team possessions
                   → off_proj / def_proj role-split representations
    L_pace MSE
    inject pace + role embeddings → token stream

  Stage 3 — matchups (how do matchups go?)
    self-attention → score/eff heads → per-segment Sinkhorn
    L_pair Poisson

  Composition:
    pair exposure E[i,j] = Σ_s Sinkhorn(scores, rotation × pace)
    λ[i,j,k] = E[i,j] × rate[i,j,k]
    player_box = assemble(pair_marginals, nonpair_rates × eff_min)
    team_box = Σ_i player_box[i]
    win_logit = (home_pts - away_pts) / γ
    L_player Poisson, L_team Poisson, L_win BCE
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

V3_SCRIPTS = Path(__file__).resolve().parents[2] / "models_cme_v3" / "scripts"
sys.path.insert(0, str(V3_SCRIPTS))
from cme_v3_model import (  # noqa: E402
    SelfBlock,
    masked_softmax,
    _zero_last_linear,
)

V2_SCRIPTS = Path(__file__).resolve().parents[2] / "models_cme_v2" / "scripts"
sys.path.insert(0, str(V2_SCRIPTS))
from cme_v2_common import (  # noqa: E402
    BOX_INDEX,
    K_BOX,
    K_PAIR,
    K_PLAYER,
    PAIR_TARGETS,
    PLAYER_TARGETS,
    BOX_TARGETS,
)

# ---- constants ---- #
TEAM_TOTAL_MINUTES = 240.0
TEAM_BASE_POSSESSIONS = 100.0
MAX_CAREER_YEAR = 25
N_SEGMENTS = 48

# Pair-target channel indices
_PAIR_FGM_IDX = PAIR_TARGETS.index("matchup_fgm")
_PAIR_3PM_IDX = PAIR_TARGETS.index("matchup_3pm")
_PAIR_FGA_IDX = PAIR_TARGETS.index("matchup_fga")
_PAIR_3PA_IDX = PAIR_TARGETS.index("matchup_3pa")
_PAIR_AST_IDX = PAIR_TARGETS.index("matchup_assists")
_PAIR_TOV_IDX = PAIR_TARGETS.index("matchup_turnovers")
_PAIR_BLK_IDX = PAIR_TARGETS.index("matchup_blocks")
_PLAYER_FTM_IDX = PLAYER_TARGETS.index("ftm")
_PLAYER_FTA_IDX = PLAYER_TARGETS.index("fta")
_PLAYER_OREB_IDX = PLAYER_TARGETS.index("oreb")
_PLAYER_DREB_IDX = PLAYER_TARGETS.index("dreb")
_PLAYER_STL_IDX = PLAYER_TARGETS.index("stl")
_PLAYER_PF_IDX = PLAYER_TARGETS.index("pf")

# Box-stat assembly: maps box-score stat -> (source, channel_idx)
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


# ---- RoPE tables ---- #

def _build_rope_tables(n_positions: int, d: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) of shape (n_positions, d/2). Row 0 -> cos=1, sin=0 (identity rotation)."""
    assert d % 2 == 0, "RoPE requires even d"
    pos = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)  # (n, 1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(base) / d))  # (d/2,)
    angles = pos * div  # (n, d/2)
    return torch.cos(angles), torch.sin(angles)


# ---- config ---- #

@dataclass
class CmeV5Config:
    vocab_size: int
    num_teams: int
    d: int = 64
    n_heads: int = 4
    n_layers_stage1: int = 1
    n_layers_stage2: int = 1
    n_layers_stage3: int = 1
    pair_hidden: int = 96
    player_hidden: int = 64
    head_hidden: int = 64
    rotation_hidden: int = 64
    dropout: float = 0.1
    pair_dropout: float = 0.2
    player_dropout: float = 0.0
    tabular_dim: int = 0
    team_emb_dim: int = 16
    player_stat_dim: int = 0
    score_clamp: float = 12.0
    rate_clamp: float = 8.0
    sinkhorn_iters: int = 8
    n_segments: int = N_SEGMENTS
    base_possessions_per_team: float = TEAM_BASE_POSSESSIONS
    team_total_minutes: float = TEAM_TOTAL_MINUTES
    init_global_scale: float = 12.0
    max_career_year: int = MAX_CAREER_YEAR
    career_year_mode: str = "rope"
    career_year_pe_base: float = 10000.0
    use_player_scalars: bool = False
    use_direct_player_head: bool = False
    direct_head_separate_emb: bool = False
    direct_emb_dim: int = 32


# ----------------------------- model ----------------------------- #


class CmeV5(nn.Module):
    def __init__(self, cfg: CmeV5Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d

        # ---- token + ctx ----
        self.E_player = nn.Embedding(cfg.vocab_size + 1, d, padding_idx=0)
        nn.init.normal_(self.E_player.weight, std=0.02)
        with torch.no_grad():
            self.E_player.weight[0].zero_()

        # Career-year embedding (RoPE only for v5)
        mode = cfg.career_year_mode
        if mode == "rope":
            cos_t, sin_t = _build_rope_tables(cfg.max_career_year + 1, d, base=cfg.career_year_pe_base)
            self.register_buffer("career_year_rope_cos", cos_t)
            self.register_buffer("career_year_rope_sin", sin_t)
        elif mode == "none":
            pass
        else:
            raise ValueError(f"unsupported career_year_mode for v5: {mode!r} (only 'rope' and 'none')")

        self.E_team = nn.Embedding(cfg.num_teams + 1, cfg.team_emb_dim, padding_idx=0)
        nn.init.normal_(self.E_team.weight, std=0.02)
        with torch.no_grad():
            self.E_team.weight[0].zero_()

        self.E_side = nn.Embedding(2, d)
        nn.init.normal_(self.E_side.weight, std=0.02)

        # Per-player scalar biases (optional)
        if cfg.use_player_scalars:
            self.E_player_min = nn.Embedding(cfg.vocab_size + 1, 1, padding_idx=0)
            nn.init.zeros_(self.E_player_min.weight)
            self.E_player_rate = nn.Embedding(cfg.vocab_size + 1, 1, padding_idx=0)
            nn.init.zeros_(self.E_player_rate.weight)
        else:
            self.E_player_min = None
            self.E_player_rate = None

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

        # ---- Stage 1: rotation ----
        self.stage1_layers = nn.ModuleList([
            SelfBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers_stage1)
        ])
        self.rotation_head = nn.Sequential(
            nn.Linear(2 * d, cfg.rotation_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.rotation_hidden, cfg.n_segments),
        )
        self.rotation_bridge = nn.Sequential(
            nn.Linear(d + cfg.n_segments, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
        )

        # ---- Stage 2: tempo + roles ----
        self.stage2_layers = nn.ModuleList([
            SelfBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers_stage2)
        ])
        self.pace_head = nn.Sequential(
            nn.Linear(2 * d, cfg.head_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        _zero_last_linear(self.pace_head)
        self.role_proj_off = nn.Linear(d, d)
        self.role_proj_def = nn.Linear(d, d)
        self.tempo_bridge = nn.Sequential(
            nn.Linear(d + 1 + 2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
        )

        # ---- Stage 3: matchups ----
        self.stage3_layers = nn.ModuleList([
            SelfBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers_stage3)
        ])
        self.off_proj = nn.Linear(d, d)
        self.def_proj = nn.Linear(d, d)

        # ---- score head (for Sinkhorn) ----
        self.score_q = nn.Linear(d, d)
        self.score_k = nn.Linear(d, d)
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * d, cfg.pair_hidden), nn.GELU(), nn.Dropout(cfg.pair_dropout),
            nn.Linear(cfg.pair_hidden, 1),
        )
        _zero_last_linear(self.score_mlp)

        # ---- efficiency head: per (off, def, channel) per-poss rate ----
        self.eff_mlp = nn.Sequential(
            nn.Linear(2 * d + 1, cfg.pair_hidden), nn.GELU(), nn.Dropout(cfg.pair_dropout),
            nn.Linear(cfg.pair_hidden, K_PAIR),
        )
        _zero_last_linear(self.eff_mlp)
        _eff_priors = torch.tensor([
            1.0,    # exposure (forced to 1 in composition)
            0.225,  # player_points (~110 FG-points / 491 player-poss)
            0.082,  # matchup_fgm  (~40 / 491)
            0.176,  # matchup_fga  (~86 / 491)
            0.027,  # matchup_3pm  (~13 / 491)
            0.075,  # matchup_3pa  (~37 / 491)
            0.051,  # matchup_assists (~25 / 491)
            0.029,  # matchup_turnovers (~14 / 491)
            0.010,  # matchup_blocks (~5 / 491)
        ])
        assert _eff_priors.numel() == K_PAIR
        self.eff_bias = nn.Parameter(_eff_priors.log())

        # ---- player non-pair head: per slot x K_PLAYER, per-minute rate ----
        _player_per_min_priors = torch.tensor([
            17.0,   # ftm / team / game
            22.0,   # fta
            10.0,   # oreb
            36.0,   # dreb
            7.0,    # stl
            20.0,   # pf
        ]) / cfg.team_total_minutes
        assert _player_per_min_priors.numel() == K_PLAYER
        self.player_mlp = nn.Sequential(
            nn.Linear(d, cfg.player_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.player_hidden, K_PLAYER),
        )
        _zero_last_linear(self.player_mlp)
        self.player_bias = nn.Parameter(_player_per_min_priors.log())

        # ---- direct player box head ----
        if cfg.use_direct_player_head:
            if cfg.direct_head_separate_emb:
                self.E_player_scoring = nn.Embedding(cfg.vocab_size + 1, cfg.direct_emb_dim, padding_idx=0)
                nn.init.normal_(self.E_player_scoring.weight, std=0.02)
                with torch.no_grad():
                    self.E_player_scoring.weight[0].zero_()
                direct_in_dim = cfg.direct_emb_dim
            else:
                self.E_player_scoring = None
                direct_in_dim = d + 1
            self.direct_player_head = nn.Sequential(
                nn.Linear(direct_in_dim, cfg.player_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.player_hidden, K_BOX),
            )
            _zero_last_linear(self.direct_player_head)
        else:
            self.direct_player_head = None
            self.E_player_scoring = None

        # ---- win_logit scale ----
        self.global_scale = nn.Parameter(
            torch.tensor(math.log(cfg.init_global_scale)), requires_grad=False,
        )

    # ------------------------ tokenize ------------------------ #

    def _apply_rope_year(self, e: torch.Tensor, year_idx: torch.Tensor) -> torch.Tensor:
        cos = self.career_year_rope_cos[year_idx]  # (..., d/2)
        sin = self.career_year_rope_sin[year_idx]
        e1 = e[..., 0::2]
        e2 = e[..., 1::2]
        e_out = torch.empty_like(e)
        e_out[..., 0::2] = e1 * cos - e2 * sin
        e_out[..., 1::2] = e1 * sin + e2 * cos
        return e_out

    def _tokenize(self, idx: torch.Tensor, stats: torch.Tensor,
                  career_year_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = self.E_player(idx)
        if self.cfg.career_year_mode == "rope":
            e = self._apply_rope_year(e, career_year_idx)
        if self.feat_mlp is not None and stats.size(-1) > 0:
            e = e + self.feat_mlp(stats)
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

    # ------------------------ heads ------------------------ #

    def _rotation(self, h: torch.Tensor, ctx: torch.Tensor,
                  mask: torch.Tensor, idx: torch.Tensor | None = None,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-slot rotation curve. Returns (pred, logits).

        pred: (B, L, S) sigmoid values in [0, 1], masked.
        logits: (B, L, S) pre-sigmoid, masked. Used for BCE loss.
        """
        B, L, _ = h.shape
        S = self.cfg.n_segments
        ctx_e = ctx.unsqueeze(1).expand(-1, L, -1)
        raw = self.rotation_head(torch.cat([h, ctx_e], dim=-1))  # (B, L, S)
        if self.E_player_min is not None and idx is not None:
            raw = raw + self.E_player_min(idx)  # (B, L, 1) broadcast to (B, L, S)
        mask_f = mask.unsqueeze(-1).to(raw.dtype)
        logits = raw * mask_f
        pred = torch.sigmoid(raw) * mask_f
        return pred, logits

    def _pace(self, own_ctx: torch.Tensor, opp_ctx: torch.Tensor) -> torch.Tensor:
        """Per-team possessions per game (softplus + base prior)."""
        raw = self.pace_head(torch.cat([own_ctx, opp_ctx], dim=-1)).squeeze(-1)
        return F.softplus(raw) + self.cfg.base_possessions_per_team

    def _compute_scores(
        self,
        off: torch.Tensor,   # (B, Lo, d) post-cross off rep
        def_: torch.Tensor,  # (B, Ld, d) post-cross def rep
    ) -> torch.Tensor:
        """Matchup score matrix (B, Lo, Ld) shared across all segments."""
        B, Lo, d = off.shape
        Ld = def_.size(1)
        Q = self.score_q(off)
        K = self.score_k(def_)
        s_dot = torch.einsum("bld,bmd->blm", Q, K) / math.sqrt(d)
        off_e = off.unsqueeze(2).expand(-1, -1, Ld, -1)
        def_e = def_.unsqueeze(1).expand(-1, Lo, -1, -1)
        pair_feat = torch.cat([off_e, def_e], dim=-1)
        s_mlp = self.score_mlp(pair_feat).squeeze(-1)
        score = s_dot + s_mlp
        return torch.clamp(score, min=-self.cfg.score_clamp, max=self.cfg.score_clamp)

    def _compute_rates(
        self,
        off: torch.Tensor,    # (B, Lo, d)
        def_: torch.Tensor,   # (B, Ld, d)
        score: torch.Tensor,  # (B, Lo, Ld)
        mask_outer: torch.Tensor,  # (B, Lo, Ld) bool
        off_idx: torch.Tensor | None = None,  # (B, Lo) for E_player_rate
    ) -> torch.Tensor:
        """Per-pair per-channel efficiency rates. Returns (B, Lo, Ld, K_PAIR)."""
        B, Lo, d = off.shape
        Ld = def_.size(1)
        off_e = off.unsqueeze(2).expand(-1, -1, Ld, -1)
        def_e = def_.unsqueeze(1).expand(-1, Lo, -1, -1)
        rate_in = torch.cat([off_e, def_e, score.unsqueeze(-1)], dim=-1)
        rate_raw = self.eff_mlp(rate_in) + self.eff_bias
        if self.E_player_rate is not None and off_idx is not None:
            off_scale = self.E_player_rate(off_idx).squeeze(-1)  # (B, Lo)
            rate_raw = rate_raw + off_scale.unsqueeze(2).unsqueeze(-1)
        rate_raw = torch.clamp(rate_raw, min=-12.0, max=self.cfg.rate_clamp)
        rate = torch.exp(rate_raw) * mask_outer.unsqueeze(-1).to(rate_raw.dtype)
        # Channel 0 (exposure) forced to 1
        rate_filled = rate.clone()
        rate_filled[..., 0] = 1.0
        rate_filled = rate_filled * mask_outer.unsqueeze(-1).to(rate_filled.dtype)
        return rate_filled

    def _per_segment_sinkhorn(
        self,
        score: torch.Tensor,          # (B, Lo, Ld) shared matchup scores
        rotation_off: torch.Tensor,    # (B, Lo, S) off-player presence curves
        rotation_def: torch.Tensor,    # (B, Ld, S) def-player presence curves
        mask_off: torch.Tensor,        # (B, Lo) bool
        mask_def: torch.Tensor,        # (B, Ld) bool
        pace_per_seg: torch.Tensor,    # (B,) pace / n_segments
    ) -> torch.Tensor:
        """Batched per-segment Sinkhorn. Returns game-level pair exposure E (B, Lo, Ld)."""
        B, Lo, Ld = score.shape
        S = self.cfg.n_segments

        # Row/col marginals per segment: rotation * pace_per_seg
        # rotation_off: (B, Lo, S), pace_per_seg: (B,) -> (B, 1, 1)
        pace_e = pace_per_seg.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        row_mass = rotation_off * pace_e  # (B, Lo, S)
        col_mass = rotation_def * pace_e  # (B, Ld, S)

        # Rescale per-segment marginals so row_sum = col_sum = 5 * pace_per_seg
        target_per_seg = (5.0 * pace_per_seg).unsqueeze(-1)  # (B, 1)
        row_total = row_mass.sum(dim=1)  # (B, S)
        col_total = col_mass.sum(dim=1)  # (B, S)
        row_scale = target_per_seg / row_total.clamp_min(1e-3).detach()  # (B, S)
        col_scale = target_per_seg / col_total.clamp_min(1e-3).detach()  # (B, S)
        row_mass = row_mass * row_scale.unsqueeze(1)  # (B, Lo, S) * (B, 1, S)
        col_mass = col_mass * col_scale.unsqueeze(1)  # (B, Ld, S) * (B, 1, S)

        # Expand scores across segments: (B, Lo, Ld) -> (B*S, Lo, Ld)
        scores_exp = score.unsqueeze(1).expand(B, S, Lo, Ld).reshape(B * S, Lo, Ld)

        # Reshape marginals: (B, Lo, S) -> (B*S, Lo)
        row_bat = row_mass.permute(0, 2, 1).reshape(B * S, Lo)
        col_bat = col_mass.permute(0, 2, 1).reshape(B * S, Ld)

        # Expand mask: (B, Lo, Ld) -> (B*S, Lo, Ld)
        mask_outer = mask_off.unsqueeze(2) & mask_def.unsqueeze(1)  # (B, Lo, Ld)
        mask_bat = mask_outer.unsqueeze(1).expand(B, S, Lo, Ld).reshape(B * S, Lo, Ld)

        # Sinkhorn
        P_bat = sinkhorn_transport(
            log_scores=scores_exp,
            row_mass=row_bat,
            col_mass=col_bat,
            mask=mask_bat,
            n_iter=self.cfg.sinkhorn_iters,
        )

        # Sum over segments -> game-level pair exposure
        E = P_bat.reshape(B, S, Lo, Ld).sum(dim=1)  # (B, Lo, Ld)
        return E

    def _player_nonpair(
        self,
        h: torch.Tensor,               # (B, L, d) post-self-attn
        effective_minutes: torch.Tensor,  # (B, L) sum of rotation curve
        mask: torch.Tensor,             # (B, L)
        idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-slot non-pair stats, scaled by effective minutes.

        Uses detached team-minute denominator for normalization (same as v4)."""
        raw = self.player_mlp(h) + self.player_bias
        if self.E_player_rate is not None and idx is not None:
            raw = raw + self.E_player_rate(idx).squeeze(-1).unsqueeze(-1)
        raw = torch.clamp(raw, min=-12.0, max=self.cfg.rate_clamp)
        rate_per_min = torch.exp(raw) * mask.unsqueeze(-1).to(raw.dtype)
        team_min = effective_minutes.sum(dim=-1, keepdim=True).clamp_min(1e-3).detach()  # (B, 1)
        scale = self.cfg.team_total_minutes / team_min
        return effective_minutes.unsqueeze(-1) * rate_per_min * scale.unsqueeze(-1)

    # ------------------------ forward ------------------------ #

    def forward(self, batch: dict) -> dict:
        home_mask = batch["home_mask"]
        away_mask = batch["away_mask"]
        if self.training and self.cfg.player_dropout > 0.0:
            p = self.cfg.player_dropout
            home_mask = home_mask & (torch.rand_like(home_mask, dtype=torch.float32) > p)
            away_mask = away_mask & (torch.rand_like(away_mask, dtype=torch.float32) > p)

        tabular = batch.get("tabular")
        B = home_mask.size(0)
        Lh = home_mask.size(1)
        La = away_mask.size(1)
        dev = home_mask.device

        # tokenize players
        home_tok = self._tokenize(
            batch["home_idx"], batch["home_stats"],
            batch["home_career_year_idx"], home_mask,
        )
        away_tok = self._tokenize(
            batch["away_idx"], batch["away_stats"],
            batch["away_career_year_idx"], away_mask,
        )

        # team tokens via ctx_proj: (B, d)
        home_team_tok = self._team_ctx(batch["home_team_idx"], tabular, batch["home_rest"])
        away_team_tok = self._team_ctx(batch["away_team_idx"], tabular, batch["away_rest"])

        # side embeddings
        side_home = self.E_side(torch.zeros(B, dtype=torch.long, device=dev))
        side_away = self.E_side(torch.ones(B, dtype=torch.long, device=dev))
        home_tok = home_tok + side_home.unsqueeze(1)
        away_tok = away_tok + side_away.unsqueeze(1)
        home_team_tok = home_team_tok + side_home
        away_team_tok = away_team_tok + side_away

        # direct player box head: separate scoring embedding (no interference with shared emb)
        if self.direct_player_head is not None and self.E_player_scoring is not None:
            home_score_emb = self.E_player_scoring(batch["home_idx"]) * home_mask.unsqueeze(-1).to(home_tok.dtype)
            away_score_emb = self.E_player_scoring(batch["away_idx"]) * away_mask.unsqueeze(-1).to(away_tok.dtype)
            home_box_direct = self.direct_player_head(home_score_emb) * home_mask.unsqueeze(-1).to(home_tok.dtype)
            away_box_direct = self.direct_player_head(away_score_emb) * away_mask.unsqueeze(-1).to(away_tok.dtype)

        # build unified sequence: [Th, Ph1..Phn, Ta, Pa1..Pam]
        seq = torch.cat([
            home_team_tok.unsqueeze(1),
            home_tok,
            away_team_tok.unsqueeze(1),
            away_tok,
        ], dim=1)  # (B, 2 + Lh + La, d)

        team_true = torch.ones(B, 1, dtype=torch.bool, device=dev)
        full_mask = torch.cat([team_true, home_mask, team_true, away_mask], dim=1)

        # ======== Stage 1: rotation ========
        for layer in self.stage1_layers:
            seq = layer(seq, full_mask)

        home_ctx1 = seq[:, 0]
        home_s1 = seq[:, 1:1 + Lh]
        away_ctx1 = seq[:, 1 + Lh]
        away_s1 = seq[:, 2 + Lh:]

        home_rotation, home_rotation_logits = self._rotation(home_s1, home_ctx1, home_mask, batch["home_idx"])
        away_rotation, away_rotation_logits = self._rotation(away_s1, away_ctx1, away_mask, batch["away_idx"])
        home_eff_min = home_rotation.sum(dim=-1)
        away_eff_min = away_rotation.sum(dim=-1)

        # direct player box head: post-Stage 1 branch (when no separate embedding)
        if self.direct_player_head is not None and self.E_player_scoring is None:
            home_direct_in = torch.cat([home_s1, home_eff_min.unsqueeze(-1)], dim=-1)
            away_direct_in = torch.cat([away_s1, away_eff_min.unsqueeze(-1)], dim=-1)
            home_box_direct = self.direct_player_head(home_direct_in) * home_mask.unsqueeze(-1).to(home_s1.dtype)
            away_box_direct = self.direct_player_head(away_direct_in) * away_mask.unsqueeze(-1).to(away_s1.dtype)

        # concat rotation curves to player tokens, project back to d
        home_bridged = self.rotation_bridge(torch.cat([home_s1, home_rotation], dim=-1))
        away_bridged = self.rotation_bridge(torch.cat([away_s1, away_rotation], dim=-1))
        seq = torch.cat([
            home_ctx1.unsqueeze(1), home_bridged,
            away_ctx1.unsqueeze(1), away_bridged,
        ], dim=1)

        # ======== Stage 2: tempo + roles ========
        for layer in self.stage2_layers:
            seq = layer(seq, full_mask)

        home_ctx2 = seq[:, 0]
        home_s2 = seq[:, 1:1 + Lh]
        away_ctx2 = seq[:, 1 + Lh]
        away_s2 = seq[:, 2 + Lh:]

        pace_home = self._pace(home_ctx2, away_ctx2)
        pace_away = self._pace(away_ctx2, home_ctx2)

        home_role_off = self.role_proj_off(home_s2) * home_mask.unsqueeze(-1).to(home_s2.dtype)
        home_role_def = self.role_proj_def(home_s2) * home_mask.unsqueeze(-1).to(home_s2.dtype)
        away_role_off = self.role_proj_off(away_s2) * away_mask.unsqueeze(-1).to(away_s2.dtype)
        away_role_def = self.role_proj_def(away_s2) * away_mask.unsqueeze(-1).to(away_s2.dtype)

        # concat pace + roles to player tokens, project back to d
        pace_h_e = pace_home.view(B, 1, 1).expand(-1, Lh, -1)
        pace_a_e = pace_away.view(B, 1, 1).expand(-1, La, -1)
        home_tempo_in = torch.cat([home_s2, pace_h_e, home_role_off, home_role_def], dim=-1)
        away_tempo_in = torch.cat([away_s2, pace_a_e, away_role_off, away_role_def], dim=-1)
        home_bridged2 = self.tempo_bridge(home_tempo_in)
        away_bridged2 = self.tempo_bridge(away_tempo_in)
        seq = torch.cat([
            home_ctx2.unsqueeze(1), home_bridged2,
            away_ctx2.unsqueeze(1), away_bridged2,
        ], dim=1)

        # ======== Stage 3: matchups ========
        for layer in self.stage3_layers:
            seq = layer(seq, full_mask)

        home_self = seq[:, 1:1 + Lh]
        away_self = seq[:, 2 + Lh:]

        # final role projections for matchup heads
        home_off = self.off_proj(home_self) * home_mask.unsqueeze(-1).to(home_self.dtype)
        home_def = self.def_proj(home_self) * home_mask.unsqueeze(-1).to(home_self.dtype)
        away_off = self.off_proj(away_self) * away_mask.unsqueeze(-1).to(away_self.dtype)
        away_def = self.def_proj(away_self) * away_mask.unsqueeze(-1).to(away_self.dtype)

        # matchup scores (shared across segments)
        score_A = self._compute_scores(home_off, away_def)  # (B, Lh, La)
        score_B = self._compute_scores(away_off, home_def)  # (B, La, Lh)

        # per-segment pace
        S = self.cfg.n_segments
        pace_home_per_seg = pace_home / S
        pace_away_per_seg = pace_away / S

        # direction A: home offense vs away defense
        E_A = self._per_segment_sinkhorn(
            score_A, home_rotation, away_rotation,
            home_mask, away_mask, pace_home_per_seg,
        )
        # direction B: away offense vs home defense
        E_B = self._per_segment_sinkhorn(
            score_B, away_rotation, home_rotation,
            away_mask, home_mask, pace_away_per_seg,
        )

        # efficiency rates
        mask_A = home_mask.unsqueeze(2) & away_mask.unsqueeze(1)
        mask_B = away_mask.unsqueeze(2) & home_mask.unsqueeze(1)
        rate_A = self._compute_rates(home_off, away_def, score_A, mask_A, batch["home_idx"])
        rate_B = self._compute_rates(away_off, home_def, score_B, mask_B, batch["away_idx"])

        # pair event rates: E * rate
        lam_A = E_A.unsqueeze(-1) * rate_A  # (B, Lh, La, K_PAIR)
        lam_B = E_B.unsqueeze(-1) * rate_B  # (B, La, Lh, K_PAIR)

        # row-sum over defenders -> per-off-player pair marginals
        home_pair_marg = lam_A.sum(dim=2)  # (B, Lh, K_PAIR)
        away_pair_marg = lam_B.sum(dim=2)  # (B, La, K_PAIR)

        # per-player non-pair stats
        home_player_np = self._player_nonpair(home_self, home_eff_min, home_mask, idx=batch["home_idx"])
        away_player_np = self._player_nonpair(away_self, away_eff_min, away_mask, idx=batch["away_idx"])

        # assemble box stats
        home_box = self._assemble_player_box(home_pair_marg, home_player_np)
        away_box = self._assemble_player_box(away_pair_marg, away_player_np)
        home_box = home_box * home_mask.unsqueeze(-1).to(home_box.dtype)
        away_box = away_box * away_mask.unsqueeze(-1).to(away_box.dtype)

        home_team_box = home_box.sum(dim=1)
        away_team_box = away_box.sum(dim=1)

        # win logit
        pts_idx = BOX_INDEX["pts"]
        home_pts = home_team_box[..., pts_idx]
        away_pts = away_team_box[..., pts_idx]
        margin_mu = home_pts - away_pts
        scale = self.global_scale.exp().clamp_min(1.0)
        win_logit = margin_mu / scale

        result = {
            # rotation
            "home_rotation": home_rotation,
            "away_rotation": away_rotation,
            "home_rotation_logits": home_rotation_logits,
            "away_rotation_logits": away_rotation_logits,
            "home_eff_min": home_eff_min,
            "away_eff_min": away_eff_min,
            # pace
            "pace_home": pace_home,
            "pace_away": pace_away,
            # pair
            "pair_dir_A": lam_A,
            "pair_dir_B": lam_B,
            "E_A": E_A,
            "E_B": E_B,
            "rate_A": rate_A,
            "rate_B": rate_B,
            "pair_mask_A": mask_A,
            "pair_mask_B": mask_B,
            # player / team
            "home_pair_marg": home_pair_marg,
            "away_pair_marg": away_pair_marg,
            "home_player_nonpair": home_player_np,
            "away_player_nonpair": away_player_np,
            "home_box": home_box,
            "away_box": away_box,
            "home_team_box": home_team_box,
            "away_team_box": away_team_box,
            # win
            "home_points": home_pts,
            "away_points": away_pts,
            "margin_mu": margin_mu,
            "win_logit": win_logit,
            "global_scale": scale,
            "home_ctx": home_ctx2,
            "away_ctx": away_ctx2,
        }

        if self.direct_player_head is not None:
            result["home_box_direct"] = home_box_direct
            result["away_box_direct"] = away_box_direct

        if self.training:
            h_self_d = home_self.detach()
            a_self_d = away_self.detach()
            h_off_d = self.off_proj(h_self_d) * home_mask.unsqueeze(-1).to(h_self_d.dtype)
            h_def_d = self.def_proj(h_self_d) * home_mask.unsqueeze(-1).to(h_self_d.dtype)
            a_off_d = self.off_proj(a_self_d) * away_mask.unsqueeze(-1).to(a_self_d.dtype)
            a_def_d = self.def_proj(a_self_d) * away_mask.unsqueeze(-1).to(a_self_d.dtype)
            rate_A_d = self._compute_rates(h_off_d, a_def_d,
                self._compute_scores(h_off_d, a_def_d), mask_A, batch["home_idx"])
            rate_B_d = self._compute_rates(a_off_d, h_def_d,
                self._compute_scores(a_off_d, h_def_d), mask_B, batch["away_idx"])
            lam_A_d = E_A.detach().unsqueeze(-1) * rate_A_d
            lam_B_d = E_B.detach().unsqueeze(-1) * rate_B_d
            h_pair_d = lam_A_d.sum(dim=2)
            a_pair_d = lam_B_d.sum(dim=2)
            h_np_d = self._player_nonpair(h_self_d, home_eff_min.detach(), home_mask, idx=batch["home_idx"])
            a_np_d = self._player_nonpair(a_self_d, away_eff_min.detach(), away_mask, idx=batch["away_idx"])
            h_box_d = self._assemble_player_box(h_pair_d, h_np_d)
            a_box_d = self._assemble_player_box(a_pair_d, a_np_d)
            h_box_d = h_box_d * home_mask.unsqueeze(-1).to(h_box_d.dtype)
            a_box_d = a_box_d * away_mask.unsqueeze(-1).to(a_box_d.dtype)
            result["home_box_sg"] = h_box_d
            result["away_box_sg"] = a_box_d

        return result

    def _assemble_player_box(
        self, pair_marginal: torch.Tensor, player_np: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = pair_marginal.shape
        out = torch.zeros(B, L, K_BOX, device=pair_marginal.device, dtype=pair_marginal.dtype)
        for box_name, (src, ch) in _BOX_SOURCE.items():
            i = BOX_INDEX[box_name]
            if src == "pair":
                out[..., i] = pair_marginal[..., ch]
            else:
                out[..., i] = player_np[..., ch]
        out[..., BOX_INDEX["pts"]] = (
            2.0 * pair_marginal[..., _PAIR_FGM_IDX]
            + pair_marginal[..., _PAIR_3PM_IDX]
            + player_np[..., _PLAYER_FTM_IDX]
        )
        return out


# ----------------------------- gathers ----------------------------- #


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


# ----------------------------- losses ----------------------------- #


def _poisson_nll(pred: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean over leading dims of Poisson NLL pred - y*log(pred)."""
    return (pred - y * torch.log(pred.clamp_min(eps))).mean()


def rotation_loss(out: dict, batch: dict) -> torch.Tensor:
    """Per-segment BCE on predicted rotation curves vs ground-truth presence.

    Supervises the full 48-dim on-court presence curve, not just the integral.
    Uses BCE with logits for numerical stability.
    """
    h_logits = out["home_rotation_logits"]  # (B, Lh, 48)
    a_logits = out["away_rotation_logits"]  # (B, La, 48)
    h_target = batch["home_rotation_target"]  # (B, Lh, 48)
    a_target = batch["away_rotation_target"]  # (B, La, 48)
    h_mask = batch["home_mask"].unsqueeze(-1).to(h_logits.dtype)
    a_mask = batch["away_mask"].unsqueeze(-1).to(a_logits.dtype)
    bce_h = F.binary_cross_entropy_with_logits(h_logits, h_target, reduction="none") * h_mask
    bce_a = F.binary_cross_entropy_with_logits(a_logits, a_target, reduction="none") * a_mask
    num = bce_h.sum() + bce_a.sum()
    den = (h_mask.sum() + a_mask.sum()).clamp_min(1.0)
    return num / den


def pace_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    h_valid = batch["home_minutes_valid"]
    a_valid = batch["away_minutes_valid"]
    err_h = (out["pace_home"] - batch["home_pace_actual"]) ** 2 * h_valid
    err_a = (out["pace_away"] - batch["away_pace_actual"]) ** 2 * a_valid
    den = (h_valid.sum() + a_valid.sum()).clamp_min(1.0)
    return (err_h.sum() + err_a.sum()) / den


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


def player_poisson_loss(out: dict, batch: dict, eps: float = 1e-6) -> torch.Tensor:
    pred = gather_player_box(out, batch["sup_pl_game"], batch["sup_pl_side"],
                             batch["sup_pl_slot"])
    y = batch["sup_pl_y"]
    if pred.size(0) == 0:
        return torch.zeros(K_BOX, device=pred.device, dtype=pred.dtype)
    return (pred - y * torch.log(pred.clamp_min(eps))).mean(dim=0)


def team_poisson_loss(out: dict, batch: dict, eps: float = 1e-6) -> torch.Tensor:
    home_pred = out["home_team_box"]
    away_pred = out["away_team_box"]
    home_y = batch["team_box_home"]
    away_y = batch["team_box_away"]
    nll_h = (home_pred - home_y * torch.log(home_pred.clamp_min(eps))).mean(dim=0)
    nll_a = (away_pred - away_y * torch.log(away_pred.clamp_min(eps))).mean(dim=0)
    return 0.5 * (nll_h + nll_a)


def win_bce_loss(out: dict, batch: dict) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        out["win_logit"], batch["label"], reduction="mean",
    )


def total_loss(
    out: dict, batch: dict,
    *,
    box_weights: torch.Tensor,
    pair_weights: torch.Tensor,
    rot_w: float = 1.0,
    pace_w: float = 0.01,
    pair_w: float = 0.0,
    player_w: float = 1.0,
    team_w: float = 0.01,
    win_w: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    L_rot = rotation_loss(out, batch)
    L_pace = pace_mse_loss(out, batch)
    pair_nll, n_pairs = pair_poisson_loss(out, batch)
    player_nll = player_poisson_loss(out, batch)
    team_nll = team_poisson_loss(out, batch)
    L_win = win_bce_loss(out, batch)

    box_w = box_weights.to(player_nll)
    pair_w_t = pair_weights.to(pair_nll)

    L_player = (box_w * player_nll).sum() if player_nll.numel() > 0 else torch.zeros((), device=L_rot.device)
    L_team = (box_w * team_nll).sum()
    L_pair = (pair_w_t * pair_nll).sum()

    total = (
        rot_w * L_rot
        + pace_w * L_pace
        + pair_w * L_pair
        + player_w * L_player
        + team_w * L_team
        + win_w * L_win
    )
    return total, {
        "L_rot": L_rot.detach(),
        "L_pace": L_pace.detach(),
        "L_pair": L_pair.detach(),
        "L_player": L_player.detach(),
        "L_team": L_team.detach(),
        "L_win": L_win.detach(),
        "player_nll_per_target": player_nll.detach(),
        "team_nll_per_target": team_nll.detach(),
        "pair_nll_per_target": pair_nll.detach(),
        "n_pairs_valid": n_pairs.detach(),
    }


# ----------------------- MSE loss variants ----------------------- #


def player_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    pred = gather_player_box(out, batch["sup_pl_game"], batch["sup_pl_side"],
                             batch["sup_pl_slot"])
    y = batch["sup_pl_y"]
    if pred.size(0) == 0:
        return torch.zeros(K_BOX, device=pred.device, dtype=pred.dtype)
    return ((pred - y) ** 2).mean(dim=0)


def direct_player_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    if "home_box_direct" not in out:
        dev = batch["sup_pl_y"].device if batch["sup_pl_y"].numel() > 0 else "cpu"
        return torch.zeros(K_BOX, device=dev)
    out_direct = {**out, "home_box": out["home_box_direct"], "away_box": out["away_box_direct"]}
    pred = gather_player_box(out_direct, batch["sup_pl_game"], batch["sup_pl_side"],
                             batch["sup_pl_slot"])
    y = batch["sup_pl_y"]
    if pred.size(0) == 0:
        return torch.zeros(K_BOX, device=pred.device, dtype=pred.dtype)
    return ((pred - y) ** 2).mean(dim=0)


def team_mse_loss(out: dict, batch: dict) -> torch.Tensor:
    home_pred = out["home_team_box"]
    away_pred = out["away_team_box"]
    home_y = batch["team_box_home"]
    away_y = batch["team_box_away"]
    mse_h = ((home_pred - home_y) ** 2).mean(dim=0)
    mse_a = ((away_pred - away_y) ** 2).mean(dim=0)
    return 0.5 * (mse_h + mse_a)


def pair_mse_loss(out: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    A = out["pair_dir_A"]
    B = out["pair_dir_B"]
    mask_A = out["pair_mask_A"].unsqueeze(-1).to(A.dtype)
    mask_B = out["pair_mask_B"].unsqueeze(-1).to(B.dtype)
    pred = gather_pair_pred(out, batch["sup_pair_game"], batch["sup_pair_side"],
                            batch["sup_pair_off"], batch["sup_pair_def"])
    y = batch["sup_pair_y"]
    n_pairs = out["pair_mask_A"].sum() + out["pair_mask_B"].sum()
    n_pairs = n_pairs.clamp_min(1).to(A.dtype)
    if pred.size(0) == 0:
        return torch.zeros(K_PAIR, device=A.device, dtype=A.dtype), n_pairs
    mse = ((pred - y) ** 2).mean(dim=0)
    return mse, n_pairs


# -------------------- stop-gradient player loss ------------------- #


def player_mse_loss_sg(out: dict, batch: dict) -> torch.Tensor:
    """MSE player loss with stop-gradient through the shared transformer.

    Uses home_box_sg / away_box_sg which are recomputed from detached
    transformer features — rate heads and player_mlp still get gradients,
    but the shared transformer backbone does not.
    """
    box_key_h = "home_box_sg" if "home_box_sg" in out else "home_box"
    box_key_a = "away_box_sg" if "away_box_sg" in out else "away_box"
    out_sg = {**out, "home_box": out[box_key_h], "away_box": out[box_key_a]}
    pred = gather_player_box(out_sg, batch["sup_pl_game"], batch["sup_pl_side"],
                             batch["sup_pl_slot"])
    y = batch["sup_pl_y"]
    if pred.size(0) == 0:
        return torch.zeros(K_BOX, device=pred.device, dtype=pred.dtype)
    return ((pred - y) ** 2).mean(dim=0)


# --------------------- loss normalizer (EMA) ---------------------- #


class LossNormalizer:
    """EMA-based loss magnitude normalizer.

    Tracks running mean of each loss and divides by it so all losses
    are ~1.0 before user weights are applied.
    """

    def __init__(self, keys: list[str], momentum: float = 0.1):
        self._ema: dict[str, float] = {k: 1.0 for k in keys}
        self._momentum = momentum
        self._initialized: set[str] = set()

    def normalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        raw = value.detach().item()
        if abs(raw) < 1e-10:
            return value
        if key not in self._initialized:
            self._ema[key] = raw
            self._initialized.add(key)
        else:
            self._ema[key] = (1 - self._momentum) * self._ema[key] + self._momentum * raw
        return value / max(self._ema[key], 1e-10)

    def stats(self) -> dict[str, float]:
        return dict(self._ema)


def total_loss_v2(
    out: dict, batch: dict,
    *,
    box_weights: torch.Tensor,
    pair_weights: torch.Tensor,
    rot_w: float = 1.0,
    pace_w: float = 0.0,
    pair_w: float = 0.1,
    player_w: float = 1.0,
    team_w: float = 0.0,
    win_w: float = 1.0,
    direct_w: float = 0.0,
    use_mse: bool = False,
    use_mse_player: bool = False,
    use_mse_pair: bool = False,
    stop_grad_player: bool = False,
    normalizer: LossNormalizer | None = None,
) -> tuple[torch.Tensor, dict]:
    L_rot = rotation_loss(out, batch)
    L_pace = pace_mse_loss(out, batch)
    L_win = win_bce_loss(out, batch)

    _mse_pair = use_mse or use_mse_pair
    _mse_player = use_mse or use_mse_player

    if _mse_pair:
        pair_raw, n_pairs = pair_mse_loss(out, batch)
    else:
        pair_raw, n_pairs = pair_poisson_loss(out, batch)

    if _mse_player:
        if stop_grad_player and player_w > 0:
            player_raw = player_mse_loss_sg(out, batch)
        else:
            player_raw = player_mse_loss(out, batch)
        team_raw = team_mse_loss(out, batch)
    else:
        player_raw = player_poisson_loss(out, batch)
        team_raw = team_poisson_loss(out, batch)

    box_w = box_weights.to(player_raw)
    pair_w_t = pair_weights.to(pair_raw)

    L_player = (box_w * player_raw).sum() if player_raw.numel() > 0 else torch.zeros((), device=L_rot.device)
    L_team = (box_w * team_raw).sum()
    L_pair = (pair_w_t * pair_raw).sum()

    if direct_w > 0 and "home_box_direct" in out:
        direct_raw = direct_player_mse_loss(out, batch)
        L_direct = (box_w * direct_raw).sum() if direct_raw.numel() > 0 else torch.zeros((), device=L_rot.device)
    else:
        L_direct = torch.zeros((), device=L_rot.device)

    losses = {
        "rot": L_rot, "pace": L_pace, "pair": L_pair,
        "player": L_player, "team": L_team, "win": L_win,
        "direct": L_direct,
    }
    weights = {
        "rot": rot_w, "pace": pace_w, "pair": pair_w,
        "player": player_w, "team": team_w, "win": win_w,
        "direct": direct_w,
    }

    if normalizer is not None:
        total = sum(
            weights[k] * normalizer.normalize(k, losses[k])
            for k in losses if weights[k] > 0
        )
    else:
        total = sum(weights[k] * losses[k] for k in losses)

    return total, {
        "L_rot": L_rot.detach(),
        "L_pace": L_pace.detach(),
        "L_pair": L_pair.detach(),
        "L_player": L_player.detach(),
        "L_direct": L_direct.detach(),
        "L_team": L_team.detach(),
        "L_win": L_win.detach(),
        "player_nll_per_target": player_raw.detach(),
        "team_nll_per_target": team_raw.detach(),
        "pair_nll_per_target": pair_raw.detach(),
        "n_pairs_valid": n_pairs.detach() if isinstance(n_pairs, torch.Tensor) else torch.tensor(0.0),
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

    home_minutes_actual = torch.rand(B, Lh) * 48 * home_mask.to(torch.float32)
    home_minutes_actual = home_minutes_actual / home_minutes_actual.sum(dim=-1, keepdim=True).clamp_min(1e-6) * 240
    away_minutes_actual = torch.rand(B, La) * 48 * away_mask.to(torch.float32)
    away_minutes_actual = away_minutes_actual / away_minutes_actual.sum(dim=-1, keepdim=True).clamp_min(1e-6) * 240

    sup_pair_game = torch.tensor([0, 0, 1, 2, 3])
    sup_pair_side = torch.tensor([0, 1, 0, 0, 1])
    sup_pair_off = torch.tensor([0, 0, 1, 2, 3])
    sup_pair_def = torch.tensor([0, 1, 2, 3, 4])
    sup_pair_y = torch.rand(5, K_PAIR) * 3.0

    sup_pl_game = torch.tensor([0, 0, 1, 2, 3, 3])
    sup_pl_side = torch.tensor([0, 1, 0, 1, 0, 1])
    sup_pl_slot = torch.tensor([0, 1, 2, 3, 4, 5])
    sup_pl_y = torch.rand(6, K_BOX) * 5.0

    batch = {
        "home_idx": home_idx, "away_idx": away_idx,
        "home_mask": home_mask, "away_mask": away_mask,
        "home_career_year_idx": torch.randint(1, cfg.max_career_year + 1, (B, Lh)),
        "away_career_year_idx": torch.randint(1, cfg.max_career_year + 1, (B, La)),
        "home_stats": torch.randn(B, Lh, cfg.player_stat_dim),
        "away_stats": torch.randn(B, La, cfg.player_stat_dim),
        "home_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "away_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "home_rest": torch.rand(B), "away_rest": torch.rand(B),
        "tabular": torch.randn(B, cfg.tabular_dim),
        "label": torch.randint(0, 2, (B,), dtype=torch.float32),
        "margin": torch.randn(B) * 10,
        "team_box_home": torch.rand(B, K_BOX) * 30,
        "team_box_away": torch.rand(B, K_BOX) * 30,
        "sup_pair_game": sup_pair_game, "sup_pair_side": sup_pair_side,
        "sup_pair_off": sup_pair_off, "sup_pair_def": sup_pair_def,
        "sup_pair_y": sup_pair_y,
        "sup_pl_game": sup_pl_game, "sup_pl_side": sup_pl_side,
        "sup_pl_slot": sup_pl_slot, "sup_pl_y": sup_pl_y,
        "home_rotation_target": torch.rand(B, Lh, 48) * home_mask.unsqueeze(-1).float(),
        "away_rotation_target": torch.rand(B, La, 48) * away_mask.unsqueeze(-1).float(),
        "home_minutes_actual": home_minutes_actual,
        "away_minutes_actual": away_minutes_actual,
        "home_pace_actual": torch.full((B,), 100.0),
        "away_pace_actual": torch.full((B,), 100.0),
        "home_minutes_valid": torch.ones(B),
        "away_minutes_valid": torch.ones(B),
    }

    out = model(batch)
    print("=== shapes ===")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:24s} {tuple(v.shape)}")

    print(f"\nhome_eff_min sum (per-game): "
          f"{[f'{s:.1f}' for s in out['home_eff_min'].sum(dim=-1).tolist()]}")
    print(f"away_eff_min sum (per-game): "
          f"{[f'{s:.1f}' for s in out['away_eff_min'].sum(dim=-1).tolist()]}")
    print(f"home_rotation shape: {tuple(out['home_rotation'].shape)}")
    print(f"home_rotation[0,0,:5] (first player, first 5 segments): "
          f"{[f'{x:.3f}' for x in out['home_rotation'][0, 0, :5].tolist()]}")
    print(f"pace_home: {[f'{p:.1f}' for p in out['pace_home'].tolist()]}")
    print(f"pace_away: {[f'{p:.1f}' for p in out['pace_away'].tolist()]}")
    print(f"home_points: {[f'{p:.1f}' for p in out['home_points'].tolist()]}")
    print(f"away_points: {[f'{p:.1f}' for p in out['away_points'].tolist()]}")
    print(f"margin_mu  : {[f'{m:.1f}' for m in out['margin_mu'].tolist()]}")

    # Sinkhorn game-level exposure check
    E_A = out["E_A"]
    row_actual = E_A[0].sum(dim=1)
    print(f"\nE_A[0] row sums (per-off-player total exposure): "
          f"{[f'{x:.2f}' for x in row_actual[:5].tolist()]}...")
    print(f"E_A total (should be ~ 5 * pace_home * 1.0): {E_A[0].sum().item():.2f}")

    box_w = torch.ones(K_BOX); box_w[BOX_INDEX["pts"]] = 3.0
    pair_w = torch.ones(K_PAIR)
    loss, diag = total_loss(out, batch, box_weights=box_w, pair_weights=pair_w, pair_w=0.1)
    print(f"\nloss = {loss.item():.4f}")
    for k in ("L_rot", "L_pace", "L_pair", "L_player", "L_team", "L_win"):
        print(f"  {k:10s} = {diag[k].item():.4f}")

    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    n_total = sum(1 for _ in model.parameters())
    print(f"\nbackward ok: {n_grad}/{n_total} parameters got gradients")


if __name__ == "__main__":
    _smoke()
