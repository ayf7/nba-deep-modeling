"""CME-v6 encoder-decoder model.

Encoder: player + team embeddings → self-attention over the roster.
Decoder: cumulative game-state decoder with RoPE self-attention and
         per-minute cross-attention to player tokens augmented with
         cumulative box scores. Rotation masking ensures only on-court
         players are visible in cross-attention.
Outputs: per-minute rotation predictions (dot product D @ P_aug^T),
         per-minute cumulative box score predictions (D * P_aug → Linear),
         win logit from final-minute predicted pts margin / gamma.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


N_SEGMENTS = 48
K_BOX = 14


def _build_rope_tables(n_positions: int, d: int, base: float = 10000.0):
    assert d % 2 == 0
    pos = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(base) / d))
    angles = pos * div
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    S = x.size(-2)
    cos = cos[:S].unsqueeze(0).unsqueeze(0)
    sin = sin[:S].unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


@dataclass
class CmeV6Config:
    vocab_size: int
    num_teams: int
    tabular_dim: int = 0
    d: int = 64
    n_heads: int = 4
    n_enc: int = 4
    n_dec: int = 4
    dropout: float = 0.1
    n_segments: int = N_SEGMENTS
    player_stats_dim: int = 0
    max_career_years: int = 25
    use_player_embeddings: bool = True
    n_box_channels: int = K_BOX
    box_delta_hidden: int = 128  # unused, kept for ckpt compat
    # Legacy fields kept so old checkpoints/CLI args don't error on construction.
    score_k: int = 16
    bilinear_init: str = "xavier"
    rate_clamp: float = 0.0
    emit_channels: tuple[str, ...] = ("pts",)
    pl_rotation: bool = False
    slot_aware_input: bool = False
    score_clamp: float = 15.0
    pl_tau: float = 1.0
    side_on_team_only: bool = False
    direct_win: str = "off"
    decoder_mode: str = "cumstate"


class EncoderBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d, d), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class CumStateDecoderBlock(nn.Module):
    """RoPE self-attention + per-minute rotation-masked cross-attention + FFN."""

    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.norm_sa = nn.LayerNorm(d)
        self.Wq_sa = nn.Linear(d, d, bias=False)
        self.Wk_sa = nn.Linear(d, d, bias=False)
        self.Wv_sa = nn.Linear(d, d, bias=False)
        self.Wo_sa = nn.Linear(d, d, bias=False)
        self.drop_sa = nn.Dropout(dropout)
        self.norm_ca = nn.LayerNorm(d)
        self.Wq_ca = nn.Linear(d, d, bias=False)
        self.Wk_ca = nn.Linear(d, d, bias=False)
        self.Wv_ca = nn.Linear(d, d, bias=False)
        self.Wo_ca = nn.Linear(d, d, bias=False)
        self.drop_ca = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d, d), nn.Dropout(dropout),
        )

    def forward(
        self,
        d_seq: torch.Tensor,       # (B, S, d)
        p_aug: torch.Tensor,        # (B, S, L, d)
        causal_mask: torch.Tensor,  # (S, S)
        ca_mask: torch.Tensor,      # (B, S, L) bool
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, S, d = d_seq.shape
        H, hd = self.n_heads, self.head_dim
        L = p_aug.size(2)

        h = self.norm_sa(d_seq)
        q = self.Wq_sa(h).view(B, S, H, hd).transpose(1, 2)
        k = self.Wk_sa(h).view(B, S, H, hd).transpose(1, 2)
        v = self.Wv_sa(h).view(B, S, H, hd).transpose(1, 2)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        sa = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
        sa = sa.transpose(1, 2).reshape(B, S, d)
        d_seq = d_seq + self.drop_sa(self.Wo_sa(sa))

        h = self.norm_ca(d_seq)
        q_ca = self.Wq_ca(h).view(B, S, H, hd)
        q_ca = q_ca.reshape(B * S, 1, H, hd).transpose(1, 2)
        kv_in = p_aug.reshape(B * S, L, d)
        k_ca = self.Wk_ca(kv_in).view(B * S, L, H, hd).transpose(1, 2)
        v_ca = self.Wv_ca(kv_in).view(B * S, L, H, hd).transpose(1, 2)
        mask_flat = ca_mask.reshape(B * S, L)
        mask_float = torch.zeros_like(mask_flat, dtype=q_ca.dtype)
        mask_float.masked_fill_(~mask_flat, float("-inf"))
        mask_float = mask_float.unsqueeze(1).unsqueeze(1)
        ca = F.scaled_dot_product_attention(q_ca, k_ca, v_ca, attn_mask=mask_float)
        ca = ca.transpose(1, 2).reshape(B, S, d)
        d_seq = d_seq + self.drop_ca(self.Wo_ca(ca))

        d_seq = d_seq + self.ffn(self.norm_ffn(d_seq))
        return d_seq

    def forward_cached(
        self,
        d_t: torch.Tensor,         # (B, 1, d)
        p_aug_t: torch.Tensor,     # (B, 1, L, d)
        ca_mask_t: torch.Tensor,   # (B, 1, L) bool
        rope_cos_t: torch.Tensor,
        rope_sin_t: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step forward with self-attention KV cache for AR rollout."""
        B, _, d = d_t.shape
        H, hd = self.n_heads, self.head_dim
        L = p_aug_t.size(2)

        h = self.norm_sa(d_t)
        q = self.Wq_sa(h).view(B, 1, H, hd).transpose(1, 2)
        k = self.Wk_sa(h).view(B, 1, H, hd).transpose(1, 2)
        v = self.Wv_sa(h).view(B, 1, H, hd).transpose(1, 2)
        q = _apply_rope(q, rope_cos_t, rope_sin_t)
        k = _apply_rope(k, rope_cos_t, rope_sin_t)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        sa = F.scaled_dot_product_attention(q, k, v)
        sa = sa.transpose(1, 2).reshape(B, 1, d)
        d_t = d_t + self.drop_sa(self.Wo_sa(sa))

        h = self.norm_ca(d_t)
        q_ca = self.Wq_ca(h).view(B, 1, H, hd).transpose(1, 2)
        kv_in = p_aug_t.squeeze(1)
        k_ca = self.Wk_ca(kv_in).view(B, L, H, hd).transpose(1, 2)
        v_ca = self.Wv_ca(kv_in).view(B, L, H, hd).transpose(1, 2)
        mask_flat = ca_mask_t.squeeze(1)
        mask_float = torch.zeros(B, L, device=d_t.device, dtype=d_t.dtype)
        mask_float.masked_fill_(~mask_flat, float("-inf"))
        mask_float = mask_float.unsqueeze(1).unsqueeze(1)
        ca = F.scaled_dot_product_attention(q_ca, k_ca, v_ca, attn_mask=mask_float)
        ca = ca.transpose(1, 2).reshape(B, 1, d)
        d_t = d_t + self.drop_ca(self.Wo_ca(ca))

        d_t = d_t + self.ffn(self.norm_ffn(d_t))
        return d_t, new_cache


class CmeV6(nn.Module):
    def __init__(self, cfg: CmeV6Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        S = cfg.n_segments

        if cfg.use_player_embeddings:
            self.E_player = nn.Embedding(cfg.vocab_size + 1, d, padding_idx=0)
            nn.init.normal_(self.E_player.weight, std=0.02)
            with torch.no_grad():
                self.E_player.weight[0].zero_()

        self.E_team = nn.Embedding(cfg.num_teams + 1, d, padding_idx=0)
        self.E_side = nn.Embedding(2, d)
        nn.init.normal_(self.E_team.weight, std=0.02)
        nn.init.normal_(self.E_side.weight, std=0.02)
        with torch.no_grad():
            self.E_team.weight[0].zero_()

        if cfg.player_stats_dim > 0:
            self.stats_proj = nn.Sequential(
                nn.Linear(cfg.player_stats_dim, d), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(d, d),
            )

        # Sinusoidal career-year encoding (fixed, not learned)
        career_pos = torch.arange(cfg.max_career_years + 1).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        career_enc = torch.zeros(cfg.max_career_years + 1, d)
        career_enc[:, 0::2] = torch.sin(career_pos * div)
        career_enc[:, 1::2] = torch.cos(career_pos * div[:d // 2])
        self.register_buffer("career_enc", career_enc)

        ctx_in = d + cfg.tabular_dim + 1
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_in, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
        )

        self.enc_layers = nn.ModuleList([
            EncoderBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_enc)
        ])
        self.enc_norm = nn.LayerNorm(d)

        self.log_gamma = nn.Parameter(torch.full((), math.log(15.0)))

        # Cumulative game-state decoder
        hd = d // cfg.n_heads
        rope_cos, rope_sin = _build_rope_tables(S, hd)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)
        self.dec_start = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.dec_start, std=0.02)
        self.cumstat_proj = nn.Sequential(
            nn.Linear(cfg.n_box_channels, d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
        self.dec_layers = nn.ModuleList([
            CumStateDecoderBlock(d, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_dec)
        ])
        self.dec_norm = nn.LayerNorm(d)
        self.box_out = nn.Linear(d, cfg.n_box_channels)

    def _encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        home_mask = batch["home_mask"]
        away_mask = batch["away_mask"]
        B = home_mask.size(0)
        Lh = home_mask.size(1)
        dev = home_mask.device

        if self.cfg.use_player_embeddings:
            home_e = self.E_player(batch["home_idx"])
            away_e = self.E_player(batch["away_idx"])
        else:
            home_e = torch.zeros(B, home_mask.size(1), self.cfg.d, device=dev)
            away_e = torch.zeros(B, away_mask.size(1), self.cfg.d, device=dev)

        if self.cfg.player_stats_dim > 0 and "home_stats" in batch:
            home_e = home_e + self.stats_proj(batch["home_stats"])
            away_e = away_e + self.stats_proj(batch["away_stats"])

        # Career-year sinusoidal encoding per player
        home_career = batch["home_career_year_idx"].clamp(0, self.cfg.max_career_years)
        away_career = batch["away_career_year_idx"].clamp(0, self.cfg.max_career_years)
        home_e = home_e + self.career_enc[home_career]
        away_e = away_e + self.career_enc[away_career]

        tab = batch["tabular"]
        home_team_e = self.E_team(batch["home_team_idx"])
        away_team_e = self.E_team(batch["away_team_idx"])
        home_rest = batch["home_rest"].unsqueeze(-1)
        away_rest = batch["away_rest"].unsqueeze(-1)
        home_team = self.ctx_proj(torch.cat([home_team_e, tab, home_rest], dim=-1)).unsqueeze(1)
        away_team = self.ctx_proj(torch.cat([away_team_e, tab, away_rest], dim=-1)).unsqueeze(1)

        side_home = self.E_side(torch.zeros(B, dtype=torch.long, device=dev))
        side_away = self.E_side(torch.ones(B, dtype=torch.long, device=dev))
        if not self.cfg.side_on_team_only:
            home_e = home_e + side_home.unsqueeze(1)
            away_e = away_e + side_away.unsqueeze(1)
        home_team = home_team + side_home.unsqueeze(1)
        away_team = away_team + side_away.unsqueeze(1)

        seq = torch.cat([home_team, home_e, away_team, away_e], dim=1)
        team_true = torch.ones(B, 1, dtype=torch.bool, device=dev)
        full_mask = torch.cat([team_true, home_mask, team_true, away_mask], dim=1)
        key_pad = ~full_mask

        for layer in self.enc_layers:
            seq = layer(seq, key_padding_mask=key_pad)
        seq = self.enc_norm(seq)

        home_team_out = seq[:, 0]
        home_p = seq[:, 1:1 + Lh]
        away_team_out = seq[:, 1 + Lh]
        away_p = seq[:, 2 + Lh:]
        P = torch.cat([home_p, away_p], dim=1)
        P_mask = torch.cat([home_mask, away_mask], dim=1)
        return P, P_mask, home_team_out, away_team_out

    def _build_cumstate_inputs(self, batch: dict, P_mask: torch.Tensor, Lh: int):
        h_box = batch["home_minute_box"]
        a_box = batch["away_minute_box"]
        minute_box = torch.cat([h_box, a_box], dim=1)
        cumstats = minute_box.cumsum(dim=2)
        cumstats_shifted = torch.cat([
            torch.zeros_like(cumstats[:, :, :1, :]),
            cumstats[:, :, :-1, :],
        ], dim=2)
        cumstats_input = cumstats_shifted.permute(0, 2, 1, 3)

        h_rot = batch["home_rotation_target"]
        a_rot = batch["away_rotation_target"]
        rot = torch.cat([h_rot, a_rot], dim=1)
        rot_shifted = torch.cat([rot[:, :, :1], rot[:, :, :-1]], dim=2)
        rot_mask = rot_shifted.permute(0, 2, 1)
        rot_mask = rot_mask * P_mask.unsqueeze(1).float()
        ca_mask = rot_mask > 0.5

        return cumstats_input, ca_mask

    def _decode_cumstate(
        self, P: torch.Tensor, P_mask: torch.Tensor,
        cumstats_input: torch.Tensor, ca_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S = cumstats_input.shape[:2]
        d = P.size(-1)

        cumstat_emb = self.cumstat_proj(cumstats_input)
        P_aug = P.unsqueeze(1) + cumstat_emb

        D = self.dec_start.expand(B, S, d)

        causal = torch.zeros(S, S, device=P.device)
        causal.masked_fill_(
            torch.triu(torch.ones(S, S, dtype=torch.bool, device=P.device), diagonal=1),
            float("-inf"),
        )

        for layer in self.dec_layers:
            D = layer(D, P_aug, causal, ca_mask, self.rope_cos, self.rope_sin)
        return self.dec_norm(D), P_aug

    def _decode_cumstate_ss(
        self, P: torch.Tensor, P_mask: torch.Tensor,
        batch: dict, Lh: int, ss_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step-by-step scheduled sampling with KV-cached decoder.

        At each minute t, coin-flip between GT and predicted cumstats from t-1.
        Gradients flow through predictions so the model learns to handle its own errors.
        """
        B, L, d = P.shape
        S = self.cfg.n_segments
        dev = P.device

        # GT cumstats and rotation mask (shifted right by 1)
        cumstats_gt, ca_mask = self._build_cumstate_inputs(batch, P_mask, Lh)

        # Seed: minute 0 uses zero cumstats (same as GT shifted right)
        cumstats = torch.zeros(B, L, self.cfg.n_box_channels, device=dev)

        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.dec_layers)
        all_D = []
        all_P_aug = []

        for t in range(S):
            # Build P_aug for this step
            cumstat_emb_t = self.cumstat_proj(cumstats.unsqueeze(1))  # (B, 1, L, d)
            P_aug_t = P.unsqueeze(1) + cumstat_emb_t  # (B, 1, L, d)
            all_P_aug.append(P_aug_t)

            ca_mask_t = ca_mask[:, t:t+1, :]  # (B, 1, L)

            D_t = self.dec_start.expand(B, 1, d)
            cos_t = self.rope_cos[t:t+1]
            sin_t = self.rope_sin[t:t+1]
            for i, layer in enumerate(self.dec_layers):
                D_t, kv_caches[i] = layer.forward_cached(
                    D_t, P_aug_t, ca_mask_t, cos_t, sin_t, kv_caches[i],
                )
            D_t = self.dec_norm(D_t).squeeze(1)  # (B, d)
            all_D.append(D_t)

            # Predict cumulative box scores at this step
            interaction = D_t.unsqueeze(1) * P  # (B, L, d)
            cum_t = F.relu(self.box_out(interaction))  # (B, L, 14)

            # Prepare cumstats for next step: coin-flip between predicted and GT
            if t + 1 < S:
                use_pred = (torch.rand(1, device=dev) < ss_ratio).float()
                cumstats = use_pred * cum_t + (1 - use_pred) * cumstats_gt[:, t + 1]

        D = torch.stack(all_D, dim=1)  # (B, S, d)
        P_aug = torch.cat(all_P_aug, dim=1)  # (B, S, L, d)
        return D, P_aug

    def _cumstate_heads(self, D: torch.Tensor, P: torch.Tensor,
                        P_aug: torch.Tensor, P_mask: torch.Tensor, Lh: int):
        B, S, d = D.shape
        mask_float = P_mask.unsqueeze(1).float()

        rot_logits = torch.einsum("bsd,bsld->bsl", D, P_aug) / (d ** 0.5)
        rotation = torch.sigmoid(rot_logits) * mask_float

        interaction = D.unsqueeze(2) * P.unsqueeze(1)
        box_cum = F.relu(self.box_out(interaction))

        # Win from final-minute cumulative pts
        final_cum = box_cum[:, -1, :, 0]  # (B, L) — pts channel
        P_mask_f = P_mask.float()
        home_pts = (final_cum[:, :Lh] * P_mask_f[:, :Lh]).sum(dim=1)
        away_pts = (final_cum[:, Lh:] * P_mask_f[:, Lh:]).sum(dim=1)
        gamma = self.log_gamma.exp()
        win_logit = (home_pts - away_pts) / gamma

        return {
            "home_rotation_logits": rot_logits[:, :, :Lh].permute(0, 2, 1),
            "away_rotation_logits": rot_logits[:, :, Lh:].permute(0, 2, 1),
            "home_rotation": rotation[:, :, :Lh].permute(0, 2, 1),
            "away_rotation": rotation[:, :, Lh:].permute(0, 2, 1),
            "box_cum": box_cum,
            "home_points": home_pts,
            "away_points": away_pts,
            "win_logit": win_logit,
        }

    @torch.no_grad()
    def _rollout_cumstate(self, P: torch.Tensor, P_mask: torch.Tensor,
                          batch: dict, Lh: int) -> dict:
        """AR eval with KV-cached self-attention so position t sees 0..t-1."""
        B, L, d = P.shape
        S = self.cfg.n_segments
        dev = P.device
        home_mask = batch["home_mask"]
        away_mask = batch["away_mask"]

        cumstats = torch.zeros(B, L, self.cfg.n_box_channels, device=dev)
        rot_history = torch.zeros(B, S, L, device=dev)
        all_box_cum = torch.zeros(B, S, L, self.cfg.n_box_channels, device=dev)
        all_rot_logits = torch.zeros(B, S, L, device=dev)

        h_rot0 = batch["home_rotation_target"][:, :, 0]
        a_rot0 = batch["away_rotation_target"][:, :, 0]
        rot_history[:, 0, :Lh] = h_rot0
        rot_history[:, 0, Lh:] = a_rot0

        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.dec_layers)

        for t in range(S):
            cumstat_emb_t = self.cumstat_proj(cumstats.unsqueeze(1))
            P_aug_t = P.unsqueeze(1) + cumstat_emb_t

            ca_mask_t = (rot_history[:, t:t+1] > 0.5) & P_mask.unsqueeze(1)

            D_t = self.dec_start.expand(B, 1, d)
            cos_t = self.rope_cos[t:t+1]
            sin_t = self.rope_sin[t:t+1]
            for i, layer in enumerate(self.dec_layers):
                D_t, kv_caches[i] = layer.forward_cached(
                    D_t, P_aug_t, ca_mask_t, cos_t, sin_t, kv_caches[i],
                )
            D_t = self.dec_norm(D_t).squeeze(1)

            interaction = D_t.unsqueeze(1) * P
            cum_t = F.relu(self.box_out(interaction))
            all_box_cum[:, t] = cum_t
            cumstats = cum_t

            # Rotation uses P_aug (game-state aware)
            P_aug_t_sq = P_aug_t.squeeze(1)
            rot_logits_t = torch.bmm(
                D_t.unsqueeze(1), P_aug_t_sq.transpose(1, 2),
            ).squeeze(1) / (d ** 0.5)
            all_rot_logits[:, t] = rot_logits_t

            if t + 1 < S:
                scores_h = rot_logits_t[:, :Lh].masked_fill(~home_mask, float("-inf"))
                scores_a = rot_logits_t[:, Lh:].masked_fill(~away_mask, float("-inf"))
                top_h = scores_h.topk(min(5, scores_h.size(-1)), dim=-1).indices
                top_a = scores_a.topk(min(5, scores_a.size(-1)), dim=-1).indices
                one_hot = torch.zeros(B, L, device=dev)
                one_hot.scatter_(1, top_h, 1.0)
                one_hot.scatter_(1, Lh + top_a, 1.0)
                rot_history[:, t + 1] = one_hot

        rotation = torch.sigmoid(all_rot_logits) * P_mask.unsqueeze(1).float()

        # Win from final-minute cumulative pts
        final_cum = all_box_cum[:, -1, :, 0]
        P_mask_f = P_mask.float()
        home_pts = (final_cum[:, :Lh] * P_mask_f[:, :Lh]).sum(dim=1)
        away_pts = (final_cum[:, Lh:] * P_mask_f[:, Lh:]).sum(dim=1)
        gamma = self.log_gamma.exp()
        win_logit = (home_pts - away_pts) / gamma

        return {
            "home_rotation_logits": all_rot_logits[:, :, :Lh].permute(0, 2, 1),
            "away_rotation_logits": all_rot_logits[:, :, Lh:].permute(0, 2, 1),
            "home_rotation": rotation[:, :, :Lh].permute(0, 2, 1),
            "away_rotation": rotation[:, :, Lh:].permute(0, 2, 1),
            "box_cum": all_box_cum,
            "home_points": home_pts,
            "away_points": away_pts,
            "win_logit": win_logit,
        }

    def forward(
        self,
        batch: dict,
        *,
        teacher_force: bool = True,
        autoregressive: bool = False,
        ss_ratio: float = 0.0,
    ) -> dict:
        home_mask = batch["home_mask"]
        Lh = home_mask.size(1)

        P, P_mask, home_team_out, away_team_out = self._encode(batch)

        if teacher_force and ss_ratio > 0:
            D, P_aug = self._decode_cumstate_ss(P, P_mask, batch, Lh, ss_ratio)
            out = self._cumstate_heads(D, P, P_aug, P_mask, Lh)
        elif teacher_force:
            cumstats_input, ca_mask = self._build_cumstate_inputs(batch, P_mask, Lh)
            D, P_aug = self._decode_cumstate(P, P_mask, cumstats_input, ca_mask)
            out = self._cumstate_heads(D, P, P_aug, P_mask, Lh)
        else:
            out = self._rollout_cumstate(P, P_mask, batch, Lh)

        out["home_team_out"] = home_team_out
        out["away_team_out"] = away_team_out
        return out


# ----------------------------- losses ----------------------------- #


def rotation_loss(out: dict, batch: dict) -> torch.Tensor:
    h_logits = out["home_rotation_logits"]
    a_logits = out["away_rotation_logits"]
    h_target = batch["home_rotation_target"]
    a_target = batch["away_rotation_target"]
    h_mask = batch["home_mask"].unsqueeze(-1).to(h_logits.dtype)
    a_mask = batch["away_mask"].unsqueeze(-1).to(a_logits.dtype)
    bce_h = F.binary_cross_entropy_with_logits(h_logits, h_target, reduction="none") * h_mask
    bce_a = F.binary_cross_entropy_with_logits(a_logits, a_target, reduction="none") * a_mask
    num = bce_h.sum() + bce_a.sum()
    den = (h_mask.sum() + a_mask.sum()).clamp_min(1.0) * h_logits.size(-1)
    return num / den


def win_bce(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, label, reduction="mean")


def win_hinge(logit: torch.Tensor, label: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    y = 2 * label - 1  # {0,1} -> {-1,+1}
    return F.relu(margin - y * logit).mean()


def box_cum_loss(out: dict, batch: dict) -> torch.Tensor:
    pred = out["box_cum"]  # (B, S, L, 14)
    h_box = batch["home_minute_box"]
    a_box = batch["away_minute_box"]
    gt_delta = torch.cat([h_box, a_box], dim=1)  # (B, L, S, 14)
    gt_cum = gt_delta.cumsum(dim=2).permute(0, 2, 1, 3)  # (B, S, L, 14)

    # Mask to on-court cells only
    h_rot = batch["home_rotation_target"]
    a_rot = batch["away_rotation_target"]
    on_court = torch.cat([h_rot, a_rot], dim=1).permute(0, 2, 1)
    mask = (on_court > 0.5).unsqueeze(-1).float()

    err = ((pred - gt_cum) ** 2) * mask
    return err.sum() / mask.sum().clamp_min(1.0) / pred.size(-1)


def total_loss_cumstate(
    out: dict, batch: dict, *,
    w_rot: float, w_delta: float, w_win: float,
    win_loss: str = "hinge",
) -> tuple[torch.Tensor, dict]:
    L_rot = rotation_loss(out, batch)
    L_cum = box_cum_loss(out, batch)
    if win_loss == "hinge":
        L_win = win_hinge(out["win_logit"], batch["label"])
    else:
        L_win = win_bce(out["win_logit"], batch["label"])
    total = w_rot * L_rot + w_delta * L_cum + w_win * L_win
    diag = {
        "L_rot": L_rot.detach(),
        "L_delta": L_cum.detach(),
        "L_win": L_win.detach(),
    }
    return total, diag
