"""Encoder-only model: roster → self-attention → win prediction. No decoder."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderOnlyConfig:
    vocab_size: int
    num_teams: int
    tabular_dim: int = 0
    d: int = 32
    n_heads: int = 4
    n_enc: int = 2
    dropout: float = 0.1
    player_stats_dim: int = 0
    max_career_years: int = 25
    use_player_embeddings: bool = True


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


class EncoderOnlyModel(nn.Module):
    def __init__(self, cfg: EncoderOnlyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d

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

        self.win_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )

    def forward(self, batch: dict) -> dict:
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
        away_team_out = seq[:, Lh + 1]
        win_logit = self.win_head(torch.cat([home_team_out, away_team_out], dim=-1)).squeeze(-1)

        return {"win_logit": win_logit}


def win_hinge(logit: torch.Tensor, label: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    y = 2 * label - 1
    return F.relu(margin - y * logit).mean()


def win_bce(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, label, reduction="mean")
