"""CME-v1: Constrained Matchup-Event World Model, minimal first build.

Forward pipeline per game:
  PlayerFusion(player_id, stats, tabular)
  -> within-team self-attention
  -> per-side involvement softmax (off + def, with log(prob_play) prior)
  -> possession volume N
  -> Sinkhorn-constrained pair exposure P that satisfies row/col marginals
  -> per-pair efficiency e (points per possession)
  -> team points = sum_ij P_ij * e_ij
  -> margin = home_pts - away_pts
  -> win_logit = margin / global_scale

Outputs are compatible with v5's gather_lambda_for_supervision for the pair
Poisson loss, and additionally expose involvement and points heads for the
new CME losses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sinkhorn import sinkhorn_transport


@dataclass
class CmeV1Config:
    vocab_size: int
    num_teams: int
    d: int = 32
    pair_hidden: int = 64
    eff_hidden: int = 32
    inv_hidden: int = 32
    dropout: float = 0.15
    pair_dropout: float = 0.5
    player_dropout: float = 0.0
    tabular_dim: int = 0
    team_emb_dim: int = 8
    player_stat_dim: int = 0
    n_self_attn_heads: int = 2
    n_self_attn_layers: int = 1
    sinkhorn_iters: int = 8
    base_possessions_per_team: float = 491.0
    base_eff: float = 0.224
    eff_amplitude: float = 0.05
    init_global_scale: float = 9.0
    use_bet_token: bool = False  # CLS-style probe slot for downstream heads


def _zero_init_last_linear(seq: nn.Sequential) -> None:
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(logits, dim=dim)


class PlayerFusion(nn.Module):
    def __init__(self, d: int, stat_dim: int, tabular_dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = d + stat_dim + tabular_dim
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )
        _zero_init_last_linear(self.fuse)

    def forward(
        self,
        e_player: torch.Tensor,             # (B, L, d)
        stats: torch.Tensor | None,         # (B, L, S) or None
        tabular: torch.Tensor | None,       # (B, T) or None
    ) -> torch.Tensor:
        B, L, d = e_player.shape
        parts = [e_player]
        if stats is not None and stats.size(-1) > 0:
            parts.append(stats)
        if tabular is not None and tabular.size(-1) > 0:
            parts.append(tabular.unsqueeze(1).expand(-1, L, -1))
        x = torch.cat(parts, dim=-1)
        return e_player + self.fuse(x)


class TeamSelfAttention(nn.Module):
    """Light within-team self-attention, padding-masked."""
    def __init__(self, d: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True),
                "ffn": nn.Sequential(
                    nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)
                ),
                "ln1": nn.LayerNorm(d),
                "ln2": nn.LayerNorm(d),
            }))
        for layer in self.layers:
            _zero_init_last_linear(layer["ffn"])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~mask
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
            x = layer["ln1"](x + attn_out)
            x = layer["ln2"](x + layer["ffn"](x))
        return x


class CmeV1(nn.Module):
    def __init__(self, cfg: CmeV1Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d

        self.E_player = nn.Embedding(cfg.vocab_size + 1, d, padding_idx=0)
        self.b_off = nn.Embedding(cfg.vocab_size + 1, 1, padding_idx=0)
        self.b_def = nn.Embedding(cfg.vocab_size + 1, 1, padding_idx=0)

        self.W_q_team = nn.Embedding(cfg.num_teams + 1, d * d)
        self.W_k_team = nn.Embedding(cfg.num_teams + 1, d * d)

        if cfg.team_emb_dim > 0:
            self.E_team = nn.Embedding(cfg.num_teams + 1, cfg.team_emb_dim, padding_idx=0)

        self.player_fusion = PlayerFusion(d, cfg.player_stat_dim, cfg.tabular_dim, cfg.dropout)
        self.team_attn = TeamSelfAttention(d, cfg.n_self_attn_heads, cfg.n_self_attn_layers, cfg.dropout)

        if cfg.use_bet_token:
            self.bet_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        ctx_dim = cfg.tabular_dim + 2 + 2 * cfg.team_emb_dim  # tabular + 2 rest + 2 team emb

        self.off_inv_head = nn.Sequential(
            nn.Linear(d + ctx_dim, cfg.inv_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.inv_hidden, 1),
        )
        self.def_inv_head = nn.Sequential(
            nn.Linear(d + ctx_dim, cfg.inv_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.inv_hidden, 1),
        )

        self.poss_head = nn.Sequential(
            nn.Linear(ctx_dim, cfg.inv_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.inv_hidden, 1),
        )
        _zero_init_last_linear(self.poss_head)

        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d + ctx_dim, cfg.pair_hidden), nn.GELU(), nn.Dropout(cfg.pair_dropout),
            nn.Linear(cfg.pair_hidden, 1),
        )
        _zero_init_last_linear(self.pair_mlp)

        if cfg.eff_hidden > 0:
            self.eff_mlp = nn.Sequential(
                nn.Linear(2 * d + ctx_dim, cfg.eff_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.eff_hidden, 1),
            )
        else:
            self.eff_mlp = nn.Sequential(nn.Linear(2 * d + ctx_dim, 1))
        _zero_init_last_linear(self.eff_mlp)

        self.global_scale = nn.Parameter(torch.tensor(math.log(cfg.init_global_scale)))

        nn.init.normal_(self.E_player.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.E_player.weight[0].zero_()
        nn.init.zeros_(self.b_off.weight)
        nn.init.zeros_(self.b_def.weight)

        with torch.no_grad():
            eye = torch.eye(d).flatten()
            self.W_q_team.weight.copy_(eye.unsqueeze(0).expand(cfg.num_teams + 1, -1))
            self.W_k_team.weight.copy_(eye.unsqueeze(0).expand(cfg.num_teams + 1, -1))
            self.W_q_team.weight[0].zero_()
            self.W_k_team.weight[0].zero_()

        if cfg.team_emb_dim > 0:
            nn.init.normal_(self.E_team.weight, std=0.02)
            with torch.no_grad():
                self.E_team.weight[0].zero_()

    def _make_ctx(self, batch: dict) -> torch.Tensor:
        parts = []
        if self.cfg.tabular_dim > 0:
            parts.append(batch["tabular"])
        parts.append(batch["home_rest"].unsqueeze(-1))
        parts.append(batch["away_rest"].unsqueeze(-1))
        if self.cfg.team_emb_dim > 0:
            parts.append(self.E_team(batch["home_team_idx"]))
            parts.append(self.E_team(batch["away_team_idx"]))
        return torch.cat(parts, dim=-1)  # (B, ctx_dim)

    def _encode_side(self, idx, stats, tabular, mask):
        """Returns (Hh, bet_emb) where bet_emb is None unless use_bet_token=True.

        With the bet token, a learnable [BET] slot is prepended to the player
        sequence, attends to all real players (and they to it), and its post-
        attention output is the per-side residual-stream summary.
        """
        e = self.E_player(idx)
        e = self.player_fusion(e, stats if stats.size(-1) > 0 else None, tabular)
        e = e * mask.unsqueeze(-1).to(e.dtype)

        if not self.cfg.use_bet_token:
            e = self.team_attn(e, mask)
            return e * mask.unsqueeze(-1).to(e.dtype), None

        B = e.size(0)
        bet = self.bet_token.expand(B, 1, -1).to(e.dtype)
        e_full = torch.cat([bet, e], dim=1)
        bet_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        mask_full = torch.cat([bet_mask, mask], dim=1)
        e_full = self.team_attn(e_full, mask_full)
        bet_emb = e_full[:, 0]
        Hh = e_full[:, 1:] * mask.unsqueeze(-1).to(e_full.dtype)
        return Hh, bet_emb

    def _direction(
        self,
        Hoff: torch.Tensor,                  # (B, Lo, d)
        Hdef: torch.Tensor,                  # (B, Ld, d)
        prob_off: torch.Tensor,              # (B, Lo)
        prob_def: torch.Tensor,              # (B, Ld)
        mask_off: torch.Tensor,              # (B, Lo)
        mask_def: torch.Tensor,              # (B, Ld)
        team_off: torch.Tensor,              # (B,)
        team_def: torch.Tensor,              # (B,)
        ctx: torch.Tensor,                   # (B, ctx_dim)
        idx_off: torch.Tensor,               # (B, Lo) for player bias
        idx_def: torch.Tensor,               # (B, Ld)
    ) -> dict:
        B, Lo, d = Hoff.shape
        Ld = Hdef.size(1)

        ctx_off = ctx.unsqueeze(1).expand(-1, Lo, -1)
        ctx_def = ctx.unsqueeze(1).expand(-1, Ld, -1)

        off_logits = self.off_inv_head(torch.cat([Hoff, ctx_off], dim=-1)).squeeze(-1)
        def_logits = self.def_inv_head(torch.cat([Hdef, ctx_def], dim=-1)).squeeze(-1)
        off_logits = off_logits + torch.log(prob_off.clamp_min(1e-6))
        def_logits = def_logits + torch.log(prob_def.clamp_min(1e-6))

        off_share = masked_softmax(off_logits, mask_off, dim=-1)
        def_share = masked_softmax(def_logits, mask_def, dim=-1)

        N = F.softplus(self.poss_head(ctx).squeeze(-1)) + self.cfg.base_possessions_per_team
        row_mass = N.unsqueeze(-1) * off_share
        col_mass = N.unsqueeze(-1) * def_share

        W_q = self.W_q_team(team_off).view(B, d, d)
        W_k = self.W_k_team(team_def).view(B, d, d)
        q = torch.einsum("bld,bde->ble", Hoff, W_q)
        k = torch.einsum("bld,bde->ble", Hdef, W_k)
        score = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(d)
        score = score + self.b_off(idx_off).squeeze(-1).unsqueeze(2)
        score = score + self.b_def(idx_def).squeeze(-1).unsqueeze(1)

        Hoff_e = Hoff.unsqueeze(2).expand(-1, -1, Ld, -1)
        Hdef_e = Hdef.unsqueeze(1).expand(-1, Lo, -1, -1)
        ctx_pair = ctx.view(B, 1, 1, -1).expand(-1, Lo, Ld, -1)
        pair_in = torch.cat([Hoff_e, Hdef_e, ctx_pair], dim=-1)
        score = score + self.pair_mlp(pair_in).squeeze(-1)
        score = torch.clamp(score, min=-12.0, max=12.0)

        mask_outer = mask_off.unsqueeze(2) & mask_def.unsqueeze(1)
        P = sinkhorn_transport(
            log_scores=score,
            row_mass=row_mass,
            col_mass=col_mass,
            mask=mask_outer,
            n_iter=self.cfg.sinkhorn_iters,
        )

        eff_raw = self.eff_mlp(pair_in).squeeze(-1)
        eff = self.cfg.base_eff + self.cfg.eff_amplitude * torch.tanh(eff_raw)
        eff = eff * mask_outer.to(eff.dtype)

        points = (P * eff).sum(dim=(1, 2))

        return {
            "off_share": off_share,
            "def_share": def_share,
            "off_involvement": row_mass,
            "def_involvement": col_mass,
            "pair_exposure": P,
            "eff": eff,
            "points": points,
            "N": N,
        }

    def forward(self, batch: dict) -> dict:
        home_mask = batch["home_mask"]
        away_mask = batch["away_mask"]
        if self.training and self.cfg.player_dropout > 0.0:
            p = self.cfg.player_dropout
            home_mask = home_mask & (torch.rand(home_mask.shape, device=home_mask.device) > p)
            away_mask = away_mask & (torch.rand(away_mask.shape, device=away_mask.device) > p)

        tabular = batch.get("tabular")

        Hh, h_bet = self._encode_side(batch["home_idx"], batch["home_stats"], tabular, home_mask)
        Ha, a_bet = self._encode_side(batch["away_idx"], batch["away_stats"], tabular, away_mask)

        # Side-summary used by downstream heads. With use_bet_token, this is the
        # post-attention [BET] token slot — proper residual-stream probe.
        # Without it, fall back to masked mean-pool of player embeddings.
        if self.cfg.use_bet_token:
            home_emb = h_bet
            away_emb = a_bet
        else:
            h_n = home_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(Hh.dtype)
            a_n = away_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(Ha.dtype)
            home_emb = Hh.sum(dim=1) / h_n
            away_emb = Ha.sum(dim=1) / a_n

        ctx = self._make_ctx(batch)

        home_off = self._direction(
            Hoff=Hh, Hdef=Ha,
            prob_off=batch["home_prob"], prob_def=batch["away_prob"],
            mask_off=home_mask, mask_def=away_mask,
            team_off=batch["home_team_idx"], team_def=batch["away_team_idx"],
            ctx=ctx,
            idx_off=batch["home_idx"], idx_def=batch["away_idx"],
        )
        away_off = self._direction(
            Hoff=Ha, Hdef=Hh,
            prob_off=batch["away_prob"], prob_def=batch["home_prob"],
            mask_off=away_mask, mask_def=home_mask,
            team_off=batch["away_team_idx"], team_def=batch["home_team_idx"],
            ctx=ctx,
            idx_off=batch["away_idx"], idx_def=batch["home_idx"],
        )

        home_points = home_off["points"]
        away_points = away_off["points"]
        margin_mu = home_points - away_points

        scale = self.global_scale.exp().clamp_min(1.0)
        win_logit = margin_mu / scale

        return {
            "win_logit": win_logit,
            "margin_mu": margin_mu,
            "home_points": home_points,
            "away_points": away_points,
            "lam_h": home_off["pair_exposure"],
            "lam_a": away_off["pair_exposure"],
            "home_off_inv": home_off["off_involvement"],
            "away_def_inv": home_off["def_involvement"],
            "away_off_inv": away_off["off_involvement"],
            "home_def_inv": away_off["def_involvement"],
            "home_off_share": home_off["off_share"],
            "away_def_share": home_off["def_share"],
            "away_off_share": away_off["off_share"],
            "home_def_share": away_off["def_share"],
            "N_home": home_off["N"],
            "N_away": away_off["N"],
            "global_scale": scale,
            "home_emb": home_emb,
            "away_emb": away_emb,
            "ctx": ctx,
        }


def gather_lambda_for_supervision(
    lam_h: torch.Tensor,
    lam_a: torch.Tensor,
    sup_game: torch.Tensor,
    sup_side: torch.Tensor,
    sup_off: torch.Tensor,
    sup_def: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros(sup_game.size(0), device=lam_h.device, dtype=lam_h.dtype)
    h_mask = sup_side == 0
    a_mask = sup_side == 1
    if h_mask.any():
        idx = h_mask.nonzero(as_tuple=False).squeeze(-1)
        out[idx] = lam_h[sup_game[idx], sup_off[idx], sup_def[idx]]
    if a_mask.any():
        idx = a_mask.nonzero(as_tuple=False).squeeze(-1)
        out[idx] = lam_a[sup_game[idx], sup_off[idx], sup_def[idx]]
    return out


def poisson_nll(lam_pred: torch.Tensor, exposure: torch.Tensor) -> torch.Tensor:
    lam = lam_pred.clamp_min(1e-8)
    return (lam - exposure * torch.log(lam)).mean()


def _smoke() -> None:
    torch.manual_seed(0)
    cfg = CmeV1Config(
        vocab_size=200, num_teams=30, d=32, tabular_dim=4, team_emb_dim=8, player_stat_dim=6,
    )
    model = CmeV1(cfg)
    B, Lo, Ld = 4, 12, 14
    batch = {
        "home_idx": torch.randint(1, cfg.vocab_size + 1, (B, Lo)),
        "away_idx": torch.randint(1, cfg.vocab_size + 1, (B, Ld)),
        "home_prob": torch.rand(B, Lo),
        "away_prob": torch.rand(B, Ld),
        "home_mask": torch.ones(B, Lo, dtype=torch.bool),
        "away_mask": torch.ones(B, Ld, dtype=torch.bool),
        "home_stats": torch.randn(B, Lo, cfg.player_stat_dim),
        "away_stats": torch.randn(B, Ld, cfg.player_stat_dim),
        "home_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "away_team_idx": torch.randint(1, cfg.num_teams + 1, (B,)),
        "home_rest": torch.randn(B),
        "away_rest": torch.randn(B),
        "tabular": torch.randn(B, cfg.tabular_dim),
    }
    batch["home_mask"][0, 10:] = False  # mask 2 home players in batch 0
    batch["away_mask"][1, 12:] = False  # mask 2 away players in batch 1

    out = model(batch)
    print("=== shapes ===")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:20s} {tuple(v.shape)}")
        else:
            print(f"{k:20s} {type(v).__name__}")

    print(f"\nwin_logit          {out['win_logit'].tolist()}")
    print(f"home_points        {out['home_points'].tolist()}")
    print(f"away_points        {out['away_points'].tolist()}")
    print(f"margin_mu          {out['margin_mu'].tolist()}")
    print(f"N_home             {out['N_home'].tolist()}")
    print(f"global_scale       {out['global_scale'].item():.3f}")

    P_h = out["lam_h"]
    print(f"\nlam_h sum per game (should ≈ N_home): {P_h.sum(dim=(1,2)).tolist()}")
    print(f"N_home from forward                  : {out['N_home'].tolist()}")

    B_idx = 0
    row_target = out["home_off_inv"][B_idx]
    row_actual = P_h[B_idx].sum(dim=1)
    print(f"\nbatch 0 row marginal max err = {(row_target - row_actual).abs().max().item():.2e}")

    loss = (out["win_logit"].pow(2).mean() + out["home_points"].mean()
            + out["lam_h"].mean() + out["home_off_inv"].mean())
    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    n_total = sum(1 for _ in model.parameters())
    print(f"\nbackward ok: {n_grad}/{n_total} parameters got gradients")


if __name__ == "__main__":
    _smoke()
