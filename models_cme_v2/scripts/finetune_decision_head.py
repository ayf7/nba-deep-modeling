#!/usr/bin/env python3
"""Fine-tune a 3-way decision head on top of a frozen CME-v2 backbone.

Action space: {bet_home, bet_away, skip}.

Inputs to the head (concatenated, per game):
    [home_team_emb (d), away_team_emb (d),
     sigmoid(win_logit), 1/h_dec, 1/a_dec]   →  2d + 3

Loss (contextual-bandit, closed-form expected reward + entropy bonus):
    L = -E_π[R]  -  β · H(π)

  R_home = (h_dec - 1) if home_won else -1
  R_away = (a_dec - 1) if away_won else -1
  R_skip = 0

Only games with `has_odds=True` are used for both train and eval.

Eval (argmax policy) reports bet rate, per-side bet rate, ROI per bet,
ROI per game (= bankroll growth per game).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_SCRIPTS))

from cme_v2_common import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH, DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB, DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    GameDatasetV2, TeamVocab, Vocab, build_records_v2, collate_v2,
    fit_tabular_stats,
    load_game_odds, load_game_player_status, load_game_scores, load_games,
    load_matchup_rows_v2, load_player_game_stats, load_player_histories,
    load_status_calibration, load_team_exposures,
)
from cme_v2_model import CmeV2, CmeV2Config  # noqa: E402
from man_xfmr_common import PlayerFormStats  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v2" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=DEFAULT_OUTPUT_ROOT / "run_v2_dd_s_tt" / "best.pt")
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)

    p.add_argument("--head-mode", choices=["full", "minimal"], default="full",
                   help="full: [home_emb, away_emb, p_win, 1/h_dec, 1/a_dec] (2d+3). "
                        "minimal: [p_win, 1/h_dec, 1/a_dec, edge] (4).")
    p.add_argument("--head-hidden", type=int, default=0,
                   help="Hidden dim for decision head; default = backbone d (full) or 16 (minimal).")
    p.add_argument("--head-dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=0.1,
                   help="Entropy regularization weight.")
    p.add_argument("--beta-anneal", type=float, default=1.0,
                   help="Multiply beta by this factor each epoch (1.0 = no anneal).")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chrono_split(games: pd.DataFrame, val_frac: float, test_frac: float):
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(games)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return (
        games.iloc[:n_train].reset_index(drop=True),
        games.iloc[n_train : n_train + n_val].reset_index(drop=True),
        games.iloc[n_train + n_val :].reset_index(drop=True),
    )


class DecisionHead(nn.Module):
    """Small MLP from feature vector → 3 logits over {bet_home, bet_away, skip}."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def build_head_features(mode: str, home_emb, away_emb,
                        p_home, h_dec, a_dec) -> torch.Tensor:
    h_imp = 1.0 / h_dec
    a_imp = 1.0 / a_dec
    if mode == "full":
        return torch.cat([
            home_emb, away_emb,
            p_home.unsqueeze(-1),
            h_imp.unsqueeze(-1),
            a_imp.unsqueeze(-1),
        ], dim=-1)
    if mode == "minimal":
        devigged_h = h_imp / (h_imp + a_imp)
        edge = p_home - devigged_h
        return torch.stack([p_home, h_imp, a_imp, edge], dim=-1)
    raise ValueError(f"Unknown head mode: {mode}")


def head_input_dim(mode: str, d: int) -> int:
    if mode == "full":
        return 2 * d + 3
    if mode == "minimal":
        return 4
    raise ValueError(f"Unknown head mode: {mode}")


def compute_rewards(label: torch.Tensor, h_dec: torch.Tensor,
                    a_dec: torch.Tensor) -> torch.Tensor:
    """Return (B, 3) rewards for [bet_home, bet_away, skip]."""
    home_won = label  # 1 if home wins else 0, already float
    away_won = 1.0 - home_won
    r_home = (h_dec - 1.0) * home_won + (-1.0) * away_won
    r_away = (a_dec - 1.0) * away_won + (-1.0) * home_won
    r_skip = torch.zeros_like(r_home)
    return torch.stack([r_home, r_away, r_skip], dim=-1)


@torch.no_grad()
def encode_backbone(model: CmeV2, batch: dict) -> dict:
    out = model(batch)
    return {
        "home_team_emb": out["home_team_emb"].detach(),
        "away_team_emb": out["away_team_emb"].detach(),
        "win_logit": out["win_logit"].detach(),
    }


def _accumulate_step(stats: dict, logits: torch.Tensor, E_R: torch.Tensor,
                     H: torch.Tensor, loss: torch.Tensor, rewards: torch.Tensor,
                     b: int) -> None:
    pi = F.softmax(logits.detach(), dim=-1)
    stats["n"] += b
    stats["sum_er"] += float(E_R.detach().sum())
    stats["sum_h"] += float(H.detach().sum())
    stats["sum_loss"] += float(loss.detach()) * b
    stats["pi_bet_home"] += float(pi[:, 0].sum())
    stats["pi_bet_away"] += float(pi[:, 1].sum())
    stats["pi_skip"] += float(pi[:, 2].sum())

    action = logits.detach().argmax(dim=-1)
    realized = rewards.gather(1, action.unsqueeze(-1)).squeeze(-1)
    stats["sum_realized"] += float(realized.sum())
    bet_mask = action != 2
    home_bets = action == 0
    away_bets = action == 1
    stats["n_bet"] += int(bet_mask.sum())
    stats["n_bet_home"] += int(home_bets.sum())
    stats["n_bet_away"] += int(away_bets.sum())
    stats["sum_realized_bet"] += float(realized[bet_mask].sum())
    stats["sum_realized_home"] += float(realized[home_bets].sum())
    stats["sum_realized_away"] += float(realized[away_bets].sum())
    stats["n_bet_home_win"] += int(((rewards[:, 0] > 0) & home_bets).sum())
    stats["n_bet_away_win"] += int(((rewards[:, 1] > 0) & away_bets).sum())


def _finalize_stats(s: dict) -> dict:
    n = max(s["n"], 1)
    n_bet = max(s["n_bet"], 1)
    n_bh = max(s["n_bet_home"], 1)
    n_ba = max(s["n_bet_away"], 1)
    return {
        "n": s["n"],
        "loss": s["sum_loss"] / n,
        "E_R": s["sum_er"] / n,
        "H": s["sum_h"] / n,
        "pi_home": s["pi_bet_home"] / n,
        "pi_away": s["pi_bet_away"] / n,
        "pi_skip": s["pi_skip"] / n,
        "roi_game": s["sum_realized"] / n,
        "roi_bet": s["sum_realized_bet"] / n_bet,
        "bet_rate": s["n_bet"] / n,
        "bet_home_rate": s["n_bet_home"] / n,
        "bet_away_rate": s["n_bet_away"] / n,
        "n_bet": s["n_bet"],
        "n_bet_home": s["n_bet_home"],
        "n_bet_away": s["n_bet_away"],
        "roi_home_bet": s["sum_realized_home"] / n_bh,
        "roi_away_bet": s["sum_realized_away"] / n_ba,
        "home_bet_winrate": s["n_bet_home_win"] / n_bh,
        "away_bet_winrate": s["n_bet_away_win"] / n_ba,
    }


def _empty_stats() -> dict:
    return {k: 0 if k.startswith("n") else 0.0 for k in [
        "n", "sum_er", "sum_h", "sum_loss",
        "pi_bet_home", "pi_bet_away", "pi_skip",
        "sum_realized", "sum_realized_bet", "sum_realized_home", "sum_realized_away",
        "n_bet", "n_bet_home", "n_bet_away", "n_bet_home_win", "n_bet_away_win",
    ]}


def _step_forward(model: CmeV2, head: DecisionHead, batch: dict,
                  head_mode: str, device: str):
    """Return (logits, E_R, H, rewards, b) for masked batch."""
    mask = batch["has_odds"] > 0.5
    if not mask.any():
        return None
    enc = encode_backbone(model, batch)
    home_emb = enc["home_team_emb"][mask]
    away_emb = enc["away_team_emb"][mask]
    p_home = torch.sigmoid(enc["win_logit"][mask])
    h_dec = batch["home_dec_odds"][mask]
    a_dec = batch["away_dec_odds"][mask]
    label = batch["label"][mask]
    x = build_head_features(head_mode, home_emb, away_emb, p_home, h_dec, a_dec)
    logits = head(x)
    log_pi = F.log_softmax(logits, dim=-1)
    pi = log_pi.exp()
    rewards = compute_rewards(label, h_dec, a_dec)
    E_R = (pi * rewards).sum(dim=-1)
    H = -(pi * log_pi).sum(dim=-1)
    return logits, E_R, H, rewards, int(mask.sum())


def epoch_train(model: CmeV2, head: DecisionHead, loader: DataLoader,
                opt: torch.optim.Optimizer, beta: float, head_mode: str,
                device: str) -> dict:
    head.train()
    stats = _empty_stats()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        res = _step_forward(model, head, batch, head_mode, device)
        if res is None:
            continue
        logits, E_R, H, rewards, b = res
        loss = (-E_R - beta * H).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        _accumulate_step(stats, logits, E_R, H, loss, rewards, b)
    return _finalize_stats(stats)


@torch.no_grad()
def epoch_eval(model: CmeV2, head: DecisionHead, loader: DataLoader,
               beta: float, head_mode: str, device: str) -> dict:
    head.eval()
    stats = _empty_stats()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        res = _step_forward(model, head, batch, head_mode, device)
        if res is None:
            continue
        logits, E_R, H, rewards, b = res
        loss = (-E_R - beta * H).mean()
        _accumulate_step(stats, logits, E_R, H, loss, rewards, b)
    return _finalize_stats(stats)


def fmt_eval(tag: str, ep: int, m: dict, beta: float) -> str:
    return (f"[{tag} ep{ep:02d}] loss={m['loss']:+.4f} E_R={m['E_R']:+.4f} "
            f"H={m['H']:.3f} (β={beta:.3f}) | "
            f"π(h/a/skip)={m['pi_home']:.2f}/{m['pi_away']:.2f}/{m['pi_skip']:.2f} | "
            f"argmax bet={m['bet_rate']:.2f} "
            f"(h={m['bet_home_rate']:.2f} a={m['bet_away_rate']:.2f}) "
            f"ROI/bet={m['roi_bet']*100:+.2f}% ROI/game={m['roi_game']*100:+.2f}%")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device}")
    print(f"[ckpt]   {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = CmeV2Config(**state["cfg"])
    if not cfg.use_team_token:
        raise ValueError(
            "Decision head requires use_team_token=True in the backbone checkpoint. "
            f"Got cfg.use_team_token={cfg.use_team_token}."
        )
    vocab = Vocab(player_to_idx=state["vocab"])
    team_vocab = TeamVocab(team_to_idx=state["team_vocab"])
    use_player_stats = cfg.player_stat_dim > 0
    player_form_stats = (
        PlayerFormStats(
            means=np.array(state["player_form_means"], dtype=np.float64),
            stds=np.array(state["player_form_stds"], dtype=np.float64),
        ) if use_player_stats else None
    )

    print("[load] games + exposures + matchup rows + status + odds")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)
    game_odds = load_game_odds(args.core_db, game_ids)
    print(f"[load] game_odds: {len(game_odds)} / {len(game_ids)} games have odds")

    train_df, val_df, test_df = chrono_split(games_all, args.val_frac, args.test_frac)
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_gids = [str(g) for g in train_df["game_id"].tolist()]
    val_gids = [str(g) for g in val_df["game_id"].tolist()]
    test_gids = [str(g) for g in test_df["game_id"].tolist()]

    print("[load] matchup rows + player game stats")
    train_matchup = load_matchup_rows_v2(args.matchup_db, train_gids)
    val_matchup = load_matchup_rows_v2(args.matchup_db, val_gids)
    test_matchup = load_matchup_rows_v2(args.matchup_db, test_gids)
    train_pl = load_player_game_stats(args.core_db, train_gids)
    val_pl = load_player_game_stats(args.core_db, val_gids)
    test_pl = load_player_game_stats(args.core_db, test_gids)

    tabular_stats = fit_tabular_stats(train_df)
    player_histories = load_player_histories(args.core_db) if use_player_stats else None

    print("[records] building train/val/test")
    common = dict(
        histories=histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=game_odds,
    )
    train_recs = build_records_v2(train_df, matchup_rows=train_matchup,
                                  player_game_stats=train_pl, **common)
    val_recs = build_records_v2(val_df, matchup_rows=val_matchup,
                                player_game_stats=val_pl, **common)
    test_recs = build_records_v2(test_df, matchup_rows=test_matchup,
                                 player_game_stats=test_pl, **common)
    n_train_o = sum(1 for r in train_recs if r.has_odds)
    n_val_o = sum(1 for r in val_recs if r.has_odds)
    n_test_o = sum(1 for r in test_recs if r.has_odds)
    print(f"[records] train={len(train_recs)} ({n_train_o} w/ odds) "
          f"val={len(val_recs)} ({n_val_o} w/ odds) "
          f"test={len(test_recs)} ({n_test_o} w/ odds)")

    train_loader = DataLoader(GameDatasetV2(train_recs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_v2)
    val_loader = DataLoader(GameDatasetV2(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_v2)
    test_loader = DataLoader(GameDatasetV2(test_recs), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_v2)

    print("[model] building + loading state (FROZEN)")
    model = CmeV2(cfg).to(device)
    model.load_state_dict(state["model_state"], strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    d = cfg.d
    in_dim = head_input_dim(args.head_mode, d)
    default_hidden = d if args.head_mode == "full" else 16
    hidden = args.head_hidden if args.head_hidden > 0 else default_hidden
    head = DecisionHead(in_dim=in_dim, hidden=hidden, dropout=args.head_dropout).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[head] mode={args.head_mode} in={in_dim} hidden={hidden} out=3  params={n_params}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    best_val_er = -float("inf")
    best_ep = -1
    epochs_since_improve = 0
    beta = args.beta
    history = []

    print("[train] starting")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr = epoch_train(model, head, train_loader, opt, beta=beta,
                         head_mode=args.head_mode, device=device)
        va = epoch_eval(model, head, val_loader, beta=beta,
                        head_mode=args.head_mode, device=device)
        te = epoch_eval(model, head, test_loader, beta=beta,
                        head_mode=args.head_mode, device=device)
        dt = time.time() - t0

        print(fmt_eval("tr", ep, tr, beta) + f"  ({dt:.1f}s)")
        print(fmt_eval("va", ep, va, beta))
        print(fmt_eval("te", ep, te, beta))

        history.append({"epoch": ep, "beta": beta, "train": tr, "val": va, "test": te})

        # Best by val expected reward (the training objective)
        if va["E_R"] > best_val_er + 1e-5:
            best_val_er = va["E_R"]
            best_ep = ep
            epochs_since_improve = 0
            torch.save({
                "head_state": head.state_dict(),
                "head_cfg": {
                    "mode": args.head_mode, "in_dim": in_dim,
                    "hidden": hidden, "dropout": args.head_dropout,
                },
                "backbone_ckpt": str(args.checkpoint),
                "epoch": ep,
                "val_metrics": va,
                "test_metrics": te,
                "args": vars(args),
            }, out_dir / "best.pt")
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= args.patience:
            print(f"[early-stop] no val_E_R improvement for {args.patience} epochs")
            break

        beta *= args.beta_anneal

    print(f"[best] epoch={best_ep} val_E_R={best_val_er:+.4f}")

    # Reload best head + final eval on test
    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    head.load_state_dict(best["head_state"])
    te_best = epoch_eval(model, head, test_loader, beta=args.beta,
                         head_mode=args.head_mode, device=device)
    va_best = epoch_eval(model, head, val_loader, beta=args.beta,
                         head_mode=args.head_mode, device=device)
    print(fmt_eval("va:best", best_ep, va_best, args.beta))
    print(fmt_eval("te:best", best_ep, te_best, args.beta))

    summary = {
        "checkpoint": str(args.checkpoint),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "best_epoch": best_ep,
        "val": va_best,
        "test": te_best,
        "history": history,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
