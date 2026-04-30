#!/usr/bin/env python3
"""Expanding-window monthly backtest for CME-v5 (cascaded world model).

Adapted from backtest_cme_v4.py. Swaps in v5's cascaded attention model,
48-dim rotation curves, and per-segment Sinkhorn. Output schema is identical
so predictions CSV is a drop-in for evaluate_betting_strategy.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
V5_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_SCRIPTS))

from cme_v5_common import (  # noqa: E402
    BOX_INDEX, BOX_TARGETS, K_BOX, K_PAIR, MAX_CAREER_YEAR, PAIR_TARGETS,
    DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    DEFAULT_PLAYER_DEBUT_DB,
    DEFAULT_PLAYER_DECISIONS_DB, DEFAULT_PLAYER_GAME_STATS_DB,
    DEFAULT_V5_FEATURES_DB,
    PLAYER_FORM_DIM, TABULAR_FEATURE_COLUMNS, TEAM_BASE_POSSESSIONS,
    GameDatasetV5, build_records_v5, build_team_vocab,
    build_vocab_from_records_v5, collate_v5,
    fit_player_form_stats, fit_tabular_stats,
    load_box_minutes_and_pace,
    load_game_odds, load_game_scores, load_games,
    load_matchup_rows_v2, load_minute_presence, load_play_decisions,
    load_player_debut, load_player_first_season,
    load_player_game_stats, load_player_histories, load_regulation_scores,
    load_team_exposures,
)
from cme_v5_model import (  # noqa: E402
    CmeV5, CmeV5Config,
    rotation_loss, pace_mse_loss, pair_poisson_loss,
    player_poisson_loss, team_poisson_loss, win_bce_loss, total_loss,
    total_loss_v2,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v5" / "artifacts"


# ----------------------------- weight defaults ----------------------------- #


def default_box_weights() -> torch.Tensor:
    w = torch.ones(K_BOX)
    w[BOX_INDEX["pts"]] = 3.0
    w[BOX_INDEX["pf"]] = 0.3
    w[BOX_INDEX["tov"]] = 0.5
    w[BOX_INDEX["blk"]] = 0.5
    w[BOX_INDEX["stl"]] = 0.5
    w[BOX_INDEX["ast"]] = 0.5
    w[BOX_INDEX["oreb"]] = 0.5
    w[BOX_INDEX["dreb"]] = 0.5
    return w


def default_pair_weights() -> torch.Tensor:
    w = torch.full((K_PAIR,), 0.5)
    w[PAIR_TARGETS.index("exposure_possessions")] = 1.0
    if "player_points" in PAIR_TARGETS:
        w[PAIR_TARGETS.index("player_points")] = 1.0
    return w


# ----------------------------- run epoch ----------------------------- #


def run_epoch(
    model: CmeV5, loader: DataLoader, *, device: str,
    optim: torch.optim.Optimizer | None,
    box_weights: torch.Tensor, pair_weights: torch.Tensor,
    rot_w: float, pace_w: float, pair_w: float,
    player_w: float, team_w: float, win_w: float,
    direct_w: float = 0.0,
    use_mse_player: bool = False,
) -> dict:
    is_train = optim is not None
    model.train(is_train)

    total_n = 0
    sum_loss = 0.0
    sum_L_rot = 0.0
    sum_L_pace = 0.0
    sum_L_pair = 0.0
    sum_L_player = 0.0
    sum_L_team = 0.0
    sum_L_win = 0.0
    sum_team_nll_per_target = torch.zeros(K_BOX)
    sum_player_nll_per_target = torch.zeros(K_BOX)
    sum_pair_nll_per_target = torch.zeros(K_PAIR)
    sum_bce = 0.0
    sum_margin_abs = 0.0
    sum_pts_h = 0.0
    sum_pts_a = 0.0
    sum_pace_h = 0.0
    sum_pace_a = 0.0
    sum_rotation_corr = 0.0
    sum_rotation_corr_n = 0
    correct = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            out = model(batch)
            loss, diag = total_loss_v2(
                out, batch,
                box_weights=box_weights.to(device),
                pair_weights=pair_weights.to(device),
                rot_w=rot_w, pace_w=pace_w, pair_w=pair_w,
                player_w=player_w, team_w=team_w, win_w=win_w,
                direct_w=direct_w,
                use_mse_player=use_mse_player,
            )

            if is_train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

            with torch.no_grad():
                bce = F.binary_cross_entropy_with_logits(out["win_logit"], batch["label"])
                margin_abs = (out["margin_mu"] - batch["margin"]).abs().mean()
                preds = (torch.sigmoid(out["win_logit"]) > 0.5).float()
                correct_bs = (preds == batch["label"]).sum().item()

                # Rotation curve correlation (effective minutes pred vs target sum)
                for side in ("home", "away"):
                    rot = out[f"{side}_rotation"]  # (B, L, 48)
                    tgt = batch[f"{side}_rotation_target"]  # (B, L, 48)
                    mask = batch[f"{side}_mask"]  # (B, L)
                    pred_eff = rot.sum(dim=-1)  # (B, L)
                    tgt_eff = tgt.sum(dim=-1)  # (B, L)
                    flat_p = (pred_eff * mask.float()).flatten()
                    flat_t = (tgt_eff * mask.float()).flatten()
                    if flat_p.numel() > 1:
                        pc = flat_p - flat_p.mean()
                        tc = flat_t - flat_t.mean()
                        denom = (pc.pow(2).sum().sqrt() * tc.pow(2).sum().sqrt()).clamp_min(1e-8)
                        corr = (pc * tc).sum() / denom
                        n = int(mask.sum().item())
                        sum_rotation_corr += corr.item() * n
                        sum_rotation_corr_n += n

        bs = batch["label"].size(0)
        total_n += bs
        sum_loss += loss.item() * bs
        sum_L_rot += diag["L_rot"].item() * bs
        sum_L_pace += diag["L_pace"].item() * bs
        sum_L_pair += diag["L_pair"].item() * bs
        sum_L_player += diag["L_player"].item() * bs
        sum_L_team += diag["L_team"].item() * bs
        sum_L_win += diag["L_win"].item() * bs
        sum_team_nll_per_target += diag["team_nll_per_target"].cpu() * bs
        sum_player_nll_per_target += diag["player_nll_per_target"].cpu() * bs
        sum_pair_nll_per_target += diag["pair_nll_per_target"].cpu() * bs
        sum_bce += bce.item() * bs
        sum_margin_abs += margin_abs.item() * bs
        sum_pts_h += out["home_points"].mean().item() * bs
        sum_pts_a += out["away_points"].mean().item() * bs
        sum_pace_h += out["pace_home"].mean().item() * bs
        sum_pace_a += out["pace_away"].mean().item() * bs
        correct += correct_bs

    return {
        "n": total_n,
        "loss": sum_loss / total_n,
        "L_rot": sum_L_rot / total_n,
        "L_pace": sum_L_pace / total_n,
        "L_pair": sum_L_pair / total_n,
        "L_player": sum_L_player / total_n,
        "L_team": sum_L_team / total_n,
        "L_win": sum_L_win / total_n,
        "team_nll_per_target": (sum_team_nll_per_target / total_n).tolist(),
        "player_nll_per_target": (sum_player_nll_per_target / total_n).tolist(),
        "pair_nll_per_target": (sum_pair_nll_per_target / total_n).tolist(),
        "bce": sum_bce / total_n,
        "margin_mae": sum_margin_abs / total_n,
        "acc": correct / total_n,
        "mean_home_pts": sum_pts_h / total_n,
        "mean_away_pts": sum_pts_a / total_n,
        "mean_pace_home": sum_pace_h / total_n,
        "mean_pace_away": sum_pace_a / total_n,
        "rotation_corr": (sum_rotation_corr / sum_rotation_corr_n
                          if sum_rotation_corr_n else 0.0),
    }


# ----------------------------- CLI ----------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--player-game-stats-db", type=Path, default=DEFAULT_PLAYER_GAME_STATS_DB)
    p.add_argument("--player-debut-db", type=Path, default=DEFAULT_PLAYER_DEBUT_DB)
    p.add_argument("--decisions-db", type=Path, default=DEFAULT_PLAYER_DECISIONS_DB)
    p.add_argument("--v5-features-db", type=Path, default=DEFAULT_V5_FEATURES_DB)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="backtest")
    p.add_argument("--initial-train-end", default="2023-12-31")
    p.add_argument("--train-start", default=None,
                   help="Earliest game_date for training data. "
                        "Use '2020-01-01' or '2017-01-01'. Default: no cutoff.")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--val-frac-of-train", type=float, default=0.15)
    # backbone config
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers-stage1", type=int, default=1)
    p.add_argument("--n-layers-stage2", type=int, default=1)
    p.add_argument("--n-layers-stage3", type=int, default=1)
    p.add_argument("--pair-hidden", type=int, default=96)
    p.add_argument("--player-hidden", type=int, default=64)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--rotation-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pair-dropout", type=float, default=0.2)
    p.add_argument("--player-dropout", type=float, default=0.0)
    p.add_argument("--team-emb-dim", type=int, default=16)
    p.add_argument("--init-scale", type=float, default=12.0)
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--base-possessions", type=float, default=TEAM_BASE_POSSESSIONS)
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--no-cyclic", action="store_true")
    p.add_argument("--career-year-mode", choices=["rope", "none"], default="rope")
    p.add_argument("--career-year-pe-base", type=float, default=10000.0)
    p.add_argument("--use-player-scalars", action="store_true")
    p.add_argument("--use-player-form", action="store_true",
                   help="Feed per-player rolling form stats (10-dim) as input features")
    # loss weights
    p.add_argument("--rot-w", type=float, default=1.0)
    p.add_argument("--pace-w", type=float, default=0.01)
    p.add_argument("--pair-w", type=float, default=0.0)
    p.add_argument("--player-w", type=float, default=1.0)
    p.add_argument("--team-w", type=float, default=0.01)
    p.add_argument("--win-w", type=float, default=1.0)
    p.add_argument("--use-mse-player", action="store_true",
                   help="Use MSE instead of Poisson for player+team losses")
    p.add_argument("--direct-w", type=float, default=0.0,
                   help="Weight for direct player head loss")
    p.add_argument("--use-direct-player-head", action="store_true",
                   help="Add direct player box head branching post-Stage 1")
    p.add_argument("--direct-head-separate-emb", action="store_true",
                   help="Use separate scoring embedding for direct head")
    p.add_argument("--direct-emb-dim", type=int, default=32,
                   help="Dimension of separate scoring embedding")
    p.add_argument("--freeze-backbone-player-ft", action="store_true",
                   help="Phase1: rot+win. Phase2: freeze backbone, train player heads only")
    p.add_argument("--freeze-player-epochs", type=int, default=30,
                   help="Epochs for player-head fine-tune phase")
    p.add_argument("--freeze-player-patience", type=int, default=6)
    # staged warmup: rotation → pace/pair → player
    p.add_argument("--staged-warmup", action="store_true",
                   help="3-phase loss warmup: rot-only → ramp pace/pair → ramp player")
    p.add_argument("--warmup-rot-epochs", type=int, default=5,
                   help="Epochs of rotation-only (phase 1)")
    p.add_argument("--warmup-aux-epochs", type=int, default=5,
                   help="Epochs to linearly ramp in pace/pair (phase 2)")
    # fine-tune: drop aux losses, train on win-only
    p.add_argument("--finetune-epochs", type=int, default=0,
                   help="Extra epochs with win-only loss after main training (0=off)")
    p.add_argument("--finetune-lr-mult", type=float, default=0.3,
                   help="LR multiplier for fine-tune phase (relative to base LR)")
    p.add_argument("--finetune-patience", type=int, default=4,
                   help="Early stopping patience for fine-tune phase")
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--scalars-lr-mult", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--save-final", action="store_true")
    p.add_argument("--inspect-players", action="store_true")
    p.add_argument("--inspect-top-k", type=int, default=10)
    p.add_argument("--save-player-predictions", action="store_true")
    # parallel sharding
    p.add_argument("--window-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


# ----------------------------- helpers ----------------------------- #


def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    periods = pd.to_datetime(dates).dt.to_period("M").drop_duplicates().sort_values()
    return [period.to_timestamp() for period in periods]


def chrono_val_split(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def predict_window(
    model: CmeV5, records: list, batch_size: int, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(GameDatasetV5(records), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_v5)
    probs: list[np.ndarray] = []
    home_pts: list[np.ndarray] = []
    away_pts: list[np.ndarray] = []
    margin_mu: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_dev)
            probs.append(torch.sigmoid(out["win_logit"]).cpu().numpy())
            home_pts.append(out["home_points"].cpu().numpy())
            away_pts.append(out["away_points"].cpu().numpy())
            margin_mu.append(out["margin_mu"].cpu().numpy())
    gids = np.array([str(r.game_id) for r in records])
    return (
        gids,
        np.concatenate(probs),
        np.concatenate(home_pts),
        np.concatenate(away_pts),
        np.concatenate(margin_mu),
    )


def _build_lr_scheduler(optim, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        denom = max(1, total_epochs - warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def _build_param_groups(model, weight_decay: float, base_lr: float, scalars_lr_mult: float = 1.0):
    scalar_names = {"E_player_min.weight", "E_player_rate.weight"}
    scalar_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in scalar_names:
            scalar_params.append(p)
        else:
            other_params.append(p)
    groups = [{"params": other_params, "weight_decay": weight_decay, "lr": base_lr}]
    if scalar_params:
        groups.append({"params": scalar_params, "weight_decay": 0.0, "lr": base_lr * scalars_lr_mult})
    return groups


# ----------------------------- training loop ----------------------------- #


def _make_weight_schedule(args) -> callable:
    """Build epoch → loss-weight dict function.

    With --staged-warmup:
      Phase 1 (ep 1..R):           rot only + win
      Phase 2 (ep R+1..R+A):       linearly ramp pace_w, pair_w from 0 → target
      Phase 3 (ep R+A+1..end):     linearly ramp player_w from 0 → target

    Without: fixed weights every epoch.
    """
    base = dict(
        rot_w=args.rot_w, pace_w=args.pace_w, pair_w=args.pair_w,
        player_w=args.player_w, team_w=args.team_w, win_w=args.win_w,
        direct_w=args.direct_w,
    )
    if not args.staged_warmup:
        return lambda ep: base

    R = args.warmup_rot_epochs
    A = args.warmup_aux_epochs

    def schedule(ep: int) -> dict:
        if ep <= R:
            return dict(rot_w=args.rot_w, pace_w=0.0, pair_w=0.0,
                        player_w=0.0, team_w=0.0, win_w=args.win_w,
                        direct_w=0.0)
        elif ep <= R + A:
            t = (ep - R) / A
            return dict(rot_w=args.rot_w,
                        pace_w=args.pace_w * t, pair_w=args.pair_w * t,
                        player_w=0.0, team_w=0.0, win_w=args.win_w,
                        direct_w=0.0)
        else:
            remaining = args.epochs - (R + A)
            t = min(1.0, (ep - R - A) / max(remaining, 1))
            return dict(rot_w=args.rot_w,
                        pace_w=args.pace_w, pair_w=args.pair_w,
                        player_w=args.player_w * t, team_w=args.team_w * t,
                        win_w=args.win_w,
                        direct_w=args.direct_w * t)
    return schedule


def _run_training_loop(
    model, train_loader, val_loader, optim, scheduler,
    epoch_kwargs: dict, num_epochs: int, patience: int, device: str,
    track_metric: str, phase_label: int,
    weight_schedule: callable | None = None,
) -> tuple[dict | None, int, float, list[dict], dict]:
    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    best_val_diag: dict = {}
    epochs_since_best = 0
    history: list[dict] = []
    for epoch in range(1, num_epochs + 1):
        ek = dict(epoch_kwargs)
        if weight_schedule is not None:
            ek.update(weight_schedule(epoch))

        t0 = time.time()
        tr = run_epoch(model, train_loader, device=device, optim=optim, **ek)
        va = run_epoch(model, val_loader, device=device, optim=None, **ek)
        scheduler.step()
        dt = time.time() - t0

        phase_tag = ""
        if weight_schedule is not None:
            w = weight_schedule(epoch)
            phase_tag = f" w:rot={w['rot_w']:.2f}/pace={w['pace_w']:.3f}/pair={w['pair_w']:.2f}/plyr={w['player_w']:.2f}"

        history.append({
            "phase": phase_label, "epoch": epoch, "secs": dt,
            "train_loss": tr["loss"], "val_loss": va["loss"],
            "train_bce": tr["bce"], "val_bce": va["bce"],
            "train_acc": tr["acc"], "val_acc": va["acc"],
            "train_rotation_corr": tr["rotation_corr"],
            "val_rotation_corr": va["rotation_corr"],
            "train_mean_pace_home": tr["mean_pace_home"],
            "val_mean_pace_home": va["mean_pace_home"],
            "val_L_rot": va["L_rot"], "val_L_pace": va["L_pace"],
            "val_L_pair": va["L_pair"], "val_L_player": va["L_player"],
            "val_L_team": va["L_team"], "val_L_win": va["L_win"],
        })
        print(f"  [p{phase_label} ep{epoch:3d}] "
              f"tr_loss={tr['loss']:.4f} va_loss={va['loss']:.4f} "
              f"tr_bce={tr['bce']:.4f} va_bce={va['bce']:.4f} "
              f"va_acc={va['acc']:.3f} rot_r={va['rotation_corr']:.3f} "
              f"({dt:.0f}s){phase_tag}")
        if va[track_metric] < best_val - 1e-5:
            best_val = va[track_metric]
            best_epoch = epoch
            epochs_since_best = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val_diag = {
                "val_L_rot": va["L_rot"], "val_L_pace": va["L_pace"],
                "val_L_pair": va["L_pair"], "val_L_player": va["L_player"],
                "val_L_team": va["L_team"], "val_L_win": va["L_win"],
                "val_rotation_corr": va["rotation_corr"],
                "val_margin_mae": va["margin_mae"],
            }
        else:
            epochs_since_best += 1
            in_warmup = (weight_schedule is not None
                         and weight_schedule(epoch).get("player_w", 1.0) == 0.0)
            if epochs_since_best >= patience and not in_warmup:
                break
    return best_state, best_epoch, best_val, history, best_val_diag


# ----------------------------- one window ----------------------------- #


def train_one_window(
    args: argparse.Namespace,
    train_fit_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    histories, scores, play_decisions, game_odds, player_first_season,
    matchup_db: Path, core_db: Path, player_game_stats_db: Path,
    minute_presence, regulation_scores,
    player_histories=None,
) -> tuple[CmeV5, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_gids = [str(g) for g in train_fit_df["game_id"].tolist()]
    val_gids = [str(g) for g in val_df["game_id"].tolist()]
    test_gids = [str(g) for g in test_df["game_id"].tolist()]

    train_matchup = load_matchup_rows_v2(matchup_db, train_gids)
    val_matchup = load_matchup_rows_v2(matchup_db, val_gids)
    test_matchup = load_matchup_rows_v2(matchup_db, test_gids)
    train_box = load_box_minutes_and_pace(player_game_stats_db, train_gids)
    val_box = load_box_minutes_and_pace(player_game_stats_db, val_gids)
    test_box = load_box_minutes_and_pace(player_game_stats_db, test_gids)
    train_pl = load_player_game_stats(core_db, train_gids)
    val_pl = load_player_game_stats(core_db, val_gids)
    test_pl = load_player_game_stats(core_db, test_gids)

    vocab = build_vocab_from_records_v5(
        train_fit_df, histories, train_matchup, play_decisions,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(train_fit_df)
    tabular_stats = fit_tabular_stats(train_fit_df)

    player_form_stats = None
    if player_histories is not None:
        player_form_stats = fit_player_form_stats(
            player_histories, train_fit_df,
            lookback_games=DEFAULT_PLAYER_FORM_LOOKBACK,
            decay=DEFAULT_PLAYER_FORM_DECAY,
        )

    common = dict(
        histories=histories, vocab=vocab, team_vocab=team_vocab,
        play_decisions=play_decisions, game_scores=scores,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_first_season=player_first_season,
        player_histories=player_histories, player_form_stats=player_form_stats,
        player_form_lookback=DEFAULT_PLAYER_FORM_LOOKBACK,
        player_form_decay=DEFAULT_PLAYER_FORM_DECAY,
        game_odds=game_odds,
        minute_presence=minute_presence,
        regulation_scores=regulation_scores,
    )
    train_recs = build_records_v5(
        train_fit_df, matchup_rows=train_matchup,
        box_minutes_pace=train_box,
        player_game_stats=train_pl, **common,
    )
    val_recs = build_records_v5(
        val_df, matchup_rows=val_matchup,
        box_minutes_pace=val_box,
        player_game_stats=val_pl, **common,
    )
    test_recs = build_records_v5(
        test_df, matchup_rows=test_matchup,
        box_minutes_pace=test_box,
        player_game_stats=test_pl, **common,
    )

    train_loader = DataLoader(GameDatasetV5(train_recs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_v5)
    val_loader = DataLoader(GameDatasetV5(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_v5)

    import cme_v5_common as _cv5c
    tabular_dim = 0 if args.no_tabular else len(_cv5c.TABULAR_FEATURE_COLUMNS)
    cfg = CmeV5Config(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, n_heads=args.n_heads,
        n_layers_stage1=args.n_layers_stage1,
        n_layers_stage2=args.n_layers_stage2,
        n_layers_stage3=args.n_layers_stage3,
        pair_hidden=args.pair_hidden, player_hidden=args.player_hidden,
        head_hidden=args.head_hidden, rotation_hidden=args.rotation_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        tabular_dim=tabular_dim, team_emb_dim=args.team_emb_dim,
        player_stat_dim=PLAYER_FORM_DIM if args.use_player_form else 0,
        sinkhorn_iters=args.sinkhorn_iters,
        base_possessions_per_team=args.base_possessions,
        init_global_scale=args.init_scale,
        max_career_year=MAX_CAREER_YEAR,
        career_year_mode=args.career_year_mode,
        career_year_pe_base=args.career_year_pe_base,
        use_player_scalars=args.use_player_scalars,
        use_direct_player_head=args.use_direct_player_head,
        direct_head_separate_emb=args.direct_head_separate_emb,
        direct_emb_dim=args.direct_emb_dim,
    )
    model = CmeV5(cfg).to(args.device)

    box_weights = default_box_weights()
    pair_weights = default_pair_weights()

    history: list[dict] = []

    optim = torch.optim.AdamW(
        _build_param_groups(model, args.weight_decay, args.lr, args.scalars_lr_mult),
    )
    scheduler = _build_lr_scheduler(optim, args.warmup_epochs, args.epochs)
    epoch_kwargs = dict(
        box_weights=box_weights, pair_weights=pair_weights,
        rot_w=args.rot_w, pace_w=args.pace_w, pair_w=args.pair_w,
        player_w=args.player_w, team_w=args.team_w, win_w=args.win_w,
        direct_w=args.direct_w,
        use_mse_player=args.use_mse_player,
    )
    weight_schedule = _make_weight_schedule(args) if args.staged_warmup else None
    if args.staged_warmup:
        print(f"  [staged-warmup] rot-only: ep 1-{args.warmup_rot_epochs}, "
              f"ramp aux: ep {args.warmup_rot_epochs+1}-{args.warmup_rot_epochs+args.warmup_aux_epochs}, "
              f"ramp player: ep {args.warmup_rot_epochs+args.warmup_aux_epochs+1}+")
    best_state, best_epoch, best_val, single_history, best_diag = _run_training_loop(
        model, train_loader, val_loader, optim, scheduler,
        epoch_kwargs, args.epochs, args.patience,
        args.device, track_metric="bce", phase_label=0,
        weight_schedule=weight_schedule,
    )
    history.extend(single_history)

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.freeze_backbone_player_ft and best_state is not None:
        model.load_state_dict(best_state)

        aux_head_prefixes = ("pace_head", "tempo_bridge", "score_k", "score_q", "score_mlp")
        player_head_prefixes = ("eff_mlp", "eff_bias", "player_mlp", "player_bias", "direct_player_head", "E_player_scoring")

        # Phase 2: freeze backbone+rot+win, train aux heads (pace, pair scoring)
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(aux_head_prefixes)
        trainable_aux = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  [freeze-p2] training aux heads: {trainable_aux:,}/{total_params:,} params")
        ft2_optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        )
        ft2_scheduler = _build_lr_scheduler(
            ft2_optim, warmup_epochs=0, total_epochs=args.freeze_player_epochs,
        )
        ft2_kwargs = dict(
            box_weights=box_weights, pair_weights=pair_weights,
            rot_w=0.0, pace_w=1.0, pair_w=1.0,
            player_w=0.0, team_w=1.0, win_w=0.0,
            direct_w=0.0,
            use_mse_player=args.use_mse_player,
        )
        ft2_state, ft2_epoch, _, ft2_history, _ = _run_training_loop(
            model, train_loader, val_loader, ft2_optim, ft2_scheduler,
            ft2_kwargs, args.freeze_player_epochs, args.freeze_player_patience,
            args.device, track_metric="loss", phase_label=2,
        )
        history.extend(ft2_history)
        if ft2_state is not None:
            model.load_state_dict(ft2_state)
            print(f"  [freeze-p2] best epoch {ft2_epoch}")

        # Phase 3: unfreeze all, player heads at full LR, backbone at 1/30th
        for name, param in model.named_parameters():
            param.requires_grad = True
        backbone_params = []
        player_params = []
        for name, param in model.named_parameters():
            if name.startswith(player_head_prefixes):
                player_params.append(param)
            else:
                backbone_params.append(param)
        print(f"  [freeze-p3] player heads: full LR={args.lr:.1e}, "
              f"backbone: LR={args.lr/30:.1e}")
        ft3_optim = torch.optim.AdamW([
            {"params": player_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr / 30},
        ])
        ft3_scheduler = _build_lr_scheduler(
            ft3_optim, warmup_epochs=0, total_epochs=args.freeze_player_epochs,
        )
        ft3_kwargs = dict(
            box_weights=box_weights, pair_weights=pair_weights,
            rot_w=args.rot_w, pace_w=0.0, pair_w=0.0,
            player_w=args.player_w, team_w=0.0, win_w=args.win_w,
            direct_w=args.direct_w,
            use_mse_player=args.use_mse_player,
        )
        ft3_state, ft3_epoch, _, ft3_history, ft3_diag = _run_training_loop(
            model, train_loader, val_loader, ft3_optim, ft3_scheduler,
            ft3_kwargs, args.freeze_player_epochs, args.freeze_player_patience,
            args.device, track_metric="loss", phase_label=3,
        )
        history.extend(ft3_history)
        if ft3_state is not None:
            best_state = ft3_state
            best_epoch = f"{best_epoch}+aux{ft2_epoch}+pl{ft3_epoch}"
            best_diag = ft3_diag

        for name, param in model.named_parameters():
            param.requires_grad = True

    if args.finetune_epochs > 0 and best_state is not None:
        print(f"  [finetune] {args.finetune_epochs} epochs, win-only, "
              f"lr={args.lr * args.finetune_lr_mult:.1e}")
        ft_optim = torch.optim.AdamW(
            _build_param_groups(model, args.weight_decay,
                                args.lr * args.finetune_lr_mult,
                                args.scalars_lr_mult),
        )
        ft_scheduler = _build_lr_scheduler(
            ft_optim, warmup_epochs=0, total_epochs=args.finetune_epochs,
        )
        ft_kwargs = dict(
            box_weights=box_weights, pair_weights=pair_weights,
            rot_w=0.0, pace_w=0.0, pair_w=0.0,
            player_w=0.0, team_w=0.0, win_w=args.win_w,
        )
        ft_state, ft_epoch, ft_val, ft_history, ft_diag = _run_training_loop(
            model, train_loader, val_loader, ft_optim, ft_scheduler,
            ft_kwargs, args.finetune_epochs, args.finetune_patience,
            args.device, track_metric="bce", phase_label=1,
        )
        history.extend(ft_history)
        if ft_state is not None and ft_val < best_val:
            best_state = ft_state
            best_epoch = f"{best_epoch}+ft{ft_epoch}"
            best_val = ft_val
            best_diag = ft_diag
            model.load_state_dict(best_state)
            print(f"  [finetune] improved: val_bce={ft_val:.4f} at ft_ep={ft_epoch}")
        else:
            model.load_state_dict(best_state)
            print(f"  [finetune] no improvement (ft_best={ft_val:.4f} vs phase0={best_val:.4f})")

    gids, probs, h_pts, a_pts, margin_mu = predict_window(
        model, test_recs, args.batch_size, args.device,
    )

    window_info = {
        "n_train_fit": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "vocab_size": vocab.size,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "best_val_diag": best_diag,
    }
    return model, window_info, gids, probs, h_pts, a_pts, margin_mu


# ----------------------------- shard aggregation ----------------------------- #


def _aggregate_shards(out_dir: Path, args: argparse.Namespace) -> None:
    shards_dir = out_dir / "_shards"
    if not shards_dir.exists():
        raise FileNotFoundError(f"No shards dir at {shards_dir}")

    pred_files = sorted(shards_dir.glob("predictions.shard*.csv"))
    wm_files = sorted(shards_dir.glob("window_metrics.shard*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions shard files in {shards_dir}")
    print(f"[aggregate] {len(pred_files)} predictions, {len(wm_files)} window_metrics shard files")

    pred_df = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True)
    pred_df = pred_df.sort_values(["window_start", "game_date", "game_id"]).reset_index(drop=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    wm_df = pd.concat([pd.read_csv(p) for p in wm_files], ignore_index=True)
    wm_df = wm_df.sort_values("window_start").reset_index(drop=True)
    wm_df.to_csv(out_dir / "window_metrics.csv", index=False)

    labels = pred_df["label_home_win"].to_numpy(dtype=float)
    probs = pred_df["pred_home_win_prob"].to_numpy(dtype=float)
    bce = float(-np.mean(labels * np.log(np.clip(probs, 1e-7, 1 - 1e-7))
                         + (1 - labels) * np.log(np.clip(1 - probs, 1e-7, 1 - 1e-7))))
    acc = float(np.mean((probs > 0.5).astype(float) == labels))
    brier = float(np.mean((probs - labels) ** 2))
    diag_keys = ["val_L_rot", "val_L_pace", "val_L_pair", "val_L_player",
                  "val_L_team", "val_L_win", "val_rotation_corr", "val_margin_mae"]
    avg_diag = {}
    for k in diag_keys:
        if k in wm_df.columns:
            avg_diag[k] = float(wm_df[k].mean())

    overall = {
        "n": int(len(pred_df)),
        "bce": bce,
        "acc": acc,
        "brier": brier,
        "mean_prob": float(probs.mean()),
        "std_prob": float(probs.std()),
        **avg_diag,
    }
    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2) + "\n")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("\n[done] overall:")
    print(json.dumps(overall, indent=2))
    print(f"[done] artifacts: {out_dir}")


# ----------------------------- main ----------------------------- #


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        _aggregate_shards(out_dir, args)
        return

    if args.window_shards < 1 or not (0 <= args.shard_idx < args.window_shards):
        raise ValueError(
            f"Bad shard args: window_shards={args.window_shards}, shard_idx={args.shard_idx}"
        )

    if args.no_cyclic:
        import cme_v5_common as _cv5c
        from cme_v2_common import TABULAR_FEATURE_COLUMNS as _orig_cols
        import man_xfmr_common as _mxc
        trimmed = tuple(c for c in _orig_cols if not c.startswith("cyc_"))
        _mxc.TABULAR_FEATURE_COLUMNS = trimmed
        print(f"[no-cyclic] tabular columns trimmed to {len(trimmed)}")

    print(f"[device] {args.device}")
    print(f"[output] {out_dir}")

    print("[load] games + exposures + odds + scores + play_decisions + v5_features")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    all_gids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, all_gids)
    play_decisions = load_play_decisions(args.decisions_db, all_gids)
    game_odds = load_game_odds(args.core_db, all_gids)
    player_first_season = load_player_debut(
        args.player_debut_db, args.player_game_stats_db
    )
    minute_presence = load_minute_presence(args.v5_features_db)
    regulation_scores = load_regulation_scores(args.v5_features_db)
    player_histories = load_player_histories(args.core_db) if args.use_player_form else None
    print(f"[load] n_games={len(games_all)} odds={len(game_odds)} "
          f"play_decisions={len(play_decisions)} "
          f"minute_presence={len(minute_presence)} player-games "
          f"regulation_scores={len(regulation_scores)} games"
          + (f" player_histories={len(player_histories)} players" if player_histories else ""))

    if args.train_start:
        train_start = pd.Timestamp(args.train_start)
        n_before = len(games_all)
        games_all = games_all[games_all["game_date"] >= train_start].copy()
        print(f"[train-start] {args.train_start}: {n_before} -> {len(games_all)} games")

    initial_train_end = pd.Timestamp(args.initial_train_end)
    windows = [
        s for s in month_starts(games_all.loc[games_all["game_date"] > initial_train_end, "game_date"])
        if s > initial_train_end
    ]
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    if not windows:
        raise ValueError("No backtest windows found after initial_train_end.")
    print(f"[plan] {len(windows)} monthly windows from {windows[0].date()} to {windows[-1].date()}")

    if args.window_shards > 1:
        windows = windows[args.shard_idx::args.window_shards]
        shards_dir = out_dir / "_shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        tag = f".shard{args.shard_idx}"
        wm_out = shards_dir / f"window_metrics{tag}.csv"
        pred_out = shards_dir / f"predictions{tag}.csv"
        print(f"[shard {args.shard_idx}/{args.window_shards}] running {len(windows)} windows")
    else:
        wm_out = out_dir / "window_metrics.csv"
        pred_out = out_dir / "predictions.csv"

    all_predictions: list[pd.DataFrame] = []
    window_metrics: list[dict] = []

    for wi, window_start in enumerate(windows):
        window_end = window_start + pd.offsets.MonthBegin(1)
        train_block = games_all[games_all["game_date"] < window_start].copy()
        test_block = games_all[
            (games_all["game_date"] >= window_start) & (games_all["game_date"] < window_end)
        ].copy()
        if len(train_block) == 0 or len(test_block) == 0:
            print(f"[skip] window {window_start.date()}: empty train or test")
            continue

        train_fit, val = chrono_val_split(train_block, args.val_frac_of_train)
        t0 = time.time()
        print(f"\n[window {wi+1}/{len(windows)}] {window_start.date()} "
              f"train_fit={len(train_fit)} val={len(val)} test={len(test_block)}")

        try:
            (_model, info, gids, probs, h_pts, a_pts, margin_mu) = train_one_window(
                args, train_fit, val, test_block, histories, scores,
                play_decisions, game_odds, player_first_season,
                args.matchup_db, args.core_db,
                args.player_game_stats_db,
                minute_presence, regulation_scores,
                player_histories=player_histories,
            )
        except Exception as exc:
            print(f"[ERROR] window {window_start.date()}: {exc}")
            raise
        dt = time.time() - t0

        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"window_{window_start.date()}.pt"
        torch.save(_model.state_dict(), ckpt_path)

        gid_to_idx = {g: i for i, g in enumerate(gids)}
        gid_set = set(gid_to_idx)
        mask = test_block["game_id"].astype(str).isin(gid_set)
        if not mask.all():
            dropped = int((~mask).sum())
            print(f"[warn] {dropped} test games missing from predictions")
            test_block = test_block.loc[mask].copy()
        order = [gid_to_idx[str(g)] for g in test_block["game_id"].astype(str).tolist()]
        probs_ordered = probs[order]
        h_pts_ordered = h_pts[order]
        a_pts_ordered = a_pts[order]
        margin_ordered = margin_mu[order]

        labels = test_block["label_home_win"].to_numpy(dtype=float)
        bce = float(-np.mean(labels * np.log(np.clip(probs_ordered, 1e-7, 1-1e-7))
                              + (1 - labels) * np.log(np.clip(1 - probs_ordered, 1e-7, 1-1e-7))))
        acc = float(np.mean((probs_ordered > 0.5).astype(float) == labels))

        win_metrics = {
            "window_start": window_start.date().isoformat(),
            "window_end": (window_end - pd.Timedelta(days=1)).date().isoformat(),
            "train_n": len(train_fit),
            "val_n": len(val),
            "test_n": len(test_block),
            "train_start": train_fit["game_date"].min().date().isoformat(),
            "train_end": train_fit["game_date"].max().date().isoformat(),
            "best_epoch": info["best_epoch"],
            "epochs_run": info["epochs_run"],
            "test_bce": bce,
            "test_acc": acc,
            "secs": dt,
            **info.get("best_val_diag", {}),
        }
        window_metrics.append(win_metrics)
        print(f"[window {wi+1}/{len(windows)}] best_ep={info['best_epoch']} "
              f"test_bce={bce:.4f} test_acc={acc:.3f} ({dt:.0f}s)")

        pred_df = test_block[
            ["game_id", "game_date", "home_team_id", "away_team_id", "label_home_win"]
        ].copy()
        pred_df["pred_home_win_prob"] = probs_ordered
        pred_df["pred_home_pts"] = h_pts_ordered
        pred_df["pred_away_pts"] = a_pts_ordered
        pred_df["margin_mu"] = margin_ordered
        pred_df["window_start"] = window_start.date().isoformat()
        all_predictions.append(pred_df)

        predictions_df_partial = pd.concat(all_predictions, ignore_index=True)
        predictions_df_partial.to_csv(pred_out, index=False)
        pd.DataFrame(window_metrics).to_csv(wm_out, index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    labels = predictions_df["label_home_win"].to_numpy(dtype=float)
    probs = predictions_df["pred_home_win_prob"].to_numpy(dtype=float)
    overall_bce = float(-np.mean(labels * np.log(np.clip(probs, 1e-7, 1-1e-7))
                                  + (1 - labels) * np.log(np.clip(1 - probs, 1e-7, 1-1e-7))))
    overall_acc = float(np.mean((probs > 0.5).astype(float) == labels))
    overall_brier = float(np.mean((probs - labels) ** 2))

    diag_keys = ["val_L_rot", "val_L_pace", "val_L_pair", "val_L_player",
                  "val_L_team", "val_L_win", "val_rotation_corr", "val_margin_mae"]
    avg_diag = {}
    for k in diag_keys:
        vals = [wm[k] for wm in window_metrics if k in wm]
        if vals:
            avg_diag[k] = float(np.mean(vals))

    overall = {
        "n": int(len(predictions_df)),
        "bce": overall_bce,
        "acc": overall_acc,
        "brier": overall_brier,
        "mean_prob": float(probs.mean()),
        "std_prob": float(probs.std()),
        **avg_diag,
    }

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    predictions_df.to_csv(pred_out, index=False)
    pd.DataFrame(window_metrics).to_csv(wm_out, index=False)

    if args.window_shards == 1:
        (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2) + "\n")
        print("\n[done] overall:")
        print(json.dumps(overall, indent=2))
        print(f"[done] artifacts: {out_dir}")
    else:
        print(f"\n[shard {args.shard_idx}/{args.window_shards}] done — "
              f"{len(window_metrics)} windows, {len(predictions_df)} games. "
              f"Run --aggregate-only to merge shards.")


if __name__ == "__main__":
    main()
