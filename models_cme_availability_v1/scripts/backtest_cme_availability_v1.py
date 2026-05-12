#!/usr/bin/env python3
"""Expanding-window monthly backtest for CME-Availability-v1.

This mirrors the rolling-month evaluation used for CME-v4.2 while adding a
learned player availability/rotation prior.  Each window trains only on games
strictly before the month, holds out the latest ``--val-frac-of-train`` of that
historical block for early stopping, and predicts the games inside the month.
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cme_availability_v1_common import (  # noqa: E402
    AVAILABILITY_FEATURE_DIM,
    BOX_INDEX,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB,
    DEFAULT_LINEUP_DECAY,
    DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB,
    DEFAULT_PLAYER_FORM_DECAY,
    DEFAULT_PLAYER_FORM_LOOKBACK,
    GameDatasetAvailabilityV1,
    K_BOX,
    K_PAIR,
    PAIR_TARGETS,
    PLAYER_FORM_DIM,
    TABULAR_FEATURE_COLUMNS,
    build_records_availability_v1,
    build_team_game_player_seconds_index,
    build_team_vocab,
    build_vocab_from_records_availability_v1,
    collate_availability_v1,
    fit_availability_feature_stats,
    fit_player_form_stats,
    fit_tabular_stats,
    load_game_odds,
    load_game_player_status,
    load_game_player_status_details,
    load_game_scores,
    load_games,
    load_matchup_rows_v2,
    load_player_game_stats,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
)
from cme_availability_v1_model import CmeAvailabilityV1, CmeAvailabilityV1Config  # noqa: E402
from train_cme_availability_v1 import (  # noqa: E402
    default_box_weights,
    default_pair_weights,
    run_epoch,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_availability_v1" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="backtest")
    p.add_argument("--initial-train-end", default="2023-12-31",
                   help="First test window is the month strictly after this date.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="If set, only run the first N monthly windows.")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--val-frac-of-train", type=float, default=0.15)
    # backbone
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-self-layers", type=int, default=2)
    p.add_argument("--n-cross-layers", type=int, default=2)
    p.add_argument("--pair-hidden", type=int, default=96)
    p.add_argument("--player-hidden", type=int, default=64)
    p.add_argument("--inv-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pair-dropout", type=float, default=0.2)
    p.add_argument("--player-dropout", type=float, default=0.0)
    p.add_argument("--team-emb-dim", type=int, default=16)
    p.add_argument("--init-scale", type=float, default=12.0)
    p.add_argument("--calibration-hidden", type=int, default=64)
    p.add_argument("--calibration-dropout", type=float, default=0.05)
    p.add_argument("--max-calibration-residual", type=float, default=0.30)
    p.add_argument("--season-calibration-hidden", type=int, default=16)
    p.add_argument("--max-season-logit-adjustment", type=float, default=0.25)
    p.add_argument("--tail-gate-center", type=float, default=1.25)
    p.add_argument("--tail-gate-sharpness", type=float, default=2.0)
    p.add_argument("--init-home-logit-bias", type=float, default=0.15)
    p.add_argument("--init-win-logit-slope", type=float, default=1.0)
    p.add_argument("--trainable-global-scale", action="store_true")
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--base-possessions", type=float, default=491.0)
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--use-player-stats", action="store_true")
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)
    # availability branch
    p.add_argument("--availability-hidden", type=int, default=64)
    p.add_argument("--availability-dropout", type=float, default=0.05)
    p.add_argument("--max-play-logit-delta", type=float, default=3.0)
    p.add_argument("--max-minute-log-delta", type=float, default=2.0)
    p.add_argument("--init-role-prior-strength", type=float, default=0.25)
    # losses
    p.add_argument("--team-w", type=float, default=1.0)
    p.add_argument("--player-w", type=float, default=0.01)
    p.add_argument("--pair-w", type=float, default=0.001)
    p.add_argument("--inv-w", type=float, default=5.0)
    p.add_argument("--win-w", type=float, default=10.0)
    p.add_argument("--margin-nll-w", type=float, default=0.0)
    p.add_argument("--calibration-reg-w", type=float, default=0.075)
    p.add_argument("--season-calibration-reg-w", type=float, default=0.02)
    p.add_argument("--calibration-slope-reg-w", type=float, default=0.001)
    p.add_argument("--availability-play-w", type=float, default=0.50)
    p.add_argument("--availability-minutes-w", type=float, default=0.05)
    p.add_argument("--availability-role-w", type=float, default=2.0)
    p.add_argument("--box-weights", type=float, nargs="+", default=None)
    p.add_argument("--pair-weights", type=float, nargs="+", default=None)
    # optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--calibration-lr-mult", type=float, default=5.0)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoints", action="store_true")
    return p.parse_args()


def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    periods = pd.to_datetime(dates).dt.to_period("M").drop_duplicates().sort_values()
    return [period.to_timestamp() for period in periods]


def chrono_val_split(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def bce_from_probs(prob: np.ndarray, label: np.ndarray) -> float:
    p = np.clip(prob.astype(float), 1e-7, 1.0 - 1e-7)
    y = label.astype(float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def metrics_from_frame(df: pd.DataFrame) -> dict:
    p = df["pred_home_win_prob"].to_numpy(dtype=float)
    y = df["label_home_win"].to_numpy(dtype=float)
    out = {
        "n": int(len(df)),
        "bce": bce_from_probs(p, y),
        "acc": float(((p > 0.5).astype(float) == y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_prob": float(np.mean(p)),
        "std_prob": float(np.std(p)),
    }
    optional = {
        "pred_home_win_prob_base": "mean_base_prob",
        "pred_home_win_prob_affine": "mean_affine_prob",
        "pred_home_win_prob_temporal": "mean_temporal_prob",
        "season_logit_adjustment": "mean_season_logit_adjustment",
        "residual_tail_gate": "mean_residual_tail_gate",
        "win_calibration_residual": "mean_win_residual",
        "home_logit_bias": "mean_home_logit_bias",
        "win_logit_slope": "mean_win_logit_slope",
        "home_avail_play_prob_mean": "mean_home_avail_play_prob",
        "away_avail_play_prob_mean": "mean_away_avail_play_prob",
        "home_avail_play_actual_rate": "mean_home_avail_play_actual_rate",
        "away_avail_play_actual_rate": "mean_away_avail_play_actual_rate",
        "home_avail_log_seconds_pred_mean": "mean_home_avail_log_seconds_pred",
        "away_avail_log_seconds_pred_mean": "mean_away_avail_log_seconds_pred",
        "availability_role_prior_strength": "mean_availability_role_prior_strength",
    }
    for col, key in optional.items():
        if col in df.columns:
            out[key] = float(df[col].mean())
    out["actual_home_win_rate"] = float(np.mean(y))
    return out


def build_optimizer(model: CmeAvailabilityV1, args: argparse.Namespace) -> torch.optim.Optimizer:
    calibration_param_names = {"home_logit_bias", "log_win_logit_slope", "global_scale"}
    backbone_params: list[torch.nn.Parameter] = []
    calibration_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name in calibration_param_names
            or name.startswith("win_calibration_head.")
            or name.startswith("season_calibration_head.")
        ):
            calibration_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": calibration_params, "lr": args.lr * args.calibration_lr_mult, "weight_decay": 0.0},
    ])


def build_scheduler(optim: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        denom = max(1, total_epochs - warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def epoch_kwargs(args: argparse.Namespace, box_weights: torch.Tensor, pair_weights: torch.Tensor) -> dict:
    return dict(
        box_weights=box_weights,
        pair_weights=pair_weights,
        team_w=args.team_w,
        player_w=args.player_w,
        pair_w=args.pair_w,
        inv_w=args.inv_w,
        win_w=args.win_w,
        margin_nll_w=args.margin_nll_w,
        calibration_reg_w=args.calibration_reg_w,
        season_calibration_reg_w=args.season_calibration_reg_w,
        calibration_slope_reg_w=args.calibration_slope_reg_w,
        availability_play_w=args.availability_play_w,
        availability_minutes_w=args.availability_minutes_w,
        availability_role_w=args.availability_role_w,
    )


def train_one_window(
    model: CmeAvailabilityV1,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    args: argparse.Namespace,
    box_weights: torch.Tensor,
    pair_weights: torch.Tensor,
) -> tuple[dict, int, float, list[dict]]:
    optim = build_optimizer(model, args)
    scheduler = build_scheduler(optim, args.warmup_epochs, args.epochs)
    kwargs = epoch_kwargs(args, box_weights, pair_weights)
    best_state: dict | None = None
    best_epoch = -1
    best_bce = float("inf")
    no_improve = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device=args.device, optim=optim, **kwargs)
        va = run_epoch(model, val_loader, device=args.device, optim=None, **kwargs)
        scheduler.step()
        dt = time.time() - t0
        history.append({
            "epoch": epoch,
            "secs": dt,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_bce": tr["bce"],
            "val_bce": va["bce"],
            "train_acc": tr["acc"],
            "val_acc": va["acc"],
            "train_avail_play": tr["L_availability_play"],
            "val_avail_play": va["L_availability_play"],
            "train_avail_minutes": tr["L_availability_minutes"],
            "val_avail_minutes": va["L_availability_minutes"],
            "train_avail_role": tr["L_availability_role"],
            "val_avail_role": va["L_availability_role"],
            "train_availability_role_prior_strength": tr["availability_role_prior_strength"],
            "val_availability_role_prior_strength": va["availability_role_prior_strength"],
        })
        print(
            f"[ep{epoch:02d}] tr_bce={tr['bce']:.4f} tr_acc={tr['acc']:.3f} "
            f"va_bce={va['bce']:.4f} va_acc={va['acc']:.3f} "
            f"avail(play/min/role)={tr['L_availability_play']:.3f}/{tr['L_availability_minutes']:.3f}/{tr['L_availability_role']:.4f} "
            f"→ {va['L_availability_play']:.3f}/{va['L_availability_minutes']:.3f}/{va['L_availability_role']:.4f} "
            f"ρ={va['availability_role_prior_strength']:.3f} ({dt:.1f}s)"
        )
        if va["bce"] < best_bce - 1e-5:
            best_bce = float(va["bce"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early-stop] no val_bce improvement for {args.patience} epochs")
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = args.epochs
        best_bce = float("nan")
    model.load_state_dict(best_state)
    return best_state, best_epoch, best_bce, history


def predict_records(
    model: CmeAvailabilityV1,
    records: list,
    availability_stats,
    *,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    model.eval()
    loader = DataLoader(
        GameDatasetAvailabilityV1(records, availability_stats),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_availability_v1,
    )
    rows: list[dict] = []
    record_i = 0
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_dev)
            probs = torch.sigmoid(out["win_logit"]).detach().cpu().numpy()
            base_probs = torch.sigmoid(out["win_logit_base"]).detach().cpu().numpy()
            affine_probs = torch.sigmoid(out["win_logit_affine"]).detach().cpu().numpy()
            temporal_probs = torch.sigmoid(out["win_logit_temporal"]).detach().cpu().numpy()
            home_pts = out["home_points"].detach().cpu().numpy()
            away_pts = out["away_points"].detach().cpu().numpy()
            margins = out["margin_mu"].detach().cpu().numpy()
            season_adj = out["season_logit_adjustment"].detach().cpu().numpy()
            tail_gate = out["residual_tail_gate"].detach().cpu().numpy()
            win_res = out["win_residual"].detach().cpu().numpy()
            home_bias = float(out["home_logit_bias"].detach().cpu().item())
            slope = float(out["win_logit_slope"].detach().cpu().item())
            role_strength = float(out["availability_role_prior_strength"].detach().cpu().item())
            h_mask = batch["home_mask"].to(torch.float32)
            a_mask = batch["away_mask"].to(torch.float32)
            h_denom = h_mask.sum(dim=1).clamp_min(1.0)
            a_denom = a_mask.sum(dim=1).clamp_min(1.0)
            h_prob_mean = ((out["home_avail_play_prob"].detach().cpu() * h_mask).sum(dim=1) / h_denom).numpy()
            a_prob_mean = ((out["away_avail_play_prob"].detach().cpu() * a_mask).sum(dim=1) / a_denom).numpy()
            h_actual_rate = ((batch["home_avail_play_actual"] * h_mask).sum(dim=1) / h_denom).numpy()
            a_actual_rate = ((batch["away_avail_play_actual"] * a_mask).sum(dim=1) / a_denom).numpy()
            h_logsecs = ((out["home_avail_log_seconds_pred"].detach().cpu() * h_mask).sum(dim=1) / h_denom).numpy()
            a_logsecs = ((out["away_avail_log_seconds_pred"].detach().cpu() * a_mask).sum(dim=1) / a_denom).numpy()
            bs = len(probs)
            for j in range(bs):
                rec = records[record_i + j].base
                rows.append({
                    "game_id": rec.game_id,
                    "game_date": pd.Timestamp(rec.game_date).strftime("%Y-%m-%d"),
                    "home_team_id": rec.home_team_id,
                    "away_team_id": rec.away_team_id,
                    "label_home_win": int(rec.label),
                    "pred_home_win_prob": float(probs[j]),
                    "pred_home_win_prob_base": float(base_probs[j]),
                    "pred_home_win_prob_affine": float(affine_probs[j]),
                    "pred_home_win_prob_temporal": float(temporal_probs[j]),
                    "pred_home_pts": float(home_pts[j]),
                    "pred_away_pts": float(away_pts[j]),
                    "margin_mu": float(margins[j]),
                    "season_logit_adjustment": float(season_adj[j]),
                    "residual_tail_gate": float(tail_gate[j]),
                    "win_calibration_residual": float(win_res[j]),
                    "home_logit_bias": home_bias,
                    "win_logit_slope": slope,
                    "home_avail_play_prob_mean": float(h_prob_mean[j]),
                    "away_avail_play_prob_mean": float(a_prob_mean[j]),
                    "home_avail_play_actual_rate": float(h_actual_rate[j]),
                    "away_avail_play_actual_rate": float(a_actual_rate[j]),
                    "home_avail_log_seconds_pred_mean": float(h_logsecs[j]),
                    "away_avail_log_seconds_pred_mean": float(a_logsecs[j]),
                    "availability_role_prior_strength": role_strength,
                })
            record_i += bs
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {args.device}")
    print("[load] games + shared artifacts")
    games = load_games(args.features_db, min_games_before=args.min_games_before)
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    status_details = load_game_player_status_details(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)
    odds = load_game_odds(args.core_db, game_ids)
    actual_seconds_index = build_team_game_player_seconds_index(histories)

    initial_train_end = pd.Timestamp(args.initial_train_end)
    test_months = [m for m in month_starts(games["game_date"]) if m > initial_train_end]
    if args.max_windows is not None:
        test_months = test_months[: args.max_windows]
    print(f"[windows] {len(test_months)} months")

    all_predictions: list[pd.DataFrame] = []
    window_rows: list[dict] = []
    for w_i, window_start in enumerate(test_months, start=1):
        window_end = window_start + pd.offsets.MonthBegin(1)
        train_all = games[games["game_date"] < window_start].reset_index(drop=True)
        test_df = games[(games["game_date"] >= window_start) & (games["game_date"] < window_end)].reset_index(drop=True)
        if train_all.empty or test_df.empty:
            continue
        train_df, val_df = chrono_val_split(train_all, args.val_frac_of_train)
        print(f"\n[window {w_i}/{len(test_months)}] {window_start:%Y-%m} train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        train_gids = [str(g) for g in train_df["game_id"].tolist()]
        val_gids = [str(g) for g in val_df["game_id"].tolist()]
        test_gids = [str(g) for g in test_df["game_id"].tolist()]
        train_matchup = load_matchup_rows_v2(args.matchup_db, train_gids)
        val_matchup = load_matchup_rows_v2(args.matchup_db, val_gids)
        test_matchup = load_matchup_rows_v2(args.matchup_db, test_gids)
        train_pl = load_player_game_stats(args.core_db, train_gids)
        val_pl = load_player_game_stats(args.core_db, val_gids)
        test_pl = load_player_game_stats(args.core_db, test_gids)
        vocab = build_vocab_from_records_availability_v1(
            train_df, histories, train_matchup,
            lookback_games=args.lookback_games, decay=args.decay,
        )
        team_vocab = build_team_vocab(train_df)
        tabular_stats = fit_tabular_stats(train_df)
        if args.use_player_stats:
            player_histories = load_player_histories(args.core_db)
            player_form_stats = fit_player_form_stats(
                player_histories, train_df,
                lookback_games=args.player_form_lookback,
                decay=args.player_form_decay,
            )
        else:
            player_histories = None
            player_form_stats = None
        common = dict(
            histories=histories,
            vocab=vocab,
            team_vocab=team_vocab,
            status_lookup=statuses,
            status_details=status_details,
            calibration=calibration,
            game_scores=scores,
            lookback_games=args.lookback_games,
            decay=args.decay,
            tabular_stats=tabular_stats,
            player_histories=player_histories,
            player_form_stats=player_form_stats,
            player_form_lookback=args.player_form_lookback,
            player_form_decay=args.player_form_decay,
            game_odds=odds,
            actual_seconds_index=actual_seconds_index,
        )
        train_recs = build_records_availability_v1(train_df, matchup_rows=train_matchup, player_game_stats=train_pl, **common)
        val_recs = build_records_availability_v1(val_df, matchup_rows=val_matchup, player_game_stats=val_pl, **common)
        test_recs = build_records_availability_v1(test_df, matchup_rows=test_matchup, player_game_stats=test_pl, **common)
        if not train_recs or not val_recs or not test_recs:
            print("[skip] record construction returned an empty split")
            continue
        availability_stats = fit_availability_feature_stats(train_recs)
        train_loader = DataLoader(
            GameDatasetAvailabilityV1(train_recs, availability_stats),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_availability_v1,
        )
        val_loader = DataLoader(
            GameDatasetAvailabilityV1(val_recs, availability_stats),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_availability_v1,
        )
        tabular_dim = 0 if args.no_tabular else len(TABULAR_FEATURE_COLUMNS)
        cfg = CmeAvailabilityV1Config(
            vocab_size=vocab.size,
            num_teams=team_vocab.size,
            d=args.d,
            n_heads=args.n_heads,
            n_self_layers=args.n_self_layers,
            n_cross_layers=args.n_cross_layers,
            pair_hidden=args.pair_hidden,
            player_hidden=args.player_hidden,
            inv_hidden=args.inv_hidden,
            dropout=args.dropout,
            pair_dropout=args.pair_dropout,
            player_dropout=args.player_dropout,
            tabular_dim=tabular_dim,
            team_emb_dim=args.team_emb_dim,
            player_stat_dim=PLAYER_FORM_DIM if args.use_player_stats else 0,
            sinkhorn_iters=args.sinkhorn_iters,
            base_possessions_per_team=args.base_possessions,
            init_global_scale=args.init_scale,
            calibration_hidden=args.calibration_hidden,
            calibration_dropout=args.calibration_dropout,
            max_calibration_residual=args.max_calibration_residual,
            season_calibration_hidden=args.season_calibration_hidden,
            max_season_logit_adjustment=args.max_season_logit_adjustment,
            tail_gate_center=args.tail_gate_center,
            tail_gate_sharpness=args.tail_gate_sharpness,
            init_home_logit_bias=args.init_home_logit_bias,
            init_win_logit_slope=args.init_win_logit_slope,
            trainable_global_scale=args.trainable_global_scale,
            availability_feature_dim=AVAILABILITY_FEATURE_DIM,
            availability_hidden=args.availability_hidden,
            availability_dropout=args.availability_dropout,
            max_play_logit_delta=args.max_play_logit_delta,
            max_minute_log_delta=args.max_minute_log_delta,
            init_role_prior_strength=args.init_role_prior_strength,
        )
        model = CmeAvailabilityV1(cfg).to(args.device)
        box_weights = torch.tensor(args.box_weights, dtype=torch.float32) if args.box_weights is not None else default_box_weights()
        pair_weights = torch.tensor(args.pair_weights, dtype=torch.float32) if args.pair_weights is not None else default_pair_weights()
        if box_weights.numel() != K_BOX:
            raise ValueError(f"--box-weights must have length {K_BOX}")
        if pair_weights.numel() != K_PAIR:
            raise ValueError(f"--pair-weights must have length {K_PAIR}")
        best_state, best_epoch, best_val_bce, history = train_one_window(
            model,
            train_loader,
            val_loader,
            args=args,
            box_weights=box_weights,
            pair_weights=pair_weights,
        )
        if args.save_checkpoints:
            ckpt_dir = out_dir / "windows" / window_start.strftime("%Y-%m-%d")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": best_state,
                "cfg": asdict(cfg),
                "vocab": vocab.player_to_idx,
                "team_vocab": team_vocab.team_to_idx,
                "availability_feature_means": availability_stats.means.tolist(),
                "availability_feature_stds": availability_stats.stds.tolist(),
                "best_epoch": best_epoch,
                "best_val_bce": best_val_bce,
                "history": history,
            }, ckpt_dir / "best.pt")
        pred_df = predict_records(
            model,
            test_recs,
            availability_stats,
            batch_size=args.batch_size,
            device=args.device,
        )
        pred_df["window_start"] = window_start.strftime("%Y-%m-%d")
        metrics = metrics_from_frame(pred_df)
        metrics["window_start"] = window_start.strftime("%Y-%m-%d")
        metrics["best_epoch"] = int(best_epoch)
        metrics["best_val_bce"] = float(best_val_bce)
        window_rows.append(metrics)
        all_predictions.append(pred_df)
        print(
            f"[window done] bce={metrics['bce']:.4f} acc={metrics['acc']:.3f} "
            f"brier={metrics['brier']:.4f} p={metrics['mean_prob']:.3f}/{metrics['actual_home_win_rate']:.3f}"
        )

    if not all_predictions:
        raise RuntimeError("No backtest windows produced predictions")
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    window_metrics = pd.DataFrame(window_rows)
    overall = metrics_from_frame(predictions)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    window_metrics.to_csv(out_dir / "window_metrics.csv", index=False)
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    config_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    print("[done] overall:")
    print(json.dumps(overall, indent=2))
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
