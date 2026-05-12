#!/usr/bin/env python3
"""Expanding-window monthly backtest for CME-Ratings-v2.

Mirrors `models_cme_v2/scripts/backtest_cme_v2.py`: for each month-start
after --initial-train-end, train on all games strictly before that month
and predict every game inside the month. Predictions are stitched into a
single CSV with the same schema as the baseline / v2 backtests
(game_id, game_date, home_team_id, away_team_id, label_home_win,
pred_home_win_prob, window_start) so it can be fed straight into
evaluate_betting_strategy.py.

Within each window the most recent --val-frac-of-train fraction of the
train block is held out chronologically for early stopping.

Defaults match `train_cme_v4_2.py`'s nine-level loss
(team=1.0, player=0.01, pair=0.001, inv=5.0, win=10.0, margin_nll=0.0,
plus regularized calibration terms).
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
RATINGS_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(RATINGS_SCRIPTS))

from cme_ratings_v2_common import (  # noqa: E402
    BOX_INDEX, K_BOX, K_PAIR, K_RATING_FEATURES,
    DEFAULT_CALIBRATION_PATH, DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB, DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    DEFAULT_RATINGS_DB, RATING_FEATURE_COLUMNS,
    PLAYER_FORM_DIM, TABULAR_FEATURE_COLUMNS,
    GameDatasetRatingsV2, build_records_ratings_v2, build_team_vocab,
    build_vocab_from_records_v42, collate_ratings_v2,
    fit_rating_feature_stats, fit_tabular_stats,
    load_game_odds, load_game_player_status, load_game_scores, load_games,
    load_matchup_rows_v2, load_player_game_stats,
    load_rating_priors, load_status_calibration, load_team_exposures,
)
from cme_ratings_v2_model import CmeRatingsV2, CmeRatingsV2Config  # noqa: E402
from train_cme_ratings_v2 import (  # noqa: E402
    default_box_weights, default_pair_weights, run_epoch,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_ratings_v2" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--ratings-db", type=Path, default=DEFAULT_RATINGS_DB,
                   help="Dynamic team ratings artifact from data/scripts/build_team_strength_ratings.py.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="backtest")
    p.add_argument("--initial-train-end", default="2023-12-31",
                   help="First test window is the month strictly after this date.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="If set, only run the first N windows (for smoke tests).")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--val-frac-of-train", type=float, default=0.15,
                   help="Within each window's train block, hold out the most recent X%% as val.")
    # v4 backbone config
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
    p.add_argument("--calibration-hidden", type=int, default=64,
                   help="Hidden size for the v4.2 context-residual head.")
    p.add_argument("--calibration-dropout", type=float, default=0.05,
                   help="Dropout inside the v4.2 context-residual head.")
    p.add_argument("--max-calibration-residual", type=float, default=0.30,
                   help="Absolute tanh-bounded context residual after affine + season calibration.")
    p.add_argument("--season-calibration-hidden", type=int, default=16,
                   help="Hidden size for the v4.2 season-phase calibration head.")
    p.add_argument("--max-season-logit-adjustment", type=float, default=0.25,
                   help="Absolute tanh-bounded season-phase logit adjustment.")
    p.add_argument("--tail-gate-center", type=float, default=1.25,
                   help="Absolute temporal logit where the residual gate is 0.5.")
    p.add_argument("--tail-gate-sharpness", type=float, default=2.0,
                   help="Sharpness of residual suppression in extreme probability tails; <=0 disables gating.")
    p.add_argument("--init-home-logit-bias", type=float, default=0.15,
                   help="Warm start for the learned home-side logit intercept.")
    p.add_argument("--init-win-logit-slope", type=float, default=1.0,
                   help="Initialization for the positive affine slope on margin/scale logits.")
    p.add_argument("--trainable-global-scale", action="store_true",
                   help="Allow the margin-to-win logistic scale to move during training. Default keeps v3's fixed scale.")
    p.add_argument("--cme-residual-hidden", type=int, default=32,
                   help="Hidden size for the ratings-first CME residual head.")
    p.add_argument("--cme-residual-dropout", type=float, default=0.05,
                   help="Dropout inside the ratings-first CME residual head.")
    p.add_argument("--rating-margin-scale", type=float, default=12.0,
                   help="Convert the pregame rating margin prior from points to raw logits.")
    p.add_argument("--init-rating-logit-bias", type=float, default=0.04,
                   help="Warm start for the ratings-prior calibration intercept.")
    p.add_argument("--init-rating-logit-slope", type=float, default=1.50,
                   help="Warm start for the positive ratings-prior calibration slope.")
    p.add_argument("--max-cme-gap-weight", type=float, default=0.35,
                   help="Absolute tanh bound on the linear CME-vs-ratings gap weight.")
    p.add_argument("--init-cme-gap-weight", type=float, default=0.0,
                   help="Initialization for the bounded CME-vs-ratings linear gap weight.")
    p.add_argument("--max-cme-feature-residual", type=float, default=0.15,
                   help="Absolute tanh-bound for the learned CME residual correction.")
    p.add_argument("--disagreement-gate-center", type=float, default=0.80,
                   help="Gate intercept before confidence/disagreement suppression.")
    p.add_argument("--disagreement-gate-rating-confidence-w", type=float, default=1.00,
                   help="Suppress CME correction when calibrated ratings are confident.")
    p.add_argument("--disagreement-gate-gap-w", type=float, default=0.75,
                   help="Suppress CME correction when ratings and CME disagree strongly.")
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--base-possessions", type=float, default=491.0)
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--no-cyclic", action="store_true",
                   help="Drop the 12 cyc_* columns from the tabular feature set (pre-cyclic baseline).")
    # nine-level loss weights (match train_cme_v4_2 defaults)
    p.add_argument("--team-w", type=float, default=1.0)
    p.add_argument("--player-w", type=float, default=0.01)
    p.add_argument("--pair-w", type=float, default=0.001)
    p.add_argument("--inv-w", type=float, default=5.0)
    p.add_argument("--win-w", type=float, default=10.0)
    p.add_argument("--margin-nll-w", type=float, default=0.0)
    p.add_argument("--calibration-reg-w", type=float, default=0.075,
                   help="L2 penalty weight on v4.2's tail-gated context residual.")
    p.add_argument("--season-calibration-reg-w", type=float, default=0.02,
                   help="L2 penalty weight on the season-phase logit adjustment.")
    p.add_argument("--calibration-slope-reg-w", type=float, default=0.001,
                   help="Soft log-slope prior around 1.0 for the affine margin calibration.")
    p.add_argument("--cme-feature-reg-w", type=float, default=0.02,
                   help="L2 penalty on the bounded CME feature residual added to the ratings anchor.")
    p.add_argument("--cme-gap-weight-reg-w", type=float, default=0.002,
                   help="L2 penalty on the bounded linear CME-vs-ratings gap weight.")
    p.add_argument("--rating-bias-reg-w", type=float, default=0.0002,
                   help="Small L2 penalty on the learned ratings calibration intercept.")
    p.add_argument("--box-weights", type=float, nargs="+", default=None,
                   help=f"Per-stat weights for team+player levels (length {K_BOX}). "
                        f"Default = default_box_weights().")
    p.add_argument("--pair-weights", type=float, nargs="+", default=None,
                   help=f"Per-target weights for pair Poisson NLL (length {K_PAIR}). "
                        f"Default = default_pair_weights().")
    # curriculum (two-phase) training
    p.add_argument("--curriculum", action="store_true",
                   help="Two-phase training: Phase 1 supervises structural losses "
                        "(team/player/pair MSE) to convergence; then hard-switch to "
                        "cfg-D (BCE + exposure-only pair Poisson) and fine-tune.")
    p.add_argument("--phase1-team-w", type=float, default=1.0)
    p.add_argument("--phase1-player-w", type=float, default=0.01)
    p.add_argument("--phase1-pair-w", type=float, default=0.1)
    p.add_argument("--phase1-inv-w", type=float, default=0.0,
                   help="Usage-share supervision (involvement MSE) weight in "
                        "Phase 1. Default 0 preserves earlier curriculum "
                        "behavior; set >0 to directly supervise alpha_off / "
                        "alpha_def per player.")
    p.add_argument("--phase1-epochs", type=int, default=50,
                   help="Max epochs for Phase 1 (curriculum mode).")
    p.add_argument("--phase1-patience", type=int, default=6,
                   help="Patience (in epochs) for Phase 1 plateau early stop. "
                        "Tracks val structural loss, not val_bce.")
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--calibration-lr-mult", type=float, default=5.0,
                   help="Learning-rate multiplier for home bias, affine slope, global scale, and residual head; these params use no weight decay.")
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoints", action="store_true",
                   help="Write best.pt per window to "
                        "<out>/windows/<window_start>/best.pt (with cfg/vocab "
                        "needed to reload the model).")
    p.add_argument("--save-final", action="store_true",
                   help="Additionally write final.pt (last-epoch weights) per "
                        "window. No effect without --save-checkpoints.")
    p.add_argument("--inspect-players", action="store_true",
                   help="After training, print per-player predicted vs actual "
                        "(exposure / pts / alpha_off) for the first 2 test "
                        "games of each window. Diagnostic only; no effect on "
                        "predictions.")
    p.add_argument("--inspect-top-k", type=int, default=10,
                   help="Top K players per roster to show in --inspect-players "
                        "(sorted by predicted exposure).")
    p.add_argument("--track-pids", type=str, default=None,
                   help="Comma-separated list of player IDs to track across "
                        "ALL test games in each window. For each (game, pid), "
                        "prints one line of predicted vs actual stats.")
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


def predict_window(
    model: CmeRatingsV2, records: list, batch_size: int, device: str,
) -> tuple[np.ndarray, ...]:
    model.eval()
    loader = DataLoader(GameDatasetRatingsV2(records), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_ratings_v2)
    probs: list[np.ndarray] = []
    base_probs: list[np.ndarray] = []
    affine_probs: list[np.ndarray] = []
    temporal_probs: list[np.ndarray] = []
    season_logit_adjustments: list[np.ndarray] = []
    residual_tail_gates: list[np.ndarray] = []
    win_residuals: list[np.ndarray] = []
    rating_margin_priors: list[np.ndarray] = []
    rating_logit_priors: list[np.ndarray] = []
    rating_probs_raw: list[np.ndarray] = []
    rating_logits_calibrated: list[np.ndarray] = []
    rating_probs_calibrated: list[np.ndarray] = []
    rating_logit_biases: list[np.ndarray] = []
    rating_logit_slopes: list[np.ndarray] = []
    cme_gaps: list[np.ndarray] = []
    disagreement_gates: list[np.ndarray] = []
    cme_gap_weights: list[np.ndarray] = []
    cme_gap_linear_residuals: list[np.ndarray] = []
    cme_feature_residuals: list[np.ndarray] = []
    home_logit_biases: list[np.ndarray] = []
    win_logit_slopes: list[np.ndarray] = []
    home_pts: list[np.ndarray] = []
    away_pts: list[np.ndarray] = []
    margin_mu: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_dev)
            probs_arr = torch.sigmoid(out["win_logit"]).cpu().numpy()
            probs.append(probs_arr)
            base_probs.append(torch.sigmoid(out["win_logit_base"]).cpu().numpy())
            affine_probs.append(torch.sigmoid(out["win_logit_affine"]).cpu().numpy())
            temporal_probs.append(torch.sigmoid(out["win_logit_temporal"]).cpu().numpy())
            season_logit_adjustments.append(out["season_logit_adjustment"].cpu().numpy())
            residual_tail_gates.append(out["residual_tail_gate"].cpu().numpy())
            win_residuals.append(out["win_residual"].cpu().numpy())
            rating_margin_priors.append(out["rating_margin_prior"].cpu().numpy())
            rating_logit_priors.append(out["rating_logit_prior"].cpu().numpy())
            rating_probs_raw.append(out["rating_prob_raw"].cpu().numpy())
            rating_logits_calibrated.append(out["rating_logit_calibrated"].cpu().numpy())
            rating_probs_calibrated.append(out["rating_prob_calibrated"].cpu().numpy())
            rating_logit_biases.append(np.full(probs_arr.shape, out["rating_logit_bias"].detach().cpu().item()))
            rating_logit_slopes.append(np.full(probs_arr.shape, out["rating_logit_slope"].detach().cpu().item()))
            cme_gaps.append(out["cme_gap"].cpu().numpy())
            disagreement_gates.append(out["disagreement_gate"].cpu().numpy())
            cme_gap_weights.append(np.full(probs_arr.shape, out["cme_gap_weight"].detach().cpu().item()))
            cme_gap_linear_residuals.append(out["cme_gap_linear_residual"].cpu().numpy())
            cme_feature_residuals.append(out["cme_feature_residual"].cpu().numpy())
            home_logit_biases.append(np.full(probs_arr.shape, out["home_logit_bias"].detach().cpu().item()))
            win_logit_slopes.append(np.full(probs_arr.shape, out["win_logit_slope"].detach().cpu().item()))
            home_pts.append(out["home_points"].cpu().numpy())
            away_pts.append(out["away_points"].cpu().numpy())
            margin_mu.append(out["margin_mu"].cpu().numpy())
    gids = np.array([str(r.game_id) for r in records])
    return (
        gids,
        np.concatenate(probs),
        np.concatenate(base_probs),
        np.concatenate(affine_probs),
        np.concatenate(temporal_probs),
        np.concatenate(season_logit_adjustments),
        np.concatenate(residual_tail_gates),
        np.concatenate(win_residuals),
        np.concatenate(rating_margin_priors),
        np.concatenate(rating_logit_priors),
        np.concatenate(rating_probs_raw),
        np.concatenate(rating_logits_calibrated),
        np.concatenate(rating_probs_calibrated),
        np.concatenate(rating_logit_biases),
        np.concatenate(rating_logit_slopes),
        np.concatenate(cme_gaps),
        np.concatenate(disagreement_gates),
        np.concatenate(cme_gap_weights),
        np.concatenate(cme_gap_linear_residuals),
        np.concatenate(cme_feature_residuals),
        np.concatenate(home_logit_biases),
        np.concatenate(win_logit_slopes),
        np.concatenate(home_pts),
        np.concatenate(away_pts),
        np.concatenate(margin_mu),
    )


def _build_ckpt_payload(model, cfg, vocab, team_vocab, box_weights, pair_weights,
                        team_w, player_w, pair_w, inv_w, win_w, margin_nll_w,
                        calibration_reg_w, season_calibration_reg_w, calibration_slope_reg_w,
                        cme_feature_reg_w, cme_gap_weight_reg_w, rating_bias_reg_w,
                        rating_feature_stats, epoch):
    return {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": asdict(cfg),
        "vocab": vocab.player_to_idx,
        "team_vocab": team_vocab.team_to_idx,
        "player_form_means": None,
        "player_form_stds": None,
        "box_weights": box_weights.tolist(),
        "pair_weights": pair_weights.tolist(),
        "rating_feature_columns": list(RATING_FEATURE_COLUMNS),
        "rating_feature_medians": (rating_feature_stats.medians.tolist() if rating_feature_stats is not None else None),
        "rating_feature_means": (rating_feature_stats.means.tolist() if rating_feature_stats is not None else None),
        "rating_feature_stds": (rating_feature_stats.stds.tolist() if rating_feature_stats is not None else None),
        "loss_level_weights": {
            "team": team_w, "player": player_w, "pair": pair_w,
            "inv": inv_w, "win": win_w, "margin_nll": margin_nll_w,
            "calibration_reg": calibration_reg_w,
            "season_calibration_reg": season_calibration_reg_w,
            "calibration_slope_reg": calibration_slope_reg_w,
            "cme_feature_reg": cme_feature_reg_w,
            "cme_gap_weight_reg": cme_gap_weight_reg_w,
            "rating_bias_reg": rating_bias_reg_w,
        },
        "epoch": epoch,
    }


def _build_optimizer(model: CmeRatingsV2, args) -> torch.optim.Optimizer:
    calibration_param_names = {"home_logit_bias", "log_win_logit_slope", "global_scale"}
    backbone_params = []
    calibration_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (name in calibration_param_names
                or name.startswith("win_calibration_head.")
                or name.startswith("season_calibration_head.")
                or name in {"rating_logit_bias", "log_rating_logit_slope", "raw_cme_gap_weight"}
                or name.startswith("cme_residual_head.")):
            calibration_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": calibration_params, "lr": args.lr * args.calibration_lr_mult, "weight_decay": 0.0},
    ])


def _build_lr_scheduler(optim, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        denom = max(1, total_epochs - warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def _per_task_grad_norms(
    model, loader, device: str,
    box_weights: torch.Tensor, pair_weights: torch.Tensor,
) -> dict:
    """Compute per-task ||grad of L_*|| on one batch. Unweighted by outer task_w."""
    import torch.nn.functional as F
    from cme_v4_2_model import (
        involvement_mse_loss, pair_poisson_loss, player_mse_loss, team_mse_loss,
    )

    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    box_w = box_weights.to(device)
    pair_w = pair_weights.to(device)

    def _norm() -> float:
        sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                sq += p.grad.detach().pow(2).sum().item()
        return sq ** 0.5

    out: dict = {}
    model.train()
    # team
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    L = (box_w * team_mse_loss(fwd, batch)).sum()
    L.backward()
    out["team"] = (_norm(), L.item())
    # player
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    pl = player_mse_loss(fwd, batch)
    if pl.numel() > 0:
        L = (box_w * pl).sum()
        L.backward()
        out["player"] = (_norm(), L.item())
    else:
        out["player"] = (0.0, 0.0)
    # pair
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    pair_nll, _ = pair_poisson_loss(fwd, batch)
    L = (pair_w * pair_nll).sum()
    L.backward()
    out["pair"] = (_norm(), L.item())
    # inv (involvement / usage MSE)
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    L = involvement_mse_loss(fwd, batch)
    L.backward()
    out["inv"] = (_norm(), L.item())
    # win
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    L = F.binary_cross_entropy_with_logits(fwd["win_logit"], batch["label"])
    L.backward()
    out["win"] = (_norm(), L.item())

    model.zero_grad(set_to_none=True)
    return out


def _fmt_grad_norms(label: str, norms: dict, task_weights: dict) -> str:
    """Render grad norms in a one-line table. task_weights gives outer w_* per task."""
    parts = [label]
    for task in ("team", "player", "pair", "inv", "win"):
        if task not in norms:
            continue
        g, L = norms[task]
        w = task_weights.get(task, 1.0)
        parts.append(f"{task}=L{L:7.3f} g{g:7.3f} w{w:.3g} wg{w*g:7.3f}")
    return " | ".join(parts)


def _run_training_loop(
    model, train_loader, val_loader, optim, scheduler,
    epoch_kwargs: dict, num_epochs: int, patience: int, device: str,
    track_metric: str, phase_label: int,
) -> tuple[dict | None, int, float, list[dict]]:
    """Run a training loop with early stopping. Returns (best_state, best_epoch, best_metric, history)."""
    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict] = []
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device=device, optim=optim, **epoch_kwargs)
        va = run_epoch(model, val_loader, device=device, optim=None, **epoch_kwargs)
        scheduler.step()
        dt = time.time() - t0
        history.append({
            "phase": phase_label, "epoch": epoch, "secs": dt,
            "train_loss": tr["loss"], "val_loss": va["loss"],
            "train_bce": tr["bce"], "val_bce": va["bce"],
            "train_acc": tr["acc"], "val_acc": va["acc"],
            "train_alpha_off_corr": tr["alpha_off_corr"],
            "val_alpha_off_corr": va["alpha_off_corr"],
            "train_mean_N": tr["mean_N"], "val_mean_N": va["mean_N"],
            "train_mean_prob": tr["mean_prob"], "val_mean_prob": va["mean_prob"],
            "train_mean_base_prob": tr["mean_base_prob"], "val_mean_base_prob": va["mean_base_prob"],
            "train_mean_affine_prob": tr["mean_affine_prob"], "val_mean_affine_prob": va["mean_affine_prob"],
            "train_mean_temporal_prob": tr["mean_temporal_prob"], "val_mean_temporal_prob": va["mean_temporal_prob"],
            "train_mean_label": tr["mean_label"], "val_mean_label": va["mean_label"],
            "train_mean_season_logit_adjustment": tr["mean_season_logit_adjustment"], "val_mean_season_logit_adjustment": va["mean_season_logit_adjustment"],
            "train_mean_residual_tail_gate": tr["mean_residual_tail_gate"], "val_mean_residual_tail_gate": va["mean_residual_tail_gate"],
            "train_mean_win_residual": tr["mean_win_residual"], "val_mean_win_residual": va["mean_win_residual"],
            "train_home_logit_bias": tr["home_logit_bias"], "val_home_logit_bias": va["home_logit_bias"],
            "train_win_logit_slope": tr["win_logit_slope"], "val_win_logit_slope": va["win_logit_slope"],
        })
        if va[track_metric] < best_val - 1e-5:
            best_val = va[track_metric]
            best_epoch = epoch
            epochs_since_best = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break
    return best_state, best_epoch, best_val, history


def _inspect_player_predictions(
    model, test_recs, vocab, device: str, n_games: int = 2, top_k: int = 10,
) -> None:
    """Print per-player predicted vs actual for the first n_games test games.

    Diagnostic only. Shows top_k home/away players sorted by predicted
    exposure with predicted vs actual exposure / pts / alpha_off.
    """
    if not test_recs:
        return
    idx_to_pid = {i: pid for pid, i in vocab.player_to_idx.items()}
    sample = test_recs[: max(1, n_games)]
    loader = DataLoader(GameDatasetRatingsV2(sample), batch_size=len(sample),
                        shuffle=False, collate_fn=collate_ratings_v2)
    pts_i = BOX_INDEX["pts"]
    fgm_i = BOX_INDEX["fgm"]
    fga_i = BOX_INDEX["fga"]
    ast_i = BOX_INDEX["ast"]

    model.eval()
    with torch.no_grad():
        batch_cpu = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch_cpu.items()}
        fwd = model(batch)
        win_prob = torch.sigmoid(fwd["win_logit"]).cpu().numpy()
        for b in range(len(sample)):
            rec = sample[b]
            pred_h = fwd["home_team_box"][b].cpu().numpy()
            pred_a = fwd["away_team_box"][b].cpu().numpy()
            act_h = batch_cpu["team_box_home"][b].cpu().numpy()
            act_a = batch_cpu["team_box_away"][b].cpu().numpy()

            print(f"\n[player-diag] gid={rec.game_id} {rec.game_date.date()} "
                  f"home={rec.home_team_id} away={rec.away_team_id} "
                  f"label_home_win={int(batch_cpu['label'][b].item())} "
                  f"win_prob={win_prob[b]:.3f}")
            print(f"  team   pred  H={pred_h[pts_i]:6.1f}  A={pred_a[pts_i]:6.1f}  "
                  f"|  actual  H={act_h[pts_i]:.0f}  A={act_a[pts_i]:.0f}")

            for side, side_name in [(0, "HOME"), (1, "AWAY")]:
                pair_marg = (fwd["home_pair_marg"] if side == 0 else fwd["away_pair_marg"])[b].cpu().numpy()
                box = (fwd["home_box"] if side == 0 else fwd["away_box"])[b].cpu().numpy()
                alpha_off = (fwd["alpha_home_off"] if side == 0 else fwd["alpha_away_off"])[b].cpu().numpy()
                mask = (batch_cpu["home_mask"] if side == 0 else batch_cpu["away_mask"])[b].cpu().numpy().astype(bool)
                idx = (batch_cpu["home_idx"] if side == 0 else batch_cpu["away_idx"])[b].cpu().numpy()
                prob = (batch_cpu["home_prob"] if side == 0 else batch_cpu["away_prob"])[b].cpu().numpy()
                actual_alpha = (batch_cpu["home_alpha_off_actual"] if side == 0
                                else batch_cpu["away_alpha_off_actual"])[b].cpu().numpy()

                # actual per-player exposure & pts from flattened sup tensors
                sup_pair_game = batch_cpu["sup_pair_game"].cpu().numpy()
                sup_pair_side = batch_cpu["sup_pair_side"].cpu().numpy()
                sup_pair_off = batch_cpu["sup_pair_off"].cpu().numpy()
                sup_pair_y = batch_cpu["sup_pair_y"].cpu().numpy()
                sup_pl_game = batch_cpu["sup_pl_game"].cpu().numpy()
                sup_pl_side = batch_cpu["sup_pl_side"].cpu().numpy()
                sup_pl_slot = batch_cpu["sup_pl_slot"].cpu().numpy()
                sup_pl_y = batch_cpu["sup_pl_y"].cpu().numpy()

                L = mask.sum()
                rows = []
                for s in range(int(mask.size)):
                    if not mask[s]:
                        continue
                    pid = idx_to_pid.get(int(idx[s]), f"OOV/{int(idx[s])}")
                    pred_exp = float(pair_marg[s, 0])
                    pred_pts = float(box[s, pts_i])
                    pred_a_off = float(alpha_off[s])
                    # actual exposure: sum sup_pair_y[..., 0] for this game/side/off_slot
                    m_pair = (sup_pair_game == b) & (sup_pair_side == side) & (sup_pair_off == s)
                    act_exp = float(sup_pair_y[m_pair, 0].sum()) if m_pair.any() else 0.0
                    # actual per-player full box for this slot
                    m_pl = (sup_pl_game == b) & (sup_pl_side == side) & (sup_pl_slot == s)
                    if m_pl.any():
                        i_row = int(np.flatnonzero(m_pl)[0])
                        act_pts = float(sup_pl_y[i_row, pts_i])
                        act_fgm = float(sup_pl_y[i_row, fgm_i])
                        act_fga = float(sup_pl_y[i_row, fga_i])
                        act_ast = float(sup_pl_y[i_row, ast_i])
                    else:
                        act_pts = act_fgm = act_fga = act_ast = 0.0
                    pred_fgm = float(box[s, fgm_i])
                    pred_fga = float(box[s, fga_i])
                    pred_ast = float(box[s, ast_i])
                    rows.append((
                        pred_exp, pid, prob[s], pred_exp, pred_pts, pred_a_off,
                        act_exp, act_pts, float(actual_alpha[s]),
                        pred_fgm, pred_fga, pred_ast,
                        act_fgm, act_fga, act_ast,
                    ))
                rows.sort(key=lambda r: r[0], reverse=True)
                top = rows[: top_k]
                pred_total_exp = sum(r[3] for r in rows)
                act_total_exp = sum(r[6] for r in rows)
                pred_total_pts = sum(r[4] for r in rows)
                act_total_pts = sum(r[7] for r in rows)
                pred_total_a = sum(r[5] for r in rows)
                act_total_a = sum(r[8] for r in rows)
                print(f"  {side_name} L={int(L)}  "
                      f"sum_pred[exp={pred_total_exp:.1f} pts={pred_total_pts:.1f} a_off={pred_total_a:.3f}]  "
                      f"sum_actual[exp={act_total_exp:.0f} pts={act_total_pts:.0f} a_off={act_total_a:.3f}]")
                print(f"    {'pid':>10} {'prob':>5} | {'pE':>5} {'aE':>5} {'pPTS':>5} {'aPTS':>5} "
                      f"{'pAoff':>6} {'aAoff':>6} | {'pFGM':>5} {'aFGM':>5} {'pFGA':>5} {'aFGA':>5} "
                      f"{'pAST':>5} {'aAST':>5}")
                for r in top:
                    (_, pid, pr, pred_exp, pred_pts, pred_a_off,
                     act_exp, act_pts, act_a,
                     pred_fgm, pred_fga, pred_ast,
                     act_fgm, act_fga, act_ast) = r
                    pid_s = pid if isinstance(pid, str) else str(pid)
                    print(f"    {pid_s:>10} {pr:5.2f} | "
                          f"{pred_exp:5.1f} {act_exp:5.0f} {pred_pts:5.1f} {act_pts:5.0f} "
                          f"{pred_a_off:6.3f} {act_a:6.3f} | "
                          f"{pred_fgm:5.1f} {act_fgm:5.0f} {pred_fga:5.1f} {act_fga:5.0f} "
                          f"{pred_ast:5.1f} {act_ast:5.0f}")


def _track_player_predictions(
    model, test_recs, vocab, device: str,
    target_pids: list[str], batch_size: int = 64,
) -> None:
    """For every test game containing each target_pid, print pred vs actual.

    One line per (game, pid). Useful for tracking a single star's
    predicted variance across opponents.
    """
    if not test_recs or not target_pids:
        return
    pid_to_idx = {pid: vocab.player_to_idx.get(pid) for pid in target_pids}
    missing = [pid for pid, i in pid_to_idx.items() if i is None]
    if missing:
        print(f"[track] pids not in vocab: {missing}")
    pts_i = BOX_INDEX["pts"]
    fgm_i = BOX_INDEX["fgm"]
    fga_i = BOX_INDEX["fga"]
    ast_i = BOX_INDEX["ast"]

    loader = DataLoader(GameDatasetRatingsV2(test_recs), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_ratings_v2)
    model.eval()
    header_printed: set[str] = set()
    rec_iter = iter(test_recs)
    with torch.no_grad():
        for batch_cpu in loader:
            batch = {k: v.to(device) for k, v in batch_cpu.items()}
            fwd = model(batch)
            B = batch_cpu["label"].shape[0]
            recs_b = [next(rec_iter) for _ in range(B)]
            home_idx = batch_cpu["home_idx"].cpu().numpy()
            away_idx = batch_cpu["away_idx"].cpu().numpy()
            home_mask = batch_cpu["home_mask"].cpu().numpy().astype(bool)
            away_mask = batch_cpu["away_mask"].cpu().numpy().astype(bool)
            home_pair_marg = fwd["home_pair_marg"].cpu().numpy()
            away_pair_marg = fwd["away_pair_marg"].cpu().numpy()
            home_box = fwd["home_box"].cpu().numpy()
            away_box = fwd["away_box"].cpu().numpy()
            alpha_h = fwd["alpha_home_off"].cpu().numpy()
            alpha_a = fwd["alpha_away_off"].cpu().numpy()
            alpha_h_act = batch_cpu["home_alpha_off_actual"].cpu().numpy()
            alpha_a_act = batch_cpu["away_alpha_off_actual"].cpu().numpy()
            sup_pair_game = batch_cpu["sup_pair_game"].cpu().numpy()
            sup_pair_side = batch_cpu["sup_pair_side"].cpu().numpy()
            sup_pair_off = batch_cpu["sup_pair_off"].cpu().numpy()
            sup_pair_y = batch_cpu["sup_pair_y"].cpu().numpy()
            sup_pl_game = batch_cpu["sup_pl_game"].cpu().numpy()
            sup_pl_side = batch_cpu["sup_pl_side"].cpu().numpy()
            sup_pl_slot = batch_cpu["sup_pl_slot"].cpu().numpy()
            sup_pl_y = batch_cpu["sup_pl_y"].cpu().numpy()

            for pid, target_idx in pid_to_idx.items():
                if target_idx is None:
                    continue
                if pid not in header_printed:
                    print(f"\n[track pid={pid}]  one row per game; "
                          f"opp = opposing team_id; pred totals = team pts; "
                          f"pE/pPTS/pAoff = predicted; aE/aPTS/aAoff = actual")
                    print(f"    {'date':>10} {'gid':>10} {'side':>4} {'opp':>11} "
                          f"{'pE':>5} {'aE':>5} {'pPTS':>5} {'aPTS':>5} "
                          f"{'pFGM':>5} {'aFGM':>5} {'pFGA':>5} {'aFGA':>5} "
                          f"{'pAST':>5} {'aAST':>5} "
                          f"{'pAoff':>6} {'aAoff':>6}")
                    header_printed.add(pid)
                for b in range(B):
                    rec = recs_b[b]
                    side = -1
                    slot = -1
                    h_slots = np.flatnonzero((home_idx[b] == target_idx) & home_mask[b])
                    a_slots = np.flatnonzero((away_idx[b] == target_idx) & away_mask[b])
                    if h_slots.size > 0:
                        side, slot = 0, int(h_slots[0])
                        opp = rec.away_team_id
                    elif a_slots.size > 0:
                        side, slot = 1, int(a_slots[0])
                        opp = rec.home_team_id
                    else:
                        continue
                    if side == 0:
                        pred_exp = float(home_pair_marg[b, slot, 0])
                        pred_pts = float(home_box[b, slot, pts_i])
                        pred_fgm = float(home_box[b, slot, fgm_i])
                        pred_fga = float(home_box[b, slot, fga_i])
                        pred_ast = float(home_box[b, slot, ast_i])
                        pred_a_off = float(alpha_h[b, slot])
                        act_a_off = float(alpha_h_act[b, slot])
                    else:
                        pred_exp = float(away_pair_marg[b, slot, 0])
                        pred_pts = float(away_box[b, slot, pts_i])
                        pred_fgm = float(away_box[b, slot, fgm_i])
                        pred_fga = float(away_box[b, slot, fga_i])
                        pred_ast = float(away_box[b, slot, ast_i])
                        pred_a_off = float(alpha_a[b, slot])
                        act_a_off = float(alpha_a_act[b, slot])
                    m_pair = ((sup_pair_game == b) & (sup_pair_side == side)
                              & (sup_pair_off == slot))
                    act_exp = float(sup_pair_y[m_pair, 0].sum()) if m_pair.any() else 0.0
                    m_pl = ((sup_pl_game == b) & (sup_pl_side == side)
                            & (sup_pl_slot == slot))
                    if m_pl.any():
                        i_row = int(np.flatnonzero(m_pl)[0])
                        act_pts = float(sup_pl_y[i_row, pts_i])
                        act_fgm = float(sup_pl_y[i_row, fgm_i])
                        act_fga = float(sup_pl_y[i_row, fga_i])
                        act_ast = float(sup_pl_y[i_row, ast_i])
                    else:
                        act_pts = act_fgm = act_fga = act_ast = 0.0
                    side_label = "HOME" if side == 0 else "AWAY"
                    print(f"    {rec.game_date.date()!s:>10} {rec.game_id:>10} "
                          f"{side_label:>4} {opp:>11} "
                          f"{pred_exp:5.1f} {act_exp:5.0f} {pred_pts:5.1f} {act_pts:5.0f} "
                          f"{pred_fgm:5.1f} {act_fgm:5.0f} {pred_fga:5.1f} {act_fga:5.0f} "
                          f"{pred_ast:5.1f} {act_ast:5.0f} "
                          f"{pred_a_off:6.3f} {act_a_off:6.3f}")


def train_one_window(
    args: argparse.Namespace,
    train_fit_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    histories, scores, statuses, calibration, game_odds, rating_priors,
    matchup_db: Path, core_db: Path,
    ckpt_dir: Path | None = None,
) -> tuple:
    """Train one window from scratch; return trained model and test predictions."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_gids = [str(g) for g in train_fit_df["game_id"].tolist()]
    val_gids = [str(g) for g in val_df["game_id"].tolist()]
    test_gids = [str(g) for g in test_df["game_id"].tolist()]

    train_matchup = load_matchup_rows_v2(matchup_db, train_gids)
    val_matchup = load_matchup_rows_v2(matchup_db, val_gids)
    test_matchup = load_matchup_rows_v2(matchup_db, test_gids)
    train_pl = load_player_game_stats(core_db, train_gids)
    val_pl = load_player_game_stats(core_db, val_gids)
    test_pl = load_player_game_stats(core_db, test_gids)

    vocab = build_vocab_from_records_v42(
        train_fit_df, histories, train_matchup,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(train_fit_df)
    tabular_stats = fit_tabular_stats(train_fit_df)
    rating_feature_stats = fit_rating_feature_stats(train_fit_df, rating_priors)

    common = dict(
        histories=histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=None, player_form_stats=None,
        player_form_lookback=DEFAULT_PLAYER_FORM_LOOKBACK,
        player_form_decay=DEFAULT_PLAYER_FORM_DECAY,
        game_odds=game_odds,
    )
    train_recs = build_records_ratings_v2(
        train_fit_df, matchup_rows=train_matchup, player_game_stats=train_pl,
        rating_lookup=rating_priors, rating_feature_stats=rating_feature_stats, **common,
    )
    val_recs = build_records_ratings_v2(
        val_df, matchup_rows=val_matchup, player_game_stats=val_pl,
        rating_lookup=rating_priors, rating_feature_stats=rating_feature_stats, **common,
    )
    test_recs = build_records_ratings_v2(
        test_df, matchup_rows=test_matchup, player_game_stats=test_pl,
        rating_lookup=rating_priors, rating_feature_stats=rating_feature_stats, **common,
    )

    train_loader = DataLoader(GameDatasetRatingsV2(train_recs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_ratings_v2)
    val_loader = DataLoader(GameDatasetRatingsV2(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_ratings_v2)

    import cme_v4_2_common as _cv4c
    tabular_dim = 0 if args.no_tabular else len(_cv4c.TABULAR_FEATURE_COLUMNS)
    cfg = CmeRatingsV2Config(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, n_heads=args.n_heads,
        n_self_layers=args.n_self_layers, n_cross_layers=args.n_cross_layers,
        pair_hidden=args.pair_hidden, player_hidden=args.player_hidden,
        inv_hidden=args.inv_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        tabular_dim=tabular_dim, team_emb_dim=args.team_emb_dim,
        player_stat_dim=0,
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
        rating_feature_dim=K_RATING_FEATURES,
        cme_residual_hidden=args.cme_residual_hidden,
        cme_residual_dropout=args.cme_residual_dropout,
        rating_margin_scale=args.rating_margin_scale,
        init_rating_logit_bias=args.init_rating_logit_bias,
        init_rating_logit_slope=args.init_rating_logit_slope,
        max_cme_gap_weight=args.max_cme_gap_weight,
        init_cme_gap_weight=args.init_cme_gap_weight,
        max_cme_feature_residual=args.max_cme_feature_residual,
        disagreement_gate_center=args.disagreement_gate_center,
        disagreement_gate_rating_confidence_w=args.disagreement_gate_rating_confidence_w,
        disagreement_gate_gap_w=args.disagreement_gate_gap_w,
    )
    model = CmeRatingsV2(cfg).to(args.device)

    box_weights = (torch.tensor(args.box_weights, dtype=torch.float32)
                   if args.box_weights is not None else default_box_weights())
    pair_weights = (torch.tensor(args.pair_weights, dtype=torch.float32)
                    if args.pair_weights is not None else default_pair_weights())
    if box_weights.numel() != K_BOX:
        raise ValueError(f"--box-weights must have length {K_BOX}")
    if pair_weights.numel() != K_PAIR:
        raise ValueError(f"--pair-weights must have length {K_PAIR}")

    history: list[dict] = []

    if args.curriculum:
        p1_task_weights = {
            "team": args.phase1_team_w, "player": args.phase1_player_w,
            "pair": args.phase1_pair_w, "inv": args.phase1_inv_w, "win": 0.0,
        }
        norms = _per_task_grad_norms(model, train_loader, args.device, box_weights, pair_weights)
        print("[grad-diag p1-init]    " + _fmt_grad_norms("", norms, p1_task_weights))

        # Phase 1: structural pretraining (no win BCE, no involvement loss).
        # Track val total loss for plateau early-stop.
        p1_optim = _build_optimizer(model, args)
        p1_scheduler = _build_lr_scheduler(p1_optim, args.warmup_epochs, args.phase1_epochs)
        p1_kwargs = dict(
            box_weights=box_weights, pair_weights=pair_weights,
            team_w=args.phase1_team_w, player_w=args.phase1_player_w,
            pair_w=args.phase1_pair_w,
            inv_w=args.phase1_inv_w, win_w=0.0, margin_nll_w=0.0,
            calibration_reg_w=args.calibration_reg_w,
            season_calibration_reg_w=args.season_calibration_reg_w,
            calibration_slope_reg_w=args.calibration_slope_reg_w,
            cme_feature_reg_w=args.cme_feature_reg_w,
            cme_gap_weight_reg_w=args.cme_gap_weight_reg_w,
            rating_bias_reg_w=args.rating_bias_reg_w,
        )
        p1_best_state, p1_best_epoch, p1_best_val, p1_history = _run_training_loop(
            model, train_loader, val_loader, p1_optim, p1_scheduler,
            p1_kwargs, args.phase1_epochs, args.phase1_patience,
            args.device, track_metric="loss", phase_label=1,
        )
        history.extend(p1_history)
        if p1_best_state is not None:
            model.load_state_dict(p1_best_state)

        norms = _per_task_grad_norms(model, train_loader, args.device, box_weights, pair_weights)
        print("[grad-diag p1-end]     " + _fmt_grad_norms("", norms, p1_task_weights))

        # Phase 2: cfg-D weights — BCE + exposure-only pair Poisson.
        ckpt_pair_weights = torch.zeros(K_PAIR)
        ckpt_pair_weights[0] = 1.0
        ckpt_box_weights = box_weights
        ckpt_team_w = 0.0
        ckpt_player_w = 0.0
        ckpt_pair_w = 1.0
        ckpt_inv_w = 0.0
        ckpt_win_w = 1.0
        ckpt_margin_nll_w = 0.0
        ckpt_calibration_reg_w = args.calibration_reg_w
        ckpt_season_calibration_reg_w = args.season_calibration_reg_w
        ckpt_calibration_slope_reg_w = args.calibration_slope_reg_w
        ckpt_cme_feature_reg_w = args.cme_feature_reg_w
        ckpt_cme_gap_weight_reg_w = args.cme_gap_weight_reg_w

        p2_optim = _build_optimizer(model, args)
        p2_scheduler = _build_lr_scheduler(p2_optim, args.warmup_epochs, args.epochs)
        p2_kwargs = dict(
            box_weights=ckpt_box_weights, pair_weights=ckpt_pair_weights,
            team_w=ckpt_team_w, player_w=ckpt_player_w, pair_w=ckpt_pair_w,
            inv_w=ckpt_inv_w, win_w=ckpt_win_w, margin_nll_w=ckpt_margin_nll_w,
            calibration_reg_w=ckpt_calibration_reg_w,
            season_calibration_reg_w=ckpt_season_calibration_reg_w,
            calibration_slope_reg_w=ckpt_calibration_slope_reg_w,
            cme_feature_reg_w=args.cme_feature_reg_w,
            cme_gap_weight_reg_w=args.cme_gap_weight_reg_w,
            rating_bias_reg_w=args.rating_bias_reg_w,
        )
        best_state, best_epoch, best_val, p2_history = _run_training_loop(
            model, train_loader, val_loader, p2_optim, p2_scheduler,
            p2_kwargs, args.epochs, args.patience,
            args.device, track_metric="bce", phase_label=2,
        )
        history.extend(p2_history)
        last_epoch = p2_history[-1]["epoch"] if p2_history else 0

        p2_task_weights = {
            "team": ckpt_team_w, "player": ckpt_player_w,
            "pair": ckpt_pair_w, "inv": ckpt_inv_w, "win": ckpt_win_w,
        }
        norms = _per_task_grad_norms(model, train_loader, args.device,
                                     ckpt_box_weights, ckpt_pair_weights)
        print("[grad-diag p2-end]     " + _fmt_grad_norms("", norms, p2_task_weights))
    else:
        optim = _build_optimizer(model, args)
        scheduler = _build_lr_scheduler(optim, args.warmup_epochs, args.epochs)
        epoch_kwargs = dict(
            box_weights=box_weights, pair_weights=pair_weights,
            team_w=args.team_w, player_w=args.player_w, pair_w=args.pair_w,
            inv_w=args.inv_w, win_w=args.win_w, margin_nll_w=args.margin_nll_w,
            calibration_reg_w=args.calibration_reg_w,
            season_calibration_reg_w=args.season_calibration_reg_w,
            calibration_slope_reg_w=args.calibration_slope_reg_w,
            cme_feature_reg_w=args.cme_feature_reg_w,
            cme_gap_weight_reg_w=args.cme_gap_weight_reg_w,
            rating_bias_reg_w=args.rating_bias_reg_w,
        )
        best_state, best_epoch, best_val, single_history = _run_training_loop(
            model, train_loader, val_loader, optim, scheduler,
            epoch_kwargs, args.epochs, args.patience,
            args.device, track_metric="bce", phase_label=0,
        )
        history.extend(single_history)
        last_epoch = single_history[-1]["epoch"] if single_history else 0

        ckpt_box_weights = box_weights
        ckpt_pair_weights = pair_weights
        ckpt_team_w = args.team_w
        ckpt_player_w = args.player_w
        ckpt_pair_w = args.pair_w
        ckpt_inv_w = args.inv_w
        ckpt_win_w = args.win_w
        ckpt_margin_nll_w = args.margin_nll_w
        ckpt_calibration_reg_w = args.calibration_reg_w
        ckpt_season_calibration_reg_w = args.season_calibration_reg_w
        ckpt_calibration_slope_reg_w = args.calibration_slope_reg_w
        ckpt_cme_feature_reg_w = args.cme_feature_reg_w
        ckpt_cme_gap_weight_reg_w = args.cme_gap_weight_reg_w

    if ckpt_dir is not None and args.save_final:
        final_payload = _build_ckpt_payload(
            model, cfg, vocab, team_vocab, ckpt_box_weights, ckpt_pair_weights,
            ckpt_team_w, ckpt_player_w, ckpt_pair_w, ckpt_inv_w, ckpt_win_w,
            ckpt_margin_nll_w, ckpt_calibration_reg_w, ckpt_season_calibration_reg_w, ckpt_calibration_slope_reg_w,
            ckpt_cme_feature_reg_w, ckpt_cme_gap_weight_reg_w, args.rating_bias_reg_w,
            rating_feature_stats, last_epoch,
        )
    else:
        final_payload = None

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.inspect_players:
        _inspect_player_predictions(
            model, test_recs, vocab, args.device,
            n_games=2, top_k=args.inspect_top_k,
        )
    if args.track_pids:
        pids = [p.strip() for p in args.track_pids.split(",") if p.strip()]
        _track_player_predictions(
            model, test_recs, vocab, args.device, pids,
            batch_size=args.batch_size,
        )

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            _build_ckpt_payload(
                model, cfg, vocab, team_vocab, ckpt_box_weights, ckpt_pair_weights,
                ckpt_team_w, ckpt_player_w, ckpt_pair_w, ckpt_inv_w, ckpt_win_w,
                ckpt_margin_nll_w, ckpt_calibration_reg_w, ckpt_season_calibration_reg_w, ckpt_calibration_slope_reg_w,
                ckpt_cme_feature_reg_w, ckpt_cme_gap_weight_reg_w, args.rating_bias_reg_w,
                rating_feature_stats, best_epoch,
            ),
            ckpt_dir / "best.pt",
        )
        if final_payload is not None:
            torch.save(final_payload, ckpt_dir / "final.pt")

    (gids, probs, base_probs, affine_probs, temporal_probs, season_logit_adjustments,
     residual_tail_gates, win_residuals, rating_margin_priors, rating_logit_priors,
     rating_probs_raw, rating_logits_calibrated, rating_probs_calibrated,
     rating_logit_biases, rating_logit_slopes, cme_gaps, disagreement_gates,
     cme_gap_weights, cme_gap_linear_residuals, cme_feature_residuals,
     home_logit_biases, win_logit_slopes, h_pts, a_pts, margin_mu) = predict_window(
        model, test_recs, args.batch_size, args.device,
    )

    window_info = {
        "n_train_fit": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "vocab_size": vocab.size,
        "best_epoch": best_epoch,
        "best_val_bce": best_val,
        "epochs_run": len(history),
        "history": history,
    }
    return (model, window_info, gids, probs, base_probs, affine_probs, temporal_probs,
            season_logit_adjustments, residual_tail_gates, win_residuals,
            rating_margin_priors, rating_logit_priors,
            rating_probs_raw, rating_logits_calibrated, rating_probs_calibrated,
            rating_logit_biases, rating_logit_slopes, cme_gaps, disagreement_gates,
            cme_gap_weights, cme_gap_linear_residuals, cme_feature_residuals,
            home_logit_biases, win_logit_slopes, h_pts, a_pts, margin_mu)


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_cyclic:
        import cme_v4_2_common as _cv4c
        import cme_v2_common as _cv2c
        trimmed = tuple(c for c in _cv2c.TABULAR_FEATURE_COLUMNS if not c.startswith("cyc_"))
        _cv2c.TABULAR_FEATURE_COLUMNS = trimmed
        _cv4c.TABULAR_FEATURE_COLUMNS = trimmed
        print(f"[no-cyclic] tabular columns trimmed to {len(trimmed)}")

    print(f"[device] {args.device}")
    print(f"[output] {out_dir}")

    print("[load] games + exposures + odds + scores + statuses + calibration")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    all_gids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, all_gids)
    statuses = load_game_player_status(args.injury_db, all_gids)
    calibration = load_status_calibration(args.calibration)
    game_odds = load_game_odds(args.core_db, all_gids)
    rating_priors = load_rating_priors(args.ratings_db, all_gids)
    print(f"[load] n_games={len(games_all)} odds={len(game_odds)} ratings={len(rating_priors)}")

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
        ckpt_dir = (out_dir / "windows" / window_start.date().isoformat()
                    if args.save_checkpoints else None)

        try:
            (_model, info, gids, probs, base_probs, affine_probs, temporal_probs,
             season_logit_adjustments, residual_tail_gates, win_residuals,
             rating_margin_priors, rating_logit_priors,
             rating_probs_raw, rating_logits_calibrated, rating_probs_calibrated,
             rating_logit_biases, rating_logit_slopes, cme_gaps, disagreement_gates,
             cme_gap_weights, cme_gap_linear_residuals, cme_feature_residuals,
             home_logit_biases, win_logit_slopes, h_pts, a_pts, margin_mu) = train_one_window(
                args, train_fit, val, test_block, histories, scores, statuses,
                calibration, game_odds, rating_priors, args.matchup_db, args.core_db,
                ckpt_dir=ckpt_dir,
            )
        except Exception as exc:
            print(f"[ERROR] window {window_start.date()}: {exc}")
            raise
        dt = time.time() - t0

        gid_to_idx = {g: i for i, g in enumerate(gids)}
        order = [gid_to_idx[str(g)] for g in test_block["game_id"].astype(str).tolist()]
        probs_ordered = probs[order]
        base_probs_ordered = base_probs[order]
        affine_probs_ordered = affine_probs[order]
        temporal_probs_ordered = temporal_probs[order]
        season_logit_adjustments_ordered = season_logit_adjustments[order]
        residual_tail_gates_ordered = residual_tail_gates[order]
        win_residuals_ordered = win_residuals[order]
        rating_margin_priors_ordered = rating_margin_priors[order]
        rating_logit_priors_ordered = rating_logit_priors[order]
        rating_probs_raw_ordered = rating_probs_raw[order]
        rating_logits_calibrated_ordered = rating_logits_calibrated[order]
        rating_probs_calibrated_ordered = rating_probs_calibrated[order]
        rating_logit_biases_ordered = rating_logit_biases[order]
        rating_logit_slopes_ordered = rating_logit_slopes[order]
        cme_gaps_ordered = cme_gaps[order]
        disagreement_gates_ordered = disagreement_gates[order]
        cme_gap_weights_ordered = cme_gap_weights[order]
        cme_gap_linear_residuals_ordered = cme_gap_linear_residuals[order]
        cme_feature_residuals_ordered = cme_feature_residuals[order]
        home_logit_biases_ordered = home_logit_biases[order]
        win_logit_slopes_ordered = win_logit_slopes[order]
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
            "best_val_bce": info["best_val_bce"],
            "epochs_run": info["epochs_run"],
            "test_bce": bce,
            "test_acc": acc,
            "test_mean_prob": float(np.mean(probs_ordered)),
            "test_mean_base_prob": float(np.mean(base_probs_ordered)),
            "test_home_rate": float(np.mean(labels)),
            "test_mean_affine_prob": float(np.mean(affine_probs_ordered)),
            "test_mean_temporal_prob": float(np.mean(temporal_probs_ordered)),
            "test_mean_season_logit_adjustment": float(np.mean(season_logit_adjustments_ordered)),
            "test_mean_residual_tail_gate": float(np.mean(residual_tail_gates_ordered)),
            "test_mean_win_residual": float(np.mean(win_residuals_ordered)),
            "test_mean_rating_margin_prior": float(np.mean(rating_margin_priors_ordered)),
            "test_mean_rating_logit_prior": float(np.mean(rating_logit_priors_ordered)),
            "test_mean_rating_prob_raw": float(np.mean(rating_probs_raw_ordered)),
            "test_mean_rating_logit_calibrated": float(np.mean(rating_logits_calibrated_ordered)),
            "test_mean_rating_prob_calibrated": float(np.mean(rating_probs_calibrated_ordered)),
            "test_mean_rating_logit_bias": float(np.mean(rating_logit_biases_ordered)),
            "test_mean_rating_logit_slope": float(np.mean(rating_logit_slopes_ordered)),
            "test_mean_cme_gap": float(np.mean(cme_gaps_ordered)),
            "test_mean_disagreement_gate": float(np.mean(disagreement_gates_ordered)),
            "test_mean_cme_gap_weight": float(np.mean(cme_gap_weights_ordered)),
            "test_mean_cme_gap_linear_residual": float(np.mean(cme_gap_linear_residuals_ordered)),
            "test_mean_cme_feature_residual": float(np.mean(cme_feature_residuals_ordered)),
            "test_home_logit_bias": float(np.mean(home_logit_biases_ordered)),
            "test_win_logit_slope": float(np.mean(win_logit_slopes_ordered)),
            "secs": dt,
        }
        window_metrics.append(win_metrics)
        print(f"[window {wi+1}/{len(windows)}] best_ep={info['best_epoch']:2d} "
              f"val_bce={info['best_val_bce']:.4f} test_bce={bce:.4f} test_acc={acc:.3f} "
              f"p={np.mean(probs_ordered):.3f}/{np.mean(labels):.3f} "
              f"base={np.mean(base_probs_ordered):.3f} "
              f"aff={np.mean(affine_probs_ordered):.3f} temp={np.mean(temporal_probs_ordered):.3f} "
              f"ζ={np.mean(season_logit_adjustments_ordered):.3f} g={np.mean(residual_tail_gates_ordered):.3f} "
              f"δ={np.mean(win_residuals_ordered):.3f} rM={np.mean(rating_margin_priors_ordered):.2f} "
              f"rPc={np.mean(rating_probs_calibrated_ordered):.3f} "
              f"gate={np.mean(disagreement_gates_ordered):.3f} gw={np.mean(cme_gap_weights_ordered):.3f} "
              f"b={np.mean(home_logit_biases_ordered):.3f} s={np.mean(win_logit_slopes_ordered):.3f} "
              f"({dt:.0f}s)")

        pred_df = test_block[
            ["game_id", "game_date", "home_team_id", "away_team_id", "label_home_win"]
        ].copy()
        pred_df["pred_home_win_prob"] = probs_ordered
        pred_df["pred_home_win_prob_base"] = base_probs_ordered
        pred_df["pred_home_win_prob_affine"] = affine_probs_ordered
        pred_df["pred_home_win_prob_temporal"] = temporal_probs_ordered
        pred_df["season_logit_adjustment"] = season_logit_adjustments_ordered
        pred_df["residual_tail_gate"] = residual_tail_gates_ordered
        pred_df["win_calibration_residual"] = win_residuals_ordered
        pred_df["rating_margin_prior"] = rating_margin_priors_ordered
        pred_df["rating_logit_prior"] = rating_logit_priors_ordered
        pred_df["rating_home_win_prob_raw"] = rating_probs_raw_ordered
        pred_df["rating_logit_calibrated"] = rating_logits_calibrated_ordered
        pred_df["rating_home_win_prob_calibrated"] = rating_probs_calibrated_ordered
        pred_df["rating_logit_bias"] = rating_logit_biases_ordered
        pred_df["rating_logit_slope"] = rating_logit_slopes_ordered
        pred_df["cme_gap"] = cme_gaps_ordered
        pred_df["disagreement_gate"] = disagreement_gates_ordered
        pred_df["cme_gap_weight"] = cme_gap_weights_ordered
        pred_df["cme_gap_linear_residual"] = cme_gap_linear_residuals_ordered
        pred_df["cme_feature_residual"] = cme_feature_residuals_ordered
        pred_df["home_logit_bias"] = home_logit_biases_ordered
        pred_df["win_logit_slope"] = win_logit_slopes_ordered
        pred_df["pred_home_pts"] = h_pts_ordered
        pred_df["pred_away_pts"] = a_pts_ordered
        pred_df["margin_mu"] = margin_ordered
        pred_df["window_start"] = window_start.date().isoformat()
        all_predictions.append(pred_df)

        predictions_df_partial = pd.concat(all_predictions, ignore_index=True)
        predictions_df_partial.to_csv(out_dir / "predictions.csv", index=False)
        pd.DataFrame(window_metrics).to_csv(out_dir / "window_metrics.csv", index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    labels = predictions_df["label_home_win"].to_numpy(dtype=float)
    probs = predictions_df["pred_home_win_prob"].to_numpy(dtype=float)
    base_probs = predictions_df["pred_home_win_prob_base"].to_numpy(dtype=float)
    affine_probs = predictions_df["pred_home_win_prob_affine"].to_numpy(dtype=float)
    temporal_probs = predictions_df["pred_home_win_prob_temporal"].to_numpy(dtype=float)
    season_logit_adjustments = predictions_df["season_logit_adjustment"].to_numpy(dtype=float)
    residual_tail_gates = predictions_df["residual_tail_gate"].to_numpy(dtype=float)
    win_residuals = predictions_df["win_calibration_residual"].to_numpy(dtype=float)
    rating_margin_priors = predictions_df["rating_margin_prior"].to_numpy(dtype=float)
    rating_logit_priors = predictions_df["rating_logit_prior"].to_numpy(dtype=float)
    rating_probs_raw = predictions_df["rating_home_win_prob_raw"].to_numpy(dtype=float)
    rating_logits_calibrated = predictions_df["rating_logit_calibrated"].to_numpy(dtype=float)
    rating_probs_calibrated = predictions_df["rating_home_win_prob_calibrated"].to_numpy(dtype=float)
    rating_logit_biases = predictions_df["rating_logit_bias"].to_numpy(dtype=float)
    rating_logit_slopes = predictions_df["rating_logit_slope"].to_numpy(dtype=float)
    cme_gaps = predictions_df["cme_gap"].to_numpy(dtype=float)
    disagreement_gates = predictions_df["disagreement_gate"].to_numpy(dtype=float)
    cme_gap_weights = predictions_df["cme_gap_weight"].to_numpy(dtype=float)
    cme_gap_linear_residuals = predictions_df["cme_gap_linear_residual"].to_numpy(dtype=float)
    cme_feature_residuals = predictions_df["cme_feature_residual"].to_numpy(dtype=float)
    home_logit_biases = predictions_df["home_logit_bias"].to_numpy(dtype=float)
    win_logit_slopes = predictions_df["win_logit_slope"].to_numpy(dtype=float)
    overall_bce = float(-np.mean(labels * np.log(np.clip(probs, 1e-7, 1-1e-7))
                                  + (1 - labels) * np.log(np.clip(1 - probs, 1e-7, 1-1e-7))))
    overall_acc = float(np.mean((probs > 0.5).astype(float) == labels))
    overall_brier = float(np.mean((probs - labels) ** 2))
    rating_raw_bce = float(-np.mean(labels * np.log(np.clip(rating_probs_raw, 1e-7, 1-1e-7))
                                      + (1 - labels) * np.log(np.clip(1 - rating_probs_raw, 1e-7, 1-1e-7))))
    rating_raw_acc = float(np.mean((rating_probs_raw > 0.5).astype(float) == labels))
    rating_raw_brier = float(np.mean((rating_probs_raw - labels) ** 2))
    rating_cal_bce = float(-np.mean(labels * np.log(np.clip(rating_probs_calibrated, 1e-7, 1-1e-7))
                                      + (1 - labels) * np.log(np.clip(1 - rating_probs_calibrated, 1e-7, 1-1e-7))))
    rating_cal_acc = float(np.mean((rating_probs_calibrated > 0.5).astype(float) == labels))
    rating_cal_brier = float(np.mean((rating_probs_calibrated - labels) ** 2))

    overall = {
        "n": int(len(predictions_df)),
        "bce": overall_bce,
        "acc": overall_acc,
        "brier": overall_brier,
        "rating_raw_bce": rating_raw_bce,
        "rating_raw_acc": rating_raw_acc,
        "rating_raw_brier": rating_raw_brier,
        "rating_calibrated_bce": rating_cal_bce,
        "rating_calibrated_acc": rating_cal_acc,
        "rating_calibrated_brier": rating_cal_brier,
        "mean_prob": float(probs.mean()),
        "std_prob": float(probs.std()),
        "mean_base_prob": float(base_probs.mean()),
        "std_base_prob": float(base_probs.std()),
        "mean_affine_prob": float(affine_probs.mean()),
        "mean_temporal_prob": float(temporal_probs.mean()),
        "mean_season_logit_adjustment": float(season_logit_adjustments.mean()),
        "mean_residual_tail_gate": float(residual_tail_gates.mean()),
        "mean_win_residual": float(win_residuals.mean()),
        "mean_rating_margin_prior": float(rating_margin_priors.mean()),
        "mean_rating_logit_prior": float(rating_logit_priors.mean()),
        "mean_rating_home_win_prob_raw": float(rating_probs_raw.mean()),
        "mean_rating_logit_calibrated": float(rating_logits_calibrated.mean()),
        "mean_rating_home_win_prob_calibrated": float(rating_probs_calibrated.mean()),
        "mean_rating_logit_bias": float(rating_logit_biases.mean()),
        "mean_rating_logit_slope": float(rating_logit_slopes.mean()),
        "mean_cme_gap": float(cme_gaps.mean()),
        "mean_disagreement_gate": float(disagreement_gates.mean()),
        "mean_cme_gap_weight": float(cme_gap_weights.mean()),
        "mean_cme_gap_linear_residual": float(cme_gap_linear_residuals.mean()),
        "mean_cme_feature_residual": float(cme_feature_residuals.mean()),
        "mean_home_logit_bias": float(home_logit_biases.mean()),
        "mean_win_logit_slope": float(win_logit_slopes.mean()),
        "actual_home_win_rate": float(labels.mean()),
    }

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    predictions_df.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(window_metrics).to_csv(out_dir / "window_metrics.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2) + "\n")

    print("\n[done] overall:")
    print(json.dumps(overall, indent=2))
    print(f"[done] artifacts: {out_dir}")


if __name__ == "__main__":
    main()
