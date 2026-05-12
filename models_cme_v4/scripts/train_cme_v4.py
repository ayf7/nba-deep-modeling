#!/usr/bin/env python3
"""Train CME-v4 (hybrid Sinkhorn + K-channel + involvement-supervised).

Six-level loss:

    L = team_w * L_team_mse + player_w * L_player_mse + pair_w * L_pair_pois
      + inv_w * L_involvement_mse + win_w * L_win_bce + margin_nll_w * L_margin_nll
      + calibration_reg_w * L_calibration_reg

The new L_involvement_mse term anchors the model's predicted involvement
softmax (alpha^off, alpha^def per team) to the observed exposure-share
derived from matchup_training_rows.exposure_seconds. This is the
diagnostic + structural anchor that lets counterfactual interventions
(set a player's prob_play to zero) produce coherent involvement
redistribution before downstream box-score gradients have to "find"
the right shares purely through indirect signal.

Margin / win accuracy / margin MAE are reported as diagnostics.
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
V4_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V4_SCRIPTS))

from cme_v4_common import (  # noqa: E402
    BOX_INDEX, BOX_TARGETS, K_BOX, K_PAIR, PAIR_TARGETS,
    DEFAULT_CALIBRATION_PATH, DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB, DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    PLAYER_FORM_DIM, TABULAR_FEATURE_COLUMNS,
    GameDatasetV4, build_records_v4, build_team_vocab,
    build_vocab_from_records_v4, collate_v4,
    fit_player_form_stats, fit_tabular_stats,
    load_game_odds, load_game_player_status, load_game_scores, load_games,
    load_matchup_rows_v2, load_player_game_stats, load_player_histories,
    load_status_calibration, load_team_exposures,
)
from cme_v4_model import CmeV4, CmeV4Config, total_loss  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v4" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="run")
    # data
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    # model
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
                   help="Hidden size for the v4 win-logit calibration residual head.")
    p.add_argument("--calibration-dropout", type=float, default=0.05,
                   help="Dropout inside the v4 win-logit calibration residual head.")
    p.add_argument("--max-calibration-residual", type=float, default=1.0,
                   help="Absolute tanh-bounded context residual added to the base margin logit.")
    p.add_argument("--init-home-logit-bias", type=float, default=0.0,
                   help="Initialization for the learned home-side calibration intercept.")
    p.add_argument("--trainable-global-scale", action="store_true",
                   help="Allow the margin-to-win logistic scale to move during training. Default keeps v3's fixed scale.")
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--base-possessions", type=float, default=491.0)
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--use-player-stats", action="store_true")
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)
    # loss weights
    p.add_argument("--box-weights", type=float, nargs="+", default=None,
                   help=f"Per-stat weights for team+player levels (length {K_BOX}). "
                        f"Default = 1.0 for all, 3.0 for pts.")
    p.add_argument("--pair-weights", type=float, nargs="+", default=None,
                   help=f"Per-target weights for pair Poisson NLL (length {K_PAIR}, "
                        f"default = 1.0 for exposure/points, 0.5 for the rest)")
    p.add_argument("--team-w", type=float, default=1.0)
    p.add_argument("--player-w", type=float, default=0.01)
    p.add_argument("--pair-w", type=float, default=0.001)
    p.add_argument("--inv-w", type=float, default=5.0,
                   help="Weight on L_involvement_mse. Anchors softmax shares to "
                        "observed exposure_seconds shares.")
    p.add_argument("--win-w", type=float, default=10.0)
    p.add_argument("--margin-nll-w", type=float, default=0.0)
    p.add_argument("--calibration-reg-w", type=float, default=0.01,
                   help="L2 penalty weight on v4's context-specific win-logit residual; the global home intercept is not penalized.")
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
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
    w[PAIR_TARGETS.index("player_points")] = 1.0
    return w


def run_epoch(
    model: CmeV4, loader: DataLoader, *, device: str,
    optim: torch.optim.Optimizer | None,
    box_weights: torch.Tensor, pair_weights: torch.Tensor,
    team_w: float, player_w: float, pair_w: float,
    inv_w: float, win_w: float, margin_nll_w: float = 0.0,
    calibration_reg_w: float = 0.01,
) -> dict:
    is_train = optim is not None
    model.train(is_train)

    total_n = 0
    sum_loss = 0.0
    sum_L_team = 0.0
    sum_L_player = 0.0
    sum_L_pair = 0.0
    sum_L_inv = 0.0
    sum_L_win = 0.0
    sum_L_margin_nll = 0.0
    sum_L_calibration_reg = 0.0
    sum_team_mse_per_target = torch.zeros(K_BOX)
    sum_player_mse_per_target = torch.zeros(K_BOX)
    sum_pair_nll_per_target = torch.zeros(K_PAIR)
    sum_bce = 0.0
    sum_margin_abs = 0.0
    sum_log_sigma = 0.0
    sum_pts_h = 0.0
    sum_pts_a = 0.0
    sum_N = 0.0
    sum_prob = 0.0
    sum_base_prob = 0.0
    sum_label = 0.0
    sum_win_residual = 0.0
    sum_home_logit_bias = 0.0
    sum_alpha_off_corr = 0.0
    sum_alpha_off_corr_n = 0
    correct = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            out = model(batch)
            loss, diag = total_loss(
                out, batch,
                box_weights=box_weights.to(device),
                pair_weights=pair_weights.to(device),
                team_w=team_w, player_w=player_w, pair_w=pair_w,
                inv_w=inv_w, win_w=win_w, margin_nll_w=margin_nll_w,
                calibration_reg_w=calibration_reg_w,
            )

            if is_train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

            with torch.no_grad():
                bce = F.binary_cross_entropy_with_logits(out["win_logit"], batch["label"])
                margin_abs = (out["margin_mu"] - batch["margin"]).abs().mean()
                probs = torch.sigmoid(out["win_logit"])
                base_probs = torch.sigmoid(out["win_logit_base"])
                preds = (probs > 0.5).float()
                correct_bs = (preds == batch["label"]).sum().item()
                # Spearman-ish diagnostic: how well does predicted alpha_off match actual
                # (only where home_off_exposure_valid).
                valid = batch["home_off_exposure_valid"].bool()
                if valid.any():
                    p_alpha = out["alpha_home_off"][valid]
                    t_alpha = batch["home_alpha_off_actual"][valid]
                    m = batch["home_mask"][valid].to(p_alpha.dtype)
                    pa = (p_alpha * m).flatten()
                    ta = (t_alpha * m).flatten()
                    if pa.numel() > 1:
                        pa_c = pa - pa.mean()
                        ta_c = ta - ta.mean()
                        denom = (pa_c.pow(2).sum().sqrt() * ta_c.pow(2).sum().sqrt()).clamp_min(1e-8)
                        corr = (pa_c * ta_c).sum() / denom
                        sum_alpha_off_corr += corr.item() * int(valid.sum().item())
                        sum_alpha_off_corr_n += int(valid.sum().item())

        bs = batch["label"].size(0)
        total_n += bs
        sum_loss += loss.item() * bs
        sum_L_team += diag["L_team"].item() * bs
        sum_L_player += diag["L_player"].item() * bs
        sum_L_pair += diag["L_pair"].item() * bs
        sum_L_inv += diag["L_inv"].item() * bs
        sum_L_win += diag["L_win"].item() * bs
        sum_L_margin_nll += diag["L_margin_nll"].item() * bs
        sum_L_calibration_reg += diag["L_calibration_reg"].item() * bs
        sum_team_mse_per_target += diag["team_mse_per_target"].cpu() * bs
        sum_player_mse_per_target += diag["player_mse_per_target"].cpu() * bs
        sum_pair_nll_per_target += diag["pair_nll_per_target"].cpu() * bs
        sum_bce += bce.item() * bs
        sum_margin_abs += margin_abs.item() * bs
        sum_log_sigma += out["margin_log_sigma"].mean().item() * bs
        sum_pts_h += out["home_points"].mean().item() * bs
        sum_pts_a += out["away_points"].mean().item() * bs
        sum_N += out["N"].mean().item() * bs
        sum_prob += probs.mean().item() * bs
        sum_base_prob += base_probs.mean().item() * bs
        sum_label += batch["label"].mean().item() * bs
        sum_win_residual += out["win_residual"].mean().item() * bs
        sum_home_logit_bias += out["home_logit_bias"].detach().item() * bs
        correct += correct_bs

    return {
        "n": total_n,
        "loss": sum_loss / total_n,
        "L_team": sum_L_team / total_n,
        "L_player": sum_L_player / total_n,
        "L_pair": sum_L_pair / total_n,
        "L_inv": sum_L_inv / total_n,
        "L_win": sum_L_win / total_n,
        "L_margin_nll": sum_L_margin_nll / total_n,
        "L_calibration_reg": sum_L_calibration_reg / total_n,
        "team_mse_per_target": (sum_team_mse_per_target / total_n).tolist(),
        "player_mse_per_target": (sum_player_mse_per_target / total_n).tolist(),
        "pair_nll_per_target": (sum_pair_nll_per_target / total_n).tolist(),
        "bce": sum_bce / total_n,
        "margin_mae": sum_margin_abs / total_n,
        "mean_log_sigma": sum_log_sigma / total_n,
        "acc": correct / total_n,
        "mean_home_pts": sum_pts_h / total_n,
        "mean_away_pts": sum_pts_a / total_n,
        "mean_N": sum_N / total_n,
        "mean_prob": sum_prob / total_n,
        "mean_base_prob": sum_base_prob / total_n,
        "mean_label": sum_label / total_n,
        "mean_win_residual": sum_win_residual / total_n,
        "home_logit_bias": sum_home_logit_bias / total_n,
        "alpha_off_corr": (
            sum_alpha_off_corr / sum_alpha_off_corr_n if sum_alpha_off_corr_n > 0 else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[device] {args.device}")
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] games + exposures + matchup rows + status + calibration")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    if args.smoke:
        games_all = games_all.sample(n=min(400, len(games_all)), random_state=args.seed)
        games_all = games_all.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        args.epochs = 3

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

    print("[load] matchup rows (K_pair=9 targets)")
    train_matchup = load_matchup_rows_v2(args.matchup_db, train_gids)
    val_matchup = load_matchup_rows_v2(args.matchup_db, val_gids)
    test_matchup = load_matchup_rows_v2(args.matchup_db, test_gids)

    print("[load] per-player non-pair stats from game_events")
    train_pl = load_player_game_stats(args.core_db, train_gids)
    val_pl = load_player_game_stats(args.core_db, val_gids)
    test_pl = load_player_game_stats(args.core_db, test_gids)

    print("[vocab] building from train rosters + train matchup rows")
    vocab = build_vocab_from_records_v4(
        train_df, histories, train_matchup,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(train_df)
    print(f"[vocab] players={vocab.size} teams={team_vocab.size}")

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
    train_recs = build_records_v4(
        train_df, matchup_rows=train_matchup, player_game_stats=train_pl, **common,
    )
    val_recs = build_records_v4(
        val_df, matchup_rows=val_matchup, player_game_stats=val_pl, **common,
    )
    test_recs = build_records_v4(
        test_df, matchup_rows=test_matchup, player_game_stats=test_pl, **common,
    )
    print(f"[records] train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

    train_loader = DataLoader(GameDatasetV4(train_recs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_v4)
    val_loader = DataLoader(GameDatasetV4(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_v4)
    test_loader = DataLoader(GameDatasetV4(test_recs), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_v4)

    tabular_dim = 0 if args.no_tabular else len(TABULAR_FEATURE_COLUMNS)
    cfg = CmeV4Config(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, n_heads=args.n_heads,
        n_self_layers=args.n_self_layers, n_cross_layers=args.n_cross_layers,
        pair_hidden=args.pair_hidden, player_hidden=args.player_hidden,
        inv_hidden=args.inv_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        tabular_dim=tabular_dim, team_emb_dim=args.team_emb_dim,
        player_stat_dim=PLAYER_FORM_DIM if args.use_player_stats else 0,
        sinkhorn_iters=args.sinkhorn_iters,
        base_possessions_per_team=args.base_possessions,
        init_global_scale=args.init_scale,
        calibration_hidden=args.calibration_hidden,
        calibration_dropout=args.calibration_dropout,
        max_calibration_residual=args.max_calibration_residual,
        init_home_logit_bias=args.init_home_logit_bias,
        trainable_global_scale=args.trainable_global_scale,
    )
    model = CmeV4(cfg).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    box_weights = (torch.tensor(args.box_weights, dtype=torch.float32)
                   if args.box_weights is not None else default_box_weights())
    pair_weights = (torch.tensor(args.pair_weights, dtype=torch.float32)
                    if args.pair_weights is not None else default_pair_weights())
    if box_weights.numel() != K_BOX:
        raise ValueError(f"--box-weights must have length {K_BOX}")
    if pair_weights.numel() != K_PAIR:
        raise ValueError(f"--pair-weights must have length {K_PAIR}")

    def lr_lambda(epoch_idx: int) -> float:
        if args.warmup_epochs > 0 and epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / args.warmup_epochs
        denom = max(1, args.epochs - args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params:,} cfg={asdict(cfg)}")
    print(f"[loss] box_weights={box_weights.tolist()}")
    print(f"[loss] pair_weights={pair_weights.tolist()}")
    print(f"[loss] team_w={args.team_w} player_w={args.player_w} "
          f"pair_w={args.pair_w} inv_w={args.inv_w} "
          f"win_w={args.win_w} margin_nll_w={args.margin_nll_w} "
          f"calibration_reg_w={args.calibration_reg_w}")
    print(f"[optim] AdamW lr={args.lr} wd={args.weight_decay} warmup={args.warmup_epochs}ep")

    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict] = []
    ckpt_path = out_dir / "best.pt"

    def save_ckpt(path: Path, epoch: int) -> None:
        torch.save(
            {"model_state": model.state_dict(), "cfg": asdict(cfg),
             "vocab": vocab.player_to_idx,
             "team_vocab": team_vocab.team_to_idx,
             "player_form_means": (player_form_stats.means.tolist()
                                    if player_form_stats is not None else None),
             "player_form_stds": (player_form_stats.stds.tolist()
                                   if player_form_stats is not None else None),
             "box_weights": box_weights.tolist(),
             "pair_weights": pair_weights.tolist(),
             "loss_level_weights": {
                 "team": args.team_w, "player": args.player_w,
                 "pair": args.pair_w, "inv": args.inv_w,
                 "win": args.win_w, "margin_nll": args.margin_nll_w,
                 "calibration_reg": args.calibration_reg_w,
             },
             "epoch": epoch},
            path,
        )

    def epoch_kwargs() -> dict:
        return dict(
            box_weights=box_weights, pair_weights=pair_weights,
            team_w=args.team_w, player_w=args.player_w, pair_w=args.pair_w,
            inv_w=args.inv_w, win_w=args.win_w, margin_nll_w=args.margin_nll_w,
            calibration_reg_w=args.calibration_reg_w,
        )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device=args.device, optim=optim, **epoch_kwargs())
        va = run_epoch(model, val_loader, device=args.device, optim=None, **epoch_kwargs())
        scheduler.step()
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch, "lr": cur_lr, "secs": dt,
            "train_loss": tr["loss"], "val_loss": va["loss"],
            "train_L_team": tr["L_team"], "val_L_team": va["L_team"],
            "train_L_player": tr["L_player"], "val_L_player": va["L_player"],
            "train_L_pair": tr["L_pair"], "val_L_pair": va["L_pair"],
            "train_L_inv": tr["L_inv"], "val_L_inv": va["L_inv"],
            "train_L_win": tr["L_win"], "val_L_win": va["L_win"],
            "train_bce": tr["bce"], "val_bce": va["bce"],
            "train_acc": tr["acc"], "val_acc": va["acc"],
            "train_margin_mae": tr["margin_mae"], "val_margin_mae": va["margin_mae"],
            "train_mean_home_pts": tr["mean_home_pts"], "val_mean_home_pts": va["mean_home_pts"],
            "train_mean_away_pts": tr["mean_away_pts"], "val_mean_away_pts": va["mean_away_pts"],
            "train_mean_N": tr["mean_N"], "val_mean_N": va["mean_N"],
            "train_alpha_off_corr": tr["alpha_off_corr"],
            "val_alpha_off_corr": va["alpha_off_corr"],
            "train_team_mse_per_target": tr["team_mse_per_target"],
            "val_team_mse_per_target": va["team_mse_per_target"],
            "train_player_mse_per_target": tr["player_mse_per_target"],
            "val_player_mse_per_target": va["player_mse_per_target"],
            "train_pair_nll_per_target": tr["pair_nll_per_target"],
            "val_pair_nll_per_target": va["pair_nll_per_target"],
            "train_L_margin_nll": tr["L_margin_nll"], "val_L_margin_nll": va["L_margin_nll"],
            "train_L_calibration_reg": tr["L_calibration_reg"], "val_L_calibration_reg": va["L_calibration_reg"],
            "train_mean_log_sigma": tr["mean_log_sigma"], "val_mean_log_sigma": va["mean_log_sigma"],
            "train_mean_prob": tr["mean_prob"], "val_mean_prob": va["mean_prob"],
            "train_mean_base_prob": tr["mean_base_prob"], "val_mean_base_prob": va["mean_base_prob"],
            "train_mean_label": tr["mean_label"], "val_mean_label": va["mean_label"],
            "train_mean_win_residual": tr["mean_win_residual"], "val_mean_win_residual": va["mean_win_residual"],
            "train_home_logit_bias": tr["home_logit_bias"], "val_home_logit_bias": va["home_logit_bias"],
        }
        history.append(row)
        extra = ""
        if args.margin_nll_w > 0:
            extra = (f" mnll={tr['L_margin_nll']:.3f}/{va['L_margin_nll']:.3f}"
                     f" lσ={tr['mean_log_sigma']:.2f}/{va['mean_log_sigma']:.2f}")
        print(
            f"[ep{epoch:02d}] tr_loss={tr['loss']:.3f} "
            f"(t={tr['L_team']:.2f} pl={tr['L_player']:.2f} pr={tr['L_pair']:.2f} "
            f"inv={tr['L_inv']:.4f} win={tr['L_win']:.4f}) "
            f"tr_bce={tr['bce']:.4f} tr_acc={tr['acc']:.3f} "
            f"tr_h={tr['mean_home_pts']:.1f}/a={tr['mean_away_pts']:.1f} "
            f"tr_N={tr['mean_N']:.1f} αr={tr['alpha_off_corr']:.3f} "
            f"p={tr['mean_prob']:.3f}/{tr['mean_label']:.3f} "
            f"δ={tr['mean_win_residual']:.3f} b={tr['home_logit_bias']:.3f} | "
            f"va_loss={va['loss']:.3f} va_bce={va['bce']:.4f} va_acc={va['acc']:.3f} "
            f"va_mae={va['margin_mae']:.2f} va_αr={va['alpha_off_corr']:.3f} "
            f"va_p={va['mean_prob']:.3f}/{va['mean_label']:.3f} "
            f"va_δ={va['mean_win_residual']:.3f} va_b={va['home_logit_bias']:.3f}{extra} "
            f"lr={cur_lr:.2e} ({dt:.1f}s)"
        )

        if va["bce"] < best_val - 1e-5:
            best_val = va["bce"]
            best_epoch = epoch
            epochs_since_best = 0
            save_ckpt(ckpt_path, epoch)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"[early-stop] no val_bce improvement for {args.patience} epochs")
                break

    final_path = out_dir / "final.pt"
    save_ckpt(final_path, epoch)
    print(f"[best] epoch={best_epoch} val_bce={best_val:.4f}")

    print("[load best] running test")
    state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model_state"])
    te_best = run_epoch(model, test_loader, device=args.device, optim=None, **epoch_kwargs())
    print(
        f"[test:best] loss={te_best['loss']:.3f} bce={te_best['bce']:.4f} "
        f"acc={te_best['acc']:.3f} mae={te_best['margin_mae']:.2f} "
        f"αr={te_best['alpha_off_corr']:.3f} "
        f"mean_h={te_best['mean_home_pts']:.1f} mean_a={te_best['mean_away_pts']:.1f} "
        f"mean_N={te_best['mean_N']:.1f} "
        f"p={te_best['mean_prob']:.3f}/{te_best['mean_label']:.3f} "
        f"base_p={te_best['mean_base_prob']:.3f} "
        f"δ={te_best['mean_win_residual']:.3f} b={te_best['home_logit_bias']:.3f}"
    )
    print("  team MSE per stat:")
    for name, v in zip(BOX_TARGETS, te_best["team_mse_per_target"]):
        print(f"    {name:6s} {v:.4f}")
    print("  player MSE per stat:")
    for name, v in zip(BOX_TARGETS, te_best["player_mse_per_target"]):
        print(f"    {name:6s} {v:.4f}")
    print("  pair NLL per channel:")
    for name, v in zip(PAIR_TARGETS, te_best["pair_nll_per_target"]):
        print(f"    {name:25s} {v:.4f}")

    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "vocab_size": vocab.size,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "best_epoch": best_epoch,
        "best_val_bce": best_val,
        "final_epoch": epoch,
        "test_best": te_best,
        "history": history,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
