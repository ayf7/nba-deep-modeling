#!/usr/bin/env python3
"""Train CME-v2 with hierarchical-marginal + win-supervision loss.

    L = team_w · L_team_mse + player_w · L_player_mse
      + pair_w · L_pair_poisson + win_w · L_win_bce

Team and player levels use per-stat MSE (fully quadratic at every scale,
so big team-points errors get strong pull-back). Pair level keeps Poisson
NLL because pair counts are 0–3 — the regime Poisson is designed for. The
win term is BCE on the composed margin → win_logit, providing the only
direct gradient on the differential signal (the box-score MSE losses can
all be satisfied by predicting league-average pace symmetrically).

Win logit / accuracy / margin MAE are reported as diagnostics. They are
computed by deterministic composition (margin = home_team_pts - away_team_pts)
and not separately supervised — global_scale is calibrated post-hoc.
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
V2_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_SCRIPTS))

from cme_v2_common import (  # noqa: E402
    BOX_INDEX, BOX_TARGETS, K_BOX, K_PAIR, PAIR_TARGETS,
    DEFAULT_CALIBRATION_PATH, DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB, DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    PLAYER_FORM_DIM, TABULAR_FEATURE_COLUMNS,
    GameDatasetV2, build_records_v2, build_team_vocab,
    build_vocab_from_records_v2, collate_v2,
    fit_player_form_stats, fit_tabular_stats,
    load_game_odds, load_game_player_status, load_game_scores, load_games,
    load_matchup_rows_v2, load_player_game_stats, load_player_histories,
    load_status_calibration, load_team_exposures,
)
from cme_v2_model import CmeV2, CmeV2Config, total_loss  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v2" / "artifacts"


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
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pair-dropout", type=float, default=0.2)
    p.add_argument("--player-dropout", type=float, default=0.0)
    p.add_argument("--team-emb-dim", type=int, default=16)
    p.add_argument("--init-scale", type=float, default=12.0)
    p.add_argument("--direct-win-head", action="store_true",
                   help="Backbone probe: bypass box/margin chain, predict win "
                        "logit directly from pooled post-cross reps + team ctx.")
    p.add_argument("--use-team-token", action="store_true",
                   help="Prepend a learnable CLS-style team token per side; "
                        "its post-cross-attn position is the team summary "
                        "fed to the decision head.")
    p.add_argument("--use-decision-head", action="store_true",
                   help="Add a binary classifier that predicts whether the "
                        "win head's chosen side will produce a winning bet. "
                        "Requires --use-team-token.")
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
    p.add_argument("--team-w", type=float, default=1.0,
                   help="Weight on L_team (default 1.0)")
    p.add_argument("--player-w", type=float, default=0.01,
                   help="Weight on L_player_marginal (default 0.01)")
    p.add_argument("--pair-w", type=float, default=0.001,
                   help="Weight on L_pair (default 0.001)")
    p.add_argument("--win-w", type=float, default=10.0,
                   help="Weight on L_win_bce (default 10.0). Direct margin "
                        "supervision; breaks the home/away symmetry.")
    p.add_argument("--margin-nll-w", type=float, default=0.0,
                   help="Weight on Gaussian NLL on margin with learned σ. "
                        "Dense per-game gradient. 0 disables.")
    p.add_argument("--decision-w", type=float, default=0.0,
                   help="Weight on BCE(decision_logit, bet_won). 0 disables. "
                        "Requires --use-decision-head.")
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
    model: CmeV2, loader: DataLoader, *, device: str,
    optim: torch.optim.Optimizer | None,
    box_weights: torch.Tensor, pair_weights: torch.Tensor,
    team_w: float, player_w: float, pair_w: float, win_w: float,
    margin_nll_w: float = 0.0, decision_w: float = 0.0,
) -> dict:
    is_train = optim is not None
    model.train(is_train)

    total_n = 0
    sum_loss = 0.0
    sum_L_team = 0.0
    sum_L_player = 0.0
    sum_L_pair = 0.0
    sum_L_win = 0.0
    sum_L_margin_nll = 0.0
    sum_L_decision = 0.0
    sum_team_mse_per_target = torch.zeros(K_BOX)
    sum_player_mse_per_target = torch.zeros(K_BOX)
    sum_pair_nll_per_target = torch.zeros(K_PAIR)
    sum_bce = 0.0
    sum_margin_abs = 0.0
    sum_log_sigma = 0.0
    sum_pts_h = 0.0
    sum_pts_a = 0.0
    correct = 0
    dec_n = 0
    dec_correct = 0
    dec_bet_won_sum = 0.0
    dec_pred_pos_sum = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            out = model(batch)
            loss, diag = total_loss(
                out, batch,
                box_weights=box_weights.to(device),
                pair_weights=pair_weights.to(device),
                team_w=team_w, player_w=player_w, pair_w=pair_w, win_w=win_w,
                margin_nll_w=margin_nll_w, decision_w=decision_w,
            )

            if is_train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

            with torch.no_grad():
                bce = F.binary_cross_entropy_with_logits(
                    out["win_logit"], batch["label"]
                )
                margin_abs = (out["margin_mu"] - batch["margin"]).abs().mean()
                preds = (torch.sigmoid(out["win_logit"]) > 0.5).float()
                correct_bs = (preds == batch["label"]).sum().item()

                if out.get("decision_logit") is not None:
                    has = batch["has_odds"].bool()
                    if has.any():
                        d_logit = out["decision_logit"][has]
                        bet_home = (out["win_logit"][has].detach() >= 0.0).float()
                        bet_won = (bet_home == batch["label"][has]).float()
                        d_pred = (torch.sigmoid(d_logit) > 0.5).float()
                        dec_n += int(has.sum().item())
                        dec_correct += int((d_pred == bet_won).sum().item())
                        dec_bet_won_sum += float(bet_won.sum().item())
                        dec_pred_pos_sum += float(d_pred.sum().item())

        bs = batch["label"].size(0)
        total_n += bs
        sum_loss += loss.item() * bs
        sum_L_team += diag["L_team"].item() * bs
        sum_L_player += diag["L_player"].item() * bs
        sum_L_pair += diag["L_pair"].item() * bs
        sum_L_win += diag["L_win"].item() * bs
        sum_L_margin_nll += diag["L_margin_nll"].item() * bs
        sum_L_decision += diag["L_decision"].item() * bs
        sum_team_mse_per_target += diag["team_mse_per_target"].cpu() * bs
        sum_player_mse_per_target += diag["player_mse_per_target"].cpu() * bs
        sum_pair_nll_per_target += diag["pair_nll_per_target"].cpu() * bs
        sum_bce += bce.item() * bs
        sum_margin_abs += margin_abs.item() * bs
        sum_log_sigma += out["margin_log_sigma"].mean().item() * bs
        sum_pts_h += out["home_points"].mean().item() * bs
        sum_pts_a += out["away_points"].mean().item() * bs
        correct += correct_bs

    return {
        "n": total_n,
        "loss": sum_loss / total_n,
        "L_team": sum_L_team / total_n,
        "L_player": sum_L_player / total_n,
        "L_pair": sum_L_pair / total_n,
        "L_win": sum_L_win / total_n,
        "L_margin_nll": sum_L_margin_nll / total_n,
        "L_decision": sum_L_decision / total_n,
        "team_mse_per_target": (sum_team_mse_per_target / total_n).tolist(),
        "player_mse_per_target": (sum_player_mse_per_target / total_n).tolist(),
        "pair_nll_per_target": (sum_pair_nll_per_target / total_n).tolist(),
        "bce": sum_bce / total_n,
        "margin_mae": sum_margin_abs / total_n,
        "mean_log_sigma": sum_log_sigma / total_n,
        "acc": correct / total_n,
        "mean_home_pts": sum_pts_h / total_n,
        "mean_away_pts": sum_pts_a / total_n,
        "dec_n": dec_n,
        "dec_acc": (dec_correct / dec_n) if dec_n > 0 else 0.0,
        "dec_bet_won_rate": (dec_bet_won_sum / dec_n) if dec_n > 0 else 0.0,
        "dec_pred_pos_rate": (dec_pred_pos_sum / dec_n) if dec_n > 0 else 0.0,
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

    print("[load] matchup rows (v2: K_pair targets)")
    train_matchup = load_matchup_rows_v2(args.matchup_db, train_gids)
    val_matchup = load_matchup_rows_v2(args.matchup_db, val_gids)
    test_matchup = load_matchup_rows_v2(args.matchup_db, test_gids)

    print("[load] per-player non-pair stats from game_events")
    train_pl = load_player_game_stats(args.core_db, train_gids)
    val_pl = load_player_game_stats(args.core_db, val_gids)
    test_pl = load_player_game_stats(args.core_db, test_gids)

    print("[vocab] building from train rosters + train matchup rows")
    vocab = build_vocab_from_records_v2(
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
    train_recs = build_records_v2(
        train_df, matchup_rows=train_matchup, player_game_stats=train_pl, **common,
    )
    val_recs = build_records_v2(
        val_df, matchup_rows=val_matchup, player_game_stats=val_pl, **common,
    )
    test_recs = build_records_v2(
        test_df, matchup_rows=test_matchup, player_game_stats=test_pl, **common,
    )
    print(f"[records] train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

    train_loader = DataLoader(GameDatasetV2(train_recs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_v2)
    val_loader = DataLoader(GameDatasetV2(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_v2)
    test_loader = DataLoader(GameDatasetV2(test_recs), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_v2)

    tabular_dim = 0 if args.no_tabular else len(TABULAR_FEATURE_COLUMNS)
    cfg = CmeV2Config(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, n_heads=args.n_heads,
        n_self_layers=args.n_self_layers, n_cross_layers=args.n_cross_layers,
        pair_hidden=args.pair_hidden, player_hidden=args.player_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        tabular_dim=tabular_dim, team_emb_dim=args.team_emb_dim,
        player_stat_dim=PLAYER_FORM_DIM if args.use_player_stats else 0,
        init_scale=args.init_scale,
        use_direct_win_head=args.direct_win_head,
        use_team_token=args.use_team_token,
        use_decision_head=args.use_decision_head,
    )
    model = CmeV2(cfg).to(args.device)
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
          f"pair_w={args.pair_w} win_w={args.win_w} "
          f"margin_nll_w={args.margin_nll_w} "
          f"decision_w={args.decision_w} "
          f"direct_win_head={args.direct_win_head} "
          f"team_token={args.use_team_token} "
          f"decision_head={args.use_decision_head}")
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
             "loss_level_weights": {"team": args.team_w, "player": args.player_w, "pair": args.pair_w},
             "epoch": epoch},
            path,
        )

    def epoch_kwargs() -> dict:
        return dict(
            box_weights=box_weights, pair_weights=pair_weights,
            team_w=args.team_w, player_w=args.player_w, pair_w=args.pair_w,
            win_w=args.win_w, margin_nll_w=args.margin_nll_w,
            decision_w=args.decision_w,
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
            "train_L_win": tr["L_win"], "val_L_win": va["L_win"],
            "train_bce": tr["bce"], "val_bce": va["bce"],
            "train_acc": tr["acc"], "val_acc": va["acc"],
            "train_margin_mae": tr["margin_mae"], "val_margin_mae": va["margin_mae"],
            "train_mean_home_pts": tr["mean_home_pts"], "val_mean_home_pts": va["mean_home_pts"],
            "train_mean_away_pts": tr["mean_away_pts"], "val_mean_away_pts": va["mean_away_pts"],
            "train_team_mse_per_target": tr["team_mse_per_target"],
            "val_team_mse_per_target": va["team_mse_per_target"],
            "train_player_mse_per_target": tr["player_mse_per_target"],
            "val_player_mse_per_target": va["player_mse_per_target"],
            "train_pair_nll_per_target": tr["pair_nll_per_target"],
            "val_pair_nll_per_target": va["pair_nll_per_target"],
            "train_L_margin_nll": tr["L_margin_nll"], "val_L_margin_nll": va["L_margin_nll"],
            "train_mean_log_sigma": tr["mean_log_sigma"], "val_mean_log_sigma": va["mean_log_sigma"],
            "train_L_decision": tr["L_decision"], "val_L_decision": va["L_decision"],
            "train_dec_acc": tr["dec_acc"], "val_dec_acc": va["dec_acc"],
            "train_dec_bet_won_rate": tr["dec_bet_won_rate"], "val_dec_bet_won_rate": va["dec_bet_won_rate"],
            "train_dec_pred_pos_rate": tr["dec_pred_pos_rate"], "val_dec_pred_pos_rate": va["dec_pred_pos_rate"],
            "train_dec_n": tr["dec_n"], "val_dec_n": va["dec_n"],
        }
        history.append(row)
        extra = ""
        if args.margin_nll_w > 0:
            extra = (f" mnll={tr['L_margin_nll']:.3f}/{va['L_margin_nll']:.3f}"
                     f" lσ={tr['mean_log_sigma']:.2f}/{va['mean_log_sigma']:.2f}")
        if args.decision_w > 0:
            extra += (f" dec={tr['L_decision']:.3f}/{va['L_decision']:.3f}"
                      f" da={tr['dec_acc']:.3f}/{va['dec_acc']:.3f}"
                      f" (bw={va['dec_bet_won_rate']:.3f}"
                      f" pp={va['dec_pred_pos_rate']:.3f}"
                      f" n={va['dec_n']})")
        print(
            f"[ep{epoch:02d}] tr_loss={tr['loss']:.3f} "
            f"(t={tr['L_team']:.2f} pl={tr['L_player']:.2f} pr={tr['L_pair']:.2f} "
            f"win={tr['L_win']:.4f}) "
            f"tr_bce={tr['bce']:.4f} tr_acc={tr['acc']:.3f} "
            f"tr_h={tr['mean_home_pts']:.1f}/a={tr['mean_away_pts']:.1f} | "
            f"va_loss={va['loss']:.3f} va_bce={va['bce']:.4f} va_acc={va['acc']:.3f} "
            f"va_mae={va['margin_mae']:.2f}{extra} lr={cur_lr:.2e} ({dt:.1f}s)"
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
        f"mean_h={te_best['mean_home_pts']:.1f} mean_a={te_best['mean_away_pts']:.1f}"
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
