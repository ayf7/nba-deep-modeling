#!/usr/bin/env python3
"""Expanding-window monthly backtest for CME-v1.5.

Adds selector-head columns to predictions.csv:
  pred_home_win_prob, sel_p_home, sel_p_away, sel_p_nobet, sel_action,
  mkt_p_home, has_odds, home_dec_odds, away_dec_odds.

sel_action = argmax over {bet_home, bet_away, no_bet}; downstream EV evaluation
uses this directly instead of hand-coded disagree(0.60).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
V5_SCRIPTS = REPO_ROOT / "models_man_xfmr" / "scripts"
CME_V1_SCRIPTS = REPO_ROOT / "models_cme_v1" / "scripts"
sys.path.insert(0, str(V5_SCRIPTS))
sys.path.insert(0, str(CME_V1_SCRIPTS))

from man_xfmr_common import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB,
    DEFAULT_LINEUP_DECAY,
    DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB,
    DEFAULT_PLAYER_FORM_DECAY,
    DEFAULT_PLAYER_FORM_LOOKBACK,
    PLAYER_FORM_DIM,
    TABULAR_FEATURE_COLUMNS,
    XfmrGameDataset,
    build_records,
    build_team_vocab,
    build_vocab_from_records,
    collate_xfmr,
    fit_player_form_stats,
    fit_tabular_stats,
    load_game_odds,
    load_game_player_status,
    load_game_scores,
    load_games,
    load_matchup_rows,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
)

from cme_v1_model import (  # noqa: E402
    CmeV1Config,
    gather_lambda_for_supervision,
    poisson_nll,
)
from cme_v1_5_model import CmeV15, CmeV15Config, selector_loss  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v1_5" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="backtest_cme_v1_5")
    p.add_argument("--initial-train-end", default="2023-12-31")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)

    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)

    p.add_argument("--d", type=int, default=32)
    p.add_argument("--pair-hidden", type=int, default=64)
    p.add_argument("--eff-hidden", type=int, default=16)
    p.add_argument("--inv-hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--pair-dropout", type=float, default=0.5)
    p.add_argument("--player-dropout", type=float, default=0.0)
    p.add_argument("--team-emb-dim", type=int, default=8)
    p.add_argument("--n-self-attn-heads", type=int, default=2)
    p.add_argument("--n-self-attn-layers", type=int, default=1)
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--base-possessions-per-team", type=float, default=491.0)
    p.add_argument("--base-eff", type=float, default=0.224)
    p.add_argument("--eff-amplitude", type=float, default=0.05)
    p.add_argument("--init-global-scale", type=float, default=9.0)
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--use-player-stats", action="store_true", default=True)
    p.add_argument("--no-player-stats", dest="use_player_stats", action="store_false")
    p.add_argument("--margin-coef", type=float, default=0.03)
    p.add_argument("--margin-huber-delta", type=float, default=10.0)

    # selector
    p.add_argument("--sel-hidden", type=int, default=96)
    p.add_argument("--sel-dropout", type=float, default=0.20)
    p.add_argument("--coverage-target", type=float, default=0.25)
    p.add_argument("--coverage-weight", type=float, default=8.0)
    p.add_argument("--sel-loss-weight", type=float, default=0.30)
    p.add_argument("--nobet-bias-init", type=float, default=0.0)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Run only first 2 windows for a quick check.")
    return p.parse_args()


def chrono_train_val(train_df: pd.DataFrame, val_frac: float):
    df = train_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_frac))
    n_train = len(df) - n_val
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    periods = pd.to_datetime(dates).dt.to_period("M").drop_duplicates().sort_values()
    return [period.to_timestamp() for period in periods]


def run_eval(model, loader, device, args, cfg):
    model.eval()
    sums = {"n": 0, "pois": 0.0, "marg": 0.0, "bce": 0.0, "correct": 0,
            "sel_ev": 0.0, "n_with_odds": 0, "coverage": 0.0, "p_nobet": 0.0}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            lam_pred = gather_lambda_for_supervision(
                out["lam_h"], out["lam_a"],
                batch["sup_game"], batch["sup_side"],
                batch["sup_off"], batch["sup_def"],
            )
            pois = poisson_nll(lam_pred, batch["sup_exp"])
            marg = F.huber_loss(out["margin_mu"], batch["margin"], delta=args.margin_huber_delta)
            bce = F.binary_cross_entropy_with_logits(out["win_logit"], batch["label"])
            preds = (torch.sigmoid(out["win_logit"]) > 0.5).float()

            _, sel_diag = selector_loss(
                out["sel_probs"], batch["label"],
                batch["home_dec_odds"], batch["away_dec_odds"],
                batch["has_odds"],
                coverage_target=cfg.coverage_target,
                coverage_weight=cfg.coverage_weight,
            )

            bs = batch["label"].size(0)
            n_odds = int(batch["has_odds"].sum().item())
            sums["n"] += bs
            sums["pois"] += pois.item() * bs
            sums["marg"] += marg.item() * bs
            sums["bce"] += bce.item() * bs
            sums["correct"] += (preds == batch["label"]).sum().item()
            sums["sel_ev"] += sel_diag["sel_ev"] * max(n_odds, 1)
            sums["coverage"] += sel_diag["coverage"] * max(n_odds, 1)
            sums["p_nobet"] += sel_diag["p_nobet_mean"] * max(n_odds, 1)
            sums["n_with_odds"] += n_odds

    n = max(sums["n"], 1)
    no = max(sums["n_with_odds"], 1)
    return {
        "n": sums["n"], "n_with_odds": sums["n_with_odds"],
        "pois": sums["pois"] / n, "marg": sums["marg"] / n,
        "combined": sums["pois"] / n + args.margin_coef * sums["marg"] / n,
        "bce": sums["bce"] / n, "acc": sums["correct"] / n,
        "sel_ev": sums["sel_ev"] / no,
        "coverage": sums["coverage"] / no,
        "p_nobet": sums["p_nobet"] / no,
    }


def predict_window_full(model, loader, device):
    """Returns dict of per-game arrays from the model output."""
    model.eval()
    cols = {"p_home": [], "sh": [], "sa": [], "sn": [],
            "mkt": [], "has": [], "h_dec": [], "a_dec": []}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            cols["p_home"].extend(torch.sigmoid(out["win_logit"]).cpu().numpy().tolist())
            sp = out["sel_probs"].cpu().numpy()
            cols["sh"].extend(sp[:, 0].tolist())
            cols["sa"].extend(sp[:, 1].tolist())
            cols["sn"].extend(sp[:, 2].tolist())
            cols["mkt"].extend(out["mkt_p_home"].cpu().numpy().tolist())
            cols["has"].extend(batch["has_odds"].cpu().numpy().tolist())
            cols["h_dec"].extend(batch["home_dec_odds"].cpu().numpy().tolist())
            cols["a_dec"].extend(batch["away_dec_odds"].cpu().numpy().tolist())
    return {k: np.asarray(v) for k, v in cols.items()}


def train_one_window(
    train_df, val_df, *,
    histories, statuses, calibration, scores,
    matchup_rows_train, matchup_rows_val,
    player_histories, args, device, game_odds,
):
    vocab = build_vocab_from_records(
        train_df, histories, matchup_rows_train,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(train_df)
    tabular_stats = fit_tabular_stats(train_df)
    if args.use_player_stats:
        player_form_stats = fit_player_form_stats(
            player_histories, train_df,
            lookback_games=args.player_form_lookback,
            decay=args.player_form_decay,
        )
    else:
        player_form_stats = None

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
    train_recs = build_records(train_df, matchup_rows=matchup_rows_train, **common)
    val_recs = build_records(val_df, matchup_rows=matchup_rows_val, **common)

    train_loader = DataLoader(XfmrGameDataset(train_recs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_xfmr)
    val_loader = DataLoader(XfmrGameDataset(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_xfmr)

    tabular_dim = 0 if args.no_tabular else len(TABULAR_FEATURE_COLUMNS)
    cme_cfg = CmeV1Config(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, pair_hidden=args.pair_hidden,
        eff_hidden=args.eff_hidden, inv_hidden=args.inv_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        tabular_dim=tabular_dim, team_emb_dim=args.team_emb_dim,
        player_stat_dim=PLAYER_FORM_DIM if args.use_player_stats else 0,
        n_self_attn_heads=args.n_self_attn_heads,
        n_self_attn_layers=args.n_self_attn_layers,
        sinkhorn_iters=args.sinkhorn_iters,
        base_possessions_per_team=args.base_possessions_per_team,
        base_eff=args.base_eff, eff_amplitude=args.eff_amplitude,
        init_global_scale=args.init_global_scale,
    )
    cfg = CmeV15Config(
        cme=cme_cfg, sel_hidden=args.sel_hidden, sel_dropout=args.sel_dropout,
        coverage_target=args.coverage_target, coverage_weight=args.coverage_weight,
        sel_loss_weight=args.sel_loss_weight, nobet_bias_init=args.nobet_bias_init,
    )
    model = CmeV15(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch_idx):
        if args.warmup_epochs > 0 and epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / args.warmup_epochs
        denom = max(1, args.epochs - args.warmup_epochs)
        progress = min(max((epoch_idx - args.warmup_epochs) / denom, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    best_meta = {}
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            lam_pred = gather_lambda_for_supervision(
                out["lam_h"], out["lam_a"],
                batch["sup_game"], batch["sup_side"],
                batch["sup_off"], batch["sup_def"],
            )
            pois = poisson_nll(lam_pred, batch["sup_exp"])
            marg = F.huber_loss(out["margin_mu"], batch["margin"], delta=args.margin_huber_delta)
            sel_total, _ = selector_loss(
                out["sel_probs"], batch["label"],
                batch["home_dec_odds"], batch["away_dec_odds"],
                batch["has_odds"],
                coverage_target=cfg.coverage_target, coverage_weight=cfg.coverage_weight,
            )
            loss = pois + args.margin_coef * marg + cfg.sel_loss_weight * sel_total
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
        scheduler.step()
        va = run_eval(model, val_loader, device, args, cfg)

        # Pick best on -sel_ev (the thing we actually care about); fall back to bce
        # if odds are too sparse to estimate it stably.
        val_score = -va["sel_ev"] if va["n_with_odds"] >= 30 else va["bce"]
        if val_score < best_val - 1e-5:
            best_val = val_score
            best_epoch = epoch
            best_meta = {"val_bce": va["bce"], "val_sel_ev": va["sel_ev"],
                         "val_cov": va["coverage"], "val_pnobet": va["p_nobet"],
                         "val_combined": va["combined"]}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, cfg, {
        "best_epoch": best_epoch,
        "best_val_score": best_val,
        **best_meta,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
    }, {
        "vocab": vocab, "team_vocab": team_vocab,
        "tabular_stats": tabular_stats, "player_form_stats": player_form_stats,
    }


def predict_window(
    model, *, test_df, histories, statuses, calibration, scores, args,
    fitted, player_histories, game_odds,
):
    test_recs = build_records(
        test_df, histories=histories, vocab=fitted["vocab"], team_vocab=fitted["team_vocab"],
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        matchup_rows=None,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=fitted["tabular_stats"],
        player_histories=player_histories,
        player_form_stats=fitted["player_form_stats"],
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=game_odds,
    )
    if not test_recs:
        return pd.DataFrame(), {"n_test": 0}
    loader = DataLoader(XfmrGameDataset(test_recs), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_xfmr)
    preds = predict_window_full(model, loader, args.device)
    sel_action = np.argmax(np.stack([preds["sh"], preds["sa"], preds["sn"]], axis=1), axis=1)
    df = pd.DataFrame({
        "game_id": [r.game_id for r in test_recs],
        "game_date": [r.game_date.strftime("%Y-%m-%d") for r in test_recs],
        "label_home_win": [int(r.label) for r in test_recs],
        "pred_home_win_prob": preds["p_home"],
        "sel_p_home": preds["sh"],
        "sel_p_away": preds["sa"],
        "sel_p_nobet": preds["sn"],
        "sel_action": sel_action,
        "mkt_p_home": preds["mkt"],
        "has_odds": preds["has"].astype(int),
        "home_dec_odds": preds["h_dec"],
        "away_dec_odds": preds["a_dec"],
    })
    eps = 1e-7
    p = np.clip(df["pred_home_win_prob"].to_numpy(), eps, 1 - eps)
    y = df["label_home_win"].to_numpy()
    bce = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    acc = float(((df["pred_home_win_prob"] >= 0.5).astype(int) == y).mean())

    # Selector realized profit on test (per the model's chosen actions, masked by has_odds)
    has = df["has_odds"].to_numpy().astype(bool)
    a = sel_action
    h_dec = df["home_dec_odds"].to_numpy()
    a_dec = df["away_dec_odds"].to_numpy()
    profit = np.zeros(len(df))
    bh = (a == 0) & has
    ba = (a == 1) & has
    profit[bh] = np.where(y[bh] == 1, h_dec[bh] - 1, -1)
    profit[ba] = np.where(y[ba] == 0, a_dec[ba] - 1, -1)
    bet_mask = (a < 2) & has
    n_bets = int(bet_mask.sum())
    sel_realized_roi = float(profit[bet_mask].mean()) if n_bets else 0.0
    coverage = n_bets / max(int(has.sum()), 1)

    return df, {
        "n_test": len(df),
        "test_bce": bce, "test_acc": acc,
        "test_sel_n_bets": n_bets,
        "test_sel_roi": sel_realized_roi,
        "test_sel_coverage": coverage,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {args.device}")
    print("[load] games + scores + status + calibration + exposures + odds (one-time)")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)
    game_odds = load_game_odds(args.core_db, game_ids)
    print(f"[odds] {len(game_odds)}/{len(game_ids)} games have moneyline odds")

    if args.use_player_stats:
        print("[load] player histories (one-time)")
        player_histories = load_player_histories(args.core_db)
    else:
        player_histories = None

    initial_train_end = pd.Timestamp(args.initial_train_end)
    windows = [
        s for s in month_starts(games_all.loc[games_all["game_date"] > initial_train_end, "game_date"])
        if s > initial_train_end
    ]
    if args.smoke:
        windows = windows[:2]
    print(f"[backtest] {len(windows)} monthly windows starting {windows[0].date()} -> {windows[-1].date()}")

    all_preds: list[pd.DataFrame] = []
    metrics_rows: list[dict] = []
    overall_t0 = time.time()
    for w_idx, window_start in enumerate(windows, 1):
        window_end = window_start + pd.offsets.MonthBegin(1)
        full_train = games_all[games_all["game_date"] < window_start].copy()
        test_df = games_all[
            (games_all["game_date"] >= window_start) & (games_all["game_date"] < window_end)
        ].copy()
        if full_train.empty or test_df.empty:
            print(f"[w{w_idx:02d}] {window_start.date()}: skip (empty train or test)")
            continue
        train_df, val_df = chrono_train_val(full_train, args.val_frac)

        matchup_rows_train = load_matchup_rows(
            args.matchup_db, [str(g) for g in train_df["game_id"].tolist()]
        )
        matchup_rows_val = load_matchup_rows(
            args.matchup_db, [str(g) for g in val_df["game_id"].tolist()]
        )

        t0 = time.time()
        model, cfg, train_meta, fitted = train_one_window(
            train_df, val_df,
            histories=histories, statuses=statuses, calibration=calibration,
            scores=scores, matchup_rows_train=matchup_rows_train,
            matchup_rows_val=matchup_rows_val,
            player_histories=player_histories, args=args, device=args.device,
            game_odds=game_odds,
        )
        train_dt = time.time() - t0

        t1 = time.time()
        preds_df, eval_meta = predict_window(
            model, test_df=test_df,
            histories=histories, statuses=statuses, calibration=calibration,
            scores=scores, args=args,
            fitted=fitted, player_histories=player_histories, game_odds=game_odds,
        )
        pred_dt = time.time() - t1

        if not preds_df.empty:
            preds_df = preds_df.assign(window_start=window_start.date().isoformat())
            all_preds.append(preds_df)

        row = {
            "window_start": window_start.date().isoformat(),
            "window_end": (window_end - pd.Timedelta(days=1)).date().isoformat(),
            "train_n": train_meta["n_train"], "val_n": train_meta["n_val"],
            "best_epoch": train_meta["best_epoch"],
            "val_bce": train_meta.get("val_bce"),
            "val_sel_ev": train_meta.get("val_sel_ev"),
            "val_cov": train_meta.get("val_cov"),
            "val_pnobet": train_meta.get("val_pnobet"),
            **eval_meta,
            "train_secs": round(train_dt, 1),
            "pred_secs": round(pred_dt, 1),
        }
        metrics_rows.append(row)
        elapsed = time.time() - overall_t0
        eta = elapsed * (len(windows) - w_idx) / w_idx if w_idx > 0 else 0
        print(
            f"[w{w_idx:02d}/{len(windows)}] {window_start.date()}: "
            f"val_sel_ev={row['val_sel_ev']:+.4f}@ep{row['best_epoch']} "
            f"test_n={row['n_test']} bce={row['test_bce']:.4f} "
            f"sel_n={row['test_sel_n_bets']} sel_roi={row['test_sel_roi']:+.3f} "
            f"cov={row['test_sel_coverage']:.2f} "
            f"({train_dt:.0f}s+{pred_dt:.0f}s) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
        )

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "window_metrics.csv", index=False)
    overall = {
        "n_total_predictions": int(len(predictions)),
        "n_windows": len(metrics_rows),
        "wall_secs": round(time.time() - overall_t0, 1),
    }
    if not predictions.empty:
        eps = 1e-7
        p = np.clip(predictions["pred_home_win_prob"].to_numpy(), eps, 1 - eps)
        y = predictions["label_home_win"].to_numpy()
        overall["overall_bce"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        overall["overall_acc"] = float(((predictions["pred_home_win_prob"] >= 0.5).astype(int) == y).mean())
        # selector aggregate
        has = predictions["has_odds"].to_numpy().astype(bool)
        a = predictions["sel_action"].to_numpy()
        bet_mask = (a < 2) & has
        bh = (a == 0) & has
        ba = (a == 1) & has
        h_dec = predictions["home_dec_odds"].to_numpy()
        a_dec = predictions["away_dec_odds"].to_numpy()
        y_arr = predictions["label_home_win"].to_numpy()
        profit = np.zeros(len(predictions))
        profit[bh] = np.where(y_arr[bh] == 1, h_dec[bh] - 1, -1)
        profit[ba] = np.where(y_arr[ba] == 0, a_dec[ba] - 1, -1)
        n_bets = int(bet_mask.sum())
        overall["sel_n_bets"] = n_bets
        overall["sel_n_home"] = int(bh.sum())
        overall["sel_n_away"] = int(ba.sum())
        overall["sel_coverage"] = float(n_bets / max(int(has.sum()), 1))
        overall["sel_roi"] = float(profit[bet_mask].mean()) if n_bets else 0.0
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[done] {out_dir}: {overall}")


if __name__ == "__main__":
    main()
