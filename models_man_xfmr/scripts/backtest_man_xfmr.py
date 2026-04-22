#!/usr/bin/env python3
"""Expanding-window monthly backtest for MAN-Xfmr (v5).

For each month from --initial-train-end onwards:
  - Train on all games before window_start (chrono val split for early stop)
  - Predict on games in [window_start, window_end)
  - Append predictions and per-window metrics

Output: predictions.csv (concat across months) + window_metrics.csv + config.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from man_xfmr_common import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB,
    DEFAULT_LINEUP_DECAY,
    DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB,
    DEFAULT_OUTPUT_ROOT,
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
from man_xfmr_model import (
    ManXfmr,
    XfmrConfig,
    gather_lambda_for_supervision,
    poisson_nll,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="backtest_man_xfmr")
    p.add_argument("--initial-train-end", default="2023-12-31",
                   help="Train cutoff before the first test window.")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Last fraction of each window's train block reserved for early-stop val.")

    # SOTA hyperparams (matched to v5_pstats_2stream_apoiss10).
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--pair-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--pair-dropout", type=float, default=0.5)
    p.add_argument("--player-dropout", type=float, default=0.15)
    p.add_argument("--team-emb-dim", type=int, default=0)
    p.add_argument("--two-stream", action="store_true", default=True)
    p.add_argument("--no-two-stream", dest="two_stream", action="store_false")
    p.add_argument("--use-player-stats", action="store_true", default=True)
    p.add_argument("--no-player-stats", dest="use_player_stats", action="store_false")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--alpha-poisson", type=float, default=10.0)
    p.add_argument("--ev-weight", type=float, default=0.0,
                   help="lambda for soft-EV loss term: bce + alpha*poisson + lambda * -E[bet_h*profit_h + bet_a*profit_a]")
    p.add_argument("--ev-bet-sharpness", type=float, default=50.0,
                   help="Sigmoid sharpness on edge for soft-bet gating (matches residual recipe)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Run only first 2 windows for a quick check.")
    p.add_argument("--save-train-preds", action="store_true",
                   help="Also save per-window in-sample predictions on train_df "
                        "to predictions_train.csv.")
    p.add_argument("--save-checkpoints", action="store_true",
                   help="Write best.pt per window to "
                        "<out>/windows/<window_start>/best.pt (with cfg/vocab "
                        "needed to reload the model).")
    p.add_argument("--save-final", action="store_true",
                   help="Additionally write final.pt (last-epoch weights) per "
                        "window. No effect without --save-checkpoints.")
    return p.parse_args()


def chrono_train_val(train_df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a single train block chronologically into (train, val)."""
    df = train_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_frac))
    n_train = len(df) - n_val
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    periods = pd.to_datetime(dates).dt.to_period("M").drop_duplicates().sort_values()
    return [period.to_timestamp() for period in periods]


def run_eval(model: ManXfmr, loader: DataLoader, device: str, alpha: float) -> dict:
    model.eval()
    total_n = 0
    total_bce = 0.0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, zero_tabular=False)
            bce = F.binary_cross_entropy_with_logits(out["win_logit"], batch["label"])
            bs = batch["label"].size(0)
            total_n += bs
            total_bce += bce.item() * bs
            preds = (torch.sigmoid(out["win_logit"]) > 0.5).float()
            correct += (preds == batch["label"]).sum().item()
    return {"n": total_n, "bce": total_bce / total_n if total_n else float("nan"),
            "acc": correct / total_n if total_n else float("nan")}


def predict_probs(model: ManXfmr, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, zero_tabular=False)
            probs.extend(torch.sigmoid(out["win_logit"]).detach().cpu().numpy().tolist())
    return np.asarray(probs, dtype=np.float64)


def _build_ckpt_payload(model, cfg, vocab, team_vocab, player_form_stats, epoch):
    return {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": asdict(cfg),
        "vocab": vocab.player_to_idx,
        "team_vocab": team_vocab.team_to_idx,
        "player_form_means": (player_form_stats.means.tolist()
                              if player_form_stats is not None else None),
        "player_form_stds": (player_form_stats.stds.tolist()
                             if player_form_stats is not None else None),
        "epoch": epoch,
    }


def train_one_window(
    train_df: pd.DataFrame, val_df: pd.DataFrame, *,
    histories, statuses, calibration, scores, matchup_rows_train,
    player_histories, args, device, game_odds=None,
    ckpt_dir: Path | None = None,
) -> tuple[ManXfmr, XfmrConfig, dict, dict]:
    """Build vocab/standardizers on train, train with early stop on val. Return model."""
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

    # Records.
    train_recs = build_records(
        train_df, histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        matchup_rows=matchup_rows_train,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories, player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=game_odds,
    )
    val_recs = build_records(
        val_df, histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        matchup_rows=None,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories, player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=game_odds,
    )
    train_loader = DataLoader(
        XfmrGameDataset(train_recs), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_xfmr,
    )
    val_loader = DataLoader(
        XfmrGameDataset(val_recs), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_xfmr,
    )

    cfg = XfmrConfig(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, pair_hidden=args.pair_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        margin_head=False,
        tabular_dim=len(TABULAR_FEATURE_COLUMNS),
        team_emb_dim=args.team_emb_dim,
        player_stat_dim=PLAYER_FORM_DIM if args.use_player_stats else 0,
        two_stream=args.two_stream,
    )
    model = ManXfmr(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch_idx: int) -> float:
        if args.warmup_epochs > 0 and epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / args.warmup_epochs
        denom = max(1, args.epochs - args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    epochs_since_best = 0
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, zero_tabular=False)
            bce = F.binary_cross_entropy_with_logits(out["win_logit"], batch["label"])
            lam_pred = gather_lambda_for_supervision(
                out["lam_h"], out["lam_a"],
                batch["sup_game"], batch["sup_side"],
                batch["sup_off"], batch["sup_def"],
            )
            poisson = poisson_nll(lam_pred, batch["sup_exp"])
            loss = bce + args.alpha_poisson * poisson
            if args.ev_weight > 0 and batch["has_odds"].sum() > 0:
                p = torch.sigmoid(out["win_logit"])
                edge_h = p - 1.0 / batch["home_dec_odds"]
                edge_a = (1.0 - p) - 1.0 / batch["away_dec_odds"]
                bet_h = torch.sigmoid(edge_h * args.ev_bet_sharpness)
                bet_a = torch.sigmoid(edge_a * args.ev_bet_sharpness)
                profit_h = batch["label"] * (batch["home_dec_odds"] - 1.0) - (1.0 - batch["label"])
                profit_a = (1.0 - batch["label"]) * (batch["away_dec_odds"] - 1.0) - batch["label"]
                expected = bet_h * profit_h + bet_a * profit_a
                denom = batch["has_odds"].sum().clamp(min=1.0)
                ev = -(expected * batch["has_odds"]).sum() / denom
                loss = loss + args.ev_weight * ev
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
        scheduler.step()
        va = run_eval(model, val_loader, device, args.alpha_poisson)
        if va["bce"] < best_val - 1e-5:
            best_val = va["bce"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                break

    if ckpt_dir is not None and args.save_final:
        final_payload = _build_ckpt_payload(
            model, cfg, vocab, team_vocab, player_form_stats, last_epoch,
        )
    else:
        final_payload = None

    if best_state is not None:
        model.load_state_dict(best_state)

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            _build_ckpt_payload(
                model, cfg, vocab, team_vocab, player_form_stats, best_epoch,
            ),
            ckpt_dir / "best.pt",
        )
        if final_payload is not None:
            torch.save(final_payload, ckpt_dir / "final.pt")

    return model, cfg, {"best_val_bce": best_val, "best_epoch": best_epoch,
                         "n_train": len(train_recs), "n_val": len(val_recs)}, {
        "vocab": vocab, "team_vocab": team_vocab,
        "tabular_stats": tabular_stats, "player_form_stats": player_form_stats,
    }


def predict_window(
    model, *, test_df, histories, statuses, calibration, scores, args,
    fitted, player_histories, game_odds=None,
) -> tuple[pd.DataFrame, dict]:
    test_recs = build_records(
        test_df, histories, vocab=fitted["vocab"], team_vocab=fitted["team_vocab"],
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
        return pd.DataFrame(), {"n_test": 0, "test_bce": float("nan"), "test_acc": float("nan")}
    loader = DataLoader(
        XfmrGameDataset(test_recs), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_xfmr,
    )
    probs = predict_probs(model, loader, args.device)
    df = pd.DataFrame({
        "game_id": [r.game_id for r in test_recs],
        "game_date": [r.game_date.strftime("%Y-%m-%d") for r in test_recs],
        "label_home_win": [int(r.label) for r in test_recs],
        "pred_home_win_prob": probs.tolist(),
    })
    eps = 1e-7
    p = np.clip(df["pred_home_win_prob"].to_numpy(), eps, 1 - eps)
    y = df["label_home_win"].to_numpy()
    bce = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    acc = float(((df["pred_home_win_prob"] >= 0.5).astype(int) == y).mean())
    return df, {"n_test": len(df), "test_bce": bce, "test_acc": acc}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {args.device}")
    print("[load] games + scores + status + calibration + exposures (one-time)")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    odds = load_game_odds(args.core_db, game_ids) if args.ev_weight > 0 else None
    if odds is not None:
        print(f"[odds] loaded for {len(odds)}/{len(game_ids)} games")
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)
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
    all_train_preds: list[pd.DataFrame] = []
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

        # Matchup rows only loaded for the train games of this window.
        matchup_rows_train = load_matchup_rows(
            args.matchup_db, [str(g) for g in train_df["game_id"].tolist()]
        )

        ckpt_dir = (out_dir / "windows" / window_start.date().isoformat()
                    if args.save_checkpoints else None)

        t0 = time.time()
        model, cfg, train_meta, fitted = train_one_window(
            train_df, val_df,
            histories=histories, statuses=statuses, calibration=calibration,
            scores=scores, matchup_rows_train=matchup_rows_train,
            player_histories=player_histories, args=args, device=args.device,
            game_odds=odds,
            ckpt_dir=ckpt_dir,
        )
        train_dt = time.time() - t0

        t1 = time.time()
        preds_df, eval_meta = predict_window(
            model, test_df=test_df,
            histories=histories, statuses=statuses, calibration=calibration,
            scores=scores, args=args,
            fitted=fitted, player_histories=player_histories,
            game_odds=odds,
        )
        pred_dt = time.time() - t1

        if not preds_df.empty:
            preds_df = preds_df.assign(window_start=window_start.date().isoformat())
            all_preds.append(preds_df)

        if args.save_train_preds:
            train_preds_df, _ = predict_window(
                model, test_df=full_train,
                histories=histories, statuses=statuses, calibration=calibration,
                scores=scores, args=args,
                fitted=fitted, player_histories=player_histories,
                game_odds=odds,
            )
            if not train_preds_df.empty:
                train_preds_df = train_preds_df.assign(
                    window_start=window_start.date().isoformat())
                all_train_preds.append(train_preds_df)

        row = {
            "window_start": window_start.date().isoformat(),
            "window_end": (window_end - pd.Timedelta(days=1)).date().isoformat(),
            "train_n": train_meta["n_train"],
            "val_n": train_meta["n_val"],
            "best_val_bce": train_meta["best_val_bce"],
            "best_epoch": train_meta["best_epoch"],
            **eval_meta,
            "train_secs": round(train_dt, 1),
            "pred_secs": round(pred_dt, 1),
        }
        metrics_rows.append(row)
        elapsed = time.time() - overall_t0
        eta = elapsed * (len(windows) - w_idx) / w_idx if w_idx > 0 else 0
        print(
            f"[w{w_idx:02d}/{len(windows)}] {window_start.date()}: "
            f"train_n={row['train_n']} val_bce={row['best_val_bce']:.4f}@ep{row['best_epoch']} "
            f"test_n={row['n_test']} test_bce={row['test_bce']:.4f} test_acc={row['test_acc']:.3f} "
            f"({train_dt:.0f}s+{pred_dt:.0f}s) elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
        )

        # Free model so memory doesn't grow across windows.
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Write outputs.
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    if args.save_train_preds:
        train_predictions = (pd.concat(all_train_preds, ignore_index=True)
                             if all_train_preds else pd.DataFrame())
        train_predictions.to_csv(out_dir / "predictions_train.csv", index=False)
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
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[done] {out_dir}: {overall}")


if __name__ == "__main__":
    main()
