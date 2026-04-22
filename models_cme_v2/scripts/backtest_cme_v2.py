#!/usr/bin/env python3
"""Expanding-window monthly backtest for CME-v2.

Mirrors the methodology used by models_baseline/scripts/backtest_baselines.py:
for each month-start after --initial-train-end, train on all games strictly
before that month and predict every game inside the month. Predictions are
stitched into a single CSV with the same schema as the baseline backtests
(game_id, game_date, home_team_id, away_team_id, label_home_win,
pred_home_win_prob, window_start) so it can be fed into
evaluate_betting_strategy.py.

Defaults reproduce the run_v2_dd_s_tt config (small backbone + CLS team token
+ direct win head, win-only loss). Within each window the most recent
--val-frac-of-train fraction is held out chronologically for early stopping.
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
V2_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_SCRIPTS))

from cme_v2_common import (  # noqa: E402
    K_BOX, K_PAIR,
    DEFAULT_CALIBRATION_PATH, DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB, DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    PLAYER_FORM_DIM, TABULAR_FEATURE_COLUMNS,
    GameDatasetV2, build_records_v2, build_team_vocab,
    build_vocab_from_records_v2, collate_v2,
    fit_tabular_stats,
    load_game_odds, load_game_player_status, load_game_scores, load_games,
    load_matchup_rows_v2, load_player_game_stats,
    load_status_calibration, load_team_exposures,
)
from cme_v2_model import CmeV2, CmeV2Config  # noqa: E402
from train_cme_v2 import (  # noqa: E402
    default_box_weights, default_pair_weights, run_epoch,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v2" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="backtest_s_tt")
    p.add_argument("--initial-train-end", default="2023-12-31",
                   help="First test window is the month strictly after this date.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="If set, only run the first N windows (for smoke tests).")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--val-frac-of-train", type=float, default=0.15,
                   help="Within each window's train block, hold out the most recent X%% as val.")
    # s_tt backbone config
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
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--no-cyclic", action="store_true",
                   help="Drop the 12 cyc_* columns from the tabular feature set (pre-cyclic baseline).")
    # loss (s_tt = win-only)
    p.add_argument("--team-w", type=float, default=0.0)
    p.add_argument("--player-w", type=float, default=0.0)
    p.add_argument("--pair-w", type=float, default=0.0)
    p.add_argument("--win-w", type=float, default=1.0)
    # optim (matches train_cme_v2 defaults)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
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
    model: CmeV2, records: list, batch_size: int, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(GameDatasetV2(records), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_v2)
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


def _build_ckpt_payload(model, cfg, vocab, team_vocab, box_weights, pair_weights,
                        team_w, player_w, pair_w, epoch):
    return {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": asdict(cfg),
        "vocab": vocab.player_to_idx,
        "team_vocab": team_vocab.team_to_idx,
        "player_form_means": None,
        "player_form_stds": None,
        "box_weights": box_weights.tolist(),
        "pair_weights": pair_weights.tolist(),
        "loss_level_weights": {"team": team_w, "player": player_w, "pair": pair_w},
        "epoch": epoch,
    }


def train_one_window(
    args: argparse.Namespace,
    train_fit_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    histories, scores, statuses, calibration, game_odds,
    matchup_db: Path, core_db: Path,
    ckpt_dir: Path | None = None,
) -> tuple[CmeV2, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    vocab = build_vocab_from_records_v2(
        train_fit_df, histories, train_matchup,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(train_fit_df)
    tabular_stats = fit_tabular_stats(train_fit_df)

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
    train_recs = build_records_v2(
        train_fit_df, matchup_rows=train_matchup, player_game_stats=train_pl, **common,
    )
    val_recs = build_records_v2(
        val_df, matchup_rows=val_matchup, player_game_stats=val_pl, **common,
    )
    test_recs = build_records_v2(
        test_df, matchup_rows=test_matchup, player_game_stats=test_pl, **common,
    )

    train_loader = DataLoader(GameDatasetV2(train_recs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_v2)
    val_loader = DataLoader(GameDatasetV2(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_v2)
    test_loader = DataLoader(GameDatasetV2(test_recs), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_v2)

    import cme_v2_common as _cv2c
    tabular_dim = 0 if args.no_tabular else len(_cv2c.TABULAR_FEATURE_COLUMNS)
    cfg = CmeV2Config(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, n_heads=args.n_heads,
        n_self_layers=args.n_self_layers, n_cross_layers=args.n_cross_layers,
        pair_hidden=args.pair_hidden, player_hidden=args.player_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        tabular_dim=tabular_dim, team_emb_dim=args.team_emb_dim,
        player_stat_dim=0,
        init_scale=args.init_scale,
        use_direct_win_head=True,
        use_team_token=True,
        use_decision_head=False,
    )
    model = CmeV2(cfg).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    box_weights = default_box_weights()
    pair_weights = default_pair_weights()

    def lr_lambda(epoch_idx: int) -> float:
        if args.warmup_epochs > 0 and epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / args.warmup_epochs
        denom = max(1, args.epochs - args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    epoch_kwargs = dict(
        box_weights=box_weights, pair_weights=pair_weights,
        team_w=args.team_w, player_w=args.player_w, pair_w=args.pair_w,
        win_w=args.win_w, margin_nll_w=0.0, decision_w=0.0,
    )

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict] = []
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        t0 = time.time()
        tr = run_epoch(model, train_loader, device=args.device, optim=optim, **epoch_kwargs)
        va = run_epoch(model, val_loader, device=args.device, optim=None, **epoch_kwargs)
        scheduler.step()
        dt = time.time() - t0
        history.append({"epoch": epoch, "secs": dt,
                        "train_bce": tr["bce"], "val_bce": va["bce"],
                        "train_acc": tr["acc"], "val_acc": va["acc"]})
        if va["bce"] < best_val - 1e-5:
            best_val = va["bce"]
            best_epoch = epoch
            epochs_since_best = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                break

    if ckpt_dir is not None and args.save_final:
        final_payload = _build_ckpt_payload(
            model, cfg, vocab, team_vocab, box_weights, pair_weights,
            args.team_w, args.player_w, args.pair_w, last_epoch,
        )
    else:
        final_payload = None

    if best_state is not None:
        model.load_state_dict(best_state)

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            _build_ckpt_payload(
                model, cfg, vocab, team_vocab, box_weights, pair_weights,
                args.team_w, args.player_w, args.pair_w, best_epoch,
            ),
            ckpt_dir / "best.pt",
        )
        if final_payload is not None:
            torch.save(final_payload, ckpt_dir / "final.pt")

    gids, probs, h_pts, a_pts, margin_mu = predict_window(
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
    return model, window_info, gids, probs, h_pts, a_pts, margin_mu


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_cyclic:
        import man_xfmr_common as _mxc
        import cme_v2_common as _cv2c
        trimmed = tuple(c for c in _mxc.TABULAR_FEATURE_COLUMNS if not c.startswith("cyc_"))
        _mxc.TABULAR_FEATURE_COLUMNS = trimmed
        _cv2c.TABULAR_FEATURE_COLUMNS = trimmed
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
    print(f"[load] n_games={len(games_all)} odds={len(game_odds)}")

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
            _model, info, gids, probs, h_pts, a_pts, margin_mu = train_one_window(
                args, train_fit, val, test_block, histories, scores, statuses,
                calibration, game_odds, args.matchup_db, args.core_db,
                ckpt_dir=ckpt_dir,
            )
        except Exception as exc:
            print(f"[ERROR] window {window_start.date()}: {exc}")
            raise
        dt = time.time() - t0

        gid_to_idx = {g: i for i, g in enumerate(gids)}
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
            "best_val_bce": info["best_val_bce"],
            "epochs_run": info["epochs_run"],
            "test_bce": bce,
            "test_acc": acc,
            "secs": dt,
        }
        window_metrics.append(win_metrics)
        print(f"[window {wi+1}/{len(windows)}] best_ep={info['best_epoch']:2d} "
              f"val_bce={info['best_val_bce']:.4f} test_bce={bce:.4f} test_acc={acc:.3f} "
              f"({dt:.0f}s)")

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
        predictions_df_partial.to_csv(out_dir / "predictions.csv", index=False)
        pd.DataFrame(window_metrics).to_csv(out_dir / "window_metrics.csv", index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    labels = predictions_df["label_home_win"].to_numpy(dtype=float)
    probs = predictions_df["pred_home_win_prob"].to_numpy(dtype=float)
    overall_bce = float(-np.mean(labels * np.log(np.clip(probs, 1e-7, 1-1e-7))
                                  + (1 - labels) * np.log(np.clip(1 - probs, 1e-7, 1-1e-7))))
    overall_acc = float(np.mean((probs > 0.5).astype(float) == labels))
    overall_brier = float(np.mean((probs - labels) ** 2))

    overall = {
        "n": int(len(predictions_df)),
        "bce": overall_bce,
        "acc": overall_acc,
        "brier": overall_brier,
        "mean_prob": float(probs.mean()),
        "std_prob": float(probs.std()),
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
