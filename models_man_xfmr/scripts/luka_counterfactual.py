#!/usr/bin/env python3
"""Train v5 on all games < 2026-03-01, then predict on Lakers games in 2026-03
twice: once with real statuses (baseline) and once with Luka Dončić forced "Out".

Reuses train_one_window / predict_window from backtest_man_xfmr.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    load_game_player_status,
    load_game_scores,
    load_games,
    load_matchup_rows,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
)
from backtest_man_xfmr import (
    chrono_train_val,
    predict_window,
    train_one_window,
)

LAKERS_ID = "1610612747"
LUKA_ID = "1629029"
WINDOW_START = "2026-03-01"
WINDOW_END = "2026-04-01"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="luka_counterfactual_2026_03")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
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
    p.add_argument("--use-player-stats", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--alpha-poisson", type=float, default=10.0)
    p.add_argument("--beta-margin", type=float, default=0.0)
    p.add_argument("--ev-weight", type=float, default=0.0)
    p.add_argument("--ev-bet-sharpness", type=float, default=50.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoint", action="store_true",
                   help="Save trained model checkpoint to model.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {args.device}")
    print("[load] games + scores + status + calibration + exposures")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)
    if args.use_player_stats:
        print("[load] player histories")
        player_histories = load_player_histories(args.core_db)
    else:
        player_histories = None

    window_start = pd.Timestamp(WINDOW_START)
    window_end = pd.Timestamp(WINDOW_END)

    full_train = games_all[games_all["game_date"] < window_start].copy()
    test_all = games_all[
        (games_all["game_date"] >= window_start)
        & (games_all["game_date"] < window_end)
    ].copy()
    lakers_mask = (
        (test_all["home_team_id"].astype(str) == LAKERS_ID)
        | (test_all["away_team_id"].astype(str) == LAKERS_ID)
    )
    test_df = test_all[lakers_mask].copy().reset_index(drop=True)
    print(
        f"[window] train_n={len(full_train)} "
        f"test_lakers_n={len(test_df)} (of {len(test_all)} games in March 2026)"
    )

    train_df, val_df = chrono_train_val(full_train, args.val_frac)
    print(f"[split] train={len(train_df)} val={len(val_df)}")

    matchup_rows_train = load_matchup_rows(
        args.matchup_db, [str(g) for g in train_df["game_id"].tolist()]
    )

    t0 = time.time()
    print("[train] starting...")
    model, cfg, train_metrics, fitted = train_one_window(
        train_df, val_df,
        histories=histories, statuses=statuses, calibration=calibration, scores=scores,
        matchup_rows_train=matchup_rows_train,
        player_histories=player_histories, args=args, device=args.device,
        game_odds=None,
    )
    train_wall = time.time() - t0
    print(
        f"[train] done in {train_wall:.1f}s, "
        f"best_val_bce={train_metrics['best_val_bce']:.4f} "
        f"best_epoch={train_metrics['best_epoch']}"
    )

    if args.save_checkpoint:
        ckpt_path = out_dir / "model.pt"
        torch.save({
            "model_state": model.state_dict(),
            "cfg": asdict(cfg),
            "vocab": fitted["vocab"].player_to_idx,
            "team_vocab": fitted["team_vocab"].team_to_idx,
        }, ckpt_path)
        print(f"[save] checkpoint -> {ckpt_path}")

    # ----- baseline (real statuses) -----
    print("[predict] baseline (real statuses)")
    baseline_df, baseline_metrics = predict_window(
        model, test_df=test_df, histories=histories, statuses=statuses,
        calibration=calibration, scores=scores, args=args,
        fitted=fitted, player_histories=player_histories, game_odds=None,
    )

    # ----- counterfactual: force Luka OUT in every LAL March game -----
    statuses_cf = dict(statuses)
    prior_status_counts: dict[str, int] = {}
    for gid in test_df["game_id"].astype(str):
        key = (gid, LUKA_ID)
        prior = statuses_cf.get(key, "NotListed")
        prior_status_counts[prior] = prior_status_counts.get(prior, 0) + 1
        statuses_cf[key] = "Out"
    print(f"[predict] counterfactual (Luka OUT, overrode {len(test_df)} games); "
          f"prior statuses: {prior_status_counts}")
    cf_df, cf_metrics = predict_window(
        model, test_df=test_df, histories=histories, statuses=statuses_cf,
        calibration=calibration, scores=scores, args=args,
        fitted=fitted, player_histories=player_histories, game_odds=None,
    )

    # ----- merge + add per-team / per-game framing -----
    baseline_df = baseline_df.rename(columns={"pred_home_win_prob": "p_home_baseline"})
    cf_df = cf_df.rename(columns={"pred_home_win_prob": "p_home_luka_out"})
    merged = baseline_df.merge(cf_df[["game_id", "p_home_luka_out"]], on="game_id")

    # Pull abbreviations from core_db.games
    import sqlite3
    with sqlite3.connect(args.core_db) as conn:
        abbrev_df = pd.read_sql_query(
            "SELECT game_id, home_team_abbr, away_team_abbr FROM games", conn
        )
    abbrev_df["game_id"] = abbrev_df["game_id"].astype(str)

    meta = test_df[["game_id", "game_date", "home_team_id", "away_team_id"]].copy()
    meta["game_id"] = meta["game_id"].astype(str)
    meta = meta.merge(abbrev_df, on="game_id", how="left")
    meta["lakers_home"] = (meta["home_team_id"].astype(str) == LAKERS_ID).astype(int)
    meta["opponent_abbr"] = np.where(
        meta["lakers_home"] == 1, meta["away_team_abbr"], meta["home_team_abbr"]
    )
    meta = meta.drop(columns=["home_team_id", "away_team_id",
                              "home_team_abbr", "away_team_abbr"])
    merged["game_id"] = merged["game_id"].astype(str)
    merged = merged.merge(meta, on="game_id", how="left")
    if "game_date_x" in merged.columns:
        merged["game_date"] = merged["game_date_x"]
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_x") or c.endswith("_y")])

    merged["p_lakers_baseline"] = np.where(
        merged["lakers_home"] == 1,
        merged["p_home_baseline"], 1 - merged["p_home_baseline"],
    )
    merged["p_lakers_luka_out"] = np.where(
        merged["lakers_home"] == 1,
        merged["p_home_luka_out"], 1 - merged["p_home_luka_out"],
    )
    merged["delta_lakers"] = merged["p_lakers_luka_out"] - merged["p_lakers_baseline"]

    cols = [
        "game_id", "game_date", "lakers_home", "opponent_abbr",
        "label_home_win",
        "p_home_baseline", "p_home_luka_out",
        "p_lakers_baseline", "p_lakers_luka_out", "delta_lakers",
    ]
    merged = merged[cols].sort_values("game_date").reset_index(drop=True)

    csv_path = out_dir / "lakers_march_predictions.csv"
    merged.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    summary = {
        "window_start": str(window_start.date()),
        "window_end": str(window_end.date()),
        "lakers_team_id": LAKERS_ID,
        "luka_player_id": LUKA_ID,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test_lakers": int(len(test_df)),
        "train_wall_secs": float(train_wall),
        "best_val_bce": float(train_metrics["best_val_bce"]),
        "best_epoch": int(train_metrics["best_epoch"]),
        "baseline_test_bce": float(baseline_metrics["test_bce"]),
        "baseline_test_acc": float(baseline_metrics["test_acc"]),
        "cf_test_bce": float(cf_metrics["test_bce"]),
        "cf_test_acc": float(cf_metrics["test_acc"]),
        "mean_p_lakers_baseline": float(merged["p_lakers_baseline"].mean()),
        "mean_p_lakers_luka_out": float(merged["p_lakers_luka_out"].mean()),
        "mean_delta_lakers": float(merged["delta_lakers"].mean()),
        "median_delta_lakers": float(merged["delta_lakers"].median()),
        "min_delta_lakers": float(merged["delta_lakers"].min()),
        "max_delta_lakers": float(merged["delta_lakers"].max()),
        "prior_status_counts": prior_status_counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n[summary]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
