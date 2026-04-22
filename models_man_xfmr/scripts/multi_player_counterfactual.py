#!/usr/bin/env python3
"""Multi-player counterfactual for Lakers March 2026 games.

Loads the trained checkpoint from `luka_counterfactual_2026_03/model.pt`,
re-fits tabular_stats and player_form_stats on the same train slice
(deterministic), and runs baseline + per-player counterfactual inference
for the players listed in `--players`. Each player is forced to status
"Out" for every Lakers March 2026 game.

Outputs:
- predictions_<player>.csv per player
- combined.csv with p_lakers_baseline + p_lakers_<player>_out columns
- summary.json with per-player aggregate deltas
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
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
    PLAYER_FORM_DIM,
    TABULAR_FEATURE_COLUMNS,
    Vocab,
    TeamVocab,
    fit_player_form_stats,
    fit_tabular_stats,
    load_game_player_status,
    load_game_scores,
    load_games,
    load_matchup_rows,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
)
from man_xfmr_model import ManXfmr, XfmrConfig
from backtest_man_xfmr import chrono_train_val, predict_window

LAKERS_ID = "1610612747"
WINDOW_START = "2026-03-01"
WINDOW_END = "2026-04-01"

DEFAULT_PLAYERS = "luka:1629029,lebron:2544,ayton:1629028"


def parse_players(s: str) -> list[tuple[str, list[str]]]:
    """name:pid[+pid+...] entries, comma-separated. '+' groups players into a joint cf."""
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        name, pids = item.split(":")
        pid_list = [p.strip() for p in pids.split("+") if p.strip()]
        out.append((name.strip(), pid_list))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="multi_player_cf_2026_03")
    p.add_argument(
        "--checkpoint", type=Path,
        default=DEFAULT_OUTPUT_ROOT / "luka_counterfactual_2026_03" / "model.pt",
        help="Path to trained v5 model.pt",
    )
    p.add_argument("--players", type=str, default=DEFAULT_PLAYERS,
                   help="Comma-separated name:player_id pairs")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)
    p.add_argument("--use-player-stats", action="store_true", default=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    players = parse_players(args.players)
    print(f"[device] {args.device}")
    print(f"[players] {players}")

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
    print(f"[window] train_n={len(full_train)} test_lakers_n={len(test_df)}")

    train_df, val_df = chrono_train_val(full_train, args.val_frac)

    print("[stats] refitting tabular + player-form stats on train_df (deterministic)")
    tabular_stats = fit_tabular_stats(train_df)
    if args.use_player_stats:
        player_form_stats = fit_player_form_stats(
            player_histories, train_df,
            lookback_games=args.player_form_lookback,
            decay=args.player_form_decay,
        )
    else:
        player_form_stats = None

    print(f"[load] checkpoint <- {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = XfmrConfig(**ckpt["cfg"])
    model = ManXfmr(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    vocab = Vocab(player_to_idx=ckpt["vocab"])
    team_vocab = TeamVocab(team_to_idx=ckpt["team_vocab"])
    print(f"[vocab] players={vocab.size} teams={team_vocab.size}")

    fitted = {
        "vocab": vocab, "team_vocab": team_vocab,
        "tabular_stats": tabular_stats, "player_form_stats": player_form_stats,
    }

    # Baseline once
    t0 = time.time()
    print("[predict] baseline")
    baseline_df, baseline_metrics = predict_window(
        model, test_df=test_df, histories=histories, statuses=statuses,
        calibration=calibration, scores=scores, args=args,
        fitted=fitted, player_histories=player_histories, game_odds=None,
    )
    baseline_df = baseline_df.rename(columns={"pred_home_win_prob": "p_home_baseline"})
    print(f"[predict] baseline done {time.time()-t0:.1f}s "
          f"bce={baseline_metrics['test_bce']:.4f} acc={baseline_metrics['test_acc']:.3f}")

    # Per-player counterfactuals
    per_player: dict[str, pd.DataFrame] = {}
    per_player_summary: dict[str, dict] = {}
    for name, pid_list in players:
        statuses_cf = dict(statuses)
        prior_counts: dict[str, dict[str, int]] = {pid: {} for pid in pid_list}
        for gid in test_df["game_id"].astype(str):
            for pid in pid_list:
                key = (gid, pid)
                prior = statuses_cf.get(key, "NotListed")
                prior_counts[pid][prior] = prior_counts[pid].get(prior, 0) + 1
                statuses_cf[key] = "Out"
        t0 = time.time()
        print(f"[predict] cf {name} ({'+'.join(pid_list)}) prior={prior_counts}")
        cf_df, cf_metrics = predict_window(
            model, test_df=test_df, histories=histories, statuses=statuses_cf,
            calibration=calibration, scores=scores, args=args,
            fitted=fitted, player_histories=player_histories, game_odds=None,
        )
        cf_df = cf_df.rename(columns={"pred_home_win_prob": f"p_home_{name}_out"})
        per_player[name] = cf_df
        per_player_summary[name] = {
            "player_ids": pid_list,
            "prior_status_counts": prior_counts,
            "wall_secs": time.time() - t0,
            "test_bce": cf_metrics["test_bce"],
            "test_acc": cf_metrics["test_acc"],
        }
        print(f"[predict] cf {name} done {time.time()-t0:.1f}s "
              f"bce={cf_metrics['test_bce']:.4f} acc={cf_metrics['test_acc']:.3f}")

    # Merge all into a single CSV
    merged = baseline_df.copy()
    merged["game_id"] = merged["game_id"].astype(str)
    for name, cf_df in per_player.items():
        cf_df = cf_df.copy()
        cf_df["game_id"] = cf_df["game_id"].astype(str)
        merged = merged.merge(
            cf_df[["game_id", f"p_home_{name}_out"]], on="game_id", how="left"
        )

    # team abbrev metadata
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
    meta = meta[["game_id", "lakers_home", "opponent_abbr"]]
    merged = merged.merge(meta, on="game_id", how="left")

    # Lakers-side probs
    merged["p_lakers_baseline"] = np.where(
        merged["lakers_home"] == 1,
        merged["p_home_baseline"], 1 - merged["p_home_baseline"],
    )
    for name, _ in players:
        merged[f"p_lakers_{name}_out"] = np.where(
            merged["lakers_home"] == 1,
            merged[f"p_home_{name}_out"], 1 - merged[f"p_home_{name}_out"],
        )
        merged[f"delta_{name}"] = merged[f"p_lakers_{name}_out"] - merged["p_lakers_baseline"]

    # Order columns
    base_cols = ["game_id", "game_date", "lakers_home", "opponent_abbr",
                 "label_home_win", "p_home_baseline", "p_lakers_baseline"]
    cf_cols = []
    for name, _pids in players:
        cf_cols += [f"p_home_{name}_out", f"p_lakers_{name}_out", f"delta_{name}"]
    merged = merged[base_cols + cf_cols].sort_values("game_date").reset_index(drop=True)

    csv_path = out_dir / "lakers_march_multi_player.csv"
    merged.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    summary = {
        "window_start": str(window_start.date()),
        "window_end": str(window_end.date()),
        "checkpoint": str(args.checkpoint),
        "n_test_lakers": int(len(test_df)),
        "baseline_test_bce": float(baseline_metrics["test_bce"]),
        "baseline_test_acc": float(baseline_metrics["test_acc"]),
        "mean_p_lakers_baseline": float(merged["p_lakers_baseline"].mean()),
        "players": [],
    }
    for name, pid_list in players:
        deltas = merged[f"delta_{name}"]
        summary["players"].append({
            "name": name,
            "player_ids": pid_list,
            "mean_delta": float(deltas.mean()),
            "median_delta": float(deltas.median()),
            "min_delta": float(deltas.min()),
            "max_delta": float(deltas.max()),
            "mean_p_lakers_out": float(merged[f"p_lakers_{name}_out"].mean()),
            **{k: v for k, v in per_player_summary[name].items() if k != "wall_secs"},
        })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n[summary]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
