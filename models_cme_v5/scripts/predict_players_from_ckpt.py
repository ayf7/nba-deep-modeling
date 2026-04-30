#!/usr/bin/env python3
"""Load a saved checkpoint and produce player-level box predictions for test games."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
V5_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V5_SCRIPTS))

from cme_v5_common import (
    BOX_INDEX, BOX_TARGETS, K_BOX, K_PAIR, MAX_CAREER_YEAR, PAIR_TARGETS,
    DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    DEFAULT_PLAYER_DEBUT_DB,
    DEFAULT_PLAYER_DECISIONS_DB, DEFAULT_PLAYER_GAME_STATS_DB,
    DEFAULT_V5_FEATURES_DB,
    PLAYER_FORM_DIM, TABULAR_FEATURE_COLUMNS,
    GameDatasetV5, build_records_v5, build_team_vocab,
    build_vocab_from_records_v5, collate_v5,
    fit_player_form_stats, fit_tabular_stats,
    load_box_minutes_and_pace,
    load_game_odds, load_game_scores, load_games,
    load_matchup_rows_v2, load_minute_presence, load_play_decisions,
    load_player_debut, load_player_histories,
    load_player_game_stats, load_regulation_scores,
    load_team_exposures,
)
from cme_v5_model import CmeV5, CmeV5Config, gather_player_box


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--window", type=str, default=None,
                   help="Window date YYYY-MM-DD. If omitted, run all windows.")
    p.add_argument("--top-n", type=int, default=20,
                   help="Show top N players by actual points per game")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--player-game-stats-db", type=Path, default=DEFAULT_PLAYER_GAME_STATS_DB)
    p.add_argument("--use-player-form", action="store_true",
                   help="Feed per-player rolling form stats as input features")
    p.add_argument("--use-direct-player-head", action="store_true",
                   help="Use direct player head predictions instead of compositional")
    p.add_argument("--direct-head-separate-emb", action="store_true",
                   help="Use separate scoring embedding for direct head")
    p.add_argument("--direct-emb-dim", type=int, default=32,
                   help="Dimension of separate scoring embedding")
    args = p.parse_args()

    ckpt_dir = args.ckpt_dir
    ckpts = sorted(ckpt_dir.glob("window_*.pt"))
    if not ckpts:
        print(f"No checkpoints in {ckpt_dir}")
        return

    if args.window:
        ckpts = [c for c in ckpts if args.window in c.stem]
        if not ckpts:
            print(f"No checkpoint for window {args.window}")
            return

    print(f"[load] Loading shared data...")
    games_all = load_games(DEFAULT_FEATURES_DB)
    histories = load_team_exposures(args.core_db)
    all_gids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, all_gids)
    play_decisions = load_play_decisions(DEFAULT_PLAYER_DECISIONS_DB, all_gids)
    player_histories = load_player_histories(args.core_db) if args.use_player_form else None
    game_odds = load_game_odds(args.core_db, all_gids)
    player_first_season = load_player_debut(DEFAULT_PLAYER_DEBUT_DB, args.player_game_stats_db)
    minute_presence = load_minute_presence(DEFAULT_V5_FEATURES_DB)
    regulation_scores = load_regulation_scores(DEFAULT_V5_FEATURES_DB)

    all_rows = []

    for ckpt_path in ckpts:
        window_date = ckpt_path.stem.replace("window_", "")
        window_start = pd.Timestamp(window_date)
        window_end = window_start + pd.offsets.MonthBegin(1)

        train_block = games_all[games_all["game_date"] < window_start].copy()
        test_block = games_all[
            (games_all["game_date"] >= window_start) & (games_all["game_date"] < window_end)
        ].copy()
        if len(test_block) == 0:
            continue

        print(f"\n[window {window_date}] {len(test_block)} test games")

        all_block = games_all[games_all["game_date"] < window_end].copy()
        all_block_gids = [str(g) for g in all_block["game_id"].tolist()]
        test_gids = [str(g) for g in test_block["game_id"].tolist()]

        all_matchup = load_matchup_rows_v2(args.matchup_db, all_block_gids)
        test_matchup = load_matchup_rows_v2(args.matchup_db, test_gids)
        all_box = load_box_minutes_and_pace(args.player_game_stats_db, all_block_gids)
        test_box = load_box_minutes_and_pace(args.player_game_stats_db, test_gids)
        all_pl = load_player_game_stats(args.core_db, all_block_gids)
        test_pl = load_player_game_stats(args.core_db, test_gids)

        vocab = build_vocab_from_records_v5(
            all_block, histories, all_matchup, play_decisions,
            lookback_games=10, decay=0.85,
        )
        team_vocab = build_team_vocab(all_block)
        tabular_stats = fit_tabular_stats(all_block)

        player_form_stats = None
        if player_histories is not None:
            player_form_stats = fit_player_form_stats(
                player_histories, train_block,
                lookback_games=DEFAULT_PLAYER_FORM_LOOKBACK,
                decay=DEFAULT_PLAYER_FORM_DECAY,
            )

        common = dict(
            histories=histories, vocab=vocab, team_vocab=team_vocab,
            play_decisions=play_decisions, game_scores=scores,
            lookback_games=10, decay=0.85,
            tabular_stats=tabular_stats,
            player_first_season=player_first_season,
            player_histories=player_histories, player_form_stats=player_form_stats,
            player_form_lookback=DEFAULT_PLAYER_FORM_LOOKBACK,
            player_form_decay=DEFAULT_PLAYER_FORM_DECAY,
            game_odds=game_odds,
            minute_presence=minute_presence,
            regulation_scores=regulation_scores,
        )
        test_recs = build_records_v5(
            test_block, matchup_rows=test_matchup,
            box_minutes_pace=test_box,
            player_game_stats=test_pl, **common,
        )

        import cme_v5_common as _cv5c
        tabular_dim = len(_cv5c.TABULAR_FEATURE_COLUMNS)
        sd = torch.load(ckpt_path, map_location=args.device, weights_only=True)
        cfg = CmeV5Config(
            vocab_size=vocab.size, num_teams=team_vocab.size,
            d=128, n_heads=8,
            tabular_dim=tabular_dim,
            player_stat_dim=PLAYER_FORM_DIM if args.use_player_form else 0,
            max_career_year=MAX_CAREER_YEAR,
            use_direct_player_head=args.use_direct_player_head,
            direct_head_separate_emb=args.direct_head_separate_emb,
            direct_emb_dim=args.direct_emb_dim,
        )
        model = CmeV5(cfg).to(args.device)
        model_sd = model.state_dict()
        for k in sd:
            if k in model_sd and sd[k].shape != model_sd[k].shape:
                # checkpoint is smaller — copy into the model's larger tensor
                slices = tuple(slice(0, s) for s in sd[k].shape)
                model_sd[k][slices] = sd[k]
                sd[k] = model_sd[k]
        model.load_state_dict(sd, strict=False)
        model.eval()

        idx2pid = {v: k for k, v in vocab.player_to_idx.items()}

        loader = DataLoader(GameDatasetV5(test_recs), batch_size=64,
                            shuffle=False, collate_fn=collate_v5)

        pts_idx = BOX_INDEX["pts"]
        rec_offset = 0

        with torch.no_grad():
            for batch in loader:
                B = batch["label"].size(0)
                batch_recs = test_recs[rec_offset:rec_offset + B]
                rec_offset += B

                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch)

                sup_game = batch["sup_pl_game"]
                sup_side = batch["sup_pl_side"]
                sup_slot = batch["sup_pl_slot"]
                sup_y = batch["sup_pl_y"]
                if args.use_direct_player_head and "home_box_direct" in out:
                    out_for_gather = {**out, "home_box": out["home_box_direct"], "away_box": out["away_box_direct"]}
                else:
                    out_for_gather = out
                pred_box = gather_player_box(out_for_gather, sup_game, sup_side, sup_slot)

                for j in range(pred_box.size(0)):
                    gi = sup_game[j].item()
                    rec = batch_recs[gi]
                    side_val = sup_side[j].item()
                    slot_val = sup_slot[j].item()
                    if side_val == 0:
                        pid = idx2pid.get(rec.home_player_idx[slot_val], "?")
                    else:
                        pid = idx2pid.get(rec.away_player_idx[slot_val], "?")
                    pred_pts = pred_box[j, pts_idx].item()
                    actual_pts = sup_y[j, pts_idx].item()
                    all_rows.append(dict(
                        window=window_date, game_id=rec.game_id, player_id=pid,
                        side="home" if side_val == 0 else "away",
                        pred_pts=pred_pts, actual_pts=actual_pts,
                    ))

        print(f"  collected {len(all_rows)} player-game predictions so far")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No predictions collected.")
        return

    print(f"\n{'='*80}")
    print(f"PLAYER POINTS PREDICTIONS — {len(df)} player-games across {df['window'].nunique()} windows")
    print(f"{'='*80}")

    pts_mae = (df["pred_pts"] - df["actual_pts"]).abs().mean()
    pts_bias = (df["pred_pts"] - df["actual_pts"]).mean()
    pts_corr = df[["pred_pts", "actual_pts"]].corr().iloc[0, 1]
    print(f"\nOverall:  MAE={pts_mae:.2f}  Bias={pts_bias:+.2f}  Corr={pts_corr:.3f}")

    if "min" in BOX_INDEX:
        min_mae = (df["pred_min"] - df["actual_min"]).abs().mean()
        min_bias = (df["pred_min"] - df["actual_min"]).mean()
        print(f"Minutes:  MAE={min_mae:.2f}  Bias={min_bias:+.2f}")

    # By PPG tier
    print(f"\nBy actual PPG tier:")
    df["ppg_tier"] = pd.cut(df["actual_pts"], bins=[0, 5, 10, 15, 20, 25, 100],
                            labels=["0-5", "5-10", "10-15", "15-20", "20-25", "25+"])
    for tier, g in df.groupby("ppg_tier", observed=True):
        mae = (g["pred_pts"] - g["actual_pts"]).abs().mean()
        bias = (g["pred_pts"] - g["actual_pts"]).mean()
        print(f"  {tier:>5s}:  n={len(g):5d}  pred_avg={g['pred_pts'].mean():5.1f}  "
              f"actual_avg={g['actual_pts'].mean():5.1f}  MAE={mae:.2f}  bias={bias:+.2f}")

    # Top players by volume
    player_agg = df.groupby("player_id").agg(
        n=("actual_pts", "count"),
        actual_ppg=("actual_pts", "mean"),
        pred_ppg=("pred_pts", "mean"),
    ).sort_values("actual_ppg", ascending=False)

    print(f"\nTop {args.top_n} players by actual PPG (min 10 games):")
    print(f"{'player_id':<12s}  {'n':>4s}  {'actual':>7s}  {'pred':>7s}  {'diff':>7s}")
    print("-" * 45)
    shown = 0
    for pid, row in player_agg.iterrows():
        if row["n"] < 10:
            continue
        diff = row["pred_ppg"] - row["actual_ppg"]
        print(f"{pid:<12}  {int(row['n']):4d}  {row['actual_ppg']:7.1f}  {row['pred_ppg']:7.1f}  {diff:+7.1f}")
        shown += 1
        if shown >= args.top_n:
            break


if __name__ == "__main__":
    main()
