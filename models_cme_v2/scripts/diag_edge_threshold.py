#!/usr/bin/env python3
"""Diagnostic: sweep edge-threshold policy over the frozen CME-v2 backbone.

For each game (with odds), get:
    p_home   = σ(win_logit)
    devigged = (1/h_dec) / (1/h_dec + 1/a_dec)
    edge     = p_home - devigged

Policy at margin m:
    bet home  if  edge >  m
    bet away  if  edge < -m
    skip      otherwise

For each split and m, report:
    n_bets, bet_rate, ROI/bet, ROI/game, per-side counts and win rates.

If even the best m on val gives -EV (after vig), the win head has no
exploitable edge over the implied probability — and decision-head
fine-tuning can't rescue it.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    DEFAULT_CALIBRATION_PATH, DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB, DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    GameDatasetV2, TeamVocab, Vocab, build_records_v2, collate_v2,
    fit_tabular_stats,
    load_game_odds, load_game_player_status, load_game_scores, load_games,
    load_matchup_rows_v2, load_player_game_stats, load_player_histories,
    load_status_calibration, load_team_exposures,
)
from cme_v2_model import CmeV2, CmeV2Config  # noqa: E402
from man_xfmr_common import PlayerFormStats  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v2" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=DEFAULT_OUTPUT_ROOT / "run_v2_dd_s_tt" / "best.pt")
    p.add_argument("--out-json", type=Path,
                   default=DEFAULT_OUTPUT_ROOT / "diag_edge_threshold.json")

    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--margins", type=str,
                   default="0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.15,0.20")
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


@torch.no_grad()
def collect_split(model: CmeV2, loader: DataLoader, device: str):
    """Return dict of numpy arrays for games with odds."""
    p_homes, h_decs, a_decs, labels = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        mask = batch["has_odds"] > 0.5
        if not mask.any():
            continue
        out = model(batch)
        p_h = torch.sigmoid(out["win_logit"])[mask]
        p_homes.append(p_h.cpu().numpy())
        h_decs.append(batch["home_dec_odds"][mask].cpu().numpy())
        a_decs.append(batch["away_dec_odds"][mask].cpu().numpy())
        labels.append(batch["label"][mask].cpu().numpy())
    return {
        "p_home": np.concatenate(p_homes),
        "h_dec": np.concatenate(h_decs),
        "a_dec": np.concatenate(a_decs),
        "label": np.concatenate(labels),
    }


def evaluate_thresholds(arrs: dict, margins: list[float]) -> list[dict]:
    p = arrs["p_home"]
    h = arrs["h_dec"]
    a = arrs["a_dec"]
    y = arrs["label"]
    n = len(p)
    devigged_h = (1.0 / h) / (1.0 / h + 1.0 / a)
    edge = p - devigged_h
    r_home = np.where(y > 0.5, h - 1.0, -1.0)
    r_away = np.where(y > 0.5, -1.0, a - 1.0)
    home_won = (y > 0.5).astype(float)
    away_won = 1.0 - home_won

    rows = []
    for m in margins:
        bet_home = edge > m
        bet_away = edge < -m
        bet_mask = bet_home | bet_away
        n_bet = int(bet_mask.sum())
        n_h = int(bet_home.sum())
        n_a = int(bet_away.sum())
        realized = np.where(bet_home, r_home, np.where(bet_away, r_away, 0.0))
        sum_realized = float(realized.sum())
        sum_home = float(realized[bet_home].sum()) if n_h else 0.0
        sum_away = float(realized[bet_away].sum()) if n_a else 0.0
        home_wins = int((bet_home & (home_won > 0.5)).sum())
        away_wins = int((bet_away & (away_won > 0.5)).sum())
        rows.append({
            "margin": m,
            "n": n, "n_bet": n_bet, "n_home": n_h, "n_away": n_a,
            "bet_rate": n_bet / n if n else 0.0,
            "roi_bet": sum_realized / n_bet if n_bet else 0.0,
            "roi_game": sum_realized / n if n else 0.0,
            "roi_home": sum_home / n_h if n_h else 0.0,
            "roi_away": sum_away / n_a if n_a else 0.0,
            "home_winrate": home_wins / n_h if n_h else 0.0,
            "away_winrate": away_wins / n_a if n_a else 0.0,
        })
    return rows


def fmt_table(name: str, rows: list[dict]) -> str:
    head = (f"\n=== {name} (n={rows[0]['n']}) ===\n"
            f"{'m':>5} {'n_bet':>6} {'bet%':>6} {'n_h':>5} {'n_a':>5} "
            f"{'ROI/bet':>9} {'ROI/gm':>9} {'ROI_h':>8} {'ROI_a':>8} "
            f"{'wr_h':>6} {'wr_a':>6}")
    lines = [head]
    for r in rows:
        lines.append(
            f"{r['margin']:>5.2f} {r['n_bet']:>6d} {r['bet_rate']*100:>5.1f}% "
            f"{r['n_home']:>5d} {r['n_away']:>5d} "
            f"{r['roi_bet']*100:>+8.2f}% {r['roi_game']*100:>+8.2f}% "
            f"{r['roi_home']*100:>+7.2f}% {r['roi_away']*100:>+7.2f}% "
            f"{r['home_winrate']*100:>5.1f}% {r['away_winrate']*100:>5.1f}%"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = args.device
    print(f"[device] {device}")
    print(f"[ckpt]   {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = CmeV2Config(**state["cfg"])
    vocab = Vocab(player_to_idx=state["vocab"])
    team_vocab = TeamVocab(team_to_idx=state["team_vocab"])
    use_player_stats = cfg.player_stat_dim > 0
    player_form_stats = (
        PlayerFormStats(
            means=np.array(state["player_form_means"], dtype=np.float64),
            stds=np.array(state["player_form_stds"], dtype=np.float64),
        ) if use_player_stats else None
    )

    print("[load] games + exposures + matchup rows + status + odds")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
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

    print("[load] matchup rows + player game stats")
    train_matchup = load_matchup_rows_v2(args.matchup_db, train_gids)
    val_matchup = load_matchup_rows_v2(args.matchup_db, val_gids)
    test_matchup = load_matchup_rows_v2(args.matchup_db, test_gids)
    train_pl = load_player_game_stats(args.core_db, train_gids)
    val_pl = load_player_game_stats(args.core_db, val_gids)
    test_pl = load_player_game_stats(args.core_db, test_gids)

    tabular_stats = fit_tabular_stats(train_df)
    player_histories = load_player_histories(args.core_db) if use_player_stats else None

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
    train_recs = build_records_v2(train_df, matchup_rows=train_matchup,
                                  player_game_stats=train_pl, **common)
    val_recs = build_records_v2(val_df, matchup_rows=val_matchup,
                                player_game_stats=val_pl, **common)
    test_recs = build_records_v2(test_df, matchup_rows=test_matchup,
                                 player_game_stats=test_pl, **common)

    train_loader = DataLoader(GameDatasetV2(train_recs), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_v2)
    val_loader = DataLoader(GameDatasetV2(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_v2)
    test_loader = DataLoader(GameDatasetV2(test_recs), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_v2)

    print("[model] loading state (FROZEN)")
    model = CmeV2(cfg).to(device)
    model.load_state_dict(state["model_state"], strict=False)
    model.eval()

    margins = [float(x) for x in args.margins.split(",")]

    print("[infer] collecting val + test (and train for context)")
    train_arrs = collect_split(model, train_loader, device)
    val_arrs = collect_split(model, val_loader, device)
    test_arrs = collect_split(model, test_loader, device)

    train_rows = evaluate_thresholds(train_arrs, margins)
    val_rows = evaluate_thresholds(val_arrs, margins)
    test_rows = evaluate_thresholds(test_arrs, margins)

    print(fmt_table("train", train_rows))
    print(fmt_table("val", val_rows))
    print(fmt_table("test", test_rows))

    # Also: pure "bet whichever side has higher σ(p_win)" (no edge filter, no devig)
    def naive_pwin(arrs):
        p, h, a, y = arrs["p_home"], arrs["h_dec"], arrs["a_dec"], arrs["label"]
        bet_home = p > 0.5
        bet_away = ~bet_home
        r_home = np.where(y > 0.5, h - 1.0, -1.0)
        r_away = np.where(y > 0.5, -1.0, a - 1.0)
        realized = np.where(bet_home, r_home, r_away)
        return {
            "n": int(len(p)),
            "n_home": int(bet_home.sum()),
            "n_away": int(bet_away.sum()),
            "roi_bet": float(realized.mean()),
        }
    print("\n=== naive (always bet σ(p)>0.5 side) ===")
    for name, arrs in [("train", train_arrs), ("val", val_arrs), ("test", test_arrs)]:
        n = naive_pwin(arrs)
        print(f"  {name}: n={n['n']} home={n['n_home']} away={n['n_away']} "
              f"ROI/bet={n['roi_bet']*100:+.2f}%")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "margins": margins,
            "train": train_rows, "val": val_rows, "test": test_rows,
        }, f, indent=2, default=float)
    print(f"\n[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
