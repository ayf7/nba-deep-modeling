#!/usr/bin/env python3
"""Load a MAN-Xfmr checkpoint and emit per-game test predictions.

Outputs a CSV with columns (game_id, label_home_win, pred_home_win_prob) so it
can be fed directly into models_baseline/scripts/evaluate_betting_strategy.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from man_xfmr_common import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB,
    DEFAULT_LINEUP_DECAY,
    DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_PLAYER_FORM_DECAY,
    DEFAULT_PLAYER_FORM_LOOKBACK,
    PLAYER_FORM_DIM,
    PlayerFormStats,
    TeamVocab,
    Vocab,
    XfmrGameDataset,
    build_records,
    collate_xfmr,
    fit_tabular_stats,
    load_game_player_status,
    load_game_scores,
    load_games,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
)
from man_xfmr_model import ManXfmr, XfmrConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to best.pt or final.pt from train_man_xfmr.py")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write predictions.csv")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def main() -> None:
    args = parse_args()
    print(f"[device] {args.device}")
    print(f"[ckpt] loading {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = state["cfg"]
    cfg = XfmrConfig(**cfg_dict)
    vocab = Vocab(player_to_idx=state["vocab"])
    team_vocab = TeamVocab(team_to_idx=state["team_vocab"])

    use_player_stats = cfg.player_stat_dim > 0
    if use_player_stats:
        means = np.array(state["player_form_means"], dtype="float32")
        stds = np.array(state["player_form_stds"], dtype="float32")
        player_form_stats = PlayerFormStats(means=means, stds=stds)
    else:
        player_form_stats = None

    print(f"[cfg] vocab={vocab.size} teams={team_vocab.size} d={cfg.d} "
          f"player_stat_dim={cfg.player_stat_dim} two_stream={cfg.two_stream}")

    print("[load] games + scores + status + calibration + exposures")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)

    train_df, val_df, test_df = chrono_split(games_all, args.val_frac, args.test_frac)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    target_df = splits[args.split]
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"-> using {args.split} ({len(target_df)})")

    print("[tabular] re-fitting median/mean/std on train (not saved in ckpt)")
    tabular_stats = fit_tabular_stats(train_df)

    if use_player_stats:
        print("[player-form] loading per-player histories")
        player_histories = load_player_histories(args.core_db)
    else:
        player_histories = None

    print(f"[records] building {args.split}")
    recs = build_records(
        target_df, histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        matchup_rows=None,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
    )
    print(f"[records] n={len(recs)}")

    loader = DataLoader(
        XfmrGameDataset(recs), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_xfmr,
    )

    model = ManXfmr(cfg).to(args.device)
    model.load_state_dict(state["model_state"])
    model.eval()

    all_probs: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch, zero_tabular=False)
            probs = torch.sigmoid(out["win_logit"]).detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    if len(all_probs) != len(recs):
        raise RuntimeError(
            f"Prediction count mismatch: probs={len(all_probs)} recs={len(recs)}"
        )

    df = pd.DataFrame({
        "game_id": [r.game_id for r in recs],
        "label_home_win": [int(r.label) for r in recs],
        "pred_home_win_prob": [float(p) for p in all_probs],
        "game_date": [r.game_date.strftime("%Y-%m-%d") for r in recs],
    })

    eps = 1e-7
    p = np.clip(df["pred_home_win_prob"].to_numpy(), eps, 1 - eps)
    y = df["label_home_win"].to_numpy()
    bce = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    acc = float(((df["pred_home_win_prob"] >= 0.5).astype(int) == y).mean())
    print(f"[{args.split}] n={len(df)} bce={bce:.4f} acc={acc:.3f} "
          f"mean_prob={df['pred_home_win_prob'].mean():.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
