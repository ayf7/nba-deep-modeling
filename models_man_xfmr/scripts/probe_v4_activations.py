#!/usr/bin/env python3
"""Probe a trained MAN-Xfmr (v4) checkpoint.

Reports test BCE/acc under three forward modes:
  - full       (lineup + tabular)
  - lineup-off (z_home/z_away zeroed)
  - tabular-off (tabular features zeroed)

Also prints mean-absolute activation magnitudes for the lineup vector vs
tabular vector, so we can see which path the head is actually using.
"""

from __future__ import annotations

import argparse
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
    TABULAR_FEATURE_COLUMNS,
    XfmrGameDataset,
    build_records,
    collate_xfmr,
    fit_tabular_stats,
    load_game_player_status,
    load_game_scores,
    load_games,
    load_status_calibration,
    load_team_exposures,
)
from man_xfmr_model import ManXfmr, XfmrConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="best.pt or final.pt")
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


@torch.no_grad()
def eval_one(
    model: ManXfmr,
    loader: DataLoader,
    *,
    device: str,
    zero_lineup: bool = False,
    zero_tabular: bool = False,
) -> dict:
    model.eval()
    n = 0
    bce_sum = 0.0
    correct = 0
    abs_z_h = 0.0
    abs_z_a = 0.0
    abs_tab = 0.0
    abs_logit = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, zero_lineup=zero_lineup, zero_tabular=zero_tabular)
        bce = F.binary_cross_entropy_with_logits(
            out["win_logit"], batch["label"], reduction="sum"
        )
        bs = batch["label"].size(0)
        n += bs
        bce_sum += bce.item()
        preds = (torch.sigmoid(out["win_logit"]) > 0.5).float()
        correct += (preds == batch["label"]).sum().item()
        abs_z_h += out["z_home_off"].abs().sum().item()
        abs_z_a += out["z_away_off"].abs().sum().item()
        if "tabular" in batch:
            abs_tab += batch["tabular"].abs().sum().item()
        abs_logit += out["win_logit"].abs().sum().item()
    d_z = out["z_home_off"].size(-1)
    d_tab = batch["tabular"].size(-1) if "tabular" in batch else 0
    return {
        "n": n,
        "bce": bce_sum / n,
        "acc": correct / n,
        "mean_abs_z_h": abs_z_h / (n * d_z),
        "mean_abs_z_a": abs_z_a / (n * d_z),
        "mean_abs_tab": abs_tab / (n * d_tab) if d_tab else 0.0,
        "mean_abs_logit": abs_logit / n,
    }


def main() -> None:
    args = parse_args()
    print(f"[device] {args.device}")
    print(f"[ckpt] {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location=args.device)
    cfg_dict = state["cfg"]
    cfg = XfmrConfig(**cfg_dict)
    print(f"[cfg] {cfg}")

    print("[load] games + exposures + matchup rows + status + calibration")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)

    train_df, val_df, test_df = chrono_split(games_all, args.val_frac, args.test_frac)
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("[tabular] fitting median/mean/std on train")
    tabular_stats = fit_tabular_stats(train_df)

    # Reconstruct vocab from checkpoint (don't trust train-side rebuild)
    from man_xfmr_common import TeamVocab, Vocab
    vocab = Vocab(player_to_idx=state["vocab"])
    assert vocab.size == cfg.vocab_size, (
        f"vocab size mismatch: ckpt={cfg.vocab_size} reconstructed={vocab.size}"
    )
    team_vocab = TeamVocab(team_to_idx=state["team_vocab"])
    assert team_vocab.size == cfg.num_teams, (
        f"team vocab mismatch: ckpt={cfg.num_teams} reconstructed={team_vocab.size}"
    )

    test_recs = build_records(
        test_df, histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        matchup_rows=None, lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
    )
    print(f"[records] test={len(test_recs)}")

    test_loader = DataLoader(
        XfmrGameDataset(test_recs), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_xfmr,
    )

    model = ManXfmr(cfg).to(args.device)
    model.load_state_dict(state["model_state"])

    print("\n=== ablation table ===")
    for label, kwargs in [
        ("full         ", {}),
        ("lineup-off   ", {"zero_lineup": True}),
        ("tabular-off  ", {"zero_tabular": True}),
    ]:
        m = eval_one(model, test_loader, device=args.device, **kwargs)
        print(
            f"[{label}] bce={m['bce']:.4f} acc={m['acc']:.3f} "
            f"|z_h|={m['mean_abs_z_h']:.4f} |z_a|={m['mean_abs_z_a']:.4f} "
            f"|tab|={m['mean_abs_tab']:.4f} |logit|={m['mean_abs_logit']:.3f}"
        )


if __name__ == "__main__":
    main()
