#!/usr/bin/env python3
"""Evaluate trained CME-v1: does Sinkhorn pair structure beat a rank-1 baseline?

Pass criterion: CME-v1's learned pair exposure P_ij produces lower test
Poisson NLL than a rank-1 outer-product baseline P^rank1_ij = r_i * c_j / N
(uses the same row/col marginals but throws away the learned pair score).

If CME-v1 beats rank-1 by >0.01 in mean Poisson, the matchup-conditional
score signal is doing real work. If it ties, the Sinkhorn structure has
not learned anything useful and we should question the design.
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
V5_SCRIPTS = REPO_ROOT / "models_man_xfmr" / "scripts"
sys.path.insert(0, str(V5_SCRIPTS))

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
    Vocab,
    TeamVocab,
    XfmrGameDataset,
    build_records,
    build_team_vocab,
    build_vocab_from_records,
    collate_xfmr,
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

from cme_v1_model import (  # noqa: E402
    CmeV1,
    CmeV1Config,
    gather_lambda_for_supervision,
    poisson_nll,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
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


def gather_rank1_lam(
    row_mass_h: torch.Tensor,    # (B, Lo)
    col_mass_h: torch.Tensor,    # (B, Ld)
    N_h: torch.Tensor,           # (B,)
    row_mass_a: torch.Tensor,
    col_mass_a: torch.Tensor,
    N_a: torch.Tensor,
    sup_game: torch.Tensor,
    sup_side: torch.Tensor,
    sup_off: torch.Tensor,
    sup_def: torch.Tensor,
) -> torch.Tensor:
    """Rank-1 baseline: P_ij = r_i * c_j / N, gathered at supervision rows."""
    out = torch.zeros(sup_game.size(0), device=row_mass_h.device, dtype=row_mass_h.dtype)
    h = sup_side == 0
    a = sup_side == 1
    if h.any():
        idx = h.nonzero(as_tuple=False).squeeze(-1)
        g = sup_game[idx]; oi = sup_off[idx]; di = sup_def[idx]
        out[idx] = row_mass_h[g, oi] * col_mass_h[g, di] / N_h[g].clamp_min(1e-6)
    if a.any():
        idx = a.nonzero(as_tuple=False).squeeze(-1)
        g = sup_game[idx]; oi = sup_off[idx]; di = sup_def[idx]
        out[idx] = row_mass_a[g, oi] * col_mass_a[g, di] / N_a[g].clamp_min(1e-6)
    return out


def evaluate(model, loader, device):
    model.eval()
    sums = {"n": 0, "pois_full": 0.0, "pois_rank1": 0.0,
            "bce": 0.0, "margin_abs": 0.0, "logit": 0.0, "correct": 0}

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)

            lam_full = gather_lambda_for_supervision(
                out["lam_h"], out["lam_a"],
                batch["sup_game"], batch["sup_side"],
                batch["sup_off"], batch["sup_def"],
            )
            pois_full = poisson_nll(lam_full, batch["sup_exp"])

            lam_rank1 = gather_rank1_lam(
                out["home_off_inv"], out["away_def_inv"], out["N_home"],
                out["away_off_inv"], out["home_def_inv"], out["N_away"],
                batch["sup_game"], batch["sup_side"],
                batch["sup_off"], batch["sup_def"],
            )
            pois_rank1 = poisson_nll(lam_rank1, batch["sup_exp"])

            bce = F.binary_cross_entropy_with_logits(out["win_logit"], batch["label"])
            margin_abs = (out["margin_mu"] - batch["margin"]).abs().mean()
            preds = (torch.sigmoid(out["win_logit"]) > 0.5).float()
            correct = (preds == batch["label"]).sum().item()

        bs = batch["label"].size(0)
        sums["n"] += bs
        sums["pois_full"] += pois_full.item() * bs
        sums["pois_rank1"] += pois_rank1.item() * bs
        sums["bce"] += bce.item() * bs
        sums["margin_abs"] += margin_abs.item() * bs
        sums["logit"] += out["win_logit"].sum().item()
        sums["correct"] += correct

    n = sums["n"]
    return {
        "n": n,
        "pois_full": sums["pois_full"] / n,
        "pois_rank1": sums["pois_rank1"] / n,
        "delta_pois": (sums["pois_full"] - sums["pois_rank1"]) / n,
        "bce": sums["bce"] / n,
        "margin_mae": sums["margin_abs"] / n,
        "acc": sums["correct"] / n,
        "mean_logit": sums["logit"] / n,
    }


def main() -> None:
    args = parse_args()
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = CmeV1Config(**state["cfg"])
    model = CmeV1(cfg).to(args.device)
    model.load_state_dict(state["model_state"])

    vocab = Vocab(state["vocab"]) if "vocab" in state else None
    team_vocab_dict = state.get("team_vocab", {})
    team_vocab = TeamVocab(team_vocab_dict)

    print(f"[ckpt] {args.ckpt} epoch={state.get('epoch', '?')}")
    print(f"[cfg] vocab={cfg.vocab_size} d={cfg.d} self_attn_layers={cfg.n_self_attn_layers}")

    print("[load] games + exposures + matchup rows + status + calibration")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)

    train_df, val_df, test_df = chrono_split(games_all, args.val_frac, args.test_frac)
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    test_matchup = load_matchup_rows(
        args.matchup_db, [str(g) for g in test_df["game_id"].tolist()]
    )

    train_matchup = load_matchup_rows(
        args.matchup_db, [str(g) for g in train_df["game_id"].tolist()]
    )
    if vocab is None:
        vocab = build_vocab_from_records(
            train_df, histories, train_matchup,
            lookback_games=args.lookback_games, decay=args.decay,
        )

    tabular_stats = fit_tabular_stats(train_df)

    use_player_stats = cfg.player_stat_dim > 0
    if use_player_stats:
        player_histories = load_player_histories(args.core_db)
        player_form_stats = fit_player_form_stats(
            player_histories, train_df,
            lookback_games=args.player_form_lookback,
            decay=args.player_form_decay,
        )
    else:
        player_histories = None
        player_form_stats = None

    test_recs = build_records(
        test_df, histories=histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        matchup_rows=test_matchup,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=None,
    )
    print(f"[records] test={len(test_recs)}")

    test_loader = DataLoader(
        XfmrGameDataset(test_recs), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_xfmr,
    )

    res = evaluate(model, test_loader, args.device)
    print()
    print("=== Test diagnostics ===")
    print(f"n games               {res['n']}")
    print(f"pois_full (CME-v1)    {res['pois_full']:.5f}")
    print(f"pois_rank1 (baseline) {res['pois_rank1']:.5f}")
    print(f"delta (full - rank1)  {res['delta_pois']:+.5f}   [<0 means CME-v1 beats rank-1]")
    print()
    print(f"bce                   {res['bce']:.4f}    (chance = 0.6931)")
    print(f"acc                   {res['acc']:.3f}")
    print(f"margin_mae            {res['margin_mae']:.2f}")
    print(f"mean_logit            {res['mean_logit']:+.4f}")

    threshold = -0.010
    if res["delta_pois"] < threshold:
        verdict = f"PASS: CME-v1 beats rank-1 by {-res['delta_pois']:.4f} in Poisson NLL"
    elif res["delta_pois"] < 0:
        verdict = f"WEAK PASS: CME-v1 marginally beats rank-1 (delta={res['delta_pois']:+.4f}, threshold={threshold})"
    else:
        verdict = f"FAIL: rank-1 ties or beats CME-v1 (delta={res['delta_pois']:+.4f}); pair score is not adding signal"
    print()
    print(verdict)

    out_path = args.ckpt.parent / "diagnostics.json"
    with open(out_path, "w") as f:
        json.dump({**res, "verdict": verdict}, f, indent=2)
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()
