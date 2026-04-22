#!/usr/bin/env python3
"""Score the v2-hier checkpoint on val + test, calibrate global_scale on val,
emit predictions.csv compatible with the v5 plot scripts in
models_baseline/scripts/.

Schema:
  game_id, game_date, label_home_win, pred_home_win_prob,
  pred_home_pts, pred_away_pts, margin_mu

The val split is used purely to fit `scale*` so that
P(home_win) = sigmoid((home_pts - away_pts) / scale*) minimizes BCE on val.
The training scale param has no gradient, so it stays at init and is
typically miscalibrated.
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
    PLAYER_FORM_DIM, TABULAR_FEATURE_COLUMNS,
    GameDatasetV2, TeamVocab, Vocab, build_records_v2, collate_v2,
    fit_player_form_stats, fit_tabular_stats,
    load_game_player_status, load_game_scores, load_games,
    load_matchup_rows_v2, load_player_game_stats, load_player_histories,
    load_status_calibration, load_team_exposures,
)
from cme_v2_model import CmeV2, CmeV2Config  # noqa: E402
from man_xfmr_common import PlayerFormStats  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "models_cme_v2" / "artifacts" / "run_v2_hier" / "best.pt")
    p.add_argument("--output", type=Path, default=None,
                   help="Predictions CSV path (default: <ckpt-dir>/predictions.csv)")
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
    p.add_argument("--split", choices=["test", "val", "train"], default="test",
                   help="Which split to dump predictions for. val is always evaluated "
                        "for scale calibration regardless of this flag.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
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
def collect_predictions(model: CmeV2, records, device: str, batch_size: int):
    """Return per-record arrays: margin_mu, home_pts, away_pts, label, win_logit (float32)."""
    loader = DataLoader(GameDatasetV2(records), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_v2)
    margins, h_pts, a_pts, labels, win_logits = [], [], [], [], []
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        margins.append(out["margin_mu"].detach().cpu().numpy())
        h_pts.append(out["home_points"].detach().cpu().numpy())
        a_pts.append(out["away_points"].detach().cpu().numpy())
        labels.append(batch["label"].detach().cpu().numpy())
        win_logits.append(out["win_logit"].detach().cpu().numpy())
    return (np.concatenate(margins), np.concatenate(h_pts),
            np.concatenate(a_pts), np.concatenate(labels),
            np.concatenate(win_logits))


def calibrate_scale(margins: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit scale* = argmin BCE(sigmoid(margin/scale), label). Returns (scale, val_bce)."""
    m = torch.tensor(margins, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    grid = np.concatenate([np.arange(1.0, 30.01, 0.1)])
    best_s, best_bce = float("nan"), float("inf")
    for s in grid:
        logits = m / float(s)
        bce = F.binary_cross_entropy_with_logits(logits, y).item()
        if bce < best_bce:
            best_s, best_bce = float(s), bce
    return best_s, best_bce


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.output is not None:
        out_csv = args.output
    elif args.split == "test":
        out_csv = args.checkpoint.parent / "predictions.csv"
    else:
        out_csv = args.checkpoint.parent / f"predictions_{args.split}.csv"
    if args.split == "test":
        summary_path = args.checkpoint.parent / "eval_summary.json"
    else:
        summary_path = args.checkpoint.parent / f"eval_summary_{args.split}.json"

    print(f"[device] {args.device}")
    print(f"[ckpt]   {args.checkpoint}")

    print("[load] checkpoint")
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = CmeV2Config(**state["cfg"])
    vocab = Vocab(player_to_idx=state["vocab"])
    team_vocab = TeamVocab(team_to_idx=state["team_vocab"])
    use_player_stats = cfg.player_stat_dim > 0
    if use_player_stats:
        player_form_stats = PlayerFormStats(
            means=np.array(state["player_form_means"], dtype=np.float64),
            stds=np.array(state["player_form_stds"], dtype=np.float64),
        )
    else:
        player_form_stats = None

    print("[load] games + odds + scores + statuses + calibration")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)

    train_df, val_df, test_df = chrono_split(games_all, args.val_frac, args.test_frac)
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    eval_df = split_dfs[args.split]
    val_gids = [str(g) for g in val_df["game_id"].tolist()]
    eval_gids = [str(g) for g in eval_df["game_id"].tolist()]

    print(f"[load] matchup rows + per-player game stats (val + {args.split})")
    val_matchup = load_matchup_rows_v2(args.matchup_db, val_gids)
    val_pl = load_player_game_stats(args.core_db, val_gids)
    if args.split == "val":
        eval_matchup, eval_pl = val_matchup, val_pl
    else:
        eval_matchup = load_matchup_rows_v2(args.matchup_db, eval_gids)
        eval_pl = load_player_game_stats(args.core_db, eval_gids)

    print("[fit] tabular stats from train (re-deriving)")
    tabular_stats = fit_tabular_stats(train_df)
    if use_player_stats:
        player_histories = load_player_histories(args.core_db)
    else:
        player_histories = None

    print(f"[records] building val + {args.split}")
    common = dict(
        histories=histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=None,
    )
    val_recs = build_records_v2(
        val_df, matchup_rows=val_matchup, player_game_stats=val_pl, **common,
    )
    if args.split == "val":
        eval_recs = val_recs
    else:
        eval_recs = build_records_v2(
            eval_df, matchup_rows=eval_matchup, player_game_stats=eval_pl, **common,
        )
    print(f"[records] val={len(val_recs)} {args.split}={len(eval_recs)}")

    print("[model] building + loading state")
    model = CmeV2(cfg).to(args.device)
    model.load_state_dict(state["model_state"], strict=False)

    print("[infer] val")
    v_margins, v_hpts, v_apts, v_labels, v_wlogits = collect_predictions(
        model, val_recs, args.device, args.batch_size,
    )
    print(f"  val: margin mean={v_margins.mean():+.2f} std={v_margins.std():.2f}  "
          f"acc={(np.sign(v_margins) == np.where(v_labels > 0.5, 1, -1)).mean():.3f}")

    use_direct = bool(getattr(cfg, "use_direct_win_head", False))
    if use_direct:
        print("[head] direct win head — using win_logit; no scale calibration")
        scale_star = 1.0
        v_logits_for_bce = v_wlogits
        val_bce = float(F.binary_cross_entropy_with_logits(
            torch.tensor(v_wlogits, dtype=torch.float32),
            torch.tensor(v_labels, dtype=torch.float32),
        ).item())
        print(f"  val_bce={val_bce:.4f}")
    else:
        print("[calibrate] fitting global_scale on val (BCE)")
        scale_star, val_bce = calibrate_scale(v_margins, v_labels)
        ckpt_scale = float(state["model_state"]["global_scale"].detach().cpu().exp().item())
        print(f"  scale*={scale_star:.2f}  val_bce={val_bce:.4f}  (ckpt scale={ckpt_scale:.2f})")

    print(f"[infer] {args.split}")
    if args.split == "val":
        t_margins, t_hpts, t_apts, t_labels, t_wlogits = (
            v_margins, v_hpts, v_apts, v_labels, v_wlogits,
        )
    else:
        t_margins, t_hpts, t_apts, t_labels, t_wlogits = collect_predictions(
            model, eval_recs, args.device, args.batch_size,
        )
    if use_direct:
        test_logits = t_wlogits
    else:
        test_logits = t_margins / scale_star
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_bce = float(F.binary_cross_entropy_with_logits(
        torch.tensor(test_logits, dtype=torch.float32),
        torch.tensor(t_labels, dtype=torch.float32),
    ).item())
    test_acc = float(((test_probs > 0.5).astype(np.float32) == t_labels).mean())
    print(f"  {args.split}: bce={test_bce:.4f}  acc={test_acc:.3f}  "
          f"mean_prob={test_probs.mean():.3f}  std_prob={test_probs.std():.3f}")

    print(f"[write] {out_csv}")
    actual_h = np.array([scores[r.game_id][0] for r in eval_recs], dtype=float)
    actual_a = np.array([scores[r.game_id][1] for r in eval_recs], dtype=float)
    df = pd.DataFrame({
        "game_id": [r.game_id for r in eval_recs],
        "game_date": [pd.Timestamp(r.game_date).strftime("%Y-%m-%d") for r in eval_recs],
        "home_team_id": [r.home_team_id for r in eval_recs],
        "away_team_id": [r.away_team_id for r in eval_recs],
        "label_home_win": t_labels.astype(int),
        "pred_home_win_prob": test_probs.astype(float),
        "pred_home_pts": t_hpts.astype(float),
        "pred_away_pts": t_apts.astype(float),
        "actual_home_pts": actual_h,
        "actual_away_pts": actual_a,
        "margin_mu": t_margins.astype(float),
        "actual_margin": actual_h - actual_a,
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "scale_star": scale_star,
        "val_bce": val_bce,
        "n_val": len(val_recs),
        f"n_{args.split}": len(eval_recs),
        f"{args.split}_bce": test_bce,
        f"{args.split}_acc": test_acc,
        f"{args.split}_mean_prob": float(test_probs.mean()),
        f"{args.split}_std_prob": float(test_probs.std()),
        f"{args.split}_mean_home_pts": float(t_hpts.mean()),
        f"{args.split}_mean_away_pts": float(t_apts.mean()),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {out_csv}  +  {summary_path}")


if __name__ == "__main__":
    main()
