#!/usr/bin/env python3
"""Re-evaluate the CME-v1 expanding-window backtest from saved checkpoints,
emitting per-game home_points / away_points / margin_mu alongside win prob.

Loads each window's best.pt under backtest_v1_ckpt/windows/<date>/best.pt and
runs the model in eval mode on that window's test slice. Produces:

  - predictions_with_points.csv (concat across windows)
  - points_summary.json (per-window mean/std/min/max)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_SCRIPTS = Path(__file__).resolve().parent
V5_SCRIPTS = REPO_ROOT / "models_man_xfmr" / "scripts"
sys.path.insert(0, str(V1_SCRIPTS))
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
    PlayerFormStats,
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
from cme_v1_model import CmeV1, CmeV1Config  # noqa: E402


DEFAULT_ART = REPO_ROOT / "models_cme_v1" / "artifacts"
DEFAULT_CKPT_ROOT = DEFAULT_ART / "backtest_v1_ckpt"


def chrono_train_val(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_frac))
    n_train = len(df) - n_val
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


@torch.no_grad()
def predict_window(
    model: CmeV1,
    test_recs: list,
    *,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    loader = DataLoader(XfmrGameDataset(test_recs), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_xfmr)
    rows: list[pd.DataFrame] = []
    offset = 0
    model.eval()
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev)
        bs = batch["label"].size(0)
        probs = torch.sigmoid(out["win_logit"]).detach().cpu().numpy()
        home_pts = out["home_points"].detach().cpu().numpy()
        away_pts = out["away_points"].detach().cpu().numpy()
        margin = out["margin_mu"].detach().cpu().numpy()
        N_home = out["N_home"].detach().cpu().numpy()
        N_away = out["N_away"].detach().cpu().numpy()
        sub = test_recs[offset : offset + bs]
        offset += bs
        rows.append(pd.DataFrame({
            "game_id": [r.game_id for r in sub],
            "game_date": [pd.Timestamp(r.game_date).strftime("%Y-%m-%d") for r in sub],
            "home_team_id": [r.home_team_id for r in sub],
            "away_team_id": [r.away_team_id for r in sub],
            "label_home_win": batch["label"].detach().cpu().numpy().astype(int),
            "label_home_score": [None] * bs,
            "label_away_score": [None] * bs,
            "pred_home_win_prob": probs.astype(float),
            "pred_home_points": home_pts.astype(float),
            "pred_away_points": away_pts.astype(float),
            "pred_margin": margin.astype(float),
            "pred_N_home": N_home.astype(float),
            "pred_N_away": N_away.astype(float),
        }))
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_ART)
    p.add_argument("--run-name", type=str, default="eval_v1_with_points")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)
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

    print(f"[device] {args.device}")
    print(f"[ckpt-root] {args.ckpt_root}")
    windows_dir = args.ckpt_root / "windows"
    window_starts = sorted(p.name for p in windows_dir.iterdir() if p.is_dir())
    print(f"[windows] {len(window_starts)} checkpoints: {window_starts[0]} -> {window_starts[-1]}")

    print("[load] games + exposures + status + calibration")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)

    with sqlite3.connect(args.core_db) as conn:
        scores_df = pd.read_sql_query(
            "SELECT game_id, home_score, away_score FROM games", conn
        )
    scores_df["game_id"] = scores_df["game_id"].astype(str)

    all_preds: list[pd.DataFrame] = []
    summary: dict[str, dict] = {}

    for w_idx, window_start_str in enumerate(window_starts, 1):
        ckpt_path = windows_dir / window_start_str / "best.pt"
        if not ckpt_path.exists():
            print(f"[w{w_idx:02d}] {window_start_str}: missing best.pt, skip")
            continue
        window_start = pd.Timestamp(window_start_str)
        window_end = window_start + pd.offsets.MonthBegin(1)

        state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        cfg = CmeV1Config(**state["cfg"])
        vocab = state["vocab"]
        team_vocab = state["team_vocab"]
        player_form_stats = None
        if state.get("player_form_means") is not None:
            player_form_stats = PlayerFormStats(
                means=np.array(state["player_form_means"], dtype=np.float64),
                stds=np.array(state["player_form_stds"], dtype=np.float64),
            )
        player_histories = load_player_histories(args.core_db) if player_form_stats is not None else None

        full_train = games_all[games_all["game_date"] < window_start].copy()
        train_df, _val_df = chrono_train_val(full_train, args.val_frac)
        test_df = games_all[
            (games_all["game_date"] >= window_start)
            & (games_all["game_date"] < window_end)
        ].copy()
        if test_df.empty:
            print(f"[w{w_idx:02d}] {window_start_str}: empty test, skip")
            continue
        tabular_stats = fit_tabular_stats(train_df)

        test_recs = build_records(
            test_df, histories=histories,
            vocab=type("VocabShim", (), {"encode": lambda _self, p: vocab.get(str(p), 0)})(),
            team_vocab=type("TeamVocabShim", (), {"encode": lambda _self, t: team_vocab.get(str(t), 0)})(),
            status_lookup=statuses,
            calibration=calibration,
            game_scores=scores,
            matchup_rows=None,
            lookback_games=args.lookback_games,
            decay=args.decay,
            tabular_stats=tabular_stats,
            player_histories=player_histories,
            player_form_stats=player_form_stats,
            player_form_lookback=args.player_form_lookback,
            player_form_decay=args.player_form_decay,
            game_odds=None,
        )
        if not test_recs:
            print(f"[w{w_idx:02d}] {window_start_str}: no records, skip")
            continue

        model = CmeV1(cfg).to(args.device)
        model.load_state_dict(state["model_state"])
        df = predict_window(model, test_recs, batch_size=args.batch_size, device=args.device)
        df["window_start"] = window_start_str

        df["game_id"] = df["game_id"].astype(str)
        df = df.drop(columns=["label_home_score", "label_away_score"]).merge(
            scores_df, on="game_id", how="left"
        ).rename(columns={"home_score": "label_home_score", "away_score": "label_away_score"})

        margin_actual = df["label_home_score"] - df["label_away_score"]
        df["err_home_points"] = df["pred_home_points"] - df["label_home_score"]
        df["err_away_points"] = df["pred_away_points"] - df["label_away_score"]
        df["err_margin"] = df["pred_margin"] - margin_actual

        all_preds.append(df)
        summary[window_start_str] = {
            "n": int(len(df)),
            "pred_home_points_mean": float(df["pred_home_points"].mean()),
            "pred_away_points_mean": float(df["pred_away_points"].mean()),
            "label_home_points_mean": float(df["label_home_score"].mean()),
            "label_away_points_mean": float(df["label_away_score"].mean()),
            "home_points_mae": float(df["err_home_points"].abs().mean()),
            "away_points_mae": float(df["err_away_points"].abs().mean()),
            "margin_mae": float(df["err_margin"].abs().mean()),
            "pred_N_home_mean": float(df["pred_N_home"].mean()),
            "pred_N_away_mean": float(df["pred_N_away"].mean()),
        }
        print(
            f"[w{w_idx:02d}] {window_start_str}: "
            f"n={len(df)} "
            f"pred_home={df['pred_home_points'].mean():.1f} "
            f"actual_home={df['label_home_score'].mean():.1f} "
            f"home_mae={df['err_home_points'].abs().mean():.1f} "
            f"margin_mae={df['err_margin'].abs().mean():.1f}"
        )

    preds = pd.concat(all_preds, ignore_index=True)
    csv_path = out_dir / "predictions_with_points.csv"
    preds.to_csv(csv_path, index=False)

    overall = {
        "n_windows": len(summary),
        "n_games": int(len(preds)),
        "pred_home_points_mean": float(preds["pred_home_points"].mean()),
        "pred_away_points_mean": float(preds["pred_away_points"].mean()),
        "label_home_points_mean": float(preds["label_home_score"].mean()),
        "label_away_points_mean": float(preds["label_away_score"].mean()),
        "home_points_mae": float(preds["err_home_points"].abs().mean()),
        "away_points_mae": float(preds["err_away_points"].abs().mean()),
        "margin_mae": float(preds["err_margin"].abs().mean()),
        "home_points_bias": float(preds["err_home_points"].mean()),
        "away_points_bias": float(preds["err_away_points"].mean()),
        "margin_bias": float(preds["err_margin"].mean()),
        "pred_N_home_mean": float(preds["pred_N_home"].mean()),
        "pred_N_away_mean": float(preds["pred_N_away"].mean()),
    }
    summary_path = out_dir / "points_summary.json"
    summary_path.write_text(json.dumps({"overall": overall, "per_window": summary}, indent=2) + "\n")
    print(f"[save] {csv_path}")
    print(f"[save] {summary_path}")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
