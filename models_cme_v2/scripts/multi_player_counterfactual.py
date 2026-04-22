#!/usr/bin/env python3
"""CME-v2 multi-player counterfactual for Lakers March 2026 games.

Trains a CME-v2 checkpoint on all games before 2026-03-01, then scores Lakers
games in March 2026 under real statuses and under player-status overrides.

The override is intentionally the same knob as the v5 counterfactual scripts:
selected (game_id, player_id) statuses are forced to "Out". This measures the
current model's availability sensitivity; it does not simulate replacement
rotations.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
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
    BOX_TARGETS,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB,
    DEFAULT_LINEUP_DECAY,
    DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB,
    DEFAULT_PLAYER_FORM_DECAY,
    DEFAULT_PLAYER_FORM_LOOKBACK,
    K_BOX,
    K_PAIR,
    PLAYER_FORM_DIM,
    TABULAR_FEATURE_COLUMNS,
    GameDatasetV2,
    build_records_v2,
    build_team_vocab,
    build_vocab_from_records_v2,
    collate_v2,
    fit_player_form_stats,
    fit_tabular_stats,
    load_game_player_status,
    load_game_scores,
    load_games,
    load_matchup_rows_v2,
    load_player_game_stats,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
)
from cme_v2_model import CmeV2, CmeV2Config  # noqa: E402
from train_cme_v2 import default_box_weights, default_pair_weights, run_epoch  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v2" / "artifacts"
LAKERS_ID = "1610612747"
WINDOW_START = "2026-03-01"
WINDOW_END = "2026-04-01"
DEFAULT_PLAYERS = "luka:1629029,lebron:2544,luka_and_lebron:1629029+2544"


def parse_players(s: str) -> list[tuple[str, list[str]]]:
    """Parse comma-separated name:pid[+pid...] entries."""
    out: list[tuple[str, list[str]]] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        name, pids = item.split(":", 1)
        pid_list = [p.strip() for p in pids.split("+") if p.strip()]
        if not name.strip() or not pid_list:
            raise ValueError(f"Bad --players entry: {item!r}")
        out.append((name.strip(), pid_list))
    return out


def chrono_train_val(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_frac))
    n_train = len(df) - n_val
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="multi_player_cf_2026_03_pair")
    p.add_argument("--players", type=str, default=DEFAULT_PLAYERS)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)
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
    p.add_argument("--direct-win-head", action="store_true")
    p.add_argument("--use-team-token", action="store_true")
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--no-player-stats", action="store_true")
    p.add_argument("--team-w", type=float, default=1.0)
    p.add_argument("--player-w", type=float, default=0.01)
    p.add_argument("--pair-w", type=float, default=0.001)
    p.add_argument("--win-w", type=float, default=10.0)
    p.add_argument("--margin-nll-w", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--early-stop-metric", choices=("loss", "bce"), default="loss")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args()


@torch.no_grad()
def collect_predictions(
    model: CmeV2,
    records: list,
    *,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    loader = DataLoader(GameDatasetV2(records), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_v2)
    rows: list[pd.DataFrame] = []
    offset = 0
    model.eval()
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev)
        bs = batch["label"].size(0)
        probs = torch.sigmoid(out["win_logit"]).detach().cpu().numpy()
        home_box = out["home_team_box"].detach().cpu().numpy()
        away_box = out["away_team_box"].detach().cpu().numpy()
        margin = out["margin_mu"].detach().cpu().numpy()
        sub_records = records[offset : offset + bs]
        offset += bs

        data = {
            "game_id": [r.game_id for r in sub_records],
            "game_date": [pd.Timestamp(r.game_date).strftime("%Y-%m-%d") for r in sub_records],
            "home_team_id": [r.home_team_id for r in sub_records],
            "away_team_id": [r.away_team_id for r in sub_records],
            "label_home_win": batch["label"].detach().cpu().numpy().astype(int),
            "pred_home_win_prob": probs.astype(float),
            "margin_mu": margin.astype(float),
        }
        for i, stat in enumerate(BOX_TARGETS):
            data[f"pred_home_{stat}"] = home_box[:, i].astype(float)
            data[f"pred_away_{stat}"] = away_box[:, i].astype(float)
        rows.append(pd.DataFrame(data))
    return pd.concat(rows, ignore_index=True)


def metrics_from_predictions(df: pd.DataFrame) -> dict:
    y = df["label_home_win"].to_numpy(dtype=np.float32)
    p = df["pred_home_win_prob"].to_numpy(dtype=np.float32)
    eps = 1e-7
    bce = -np.mean(y * np.log(np.clip(p, eps, 1 - eps))
                   + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps)))
    acc = np.mean((p >= 0.5).astype(np.float32) == y)
    return {"test_bce": float(bce), "test_acc": float(acc)}


def build_eval_records(
    test_df: pd.DataFrame,
    *,
    histories,
    vocab,
    team_vocab,
    statuses,
    calibration,
    scores,
    test_matchup,
    test_player_stats,
    args,
    tabular_stats,
    player_histories,
    player_form_stats,
) -> list:
    return build_records_v2(
        test_df,
        histories=histories,
        vocab=vocab,
        team_vocab=team_vocab,
        status_lookup=statuses,
        calibration=calibration,
        game_scores=scores,
        matchup_rows=test_matchup,
        player_game_stats=test_player_stats,
        lookback_games=args.lookback_games,
        decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=None,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    players = parse_players(args.players)

    print(f"[device] {args.device}")
    print(f"[players] {players}")
    print("[load] games + scores + statuses + calibration + exposures")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)
    use_player_stats = not args.no_player_stats
    player_histories = load_player_histories(args.core_db) if use_player_stats else None

    window_start = pd.Timestamp(WINDOW_START)
    window_end = pd.Timestamp(WINDOW_END)
    full_train = games_all[games_all["game_date"] < window_start].copy()
    test_all = games_all[
        (games_all["game_date"] >= window_start)
        & (games_all["game_date"] < window_end)
    ].copy()
    test_df = test_all[
        (test_all["home_team_id"].astype(str) == LAKERS_ID)
        | (test_all["away_team_id"].astype(str) == LAKERS_ID)
    ].copy().reset_index(drop=True)
    train_df, val_df = chrono_train_val(full_train, args.val_frac)
    print(f"[window] train={len(train_df)} val={len(val_df)} test_lakers={len(test_df)}")

    train_gids = [str(g) for g in train_df["game_id"].tolist()]
    val_gids = [str(g) for g in val_df["game_id"].tolist()]
    test_gids = [str(g) for g in test_df["game_id"].tolist()]
    print("[load] matchup rows + player box labels")
    train_matchup = load_matchup_rows_v2(args.matchup_db, train_gids)
    val_matchup = load_matchup_rows_v2(args.matchup_db, val_gids)
    test_matchup = load_matchup_rows_v2(args.matchup_db, test_gids)
    train_pl = load_player_game_stats(args.core_db, train_gids)
    val_pl = load_player_game_stats(args.core_db, val_gids)
    test_pl = load_player_game_stats(args.core_db, test_gids)

    print("[fit] vocab + standardizers")
    vocab = build_vocab_from_records_v2(
        train_df, histories, train_matchup,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(train_df)
    tabular_stats = fit_tabular_stats(train_df)
    if use_player_stats:
        player_form_stats = fit_player_form_stats(
            player_histories, train_df,
            lookback_games=args.player_form_lookback,
            decay=args.player_form_decay,
        )
    else:
        player_form_stats = None

    common = dict(
        histories=histories,
        vocab=vocab,
        team_vocab=team_vocab,
        status_lookup=statuses,
        calibration=calibration,
        game_scores=scores,
        lookback_games=args.lookback_games,
        decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=None,
    )
    train_recs = build_records_v2(
        train_df, matchup_rows=train_matchup, player_game_stats=train_pl, **common,
    )
    val_recs = build_records_v2(
        val_df, matchup_rows=val_matchup, player_game_stats=val_pl, **common,
    )
    print(f"[records] train={len(train_recs)} val={len(val_recs)}")

    train_loader = DataLoader(GameDatasetV2(train_recs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_v2)
    val_loader = DataLoader(GameDatasetV2(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_v2)

    cfg = CmeV2Config(
        vocab_size=vocab.size,
        num_teams=team_vocab.size,
        d=args.d,
        n_heads=args.n_heads,
        n_self_layers=args.n_self_layers,
        n_cross_layers=args.n_cross_layers,
        pair_hidden=args.pair_hidden,
        player_hidden=args.player_hidden,
        dropout=args.dropout,
        pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        tabular_dim=0 if args.no_tabular else len(TABULAR_FEATURE_COLUMNS),
        team_emb_dim=args.team_emb_dim,
        player_stat_dim=PLAYER_FORM_DIM if use_player_stats else 0,
        init_scale=args.init_scale,
        use_direct_win_head=args.direct_win_head,
        use_team_token=args.use_team_token,
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
        box_weights=box_weights,
        pair_weights=pair_weights,
        team_w=args.team_w,
        player_w=args.player_w,
        pair_w=args.pair_w,
        win_w=args.win_w,
        margin_nll_w=args.margin_nll_w,
        decision_w=0.0,
    )

    print(f"[model] params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} cfg={asdict(cfg)}")
    best_value = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict] = []
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device=args.device, optim=optim, **epoch_kwargs)
        va = run_epoch(model, val_loader, device=args.device, optim=None, **epoch_kwargs)
        scheduler.step()
        metric = va["loss"] if args.early_stop_metric == "loss" else va["bce"]
        row = {
            "epoch": epoch,
            "secs": time.time() - t0,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_bce": tr["bce"],
            "val_bce": va["bce"],
            "train_acc": tr["acc"],
            "val_acc": va["acc"],
            "train_margin_mae": tr["margin_mae"],
            "val_margin_mae": va["margin_mae"],
        }
        history.append(row)
        print(
            f"[ep{epoch:02d}] tr_loss={tr['loss']:.3f} va_loss={va['loss']:.3f} "
            f"tr_bce={tr['bce']:.4f} va_bce={va['bce']:.4f} "
            f"va_acc={va['acc']:.3f} va_mae={va['margin_mae']:.2f} "
            f"({row['secs']:.1f}s)"
        )
        if metric < best_value - 1e-5:
            best_value = metric
            best_epoch = epoch
            epochs_since_best = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"[early-stop] no {args.early_stop_metric} improvement for {args.patience} epochs")
                break

    train_wall = time.time() - t_train
    if best_state is not None:
        model.load_state_dict(best_state)
    if args.save_checkpoint:
        torch.save(
            {
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "vocab": vocab.player_to_idx,
                "team_vocab": team_vocab.team_to_idx,
                "player_form_means": player_form_stats.means.tolist() if player_form_stats is not None else None,
                "player_form_stds": player_form_stats.stds.tolist() if player_form_stats is not None else None,
                "box_weights": box_weights.tolist(),
                "pair_weights": pair_weights.tolist(),
                "epoch": best_epoch,
            },
            out_dir / "model.pt",
        )

    print("[predict] baseline")
    baseline_recs = build_eval_records(
        test_df,
        histories=histories,
        vocab=vocab,
        team_vocab=team_vocab,
        statuses=statuses,
        calibration=calibration,
        scores=scores,
        test_matchup=test_matchup,
        test_player_stats=test_pl,
        args=args,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
    )
    baseline_df = collect_predictions(
        model, baseline_recs, batch_size=args.batch_size, device=args.device,
    ).rename(columns={"pred_home_win_prob": "p_home_baseline"})
    baseline_metrics = metrics_from_predictions(
        baseline_df.rename(columns={"p_home_baseline": "pred_home_win_prob"})
    )

    per_player: dict[str, pd.DataFrame] = {}
    per_player_summary: dict[str, dict] = {}
    for name, pid_list in players:
        statuses_cf = dict(statuses)
        prior_counts: dict[str, dict[str, int]] = {pid: {} for pid in pid_list}
        for gid in test_df["game_id"].astype(str):
            for pid in pid_list:
                prior = statuses_cf.get((gid, pid), "NotListed")
                prior_counts[pid][prior] = prior_counts[pid].get(prior, 0) + 1
                statuses_cf[(gid, pid)] = "Out"
        print(f"[predict] cf {name} prior={prior_counts}")
        cf_recs = build_eval_records(
            test_df,
            histories=histories,
            vocab=vocab,
            team_vocab=team_vocab,
            statuses=statuses_cf,
            calibration=calibration,
            scores=scores,
            test_matchup=test_matchup,
            test_player_stats=test_pl,
            args=args,
            tabular_stats=tabular_stats,
            player_histories=player_histories,
            player_form_stats=player_form_stats,
        )
        cf_df = collect_predictions(
            model, cf_recs, batch_size=args.batch_size, device=args.device,
        ).rename(columns={"pred_home_win_prob": f"p_home_{name}_out"})
        per_player[name] = cf_df
        per_player_summary[name] = {
            "player_ids": pid_list,
            "prior_status_counts": prior_counts,
            **metrics_from_predictions(
                cf_df.rename(columns={f"p_home_{name}_out": "pred_home_win_prob"})
            ),
        }

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

    merged = baseline_df.copy()
    merged["game_id"] = merged["game_id"].astype(str)
    merged = merged.merge(meta, on="game_id", how="left")
    merged["p_lakers_baseline"] = np.where(
        merged["lakers_home"] == 1,
        merged["p_home_baseline"],
        1.0 - merged["p_home_baseline"],
    )
    merged["pred_lakers_pts_baseline"] = np.where(
        merged["lakers_home"] == 1,
        merged["pred_home_pts"],
        merged["pred_away_pts"],
    )

    for name, cf_df in per_player.items():
        cf_df = cf_df.copy()
        cf_df["game_id"] = cf_df["game_id"].astype(str)
        cols = ["game_id", f"p_home_{name}_out", "pred_home_pts", "pred_away_pts"]
        merged = merged.merge(cf_df[cols], on="game_id", how="left", suffixes=("", f"_{name}_out"))
        merged[f"p_lakers_{name}_out"] = np.where(
            merged["lakers_home"] == 1,
            merged[f"p_home_{name}_out"],
            1.0 - merged[f"p_home_{name}_out"],
        )
        merged[f"delta_{name}"] = merged[f"p_lakers_{name}_out"] - merged["p_lakers_baseline"]
        merged[f"pred_lakers_pts_{name}_out"] = np.where(
            merged["lakers_home"] == 1,
            merged[f"pred_home_pts_{name}_out"],
            merged[f"pred_away_pts_{name}_out"],
        )
        merged[f"delta_lakers_pts_{name}"] = (
            merged[f"pred_lakers_pts_{name}_out"] - merged["pred_lakers_pts_baseline"]
        )

    base_cols = [
        "game_id", "game_date", "lakers_home", "opponent_abbr", "label_home_win",
        "p_home_baseline", "p_lakers_baseline", "pred_lakers_pts_baseline",
    ]
    cf_cols: list[str] = []
    for name, _pid_list in players:
        cf_cols += [
            f"p_home_{name}_out", f"p_lakers_{name}_out", f"delta_{name}",
            f"pred_lakers_pts_{name}_out", f"delta_lakers_pts_{name}",
        ]
    merged = merged[base_cols + cf_cols].sort_values("game_date").reset_index(drop=True)
    csv_path = out_dir / "lakers_march_multi_player.csv"
    merged.to_csv(csv_path, index=False)

    long_rows = []
    scenario_frames = [("baseline", baseline_df.rename(columns={"p_home_baseline": "pred_home_win_prob"}))]
    for name, cf_df in per_player.items():
        scenario_frames.append((f"{name}_out", cf_df.rename(columns={f"p_home_{name}_out": "pred_home_win_prob"})))
    for scenario, frame in scenario_frames:
        f = frame.copy()
        f["game_id"] = f["game_id"].astype(str)
        f = f.merge(meta, on="game_id", how="left")
        for row in f.itertuples(index=False):
            lakers_home = int(row.lakers_home) == 1
            item = {
                "game_id": row.game_id,
                "game_date": row.game_date,
                "scenario": scenario,
                "lakers_home": int(row.lakers_home),
                "opponent_abbr": row.opponent_abbr,
                "p_home": float(row.pred_home_win_prob),
            }
            for stat in BOX_TARGETS:
                h_val = float(getattr(row, f"pred_home_{stat}"))
                a_val = float(getattr(row, f"pred_away_{stat}"))
                item[f"lakers_{stat}"] = h_val if lakers_home else a_val
                item[f"opponent_{stat}"] = a_val if lakers_home else h_val
            long_rows.append(item)
    team_box_path = out_dir / "lakers_march_team_box_long.csv"
    pd.DataFrame(long_rows).to_csv(team_box_path, index=False)

    summary = {
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "lakers_team_id": LAKERS_ID,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test_lakers": int(len(test_df)),
        "train_wall_secs": float(train_wall),
        "best_epoch": int(best_epoch),
        "early_stop_metric": args.early_stop_metric,
        "best_value": float(best_value),
        "baseline_test_bce": baseline_metrics["test_bce"],
        "baseline_test_acc": baseline_metrics["test_acc"],
        "mean_p_lakers_baseline": float(merged["p_lakers_baseline"].mean()),
        "mean_lakers_pts_baseline": float(merged["pred_lakers_pts_baseline"].mean()),
        "players": [],
        "cfg": asdict(cfg),
    }
    for name, pid_list in players:
        deltas = merged[f"delta_{name}"]
        pts_deltas = merged[f"delta_lakers_pts_{name}"]
        player_metrics = per_player_summary[name]
        summary["players"].append({
            "name": name,
            "player_ids": pid_list,
            "mean_delta": float(deltas.mean()),
            "median_delta": float(deltas.median()),
            "min_delta": float(deltas.min()),
            "max_delta": float(deltas.max()),
            "mean_p_lakers_out": float(merged[f"p_lakers_{name}_out"].mean()),
            "mean_lakers_pts_out": float(merged[f"pred_lakers_pts_{name}_out"].mean()),
            "mean_delta_lakers_pts": float(pts_deltas.mean()),
            "median_delta_lakers_pts": float(pts_deltas.median()),
            "prior_status_counts": player_metrics["prior_status_counts"],
            "test_bce": player_metrics["test_bce"],
            "test_acc": player_metrics["test_acc"],
        })

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[save] {csv_path}")
    print(f"[save] {team_box_path}")
    print(f"[save] {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
