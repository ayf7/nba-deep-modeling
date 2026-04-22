#!/usr/bin/env python3
"""Expanding-window monthly backtest for CME-v3.

Mirrors `models_cme_v2/scripts/backtest_cme_v2.py`: for each month-start
after --initial-train-end, train on all games strictly before that month
and predict every game inside the month. Predictions are stitched into a
single CSV with the same schema as the baseline / v2 backtests
(game_id, game_date, home_team_id, away_team_id, label_home_win,
pred_home_win_prob, window_start) so it can be fed straight into
evaluate_betting_strategy.py.

Within each window the most recent --val-frac-of-train fraction of the
train block is held out chronologically for early stopping.

Defaults match `train_cme_v3.py`'s six-level loss
(team=1.0, player=0.01, pair=0.001, inv=5.0, win=10.0, margin_nll=0.0).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
V3_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V3_SCRIPTS))

from cme_v3_common import (  # noqa: E402
    BOX_INDEX, K_BOX, K_PAIR,
    DEFAULT_CALIBRATION_PATH, DEFAULT_CORE_DB, DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB, DEFAULT_LINEUP_DECAY, DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB, DEFAULT_PLAYER_FORM_DECAY, DEFAULT_PLAYER_FORM_LOOKBACK,
    PLAYER_FORM_DIM, TABULAR_FEATURE_COLUMNS,
    GameDatasetV3, build_records_v3, build_team_vocab,
    build_vocab_from_records_v3, collate_v3,
    fit_tabular_stats,
    load_game_odds, load_game_player_status, load_game_scores, load_games,
    load_matchup_rows_v2, load_player_game_stats,
    load_status_calibration, load_team_exposures,
)
from cme_v3_model import CmeV3, CmeV3Config  # noqa: E402
from train_cme_v3 import (  # noqa: E402
    default_box_weights, default_pair_weights, run_epoch,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v3" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="backtest")
    p.add_argument("--initial-train-end", default="2023-12-31",
                   help="First test window is the month strictly after this date.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="If set, only run the first N windows (for smoke tests).")
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--val-frac-of-train", type=float, default=0.15,
                   help="Within each window's train block, hold out the most recent X%% as val.")
    # v3 backbone config
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-self-layers", type=int, default=2)
    p.add_argument("--n-cross-layers", type=int, default=2)
    p.add_argument("--pair-hidden", type=int, default=96)
    p.add_argument("--player-hidden", type=int, default=64)
    p.add_argument("--inv-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pair-dropout", type=float, default=0.2)
    p.add_argument("--player-dropout", type=float, default=0.0)
    p.add_argument("--team-emb-dim", type=int, default=16)
    p.add_argument("--init-scale", type=float, default=12.0)
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--base-possessions", type=float, default=491.0)
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--no-cyclic", action="store_true",
                   help="Drop the 12 cyc_* columns from the tabular feature set (pre-cyclic baseline).")
    # six-level loss weights (match train_cme_v3 defaults)
    p.add_argument("--team-w", type=float, default=1.0)
    p.add_argument("--player-w", type=float, default=0.01)
    p.add_argument("--pair-w", type=float, default=0.001)
    p.add_argument("--inv-w", type=float, default=5.0)
    p.add_argument("--win-w", type=float, default=10.0)
    p.add_argument("--margin-nll-w", type=float, default=0.0)
    p.add_argument("--box-weights", type=float, nargs="+", default=None,
                   help=f"Per-stat weights for team+player levels (length {K_BOX}). "
                        f"Default = default_box_weights().")
    p.add_argument("--pair-weights", type=float, nargs="+", default=None,
                   help=f"Per-target weights for pair Poisson NLL (length {K_PAIR}). "
                        f"Default = default_pair_weights().")
    # curriculum (two-phase) training
    p.add_argument("--curriculum", action="store_true",
                   help="Two-phase training: Phase 1 supervises structural losses "
                        "(team/player/pair MSE) to convergence; then hard-switch to "
                        "cfg-D (BCE + exposure-only pair Poisson) and fine-tune.")
    p.add_argument("--phase1-team-w", type=float, default=1.0)
    p.add_argument("--phase1-player-w", type=float, default=0.01)
    p.add_argument("--phase1-pair-w", type=float, default=0.1)
    p.add_argument("--phase1-inv-w", type=float, default=0.0,
                   help="Usage-share supervision (involvement MSE) weight in "
                        "Phase 1. Default 0 preserves earlier curriculum "
                        "behavior; set >0 to directly supervise alpha_off / "
                        "alpha_def per player.")
    p.add_argument("--phase1-epochs", type=int, default=50,
                   help="Max epochs for Phase 1 (curriculum mode).")
    p.add_argument("--phase1-patience", type=int, default=6,
                   help="Patience (in epochs) for Phase 1 plateau early stop. "
                        "Tracks val structural loss, not val_bce.")
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoints", action="store_true",
                   help="Write best.pt per window to "
                        "<out>/windows/<window_start>/best.pt (with cfg/vocab "
                        "needed to reload the model).")
    p.add_argument("--save-final", action="store_true",
                   help="Additionally write final.pt (last-epoch weights) per "
                        "window. No effect without --save-checkpoints.")
    p.add_argument("--inspect-players", action="store_true",
                   help="After training, print per-player predicted vs actual "
                        "(exposure / pts / alpha_off) for the first 2 test "
                        "games of each window. Diagnostic only; no effect on "
                        "predictions.")
    p.add_argument("--inspect-top-k", type=int, default=10,
                   help="Top K players per roster to show in --inspect-players "
                        "(sorted by predicted exposure).")
    p.add_argument("--track-pids", type=str, default=None,
                   help="Comma-separated list of player IDs to track across "
                        "ALL test games in each window. For each (game, pid), "
                        "prints one line of predicted vs actual stats.")
    return p.parse_args()


def month_starts(dates: pd.Series) -> list[pd.Timestamp]:
    periods = pd.to_datetime(dates).dt.to_period("M").drop_duplicates().sort_values()
    return [period.to_timestamp() for period in periods]


def chrono_val_split(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def predict_window(
    model: CmeV3, records: list, batch_size: int, device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(GameDatasetV3(records), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_v3)
    probs: list[np.ndarray] = []
    home_pts: list[np.ndarray] = []
    away_pts: list[np.ndarray] = []
    margin_mu: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_dev)
            probs.append(torch.sigmoid(out["win_logit"]).cpu().numpy())
            home_pts.append(out["home_points"].cpu().numpy())
            away_pts.append(out["away_points"].cpu().numpy())
            margin_mu.append(out["margin_mu"].cpu().numpy())
    gids = np.array([str(r.game_id) for r in records])
    return (
        gids,
        np.concatenate(probs),
        np.concatenate(home_pts),
        np.concatenate(away_pts),
        np.concatenate(margin_mu),
    )


def _build_ckpt_payload(model, cfg, vocab, team_vocab, box_weights, pair_weights,
                        team_w, player_w, pair_w, inv_w, win_w, margin_nll_w, epoch):
    return {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": asdict(cfg),
        "vocab": vocab.player_to_idx,
        "team_vocab": team_vocab.team_to_idx,
        "player_form_means": None,
        "player_form_stds": None,
        "box_weights": box_weights.tolist(),
        "pair_weights": pair_weights.tolist(),
        "loss_level_weights": {
            "team": team_w, "player": player_w, "pair": pair_w,
            "inv": inv_w, "win": win_w, "margin_nll": margin_nll_w,
        },
        "epoch": epoch,
    }


def _build_lr_scheduler(optim, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        denom = max(1, total_epochs - warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def _per_task_grad_norms(
    model, loader, device: str,
    box_weights: torch.Tensor, pair_weights: torch.Tensor,
) -> dict:
    """Compute per-task ||grad of L_*|| on one batch. Unweighted by outer task_w."""
    import torch.nn.functional as F
    from cme_v3_model import (
        involvement_mse_loss, pair_poisson_loss, player_mse_loss, team_mse_loss,
    )

    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    box_w = box_weights.to(device)
    pair_w = pair_weights.to(device)

    def _norm() -> float:
        sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                sq += p.grad.detach().pow(2).sum().item()
        return sq ** 0.5

    out: dict = {}
    model.train()
    # team
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    L = (box_w * team_mse_loss(fwd, batch)).sum()
    L.backward()
    out["team"] = (_norm(), L.item())
    # player
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    pl = player_mse_loss(fwd, batch)
    if pl.numel() > 0:
        L = (box_w * pl).sum()
        L.backward()
        out["player"] = (_norm(), L.item())
    else:
        out["player"] = (0.0, 0.0)
    # pair
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    pair_nll, _ = pair_poisson_loss(fwd, batch)
    L = (pair_w * pair_nll).sum()
    L.backward()
    out["pair"] = (_norm(), L.item())
    # inv (involvement / usage MSE)
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    L = involvement_mse_loss(fwd, batch)
    L.backward()
    out["inv"] = (_norm(), L.item())
    # win
    model.zero_grad(set_to_none=True)
    fwd = model(batch)
    L = F.binary_cross_entropy_with_logits(fwd["win_logit"], batch["label"])
    L.backward()
    out["win"] = (_norm(), L.item())

    model.zero_grad(set_to_none=True)
    return out


def _fmt_grad_norms(label: str, norms: dict, task_weights: dict) -> str:
    """Render grad norms in a one-line table. task_weights gives outer w_* per task."""
    parts = [label]
    for task in ("team", "player", "pair", "inv", "win"):
        if task not in norms:
            continue
        g, L = norms[task]
        w = task_weights.get(task, 1.0)
        parts.append(f"{task}=L{L:7.3f} g{g:7.3f} w{w:.3g} wg{w*g:7.3f}")
    return " | ".join(parts)


def _run_training_loop(
    model, train_loader, val_loader, optim, scheduler,
    epoch_kwargs: dict, num_epochs: int, patience: int, device: str,
    track_metric: str, phase_label: int,
) -> tuple[dict | None, int, float, list[dict]]:
    """Run a training loop with early stopping. Returns (best_state, best_epoch, best_metric, history)."""
    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict] = []
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device=device, optim=optim, **epoch_kwargs)
        va = run_epoch(model, val_loader, device=device, optim=None, **epoch_kwargs)
        scheduler.step()
        dt = time.time() - t0
        history.append({
            "phase": phase_label, "epoch": epoch, "secs": dt,
            "train_loss": tr["loss"], "val_loss": va["loss"],
            "train_bce": tr["bce"], "val_bce": va["bce"],
            "train_acc": tr["acc"], "val_acc": va["acc"],
            "train_alpha_off_corr": tr["alpha_off_corr"],
            "val_alpha_off_corr": va["alpha_off_corr"],
            "train_mean_N": tr["mean_N"], "val_mean_N": va["mean_N"],
        })
        if va[track_metric] < best_val - 1e-5:
            best_val = va[track_metric]
            best_epoch = epoch
            epochs_since_best = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break
    return best_state, best_epoch, best_val, history


def _inspect_player_predictions(
    model, test_recs, vocab, device: str, n_games: int = 2, top_k: int = 10,
) -> None:
    """Print per-player predicted vs actual for the first n_games test games.

    Diagnostic only. Shows top_k home/away players sorted by predicted
    exposure with predicted vs actual exposure / pts / alpha_off.
    """
    if not test_recs:
        return
    idx_to_pid = {i: pid for pid, i in vocab.player_to_idx.items()}
    sample = test_recs[: max(1, n_games)]
    loader = DataLoader(GameDatasetV3(sample), batch_size=len(sample),
                        shuffle=False, collate_fn=collate_v3)
    pts_i = BOX_INDEX["pts"]
    fgm_i = BOX_INDEX["fgm"]
    fga_i = BOX_INDEX["fga"]
    ast_i = BOX_INDEX["ast"]

    model.eval()
    with torch.no_grad():
        batch_cpu = next(iter(loader))
        batch = {k: v.to(device) for k, v in batch_cpu.items()}
        fwd = model(batch)
        win_prob = torch.sigmoid(fwd["win_logit"]).cpu().numpy()
        for b in range(len(sample)):
            rec = sample[b]
            pred_h = fwd["home_team_box"][b].cpu().numpy()
            pred_a = fwd["away_team_box"][b].cpu().numpy()
            act_h = batch_cpu["team_box_home"][b].cpu().numpy()
            act_a = batch_cpu["team_box_away"][b].cpu().numpy()

            print(f"\n[player-diag] gid={rec.game_id} {rec.game_date.date()} "
                  f"home={rec.home_team_id} away={rec.away_team_id} "
                  f"label_home_win={int(batch_cpu['label'][b].item())} "
                  f"win_prob={win_prob[b]:.3f}")
            print(f"  team   pred  H={pred_h[pts_i]:6.1f}  A={pred_a[pts_i]:6.1f}  "
                  f"|  actual  H={act_h[pts_i]:.0f}  A={act_a[pts_i]:.0f}")

            for side, side_name in [(0, "HOME"), (1, "AWAY")]:
                pair_marg = (fwd["home_pair_marg"] if side == 0 else fwd["away_pair_marg"])[b].cpu().numpy()
                box = (fwd["home_box"] if side == 0 else fwd["away_box"])[b].cpu().numpy()
                alpha_off = (fwd["alpha_home_off"] if side == 0 else fwd["alpha_away_off"])[b].cpu().numpy()
                mask = (batch_cpu["home_mask"] if side == 0 else batch_cpu["away_mask"])[b].cpu().numpy().astype(bool)
                idx = (batch_cpu["home_idx"] if side == 0 else batch_cpu["away_idx"])[b].cpu().numpy()
                prob = (batch_cpu["home_prob"] if side == 0 else batch_cpu["away_prob"])[b].cpu().numpy()
                actual_alpha = (batch_cpu["home_alpha_off_actual"] if side == 0
                                else batch_cpu["away_alpha_off_actual"])[b].cpu().numpy()

                # actual per-player exposure & pts from flattened sup tensors
                sup_pair_game = batch_cpu["sup_pair_game"].cpu().numpy()
                sup_pair_side = batch_cpu["sup_pair_side"].cpu().numpy()
                sup_pair_off = batch_cpu["sup_pair_off"].cpu().numpy()
                sup_pair_y = batch_cpu["sup_pair_y"].cpu().numpy()
                sup_pl_game = batch_cpu["sup_pl_game"].cpu().numpy()
                sup_pl_side = batch_cpu["sup_pl_side"].cpu().numpy()
                sup_pl_slot = batch_cpu["sup_pl_slot"].cpu().numpy()
                sup_pl_y = batch_cpu["sup_pl_y"].cpu().numpy()

                L = mask.sum()
                rows = []
                for s in range(int(mask.size)):
                    if not mask[s]:
                        continue
                    pid = idx_to_pid.get(int(idx[s]), f"OOV/{int(idx[s])}")
                    pred_exp = float(pair_marg[s, 0])
                    pred_pts = float(box[s, pts_i])
                    pred_a_off = float(alpha_off[s])
                    # actual exposure: sum sup_pair_y[..., 0] for this game/side/off_slot
                    m_pair = (sup_pair_game == b) & (sup_pair_side == side) & (sup_pair_off == s)
                    act_exp = float(sup_pair_y[m_pair, 0].sum()) if m_pair.any() else 0.0
                    # actual per-player full box for this slot
                    m_pl = (sup_pl_game == b) & (sup_pl_side == side) & (sup_pl_slot == s)
                    if m_pl.any():
                        i_row = int(np.flatnonzero(m_pl)[0])
                        act_pts = float(sup_pl_y[i_row, pts_i])
                        act_fgm = float(sup_pl_y[i_row, fgm_i])
                        act_fga = float(sup_pl_y[i_row, fga_i])
                        act_ast = float(sup_pl_y[i_row, ast_i])
                    else:
                        act_pts = act_fgm = act_fga = act_ast = 0.0
                    pred_fgm = float(box[s, fgm_i])
                    pred_fga = float(box[s, fga_i])
                    pred_ast = float(box[s, ast_i])
                    rows.append((
                        pred_exp, pid, prob[s], pred_exp, pred_pts, pred_a_off,
                        act_exp, act_pts, float(actual_alpha[s]),
                        pred_fgm, pred_fga, pred_ast,
                        act_fgm, act_fga, act_ast,
                    ))
                rows.sort(key=lambda r: r[0], reverse=True)
                top = rows[: top_k]
                pred_total_exp = sum(r[3] for r in rows)
                act_total_exp = sum(r[6] for r in rows)
                pred_total_pts = sum(r[4] for r in rows)
                act_total_pts = sum(r[7] for r in rows)
                pred_total_a = sum(r[5] for r in rows)
                act_total_a = sum(r[8] for r in rows)
                print(f"  {side_name} L={int(L)}  "
                      f"sum_pred[exp={pred_total_exp:.1f} pts={pred_total_pts:.1f} a_off={pred_total_a:.3f}]  "
                      f"sum_actual[exp={act_total_exp:.0f} pts={act_total_pts:.0f} a_off={act_total_a:.3f}]")
                print(f"    {'pid':>10} {'prob':>5} | {'pE':>5} {'aE':>5} {'pPTS':>5} {'aPTS':>5} "
                      f"{'pAoff':>6} {'aAoff':>6} | {'pFGM':>5} {'aFGM':>5} {'pFGA':>5} {'aFGA':>5} "
                      f"{'pAST':>5} {'aAST':>5}")
                for r in top:
                    (_, pid, pr, pred_exp, pred_pts, pred_a_off,
                     act_exp, act_pts, act_a,
                     pred_fgm, pred_fga, pred_ast,
                     act_fgm, act_fga, act_ast) = r
                    pid_s = pid if isinstance(pid, str) else str(pid)
                    print(f"    {pid_s:>10} {pr:5.2f} | "
                          f"{pred_exp:5.1f} {act_exp:5.0f} {pred_pts:5.1f} {act_pts:5.0f} "
                          f"{pred_a_off:6.3f} {act_a:6.3f} | "
                          f"{pred_fgm:5.1f} {act_fgm:5.0f} {pred_fga:5.1f} {act_fga:5.0f} "
                          f"{pred_ast:5.1f} {act_ast:5.0f}")


def _track_player_predictions(
    model, test_recs, vocab, device: str,
    target_pids: list[str], batch_size: int = 64,
) -> None:
    """For every test game containing each target_pid, print pred vs actual.

    One line per (game, pid). Useful for tracking a single star's
    predicted variance across opponents.
    """
    if not test_recs or not target_pids:
        return
    pid_to_idx = {pid: vocab.player_to_idx.get(pid) for pid in target_pids}
    missing = [pid for pid, i in pid_to_idx.items() if i is None]
    if missing:
        print(f"[track] pids not in vocab: {missing}")
    pts_i = BOX_INDEX["pts"]
    fgm_i = BOX_INDEX["fgm"]
    fga_i = BOX_INDEX["fga"]
    ast_i = BOX_INDEX["ast"]

    loader = DataLoader(GameDatasetV3(test_recs), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_v3)
    model.eval()
    header_printed: set[str] = set()
    rec_iter = iter(test_recs)
    with torch.no_grad():
        for batch_cpu in loader:
            batch = {k: v.to(device) for k, v in batch_cpu.items()}
            fwd = model(batch)
            B = batch_cpu["label"].shape[0]
            recs_b = [next(rec_iter) for _ in range(B)]
            home_idx = batch_cpu["home_idx"].cpu().numpy()
            away_idx = batch_cpu["away_idx"].cpu().numpy()
            home_mask = batch_cpu["home_mask"].cpu().numpy().astype(bool)
            away_mask = batch_cpu["away_mask"].cpu().numpy().astype(bool)
            home_pair_marg = fwd["home_pair_marg"].cpu().numpy()
            away_pair_marg = fwd["away_pair_marg"].cpu().numpy()
            home_box = fwd["home_box"].cpu().numpy()
            away_box = fwd["away_box"].cpu().numpy()
            alpha_h = fwd["alpha_home_off"].cpu().numpy()
            alpha_a = fwd["alpha_away_off"].cpu().numpy()
            alpha_h_act = batch_cpu["home_alpha_off_actual"].cpu().numpy()
            alpha_a_act = batch_cpu["away_alpha_off_actual"].cpu().numpy()
            sup_pair_game = batch_cpu["sup_pair_game"].cpu().numpy()
            sup_pair_side = batch_cpu["sup_pair_side"].cpu().numpy()
            sup_pair_off = batch_cpu["sup_pair_off"].cpu().numpy()
            sup_pair_y = batch_cpu["sup_pair_y"].cpu().numpy()
            sup_pl_game = batch_cpu["sup_pl_game"].cpu().numpy()
            sup_pl_side = batch_cpu["sup_pl_side"].cpu().numpy()
            sup_pl_slot = batch_cpu["sup_pl_slot"].cpu().numpy()
            sup_pl_y = batch_cpu["sup_pl_y"].cpu().numpy()

            for pid, target_idx in pid_to_idx.items():
                if target_idx is None:
                    continue
                if pid not in header_printed:
                    print(f"\n[track pid={pid}]  one row per game; "
                          f"opp = opposing team_id; pred totals = team pts; "
                          f"pE/pPTS/pAoff = predicted; aE/aPTS/aAoff = actual")
                    print(f"    {'date':>10} {'gid':>10} {'side':>4} {'opp':>11} "
                          f"{'pE':>5} {'aE':>5} {'pPTS':>5} {'aPTS':>5} "
                          f"{'pFGM':>5} {'aFGM':>5} {'pFGA':>5} {'aFGA':>5} "
                          f"{'pAST':>5} {'aAST':>5} "
                          f"{'pAoff':>6} {'aAoff':>6}")
                    header_printed.add(pid)
                for b in range(B):
                    rec = recs_b[b]
                    side = -1
                    slot = -1
                    h_slots = np.flatnonzero((home_idx[b] == target_idx) & home_mask[b])
                    a_slots = np.flatnonzero((away_idx[b] == target_idx) & away_mask[b])
                    if h_slots.size > 0:
                        side, slot = 0, int(h_slots[0])
                        opp = rec.away_team_id
                    elif a_slots.size > 0:
                        side, slot = 1, int(a_slots[0])
                        opp = rec.home_team_id
                    else:
                        continue
                    if side == 0:
                        pred_exp = float(home_pair_marg[b, slot, 0])
                        pred_pts = float(home_box[b, slot, pts_i])
                        pred_fgm = float(home_box[b, slot, fgm_i])
                        pred_fga = float(home_box[b, slot, fga_i])
                        pred_ast = float(home_box[b, slot, ast_i])
                        pred_a_off = float(alpha_h[b, slot])
                        act_a_off = float(alpha_h_act[b, slot])
                    else:
                        pred_exp = float(away_pair_marg[b, slot, 0])
                        pred_pts = float(away_box[b, slot, pts_i])
                        pred_fgm = float(away_box[b, slot, fgm_i])
                        pred_fga = float(away_box[b, slot, fga_i])
                        pred_ast = float(away_box[b, slot, ast_i])
                        pred_a_off = float(alpha_a[b, slot])
                        act_a_off = float(alpha_a_act[b, slot])
                    m_pair = ((sup_pair_game == b) & (sup_pair_side == side)
                              & (sup_pair_off == slot))
                    act_exp = float(sup_pair_y[m_pair, 0].sum()) if m_pair.any() else 0.0
                    m_pl = ((sup_pl_game == b) & (sup_pl_side == side)
                            & (sup_pl_slot == slot))
                    if m_pl.any():
                        i_row = int(np.flatnonzero(m_pl)[0])
                        act_pts = float(sup_pl_y[i_row, pts_i])
                        act_fgm = float(sup_pl_y[i_row, fgm_i])
                        act_fga = float(sup_pl_y[i_row, fga_i])
                        act_ast = float(sup_pl_y[i_row, ast_i])
                    else:
                        act_pts = act_fgm = act_fga = act_ast = 0.0
                    side_label = "HOME" if side == 0 else "AWAY"
                    print(f"    {rec.game_date.date()!s:>10} {rec.game_id:>10} "
                          f"{side_label:>4} {opp:>11} "
                          f"{pred_exp:5.1f} {act_exp:5.0f} {pred_pts:5.1f} {act_pts:5.0f} "
                          f"{pred_fgm:5.1f} {act_fgm:5.0f} {pred_fga:5.1f} {act_fga:5.0f} "
                          f"{pred_ast:5.1f} {act_ast:5.0f} "
                          f"{pred_a_off:6.3f} {act_a_off:6.3f}")


def train_one_window(
    args: argparse.Namespace,
    train_fit_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    histories, scores, statuses, calibration, game_odds,
    matchup_db: Path, core_db: Path,
    ckpt_dir: Path | None = None,
) -> tuple[CmeV3, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train one window from scratch; return trained model and test predictions."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_gids = [str(g) for g in train_fit_df["game_id"].tolist()]
    val_gids = [str(g) for g in val_df["game_id"].tolist()]
    test_gids = [str(g) for g in test_df["game_id"].tolist()]

    train_matchup = load_matchup_rows_v2(matchup_db, train_gids)
    val_matchup = load_matchup_rows_v2(matchup_db, val_gids)
    test_matchup = load_matchup_rows_v2(matchup_db, test_gids)
    train_pl = load_player_game_stats(core_db, train_gids)
    val_pl = load_player_game_stats(core_db, val_gids)
    test_pl = load_player_game_stats(core_db, test_gids)

    vocab = build_vocab_from_records_v3(
        train_fit_df, histories, train_matchup,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(train_fit_df)
    tabular_stats = fit_tabular_stats(train_fit_df)

    common = dict(
        histories=histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=None, player_form_stats=None,
        player_form_lookback=DEFAULT_PLAYER_FORM_LOOKBACK,
        player_form_decay=DEFAULT_PLAYER_FORM_DECAY,
        game_odds=game_odds,
    )
    train_recs = build_records_v3(
        train_fit_df, matchup_rows=train_matchup, player_game_stats=train_pl, **common,
    )
    val_recs = build_records_v3(
        val_df, matchup_rows=val_matchup, player_game_stats=val_pl, **common,
    )
    test_recs = build_records_v3(
        test_df, matchup_rows=test_matchup, player_game_stats=test_pl, **common,
    )

    train_loader = DataLoader(GameDatasetV3(train_recs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_v3)
    val_loader = DataLoader(GameDatasetV3(val_recs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_v3)

    import cme_v3_common as _cv3c
    tabular_dim = 0 if args.no_tabular else len(_cv3c.TABULAR_FEATURE_COLUMNS)
    cfg = CmeV3Config(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, n_heads=args.n_heads,
        n_self_layers=args.n_self_layers, n_cross_layers=args.n_cross_layers,
        pair_hidden=args.pair_hidden, player_hidden=args.player_hidden,
        inv_hidden=args.inv_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        tabular_dim=tabular_dim, team_emb_dim=args.team_emb_dim,
        player_stat_dim=0,
        sinkhorn_iters=args.sinkhorn_iters,
        base_possessions_per_team=args.base_possessions,
        init_global_scale=args.init_scale,
    )
    model = CmeV3(cfg).to(args.device)

    box_weights = (torch.tensor(args.box_weights, dtype=torch.float32)
                   if args.box_weights is not None else default_box_weights())
    pair_weights = (torch.tensor(args.pair_weights, dtype=torch.float32)
                    if args.pair_weights is not None else default_pair_weights())
    if box_weights.numel() != K_BOX:
        raise ValueError(f"--box-weights must have length {K_BOX}")
    if pair_weights.numel() != K_PAIR:
        raise ValueError(f"--pair-weights must have length {K_PAIR}")

    history: list[dict] = []

    if args.curriculum:
        p1_task_weights = {
            "team": args.phase1_team_w, "player": args.phase1_player_w,
            "pair": args.phase1_pair_w, "inv": args.phase1_inv_w, "win": 0.0,
        }
        norms = _per_task_grad_norms(model, train_loader, args.device, box_weights, pair_weights)
        print("[grad-diag p1-init]    " + _fmt_grad_norms("", norms, p1_task_weights))

        # Phase 1: structural pretraining (no win BCE, no involvement loss).
        # Track val total loss for plateau early-stop.
        p1_optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        p1_scheduler = _build_lr_scheduler(p1_optim, args.warmup_epochs, args.phase1_epochs)
        p1_kwargs = dict(
            box_weights=box_weights, pair_weights=pair_weights,
            team_w=args.phase1_team_w, player_w=args.phase1_player_w,
            pair_w=args.phase1_pair_w,
            inv_w=args.phase1_inv_w, win_w=0.0, margin_nll_w=0.0,
        )
        p1_best_state, p1_best_epoch, p1_best_val, p1_history = _run_training_loop(
            model, train_loader, val_loader, p1_optim, p1_scheduler,
            p1_kwargs, args.phase1_epochs, args.phase1_patience,
            args.device, track_metric="loss", phase_label=1,
        )
        history.extend(p1_history)
        if p1_best_state is not None:
            model.load_state_dict(p1_best_state)

        norms = _per_task_grad_norms(model, train_loader, args.device, box_weights, pair_weights)
        print("[grad-diag p1-end]     " + _fmt_grad_norms("", norms, p1_task_weights))

        # Phase 2: cfg-D weights — BCE + exposure-only pair Poisson.
        ckpt_pair_weights = torch.zeros(K_PAIR)
        ckpt_pair_weights[0] = 1.0
        ckpt_box_weights = box_weights
        ckpt_team_w = 0.0
        ckpt_player_w = 0.0
        ckpt_pair_w = 1.0
        ckpt_inv_w = 0.0
        ckpt_win_w = 1.0
        ckpt_margin_nll_w = 0.0

        p2_optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        p2_scheduler = _build_lr_scheduler(p2_optim, args.warmup_epochs, args.epochs)
        p2_kwargs = dict(
            box_weights=ckpt_box_weights, pair_weights=ckpt_pair_weights,
            team_w=ckpt_team_w, player_w=ckpt_player_w, pair_w=ckpt_pair_w,
            inv_w=ckpt_inv_w, win_w=ckpt_win_w, margin_nll_w=ckpt_margin_nll_w,
        )
        best_state, best_epoch, best_val, p2_history = _run_training_loop(
            model, train_loader, val_loader, p2_optim, p2_scheduler,
            p2_kwargs, args.epochs, args.patience,
            args.device, track_metric="bce", phase_label=2,
        )
        history.extend(p2_history)
        last_epoch = p2_history[-1]["epoch"] if p2_history else 0

        p2_task_weights = {
            "team": ckpt_team_w, "player": ckpt_player_w,
            "pair": ckpt_pair_w, "inv": ckpt_inv_w, "win": ckpt_win_w,
        }
        norms = _per_task_grad_norms(model, train_loader, args.device,
                                     ckpt_box_weights, ckpt_pair_weights)
        print("[grad-diag p2-end]     " + _fmt_grad_norms("", norms, p2_task_weights))
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = _build_lr_scheduler(optim, args.warmup_epochs, args.epochs)
        epoch_kwargs = dict(
            box_weights=box_weights, pair_weights=pair_weights,
            team_w=args.team_w, player_w=args.player_w, pair_w=args.pair_w,
            inv_w=args.inv_w, win_w=args.win_w, margin_nll_w=args.margin_nll_w,
        )
        best_state, best_epoch, best_val, single_history = _run_training_loop(
            model, train_loader, val_loader, optim, scheduler,
            epoch_kwargs, args.epochs, args.patience,
            args.device, track_metric="bce", phase_label=0,
        )
        history.extend(single_history)
        last_epoch = single_history[-1]["epoch"] if single_history else 0

        ckpt_box_weights = box_weights
        ckpt_pair_weights = pair_weights
        ckpt_team_w = args.team_w
        ckpt_player_w = args.player_w
        ckpt_pair_w = args.pair_w
        ckpt_inv_w = args.inv_w
        ckpt_win_w = args.win_w
        ckpt_margin_nll_w = args.margin_nll_w

    if ckpt_dir is not None and args.save_final:
        final_payload = _build_ckpt_payload(
            model, cfg, vocab, team_vocab, ckpt_box_weights, ckpt_pair_weights,
            ckpt_team_w, ckpt_player_w, ckpt_pair_w, ckpt_inv_w, ckpt_win_w,
            ckpt_margin_nll_w, last_epoch,
        )
    else:
        final_payload = None

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.inspect_players:
        _inspect_player_predictions(
            model, test_recs, vocab, args.device,
            n_games=2, top_k=args.inspect_top_k,
        )
    if args.track_pids:
        pids = [p.strip() for p in args.track_pids.split(",") if p.strip()]
        _track_player_predictions(
            model, test_recs, vocab, args.device, pids,
            batch_size=args.batch_size,
        )

    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            _build_ckpt_payload(
                model, cfg, vocab, team_vocab, ckpt_box_weights, ckpt_pair_weights,
                ckpt_team_w, ckpt_player_w, ckpt_pair_w, ckpt_inv_w, ckpt_win_w,
                ckpt_margin_nll_w, best_epoch,
            ),
            ckpt_dir / "best.pt",
        )
        if final_payload is not None:
            torch.save(final_payload, ckpt_dir / "final.pt")

    gids, probs, h_pts, a_pts, margin_mu = predict_window(
        model, test_recs, args.batch_size, args.device,
    )

    window_info = {
        "n_train_fit": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "vocab_size": vocab.size,
        "best_epoch": best_epoch,
        "best_val_bce": best_val,
        "epochs_run": len(history),
        "history": history,
    }
    return model, window_info, gids, probs, h_pts, a_pts, margin_mu


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_cyclic:
        import cme_v3_common as _cv3c
        import cme_v2_common as _cv2c
        trimmed = tuple(c for c in _cv2c.TABULAR_FEATURE_COLUMNS if not c.startswith("cyc_"))
        _cv2c.TABULAR_FEATURE_COLUMNS = trimmed
        _cv3c.TABULAR_FEATURE_COLUMNS = trimmed
        print(f"[no-cyclic] tabular columns trimmed to {len(trimmed)}")

    print(f"[device] {args.device}")
    print(f"[output] {out_dir}")

    print("[load] games + exposures + odds + scores + statuses + calibration")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    all_gids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, all_gids)
    statuses = load_game_player_status(args.injury_db, all_gids)
    calibration = load_status_calibration(args.calibration)
    game_odds = load_game_odds(args.core_db, all_gids)
    print(f"[load] n_games={len(games_all)} odds={len(game_odds)}")

    initial_train_end = pd.Timestamp(args.initial_train_end)
    windows = [
        s for s in month_starts(games_all.loc[games_all["game_date"] > initial_train_end, "game_date"])
        if s > initial_train_end
    ]
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    if not windows:
        raise ValueError("No backtest windows found after initial_train_end.")
    print(f"[plan] {len(windows)} monthly windows from {windows[0].date()} to {windows[-1].date()}")

    all_predictions: list[pd.DataFrame] = []
    window_metrics: list[dict] = []

    for wi, window_start in enumerate(windows):
        window_end = window_start + pd.offsets.MonthBegin(1)
        train_block = games_all[games_all["game_date"] < window_start].copy()
        test_block = games_all[
            (games_all["game_date"] >= window_start) & (games_all["game_date"] < window_end)
        ].copy()
        if len(train_block) == 0 or len(test_block) == 0:
            print(f"[skip] window {window_start.date()}: empty train or test")
            continue

        train_fit, val = chrono_val_split(train_block, args.val_frac_of_train)
        t0 = time.time()
        print(f"\n[window {wi+1}/{len(windows)}] {window_start.date()} "
              f"train_fit={len(train_fit)} val={len(val)} test={len(test_block)}")
        ckpt_dir = (out_dir / "windows" / window_start.date().isoformat()
                    if args.save_checkpoints else None)

        try:
            _model, info, gids, probs, h_pts, a_pts, margin_mu = train_one_window(
                args, train_fit, val, test_block, histories, scores, statuses,
                calibration, game_odds, args.matchup_db, args.core_db,
                ckpt_dir=ckpt_dir,
            )
        except Exception as exc:
            print(f"[ERROR] window {window_start.date()}: {exc}")
            raise
        dt = time.time() - t0

        gid_to_idx = {g: i for i, g in enumerate(gids)}
        order = [gid_to_idx[str(g)] for g in test_block["game_id"].astype(str).tolist()]
        probs_ordered = probs[order]
        h_pts_ordered = h_pts[order]
        a_pts_ordered = a_pts[order]
        margin_ordered = margin_mu[order]

        labels = test_block["label_home_win"].to_numpy(dtype=float)
        bce = float(-np.mean(labels * np.log(np.clip(probs_ordered, 1e-7, 1-1e-7))
                              + (1 - labels) * np.log(np.clip(1 - probs_ordered, 1e-7, 1-1e-7))))
        acc = float(np.mean((probs_ordered > 0.5).astype(float) == labels))

        win_metrics = {
            "window_start": window_start.date().isoformat(),
            "window_end": (window_end - pd.Timedelta(days=1)).date().isoformat(),
            "train_n": len(train_fit),
            "val_n": len(val),
            "test_n": len(test_block),
            "train_start": train_fit["game_date"].min().date().isoformat(),
            "train_end": train_fit["game_date"].max().date().isoformat(),
            "best_epoch": info["best_epoch"],
            "best_val_bce": info["best_val_bce"],
            "epochs_run": info["epochs_run"],
            "test_bce": bce,
            "test_acc": acc,
            "secs": dt,
        }
        window_metrics.append(win_metrics)
        print(f"[window {wi+1}/{len(windows)}] best_ep={info['best_epoch']:2d} "
              f"val_bce={info['best_val_bce']:.4f} test_bce={bce:.4f} test_acc={acc:.3f} "
              f"({dt:.0f}s)")

        pred_df = test_block[
            ["game_id", "game_date", "home_team_id", "away_team_id", "label_home_win"]
        ].copy()
        pred_df["pred_home_win_prob"] = probs_ordered
        pred_df["pred_home_pts"] = h_pts_ordered
        pred_df["pred_away_pts"] = a_pts_ordered
        pred_df["margin_mu"] = margin_ordered
        pred_df["window_start"] = window_start.date().isoformat()
        all_predictions.append(pred_df)

        predictions_df_partial = pd.concat(all_predictions, ignore_index=True)
        predictions_df_partial.to_csv(out_dir / "predictions.csv", index=False)
        pd.DataFrame(window_metrics).to_csv(out_dir / "window_metrics.csv", index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    labels = predictions_df["label_home_win"].to_numpy(dtype=float)
    probs = predictions_df["pred_home_win_prob"].to_numpy(dtype=float)
    overall_bce = float(-np.mean(labels * np.log(np.clip(probs, 1e-7, 1-1e-7))
                                  + (1 - labels) * np.log(np.clip(1 - probs, 1e-7, 1-1e-7))))
    overall_acc = float(np.mean((probs > 0.5).astype(float) == labels))
    overall_brier = float(np.mean((probs - labels) ** 2))

    overall = {
        "n": int(len(predictions_df)),
        "bce": overall_bce,
        "acc": overall_acc,
        "brier": overall_brier,
        "mean_prob": float(probs.mean()),
        "std_prob": float(probs.std()),
    }

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    predictions_df.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(window_metrics).to_csv(out_dir / "window_metrics.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2) + "\n")

    print("\n[done] overall:")
    print(json.dumps(overall, indent=2))
    print(f"[done] artifacts: {out_dir}")


if __name__ == "__main__":
    main()
