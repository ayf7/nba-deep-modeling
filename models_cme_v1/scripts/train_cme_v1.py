#!/usr/bin/env python3
"""Train CME-v1 with a single training signal: per-pair Poisson NLL.

Win logit, margin, and accuracy are reported as diagnostics, not used in the
loss. The hypothesis is that the Sinkhorn-constrained world model can learn
the matchup structure from pair-event supervision alone, and that downstream
quantities (team points -> margin -> win) emerge correctly from the forward
equations.
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


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models_cme_v1" / "artifacts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--pair-hidden", type=int, default=64)
    p.add_argument("--eff-hidden", type=int, default=32)
    p.add_argument("--inv-hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--pair-dropout", type=float, default=0.5)
    p.add_argument("--player-dropout", type=float, default=0.0)
    p.add_argument("--team-emb-dim", type=int, default=8)
    p.add_argument("--n-self-attn-heads", type=int, default=2)
    p.add_argument("--n-self-attn-layers", type=int, default=1)
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--base-possessions-per-team", type=float, default=491.0)
    p.add_argument("--base-eff", type=float, default=0.224)
    p.add_argument("--eff-amplitude", type=float, default=0.05)
    p.add_argument("--init-global-scale", type=float, default=9.0)
    p.add_argument("--no-tabular", action="store_true")
    p.add_argument("--use-player-stats", action="store_true")
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--margin-coef", type=float, default=0.03,
                   help="Coefficient on margin Huber loss; supervises eff head")
    p.add_argument("--margin-huber-delta", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def chrono_split(games: pd.DataFrame, val_frac: float, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def run_epoch(
    model: CmeV1,
    loader: DataLoader,
    *,
    device: str,
    optim: torch.optim.Optimizer | None,
    margin_coef: float,
    margin_huber_delta: float,
) -> dict:
    is_train = optim is not None
    model.train(is_train)

    total_n = 0
    sum_poisson = 0.0
    sum_margin_loss = 0.0
    sum_bce = 0.0
    sum_margin_abs = 0.0
    sum_logit = 0.0
    correct = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            out = model(batch)

            lam_pred = gather_lambda_for_supervision(
                out["lam_h"], out["lam_a"],
                batch["sup_game"], batch["sup_side"],
                batch["sup_off"], batch["sup_def"],
            )
            poisson = poisson_nll(lam_pred, batch["sup_exp"])

            margin_loss = F.huber_loss(
                out["margin_mu"], batch["margin"], delta=margin_huber_delta
            )

            loss = poisson + margin_coef * margin_loss

            if is_train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

            with torch.no_grad():
                bce = F.binary_cross_entropy_with_logits(out["win_logit"], batch["label"])
                margin_abs = (out["margin_mu"] - batch["margin"]).abs().mean()
                preds = (torch.sigmoid(out["win_logit"]) > 0.5).float()
                correct_bs = (preds == batch["label"]).sum().item()

        bs = batch["label"].size(0)
        total_n += bs
        sum_poisson += poisson.item() * bs
        sum_margin_loss += margin_loss.item() * bs
        sum_bce += bce.item() * bs
        sum_margin_abs += margin_abs.item() * bs
        sum_logit += out["win_logit"].sum().item()
        correct += correct_bs

    return {
        "n": total_n,
        "poisson": sum_poisson / total_n,
        "margin_loss": sum_margin_loss / total_n,
        "bce": sum_bce / total_n,
        "margin_mae": sum_margin_abs / total_n,
        "acc": correct / total_n,
        "mean_logit": sum_logit / total_n,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[device] {args.device}")
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] games + exposures + matchup rows + status + calibration")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    if args.smoke:
        games_all = games_all.sample(n=min(400, len(games_all)), random_state=args.seed)
        games_all = games_all.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        args.epochs = 3

    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)

    train_df, val_df, test_df = chrono_split(games_all, args.val_frac, args.test_frac)
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_matchup = load_matchup_rows(
        args.matchup_db, [str(g) for g in train_df["game_id"].tolist()]
    )
    val_matchup = load_matchup_rows(
        args.matchup_db, [str(g) for g in val_df["game_id"].tolist()]
    )
    test_matchup = load_matchup_rows(
        args.matchup_db, [str(g) for g in test_df["game_id"].tolist()]
    )

    print("[vocab] building from training rosters + matchup rows")
    vocab = build_vocab_from_records(
        train_df, histories, train_matchup,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(train_df)
    print(f"[vocab] players={vocab.size} teams={team_vocab.size}")

    tabular_stats = fit_tabular_stats(train_df)

    if args.use_player_stats:
        player_histories = load_player_histories(args.core_db)
        player_form_stats = fit_player_form_stats(
            player_histories, train_df,
            lookback_games=args.player_form_lookback,
            decay=args.player_form_decay,
        )
    else:
        player_histories = None
        player_form_stats = None

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
        game_odds=None,
    )
    train_recs = build_records(train_df, matchup_rows=train_matchup, **common)
    val_recs = build_records(val_df, matchup_rows=val_matchup, **common)
    test_recs = build_records(test_df, matchup_rows=test_matchup, **common)
    print(f"[records] train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

    train_loader = DataLoader(
        XfmrGameDataset(train_recs), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_xfmr,
    )
    val_loader = DataLoader(
        XfmrGameDataset(val_recs), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_xfmr,
    )
    test_loader = DataLoader(
        XfmrGameDataset(test_recs), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_xfmr,
    )

    tabular_dim = 0 if args.no_tabular else len(TABULAR_FEATURE_COLUMNS)
    cfg = CmeV1Config(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, pair_hidden=args.pair_hidden,
        eff_hidden=args.eff_hidden, inv_hidden=args.inv_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        tabular_dim=tabular_dim,
        team_emb_dim=args.team_emb_dim,
        player_stat_dim=PLAYER_FORM_DIM if args.use_player_stats else 0,
        n_self_attn_heads=args.n_self_attn_heads,
        n_self_attn_layers=args.n_self_attn_layers,
        sinkhorn_iters=args.sinkhorn_iters,
        base_possessions_per_team=args.base_possessions_per_team,
        base_eff=args.base_eff,
        eff_amplitude=args.eff_amplitude,
        init_global_scale=args.init_global_scale,
    )
    model = CmeV1(cfg).to(args.device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(epoch_idx: int) -> float:
        if args.warmup_epochs > 0 and epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / args.warmup_epochs
        denom = max(1, args.epochs - args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params:,} cfg={asdict(cfg)}")
    print(f"[optim] AdamW lr={args.lr} wd={args.weight_decay} warmup={args.warmup_epochs}ep")
    print(
        f"[loss] pair_poisson + {args.margin_coef} * margin_huber(delta={args.margin_huber_delta}); "
        f"BCE/acc reported as diagnostics"
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict] = []
    ckpt_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device=args.device, optim=optim,
                       margin_coef=args.margin_coef, margin_huber_delta=args.margin_huber_delta)
        va = run_epoch(model, val_loader, device=args.device, optim=None,
                       margin_coef=args.margin_coef, margin_huber_delta=args.margin_huber_delta)
        scheduler.step()
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]

        val_combined = va["poisson"] + args.margin_coef * va["margin_loss"]
        row = {
            "epoch": epoch,
            "train_poisson": tr["poisson"], "train_margin_loss": tr["margin_loss"],
            "train_bce": tr["bce"], "train_acc": tr["acc"], "train_margin_mae": tr["margin_mae"],
            "val_poisson": va["poisson"], "val_margin_loss": va["margin_loss"],
            "val_bce": va["bce"], "val_acc": va["acc"], "val_margin_mae": va["margin_mae"],
            "val_combined": val_combined,
            "lr": cur_lr, "secs": dt,
        }
        history.append(row)
        print(
            f"[ep{epoch:02d}] tr_pois={tr['poisson']:.4f} tr_marg={tr['margin_loss']:.2f} "
            f"tr_bce={tr['bce']:.4f} tr_acc={tr['acc']:.3f} tr_mae={tr['margin_mae']:.2f} | "
            f"va_pois={va['poisson']:.4f} va_marg={va['margin_loss']:.2f} "
            f"va_bce={va['bce']:.4f} va_acc={va['acc']:.3f} va_mae={va['margin_mae']:.2f} "
            f"va_comb={val_combined:.4f} lr={cur_lr:.2e} ({dt:.1f}s)"
        )

        # Best is selected on combined val loss.
        if val_combined < best_val - 1e-5:
            best_val = val_combined
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {"model_state": model.state_dict(), "cfg": asdict(cfg),
                 "vocab": vocab.player_to_idx,
                 "team_vocab": team_vocab.team_to_idx,
                 "player_form_means": (player_form_stats.means.tolist()
                                        if player_form_stats is not None else None),
                 "player_form_stds": (player_form_stats.stds.tolist()
                                       if player_form_stats is not None else None),
                 "epoch": epoch},
                ckpt_path,
            )
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"[early-stop] no val improvement for {args.patience} epochs")
                break

    final_path = out_dir / "final.pt"
    torch.save(
        {"model_state": model.state_dict(), "cfg": asdict(cfg),
         "vocab": vocab.player_to_idx,
         "team_vocab": team_vocab.team_to_idx,
         "player_form_means": (player_form_stats.means.tolist()
                                if player_form_stats is not None else None),
         "player_form_stds": (player_form_stats.stds.tolist()
                               if player_form_stats is not None else None),
         "epoch": epoch},
        final_path,
    )
    print(f"[best] epoch={best_epoch} val_combined={best_val:.4f}")

    print("[load best] running test")
    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state["model_state"])
    te_best = run_epoch(model, test_loader, device=args.device, optim=None,
                        margin_coef=args.margin_coef, margin_huber_delta=args.margin_huber_delta)
    print(
        f"[test:best] pois={te_best['poisson']:.4f} bce={te_best['bce']:.4f} "
        f"acc={te_best['acc']:.3f} mae={te_best['margin_mae']:.2f} "
        f"mean_logit={te_best['mean_logit']:.3f}"
    )

    print("[load final] running test")
    state = torch.load(final_path, map_location=args.device)
    model.load_state_dict(state["model_state"])
    te_final = run_epoch(model, test_loader, device=args.device, optim=None,
                         margin_coef=args.margin_coef, margin_huber_delta=args.margin_huber_delta)
    print(
        f"[test:final] pois={te_final['poisson']:.4f} bce={te_final['bce']:.4f} "
        f"acc={te_final['acc']:.3f} mae={te_final['margin_mae']:.2f} "
        f"mean_logit={te_final['mean_logit']:.3f}"
    )

    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "vocab_size": vocab.size,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "best_epoch": best_epoch,
        "best_val_combined": best_val,
        "final_epoch": epoch,
        "test_best": {k: float(v) if k != "n" else int(v) for k, v in te_best.items()},
        "test_final": {k: float(v) if k != "n" else int(v) for k, v in te_final.items()},
        "history": history,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
