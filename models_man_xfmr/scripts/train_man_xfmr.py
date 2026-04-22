#!/usr/bin/env python3
"""Train MAN-Xfmr (v4). Chronological train/val/test split."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
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
    DEFAULT_OUTPUT_ROOT,
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
    load_game_odds,
    load_game_player_status,
    load_game_scores,
    load_games,
    load_matchup_rows,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
)
from man_xfmr_model import (
    ManXfmr,
    XfmrConfig,
    gather_lambda_for_supervision,
    poisson_nll,
)


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
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pair-dropout", type=float, default=None,
                   help="Dropout inside pair_mlp (defaults to --dropout if unset)")
    p.add_argument("--player-dropout", type=float, default=0.0,
                   help="Per-slot probability of zeroing a player mask during training")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Linear LR warmup epochs before cosine decay")
    p.add_argument("--curriculum-epochs", type=int, default=0,
                   help="Train with zero_tabular=True for the first N epochs "
                        "to force lineup branch to learn (0 = off)")
    p.add_argument("--alpha-poisson", type=float, default=0.1)
    p.add_argument("--beta-margin", type=float, default=0.0)
    p.add_argument("--ev-weight", type=float, default=0.0,
                   help="lambda for soft-EV loss term: bce + alpha*poisson + lambda * -E[bet_h*profit_h + bet_a*profit_a]")
    p.add_argument("--ev-bet-sharpness", type=float, default=50.0,
                   help="Sigmoid sharpness on edge for soft-bet gating (matches residual recipe)")
    p.add_argument("--margin-head", action="store_true")
    p.add_argument("--two-stream", action="store_true",
                   help="Sum two independent logits (lineup head + tabular head)")
    p.add_argument("--no-tabular", action="store_true",
                   help="Disable the 41 team-form tabular features (lineup-encoder only)")
    p.add_argument("--team-emb-dim", type=int, default=0,
                   help="Dimension of learned per-team identity embedding fed into "
                        "the head (0 = off)")
    p.add_argument("--use-player-stats", action="store_true",
                   help="Inject per-player rolling form features into E_player "
                        f"(dim={PLAYER_FORM_DIM})")
    p.add_argument("--player-form-lookback", type=int,
                   default=DEFAULT_PLAYER_FORM_LOOKBACK,
                   help="N most-recent games used to compute rolling form")
    p.add_argument("--player-form-decay", type=float,
                   default=DEFAULT_PLAYER_FORM_DECAY,
                   help="Decay factor for rolling form (most recent = 1.0)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end pass.")
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


def ev_loss_term(
    logits: torch.Tensor,
    label: torch.Tensor,
    h_dec: torch.Tensor,
    a_dec: torch.Tensor,
    has_odds: torch.Tensor,
    bet_sharpness: float,
) -> torch.Tensor:
    """-E[bet_h*profit_h + bet_a*profit_a] over games with odds.

    Soft-bet gates: bet_h = sigmoid((p - 1/h_dec) * sharpness), nearly 1 when
    edge is positive. Mirrors residual baseline composite_loss recipe.
    """
    p = torch.sigmoid(logits)
    edge_h = p - 1.0 / h_dec
    edge_a = (1.0 - p) - 1.0 / a_dec
    bet_h = torch.sigmoid(edge_h * bet_sharpness)
    bet_a = torch.sigmoid(edge_a * bet_sharpness)
    profit_h = label * (h_dec - 1.0) - (1.0 - label)
    profit_a = (1.0 - label) * (a_dec - 1.0) - label
    expected = bet_h * profit_h + bet_a * profit_a
    denom = has_odds.sum().clamp(min=1.0)
    return -(expected * has_odds).sum() / denom


def run_epoch(
    model: ManXfmr,
    loader: DataLoader,
    *,
    device: str,
    optim: torch.optim.Optimizer | None,
    alpha: float,
    beta: float,
    ev_weight: float = 0.0,
    ev_bet_sharpness: float = 50.0,
    zero_tabular: bool = False,
) -> dict:
    is_train = optim is not None
    model.train(is_train)
    total_n = 0
    total_bce = 0.0
    total_poisson = 0.0
    total_margin = 0.0
    total_ev = 0.0
    correct = 0
    sum_logit = 0.0  # for sanity
    all_logits: list[float] = []
    all_labels: list[float] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            out = model(batch, zero_tabular=zero_tabular)
            bce = F.binary_cross_entropy_with_logits(out["win_logit"], batch["label"])

            lam_pred = gather_lambda_for_supervision(
                out["lam_h"], out["lam_a"],
                batch["sup_game"], batch["sup_side"],
                batch["sup_off"], batch["sup_def"],
            )
            poisson = poisson_nll(lam_pred, batch["sup_exp"])

            loss = bce + alpha * poisson
            margin_loss = torch.zeros((), device=device)
            if "margin_pred" in out and beta > 0:
                margin_loss = F.mse_loss(out["margin_pred"], batch["margin"])
                loss = loss + beta * margin_loss

            ev = torch.zeros((), device=device)
            if ev_weight > 0 and batch["has_odds"].sum() > 0:
                ev = ev_loss_term(
                    out["win_logit"], batch["label"],
                    batch["home_dec_odds"], batch["away_dec_odds"],
                    batch["has_odds"], ev_bet_sharpness,
                )
                loss = loss + ev_weight * ev

            if is_train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

        bs = batch["label"].size(0)
        total_n += bs
        total_bce += bce.item() * bs
        total_poisson += poisson.item() * bs
        total_margin += margin_loss.item() * bs
        total_ev += ev.item() * bs
        preds = (torch.sigmoid(out["win_logit"]) > 0.5).float()
        correct += (preds == batch["label"]).sum().item()
        sum_logit += out["win_logit"].sum().item()
        all_logits.extend(out["win_logit"].detach().cpu().numpy().tolist())
        all_labels.extend(batch["label"].detach().cpu().numpy().tolist())

    return {
        "n": total_n,
        "bce": total_bce / total_n,
        "poisson": total_poisson / total_n,
        "margin": total_margin / total_n,
        "ev": total_ev / total_n,
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
    games_all = load_games(
        args.features_db, min_games_before=args.min_games_before
    )
    if args.smoke:
        games_all = games_all.sample(n=min(400, len(games_all)), random_state=args.seed)
        games_all = games_all.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        args.epochs = 3
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    odds = load_game_odds(args.core_db, game_ids) if args.ev_weight > 0 else None
    if odds is not None:
        print(f"[odds] loaded for {len(odds)}/{len(game_ids)} games")
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)

    train_df, val_df, test_df = chrono_split(games_all, args.val_frac, args.test_frac)
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_matchup = load_matchup_rows(
        args.matchup_db, [str(g) for g in train_df["game_id"].tolist()]
    )

    print("[vocab] building from training rosters + training matchup rows")
    vocab = build_vocab_from_records(
        train_df, histories, train_matchup,
        lookback_games=args.lookback_games, decay=args.decay,
    )
    team_vocab = build_team_vocab(train_df)
    print(f"[vocab] players={vocab.size} teams={team_vocab.size}")

    print("[tabular] fitting median/mean/std on train")
    tabular_stats = fit_tabular_stats(train_df)

    if args.use_player_stats:
        print("[player-form] loading per-player histories")
        player_histories = load_player_histories(args.core_db)
        print(f"[player-form] {len(player_histories)} players")
        print("[player-form] fitting standardization on train window")
        player_form_stats = fit_player_form_stats(
            player_histories, train_df,
            lookback_games=args.player_form_lookback,
            decay=args.player_form_decay,
        )
    else:
        player_histories = None
        player_form_stats = None

    print("[records] building train/val/test")
    train_recs = build_records(
        train_df, histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        matchup_rows=train_matchup,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=odds,
    )
    val_recs = build_records(
        val_df, histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        matchup_rows=None,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=odds,
    )
    test_recs = build_records(
        test_df, histories, vocab=vocab, team_vocab=team_vocab,
        status_lookup=statuses, calibration=calibration, game_scores=scores,
        matchup_rows=None,
        lookback_games=args.lookback_games, decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=odds,
    )
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
    cfg = XfmrConfig(
        vocab_size=vocab.size, num_teams=team_vocab.size,
        d=args.d, pair_hidden=args.pair_hidden,
        dropout=args.dropout, pair_dropout=args.pair_dropout,
        player_dropout=args.player_dropout,
        margin_head=args.margin_head,
        tabular_dim=tabular_dim,
        team_emb_dim=args.team_emb_dim,
        player_stat_dim=PLAYER_FORM_DIM if args.use_player_stats else 0,
        two_stream=args.two_stream,
    )
    model = ManXfmr(cfg).to(args.device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(epoch_idx: int) -> float:
        # epoch_idx is 0-indexed; warmup linearly from 1/warmup to 1.0,
        # then cosine decay from 1.0 to 0 over remaining epochs.
        if args.warmup_epochs > 0 and epoch_idx < args.warmup_epochs:
            return (epoch_idx + 1) / args.warmup_epochs
        denom = max(1, args.epochs - args.warmup_epochs)
        progress = (epoch_idx - args.warmup_epochs) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params:,} cfg={asdict(cfg)}")
    print(
        f"[optim] AdamW lr={args.lr} wd={args.weight_decay} "
        f"warmup={args.warmup_epochs}ep curriculum={args.curriculum_epochs}ep"
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict] = []
    ckpt_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        in_curriculum = epoch <= args.curriculum_epochs
        if epoch == args.curriculum_epochs + 1 and args.curriculum_epochs > 0:
            print("[curriculum] phase ending; resetting best-val + patience counter")
            best_val = float("inf")
            epochs_since_best = 0
        tr = run_epoch(
            model, train_loader, device=args.device, optim=optim,
            alpha=args.alpha_poisson, beta=args.beta_margin,
            ev_weight=args.ev_weight, ev_bet_sharpness=args.ev_bet_sharpness,
            zero_tabular=in_curriculum,
        )
        va = run_epoch(
            model, val_loader, device=args.device, optim=None,
            alpha=args.alpha_poisson, beta=args.beta_margin,
            ev_weight=args.ev_weight, ev_bet_sharpness=args.ev_bet_sharpness,
            zero_tabular=False,
        )
        scheduler.step()
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]
        row = {
            "epoch": epoch,
            "train_bce": tr["bce"], "train_acc": tr["acc"], "train_poisson": tr["poisson"],
            "train_ev": tr["ev"],
            "val_bce": va["bce"], "val_acc": va["acc"], "val_poisson": va["poisson"],
            "val_ev": va["ev"],
            "lr": cur_lr,
            "curriculum": bool(in_curriculum),
            "secs": dt,
        }
        history.append(row)
        tag = "*" if in_curriculum else " "
        ev_str = f" tr_ev={tr['ev']:+.4f} va_ev={va['ev']:+.4f}" if args.ev_weight > 0 else ""
        print(
            f"[ep{epoch:02d}{tag}] tr_bce={tr['bce']:.4f} tr_acc={tr['acc']:.3f} "
            f"tr_pois={tr['poisson']:.4f} | va_bce={va['bce']:.4f} "
            f"va_acc={va['acc']:.3f} va_pois={va['poisson']:.4f}"
            f"{ev_str} lr={cur_lr:.2e} ({dt:.1f}s)"
        )

        if in_curriculum:
            # Don't track best or trigger early stop while curriculum is active —
            # val uses full inputs but head_tab is being held at init.
            continue

        if va["bce"] < best_val - 1e-5:
            best_val = va["bce"]
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
    print(f"[best] epoch={best_epoch} val_bce={best_val:.4f}")

    print("[load best] running test")
    state = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state["model_state"])
    te_best = run_epoch(
        model, test_loader, device=args.device, optim=None,
        alpha=args.alpha_poisson, beta=args.beta_margin,
        ev_weight=args.ev_weight, ev_bet_sharpness=args.ev_bet_sharpness,
    )
    print(
        f"[test:best] bce={te_best['bce']:.4f} acc={te_best['acc']:.3f} "
        f"poisson={te_best['poisson']:.4f} mean_logit={te_best['mean_logit']:.3f}"
    )

    print("[load final] running test")
    state = torch.load(final_path, map_location=args.device)
    model.load_state_dict(state["model_state"])
    te_final = run_epoch(
        model, test_loader, device=args.device, optim=None,
        alpha=args.alpha_poisson, beta=args.beta_margin,
        ev_weight=args.ev_weight, ev_bet_sharpness=args.ev_bet_sharpness,
    )
    print(
        f"[test:final] bce={te_final['bce']:.4f} acc={te_final['acc']:.3f} "
        f"poisson={te_final['poisson']:.4f} mean_logit={te_final['mean_logit']:.3f}"
    )

    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "vocab_size": vocab.size,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "n_test": len(test_recs),
        "best_epoch": best_epoch,
        "best_val_bce": best_val,
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
