#!/usr/bin/env python3
"""Train baseline residual-head models with three market-conditioning modes.

Models  : logistic | xgboost | mlp
Modes   : off (control) | feature (market as input) | residual (skip-add structure)

Logistic and MLP modes use a small PyTorch shim so the residual skip is
literal: `final_logit = market_logit + f(features)` when --use-market=residual.
XGBoost residual mode uses `base_margin = market_logit` natively.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from modeling_common import (
    DEFAULT_FEATURES_DB,
    DEFAULT_OUTPUT_ROOT,
    LABEL_COLUMN,
    build_preprocessor,
    chronological_split,
    evaluate,
    load_table,
    prepare_model_frame,
)
from residual_common import (
    MARKET_KEEP_COLS,
    MARKET_LOGIT_COL,
    USE_MARKET_CHOICES,
    attach_market,
    disagree_metrics,
    feat_cols_for_mode,
)

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "residual"


# ---------------------------------------------------------------------- models


class LinearResidual(nn.Module):
    """logit = (market_logit if use_offset else 0) + linear(features)."""

    def __init__(self, input_dim: int, use_offset: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.use_offset = use_offset

    def forward(self, features: torch.Tensor, market_logit: torch.Tensor) -> torch.Tensor:
        delta = self.linear(features).squeeze(-1)
        return delta + market_logit if self.use_offset else delta


class MLPResidual(nn.Module):
    """Same MLP as baseline, but final logit = (market_logit if offset) + mlp(x)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        use_offset: bool,
    ) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        head = nn.Linear(dims[-1], 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.network = nn.Sequential(*layers)
        self.use_offset = use_offset

    def forward(self, features: torch.Tensor, market_logit: torch.Tensor) -> torch.Tensor:
        delta = self.network(features).squeeze(-1)
        return delta + market_logit if self.use_offset else delta


# ------------------------------------------------------------------- training


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def composite_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    h_dec: torch.Tensor,
    a_dec: torch.Tensor,
    market_p: torch.Tensor,
    loss_type: str,
    ev_weight: float,
    bet_sharpness: float = 50.0,
) -> torch.Tensor:
    """Composite BCE + soft-EV loss.

    EV term: for each game, soft-bet strength = sigmoid((edge) * sharpness),
    where edge = model_p - 1/dec_odds (positive when model says bet is +EV).
    Realized profit = (decimal_odds - 1) on win, -1 on loss. EV loss is
    negative expected profit; minimizing it pushes predictions outward when
    they would yield realized profit.

    market_p is unused in this implementation but kept for symmetry/future use.
    """
    bce = F.binary_cross_entropy_with_logits(logits, y)
    if loss_type == "bce":
        return bce
    p = torch.sigmoid(logits)
    edge_h = p - 1.0 / h_dec
    edge_a = (1.0 - p) - 1.0 / a_dec
    bet_h = torch.sigmoid(edge_h * bet_sharpness)
    bet_a = torch.sigmoid(edge_a * bet_sharpness)
    profit_h = y * (h_dec - 1.0) + (1.0 - y) * (-1.0)
    profit_a = (1.0 - y) * (a_dec - 1.0) + y * (-1.0)
    expected = bet_h * profit_h + bet_a * profit_a
    ev_loss = -expected.mean()
    if loss_type == "ev":
        return ev_loss
    if loss_type == "mixed":
        return bce + ev_weight * ev_loss
    raise ValueError(f"unknown loss_type: {loss_type}")


def train_torch(
    model: nn.Module,
    x_tr: np.ndarray,
    m_tr: np.ndarray,
    y_tr: np.ndarray,
    h_tr: np.ndarray,
    a_tr: np.ndarray,
    mp_tr: np.ndarray,
    x_val: np.ndarray,
    m_val: np.ndarray,
    y_val: np.ndarray,
    h_val: np.ndarray,
    a_val: np.ndarray,
    mp_val: np.ndarray,
    *,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
    loss_type: str = "bce",
    ev_weight: float = 0.0,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_tr = len(x_tr)
    rng = np.random.default_rng(seed)
    idx = np.arange(n_tr)
    has_val = len(x_val) > 0
    if has_val:
        x_val_t = to_tensor(x_val, device)
        m_val_t = to_tensor(m_val, device)
        y_val_t = to_tensor(y_val, device)
        h_val_t = to_tensor(h_val, device)
        a_val_t = to_tensor(a_val, device)
        mp_val_t = to_tensor(mp_val, device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad = 0
    last_epoch = 0
    for ep in range(1, epochs + 1):
        last_epoch = ep
        model.train()
        rng.shuffle(idx)
        for s in range(0, n_tr, batch_size):
            ii = idx[s : s + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(to_tensor(x_tr[ii], device), to_tensor(m_tr[ii], device))
            loss = composite_loss(
                logits,
                to_tensor(y_tr[ii], device),
                to_tensor(h_tr[ii], device),
                to_tensor(a_tr[ii], device),
                to_tensor(mp_tr[ii], device),
                loss_type=loss_type,
                ev_weight=ev_weight,
            )
            loss.backward()
            optimizer.step()
        if not has_val:
            continue
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t, m_val_t)
            # Track BCE on val regardless of train-loss type — this stabilizes
            # early stopping (EV loss is too noisy for stop signal).
            val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val_t).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "epochs_completed": last_epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    }


def predict_torch(
    model: nn.Module,
    x: np.ndarray,
    m: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            logits = model(
                to_tensor(x[s : s + batch_size], device),
                to_tensor(m[s : s + batch_size], device),
            )
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype="float32")


# --------------------------------------------------------------------- runner


def train_split(train_df: pd.DataFrame, val_fraction: float):
    if val_fraction <= 0:
        return train_df, train_df.iloc[0:0].copy()
    n_val = max(1, int(len(train_df) * val_fraction))
    n_val = min(n_val, len(train_df) - 1)
    return train_df.iloc[:-n_val].copy(), train_df.iloc[-n_val:].copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("logistic", "xgboost", "mlp"), required=True)
    p.add_argument("--use-market", choices=USE_MARKET_CHOICES, default="residual")
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--table", default="model_games")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    # torch shared
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--val-fraction", type=float, default=0.15)
    # loss
    p.add_argument("--loss", choices=("bce", "ev", "mixed"), default="bce")
    p.add_argument("--ev-weight", type=float, default=0.0,
                   help="lambda for mixed loss: BCE + lambda * EV_loss")
    # MLP
    p.add_argument("--hidden-dims", default="128,64,32,16")
    p.add_argument("--dropout", type=float, default=0.10)
    # XGBoost
    p.add_argument("--xgb-n-estimators", type=int, default=200)
    p.add_argument("--xgb-max-depth", type=int, default=2)
    p.add_argument("--xgb-learning-rate", type=float, default=0.05)
    p.add_argument("--xgb-subsample", type=float, default=0.9)
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    return p.parse_args()


def _extract_odds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (home_decimal_odds, away_decimal_odds, home_implied_prob) as float32."""
    h = df["home_avg_decimal_odds"].to_numpy(dtype="float32")
    a = df["away_avg_decimal_odds"].to_numpy(dtype="float32")
    mp = df["home_implied_prob_normalized"].to_numpy(dtype="float32")
    return h, a, mp


def fit_logistic(args, train_df, test_df, feat_cols, use_offset):
    device = torch.device(args.device)
    train_fit_df, val_df = train_split(train_df, args.val_fraction)
    preprocessor = build_preprocessor(feat_cols, scale=True)
    x_tr = preprocessor.fit_transform(train_fit_df[feat_cols]).astype("float32")
    x_val = preprocessor.transform(val_df[feat_cols]).astype("float32")
    x_full = preprocessor.transform(train_df[feat_cols]).astype("float32")
    x_te = preprocessor.transform(test_df[feat_cols]).astype("float32")
    m_tr = train_fit_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    m_val = val_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    m_full = train_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    m_te = test_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    y_tr = train_fit_df[LABEL_COLUMN].to_numpy(dtype="float32")
    y_val = val_df[LABEL_COLUMN].to_numpy(dtype="float32")
    h_tr, a_tr, mp_tr = _extract_odds(train_fit_df)
    h_val, a_val, mp_val = _extract_odds(val_df)
    model = LinearResidual(x_tr.shape[1], use_offset=use_offset).to(device)
    summary = train_torch(
        model, x_tr, m_tr, y_tr, h_tr, a_tr, mp_tr,
        x_val, m_val, y_val, h_val, a_val, mp_val,
        lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience, device=device, seed=args.seed,
        loss_type=args.loss, ev_weight=args.ev_weight,
    )
    return (
        predict_torch(model, x_full, m_full, device),
        predict_torch(model, x_te, m_te, device),
        summary,
    )


def fit_mlp(args, train_df, test_df, feat_cols, use_offset):
    device = torch.device(args.device)
    hidden = [int(x) for x in args.hidden_dims.split(",")]
    train_fit_df, val_df = train_split(train_df, args.val_fraction)
    preprocessor = build_preprocessor(feat_cols, scale=True)
    x_tr = preprocessor.fit_transform(train_fit_df[feat_cols]).astype("float32")
    x_val = preprocessor.transform(val_df[feat_cols]).astype("float32")
    x_full = preprocessor.transform(train_df[feat_cols]).astype("float32")
    x_te = preprocessor.transform(test_df[feat_cols]).astype("float32")
    m_tr = train_fit_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    m_val = val_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    m_full = train_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    m_te = test_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    y_tr = train_fit_df[LABEL_COLUMN].to_numpy(dtype="float32")
    y_val = val_df[LABEL_COLUMN].to_numpy(dtype="float32")
    h_tr, a_tr, mp_tr = _extract_odds(train_fit_df)
    h_val, a_val, mp_val = _extract_odds(val_df)
    model = MLPResidual(x_tr.shape[1], hidden, args.dropout, use_offset=use_offset).to(device)
    summary = train_torch(
        model, x_tr, m_tr, y_tr, h_tr, a_tr, mp_tr,
        x_val, m_val, y_val, h_val, a_val, mp_val,
        lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience, device=device, seed=args.seed,
        loss_type=args.loss, ev_weight=args.ev_weight,
    )
    return (
        predict_torch(model, x_full, m_full, device),
        predict_torch(model, x_te, m_te, device),
        summary,
    )


def fit_xgboost(args, train_df, test_df, feat_cols, use_offset):
    import xgboost as xgb
    preprocessor = build_preprocessor(feat_cols, scale=False)
    x_tr = preprocessor.fit_transform(train_df[feat_cols]).astype("float32")
    x_te = preprocessor.transform(test_df[feat_cols]).astype("float32")
    y_tr = train_df[LABEL_COLUMN].to_numpy()
    m_tr = train_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    m_te = test_df[MARKET_LOGIT_COL].to_numpy(dtype="float32")
    params = dict(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
    )
    clf = xgb.XGBClassifier(**params)
    if use_offset:
        clf.fit(x_tr, y_tr, base_margin=m_tr)
        train_prob = clf.predict_proba(x_tr, base_margin=m_tr)[:, 1]
        test_prob = clf.predict_proba(x_te, base_margin=m_te)[:, 1]
    else:
        clf.fit(x_tr, y_tr)
        train_prob = clf.predict_proba(x_tr)[:, 1]
        test_prob = clf.predict_proba(x_te)[:, 1]
    return train_prob, test_prob, params


FITTERS = {"logistic": fit_logistic, "xgboost": fit_xgboost, "mlp": fit_mlp}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = args.output_dir or (DEFAULT_OUTPUT_DIR / f"{args.model}_{args.use_market}")

    df = load_table(args.features_db, args.table)
    df, features = prepare_model_frame(df, args.table, args.min_games_before)
    df = attach_market(df)
    train_df, test_df = chronological_split(df, args.train_fraction)
    feat_cols = feat_cols_for_mode(features, args.use_market)
    use_offset = args.use_market == "residual"

    train_prob, test_prob, summary = FITTERS[args.model](
        args, train_df, test_df, feat_cols, use_offset
    )

    test_predictions = test_df[
        ["game_id", "game_date", "home_team_id", "away_team_id", LABEL_COLUMN, *MARKET_KEEP_COLS]
    ].copy()
    test_predictions["pred_home_win_prob"] = test_prob

    metrics = {
        "train": evaluate(train_df[LABEL_COLUMN], train_prob),
        "test": evaluate(test_df[LABEL_COLUMN], test_prob),
        "test_disagree_60": disagree_metrics(test_predictions, 0.60),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    config = {
        "model": args.model,
        "use_market": args.use_market,
        "use_offset": use_offset,
        "n_features_input": len(feat_cols),
        "feature_columns": feat_cols,
        "summary": summary,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
