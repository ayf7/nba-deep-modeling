#!/usr/bin/env python3
"""Train the PyTorch multilayer-perceptron baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "baseline_mlp"


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2-layer PyTorch MLP baseline.")
    parser.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    parser.add_argument("--table", default="model_games")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--min-games-before", type=int, default=10)
    parser.add_argument("--hidden-dim-1", type=int, default=64)
    parser.add_argument("--hidden-dim-2", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_validation(
    train_df: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if validation_fraction <= 0:
        return ordered, ordered.iloc[0:0].copy()
    validation_size = max(1, int(len(ordered) * validation_fraction))
    validation_size = min(validation_size, len(ordered) - 1)
    return ordered.iloc[:-validation_size].copy(), ordered.iloc[-validation_size:].copy()


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = to_tensor(features[start : start + batch_size], device)
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, int | float]:
    dataset = TensorDataset(to_tensor(train_x, device), to_tensor(train_y, device))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.random_state)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    validation_features = to_tensor(validation_x, device) if len(validation_x) else None
    validation_labels = to_tensor(validation_y, device) if len(validation_y) else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        if validation_features is None or validation_labels is None:
            continue

        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(validation_features), validation_labels).item()

        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)

    df = load_table(args.features_db, args.table)
    df, features = prepare_model_frame(df, args.table, args.min_games_before)
    train_df, test_df = chronological_split(df, args.train_fraction)
    train_fit_df, validation_df = split_train_validation(train_df, args.validation_fraction)

    preprocessor = build_preprocessor(features, scale=True)
    train_fit_x = preprocessor.fit_transform(train_fit_df[features]).astype("float32")
    validation_x = preprocessor.transform(validation_df[features]).astype("float32")
    train_x = preprocessor.transform(train_df[features]).astype("float32")
    test_x = preprocessor.transform(test_df[features]).astype("float32")

    train_fit_y = train_fit_df[LABEL_COLUMN].to_numpy(dtype="float32")
    validation_y = validation_df[LABEL_COLUMN].to_numpy(dtype="float32")

    model = TwoLayerMLP(
        input_dim=len(features),
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
        dropout=args.dropout,
    ).to(device)
    training_summary = train_model(
        model,
        train_fit_x,
        train_fit_y,
        validation_x,
        validation_y,
        args,
        device,
    )

    train_prob = predict_probabilities(model, train_x, device, args.batch_size)
    test_prob = predict_probabilities(model, test_x, device, args.batch_size)
    metrics = {
        "train": evaluate(train_df[LABEL_COLUMN], train_prob),
        "test": evaluate(test_df[LABEL_COLUMN], test_prob),
    }

    model_params = {
        "hidden_dim_1": args.hidden_dim_1,
        "hidden_dim_2": args.hidden_dim_2,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "validation_fraction": args.validation_fraction,
        "patience": args.patience,
        "random_state": args.random_state,
        "device": str(device),
        **training_summary,
    }
    config = {
        "features_db": str(args.features_db),
        "table": args.table,
        "model": "baseline_mlp",
        "train_fraction": args.train_fraction,
        "min_games_before": args.min_games_before,
        "model_params": model_params,
        "feature_columns": features,
        "train_date_min": str(train_df["game_date"].min()),
        "train_date_max": str(train_df["game_date"].max()),
        "test_date_min": str(test_df["game_date"].min()),
        "test_date_max": str(test_df["game_date"].max()),
    }

    predictions = test_df[
        ["game_id", "game_date", "home_team_id", "away_team_id", LABEL_COLUMN]
    ].copy()
    predictions["pred_home_win_prob"] = test_prob

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    joblib.dump(preprocessor, args.output_dir / "preprocessor.joblib")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(features),
            "feature_columns": features,
            "model_params": model_params,
        },
        args.output_dir / "model.pt",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved run artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
