"""Shared utilities for baseline model training."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_DB = PROJECT_ROOT / "data" / "artifacts" / "features.sqlite"
DEFAULT_OUTPUT_ROOT = MODELS_ROOT / "artifacts"
LABEL_COLUMN = "label_home_win"
ID_COLUMNS = {
    "game_id",
    "season",
    "game_date",
    "home_team_id",
    "away_team_id",
}


def load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = ID_COLUMNS | {LABEL_COLUMN}
    candidates = [column for column in df.columns if column not in excluded]
    numeric_candidates = []
    for column in candidates:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            df[column] = converted
            numeric_candidates.append(column)
    return numeric_candidates


CYCLIC_DOW_HARMONICS = 2
CYCLIC_DOS_HARMONICS = 3
CYCLIC_DOS_PERIOD = 200.0
CYCLIC_MOY_HARMONICS = 1


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["game_date"])
    dow = ts.dt.dayofweek.to_numpy()
    for k in range(1, CYCLIC_DOW_HARMONICS + 1):
        df[f"cyc_dow_sin{k}"] = np.sin(2.0 * np.pi * k * dow / 7.0)
        df[f"cyc_dow_cos{k}"] = np.cos(2.0 * np.pi * k * dow / 7.0)

    season_start = df.groupby("season")["game_date"].transform("min")
    dos = (ts - pd.to_datetime(season_start)).dt.days.to_numpy()
    for k in range(1, CYCLIC_DOS_HARMONICS + 1):
        df[f"cyc_dos_sin{k}"] = np.sin(2.0 * np.pi * k * dos / CYCLIC_DOS_PERIOD)
        df[f"cyc_dos_cos{k}"] = np.cos(2.0 * np.pi * k * dos / CYCLIC_DOS_PERIOD)

    moy = ts.dt.month.to_numpy()
    for k in range(1, CYCLIC_MOY_HARMONICS + 1):
        df[f"cyc_moy_sin{k}"] = np.sin(2.0 * np.pi * k * moy / 12.0)
        df[f"cyc_moy_cos{k}"] = np.cos(2.0 * np.pi * k * moy / 12.0)

    return df


def prepare_model_frame(
    df: pd.DataFrame, table: str, min_games_before: int,
    *, cyclic: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    required = {LABEL_COLUMN, "game_id", "game_date"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {table}: {', '.join(missing)}")

    if min_games_before > 0:
        df = df[
            (df["home_games_played_before"] >= min_games_before)
            & (df["away_games_played_before"] >= min_games_before)
        ].copy()

    if cyclic:
        df = add_cyclic_features(df)

    features = feature_columns(df)
    if not features:
        raise ValueError("No numeric feature columns found.")
    return df, features


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    split_idx = int(len(ordered) * train_fraction)
    split_idx = min(max(split_idx, 1), len(ordered) - 1)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def evaluate(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }
    if y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def build_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=steps),
                features,
            )
        ]
    )


def fit_and_write_run(
    *,
    model_name: str,
    classifier: Any,
    features_db: Path,
    table: str,
    output_dir: Path,
    train_fraction: float,
    min_games_before: int,
    scale: bool,
    model_params: dict[str, Any],
) -> None:
    df = load_table(features_db, table)
    df, features = prepare_model_frame(df, table, min_games_before)
    train_df, test_df = chronological_split(df, train_fraction)

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(features, scale=scale)),
            ("classifier", classifier),
        ]
    )
    model.fit(train_df[features], train_df[LABEL_COLUMN])

    train_prob = model.predict_proba(train_df[features])[:, 1]
    test_prob = model.predict_proba(test_df[features])[:, 1]
    metrics = {
        "train": evaluate(train_df[LABEL_COLUMN], train_prob),
        "test": evaluate(test_df[LABEL_COLUMN], test_prob),
    }

    config = {
        "features_db": str(features_db),
        "table": table,
        "model": model_name,
        "train_fraction": train_fraction,
        "min_games_before": min_games_before,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    joblib.dump(model, output_dir / "model.joblib")

    print(json.dumps(metrics, indent=2))
    print(f"Saved run artifacts to {output_dir}")
