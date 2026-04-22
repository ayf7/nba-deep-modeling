# Baseline Modeling

Baseline modeling code treats the feature database as an input. This lets us train on:

```text
data/artifacts/features.sqlite
data/artifacts/features_v2.sqlite
data/artifacts/features_<experiment>.sqlite
```

without changing model code.

## Layout

```text
models_baseline/
  scripts/
    train_baseline_logistic_regression.py
    train_baseline_xgboost.py
    train_baseline_mlp.py
    backtest_baselines.py
    evaluate_betting_strategy.py
    modeling_common.py
  artifacts/        # ignored model outputs
```

## Baseline Training

Train a logistic-regression baseline from the default feature database:

```bash
python models_baseline/scripts/train_baseline_logistic_regression.py
```

Train an XGBoost baseline:

```bash
python models_baseline/scripts/train_baseline_xgboost.py
```

Train a 2-layer PyTorch MLP baseline:

```bash
python models_baseline/scripts/train_baseline_mlp.py
```

Use a different feature database:

```bash
python models_baseline/scripts/train_baseline_logistic_regression.py \
  --features-db data/artifacts/features_v2.sqlite
```

The script reads `model_games` and uses a chronological train/test split. It filters to games where both teams have at least 10 prior games by default.

The default output directories are fixed by model:

```text
models_baseline/artifacts/baseline_logistic/
models_baseline/artifacts/baseline_xgboost/
models_baseline/artifacts/baseline_mlp/
```

Each output directory contains:

```text
config.json
metrics.json
predictions.csv
model.joblib
```

The MLP writes `model.pt` and `preprocessor.joblib` instead of `model.joblib`.

## Feature Contract

The training table should have:

```text
game_id
season
game_date
home_team_id
away_team_id
label_home_win
```

All other numeric columns are treated as candidate features unless explicitly excluded.

## Backtesting

Run an expanding-window monthly backtest:

```bash
python models_baseline/scripts/backtest_baselines.py --model logistic
```

XGBoost:

```bash
python models_baseline/scripts/backtest_baselines.py --model xgboost
```

MLP:

```bash
python models_baseline/scripts/backtest_baselines.py --model mlp
```

The MLP defaults to 100 maximum epochs with early stopping patience of 15 for
each monthly refit.

The default setup trains on games through `2023-12-31`, then evaluates one calendar month at a time. Before each monthly window, the model is refit using all prior games.

Outputs:

```text
models_baseline/artifacts/backtest_logistic/
models_baseline/artifacts/backtest_xgboost/
models_baseline/artifacts/backtest_mlp/
  config.json
  overall_metrics.json
  window_metrics.csv
  predictions.csv
```

Pass `--output-dir` to write a run somewhere else.

## Betting Strategy Evaluation

Evaluate a model's backtest predictions against matched moneyline odds:

```bash
python models_baseline/scripts/evaluate_betting_strategy.py \
  --predictions models_baseline/artifacts/backtest_xgboost/predictions.csv \
  --strategy edge_threshold \
  --threshold 0.03 \
  --output-dir models_baseline/artifacts/betting_xgboost_edge_003
```

Expected-value strategy:

```bash
python models_baseline/scripts/evaluate_betting_strategy.py \
  --predictions models_baseline/artifacts/backtest_xgboost/predictions.csv \
  --strategy expected_value \
  --min-ev 0.02 \
  --output-dir models_baseline/artifacts/betting_xgboost_ev_002
```

Predicted-winner strategy:

```bash
python models_baseline/scripts/evaluate_betting_strategy.py \
  --predictions models_baseline/artifacts/backtest_xgboost/predictions.csv \
  --strategy predicted_winner \
  --output-dir models_baseline/artifacts/betting_xgboost_predicted_winner
```

Market-only baselines use the same prediction file to define the evaluation
horizon, but they ignore the model probability:

```bash
python models_baseline/scripts/evaluate_betting_strategy.py \
  --predictions models_baseline/artifacts/backtest_xgboost/predictions.csv \
  --strategy favorite \
  --output-dir models_baseline/artifacts/betting_baseline_favorite

python models_baseline/scripts/evaluate_betting_strategy.py \
  --predictions models_baseline/artifacts/backtest_xgboost/predictions.csv \
  --strategy underdog \
  --output-dir models_baseline/artifacts/betting_baseline_underdog

python models_baseline/scripts/evaluate_betting_strategy.py \
  --predictions models_baseline/artifacts/backtest_xgboost/predictions.csv \
  --strategy random \
  --random-seed 0 \
  --output-dir models_baseline/artifacts/betting_baseline_random
```

All strategies use flat `$1` stakes by default and average decimal odds from
`data/artifacts/nba_core.sqlite`.

Outputs:

```text
summary.json
decisions.csv
bets.csv
monthly_summary.csv
```
