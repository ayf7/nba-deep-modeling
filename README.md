# NBA World Modeling

NBA data and modeling workspace for game prediction, backtesting, player-aware
world models, and market-disagreement research.

The main onboarding goal is:

1. create the Python/sqlite environment
2. build the local SQLite databases under `data/artifacts`
3. run a baseline train or backtest
4. move on to the CME world-model family when the data layer is working

The active modeling direction is CME-v1/v2/v3. The older v5 / Man Transformer
code is still kept as a useful reference point and counterfactual baseline, but
new architecture work should generally start from the CME models.

Generated databases and model outputs are intentionally ignored by git.

## Environment

Create the local conda environment under the project directory:

```bash
conda env create \
  --prefix "$(pwd)/env" \
  --file environment.yml
```

Activate it:

```bash
conda activate "$(pwd)/env"
```

If you prefer to create the base Python environment first and install from the
spec afterward:

```bash
conda create \
  --prefix "$(pwd)/env" \
  python=3.12.3

conda activate "$(pwd)/env"
conda env update --prefix "$(pwd)/env" --file environment.yml
```

Confirm that Python and sqlite resolve from the project environment:

```bash
which python
which sqlite3
python --version
sqlite3 --version
```

The expected sqlite path is:

```text
<repo>/env/bin/sqlite3
```

For the VS Code SQLite extension, set this in settings JSON:

```json
"sqlite.sqlite3": "<absolute-path-to-this-repo>/env/bin/sqlite3"
```

## Raw Data Checkout

The raw NBA CSV archives live in the `external_repos/nba_data` submodule. If
this is a fresh checkout, initialize it before building databases:

```bash
git submodule update --init --recursive
```

The converter expects archives under:

```text
external_repos/nba_data/datasets
```

## Build SQLite Databases

All generated SQLite artifacts live under:

```text
data/artifacts/
```

The normal build order is:

1. `nba_raw.sqlite`
2. `oddsportal_moneyline.sqlite`, optional but useful for market comparison
3. `nba_core.sqlite`
4. `player_matchup_training.sqlite`
5. `features.sqlite`
6. `nba_injury_history.sqlite` and `status_play_calibration.json` for player-aware models

### 1. Convert NBA Archives

```bash
python data/scripts/convert_nba_archives.py \
  --seasons 2020 2021 2022 2023 2024 2025
```

This creates:

```text
data/artifacts/nba_raw.sqlite
```

For a smaller smoke build, restrict sources or seasons:

```bash
python data/scripts/convert_nba_archives.py \
  --seasons 2024 2025 \
  --sources cdnnba shotdetail matchups
```

### 2. Build or Rebuild Moneyline Odds

If cached OddsPortal page JSON already exists, rebuild only SQLite:

```bash
python data/scripts/probe_oddsportal.py --rebuild-sqlite-only
```

To fetch or resume several NBA seasons:

```bash
python data/scripts/probe_oddsportal.py \
  --season-start-years 2020 2021 2022 2023 2024 2025 \
  --current-season-start-year 2025 \
  --all-pages \
  --sleep-seconds 1.0
```

This creates:

```text
data/artifacts/oddsportal_moneyline.sqlite
```

### 3. Build Core Tables

```bash
python data/scripts/build_nba_core.py \
  --seasons 2020 2021 2022 2023 2024 2025
```

This creates:

```text
data/artifacts/nba_core.sqlite
```

If odds are not available yet, use:

```bash
python data/scripts/build_nba_core.py \
  --seasons 2020 2021 2022 2023 2024 2025 \
  --skip-odds
```

### 4. Build Player Matchup Rows

Player-aware models use grouped offensive-player versus defender rows:

```bash
python data/scripts/build_player_matchup_training.py
```

This creates:

```text
data/artifacts/player_matchup_training.sqlite
```

### 5. Build Game Feature Tables

Baseline models and neural models both use the model-facing game table:

```bash
python data/scripts/build_features.py
```

This creates:

```text
data/artifacts/features.sqlite
```

### 6. Build Injury Availability

Player-aware models expect:

```text
data/artifacts/nba_injury_history.sqlite
data/artifacts/status_play_calibration.json
```

If the injury database is missing, build and resolve snapshots:

```bash
python data/scripts/build_injury_history.py \
  --start-date 2021-10-19 \
  --sleep 0.0

python data/scripts/resolve_injury_player_ids.py

python data/scripts/build_game_player_availability.py
```

`status_play_calibration.json` maps injury-report statuses to play
probabilities. Player-aware scripts read it by default from
`data/artifacts/status_play_calibration.json`; pass `--calibration` to use a
different file.

### Sanity Check Databases

```bash
sqlite3 data/artifacts/nba_core.sqlite '.tables'
sqlite3 data/artifacts/features.sqlite '.tables'
sqlite3 data/artifacts/features.sqlite 'SELECT COUNT(*) FROM model_games;'
sqlite3 data/artifacts/player_matchup_training.sqlite \
  'SELECT COUNT(*) FROM matchup_training_rows;'
```

For table-level details, see [data/README.md](data/README.md).

## Train Baseline Models

The fastest way to verify that the data layer works is to train one baseline
on `data/artifacts/features.sqlite`.

Logistic regression:

```bash
python models_baseline/scripts/train_baseline_logistic_regression.py
```

XGBoost:

```bash
python models_baseline/scripts/train_baseline_xgboost.py
```

MLP:

```bash
python models_baseline/scripts/train_baseline_mlp.py --device cpu
```

Default baseline outputs go to:

```text
models_baseline/artifacts/baseline_logistic/
models_baseline/artifacts/baseline_xgboost/
models_baseline/artifacts/baseline_mlp/
```

Each run writes metrics, predictions, config, and the fitted model artifact.

## Backtest Baseline Models

Baseline backtests use expanding monthly windows. For each month after
`--initial-train-end`, the script trains on all prior games and predicts the
games in that month.

```bash
python models_baseline/scripts/backtest_baselines.py --model logistic
python models_baseline/scripts/backtest_baselines.py --model xgboost
python models_baseline/scripts/backtest_baselines.py --model mlp --mlp-device cpu
```

Default outputs:

```text
models_baseline/artifacts/backtest_logistic/
models_baseline/artifacts/backtest_xgboost/
models_baseline/artifacts/backtest_mlp/
```

Important files:

```text
overall_metrics.json
window_metrics.csv
predictions.csv
```

## Evaluate Betting Strategies

Backtest prediction CSVs can be evaluated against matched moneyline odds from
`nba_core.sqlite`.

Expected-value threshold:

```bash
python models_baseline/scripts/evaluate_betting_strategy.py \
  --predictions models_baseline/artifacts/backtest_xgboost/predictions.csv \
  --strategy expected_value \
  --min-ev 0.02 \
  --output-dir models_baseline/artifacts/betting_xgboost_ev_002
```

Simple edge threshold:

```bash
python models_baseline/scripts/evaluate_betting_strategy.py \
  --predictions models_baseline/artifacts/backtest_xgboost/predictions.csv \
  --strategy edge_threshold \
  --threshold 0.03 \
  --output-dir models_baseline/artifacts/betting_xgboost_edge_003
```

Outputs include:

```text
summary.json
decisions.csv
bets.csv
monthly_summary.csv
```

## Train Player-Aware Models

Once the core, feature, matchup, injury, and calibration artifacts exist, the
active CME models can be trained.

CME-v1 smoke run:

```bash
python models_cme_v1/scripts/train_cme_v1.py \
  --run-name smoke \
  --smoke
```

CME-v2 smoke run:

```bash
python models_cme_v2/scripts/train_cme_v2.py \
  --run-name smoke \
  --smoke
```

CME-v3 smoke run:

```bash
python models_cme_v3/scripts/train_cme_v3.py \
  --run-name smoke \
  --smoke
```

The v5 / Man Transformer path is retained for comparison and legacy
counterfactuals:

```bash
python models_man_xfmr/scripts/train_man_xfmr.py \
  --run-name smoke \
  --smoke
```

For real runs, remove `--smoke`, set a useful `--run-name`, and use
`--device cuda` when a GPU is available.

## Backtest Player-Aware Models

Neural backtests are much more expensive than the tabular baselines because
they retrain one model per monthly window.

CME-v1:

```bash
python models_cme_v1/scripts/backtest_cme_v1.py \
  --run-name backtest_cme_v1 \
  --save-checkpoints
```

CME-v2 one-window smoke backtest:

```bash
python models_cme_v2/scripts/backtest_cme_v2.py \
  --run-name backtest_cme_v2_smoke \
  --max-windows 1
```

CME-v2 full backtest:

```bash
python models_cme_v2/scripts/backtest_cme_v2.py \
  --run-name backtest_s_tt \
  --save-checkpoints
```

CME-v3 one-window smoke backtest:

```bash
python models_cme_v3/scripts/backtest_cme_v3.py \
  --run-name backtest_cme_v3_smoke \
  --max-windows 1
```

The v5 / Man Transformer backtest remains available as a legacy comparison:

```bash
python models_man_xfmr/scripts/backtest_man_xfmr.py \
  --run-name backtest_man_xfmr \
  --save-checkpoints
```

The neural backtest prediction CSVs use the same broad schema as the baseline
backtests, so they can usually be fed into the same betting-evaluation scripts.

## Repository Map

```text
data/
  scripts/          # SQLite builders and raw-data conversion
  artifacts/        # ignored generated databases

models_baseline/
  scripts/          # logistic, XGBoost, MLP, backtests, betting eval
  artifacts/        # ignored baseline outputs

models_cme_v1/
  scripts/          # constrained matchup-event model
  artifacts/        # ignored CME-v1 outputs

models_cme_v2/
  scripts/          # structured stat/world-model experiments
  artifacts/        # ignored CME-v2 outputs

models_cme_v3/
  scripts/          # active CME world-model iteration
  artifacts/        # ignored CME-v3 outputs

models_man_xfmr/
  scripts/          # legacy v5 player-aware transformer and counterfactuals
  artifacts/        # ignored v5 outputs

docs/
  context.md        # project-level research context
```

## Common Failure Points

- Missing raw archive tables usually means the submodule was not initialized or
  the requested seasons/sources were not converted into `nba_raw.sqlite`.
- Missing odds tables only block odds-aware workflows. Rebuild core with
  `--skip-odds` if you only need basketball features.
- Player-aware scripts require `player_matchup_training.sqlite`,
  `nba_injury_history.sqlite`, and `status_play_calibration.json`.
- If sqlite files exist but look stale, rebuild downstream artifacts after
  rebuilding upstream ones. For example, rebuild `features.sqlite` after
  rebuilding `nba_core.sqlite`.
