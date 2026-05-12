# Process-CME v1

`models_cme_process_v1/` is the first implementation of **Experiment C**:
use a leakage-safe pregame process model to predict *how a game will be played*
and feed that learned process state into the win-probability head.

It is intentionally built on top of the current best branch, **CME-v4.2**:

- v4.2 structured player/matchup world model
- v4.2 affine + season calibration
- v4.2 tail-gated context residual
- **new:** process-history pregame features
- **new:** auxiliary process-target prediction loss
- **new:** tail-gated process logit residual

## Process targets

For both home and away teams, Process-CME predicts standardized versions of:

1. possessions
2. 2PA per possession
3. 3PA per possession
4. turnovers per possession
5. offensive rebounds per possession
6. shooting fouls drawn per possession

These are intentionally **game-style** targets, not direct duplicate labels of
final score or winner.

## Data artifact

Build the process artifact after `nba_core.sqlite` exists:

```bash
python data/scripts/build_game_process_features.py
```

The artifact is written to:

```text
data/artifacts/possession_process.sqlite
```

It contains:

- `team_game_process_actual`
- `game_process_targets`
- `game_process_pregame_features`

The builder uses imported raw `pbpstats_<season>` tables where available. For
missing game/team rows it fills with a conservative `core_proxy` derived from
shots/events unless you pass `--no-core-proxy`.

## Importing pbpstats

This drop-in updates `data/scripts/convert_nba_archives.py` so `pbpstats` is a
supported source. After downloading any desired `pbpstats_<season>.tar.xz`
archives into `external_repos/nba_data/datasets/`, import them with:

```bash
python data/scripts/convert_nba_archives.py \
  --seasons 2024 \
  --sources pbpstats
```

The public manifest available in the source data repo includes regular-season
`pbpstats` through 2024. For later seasons currently present in your CME
backtest, `build_game_process_features.py` will use `core_proxy` rows by
default so the model remains runnable.

## Smoke test

```bash
python models_cme_process_v1/scripts/train_cme_process_v1.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## Full rolling backtest

```bash
python models_cme_process_v1/scripts/backtest_cme_process_v1.py \
  --run-name backtest_cme_process_v1_full \
  --save-checkpoints \
  --device cuda
```

Outputs land in:

```text
models_cme_process_v1/artifacts/backtest_cme_process_v1_full/
```

Important files:

- `predictions.csv`
- `window_metrics.csv`
- `overall_metrics.json`

## New backtest diagnostics

The predictions file adds:

- `process_logit_residual`

The model checkpoints additionally store process feature/target normalization
statistics so reloads can use the same transformations.

## Recommended first benchmark

Compare directly against your v4.2 benchmark:

```json
{
  "bce": 0.6393162641,
  "acc": 0.6428241809,
  "brier": 0.2238565950
}
```

A meaningful Process-CME win would ideally improve accuracy and AUC without
sacrificing BCE/Brier.
