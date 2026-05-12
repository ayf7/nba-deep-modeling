# CME-v4.4: Rolling Process-Feature Input Ablation

CME-v4.4 is a data-centric follow-up to the Process-CME v1 experiment.
Process-CME v1 added noisy realized-process auxiliary targets and regressed.
v4.4 keeps the successful **CME-v4.2** architecture and loss stack unchanged,
and only appends leakage-safe rolling pregame process-style features to the
existing tabular context.

## What is new

The process artifact is built by:

```bash
python data/scripts/build_game_process_features.py
```

It creates `data/artifacts/possession_process.sqlite`, including rolling pregame
features for:

- possessions / pace proxy
- 2PA rate
- 3PA rate
- turnover rate
- offensive rebound rate
- shooting-foul-drawn rate
- opponent-allowed versions of each
- offense-vs-defense interaction gaps

v4.4 z-scores these features **within each train split/window only**, then
appends them to the normal CME-v4.2 tabular feature vector. There is no process
auxiliary target loss and no new win head.

## Usage

### 1. Build or reuse the process artifact

If `possession_process.sqlite` already exists from the Process-CME v1 experiment,
you can reuse it. Otherwise run:

```bash
python data/scripts/build_game_process_features.py
```

### 2. Smoke training run

```bash
python models_cme_v4_4/scripts/train_cme_v4_4.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

### 3. Full rolling backtest

```bash
python models_cme_v4_4/scripts/backtest_cme_v4_4.py \
  --run-name backtest_cme_v4_4_full \
  --save-checkpoints \
  --device cuda
```

### 4. Read overall metrics

```bash
cat models_cme_v4_4/artifacts/backtest_cme_v4_4_full/overall_metrics.json
```

## Comparison target

The benchmark to beat is CME-v4.2:

```text
BCE       0.6393162641
Accuracy  0.6428241809
Brier     0.2238565950
```

## Ablation switch

`--no-process-features` disables the new inputs while leaving the v4.4 scripts
and run setup otherwise intact. This is useful for confirming that any gain is
actually coming from the process-history features.
