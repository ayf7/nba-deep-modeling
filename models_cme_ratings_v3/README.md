# CME-Ratings-v3

`models_cme_ratings_v3` is a ratings-first non-market winner model built on top of the v4.2 CME world model.

## Motivation

Ratings-v2 showed that a dynamic opponent-adjusted team-strength prior was stronger than a CME-first hybrid. Ratings-v3 keeps that hierarchy, but improves the ratings engine itself with **three strictly pregame rating tracks**:

- **slow**: stronger season carryover and conservative updates
- **medium**: balanced v2-like updates
- **fast**: lower season carryover and more reactive updates

The model learns a compact ratings-only mixture of those tracks, calibrates the blended ratings logit, and permits only a small disagreement-gated CME residual.

## Architecture

```text
slow / medium / fast rating priors
            ↓
ratings-only mixture gate
            ↓
calibrated ratings anchor
            ↓
small disagreement-gated CME residual
            ↓
final home-win probability
```

This is designed to address the weak windows observed after Ratings-v2, especially periods where a single update speed may be too sluggish or too reactive.

## Build the ratings artifact

```bash
python data/scripts/build_team_strength_ratings_v3.py
```

This creates:

```text
data/artifacts/team_strength_ratings_v3.sqlite
```

## Smoke test

```bash
python models_cme_ratings_v3/scripts/train_cme_ratings_v3.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## Full backtest

```bash
python models_cme_ratings_v3/scripts/backtest_cme_ratings_v3.py \
  --run-name backtest_cme_ratings_v3_full \
  --save-checkpoints \
  --device cuda
```

Then inspect:

```bash
cat models_cme_ratings_v3/artifacts/backtest_cme_ratings_v3_full/overall_metrics.json
```

## Benchmark

Ratings-v2 reached approximately:

```text
BCE       0.59936
Accuracy  67.37%
Brier     0.20624
```

Ratings-v3 should be judged against that benchmark.
