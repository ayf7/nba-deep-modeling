# CME-Ratings-v4.1

Ratings-v4 showed that the **roster-adjusted ratings stage** outperformed the
full ratings + roster shock + CME residual final output.  Ratings-v4.1 promotes
that stage to the official winner-probability path.

## Architecture

```text
calibrated dynamic ratings prior
        + bounded roster-shock adjustment
        -> roster-adjusted ratings logit
        -> optional post-shock affine calibration
        -> final home-win probability
```

The CME/world-model path is still computed for diagnostics, but it **does not
change the final prediction** in v4.1.

## Post-shock calibration

By default v4.1 learns a small final affine correction:

```text
post_shock_logit = slope * roster_adjusted_logit + bias
```

This can be disabled to test the exact v4 winning stage directly:

```bash
--no-post-shock-calibration
```

## Prerequisites

Reuse the artifacts already built for Ratings-v4:

```text
data/artifacts/team_strength_ratings.sqlite
data/artifacts/roster_shock_features.sqlite
```

If the roster-shock artifact is missing, rebuild it with:

```bash
python data/scripts/build_roster_shock_features.py
```

## Run

```bash
python models_cme_ratings_v4_1/scripts/train_cme_ratings_v4_1.py \
  --run-name smoke \
  --smoke \
  --device cuda

python models_cme_ratings_v4_1/scripts/backtest_cme_ratings_v4_1.py \
  --run-name backtest_cme_ratings_v4_1_full \
  --save-checkpoints \
  --device cuda
```

## New diagnostics

The backtest writes:

- `rating_home_win_prob_roster_adjusted`
- `rating_logit_post_shock_calibrated`
- `rating_home_win_prob_post_shock_calibrated`
- `post_shock_logit_bias`
- `post_shock_logit_slope`
- the existing roster-shock diagnostic fields

The overall JSON reports three distinct stages:

1. calibrated ratings prior;
2. roster-adjusted ratings;
3. post-shock calibrated ratings / final model.
