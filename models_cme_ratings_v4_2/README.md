# CME-Ratings-v4.2

This is an **isolated experimental branch** of Ratings-v4.1 for
player-impact-aware roster-shock modeling.  It preserves the existing v4 and
v4.1 folders and uses its own roster-shock artifact database.

## What changes from v4.1

The final prediction path stays ratings-centric:

```text
calibrated dynamic ratings prior
        + bounded player-impact-aware roster-shock adjustment
        -> roster-adjusted ratings logit
        -> optional post-shock affine calibration
        -> final home-win probability
```

The CME/world-model branch remains diagnostic-only, matching v4.1.

The v4.2 roster-shock builder adds a leakage-safe offensive player-impact proxy:

```text
impact_units = points + 0.70 * assists - 1.00 * turnovers
```

That proxy is tracked with the same pregame EMA logic as existing rotation and
recent-point channels, then exposed through features such as:

- unavailable impact share;
- top-1 and top-3 unavailable impact share;
- home-away impact-shock differentials;
- impact-aware `roster_shock_advantage_signal`.

## Isolation / preservation

v4.2 writes to a separate artifact:

```text
data/artifacts/roster_shock_features_v4_2.sqlite
```

The existing v4.1 artifact remains:

```text
data/artifacts/roster_shock_features.sqlite
```

So running v4.2 does **not** overwrite v4.1 model code or v4.1 roster-shock
inputs.

## Prerequisites

The standard ratings artifact is still shared:

```text
data/artifacts/team_strength_ratings.sqlite
```

Build it if needed:

```bash
python data/scripts/build_team_strength_ratings.py
```

Build the isolated v4.2 shock artifact:

```bash
python data/scripts/build_roster_shock_features_v4_2.py --require-injury-db
```

## Run a smoke training job

```bash
python models_cme_ratings_v4_2/scripts/train_cme_ratings_v4_2.py   --run-name ratings_v4_2_player_impact_smoke   --smoke   --device cuda
```

For CPU, use `--device cpu`.

## Run a full expanding-window backtest

```bash
python models_cme_ratings_v4_2/scripts/backtest_cme_ratings_v4_2.py   --run-name backtest_ratings_v4_2_player_impact_full   --save-checkpoints   --device cuda
```

## Optional no-postcal ablation

```bash
python models_cme_ratings_v4_2/scripts/backtest_cme_ratings_v4_2.py   --run-name backtest_ratings_v4_2_player_impact_no_postcal   --save-checkpoints   --no-post-shock-calibration   --device cuda
```

## Core stage to compare

Keep comparing:

```text
rating_home_win_prob_roster_adjusted
```

against the prior Ratings-v4.1 hard-accuracy baseline.
