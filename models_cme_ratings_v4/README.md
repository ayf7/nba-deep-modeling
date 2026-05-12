# CME-Ratings-v4

Ratings-v4 keeps the ratings-first architecture that made Ratings-v2 the best
non-market winner model, then adds a **pregame roster-shock correction**.

## New idea

The team-strength prior can be stale when current roster strength differs from
recent rated strength.  `build_roster_shock_features.py` combines:

- latest pre-tip injury-report availability rows;
- each listed player's recent rotation mass estimated from completed historical
  matchup exposure seconds.

It produces a strictly pregame artifact:

```
data/artifacts/roster_shock_features.sqlite
```

Positive `roster_shock_advantage_signal` means the away team appears more
rotation-shocked than the home team.

## Architecture

```
calibrated dynamic ratings prior
        + bounded roster-shock adjustment
        + tiny disagreement-gated CME residual
        -> final win probability
```

## Run

```bash
python data/scripts/build_roster_shock_features.py
python models_cme_ratings_v4/scripts/train_cme_ratings_v4.py --run-name smoke --smoke --device cuda
python models_cme_ratings_v4/scripts/backtest_cme_ratings_v4.py --run-name backtest_cme_ratings_v4_full --save-checkpoints --device cuda
```

## New diagnostics

The backtest writes:

- `rating_home_win_prob_roster_adjusted`
- `roster_shock_advantage_signal`
- `roster_shock_signal_weight`
- `roster_shock_linear_adjustment`
- `roster_shock_feature_adjustment`
- `roster_shock_logit_adjustment`
- `has_roster_shock`

and the overall JSON reports calibrated-rating metrics, roster-adjusted-rating
metrics, and final model metrics separately.
