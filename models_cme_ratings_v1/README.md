# CME-Ratings-v1: Dynamic Opponent-Adjusted Team Strength Prior

This branch keeps the successful **CME-v4.2** player/matchup world model and
adds a strictly pregame **dynamic opponent-adjusted team-strength prior**.

## Motivation

The post-v4.2 experiments suggested that more auxiliary heads or noisy extra
features often degraded the model.  A stronger next signal is an explicit
pregame team-strength prior that summarizes what has happened against the
quality of prior opponents.

The ratings artifact is built sequentially from completed games only.  For each
game it records the pregame ratings *before* that game's final score is used to
update them.

## Rating model

For each team the builder maintains:

- offensive points strength
- defensive points suppression strength
- a rolling league scoring baseline
- a rolling home-advantage margin baseline

The pregame score prior is:

```text
home_pts_prior = league_pts + off_home - def_away + 0.5 * home_adv_margin
away_pts_prior = league_pts + off_away - def_home - 0.5 * home_adv_margin
rating_margin_prior = home_pts_prior - away_pts_prior
```

The emitted artifact also stores net strengths, attack-versus-defense gaps,
team games seen, and a sigmoid-implied home win prior.

## Neural integration

The new model uses the rating prior through a final-logit residual path:

```text
final_logit = v4.2_logit
            + rating_weight * (rating_margin_prior / rating_margin_scale)
            + bounded_rating_feature_residual(...)
```

If the ratings do not help, training can push the learned rating weight and
feature residual toward zero, recovering behavior close to v4.2.

## Build the ratings artifact

```bash
python data/scripts/build_team_strength_ratings.py
```

This creates:

```text
data/artifacts/team_strength_ratings.sqlite
```

## Smoke test

```bash
python models_cme_ratings_v1/scripts/train_cme_ratings_v1.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## Full backtest

```bash
python models_cme_ratings_v1/scripts/backtest_cme_ratings_v1.py \
  --run-name backtest_cme_ratings_v1_full \
  --save-checkpoints \
  --device cuda
```

## Diagnostics

`predictions.csv` adds:

- `rating_margin_prior`
- `rating_logit_prior`
- `rating_logit_adjustment`
- `rating_feature_residual`
- `rating_logit_weight`

`overall_metrics.json` reports the mean of those fields.

## Ablation

To make the ratings contribution approximately inert while retaining the code
path, run with:

```bash
--init-rating-logit-weight 0.0 --max-rating-feature-residual 0.0
```

This is not a perfect v4.2 reproduction because the extra parameter group still
exists, but it is useful as a fast sanity check.
