# CME-Ratings-v2

`CME-Ratings-v2` is the ratings-first follow-up to `CME-Ratings-v1`.

The v1 backtest showed a surprising but decisive result: the **raw dynamic
opponent-adjusted ratings prior** was stronger than the v1 combined model. v2
therefore inverts the winner head:

```text
raw_rating_logit = rating_margin_prior / rating_margin_scale
ratings_anchor   = rating_slope * raw_rating_logit + rating_bias
final_win_logit  = ratings_anchor
                 + gated linear CME-vs-ratings gap correction
                 + gated bounded CME feature residual
```

The full player-aware CME-v4.2 world model is still trained and still produces
its margin, box-score, and matchup outputs. The key change is that the **ratings
prior is now the winner-prediction anchor**, while CME acts as a small residual
specialist.

## Why this design exists

From the Ratings-v1 CSV analysis:

- Raw ratings prior alone beat the v1 hybrid model.
- Ratings beat CME clearly when the two disagreed.
- The v1 hybrid appeared to under-trust ratings.
- The raw ratings prior needed a modest learned calibration sharpening.

v2 responds directly:

- learn a ratings calibration intercept and positive slope;
- anchor the final winner probability on ratings;
- allow CME to correct ratings only through a bounded, regularized path;
- suppress CME corrections when ratings are confident or ratings/CME disagree
  sharply.

## Install / artifact dependency

Unzip from the repo root:

```bash
unzip -o cme_ratings_v2_dropin.zip
```

Build or reuse the dynamic rating artifact:

```bash
python data/scripts/build_team_strength_ratings.py
```

This creates:

```text
data/artifacts/team_strength_ratings.sqlite
```

## Smoke test

```bash
python models_cme_ratings_v2/scripts/train_cme_ratings_v2.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## Full rolling backtest

```bash
python models_cme_ratings_v2/scripts/backtest_cme_ratings_v2.py \
  --run-name backtest_cme_ratings_v2_full \
  --save-checkpoints \
  --device cuda
```

Then inspect:

```bash
cat models_cme_ratings_v2/artifacts/backtest_cme_ratings_v2_full/overall_metrics.json
```

## Main diagnostics

The predictions CSV includes:

- `rating_margin_prior`
- `rating_logit_prior`
- `rating_home_win_prob_raw`
- `rating_logit_calibrated`
- `rating_home_win_prob_calibrated`
- `rating_logit_bias`
- `rating_logit_slope`
- `cme_gap`
- `disagreement_gate`
- `cme_gap_weight`
- `cme_gap_linear_residual`
- `cme_feature_residual`

The overall JSON also reports raw- and calibrated-ratings-only metrics:

- `rating_raw_bce`, `rating_raw_acc`, `rating_raw_brier`
- `rating_calibrated_bce`, `rating_calibrated_acc`, `rating_calibrated_brier`

These let you answer the most important v2 questions:

1. Did calibrating the ratings prior improve on raw ratings?
2. Did the final residual model improve on calibrated ratings?
3. Did the model keep the CME correction small when ratings were strong?

## Benchmark

The prior best non-market backtest was Ratings-v1:

```text
BCE       0.6261573536
Accuracy  0.6580526073
Brier     0.2180229467
```

The raw ratings prior analyzed from the Ratings-v1 CSV was stronger still, so
v2 is designed to target that level and improve on it with a disciplined CME
residual.
