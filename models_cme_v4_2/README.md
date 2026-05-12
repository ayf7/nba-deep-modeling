# CME-v4.2 temporal-calibration world model

`models_cme_v4_2/` is a drop-in successor to CME-v4.1. It keeps the full
CME-v3/v4.1 world-model backbone intact:

- player availability and lineup tokens;
- involvement-share supervision;
- Sinkhorn-constrained pair exposure;
- pair-event rates and player box-score assembly;
- team box-score, margin, and win supervision.

v4.2 changes only the **final win-probability calibration stack**.

## Why v4.2 exists

v4.1 fixed the dominant global issue in v3: the model underpredicted home-team
win probability. In the full backtest, v4.1 learned a home-logit bias near the
post-hoc diagnostic estimate and improved BCE/Brier materially.

The remaining failure mode was narrower:

1. **calibration still drifted by rolling window**;
2. the **context residual** slightly sharpened the upper probability tail;
3. the residual helped BCE a little, so simply deleting it was unattractive.

v4.2 addresses those points without changing the structured scoring backbone.

## Probability head

```text
base_logit      = predicted_margin / global_scale
affine_logit    = positive_slope * base_logit + home_logit_bias
temporal_logit  = affine_logit + season_delta(season_phase)
final_logit     = temporal_logit
                  + tail_gate(|temporal_logit|) * bounded_context_residual
```

### New pieces

- `season_delta(season_phase)`:
  - deterministic pregame season-phase features are derived from the game date;
  - the learned logit correction is bounded and regularized;
  - this is intended to reduce repeatable early-/late-season calibration drift.

- `tail_gate(|temporal_logit|)`:
  - residual corrections remain useful around close and moderately favored games;
  - they are suppressed once the affine+season logit is already extreme;
  - this targets the small tail-overconfidence pattern seen in v4.1.

## Default calibration settings

```text
--init-home-logit-bias 0.15
--init-win-logit-slope 1.0
--max-calibration-residual 0.30
--season-calibration-hidden 16
--max-season-logit-adjustment 0.25
--tail-gate-center 1.25
--tail-gate-sharpness 2.0
--calibration-reg-w 0.075
--season-calibration-reg-w 0.02
--calibration-slope-reg-w 0.001
--calibration-lr-mult 5.0
```

The global home bias, affine slope, optional trainable global scale, season
calibration head, and residual head train with **no weight decay** and the
calibration learning-rate multiplier.

## Install

From the root of your modeling repo:

```bash
unzip cme_v4_2_dropin.zip
```

## Smoke test

```bash
python models_cme_v4_2/scripts/train_cme_v4_2.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## One-window backtest

```bash
python models_cme_v4_2/scripts/backtest_cme_v4_2.py \
  --run-name backtest_cme_v4_2_smoke \
  --max-windows 1 \
  --device cuda
```

## Full backtest

```bash
python models_cme_v4_2/scripts/backtest_cme_v4_2.py \
  --run-name backtest_cme_v4_2_full \
  --save-checkpoints \
  --device cuda
```

## New diagnostics

The v4.2 backtest `predictions.csv` includes:

- `pred_home_win_prob_base`
- `pred_home_win_prob_affine`
- `pred_home_win_prob_temporal`
- `season_logit_adjustment`
- `residual_tail_gate`
- `win_calibration_residual`
- `home_logit_bias`
- `win_logit_slope`

The `overall_metrics.json` also reports:

- `mean_temporal_prob`
- `mean_season_logit_adjustment`
- `mean_residual_tail_gate`
- `mean_win_residual`

These diagnostics let you separate:

1. margin-only probability,
2. global affine calibration,
3. season-phase calibration,
4. tail-gated context residual.
