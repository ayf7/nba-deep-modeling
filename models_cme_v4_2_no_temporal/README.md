# CME-v4.2-no-temporal ablation

`models_cme_v4_2_no_temporal/` is an apples-to-apples ablation of CME-v4.2.
It keeps:

- the full CME-v3/v4.1 structured world-model backbone;
- the v4.1 affine win-logit calibration head;
- the v4.2 tail-gated residual correction.

It **removes the season/temporal calibration branch** from the prediction path.

## Probability head

```text
base_logit      = predicted_margin / global_scale
affine_logit    = positive_slope * base_logit + home_logit_bias
final_logit     = affine_logit
                  + tail_gate(|affine_logit|) * bounded_context_residual
```

For CSV-schema comparability with v4.2, the model still emits:

- `pred_home_win_prob_temporal`, which is identical to `pred_home_win_prob_affine`;
- `season_logit_adjustment`, which is always zero.

No temporal/season head is trained or used.

## Why run this

Your v4.2 analysis suggested that:

- the affine head remained highly valuable;
- the tail-gated residual helped final quality;
- the season/temporal branch was not clearly responsible for the gain.

This ablation tests exactly that.

## Install

From the root of your modeling repo:

```bash
unzip cme_v4_2_no_temporal_dropin.zip
```

## Smoke test

```bash
python models_cme_v4_2_no_temporal/scripts/train_cme_v4_2_no_temporal.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## One-window backtest

```bash
python models_cme_v4_2_no_temporal/scripts/backtest_cme_v4_2_no_temporal.py \
  --run-name backtest_cme_v4_2_no_temporal_smoke \
  --max-windows 1 \
  --device cuda
```

## Full backtest

```bash
python models_cme_v4_2_no_temporal/scripts/backtest_cme_v4_2_no_temporal.py \
  --run-name backtest_cme_v4_2_no_temporal_full \
  --save-checkpoints \
  --device cuda
```

## What to compare against v4.2

Your v4.2 benchmark was:

```text
BCE       0.6393163
Accuracy  0.6428242
Brier     0.2238566
Mean p    0.5395977
Home rate 0.5459160
```

The ablation is successful if it matches or improves those metrics while showing:

```text
mean_temporal_prob ~= mean_affine_prob
mean_season_logit_adjustment = 0
```
