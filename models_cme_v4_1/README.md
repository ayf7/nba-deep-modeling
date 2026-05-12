# CME-v4.1 calibrated-margin world model

`models_cme_v4_1/` is a drop-in successor to the v4 experiment.  It keeps the
entire CME-v3/v4 player-matchup world model intact—player availability,
involvement shares, Sinkhorn-constrained pair exposure, pair event rates,
player box-score assembly, and team/margin supervision—but replaces the final
win-probability calibration path with a more targeted **affine calibrated
margin head**.

## Why v4.1 exists

The v3 backtest was informative:

- the model had useful ranking signal, but aggregate home-win probability was
  too low;
- `pred_home_win_prob` was essentially `sigmoid(margin / 12)`, so a margin bias
  became a probability bias.

v4 added a home intercept plus a residual context head.  Your v4 full backtest
moved mean home probability from roughly `0.508` to `0.522` and slightly
improved BCE/accuracy, but the learned **global home bias stayed small** while
the residual branch carried most of the correction.

v4.1 changes the probability head to:

```text
base_logit   = predicted_margin / global_scale
affine_logit = positive_slope * base_logit + home_logit_bias
final_logit  = affine_logit + bounded_context_residual
```

Defaults are intentionally targeted:

- `--init-home-logit-bias 0.15`
- `--init-win-logit-slope 1.0`
- `--max-calibration-residual 0.35`
- `--calibration-reg-w 0.05`
- `--calibration-slope-reg-w 0.001`
- `--calibration-lr-mult 5.0`

The home intercept, affine slope, global scale if enabled, and residual head
train with **no weight decay** and a separate learning-rate multiplier.

## Install

Unzip this bundle from the repo root:

```bash
unzip cme_v4_1_dropin.zip
```

## Smoke test

```bash
python models_cme_v4_1/scripts/train_cme_v4_1.py   --run-name smoke   --smoke   --device cuda
```

## Full backtest

```bash
python models_cme_v4_1/scripts/backtest_cme_v4_1.py   --run-name backtest_cme_v4_1_full   --save-checkpoints   --device cuda
```

## New diagnostics

The backtest `predictions.csv` adds:

- `pred_home_win_prob_base`: v3-style margin-only probability
- `pred_home_win_prob_affine`: after slope + home bias, before residual
- `win_calibration_residual`: final bounded context correction
- `home_logit_bias`: learned global home intercept
- `win_logit_slope`: learned affine slope on the base margin logit

The `overall_metrics.json` adds corresponding means so you can tell whether
v4.1 is fixing calibration through the intended affine path instead of letting
the residual absorb the whole shift.
