# CME-v4

CME-v4 is a targeted revision of CME-v3.  It keeps v3's structured world model
intact:

- player tokenization and team context
- within-team self-attention
- cross-team offensive/defensive attention
- involvement-share heads
- Sinkhorn-constrained pair exposure matrices
- pair event rates, player box scores, team box scores, and margin prediction

The v4 change is focused on **win-probability calibration**.

## Why v4 exists

The v3 full backtest predictions showed a persistent aggregate home-side
calibration shortfall:

- mean predicted home win probability was materially below the realized home
  win rate
- the emitted probability was effectively `sigmoid(margin_mu / 12)`
- a diagnostic post-hoc calibration fit wanted almost the same slope but a
  positive intercept

That means v3 had useful ranking signal, but its probability layer had no cheap
way to correct a stable home/intercept bias without distorting the structured
score/margin world model.

## Architectural change

v4 replaces v3's final line

```python
win_logit = margin_mu / global_scale
```

with

```python
win_logit_base = margin_mu / global_scale
win_logit = win_logit_base + home_logit_bias + win_residual
```

where:

- `home_logit_bias` is a trainable scalar intercept
- `win_residual` is a small bounded context residual from a zero-initialized MLP
- the residual is regularized so it corrects calibration without replacing the
  margin/world-model path

The residual head consumes:

- home context
- away context
- context difference
- context interaction
- the base margin-derived win logit

A fresh v4 model starts behaviorally close to v3 because the residual head is
zero-initialized and the home intercept starts at `0.0` by default.

## New configuration and CLI flags

- `--calibration-hidden 64`
- `--calibration-dropout 0.05`
- `--max-calibration-residual 1.0`
- `--init-home-logit-bias 0.0`
- `--calibration-reg-w 0.01`
- `--trainable-global-scale` (optional; default remains v3-style fixed scale)

## New diagnostics

Training logs now report:

- calibrated mean predicted probability vs. realized home win rate
- base probability before calibration
- mean context residual `δ`
- learned home intercept `b`

Backtest `predictions.csv` adds:

- `pred_home_win_prob_base`
- `win_calibration_residual`
- `home_logit_bias`

Backtest `overall_metrics.json` adds:

- `mean_base_prob`
- `std_base_prob`
- `mean_win_residual`
- `mean_home_logit_bias`
- `actual_home_win_rate`

## Run commands

Smoke train:

```bash
python models_cme_v4/scripts/train_cme_v4.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

Single full train:

```bash
python models_cme_v4/scripts/train_cme_v4.py \
  --run-name cme_v4_full \
  --device cuda
```

Rolling backtest:

```bash
python models_cme_v4/scripts/backtest_cme_v4.py \
  --run-name backtest_cme_v4_full \
  --save-checkpoints \
  --device cuda
```

## First comparison to run

Use the same dataset/backtest window settings you used for v3 and compare:

- `bce`
- `brier`
- `acc`
- `mean_prob` versus `actual_home_win_rate`
- per-window `test_bce`

The main hypothesis is that v4 should reduce the home-side calibration gap and
improve BCE/Brier, even if raw accuracy moves only modestly.
