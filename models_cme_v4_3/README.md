# CME-v4.3 matchup-evidence ablation

`models_cme_v4_3/` is a clean ablation built on top of the best-performing
CME-v4.2 architecture.  It keeps the full v4.2 probability stack:

- affine margin calibration;
- season-phase calibration;
- tail-gated context residual;
- the original CME world-model backbone.

It adds exactly one new idea from the failed v5 experiment:

> a tightly bounded, strongly regularized **pooled lineup / matchup evidence residual**.

This isolates whether the player-state evidence path is useful **without** the
v5 uncertainty-temperature machinery or margin-NLL training changes that hurt
backtest performance.

## Probability head

```text
base_logit      = predicted_margin / global_scale
affine_logit    = positive_slope * base_logit + home_logit_bias
temporal_logit  = affine_logit + season_delta(season_phase)
final_logit     = temporal_logit
                  + tail_gate(|temporal_logit|) * bounded_context_residual
                  + tail_gate(|temporal_logit|) * bounded_matchup_evidence
```

## New matchup-evidence branch

The new branch pools:

- home lineup self-attention states;
- away lineup self-attention states;
- home-away lineup difference;
- home-away lineup interaction;
- cross-attended offensive matchup-state difference;
- the current temporal logit.

It is deliberately constrained:

```text
--matchup-evidence-hidden 64
--max-matchup-evidence-residual 0.20
--matchup-evidence-reg-w 0.10
```

The branch is zero-initialized, so the model starts exactly on the v4.2
probability path and must earn any deviation during training.

## Install

From the root of your modeling repo:

```bash
unzip cme_v4_3_dropin.zip
```

## Smoke test

```bash
python models_cme_v4_3/scripts/train_cme_v4_3.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## One-window backtest

```bash
python models_cme_v4_3/scripts/backtest_cme_v4_3.py \
  --run-name backtest_cme_v4_3_smoke \
  --max-windows 1 \
  --device cuda
```

## Full backtest

```bash
python models_cme_v4_3/scripts/backtest_cme_v4_3.py \
  --run-name backtest_cme_v4_3_full \
  --save-checkpoints \
  --device cuda
```

## New diagnostics

The backtest `predictions.csv` includes:

- `matchup_evidence_residual`

The `overall_metrics.json` includes:

- `mean_matchup_evidence_residual`

These let you see whether the branch contributes a real average correction or
mostly remains near zero under regularization.
