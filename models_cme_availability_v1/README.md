# CME-Availability-v1

A drop-in non-market experiment built on top of the proven `models_cme_v4_2`
branch.

## Purpose

The original CME path uses a status-only global play prior:

```text
P(player plays | injury status)
```

This branch replaces that with a learned pregame availability/rotation module
that predicts:

- player appearance probability,
- expected involvement-seconds proxy, and
- within-roster role share.

Those predictions are injected into the *same* causal bottleneck CME already
uses: player token gating and the Sinkhorn involvement softmax priors.

## Inputs used

All availability features are pregame-only:

- latest pre-tip injury status, reason text buckets, and minutes-to-tip,
- the existing status-only play calibration as a stable prior,
- strictly-before-game rolling team/player exposure history.

## Architecture sketch

```text
status prior + report timing + recent role/history
                    ↓
      learned availability / minute corrections
                    ↓
 predicted play probability + predicted log involvement seconds
                    ↓
 token gating + involvement softmax priors
                    ↓
       existing CME-v4.2 world model
```

## Install

Unzip the drop-in at the root of the modeling repo.  It expects the previously
installed `models_cme_v4_2/` branch to remain present.

## Smoke test

```bash
python models_cme_availability_v1/scripts/train_cme_availability_v1.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## Full rolling backtest

```bash
python models_cme_availability_v1/scripts/backtest_cme_availability_v1.py \
  --run-name backtest_cme_availability_v1_full \
  --save-checkpoints \
  --device cuda
```

Key output:

```bash
cat models_cme_availability_v1/artifacts/backtest_cme_availability_v1_full/overall_metrics.json
```

## Benchmark

The current non-market champion is CME-v4.2:

```json
{
  "bce": 0.6393162641,
  "acc": 0.6428241809,
  "brier": 0.2238565950
}
```

## New prediction diagnostics

The availability backtest adds:

- `home_avail_play_prob_mean`
- `away_avail_play_prob_mean`
- `home_avail_play_actual_rate`
- `away_avail_play_actual_rate`
- `home_avail_log_seconds_pred_mean`
- `away_avail_log_seconds_pred_mean`
- `availability_role_prior_strength`

These help diagnose whether the new branch is learning meaningful roster
availability and rotation signals.
