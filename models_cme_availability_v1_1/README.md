# CME-Availability-v1.1

A safer non-market availability experiment built on top of the proven
`models_cme_v4_2` branch.

## Why v1.1 exists

Availability-v1 regressed because its learned play probabilities were badly
miscalibrated: it predicted candidate-player appearance rates around 84%, while
the realized rate was about 67%.  Those overconfident probabilities were then
fed directly into CME's player-token and exposure bottlenecks.

Availability-v1.1 keeps the same basic idea but adds safeguards:

1. **Explicit play-logit calibration**
   - a learned affine calibration layer on the slot-level play logit,
   - warm-started with a negative intercept (`-0.90`) to correct the v1 bias.
2. **Conservative injection into CME**
   - calibrated play probabilities are blended with the original status prior,
   - the blend is learned and capped rather than replacing the prior outright.
3. **Small centered rotation-role prior**
   - expected involvement only nudges the exposure prior,
   - the role prior starts near zero and is capped.
4. **Extra regularization and diagnostics**
   - residual-delta regularization,
   - play-slope regularization,
   - prediction outputs showing calibrated, uncalibrated, and CME-blended play
     probabilities.

## Inputs used

All availability features remain pregame-only:

- latest pre-tip injury status, reason text buckets, and minutes-to-tip,
- the existing status-only play calibration as a stable prior,
- strictly-before-game rolling team/player exposure history.

## Architecture sketch

```text
status prior + report timing + recent role/history
                    ↓
      player-specific play/minute residuals
                    ↓
      global calibrated play probability
                    ↓
conservative blend with original status prior
                    ↓
 token gating + centered rotation-role prior
                    ↓
       existing CME-v4.2 world model
```

## Install

Unzip the drop-in at the root of the modeling repo.  It expects the previously
installed `models_cme_v4_2/` branch to remain present.

## Smoke test

```bash
python models_cme_availability_v1_1/scripts/train_cme_availability_v1_1.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## Full rolling backtest

```bash
python models_cme_availability_v1_1/scripts/backtest_cme_availability_v1_1.py \
  --run-name backtest_cme_availability_v1_1_full \
  --save-checkpoints \
  --device cuda
```

Key output:

```bash
cat models_cme_availability_v1_1/artifacts/backtest_cme_availability_v1_1_full/overall_metrics.json
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

The v1.1 backtest adds:

- `home_avail_play_prob_mean`, `away_avail_play_prob_mean`
- `home_avail_play_prob_uncalibrated_mean`, `away_avail_play_prob_uncalibrated_mean`
- `home_avail_cme_play_prob_mean`, `away_avail_cme_play_prob_mean`
- `home_avail_play_actual_rate`, `away_avail_play_actual_rate`
- `availability_play_logit_bias`, `availability_play_logit_slope`
- `availability_play_prior_mix_strength`
- `availability_role_prior_strength`

These let us check separately:

- whether the learned availability model calibrates participation correctly,
- whether CME actually uses those calibrated estimates,
- whether the conservative blend prevents the branch from damaging the v4.2
  backbone.
