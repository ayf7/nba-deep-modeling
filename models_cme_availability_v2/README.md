# CME-Availability-v2

A rotation-aware non-market experiment built on top of the proven
`models_cme_v4_2` branch.

## Why v2 exists

Availability-v1/v1.1 regressed because they mixed two different concepts:

1. **medical availability**: is a player plausibly eligible to play?
2. **rotation impact**: is a player likely to receive meaningful game exposure?

v4.2's status-derived `prob_play` is closer to the first concept.  Candidate
roster-slot appearance is closer to the second, and the v1/v1.1 branch
therefore overpredicted player appearance and diluted the player-exposure path.

Availability-v2 keeps the v4.2 medical play mask **unchanged**.  It learns only
a separate, conservative **rotation/impact prior** that nudges the Sinkhorn
involvement softmaxes.

## Architecture sketch

```text
status-derived medical prob_play  ───────────────→ token gate (unchanged v4.2)

status + report timing + recent role/history
                    ↓
 meaningful-rotation probability + expected player-seconds
                    ↓
 centered rotation/impact role score
                    ↓
 bounded prior inside involvement softmax
                    ↓
 existing CME-v4.2 world model
```

## Supervised auxiliary targets

The v2 branch predicts:

- a **meaningful-rotation** indicator using a material player-seconds threshold,
- `log1p(involvement_seconds)`,
- within-roster role share.

These targets shape rotation/impact estimates, but they do **not** replace the
medical `prob_play` mask.

## Install

Unzip the drop-in at the root of the modeling repo.  It expects the previously
installed `models_cme_v4_2/` branch to remain present.

## Smoke test

```bash
python models_cme_availability_v2/scripts/train_cme_availability_v2.py   --run-name smoke   --smoke   --device cuda
```

## Full rolling backtest

```bash
python models_cme_availability_v2/scripts/backtest_cme_availability_v2.py   --run-name backtest_cme_availability_v2_full   --save-checkpoints   --device cuda
```

Key output:

```bash
cat models_cme_availability_v2/artifacts/backtest_cme_availability_v2_full/overall_metrics.json
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

The v2 backtest emits:

- `home_rotation_prob_mean`, `away_rotation_prob_mean`
- `home_rotation_actual_rate`, `away_rotation_actual_rate`
- `home_avail_log_seconds_pred_mean`, `away_avail_log_seconds_pred_mean`
- `availability_role_prior_strength`

For backward compatibility, legacy `home_avail_play_prob_mean` /
`away_avail_play_prob_mean` columns are also emitted and mirror the rotation
probability diagnostics.  They do **not** mean that v2 overwrites CME's medical
play prior.
