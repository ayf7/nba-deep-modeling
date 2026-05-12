# CME-v5: uncertainty-aware matchup-evidence world model

`models_cme_v5/` is a drop-in successor to **CME-v4.2**. It keeps the full
structured player-matchup world model:

- player availability gating
- involvement-share supervision
- Sinkhorn matchup exposure transport
- pair / player / team stat heads
- margin-based win probability
- v4.2 affine home calibration
- v4.2 season-phase correction
- v4.2 tail-gated context residual

and adds two architectural changes aimed at the remaining v4.2 failure modes:

1. **Pooled lineup/matchup evidence residual**
   - v4.2's final correction only saw coarse team/game context.
   - v5 pools the learned player states and cross-attended offensive states,
     then predicts a bounded extra evidence residual.
   - This is intended to improve discrimination / AUC when the structured
     score model misses a game-specific roster or matchup signal.

2. **Uncertainty-aware temperature**
   - v4.2 still had some high-confidence misses and month-to-month fragility.
   - v5 trains the existing heteroscedastic margin uncertainty head using a
     small margin-NLL loss by default.
   - It combines that learned margin uncertainty with pregame player
     availability entropy and dampens final logits by a non-negative
     temperature when uncertainty is high.

## Final probability path

```text
base_logit     = predicted_margin / scale
affine_logit   = slope * base_logit + home_bias
temporal_logit = affine_logit + season_phase_adjustment
context_logit  = temporal_logit + tail_gate * context_residual
pretemp_logit  = context_logit + tail_gate * lineup_matchup_evidence
final_logit    = pretemp_logit / uncertainty_temperature
p(home win)    = sigmoid(final_logit)
```

## New diagnostics written by `predictions.csv`

In addition to the v4.2 columns, v5 writes:

- `pred_home_win_prob_pretemp`
- `matchup_evidence_residual`
- `uncertainty_temperature`
- `uncertainty_temperature_delta`
- `margin_sigma`
- `home_play_entropy`
- `away_play_entropy`
- `home_unavailable_prob`
- `away_unavailable_prob`

These make it possible to inspect whether v5 is gaining through actual
lineup evidence, conservative uncertainty damping, or both.

## Default new hyperparameters

```text
--matchup-evidence-hidden 96
--max-matchup-evidence-residual 0.35
--uncertainty-hidden 16
--max-uncertainty-temperature-delta 0.75
--uncertainty-temperature-init-logit -4.0
--init-margin-sigma 13.0
--margin-nll-w 0.02
--matchup-evidence-reg-w 0.05
--uncertainty-temp-reg-w 0.02
```

## Install

Unzip this folder into the repo root so you get:

```text
models_cme_v5/
```

## Smoke test

```bash
python models_cme_v5/scripts/train_cme_v5.py \
  --run-name smoke \
  --smoke \
  --device cuda
```

## One-window backtest

```bash
python models_cme_v5/scripts/backtest_cme_v5.py \
  --run-name backtest_cme_v5_smoke \
  --max-windows 1 \
  --device cuda
```

## Full backtest

```bash
python models_cme_v5/scripts/backtest_cme_v5.py \
  --run-name backtest_cme_v5_full \
  --save-checkpoints \
  --device cuda
```

## Suggested first comparison

Compare `overall_metrics.json` against your v4.2 benchmark:

```json
{
  "bce": 0.6393162641,
  "acc": 0.6428241809,
  "brier": 0.2238565950
}
```

The high-value questions are:

- Does BCE/Brier improve without losing accuracy?
- Does AUC/ranking improve in the predictions CSV?
- Does the uncertainty temperature grow above neutral on risky games?
- Does the lineup evidence residual remain bounded and non-global?
