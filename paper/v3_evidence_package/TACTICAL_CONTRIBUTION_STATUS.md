# Tactical Contribution Status

**State:** `TACTICAL_ACTION_LEVEL_ONLY_SUPPORTED`

## Summary

The tactical branch produces nonzero, variegated scores and changes action selection in diagnostic evaluation across 5 seeds. However, outcome-level contribution (win rate, rating, promotion) cannot be evaluated because the full agent cannot load intent/situation models (pre-existing architecture mismatch).

## Key numbers

| Metric | Value |
|--------|-------|
| Candidates evaluated | 120 across 5 seeds |
| Nonzero scores | 120/120 (100%) |
| Changed actions vs off | 120/120 (100%) |
| Changed action rate | 1.0 |
| Score consistency | Std of seed means = 0.010 |
| Game-level metrics | Not available |

## Current limitation

The `intention_lstm.pth` checkpoint was saved with `fc_intent` output head, but the current `MultiHorizonIntentionModel` defines a `head` sequential module. This was fixed in V3A-FIX via `LegacyIntentModel` adapter. The SDK MAC binding that blocked V3B replay was recovered in V3C.

## SDK auth recovery (V3C)

SDK replay authentication was restored at smoke level. Formal outcome ablation must be rerun before any tactical outcome contribution claim.

## Recommended path

1. ~~Fix model weight compatibility (retrain or adapt intent/situation models)~~ — **DONE** (V3A-FIX)
2. ~~Fix SDK MAC binding~~ — **DONE** (V3C)
3. Run full-game tactical-on/off ablation with fixed auth
4. Determine outcome-level contribution

## Paper claim classification

Tactical is classified as **diagnostic/action-selection contribution** — not a performance contribution. Include as bounded technical analysis in the paper, not as a main result.
