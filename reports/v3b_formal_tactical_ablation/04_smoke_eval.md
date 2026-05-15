# V3B Phase 4 — Smoke Evaluation

**Date:** 2026-05-15

## Result

Smoke evaluation completed: 1 paired replay ran successfully.

## Protocol

| Property | Value |
|----------|-------|
| Tactical modes | on, off |
| Scenario | `1231` (map_123) |
| Seed | 0 |
| SDK data | Loaded from `land_wargame_sdk/Data/Data/` |

## Smoke results

| Metric | Tactical On | Tactical Off | Delta |
|--------|-------------|--------------|-------|
| Status | completed | completed | — |
| Steps | 1801 | 1801 | 0 |
| Winner | blue | blue | — |
| Score (red) | 104 | 104 | 0 |
| Score (blue) | 132 | 132 | 0 |
| Route empty rate | 0.0 | 0.0 | 0.0 |
| Duration (s) | 13.85 | 11.15 | 2.70 |

## Observations

1. Both modes execute without crashes.
2. Both modes produce identical results (game stuck in deployment stage due to SDK auth).
3. Tactical model is loaded (`tactical_model_loaded=True`) but never invoked during gameplay.
4. Route metrics are not recorded (gameplay never starts).

## Verdict

Smoke completes. Pipeline works end-to-end but SDK auth blocks meaningful gameplay measurement.

## Next step

Proceed to multi-seed evaluation with documented auth limitation.
