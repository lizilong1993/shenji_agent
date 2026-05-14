# Phase 5 — Multi-Seed Tactical Ablation

**Date:** 2026-05-14

## Result

**MULTISEED_COMPLETED** — Diagnostic-level multi-seed evaluation completed.

## Protocol

- **Type:** Diagnostic-level (not formal game-level evaluation)
- **Seeds:** 0, 1, 2, 3, 4
- **Candidates per seed:** 24 (12 shoot + 12 move_city)
- **Total candidates:** 120
- **Tactical-off baseline:** Always 0.0

## Per-seed results

| Seed | Reached | Nonzero | Changed | Score Mean | Score Std |
|------|---------|---------|---------|------------|-----------|
| 0    | 24/24   | 24/24   | 24/24   | -0.126     | 0.148     |
| 1    | 24/24   | 24/24   | 24/24   | -0.143     | 0.130     |
| 2    | 24/24   | 24/24   | 24/24   | -0.148     | 0.126     |
| 3    | 24/24   | 24/24   | 24/24   | -0.147     | 0.161     |
| 4    | 24/24   | 24/24   | 24/24   | -0.126     | 0.116     |

## Aggregate statistics

| Metric | Value |
|--------|-------|
| Changed action rate | 1.0 |
| Mean of seed means | -0.138 |
| Std of seed means | 0.010 |
| Pooled score mean | -0.138 |
| Pooled score std | 0.138 |
| Bootstrap 95% CI (mean) | [-0.146, -0.129] |
| All seeds negative | Yes |
| Game-level metrics | Not available |

## Key findings

1. **Stable across seeds:** The std of seed means is 0.010, showing highly consistent behavior regardless of seed variation.
2. **Nonzero action impact:** All 120/120 candidates receive nonzero scores, and the model differentiates between candidates (std = 0.138).
3. **Negative bias:** All scores are negative (BCE logits below 0), corresponding to predicted probabilities below 0.5. This is consistent with synthetic candidates that are structurally suboptimal.

## Limitations

1. **No game-level metrics** — win rate, rating, and promotion metrics are unavailable. Outcome-level contribution is unproven.
2. **Synthetic candidates only** — real-game state distributions may produce different score distributions.
3. **Negative score bias** — the model may need calibration or fine-tuning on in-distribution game states.

## Next step

Proceed to tactical contribution decision (Phase 6).
