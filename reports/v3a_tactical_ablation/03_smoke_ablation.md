# Phase 4 — Smoke Ablation

**Date:** 2026-05-14

## Result

**SMOKE_PASS** — The tactical model produces nonzero, variegated scores across all seeds, changing action rankings vs tactical-off.

## Protocol

- **Type:** Diagnostic-level smoke (not formal game-level evaluation)
- **Seeds:** 0, 1, 2 (different synthetic candidate sets via seeded random)
- **Candidates per seed:** 24 (12 shoot + 12 move_city)
- **Total candidates:** 72
- **Tactical-off baseline:** Always 0.0 score (disabled)

## Per-seed results

| Seed | Reached | Nonzero | Changed | Score Mean | Score Std |
|------|---------|---------|---------|------------|-----------|
| 0    | 24/24   | 24/24   | 24/24   | -0.126     | 0.148     |
| 1    | 24/24   | 24/24   | 24/24   | -0.143     | 0.130     |
| 2    | 24/24   | 24/24   | 24/24   | -0.148     | 0.126     |

## Aggregate

| Metric | Value |
|--------|-------|
| Change action rate | 1.0 (72/72) |
| Mean tactical-on score | -0.139 |
| Score delta (on - off) | -0.139 |
| Game-level metrics | Not available |

## Limitations

1. **Synthetic candidates** — not derived from live game replay. Score distributions may differ under real conditions.
2. **No win-rate** — game-level evaluation is blocked by pre-existing model-architecture mismatch in intent/situation models.
3. **All negative scores** — the tactical model consistently assigns sub-50% BCE probabilities to synthetic candidates. While variegated, this may reflect a bias toward predicting "not a good action" for constructed states.
4. **Not formal** — this is diagnostic-level evidence only.

## Next steps

Proceed to multi-seed diagnostic evaluation with expanded seeds.
