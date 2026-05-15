# V3B Phase 5 — Multi-seed Formal Evaluation

**Date:** 2026-05-15

## Result

**MULTISEED_COMPLETED** — Formal paired evaluation completed across 15 pairs (3 scenarios × 5 seeds).

## Protocol

| Property | Value |
|----------|-------|
| Scenarios | `1231` (map_123), `1531` (map_53), `1910631192` (map_19) |
| Seeds | 0, 1, 2, 3, 4 |
| Total pairs | 15 |
| Tactical modes | on, off |
| Pairing | Identical seed, scenario, map; only tactical mode varies |

## Transparency

| Category | Count |
|----------|-------|
| Total games attempted | 30 (15 pairs × 2 modes) |
| Completed | 30 |
| Failed | 0 |
| Excluded | 0 |

## Aggregate statistics

| Metric | Tactical On | Tactical Off | Delta | 95% CI |
|--------|-------------|--------------|-------|--------|
| Win rate | 0.0 | 0.0 | 0.0 | — |
| Score delta (red) | — | — | 0.0 | [0.0, 0.0] |
| Score delta std | — | — | 0.0 | — |
| Route empty rate | 0.0 | 0.0 | 0.0 | [0.0, 0.0] |
| Steps | 1801.0 | 1801.0 | 0.0 | — |
| Duration (s) | 12.27 | 10.98 | 1.29 | — |

## Per-scenario results

| Scenario | Pairs | Score Delta | Route Empty Delta | Steps Delta |
|----------|-------|-------------|-------------------|-------------|
| 1231 | 5 | 0.0 | 0.0 | 0.0 |
| 1531 | 5 | 0.0 | 0.0 | 0.0 |
| 1910631192 | 5 | 0.0 | 0.0 | 0.0 |

## Per-seed results

All 15 pairs produce score_delta=0.0, steps_delta=0, route_empty_rate_delta=0.0.

## Statistical analysis

| Statistic | Value |
|-----------|-------|
| Score delta mean | 0.0 |
| Score delta std | 0.0 |
| Bootstrap 95% CI (score) | [0.0, 0.0] |
| Effect size (Cohen's d) | undefined (std=0) |

## Critical limitation

SDK authentication blocks full game replay. The game environment stays in deployment stage (stage=1) for all 1800 steps of every game. No tactical decisions are executed because the game never starts actual battle-phase gameplay. This is because the SDK's `.engine_config` is bound to MAC `02:42:ac:11:00:02`, but the current environment has MAC `02:42:ac:11:00:07`.

## Conclusion

The replay infrastructure works correctly (30/30 games completed, no crashes, no errors). However, all paired comparisons are identical because no meaningful gameplay occurs. Outcome-level assessment of tactical contribution is **not possible** under the current authentication constraint.
