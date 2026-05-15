# V3D Statistical Summary

## Design

- 5 seeds × 3 scenarios = 15 paired comparisons
- Bootstrap: 1000 resamples, 95% CI, seed=42

## Completion

- Attempted: 15 pairs
- Completed: 15 pairs (100%)
- Errors: 0

## Primary metric: Score delta (tactical-on minus tactical-off)

| Statistic | Value |
|---|---|
| Mean | -1.27 |
| Median | 0.0 |
| Standard deviation | 85.84 |
| Bootstrap 95% CI | [-50.07, 34.82] |
| Cohen's d | -0.015 (negligible) |
| Positive deltas | 6 |
| Negative deltas | 3 |
| Zero deltas | 6 |

## Win rate

| Mode | Wins | Games | Win rate |
|---|---|---|---|
| Tactical ON | 8 | 15 | 0.533 |
| Tactical OFF | 8 | 15 | 0.533 |
| Delta | 0 | — | 0.0 |

## Per-scenario breakdown

### Scenario 1231 (map 123)
| Seed | ON score | OFF score | Delta | ON nonzero | OFF nonzero |
|---|---|---|---|---|---|
| 0 | 30 | 20 | +10 | 15 | 0 |
| 1 | 20 | 0 | +20 | 8 | 0 |
| 2 | 10 | 30 | -20 | 8 | 0 |
| 3 | 20 | 20 | 0 | 10 | 0 |
| 4 | 122 | 44 | +78 | 23 | 0 |

### Scenario 1531 (map 53)
| Seed | ON score | OFF score | Delta | ON nonzero | OFF nonzero |
|---|---|---|---|---|---|
| 0 | 366 | 346 | +20 | 45 | 0 |
| 1 | 40 | 304 | -264 | 17 | 0 |
| 2 | 202 | 44 | +158 | 22 | 0 |
| 3 | 22 | 54 | -32 | 13 | 0 |
| 4 | 335 | 324 | +11 | 32 | 0 |

### Scenario 1910631192 (map 19) — DETERMINISTIC
| Seed | ON score | OFF score | Delta | ON nonzero | OFF nonzero |
|---|---|---|---|---|---|
| 0 | 234 | 234 | 0 | 32 | 0 |
| 1 | 234 | 234 | 0 | 32 | 0 |
| 2 | 234 | 234 | 0 | 31 | 0 |
| 3 | 234 | 234 | 0 | 32 | 0 |
| 4 | 234 | 234 | 0 | 31 | 0 |

## Route metric comparison

| Metric | Tactical ON | Tactical OFF | Delta |
|---|---|---|---|
| Route empty rate | 0.0 | 0.0 | 0.0 |
| Bootstrap 95% CI | — | — | [0.0, 0.0] |

## Tactical invocation comparison

| Metric | Tactical ON | Tactical OFF |
|---|---|---|
| Nonzero count (across all pairs) | 351 | 0 |
| Mean nonzero count per pair | 23.4 | 0.0 |

## Key observations

1. Tactical branch is reliably invoked in tactical-on mode (nonzero scores in every game)
2. Tactical-off never produces nonzero tactical scores (correct deactivation)
3. Route empty rate is 0% across ALL 30 games — tactical does not degrade routing
4. Outcome delta is highly variable: range [-264, +158], std = 85.84
5. Mean score delta is essentially zero (-1.27) with CI crossing zero
6. Scenario 1910631192 is fully deterministic (identical scores regardless of tactical mode)
7. Win rate is identical between modes (0.533)

## Assessment

Tactical outcome contribution is **weak or inconclusive**. The tactical branch affects gameplay (nonzero tactical scores, variable score deltas), but the net effect on outcome is zero on average with very high variance. No consistent positive outcome benefit is demonstrated.
