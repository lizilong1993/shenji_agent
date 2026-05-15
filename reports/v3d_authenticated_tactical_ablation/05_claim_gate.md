# V3D Claim Gate Decision

## Gate evaluation

### Outcome contribution supported? ❌

Required:
- ✅ Authenticated paired replay completes (15/15 pairs, 0 errors)
- ❌ Tactical-on shows positive outcome delta (mean = -1.27, CI crosses zero)
- ❌ Effect visible across multiple seeds or CI meaningfully positive (CI [-50, 34])
- ✅ Tactical branch is actually invoked (nonzero scores in all ON runs, none in OFF runs)
- ✅ Route metrics do not degrade unacceptably (0% empty rate for all runs)

**Result: FAIL** — outcome delta is not meaningfully positive.

### Weak or inconclusive? ✅

Required:
- ✅ Paired replay completes (15/15)
- ✅ Tactical is invoked (nonzero scores in every tactical-on run)
- ✅ Outcome delta is small, unstable, mixed, or CI crosses zero (mean -1.27, CI [-50, 34])

**Result: PASS**

### No contribution shown? ❌

Tactical is invoked and does affect gameplay in some seeds/scenarios (range [-264, +158]).
Mean delta is near zero but not uniformly negative, and variance is high.
Tactical shows diagnostic/action-level effect even if outcome-level benefit is absent.

**Result: FAIL** — not appropriate to declare "no contribution" when tactical clearly affects gameplay.

## Claim state

```
TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE
```

## Rationale

1. Authenticated paired tactical-on/off evaluation executes successfully (15/15 pairs)
2. Tactical branch is reliably invoked (nonzero scores in all ON runs, zero in all OFF runs)
3. Route metrics are clean (0% empty rate across all 30 games)
4. Outcome delta mean = -1.27, bootstrap 95% CI = [-50.07, 34.82], Cohen's d = -0.015
5. Win rate is identical: 0.533 for both modes
6. Scenario 1910631192 is deterministic (identical scores regardless of mode) — 6 zero deltas
7. In variable scenarios (1231, 1531), tactical-on shows positive deltas in 6/10 pairs but negative in 3/10
8. Very high variance (std=85.84) suggests seed/scenario-dependent effects
9. No evidence of route degradation, illegal actions, or other negative side effects

The evidence supports that tactical executes and affects gameplay, but consistent outcome-level benefit is not demonstrated under the tested protocol.
