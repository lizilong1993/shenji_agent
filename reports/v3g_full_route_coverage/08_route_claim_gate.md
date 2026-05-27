# V3G Route Claim Gate

**Date:** 2026-05-26

(Pending full evaluation completion — this is a preliminary draft based on partial results.)

## Partial Results (14/43 scenarios)

| Metric | Value |
|---|---|
| Scenarios with 0% empty | 11/14 |
| Scenarios with high empty | 2/14 (both map 94) |
| Crashed | 1/14 (map 212) |
| Highest empty rate | 99.74% (2010131194, map 94) |
| Second highest | 99.26% (2010141294, map 94) |

## Key finding: The route failure is scenario-specific, not SDK-general

The V3F analysis claimed that `no_valid_move_candidate` is "an SDK candidate generation limitation" and "not a route planner bug." The V3G evidence shows this is partially correct but misses important nuance:

1. **Map 94 scenarios (2010141294, 2010131194)**: ~99% empty rate UNDER CURRENT SDK. The route planner (A*) genuinely cannot find paths with any move type on this map for these forces. This IS a route planner failure (not candidate generation), caused by map terrain geometry.

2. **Map 19 scenarios (1910631192)**: 0% empty rate under current authenticated SDK. The league's 99% was an auth-state artifact, NOT a route reliability issue.

3. **All other tested scenarios**: 0% empty rate — consistent with the league's own data for 38/43 scenarios.

## Revised SDK diagnostic finding

The Phase 3 SDK diagnostic revealed that the `no_valid_move_candidate` label is a misnomer:
- SDK Move valid_actions entry is `None` (binary availability flag), not a candidate list
- `missing_move_candidate = not candidate` is always `True` for Move
- The route planner IS always called before the failure is recorded
- The true root cause is: "A* route planner cannot find a path to any city with any move type, AND fallback neighbor search also fails"

This means the V3F classification `ROUTE_INSTRUMENTATION_CORRECT` should be revised to `ROUTE_INSTRUMENTATION_COUNTS_CORRECT_BUT_LABEL_MISLEADING`.

## Route status (provisional)

```
ROUTE_RELIABILITY_PARTIALLY_SUPPORTED_SCENARIO_DEPENDENT
```

### Rationale

- Authenticated full coverage evaluation is in progress (14/43 scenarios complete)
- Route-empty behavior is strongly scenario-dependent:
  - Map 94 scenarios: ~99% empty (legitimate route planning failures)
  - Map 19 scenarios: 0% empty (league's 99% was auth artifact)
  - Majority of scenarios: 0% empty
- No evidence of a route planner code bug (A* algorithm is correct)
- Persistent SDK candidate-generation limitations in specific map/scenario combinations

### Conditions

- [x] Authenticated replay enters battle
- [x] Denominator is meaningful
- [~] Route-empty rate is low under bounded evaluation (0% for 11/14 scenarios, but >99% for 2)
- [~] Archived high empty-rate is partially explained (map 19 was auth artifact, map 94 is genuine terrain issue)
- [ ] No major route failure modes remain in tested protocol (map 94 failures are real)

## Allowed wording

```
Authenticated route evaluation indicates that route-empty behavior is strongly scenario-dependent.
The map 94 scenarios (2010141294, 2010131194) show near-complete route planner failure (~99% empty)
under current authenticated SDK replay, consistent with or worse than the archived league results.
However, the map 19 scenarios show 0% empty (resolving the V3F contradiction as an auth artifact).
No route planner code bug was identified — the failures are caused by terrain/pathfinding constraints
on specific maps that prevent A* from finding paths to any city with any move type.
```

## Forbidden claims (unchanged from V3F)

- Route reliability is solved generally
- The route planner is robust across the SDK
- The archived 66.3% failure is irrelevant
- All scenarios are route-safe
