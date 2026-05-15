# V3D Preflight Smoke Report

## Parameters

| Field | Value |
|---|---|
| Scenario | 1231 (map 123) |
| Seed | 0 |
| Tactical on/off | Paired |
| Timeout steps | 2000 |
| Policy profile | challenger |

## Results

| Metric | Tactical ON | Tactical OFF | Delta |
|---|---|---|---|
| Status | completed | completed | — |
| Steps | 1801 | 1801 | 0 |
| Score (red) | 20 | 30 | -10 |
| Winner | blue (1) | blue (1) | — |
| Tactical invocations | 6 | 14 | -8 |
| Tactical nonzero count | 6 | 0 | +6 |
| Route empty rate | 0.0 | 0.0 | 0.0 |
| Duration | 33.37s | 9.48s | +23.88s |

## Required checks

| Check | Result |
|---|---|
| Authentication passes | ✅ (x2) |
| Deployment phase entered | ✅ |
| Battle phase entered | ✅ |
| Game completes | ✅ (both completed) |
| Tactical-on invokes tactical branch | ✅ (6 nonzero tactical scores) |
| Tactical-off does not invoke | ✅ (0 nonzero tactical scores) |
| Route diagnostics recorded | ✅ (empty_rate=0.0 both) |
| Outcome metric emitted | ✅ (score_red: 20 vs 30) |

## Conclusion

Preflight smoke passes. Authenticated replay pipeline is functional.
Tactical branch is correctly toggled: nonzero scores when ON, zero when OFF.
Route metrics are clean (0% empty rate).
Proceeding to formal multi-seed ablation protocol.
