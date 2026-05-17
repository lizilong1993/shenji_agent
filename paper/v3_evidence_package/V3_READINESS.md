# V3 Readiness

**State:** `IN_PROGRESS`

## Claim classification updates

| Claim | V2 status | V3A status | V3A-FIX status | V3B status | V3C status | V3D status | Change |
|-------|-----------|------------|----------------|------------|------------|------------|--------|
| tactical contribution | forbidden | partially supported (action-level only) | partially supported (action-level only, outcome executable) | inconclusive (auth-blocked) | auth recovered | weak/inconclusive (outcome) | upgraded from V3B (auth no longer blocks) |
| route reliability | forbidden | forbidden (unchanged) | forbidden (unchanged) | inconclusive (no move routes) | — | no degradation observed (3 scenarios) | V3D shows 0% empty over 590 attempts (3 scenarios) but league shows 66.3% empty — not broadly proven |
| two-window success | forbidden | forbidden (unchanged) | forbidden (unchanged) | forbidden (unchanged) | — | — | — |
| baseline superiority | forbidden | forbidden (unchanged) | forbidden (unchanged) | forbidden (unchanged) | — | — | — |
| method novelty | unsupported | unsupported (unchanged) | unsupported (unchanged) | unsupported (unchanged) | — | — | — |
| reproducibility | partially repaired | same (unchanged) | same (unchanged) | same (unchanged) | — | — | — |
| public SOTA | forbidden | forbidden (unchanged) | forbidden (unchanged) | forbidden (unchanged) | — | — | — |
| top-tier readiness | forbidden | forbidden (unchanged) | forbidden (unchanged) | forbidden (unchanged) | — | — | — |

## Completed this round

### V3F (route reliability repair and evaluation)
- ✅ V3D vs league discrepancy analyzed: 0% empty (3 scenarios) vs 66.3% (100 pairs)
- ✅ Root cause identified: `no_valid_move_candidate` is SDK candidate generation limitation, NOT route planner bug
- ✅ Route instrumentation audited and confirmed correct
- ✅ Route reliability evaluator created (`scripts/run_v3f_route_reliability_evaluation.py`)
- ✅ Bounded evaluation with league scenarios executed (5 pairs, 0% empty in sampled subset)
- ✅ No code repair applied — no route planner bug found
- ✅ Claim gate: ROUTE_NO_DEGRADATION_OBSERVED_ONLY (unchanged from V3E)
- ✅ Tests: 9/9 pass; compileall clean

## Remaining blockers

1. ~~**SDK authentication**~~ — **RESOLVED**. MAC binding recovered.
2. **Route reliability unproven** — V3D shows 0% empty rate over 590 attempts (3 scenarios). V3F confirms this is valid but scenario-limited. League evaluation (23,580 attempts, ~100 scenarios) shows 66.3% empty rate from SDK candidate generation constraints, not route planner bugs. Route reliability not broadly proven — claim gate remains `ROUTE_NO_DEGRADATION_OBSERVED_ONLY`. **V3F recommended next:** V3G route coverage expansion (evaluate full league scenario set).
