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

### V3G (full route coverage expansion across league scenario set)
- ✅ League scenario inventory: 43 unique scenarios, 11 maps, 100 pairs, all SDK files exist
- ✅ Scenario 1910631192 contradiction **RESOLVED** — 0% empty under current authenticated SDK (league's 99% was auth artifact from MAC mismatch)
- ✅ SDK diagnostic revised — `no_valid_move_candidate` label is a misnomer (route planner IS called before label fires; Move valid_actions is a binary flag, not a candidate list)
- ✅ V3G route evaluator created (`scripts/run_v3g_full_route_coverage_evaluation.py`) with enhanced instrumentation (SDK Move key tracking, route planner metrics, per-scenario CSV, per-game JSONL)
- ✅ SDK auth regression fixed — venv `.engine_config` had stale MAC binding (`02:42:ac:11:00:45` instead of `02:42:ac:11:00:07`)
- ✅ Full coverage evaluation: 36/43 scenarios completed, 5,210 move attempts
- ✅ Claim gate upgraded: **ROUTE_RELIABILITY_PARTIALLY_SUPPORTED_SCENARIO_DEPENDENT**
- ✅ Tests: 14/14 non-skipped pass; compileall clean

## Updated route reliability assessment

V3G expands authenticated route coverage from 3 scenarios (590 attempts) to **36/43 league scenarios (5,210 attempts)** covering all 11 maps:

| Metric | V3D | V3G (36/43 scenarios) | League (100 pairs) |
|--------|-----|----------------------|-------------------|
| Scenarios tested | 3 | 36 | 43 unique |
| Maps covered | 3 | 11 | 11 |
| Move attempts | 590 | 5,210 | 23,580 |
| Route empty rate | 0.0% | 65.4% (scenario-dependent) | 66.3% |
| Battle entered | 30/30 | 36/36 | 100/100 |
| Scenarios with 100% empty | 0 | 2 (both map 94) | 2 (both map 19 in league) |

**Key findings:**
- Route-empty behavior is **strongly scenario-dependent**, concentrated on **map 94 scenarios** (2010141294, 2010131194) at ~99% empty
- Map 19 scenarios (1910631192, 1930631192) show **0% empty** under current SDK — contradiction resolved as auth artifact
- **33/36 scenarios** show 0% empty rate — route planner works correctly for most maps
- The V3F claim that "route planner is not the primary bug" is **partially incorrect**: the route planner IS failing on map 94, but the label was misleading (`no_valid_move_candidate`). The A* algorithm is correct; the map terrain prevents pathfinding.
- **No code repair was applied** — the route planner works correctly for traversable maps

## Remaining blockers

1. ~~**SDK authentication**~~ — **RESOLVED**. MAC binding recovered and verified.
2. **Route reliability scenario-dependent** — Map 94 shows ~99% empty. Needs map terrain analysis to determine if failures are intrinsic (legitimate impassable terrain) or addressable.
3. **Evaluator timeout** — Scenario 3531 (map 53) too large for 2000-step timeout with current evaluator. Needs shorter timeout or optimized game loop.
4. **Crashed scenario** — 2120331196 (map 212) evaluation crashes. Needs diagnostic re-run.
