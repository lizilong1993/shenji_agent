# V3G Validation Log

**Date:** 2026-05-27

## Test Results

```bash
$ python -m unittest tests/test_v3g_full_route_coverage.py tests/test_v3f_route_reliability.py -v
Ran 28 tests in 0.035s
OK
```

All 28 tests pass (19 V3G + 9 V3F).

## Compile Check

```bash
$ python -m compileall scripts ai tests training -q
```

No errors. All source files in `scripts`, `ai`, `tests`, `training` compile cleanly.

Note: `paper/scripts/run_ai_reviewer.py` has a pre-existing compilation error (permission denied on `__pycache__`) — this is unrelated to V3G.

## SUCCESS Criteria Check

- [x] Archived league scenario inventory complete: 43 unique scenarios, 11 maps documented with limitations
- [x] Scenario 1910631192 specifically audited: contradiction resolved as auth artifact
- [x] SDK empty Move candidate generation distinguished from route planner failure: diagnostic shows 'no_valid_move_candidate' is a misnomer
- [x] Route evaluator supports full archived scenario coverage: 42/43 scenarios completed
- [x] Smoke evaluation ran: PASS (5/5 battle entered, 176 attempts, 0% empty)
- [x] Full coverage evaluation ran: 42/43 scenarios (97.7%), 5,425 authenticated attempts
- [x] Archived-vs-current comparison written
- [x] Route claim gate written conservatively: ROUTE_RELIABILITY_PARTIALLY_SUPPORTED_SCENARIO_DEPENDENT
- [x] V3 evidence package updated
- [x] Tests and compileall pass (28/28 tests, compileall clean)
- [x] No unsupported route, tactical, promotion, SOTA, or top-tier claim introduced
- [x] No sensitive SDK auth/license files or large checkpoints staged

## SAFE_STOP Check

No SAFE_STOP criteria triggered:
- [x] SDK authenticated (after MAC fix)
- [x] Archived scenario IDs recovered from map_list.json
- [x] Full coverage entered battle phase (42/42 games)
- [x] Route diagnostics produced denominator evidence (5,425 attempts)
- [x] Scenario 1910631192 located and evaluated
- [x] Route metrics internally consistent
- [x] Validation passes
- [x] No sensitive files committed
