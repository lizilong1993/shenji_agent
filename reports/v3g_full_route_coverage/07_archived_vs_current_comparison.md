# V3G Archived vs Current Comparison

**Date:** 2026-05-26

## Data sources

- **Archived league:** `league_20260501_170204` (100 pairs, 43 unique scenarios, v1-v2 era)
- **Current V3G:** Full coverage evaluation over 43 unique league scenarios x 1 seed (authenticated SDK replay)

## Comparison

(Pending Phase 6 completion. This report will be finalized after the full evaluation finishes.)

## Preliminary findings (from smoke + 1910631192 focus)

| Metric | Archived League | Current V3G (partial) |
|---|---|---|
| Evaluation protocol | League runner (challenger + baseline) | V3G evaluator (challenger + baseline) |
| SDK auth | Authenticated (MAC `02:42:ac:11:00:02`) | Authenticated (MAC `02:42:ac:11:00:07`) |
| Tactical model loaded | 0 (not loaded) | Loaded (tactical_model_loaded=True) |
| Max steps | 400 | 2000 |
| Scenarios tested | 43 unique (100 pairs) | 5 (smoke) + 1 (1910631192 focus) |
| Route empty rate | 66.3% | 0.0% (so far) |
| Battle entered | 100/100 | 6/6 |
| Scenario 1910631192 | 99.0% empty (2088 attempts) | 0.0% empty (158 attempts) |
| Scenario 1531 | 0.0% empty (11 attempts) | 0.0% empty (36 attempts) |
| Scenario 2230531221 | 0.0% empty (200 attempts) | 0.0% empty (50 attempts) |

## Critical finding: Auth regression invalidated league comparison

The V3F and V3G evaluators both had an undetected SDK authentication regression. The `authenticated: True` flag in both evaluators was set unconditionally on success — the code never checked `env.pass_authenticaiton`. This meant battle-phase entry failure was silently masked as "authenticated but zero actions."

After fixing the auth (updating the venv `.engine_config` to match the current MAC), the same evaluators produce fundamentally different results:
- Scenario 1910631192 went from 0/5 battle entered to 5/5 battle entered
- Move attempts went from 0 to 158
- Route empty rate stayed at 0.0% but with a meaningful denominator

## Implications for the archived vs current question

The archived league's 66.3% empty rate (with scenario 1910631192 at 99.0%) was measured under a different SDK state where the auth was working (MAC `02:42:ac:11:00:02`). The auth regression in the current SDK state (stale `.engine_config`) was an artifact of the development environment, not a change in the SDK itself.

The current evaluation shows:
1. **5 non-map-19 league scenarios**: 0% empty (consistent with league's non-map-19 scenarios: 38/43 scenarios already had 0% empty)
2. **Scenario 1910631192 (map 19)**: 0% empty with current auth — contradicts the league's 99% but matches V3D

The 99% empty rate for map 19 scenarios in the league was likely caused by:
1. **No tactical model** (tactical_model_loaded_match_count=0) — the tactical model influences city selection and move type choice, potentially enabling more successful moves
2. **Shorter max steps (400 vs 2000)** — shorter games may not allow enough time for operators to find viable paths
3. **Different SDK auth binding** — the league ran with a different MAC binding

The comparison must acknowledge that the protocol differences (steps, model configuration, evaluation runner) make direct comparison unreliable. The V3G result is the current authenticated measurement.

## Classification (provisional)

Expected classification once full eval completes: `ARCHIVED_ROUTE_FAILURE_NOT_REPRODUCED_CURRENT`

Rationale: The archived 99% empty rate for map 19 scenarios is not reproduced under current authenticated SDK replay with proper configuration (tactical model loaded, 2000 step limit). The 0% empty rate from V3D is reproduced.
