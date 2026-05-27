# V3G Start State

Date: 2026-05-26T20:02

## Current branch

- **Branch:** `main`
- **HEAD:** `c03b868` (sync: push updates from Windows 2026-05-25 00:00:09)
- **V3F commit:** `c03b868` (current HEAD — V3F was completed on this branch without an explicit separate commit)

## Worktree status

```
?? .claude/
?? goals/codex_goal_v3g_full_route_coverage_expansion.md
```

No unrelated edits. The only untracked files are `.claude/` (new session data) and the V3G goal file.

## V3F final decision

- **Classification:** `ROUTE_EVIDENCE_RECONCILED_NO_CODE_REPAIR_NEEDED`
- **Route planner is not buggy.** `Map.gen_move_route` (A\*) works correctly when called.
- **`no_valid_move_candidate`** is an SDK candidate generation issue, not a route planner failure.
- **Route instrumentation is correct** in both V3D and league — same code path, same diagnostics.
- **Scenario coverage is the critical variable.** V3D's 3 scenarios happen to have consistent move candidate availability.
- **Critical unresolved contradiction:** Scenario 1910631192: 0% empty in V3D/V3F but 99%+ empty in league.

## V3F route claim gate

- **Classification:** `ROUTE_NO_DEGRADATION_OBSERVED_ONLY`
- V3D authenticated: 0% empty (590 attempts, 3 scenarios, 30 games)
- League full eval: 66.3% empty (23,580 attempts, ~43 unique scenarios)
- Conditions for `ROUTE_RELIABILITY_SUPPORTED`: NOT met
- Conditions for `ROUTE_NO_DEGRADATION_OBSERVED_ONLY`: met

## Known 1910631192 contradiction

| Source | Empty rate | Move attempts | Context |
|--------|-----------|---------------|---------|
| V3D authenticated (5 seeds × 2 modes) | 0.0% | 316 | Tactical ablation, current SDK |
| League (multiple pairs, ~500 games) | ~99% | Not separated per-scenario in aggregate | Full league evaluation, v1-v2 era |

## SDK auth status

- SDK authentication was confirmed working in V3D/V3F (all games authenticated, battle phase entered)
- SDK `TrainEnv` is available and imports correctly
- Model weights present at `ai/weights/model_manifest.json`
- No recent auth regression detected

## V3F reports exist

- `reports/v3f_route_reliability/` — complete with 10 phases plus FINAL_DECISION
- Key reports: 02_route_discrepancy_analysis, 07_final_route_evaluation, 08_route_claim_gate
- Evaluator script: `scripts/run_v3f_route_reliability_evaluation.py`
- Tests: `tests/test_v3f_route_reliability.py`
- League sample: `experiments/shenji_agent/v3f_route_reliability_league_sample/` (10 games, 0% empty)

## Archived league sidecars exist

- `experiments/shenji_agent/league_20260501_170204/` — full 100-pair league
  - `map_list.json`: 100 items, 43 unique scenarios, duplicates used
  - `rating.json`: aggregate metrics including `move_route_aggregate` (23,580 attempts, 15,641 empty)
  - `summary.md`: league metadata, auth status, phase entry
  - Various other sidecar files
- No per-scenario route metrics could be extracted from league aggregate alone

## Backup branch

Created: `backup/before_v3g_full_route_coverage_20260526_200247`

## Phase plan

Following the goal file phases:
1. Full archived league scenario inventory
2. Scenario 1910631192 contradiction audit
3. SDK candidate generation diagnostic
4. Extend full-coverage route evaluator
5. Smoke evaluation
6. Full coverage evaluation
7. Archived vs current comparison
8. Route claim gate
9. V3 evidence package update
10. Tests and validation
11. Manuscript readiness update
12. Final decision and commit
