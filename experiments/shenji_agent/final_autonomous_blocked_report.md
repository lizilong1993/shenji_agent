# Final Autonomous Blocked Report — Iteration 002

**Status: B — blocked with hard evidence**
**Date: 2026-04-28**

## 1. Final Status

B (blocked-with-evidence). Code fixes are verified correct, learned policy loads and engages, but model quality is insufficient to pass promotion gates.

## 2. Git HEAD

- Original repo: `2a5ac08` — Enable GPU policy inference for learned policy
- Fresh clone (on gpfsdata): `a33cdee` — fix: restore learned move and shoot engagement in selfplay

The original repo cannot accept new commits due to root filesystem (overlay) being 100% full. A fresh clone was created at `/remote-home/lizilong/shenji_agent_fresh` on gpfsdata (144TB free) where the commit succeeded.

## 3. New Commits

- `a33cdee` in fresh clone: "fix: restore learned move and shoot engagement in selfplay"
  - 13 files changed, 3381 insertions(+), 180 deletions(-)

## 4. Reverts

None.

## 5. Modified Files

- `ai/agent.py` — Priority reorder: Shoot(0), Move(1), Occupy(2)
- `ai/models.py` — Model loading improvements
- `ai/policy.py` — Policy inference improvements
- `training/selfplay.py` — Incumbent profile resolution, map normalization, progress snapshots, heartbeats
- `training/evaluate.py` — Coverage gate enforcement in promotion_decision
- `training/common.py`, `training/maps.py` — Map resolution improvements
- `tests/test_mainline_contract.py` — 35 contract tests
- `tests/test_learned_policy_engagement_contract.py` — NEW: 3 tests
- `tests/test_selfplay_participant_wiring.py` — NEW: 9 tests
- `tests/test_promotion_artifact_contract.py` — NEW: 10 tests
- `configs/selfplay_baseline.yaml`, `mock_train_env.py`

## 6. Test Results

All 57 tests pass:
```
Ran 57 tests in 0.569s
OK
```

## 7. Smoke Self-Play

**Config:** 2 paired matches, 50 max steps
**Command:** `python3 -m training.selfplay --config configs/selfplay_smoke_learned_engagement.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule`

**Results:**
- learned_policy_enabled: 2/2 matches ✅
- learned_path_coverage: 1.0 ✅
- move_path_coverage: 1.0 ✅
- shoot_path_coverage: 0.0 (0 total shoot actions)
- fallback_reason: [] (clean) ✅

## 8. Formal Self-Play

**Config:** 30 paired matches, 200 max steps
**Command:** `python3 -m training.selfplay --config configs/selfplay_formal_learned.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule`

**Results — PROMOTION NOT ACCEPTED:**

| Gate | Required | Actual | Status |
|------|----------|--------|--------|
| overall_point_estimate | >= 0.60 | 0.50 | FAIL |
| overall_ci_lower | >= 0.55 | 0.50 | FAIL |
| complex_point_estimate | >= 0.60 | 0.50 | FAIL |
| complex_ci_lower | >= 0.55 | 0.50 | FAIL |
| learned_path_coverage | >= 0.30 | 0.155 | FAIL |
| move_path_coverage | >= 0.20 | 0.147 | FAIL |
| shoot_path_coverage | >= 0.20 | 1.0 | PASS |
| elo_delta_lower | >= 35.0 | 0.0 | FAIL |

## 9. Participants Artifact

`/remote-home/lizilong/shenji_agent_fresh/experiments/shenji_agent/league_20260428_171537/participants.json`

- candidate: `ai/weights`, profile="challenger", manifest_loaded=true
- incumbent: `ai/weights_incumbent_rule`, profile="baseline", snapshot_role="bootstrap_rule_baseline"
- candidate ≠ incumbent ✅

## 10. Promotion Artifact

`/remote-home/lizilong/shenji_agent_fresh/experiments/shenji_agent/league_20260428_171537/promotion.json`

```
accepted: false
failures: [overall_point_estimate, overall_ci_lower, complex_point_estimate, complex_ci_lower, learned_path_coverage, move_path_coverage, elo_delta_lower]
```

## 11. Learned Policy Engagement Evidence

**Positive:** The policy loads correctly:
- All 30 matches: manifest_loaded=true, intent_model_loaded, situation_model_loaded
- 0 matches with manifest_missing or policy_models_unavailable fallback
- Shoot actions: 50/50 (100%) use learned model
- Move actions: 698/4764 (14.7%) use learned model

**Negative:** Most move attempts fall back with "empty_route":
- 4764 total move actions, only 698 reach a city
- 85.3% of moves have no valid route to any city
- This is the primary driver of low learned_path_coverage

## 12. Coverage Data

- learned_path_coverage: 0.155
- move_path_coverage: 0.147
- shoot_path_coverage: 1.0
- Fallback reason: "empty_route" (no reachable city for 85% of move actions)

## 13. Promotion Gate Result

**FAILED.** 7 of 8 gates failed. Only shoot_path_coverage passed.

## 14. Root Cause of Failures

### Immediate Cause: All games end 0-0
Every match in the formal self-play ended with both sides scoring 0.0. This means:
- Neither the learned policy nor the baseline produced any scoring actions
- Games run for 200 steps with no objectives achieved
- The learned policy makes moves but they don't lead to score

### Contributing Factor 1: Low move engagement (15%)
85% of move actions fail with "empty_route" — the map position has no reachable city. This dramatically reduces the learned policy's influence on the game.

### Contributing Factor 2: Model quality
- intent_bias for move: -0.636 (negative — the model biases against cities)
- Despite loading correctly, the models don't produce winning strategies
- The learned policy doesn't outperform the baseline in any bucket

### Contributing Factor 3: Root FS full
The root filesystem (overlay) is 100% full, preventing git commits in the original repo. This forced working from a fresh clone on gpfsdata.

## 15. Next Steps (Minimum Executable Plan)

1. **Free root filesystem space** — Required before any further git operations in the original repo
2. **Investigate "empty_route" rate** — 85% of move actions fail to find a reachable city. This might be a map/pathfinding configuration issue or a game mechanics limitation
3. **Retrain models** — The models load but don't produce winning strategies. Need to investigate training data quality, model architecture, or hyperparameters
4. **Improve action diversity** — The learned policy only handles Move and Shoot. Consider adding learned Occupy decision-making to increase impact
5. **Verify game scoring** — Confirm that 0-0 draws are the expected outcome when neither side captures objectives, vs a scoring bug

## 16. Git Status (Original Repo)

Uncommitted modifications in 31 files. Cannot commit due to root FS ENOSPC.

## 17. Uncommitted Files

All changes were applied to the fresh clone (`/remote-home/lizilong/shenji_agent_fresh`) and committed as `a33cdee`. The original repo at `/remote-home/lizilong/shenji_agent` retains uncommitted changes that mirror the fresh clone commit, minus the 3 new test files and formal config.

Key uncommitted: ai/agent.py, training/selfplay.py, training/evaluate.py, training/common.py, training/maps.py, ai/models.py, ai/policy.py, tests/test_mainline_contract.py, configs/selfplay_baseline.yaml, mock_train_env.py

## 18. Verification Summary

| Fix | Status | Evidence |
|-----|--------|----------|
| Priority Shoot>Move>Occupy | ✅ Working | smoke: move_path_coverage 0→1.0; formal: learned_move 698/4764 |
| Incumbent profile baseline | ✅ Working | snapshot_role→baseline verified in participants.json |
| Model manifest traceable | ✅ Working | source_commit_sha matches HEAD |
| Logger critical | ✅ Working | Code confirmed |
| Contract tests | ✅ Working | 57/57 pass |
| Learned policy loading | ✅ Working | 30/30 matches loaded |
| Promotion wiring | ✅ Working | participants differ, profiles correct |
| Coverage diagnostics | ✅ Working | All fields populated in rating/promotion |
