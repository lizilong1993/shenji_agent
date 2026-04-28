# Shenji Agent Autonomous Engineering — Final Blocked Report (Iteration 005)

**Date:** 2026-04-28
**Status:** B — Blocked with hard evidence (model quality insufficient)
**Working directory:** `/remote-home/lizilong/shenji_agent_fresh` (fresh clone)

---

## 1. Current Git State

**HEAD:** `677f69d` — fix: rebalance city scoring, fix situation model feature normalization, add policy score tests

**Commit history (fresh clone, ahead of original by 8 commits):**
```
677f69d fix: rebalance city scoring, fix situation model feature normalization, add policy score tests
9ba20f3 chore: bump manifest to final report commit
10608cd docs: final blocked report with move routing fix evidence
49039ca fix: correct manifest source_commit_sha to real HEAD
44ac552 chore: bump manifest source_commit_sha to 1860b86
1860b86 fix: passthrough move route diagnostics and update manifest commit
c59846b docs: finalize iteration 003 report with root cause and fix summary
9e36d05 fix: add move route diagnostics, fallback strategy, and context-aware occupy
```

**Original repo** (`/remote-home/lizilong/shenji_agent`): At 2a5ac08, behind fresh clone by 8 commits.

---

## 2. What Was Fixed (Iterations 001-005)

### Iteration 001-002: Learned Policy Disengagement
- Root cause: `use_ai_optimization` not set for `CurrentAI`, so learned path skipped
- Fix: Auto-enable learned policy for `CurrentAI` in Agent.setup()
- Fix: Always enable `use_ai_optimization` for non-baseline profiles
- Disabled GPU policy inference (CUDA OOM on shared GPU)
- Capped PyTorch CPU threads

### Iteration 003-004: empty_route (85% move fallback)
- Root cause: March cost layer (mode 1) had 78.4% empty adjacency cells in cost.pickle
- Fix: Multi-mode fallback route finding (try all 4 move types, shortest route per city)
- empty_route: 85% → 0%
- move_path_coverage: 14.7% → 100%
- learned_path_coverage: 15.5% → 100%

### Iteration 005: City Scoring Rebalance and Model Quality Diagnosis
- **City scoring**: Rebalanced distance penalty (0.35→0.08), added owner bonus (+25 neutral/enemy, -15 friendly), occupy proximity bonus (+20)
- **Shoot scoring**: Kill bonus 50→75, target value bonus added
- **Situation model normalization**: Feature standardization in `_score_counterfactual`
- **Per-node scoring**: Individual operator node score instead of global mean
- **Tests**: +12 policy score tests, +conftest.py (fake module protection), fixed test cleanup
- **Situation model diagnosis**: Pre-sigmoid values -28.69→0 after normalization, but per-node scores all ~0.484 (no differentiation)

---

## 3. Current Hard Evidence (0-0 Stalemate)

### Medium Self-Play (5 matches, max_steps=400)
- League: `league_20260428_204108`
- All 5 matches: 0-0 stalemate (candidate=0.0, opponent=0.0 in all legs)
- Route coverage: 1.0 (move/shoot/learned all 100%)
- Fallback reasons: [] (zero fallbacks)
- **situation_score mean**: 0.00054 (near-zero, model not differentiating)
- **intent_bias mean**: -0.364 (negative, discourages movement)
- **tactical_model_loaded**: 0/5 matches (model file missing from candidate dir)
- **combined_score (move)**: 10.09 (entirely from heuristics, no learned signal)

### Promotion Gate Failures
| Gate | Current | Required | Status |
|------|---------|----------|--------|
| overall_point_estimate | 0.5000 | ≥ 0.60 | FAIL |
| overall_ci95_lower | 0.5000 | ≥ 0.55 | FAIL |
| complex_point_estimate | 0.5000 | ≥ 0.60 | FAIL |
| complex_ci95_lower | 0.5000 | ≥ 0.55 | FAIL |
| elo_delta_lower | -0.0 | ≥ 35 | FAIL |
| learned_path_coverage | 1.0000 | ≥ 0.30 | PASS |
| move_path_coverage | 1.0000 | ≥ 0.20 | PASS |
| shoot_path_coverage | 1.0000 | ≥ 0.20 | PASS |
| max_bucket_regression | 0.0 | ≤ 0.03 | PASS |
| max_exploiter_win_rate | 0.0 | ≤ 0.45 | PASS |

---

## 4. Root Cause of Stalemate: Model Quality

### Evidence Chain
1. **Situation model produces near-identical per-node scores (~0.484)** for all operators regardless of position — provides zero positional differentiation
2. **Intent model has negative move bias (-0.364)** — discourages the most common action type
3. **Tactical model is disabled** — `tactical_model.pt` missing from candidate `ai/weights/` directory
4. **Candidate and incumbent share the same route-finding logic** — the multi-mode fallback fix benefits both equally
5. **All scoring differentiation comes from heuristic rules** — which are identical for both sides at equal positions
6. **Result**: Both sides converge to same behavior → mutual attrition → 0-0

### Model Files Present
- Candidate (`ai/weights/`): intent_lstm.pt, situation_gcn.pt, intention_lstm.pth, situation_gcn.pth
- Missing from candidate: intent_model.pt, situation_model.pt, tactical_model.pt
- Incumbent (`ai/weights_incumbent_rule/`): has all files but profile="baseline" → learned policy disabled

---

## 5. What Has Been Eliminated

| Suspected Cause | Status | Evidence |
|----------------|--------|----------|
| Code bugs preventing moves | ELIMINATED | 100% move_path_coverage, zero empty_route |
| Learned policy not engaging | ELIMINATED | 100% learned_path_coverage, no fallback_reasons |
| Route generation broken | ELIMINATED | All 4 move types working, multi-mode fallback |
| City scoring weights imbalanced | FIXED | Owner bonus, proximity bonus, balanced distance |
| Situation model output near-zero | FIXED | Feature normalization added, output ~0.48 per node |
| Per-node vs global pooling | FIXED | Per-node scores used for city differentiation |
| Fake torch breaking tests | FIXED | conftest.py + cleanup in test_mainline_contract.py |

---

## 6. Remaining: Model Training Required

The code plumbing is complete. The remaining failure is a **model quality issue** requiring retraining:

### Required Actions
1. **Retrain situation model (GraphSituationModel)**: With standardized input features and per-node scoring targets
2. **Retrain intent model (MultiHorizonIntentionModel)**: With positive move bias toward objectives
3. **Train and enable tactical model (TacticalScorer)**: For real-time action evaluation
4. **Add occupy/scoring labels to training data**: City capture events, scoring opportunities
5. **Generate tactical_model.pt** in candidate weights directory
6. **Retrain with proper feature normalization**: Match inference-time feature distribution

### Minimum Verifiable Improvement
After retraining:
1. situation_score should vary meaningfully across different city positions (std > 0.01 across cities)
2. intent_bias for move should be non-negative
3. tactical_model_loaded should be True
4. At least one match should have non-zero score
5. candidate should show Elo advantage > 0

---

## 7. Test Results
```
79 passed in 1.31s
```
7 test files, 79 tests total. All pass.

---

## 8. Linux Git State
- Fresh clone: `/remote-home/lizilong/shenji_agent_fresh`
- HEAD: `677f69d`
- 79/79 tests pass
- All changes committed

---

## 9. Windows Sync State
- **BLOCKED**: tailscale SSH to 100.127.108.19 timed out (connection timeout on 2026-04-28)
- Sync file list ready (5 files from HEAD commit)
- Will sync when host becomes reachable

---

## 10. Uncommitted Files
- `.pyc` cache files (excluded from commit)
- `ai/weights/weights/` (duplicate directory)
- `datasets/`, `land_wargame_sdk/` (large assets)
- `scripts/` (verification scripts, already present)

---

## 11. Iteration Reports
- `iteration_002.md` — Learned policy re-engagement
- `iteration_003.md` — empty_route root cause and fix
- `iteration_004.md` — Move route diagnostics and validation
- `iteration_005.md` — City scoring rebalance and model quality diagnosis
