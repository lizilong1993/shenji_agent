# Iteration 005: City Scoring Rebalance and Model Quality Diagnosis

## Date
2026-04-28

## Status
B blocked-with-hard-evidence (model quality insufficient)

## Objective
Break 0-0 stalemate in formal self-play by analyzing and fixing score/policy root causes. Previous iterations (003-004) fixed route plumbing (empty_route → 0%).

## Current HEAD
`677f69d` (Linux fresh clone: `/remote-home/lizilong/shenji_agent_fresh`)

## Previous State (from Iteration 004)
- empty_route rate: 85% → 0% (fixed)
- move_path_coverage: 14.7% → 100% (fixed)
- learned_path_coverage: 15.5% → 100% (fixed)
- All 5 formal matches: 0-0 stalemate
- 45/45 tests pass

## Root Cause Analysis

### 1. Situation Model Dead Output
**Evidence**: pre-sigmoid values ≈ -28.69, sigmoid(-28.69) ≈ 3.5e-13
**Root cause**: Feature scaling mismatch between training and inference. The `operator_to_feature` function produces very sparse features with small values (obj_id/10000, blood/100), while the model expects larger values.
**Impact**: `situation_score * 100` contributes ~0 to city/shoot scoring, providing no differentiation.

### 2. Distance-Dominated City Scoring
**Evidence**: `distance * 0.35` (max ~1750) dwarfs `city_value * 0.2` (max 20)
**Root cause**: Scoring weights were never tuned for balanced decision-making.
**Impact**: Agent always picks the nearest city regardless of tactical value, ownership, or objectives.

### 3. No City Ownership Awareness
**Evidence**: City `flag` field ignored in scoring.
**Impact**: Agent doesn't prefer capturing neutral/enemy cities over staying on friendly cities.

### 4. Both Sides Use Same Strategy
**Evidence**: Candidate (learned) and incumbent (rule) use the same multi-mode fallback route finding.
**Impact**: Both sides converge to same cities, leading to mutual attrition without scoring differentiation.

### 5. Global Mean Pooling in Situation Model
**Evidence**: `GraphSituationModel.forward()` does `node_scores.mean(dim=0)`.
**Impact**: All cities get the same situation score, eliminating any positional differentiation.

## Changes Made

### ai/policy.py
- **City scoring rebalance**: distance penalty 0.35→0.08, city value weight 0.2→0.5
- **Owner bonus**: +25 for neutral/enemy cities, -15 for friendly cities
- **Occupy proximity bonus**: +20 when operator is at or adjacent to city
- **Shoot scoring**: kill bonus 50→75, added target value bonus (value * 0.3)
- **Situation model fix**: Added feature standardization (z-score normalization across operator batch)
- **Per-node scoring**: Use specific operator node score instead of global mean for city evaluation
- **Situation contribution**: Only include situation_score in combined_score when > 1e-6

### ai/weights/model_manifest.json
- Fixed `source_commit_sha` from incorrect `10608cd56263a...` to current HEAD `9ba20f3b2...`

### tests/test_policy_score_tuning.py (new, 12 tests)
- City scoring: higher value preferred, enemy/neutral preferred over friendly, proximity bonus
- Shoot scoring: high-value kill preferred, kill beats non-kill
- Situation model health: produces non-zero output after normalization
- Diagnostics: score means included in policy diagnostics

### tests/conftest.py (new)
- Imports real torch/numpy at collection time to prevent fake module installers from breaking real imports

### tests/test_mainline_contract.py
- Fixed `test_selfplay_can_load_agent_class_from_git_commit` to restore real torch modules via `addCleanup`

## Test Results
```
79 passed in 1.31s
```

## Smoke Self-Play (2 matches, max_steps=50)
- League: `league_20260428_203705`
- All 2 matches: 0-0 stalemate
- learned_path_coverage: 1.0
- move_path_coverage: 1.0
- shoot_path_coverage: 0.0 (no combat in short games)
- situation_score mean (move): 0.00018 (improved from 4e-13, but still minimal)
- intent_bias (move): 0.0 (improved from -0.371)

## Medium Self-Play (5 matches, max_steps=400)
- League: `league_20260428_204108`
- All 5 matches: 0-0 stalemate
- learned_path_coverage: 1.0
- move_path_coverage: 1.0
- shoot_path_coverage: 1.0
- situation_score mean (move): 0.00054 (still near-zero, model not differentiating)
- intent_bias (move): -0.364 (negative bias persists)
- combined_score (move): 10.09
- Promotion: FAILED
  - overall_point_estimate: 0.5 (required ≥ 0.60)
  - overall_ci95_lower: 0.5 (required ≥ 0.55)
  - complex_point_estimate: 0.5 (required ≥ 0.60)
  - complex_ci95_lower: 0.5 (required ≥ 0.55)
  - elo_delta_lower: -0.0 (required ≥ 35)

## Hard Evidence for Blocked State

1. **Route plumbing fixed**: All move/shoot/learned coverages at 1.0, zero fallback reasons
2. **Situation model untrained/undertrained**: Outputs near-constant ~0.0005 for all inputs even after feature normalization
3. **Per-node scores identical**: All operators get ~0.484 sigmoid output, zero positional differentiation
4. **Intent model has negative move bias** (-0.364): Discourages movement toward objectives
5. **Tactical model disabled**: `tactical_model_loaded_match_count: 0` in all runs
6. **Heuristic scoring insufficient**: Without meaningful learned model signal, both sides converge to same behavior
7. **Both sides use same route fallback logic**: The multi-mode fallback fix benefits both candidate AND incumbent equally

## Blocked Conclusion
The code plumbing (routes, actions, scoring weights, diagnostics) is complete and verified. The remaining 0-0 stalemate is a **model quality issue**:

- Models need retraining with proper feature normalization
- Training data needs scoring-relevant labels (occupy opportunities, city capture events)
- Tactical model should be enabled
- Intent model needs positive move/occupy bias

## Minimal Next Step
1. Retrain intent/situation models with standardized input features
2. Enable tactical model in training and inference
3. Add occupy/scoring labels to training data
4. Re-run supervised training → smoke → formal self-play

## Linux Commit
- `677f69d` - fix: rebalance city scoring, fix situation model feature normalization, add policy score tests

## Windows Sync
- **BLOCKED**: tailscale SSH to 100.127.108.19 timed out (connection timeout)
- Sync files ready: ai/policy.py, ai/weights/model_manifest.json, tests/conftest.py, tests/test_mainline_contract.py, tests/test_policy_score_tuning.py
- Will sync when host becomes reachable

## Windows Commit
- **BLOCKED**: pending Windows host connectivity
