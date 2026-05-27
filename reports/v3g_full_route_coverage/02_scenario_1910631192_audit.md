# Scenario 1910631192 (Map 19) Focused Route Reliability Audit

## 1. Current Authenticated Replay Results (V3F Evaluator, 5 Seeds)

### Aggregate

| Metric | Value |
|--------|-------|
| Games attempted | 5 |
| Games completed | 5 |
| Battle entered | **0** (0%) |
| Deployment entered | 5 (100%) |
| Crashed | 0 |
| Authenticated | 5 (100%) |
| Route-eligible move actions | **0** |
| Route empty count | **0** |
| Route empty rate | **0.0% (ZERO DENOMINATOR)** |
| No-valid-move-candidate count | 0 |
| Fallback move count | 0 |
| Zero-denominator games | **5/5** |
| Winner | Blue (1) for all 5 games |
| Steps | 1801 for all 5 games (max) |
| Duration | 12.4-29.8 seconds |

### Per-Game Breakdown

| Seed | Status | Battle Entered | Route Actions | Route Empty | Empty Rate | Winner | Steps | Duration (s) |
|------|--------|---------------|--------------|-------------|------------|--------|-------|-------------|
| 0 | completed | false | 0 | 0 | 0.0% | 1 (blue) | 1801 | 29.8 |
| 1 | completed | false | 0 | 0 | 0.0% | 1 (blue) | 1801 | 13.1 |
| 2 | completed | false | 0 | 0 | 0.0% | 1 (blue) | 1801 | 12.4 |
| 3 | completed | false | 0 | 0 | 0.0% | 1 (blue) | 1801 | 13.4 |
| 4 | completed | false | 0 | 0 | 0.0% | 1 (blue) | 1801 | 12.9 |

### Candidate Count Distribution

All 5 games: min=0, max=0, mean=0.0, median=0 (no route attempts occurred).

### Move Type Breakdown

None. No move actions were generated across all 5 games.

### Route Failure Reasons

None. No route planning was attempted.

### Policy Diagnostics

- intent_model_loaded: true (all 5 games)
- situation_model_loaded: true (all 5 games)
- tactical_model_loaded: true (all 5 games)
- learned_path_coverage: 0.0 (all 5 games)
- move_path_coverage: 0.0 (all 5 games)

### Key Observation

**The scenario never transitions from deployment to battle phase.** All 5 games complete all 1801 turns in deployment phase. The agents produce zero actions (no moves, no shoots) across the entire game. The V3F evaluator's battle detection (`stage != 1` check on the `time.stage` observation field) never triggers.

---

## 2. Comparison with V3D Data

V3D per-game data (from `scripts/run_v3b_formal_tactical_ablation.py`, commit `82e4ba0`, run on 2026-05-15):

| Metric | V3D (tactical_on) | V3D (tactical_off) | Fresh V3F |
|--------|-------------------|-------------------|-----------|
| Games per seed | 5 | 5 | 5 |
| Route attempts per game | 31-32 | 31-32 | **0** |
| Total route attempts | 158 | 158 | **0** |
| Route empty count | 0 | 0 | 0 |
| Route empty rate | 0.0% | 0.0% | 0.0% (zero denominator) |
| Winner | **0 (red)** | **0 (red)** | **1 (blue)** |
| Steps | 1801 | 1801 | 1801 |
| Battle entered | Yes (implicitly) | Yes (implicitly) | **No** |
| Duration (mean) | 22.89s | 22.66s | 16.3s |

### Critical Discrepancies

1. **Battle entry**: V3D entered battle (31-32 route-eligible moves per game), fresh run did not (0 moves). This is the root cause of all other differences.

2. **Winner**: V3D winner = 0 (red), fresh run winner = 1 (blue). With no actions from either side, the blue agent inexplicably wins by default.

3. **Duration**: V3D games took ~23s each; fresh games took 12-29s. The longer V3D duration is consistent with actual route computation (32 route computations per game).

### Setup Similarity

Both evaluators use nearly identical setup:
- Same `Agent(seed=seed)` for red, `Agent(seed=seed+1000)` for blue
- Same `TrainEnv` with same `env_step_info` structure
- Same `policy_profile="challenger"` for red, `"baseline"` for blue
- Same `use_ai_optimization=True/False`
- Same map loading (map_19)
- Same player info (seats 1 and 11, role=1)

The only relevant difference is that V3D explicitly sets `agent_red.policy.tactical_mode = tactical_mode` after setup, while V3F uses whatever default the policy initializes. However, V3D's tactical_off mode also produced 31-32 moves, so tactical_mode is not the cause.

---

## 3. Comparison with Archived League Data

Archived league data (100 matches, 1910631192 appeared 4 times):

| Metric | League Value |
|--------|-------------|
| Match count | 4 |
| Total move attempts | 2,088 |
| Empty route count | 2,068 |
| **Empty rate** | **99.04%** |
| By move type: Maneuver | 2,020 attempts, 2,000 empty (99.01%) |
| By move type: Walk | 68 attempts, 68 empty (100%) |
| Reachable city total | 40 |
| Learned move count | 20 (0.96%) |
| Total move count | 2,088 |

League aggregate (all 43 scenarios, 100 matches):
- 23,580 total attempts
- 15,641 empty routes
- **Overall empty rate: 66.3%**
- 5 scenarios with non-zero empty routes: 1910631192, 1930631192, 2010141294, 2010131194, 2120331196

### Key Difference vs V3D/V3F

The league runs with **max_steps=400** (vs 2000 in V3D/V3F) and has **tactical_model_loaded_count=0** (no tactical model deployed). The V3D/V3F evaluation uses the full AI pipeline including tactical model. This explains the V3D vs league difference (0% vs 99% empty) but does NOT explain the fresh run's failure to enter battle.

---

## 4. Was the 99% Archived Behavior Reproduced?

**No.** The fresh V3F evaluation does not reproduce the archived league's 99% empty route behavior. But it also does not reproduce V3D's 0% empty route behavior (with 316 attempts). **The fresh run produces a fundamentally different outcome: the scenario never enters battle phase, generating zero route-eligible actions.**

This is worse than either prior result for diagnostic purposes -- with zero denominator, no route reliability measurement is possible at all.

---

## 5. Exact Reason for Non-Reproduction

The root cause is that **scenario 1910631192 fails to transition from deployment phase to battle phase** under the V3F evaluator's game loop. All 1801 steps execute in deployment phase with no move actions generated by either agent.

Three possibilities:

1. **Unseeded SDK non-determinism**: The `TrainEnv.setup()` or first `TrainEnv.step()` may use random sources not controlled by the seed parameter, producing different initial deployment conditions between V3D and V3F runs.

2. **Agent behavior change at deployment boundary**: The agent policy might require a specific observation pattern to generate its first action, and if the deployment phase "ends" without any agent action, the game might stay in deployment for all remaining steps.

3. **Environmental sensitivity**: GPU state, CUDA non-determinism, or Python runtime state differences between runs could cause the same model/seed to produce different policy outputs.

The fact that V3D and V3F evaluators use nearly identical code (same epochs, same model weights, same seeds) but produce completely different game outcomes (32 vs 0 moves, red vs blue winner) strongly suggests the SDK or environment has non-deterministic behavior that is not fully controlled by the seed parameter.

---

## 6. Per-Game Candidate Count Distribution

| Metric | Fresh V3F | V3D (both modes) | League |
|--------|-----------|------------------|--------|
| Min | 0 | 31 | 522/map |
| Max | 0 | 32 | 522/map |
| Mean | 0.0 | 31.6 | 522/map |
| Median | 0 | 32 | 522/map |
| Total | 0 | 316 | 2,088 |

(League counts are from per-match move_policy invocation records, not per-game route candidate counts.)

---

## 7. Move Type Breakdown

| Source | Walk | Maneuver | Fly | Total |
|--------|------|----------|-----|-------|
| Fresh V3F | 0 | 0 | 0 | 0 |
| V3D (tactical_on, seed 0) | N/A* | N/A* | N/A* | 32 |
| League (4 matches) | 68 (100% empty) | 2,020 (99.01% empty) | 0 | 2,088 (99.04% empty) |

*Move type breakdown was not captured in V3D per-game data (only aggregate `route_metrics` was recorded).

---

## 8. Contradiction Classification

**Classification: SCENARIO_1910631192_CONTRADICTION_PROTOCOL_DIFFERENCE**

### Rationale

The fresh V3F evaluation protocol produces an outcome that matches neither V3D (0% empty with 316 attempts) nor the archived league (99% empty with 2,088 attempts). The scenario fails to enter battle phase at all under the V3F protocol, producing zero route-eligible actions and rendering route reliability measurement impossible.

This is classified as a **protocol difference** because:
1. The V3D evaluator (`run_v3b_formal_tactical_ablation` via `v3b_pairing.py`) enters battle and generates 31-32 route attempts per game for this scenario
2. The V3F evaluator (`run_v3f_route_reliability_evaluation.py`) with nearly identical setup produces 0 route attempts
3. The evaluator scripts differ in ways that apparently affect battle phase transition for this specific scenario
4. The source code and model weights have not changed (only `__pycache__` bytecode caches differ)
5. The same seeds produce completely different outcomes (32 moves vs 0, red winner vs blue winner)

### Implications

The V3F evaluator cannot be used to test route reliability on scenario 1910631192 because the scenario never enters battle phase under this protocol. This is a blocking issue: to compare V3D's 0% empty rate with the league's 99% empty rate, we need an evaluation protocol that reliably enters battle phase for this scenario.

The underlying contradiction between V3D's 0% empty rate and the league's ~99% empty rate remains **unresolved**. The V3F evaluation was intended to bridge these two datasets but instead reveals that the evaluation protocol itself is fragile for this scenario.

### Recommended Next Steps

1. Run the V3D evaluator (`run_v3b_formal_tactical_ablation.py`) on this scenario to verify it still produces 32 moves/battle entry
2. If V3D evaluator also fails (confirming non-determinism), investigate SDK env.setup for unseeded randomness
3. If V3D evaluator succeeds, identify the specific difference between the two evaluators that causes battle phase entry
4. Consider whether scenario 1910631192 has a deployment-phase structure that is particularly sensitive to agent action timing
