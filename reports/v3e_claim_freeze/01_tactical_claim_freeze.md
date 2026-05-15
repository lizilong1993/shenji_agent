# V3E Phase 1 — Tactical Claim Freeze

**Date:** 2026-05-15
**Audit scope:** V3A, V3B, V3C, V3D evidence
**Purpose:** Determine which claims about the tactical branch are supported by executed evidence, classify the claim freeze boundaries, and enumerate forbidden claims.

---

## Audit Questions

### 1. Did V3A establish action-level tactical invocation?

**Yes.** V3A ran an action-level tactical diagnostic using synthetic candidate states. The results are:

| Metric | Value |
|--------|-------|
| Status | `ACTION_LEVEL_TACTICAL_CONTRIBUTION_SUPPORTED` |
| Candidate actions examined | 18 (9 shoot + 9 move_city) |
| Nonzero score count | 18 / 18 (100%) |
| Changed action rate (vs off) | 1.0 |
| Score mean (all candidates) | -0.138 |
| Score std | 0.101 |
| Variegated scores | Yes -- all nonzero, std >> 0 |

The tactical model produces nonzero, non-constant scores across all candidate actions. Under a tactical-off baseline (always 0.0), every nonzero score constitutes a changed action ranking.

**Source:** `reports/v3a_tactical_ablation/02_action_level_diagnostic.md`, `reports/v3a_tactical_ablation/action_level_diagnostic.json`

### 2. Did V3B complete formal paired evaluation?

**Yes -- but auth-blocked, outcome declared INCONCLUSIBLE.**

The V3B formal paired evaluation infrastructure was built and executed:
- 15 pairs (3 scenarios x 5 seeds), 30 games, 0 errors
- Protocol defined with strict pairing constraints, bootstrap CI, and effect size reporting
- All 30 games completed with 0 crashes

However, the SDK environment authentication was bound to MAC `02:42:ac:11:00:02`, while the current environment had MAC `02:42:ac:11:00:07`. This blocked battle-phase gameplay entirely:
- No move routes were attempted (0 move attempts across all 30 games)
- Tactical-on and tactical-off produced identical outcomes (score delta = 0.0 across all pairs)
- Action-level divergence and route reliability were not measurable

The terminal state was `TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE`, meaning the outcome question could not be definitively answered.

**Source:** `reports/v3b_formal_tactical_ablation/FINAL_DECISION.md`, `reports/v3b_formal_tactical_ablation/route_reliability_audit.json`, `reports/v3b_formal_tactical_ablation/07_claim_boundary.md`

### 3. Did V3C recover SDK auth replay?

**Yes.** V3C diagnosed the MAC binding issue and applied a recovery:

| Detail | Value |
|--------|-------|
| Classification | `SDK_MAC_BINDING_CONFIRMED` |
| Bound MAC (engine_config) | `02:42:ac:11:00:02` |
| Current MAC | `02:42:ac:11:00:07` |
| Recovery method | `authenticator.set_mac_time()` (rebind to current MAC) |
| Pre-fix auth | Fail |
| Post-fix auth | Pass |
| Smoke test status | Authenticated replay confirmed |

The minimal replay smoke test (1 pair, scenario 1231) confirmed:
- Authentication passed ("did pass authentication" printed)
- Deployment phase entered and passed
- Battle phase entered with 3 move attempts in both runs
- Tactical mode confirmed: nonzero count = 3 (on) vs 0 (off)
- Route empty rate = 0.0
- No SDK binaries modified, no authenticator code patched

**Source:** `reports/v3c_sdk_auth_recovery/01_sdk_auth_audit.json`, `reports/v3c_sdk_auth_recovery/03_replay_smoke.md`, `reports/v3c_sdk_auth_recovery/replay_smoke.json`

### 4. Did V3D execute authenticated paired outcome ablation?

**Yes.** V3D ran the full formal paired evaluation under authenticated replay:

| Metric | Value |
|--------|-------|
| Protocol | 5 seeds x 3 scenarios = 15 pairs |
| Completed | 15 / 15 (100%) |
| Errors | 0 |
| Tactical nonzero (ON) | 351 total across all pairs |
| Tactical nonzero (OFF) | 0 |
| Route empty rate (ON) | 0.0 |
| Route empty rate (OFF) | 0.0 |

Every tactical-on game produced nonzero tactical scores. Every tactical-off game produced zero tactical scores. Route empty rate was 0.0 across all 30 games. The authenticated ablation completes successfully.

**Source:** `reports/v3d_authenticated_tactical_ablation/02_protocol.md`, `reports/v3d_authenticated_tactical_ablation/04_statistics.md`, `reports/v3d_authenticated_tactical_ablation/statistics.json`

### 5. What is the outcome delta?

| Statistic | Value |
|-----------|-------|
| Score delta mean | -1.27 |
| Score delta median | 0.0 |
| Score delta std | 85.84 |
| Bootstrap 95% CI | [-50.07, 34.82] |
| Cohen's d | -0.015 (negligible) |
| Positive deltas | 6 |
| Negative deltas | 3 |
| Zero deltas | 6 |

The mean score delta is slightly negative (-1.27), the confidence interval crosses zero widely, and the effect size is negligible (d = -0.015). There is no reliable positive performance benefit.

**Source:** `reports/v3d_authenticated_tactical_ablation/statistics.json`, `reports/v3d_authenticated_tactical_ablation/04_statistics.md`

### 6. Is win rate affected?

**No.** Win rate is identical between modes:

| Mode | Wins | Games | Win rate |
|------|------|-------|----------|
| Tactical ON | 8 | 15 | 0.533 |
| Tactical OFF | 8 | 15 | 0.533 |
| Delta | 0 | -- | 0.0 |

Win rate delta = 0.0. Tactical mode has zero impact on win/loss outcomes under the tested protocol.

**Source:** `reports/v3d_authenticated_tactical_ablation/statistics.json`

### 7. Is route empty rate affected?

**No.** Route empty rate is 0.0 in both modes across all 30 games:

| Metric | Tactical ON | Tactical OFF | Delta |
|--------|-------------|--------------|-------|
| Route empty rate | 0.0 | 0.0 | 0.0 |
| Bootstrap 95% CI | -- | -- | [0.0, 0.0] |

Tactical mode does not degrade or improve route reliability.

**Source:** `reports/v3d_authenticated_tactical_ablation/statistics.json`, `reports/v3d_authenticated_tactical_ablation/04_statistics.md`

---

## Classification

```
TACTICAL_ACTION_LEVEL_SUPPORTED
TACTICAL_OUTCOME_WEAK_OR_INCONCLUSIVE
```

### Rationale

**TACTICAL_ACTION_LEVEL_SUPPORTED** -- V3A demonstrated that the tactical branch produces nonzero, variegated scores across candidate actions (18/18 candidates nonzero, changed action rate = 1.0). V3D confirmed under authenticated replay that tactical is invoked in every tactical-on game (351 nonzero scores across 15 pairs) and never invoked in tactical-off games (0 nonzero scores). The tactical branch clearly affects action selection.

**TACTICAL_OUTCOME_WEAK_OR_INCONCLUSIVE** -- V3D's authenticated paired ablation (15 pairs, 30 games) shows: score delta mean = -1.27, bootstrap 95% CI [-50.07, 34.82] crosses zero, Cohen's d = -0.015 (negligible), win rate identical (0.533), route empty rate unaffected (0.0). Tactical clearly executes and affects gameplay (nonzero score deltas ranging from -264 to +158), but no consistent positive outcome benefit is demonstrated.

---

## Allowed Claim

> "The tactical branch is invoked and produces nonzero diagnostic/action-level effects under authenticated replay, but paired multi-seed outcome ablation does not show a reliable performance benefit."

## Forbidden Claims (must not appear in paper)

1. **Tactical improves win rate** -- Win rate is identical (0.533) between tactical-on and tactical-off across 15 pairs.
2. **Tactical improves outcome performance** -- Score delta mean is -1.27, CI crosses zero, Cohen's d is -0.015.
3. **Tactical is a main performance contribution** -- Outcome-level effect is weak/inconclusive; no consistent positive benefit.
4. **Tactical contribution is formally proven** -- The evidence supports action-level effects but does not formally prove outcome-level benefit.

---

## Evidence Source Summary

| Phase | Key Finding | File |
|-------|-------------|------|
| V3A | Action-level contribution supported (18/18 nonzero, changed_action_rate = 1.0) | `reports/v3a_tactical_ablation/action_level_diagnostic.json` |
| V3B | Formal evaluation completed but auth-blocked (MAC mismatch) | `reports/v3b_formal_tactical_ablation/FINAL_DECISION.md` |
| V3C | SDK auth recovered via `set_mac_time()`, smoke pass confirmed | `reports/v3c_sdk_auth_recovery/01_sdk_auth_audit.json` |
| V3C | Authenticated replay: tactical nonzero (3) vs off (0), route empty rate 0.0 | `reports/v3c_sdk_auth_recovery/replay_smoke.json` |
| V3D | 15/15 paired comparisons completed, tactical invoked 351x (ON) vs 0 (OFF) | `reports/v3d_authenticated_tactical_ablation/statistics.json` |
| V3D | Outcome delta: mean -1.27, CI [-50.07, 34.82], d = -0.015 | `reports/v3d_authenticated_tactical_ablation/statistics.json` |
| V3D | Win rate: identical 0.533, route empty: 0.0 both modes | `reports/v3d_authenticated_tactical_ablation/statistics.json` |
| V3D | Claim gate: `TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE` | `reports/v3d_authenticated_tactical_ablation/claim_gate.json` |
