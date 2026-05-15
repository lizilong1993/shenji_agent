# V3E Phase 5 — Claim Matrix Freeze

**Date:** 2026-05-15
**Scope:** All 16 major claims in the V3 evidence package
**Purpose:** Audit each claim against executed evidence, classify freeze boundary, and correct misleading entries.

---

## Claim-by-Claim Audit

### 1. closed-loop SDK evaluation exists

**Classification: SUPPORTED**

V3D executed authenticated paired evaluation: 15 pairs, 30 games, 0 errors, across 3 scenarios x 5 seeds. V3C recovered SDK MAC binding. Formal protocol with bootstrap CIs completed. The SDK evaluation loop is operational and has produced gated results.

**Source:** V3C SDK auth recovery, V3D authenticated multi-seed ablation, `reports/v3d_authenticated_tactical_ablation/`

### 2. authenticated replay is available

**Classification: SUPPORTED**

V3C diagnosed and recovered SDK MAC binding (`authenticator.set_mac_time()`), smoke tested with 1 authenticated pair. V3D executed 15 authenticated pairs. `mac_match=true`, `auth_environment_ready=true`.

**Source:** `reports/v3c_sdk_auth_recovery/`, `reports/v3d_authenticated_tactical_ablation/`

### 3. learned candidate uses intent/situation components

**Classification: SUPPORTED**

Checkpoint architecture: `intention_lstm.pth` and `situation_gcn.pth`. Legacy checkpoint compatibility restored via `LegacyIntentModel` and `LegacySituationModel` adapters. Policy auto-detects old vs new format. Architecture is clearly documented and executable.

**Source:** `ai/models.py`, `ai/policy.py`, `paper/v3_evidence_package/INTENT_CHECKPOINT_COMPATIBILITY_STATUS.md`

### 4. tactical branch is invoked

**Classification: SUPPORTED**

V3A action-level diagnostic: 18/18 candidates produced nonzero tactical scores, changed action rate = 1.0. V3D confirmed 351 nonzero tactical scores across all 15 ON runs, 0 in OFF runs. Tactical checkpoint loads as `TacticalScorer` and produces nonzero variegated scores in every ON game.

**Source:** `reports/v3a_tactical_ablation/`, `reports/v3d_authenticated_tactical_ablation/statistics.json`

### 5. tactical changes action selection

**Classification: PARTIALLY_SUPPORTED**

V3A demonstrated changed action rate = 1.0 (18/18 synthetic candidates). V3D confirmed nonzero tactical scores in all ON runs. However, the V3A diagnostic uses synthetic candidate states (17 actions from rule policy, 1 from full model) rather than real in-game action decisions. The effect on actual gameplay action selection is inferred from nonzero tactical scores, not directly measured. Action-level contribution is supported, but the scope is limited to diagnostic evidence.

**Source:** `reports/v3a_tactical_ablation/02_action_level_diagnostic.md`, `reports/v3d_authenticated_tactical_ablation/`

### 6. tactical improves outcome performance

**Classification: WEAK_OR_INCONCLUSIVE**

Authenticated paired evaluation (15 pairs, 30 games):
- Score delta mean: -1.27
- Bootstrap 95% CI: [-50.07, 34.82] (crosses zero)
- Cohen's d: -0.015 (negligible)
- Win rate: 0.533 in both modes (identical)
- 6 positive deltas, 3 negative, 6 zero

No reliable positive performance benefit is demonstrated. The claim "tactical improves outcome performance" is not supported by evidence.

**Source:** `reports/v3d_authenticated_tactical_ablation/04_statistics.md`, `reports/v3d_authenticated_tactical_ablation/statistics.json`

### 7. route empty issue is solved

**Classification: UNSUPPORTED**

League sidecar (`league_20260501_170204`) shows 66.3% empty route rate (15,641 empty events out of 23,580 move attempts). V3D's 0.0% is a zero-denominator result (0 move attempts occurred). The route empty issue has not been re-evaluated with the V3 codebase and remains the most critical unresolved technical problem.

**Source:** Phase 2 route reliability audit, `scripts/round005_route_diagnostic.py` output on `league_20260501_170204`

### 8. route degradation absent in V3D

**Classification: SUPPORTED (limited scenario coverage)**

V3D per-game metrics show 590 genuine move attempts (288 ON, 302 OFF) across all 30 games with 0 empty routes. This is a non-zero-denominator measurement with detailed per-attempt diagnostics (`move_route_samples` arrays in each per-game JSON). All 590 attempts used learned policy (learned_path_coverage=1.0). However, only 3 scenarios (1231, 1531, 1910631192) were tested — insufficient for broad route reliability claims.

**Correction note:** Prior audit incorrectly reported V3D total_move_attempts=0 (confused V3B pre-auth data with V3D authenticated data). The zero-denominator concern was valid for V3B but not V3D.

**Source:** Phase 2 route reliability audit (corrected), `reports/v3e_claim_freeze/02_route_reliability_audit.md`, V3D per-game metrics

### 9. candidate beats rule incumbent

**Classification: SUPPORTED**

Formal evidence from `league_20260501_170204`:
- Rating: 0.87 (95% CI [0.80, 0.93])
- Promotion: accepted=true
- Elo delta lower bound: 240.8
- Exploiter win rate: max 0.25
- Coverage: learned_path=0.37, move_path=0.34, shoot_path=1.0

The learned candidate convincingly beats the rule incumbent under the gated promotion protocol (100 paired maps).

**Source:** `experiments/shenji_agent/league_20260501_170204/promotion.json`, `rating.json`

### 10. promotion gate exists

**Classification: SUPPORTED**

`promotion.json` exists with `accepted=true`, `elo_delta_lower_bound=240.8`, `max_exploiter_win_rate=0.25`. All gate criteria are present: win rate comparison, Elo delta, exploitability, path coverage. Sidecars are complete.

**Source:** `experiments/shenji_agent/league_20260501_170204/promotion.json`

### 11. one promotion window exists

**Classification: SUPPORTED**

Multiple single-window promotions exist across May 2026 league runs. The latest (`league_20260501_170204`) has `accepted=true`. Evidence is consistent with prewrite package claims.

**Source:** Phase 4 promotion window audit, `experiments/shenji_agent/league_20260501_170204/`

### 12. two-window success exists

**Classification: FORBIDDEN**

`consecutive_windows=0` in latest league. Two-window hard-stop success is not proven. No evidence of two consecutive promotion windows in the formal evaluation series (champion is always `ai/weights`).

**Source:** Phase 4 promotion window audit, `league_state.json`

### 13. public SOTA exists

**Classification: FORBIDDEN**

No SOTA comparison has been conducted. The only comparison is against the rule incumbent. No external baselines, no public leaderboard entries.

**Source:** Phase 3 baseline audit

### 14. top-tier readiness exists

**Classification: FORBIDDEN**

No formal publication plan or venue targeting has been established. Manuscript does not exist. Key evidence gaps (route reliability, baselines, two-window) preclude top-tier readiness.

**Source:** `paper/prewrite_package/PAPER_WRITING_READY.md` (PARTIAL_READY_WITH_BLOCKERS)

### 15. baseline coverage is strong

**Classification: UNSUPPORTED**

Existing baselines: rule incumbent comparison (formal), tactical on/off (formal), promotion windows (single), exploiter eval (formal), authenticated replay (smoke + multi-seed).

Missing: intent-only ablation, situation-only ablation, objective-only baseline, random/legal baseline, route-disabled variant, two-window promotion.

Coverage is sufficient for a conservative technical report but not for strong comparative claims.

**Source:** Phase 3 baseline audit, `reports/v3e_claim_freeze/03_baseline_ablation_audit.md`

### 16. manuscript-ready evidence package exists

**Classification: PARTIALLY_SUPPORTED**

An evidence package exists (`paper/v3_evidence_package/`) with:
- Claim-to-evidence matrix ✓
- V3 readiness table ✓
- Tactical contribution status ✓
- Intent checkpoint compatibility ✓
- Results tables ✓

However, PAPER_WRITING_READY.md reports PARTIAL_READY_WITH_BLOCKERS. Three critical blockers:
1. Route reliability unproven (66.3% empty rate unresolved)
2. Baseline/ablation coverage limited (no component-level ablations)
3. Two-window promotion not achieved

The package is structurally present but has substantive gaps.

---

## Summary

| # | Claim | Classification |
|---|-------|---------------|
| 1 | closed-loop SDK evaluation exists | SUPPORTED |
| 2 | authenticated replay is available | SUPPORTED |
| 3 | learned candidate uses intent/situation components | SUPPORTED |
| 4 | tactical branch is invoked | SUPPORTED |
| 5 | tactical changes action selection | PARTIALLY_SUPPORTED |
| 6 | tactical improves outcome performance | WEAK_OR_INCONCLUSIVE |
| 7 | route empty issue is solved | UNSUPPORTED |
| 8 | route degradation absent in V3D | SUPPORTED (limited to 3 scenarios) |
| 9 | candidate beats rule incumbent | SUPPORTED |
| 10 | promotion gate exists | SUPPORTED |
| 11 | one promotion window exists | SUPPORTED |
| 12 | two-window success exists | FORBIDDEN |
| 13 | public SOTA exists | FORBIDDEN |
| 14 | top-tier readiness exists | FORBIDDEN |
| 15 | baseline coverage is strong | UNSUPPORTED |
| 16 | manuscript-ready evidence package exists | PARTIALLY_SUPPORTED |

## Corrections Applied

1. **claim_to_evidence_matrix_v3.md C17** — Corrected from "unproven (zero-denominator)" to "no degradation observed (0% over 590 attempts, 3 scenarios)" with accurate description of V3D per-game metrics.

2. **V3_READINESS.md route row** — Corrected from "unproven (zero-denominator)" to "no degradation observed (3 scenarios)" with accurate league comparison.

3. **V3_READINESS.md V3D section** — Corrected warning about zero-denominator to reflect genuine 590-attempt measurement.

4. **source_evidence.md** — Corrected V3D row and route diagnostics row to reflect accurate V3D route data. V3E row updated with corrected claim scope.
