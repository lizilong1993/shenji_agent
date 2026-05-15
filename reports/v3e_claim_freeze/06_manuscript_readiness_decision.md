# V3E Phase 6 — Manuscript Readiness Decision

**Date:** 2026-05-15
**Based on:** Phases 1-5 findings, evidence package audit, claim matrix freeze

---

## Input Summary

### Phase 1 — Tactical Claim Freeze
- Classification: `TACTICAL_ACTION_LEVEL_SUPPORTED` + `TACTICAL_OUTCOME_WEAK_OR_INCONCLUSIVE`
- Properly bounded as diagnostic/weak claim
- No further uncertainty on tactical outcome
- Status: **Frozen, not a manuscript blocker**

### Phase 2 — Route Reliability Audit (CRITICAL FINDING)
- Classification: `ROUTE_RELIABILITY_STILL_UNPROVEN`
- V3D route_empty_rate=0.0 is a zero-denominator artifact (0 move attempts in all 30 games)
- League sidecar shows 66.3% empty rate (15,641/23,580 move attempts)
- Status: **Critical unresolved technical issue — manuscript blocker**

### Phase 3 — Baseline/Ablation Coverage
- Classification: `BASELINE_COVERAGE_INSUFFICIENT_FOR_STRONG_PAPER`
- Rule incumbent comparison exists (formal)
- Tactical on/off ablation exists (formal but outcome weak)
- Missing: intent-only, situation-only, objective-only, random/legal, route-disabled
- Status: **Limits manuscript strength — secondary blocker**

### Phase 4 — Promotion/Two-Window
- Classification: `PROMOTION_SINGLE_WINDOW_ONLY`
- One formal window: accepted=true, rating 0.87 CI [0.80, 0.93]
- Two-window not achieved: consecutive_windows=0
- Status: **Missing evidence for promotion claims — secondary blocker**

### Phase 5 — Claim Matrix Freeze
- 16 claims audited and classified
- Route claim corrected (UNSUPPORTED from previously "measured")
- All classifications consistent with executable evidence
- Status: **Claim matrix is consistent and conservative**

---

## Decision

```
NEEDS_ROUTE_RELIABILITY_EVIDENCE_BEFORE_MANUSCRIPT
```

### Rationale

Three blockers were identified across the audit phases (route, baseline, promotion). Of these, route reliability is the most fundamental: if the planner produces empty routes for 66.3% of move attempts, the agent is not reliably executing a core gameplay function. Without resolving this, any comparison results (including the rule incumbent comparison and tactical ablation) rest on an unreliable substrate.

The baseline and promotion gaps are significant but secondary — a conservative technical report could be written with the existing single-window promotion and rule-incumbent comparison, acknowledging limited baseline coverage. But the route issue affects the validity of every evaluation result.

### What Would Change the Decision

- **Route reliability resolved** (e.g., route empty rate brought below 10%): decision would become `NEEDS_BASELINE_EXPANSION_BEFORE_MANUSCRIPT` or `READY_FOR_CONSERVATIVE_TECHNICAL_REPORT`
- **Route AND baselines AND two-window resolved**: decision would become `READY_FOR_CONSERVATIVE_SDK_EVALUATION_PAPER_WITH_LIMITATIONS`

### Path to Manuscript Readiness

1. **Primary (required):** Fix the route planner to reduce empty route rate. Re-evaluate with the same league protocol (100 paired maps) to confirm improvement.
2. **Secondary (recommended):** Add component-level ablations (intent-only, situation-only) and at minimum a random/legal action baseline.
3. **Secondary (recommended):** Demonstrate two-window promotion or adjust claims to single-window only.

---

## Appendix: Blockers Detail

| Blocker | Severity | Impact | Resolution Path |
|---------|----------|--------|-----------------|
| Route reliability unproven (66.3% empty rate) | CRITICAL | Undermines all gameplay evaluation | Fix route planner, re-evaluate |
| Baseline/ablation coverage limited | MODERATE | Limits comparative strength | Add component ablations |
| Two-window promotion not achieved | MODERATE | Limits promotion claims | Continue league iterations or relax claim |

**Recommended next goal:** `v3f_route_reliability_repair_and_evaluation` — Focus on diagnosing and fixing the route planner, then running a full league evaluation to verify improvement.
