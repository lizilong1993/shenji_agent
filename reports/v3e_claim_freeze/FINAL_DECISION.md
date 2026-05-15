# V3E Final Decision

Date: 2026-05-15

## Decision

```
CLAIMS_FROZEN_NEEDS_ROUTE_EVIDENCE
```

## What was audited

| Phase | Scope | Result |
|---|---|---|
| Phase 0 | Start state | Branch main at 0f5bd0d (V3D commit). All reports present. |
| Phase 1 | Tactical claim freeze | Action-level: SUPPORTED. Outcome: WEAK_OR_INCONCLUSIVE. Frozen. |
| Phase 2 | Route reliability audit | 0% empty over 590 attempts (V3D, 3 scenarios). 66.3% empty over 23,580 attempts (league, ~100 scenarios). **ROUTE_NO_DEGRADATION_OBSERVED_ONLY.** |
| Phase 3 | Baseline/ablation coverage | Rule incumbent + tactical ablation exist. Component ablations, two-window, random baseline missing. **INSUFFICIENT_FOR_STRONG_PAPER.** |
| Phase 4 | Promotion/two-window audit | Single window: verified (accepted=true, rating 0.87). Two-window: not proven. **SINGLE_WINDOW_ONLY.** |
| Phase 5 | Claim-to-evidence matrix | 16 claims audited. Tactical bounded. Route corrected from zero-denominator artifact to genuine measurement. All claims consistent with evidence. |
| Phase 6 | Manuscript readiness | **NEEDS_ROUTE_RELIABILITY_EVIDENCE_BEFORE_MANUSCRIPT.** |
| Phase 7 | Consistency tests | 21 tests written, all pass. |
| Phase 8 | Validation | 21/21 tests pass. Compileall clean. |

## Corrected finding (vs initial Phase 2 error)

The initial Phase 2 audit erroneously reported V3D had 0 move attempts (confused V3B pre-auth data with V3D authenticated data). **V3D actually has 590 genuine move attempts with 0% empty rate.** This finding was corrected and propagated through the evidence package.

## Frozen claim boundaries

- **Tactical outcome**: WEAK_OR_INCONCLUSIVE — frozen, not a manuscript blocker
- **Route reliability**: Not broadly proven — **primary blocker** (66.3% league rate unresolved)
- **Baseline coverage**: Limited — secondary blocker (missing component-level ablations)
- **Two-window promotion**: Not achieved — secondary blocker
- **Public SOTA**: FORBIDDEN
- **Top-tier readiness**: FORBIDDEN

## Rationale

All claims are now internally consistent and consistent with executed evidence. The claim matrix is frozen and conservative. No unsupported claims remain. However, the 66.3% route-empty rate from the league evaluation is the most critical unresolved technical issue, as it affects the validity of every game-level evaluation result.

## Recommended next goal

```
v3f_route_reliability_repair_and_evaluation
```

Fix the route planner to reduce empty route rate, then re-evaluate with the league protocol (100 paired maps) to confirm improvement.
