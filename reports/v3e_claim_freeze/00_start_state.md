# V3E Start State

Date: 2026-05-15

## Current state

| Field | Value |
|---|---|
| Current branch | `main` |
| HEAD | `0f5bd0d` |
| V3D commit | `0f5bd0d` ("rerun tactical outcome ablation with authenticated replay") |
| Worktree status | clean (only __pycache__ and untracked goal files + `selective_tactical_restore/`) |
| V3D reports exist | ✅ `reports/v3d_authenticated_tactical_ablation/` (FINAL_DECISION.md, claim_gate.json, statistics.json, multiseed_ablation.json, 00-07 reports) |
| V3 evidence package exists | ✅ `paper/v3_evidence_package/` (TACTICAL_CONTRIBUTION_STATUS.md, V3_READINESS.md, claim_to_evidence_matrix_v3.md, results_tables_v3.*, etc.) |
| PAPER_WRITING_READY.md exists | ✅ `paper/prewrite_package/PAPER_WRITING_READY.md` — state: PARTIAL_READY_WITH_BLOCKERS |
| main.tex exists | ❌ No LaTeX manuscript file found |

## Known tactical outcome conclusion

- Claim gate: `TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE`
- V3D final decision: `RESUME_V3_WITH_TACTICAL_DIAGNOSTIC_OR_WEAK_CLAIM`
- 15/15 paired comparisons completed, authenticated
- Score delta mean: -1.27, CI [-50.07, 34.82], Cohen's d: -0.015
- Win rate identical (0.533) between tactical-on and tactical-off
- Route empty rate: 0.0 across all 30 games
- Tactical action-level effect: supported (nonzero scores in all ON runs, zero in OFF)

## Backup branch

Already created: `backup/before_v3e_claim_freeze_20260515_162258`

## Goal directive

Follow `goals/codex_goal_v3e_baseline_route_claim_freeze.md`:
- Freeze V3 claims after V3D authenticated tactical outcome ablation
- Treat tactical as diagnostic/weak
- Audit authenticated route reliability, baseline/ablation coverage, promotion evidence, claim-to-evidence consistency
- Do not run long training, do not edit archived artifacts, do not strengthen claims beyond executable evidence
