# Shenji Agent Autonomous Engineering — Iteration 004

## Status: In Progress (smoke self-play running)

**Date:** 2026-04-28
**Working directory:** `/remote-home/lizilong/shenji_agent_fresh` (fresh clone)
**Root FS:** 97% (107G free, up from 100%)

---

## Smoke Self-Play Results (league_20260428_193104)

### Key Metrics (Pre-Fix vs Post-Fix)

| Metric | Pre-Fix (formal) | Post-Fix (smoke) | Change |
|--------|------------------|------------------|--------|
| **move_path_coverage** | 0.147 | **1.000** | +580% |
| **learned_path_coverage** | 0.155 | **1.000** | +545% |
| **move empty_route rate** | ~85% | **0%** | eliminated |
| **move_total** | 4,764 | 36 | smoke (2 matches × 50 steps) |
| **move_learned** | 698 (14.7%) | 36 (100%) | all learned |
| **fallback_reasons** | ["empty_route"] | **[] (empty)** | no failures |
| **intent_bias (move)** | -0.636 | **0.0** | neutralized |
| **promotion_accepted** | False (1/8 gates) | **True (all gates)** | ✅ |

### Per-Move Diagnostics (after re-run)
*Pending — smoke re-run with diagnostics passthrough in progress*

---

## Changes in Iteration 004

### 1. Move Route Diagnostics Passthrough
**Files:** `training/selfplay.py`, `training/evaluate.py`

Added `move_route_aggregate` and `move_route_samples` to the merge functions:
- `selfplay._merge_policy_stats`: merges per-leg diagnostics into match-level stats
- `evaluate._merge_policy_stats`: merges match-level diagnostics into global rating/policy stats

This ensures per-move route diagnostics appear in:
- `match_results.json` → `candidate_policy_stats.move_route_aggregate`
- `rating.json` → `policy.move_route_aggregate`

---

## Smoke Run Details

- Config: `configs/selfplay_smoke_learned_engagement.yaml`
- Paired matches: 2
- Max steps: 50
- Candidate: `ai/weights` (challenger profile, learned policy)
- Incumbent: `ai/weights_incumbent_rule` (baseline profile, rule-based)
- Device: CPU (CUDA_VISIBLE_DEVICES="" SHENJI_POLICY_DEVICE=cpu)

### Match Results

| Scenario | Bucket | Candidate Score | Opponent Score | Paired |
|----------|--------|-----------------|----------------|--------|
| 2230531221 | complex | 0.0 | 0.0 | 0.5 |
| 2010211129 | complex | 0.0 | 0.0 | 0.5 |

Both games ended 0-0 (no scoring), but all moves were learned and found routes. The lack of scoring suggests:
1. 50 steps may not be enough to reach and occupy cities
2. Opponents may block occupation
3. Shoot actions may be needed to clear enemies before occupying

---

## Next Steps

1. Smoke self-play re-run (in progress) — get full move_route_aggregate in JSON
2. If diagnostics confirm 0% empty_route, run formal self-play
3. If formal shows scoring improvement, proceed to A completed
4. If still 0-0 but routes work, investigate Occupy timing and enemy interference
5. Commit diagnostics passthrough changes
