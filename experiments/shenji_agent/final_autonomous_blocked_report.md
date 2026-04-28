# Shenji Agent Autonomous Engineering — Final Blocked Report (Iteration 003-004)

**Date:** 2026-04-28
**Status:** B — Blocked with hard evidence
**Working directory:** `/remote-home/lizilong/shenji_agent_fresh` (fresh clone)
**Root FS:** 97% (107G free; was 100% full at start)

---

## 1. Current Git State

**HEAD:** `49039ca` — fix: correct manifest source_commit_sha to real HEAD

**Commit history (fresh clone, ahead of original by 5 commits):**
```
49039ca fix: correct manifest source_commit_sha to real HEAD
44ac552 chore: bump manifest source_commit_sha to 1860b86
1860b86 fix: passthrough move route diagnostics and update manifest commit
c59846b docs: finalize iteration 003 report with root cause and fix summary
9e36d05 fix: add move route diagnostics, fallback strategy, and context-aware occupy
cee21ce docs: add iteration 002 report and final blocked report
a33cdee fix: restore learned move and shoot engagement in selfplay
```

**Original repo** (`/remote-home/lizilong/shenji_agent`): Root FS was 100% full during Iteration 002. Now has space but all work was done in fresh clone. Fresh clone is ahead by 5 commits.

---

## 2. Root Cause: empty_route (85% move fallback rate)

### Finding
The **March movement cost layer (cost[1])** in `cost.pickle` has **78.4% empty adjacency cells** (5,556/7,084 on map_21). When a Vehicle is in March state, `get_move_type()` returns 1 (March), and `gen_move_route()` indexes into `cost[1]` where most cells have `{}`. The A* frontier exhausts → empty route → move fails.

### Fix
Multi-mode fallback: try all 4 move types (Maneuver, March, Walk, Fly), use shortest route per city. March failures fall back to Maneuver (100% coverage). Neighbor-step heuristic for dead-end positions.

---

## 3. Code Changes

### ai/agent.py (+176 lines)
- Multi-mode fallback route finding with dedup by shortest route
- Neighbor-step heuristic when no city reachable via any move type
- Context-aware Occupy: skip Move when operator stands on neutral/enemy city
- Guard: prevent learned path fallthrough to rule-based for Shoot/Move
- Per-move diagnostics: `record_move_route()` at all Move decision points

### ai/policy.py (+30 lines)
- `_reset_move_route_diagnostics()` — aggregate counters + sample buffer
- `record_move_route()` — records per-move diagnostics and updates aggregates

### training/selfplay.py (+40 lines)
- `_merge_policy_stats()` — passthrough `move_route_aggregate` and `move_route_samples`

### training/evaluate.py (+30 lines)
- `_merge_policy_stats()` — aggregate `move_route_aggregate` across matches

### tests/test_move_route_diagnostics.py (new, 10 tests)
### tests/test_mainline_contract.py (+45 lines, 3 tests updated)

---

## 4. Test Results

```
tests/test_mainline_contract.py — 35 passed
tests/test_move_route_diagnostics.py — 10 passed
Total: 45 tests, 0 failures
```

---

## 5. Self-Play Results

### Smoke (2 matches, 50 steps)

| Metric | Pre-Fix (formal) | Post-Fix (smoke) |
|--------|-------------------|-------------------|
| move_path_coverage | 0.147 (14.7%) | **1.000 (100%)** |
| learned_path_coverage | 0.155 (15.5%) | **1.000 (100%)** |
| empty_route_count | ~4,066 (85%) | **0 (0%)** |
| fallback_reasons | ["empty_route"] | **[] (none)** |
| intent_bias (move) | -0.636 | 0.0 |
| promotion_accepted | False | True |

Per-move diagnostics confirm: 37 moves, 0 empty routes, March completely avoided.

### Medium (5 matches, 400 steps)

| Metric | Pre-Fix | Post-Fix |
|--------|---------|----------|
| move_path_coverage | 0.147 | **1.000** |
| learned_path_coverage | 0.155 | **1.000** |
| shoot_path_coverage | 1.000 | **1.000** |
| empty_route_count | ~4,066 | **0** |
| overall_point_estimate | 0.500 | 0.500 (still 0-0) |
| promotion_accepted | False | False (score gates fail) |

All coverage gates pass. Score gates fail (all matches 0-0).

---

## 6. Why Scores Remain 0-0

The multi-mode fallback improved **both** candidate and incumbent movement:
- Candidate: 43 moves, all learned, reaching cities
- Opponent: 46 moves, rule-based, also reaching cities (was 85% empty before!)
- Both sides: 12 shoot actions per match → mutual attrition → stalemate

The learned policy's `choose_city` makes strategic decisions, but the rule-based opponent's random city selection is now equally effective since movement works for both. The policy's scoring function (`situation_score * 100 + city_value * 0.2 - distance * 0.35`) doesn't create enough strategic advantage.

---

## 7. Promotion Gate Status (Medium Run)

| Gate | Threshold | Actual | Status |
|------|-----------|--------|--------|
| overall_point_estimate | ≥ 0.55 | 0.50 | ❌ |
| overall_ci_lower | ≥ 0.50 | 0.50 | ❌ |
| complex_point_estimate | ≥ 0.55 | 0.50 | ❌ |
| complex_ci_lower | ≥ 0.50 | 0.50 | ❌ |
| learned_path_coverage | ≥ 0.30 | **1.00** | ✅ |
| move_path_coverage | ≥ 0.20 | **1.00** | ✅ |
| shoot_path_coverage | ≥ 0.20 | **1.00** | ✅ |
| elo_delta_lower | ≥ 15.0 | -0.0 | ❌ |
| max_bucket_regression | ≤ 0.05 | 0.00 | ✅ |

**Coverage gates**: All pass (were failing before)
**Score gates**: All fail (0-0 stalemate)

---

## 8. Remaining Blockers

### Code/plumbing: NONE — all verified fixed
### Model quality: Needs training data improvement or score function tuning

---

## 9. How to Sync Back to Original Repo

```bash
cd /remote-home/lizilong/shenji_agent
git remote add fresh /remote-home/lizilong/shenji_agent_fresh 2>/dev/null || true
git fetch fresh
git merge fresh/main
```
