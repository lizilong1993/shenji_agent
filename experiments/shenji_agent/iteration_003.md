# Shenji Agent Autonomous Engineering — Iteration 003

## Status: In Progress

Working directory: `/remote-home/lizilong/shenji_agent_fresh` (fresh clone)

---

## Root Cause Analysis: empty_route (85% move fallback rate)

### Finding: Sparse March Cost Layer

The exploration of `cost.pickle` for map_21 revealed that the **March movement cost layer (cost[1])** has **78.4% of cells (5556/7084) with empty adjacency dicts `{}`**.

This means when a Vehicle operator's `move_state == MoveType.March` (value=1), `get_move_type()` returns `1` (March), and `gen_move_route()` indexes into `cost[1][row][col].items()`. For 78.4% of grid cells including:
- Starting positions (e.g., hex 2730 at row=27, col=30 has 0 items)
- 3 of 7 cities on map_21 (4838, 4942, 5038)

the A* pathfinder finds no neighbors, exhausts the frontier, and returns `[]`.

### Cost Layer Summary (map_21)

| Mode | MoveType | Enum | Cells with Data | Empty Cells |
|------|----------|------|-----------------|-------------|
| 0    | Maneuver | 0    | 7084 (100%)     | 0           |
| 1    | March    | 1    | 1528 (21.6%)    | 5556 (78.4%)|
| 2    | Walk     | 2    | 7084 (100%)     | 0           |
| 3    | Fly      | 3    | 7084 (100%)     | 0           |

### Additional Concerns

Even with Maneuver (100% cell coverage), the cost graph might not be fully connected. Some cells may only have edges in one direction or to unreachable locations. This requires runtime diagnostics to confirm.

---

## Changes Implemented

### 1. Per-Move Route Diagnostics (policy.py + agent.py)

Added `record_move_route()` method to `LearnedPolicy` that records per-move diagnostics:
- `operator_id`, `operator_type`, `current_hex`
- `move_type`, `move_type_name`
- `target_kind`, `target_source`, `target_city_id`, `target_hex`
- `reachable_city_count`, `total_cities`
- `route_len`, `route_empty`, `route_failure_reason`
- `selected_by_learned`

Aggregated in `get_diagnostics()` under `move_route_aggregate`:
- `total_attempts`, `empty_route_count`
- `reachable_city_total`
- `by_move_type` distribution
- `route_failure_reasons` counts

Detailed samples (up to 50) in `move_route_samples`.

### 2. Move Type Fallback Strategy

When no city is reachable with the primary move type:
1. Try ALL alternative move types (Maneuver→March→Walk→Fly) in priority order
2. Use the best (shortest) route per city across all tried move types
3. If still no route, try neighbor-step heuristic: step one hex toward nearest city
4. Record diagnostics at each stage with target_source indicating which strategy was used

### 3. Context-Aware Occupy

When an operator is on a neutral/enemy city:
- Skip Move (return None) for that operator
- This allows Occupy (next in priority) to execute
- Prevents the agent from moving away from a scorable city

Rule-based Move also gets the same check.

### 4. Resetting Diagnostics

`reset()` now properly clears all move route diagnostics through `_reset_move_route_diagnostics()`.

---

## Tests

### New: tests/test_move_route_diagnostics.py
10 tests covering:
- empty_route recording with route_empty=true and failure_reason
- fallback_neighbor not counted as learned success
- selected_by_learned accuracy
- move_path_coverage not inflated by shoot
- missing target recorded as failure
- empty_route aggregate count
- diagnostic persistence in get_diagnostics output
- move_type distribution recording
- reset clearing diagnostics
- fallback move type recorded in target_source

### Updated: tests/test_mainline_contract.py
Added `get_distance()`, `get_neighbors()`, and `record_move_route()` to 3 FakeMap + FakePolicy classes.

---

## Pre-Fix Formal Run Analysis (league_20260428_171537)

| Metric | Value |
|--------|-------|
| Total matches | 30 |
| All scores | 0-0 |
| Move total attempts | 4,764 |
| Move learned | 698 (14.65%) |
| empty_route rate | ~85% |
| Shoot total | 50 |
| Shoot learned | 50 (100%) |
| Promotion gates passed | 1/8 (shoot_path_coverage only) |
| Policy loaded matches | 30/30 |

### Answering the 17 Questions:

1. **30 matches complete?** Yes, all 30 completed without errors.
2. **candidate profile/artifact/manifest:** challenger, ai/weights, manifest loaded.
3. **incumbent profile/artifact/manifest:** baseline, ai/weights_incumbent_rule, manifest loaded.
4. **Candidate vs incumbent distinction:** Clear — candidate uses learned policy, incumbent uses rule baseline.
5. **Policy load success:** 30/30 matches, learned policy enabled.
6. **learned_score_calls total:** 698 (move) + 50 (shoot) = 748.
7. **learned_action_candidates:** 698 moves used choose_city.
8. **learned_action_selected:** 698 moves (all that found reachable cities).
9. **move attempts total:** 4,764.
10. **move fallback count:** 4,066 non-learned moves (most empty_route).
11. **empty_route count:** ~4,066 (85% of 4,764).
12. **empty_route / move attempts:** ~85.4%.
13. **reach city rate:** ~14.6% (698/4764).
14. **Why 0-0?** No cities reached = no occupation = no scoring. Both agents fail to reach cities consistently.
15. **Promotion gate failures:** overall_point_estimate, overall_ci_lower, complex_point_estimate, complex_ci_lower, learned_path_coverage, move_path_coverage, elo_delta_lower.
16. **Why shoot_path_coverage=1.0 but learned/move fail?** Shoot coverage is 50/50 because all shoot attempts use learned policy. Move coverage fails because 85% of move attempts can't find any reachable city.
17. **Current diagnostics sufficient?** No — previous diagnostics only had `empty_route` as a single fallback reason. New per-move diagnostics now capture move_type, reachable_city_count, and target_source.

---

## Next Steps (Pending)

1. Confirm tests pass (mainline_contract + move_route_diagnostics)
2. Commit changes
3. Run smoke self-play with diagnostics enabled
4. Compare empty_route rate pre/post fix
5. If smoke shows improvement, run formal self-play
6. Update final report

---

## Git State

Working in fresh clone at `/remote-home/lizilong/shenji_agent_fresh`
Original repo root FS is still 100% full — cannot write commits to original.

Files modified:
- `ai/agent.py` — move fallback, occupy context, route diagnostics
- `ai/policy.py` — record_move_route, get_diagnostics, reset
- `tests/test_mainline_contract.py` — FakeMap/Policy updates
- `tests/test_move_route_diagnostics.py` — new tests
