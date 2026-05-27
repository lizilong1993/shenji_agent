# V3G SDK Candidate Generation Diagnostic

## Method

The trace was performed by reading the following source files and cross-referencing with SDK documentation:

- `/remote-home/lizilong/shenji_agent/ai/agent.py` — Agent action generation, `_generate_action()` method (lines 734-918), `step()` method (lines 609-670), `Map.gen_move_route()` (lines 117-149)
- `/remote-home/lizilong/shenji_agent/ai/policy.py` — `record_move_route()` (lines 325-340), `choose_city()` (lines 668-761), `note_action()` (lines 291-313), diagnostic aggregation (lines 342-398)
- `/remote-home/lizilong/shenji_agent/ai/features.py` — Feature extraction (not used in Move candidate flow)
- `/remote-home/lizilong/shenji_agent/land_wargame_sdk/docs/observation_note.json` — SDK valid_actions structure specification
- `/remote-home/lizilong/shenji_agent/land_wargame_sdk/docs/observation_example.json` — SDK valid_actions example
- `/remote-home/lizilong/shenji_agent/scripts/run_v3f_route_reliability_evaluation.py` — Diagnostic collection and aggregation
- `/remote-home/lizilong/shenji_agent/tests/test_v3f_route_reliability.py` — Existing test patterns
- `/remote-home/lizilong/shenji_agent/reports/v3f_route_reliability/03_route_instrumentation_audit.md` — Existing V3F audit
- `/remote-home/lizilong/shenji_agent/reports/v3f_route_reliability/route_instrumentation_audit.json` — Existing V3F audit JSON

The SDK `TrainEnv` is a compiled C++ module (`train_env.cpython-310-x86_64-linux-gnu.so`) whose internal operation cannot be inspected directly. The valid_actions structure was determined from SDK documentation.

---

## Code Path Diagrams

### Path A: SDK to valid_actions to Move candidates

```
TrainEnv.setup/step (compiled .so)
  |
  v
observation["valid_actions"]      -- agent.py:625
  |
  |  Structure: {obj_id: {action_type_int: candidate_or_none}}
  |  Format documented in observation_note.json:92-137
  |  Keys are integers (from compiled C++ module)
  |  For Move (type 1): value is always None (JSON null)
  |
  v
agent.step() loops over valid_actions  -- agent.py:654-660
  |
  |  for obj_id, action_map in valid_actions.items():
  |      for action_type in self.priority:
  |          if action_type not in action_map:
  |              continue  # Skip if action type not available
  |          action = self._generate_action(obj_id, action_type, action_map[action_type])
  |
  v
_generate_action(obj_id, ActionType.Move (=1), candidate=None)  -- agent.py:734,751
  |
  |  For Move: candidate is always None (SDK returns null for Move type)
  |  For Shoot: candidate is a list of target candidates
  |  For Occupy: candidate is always None
  |
  v
missing_move_candidate = not candidate  -- agent.py:755
  |
  |  candidate = None => not None = True
  |  This is ALWAYS True for Move actions
  |  This check is MEANINGLESS for Move -- no candidate list exists for Move
```

**SDK documentation evidence** (observation_note.json:94):
```json
"1-机动": null,
```

**SDK example evidence** (observation_example.json:2550):
```json
"14231": {
    "1": null,
    "6": [...list...],
    "11": null
}
```

Actions with `null` candidates: Move (1), Occupy (5), RemoveKeep (7), StopMove (10), WeaponLock (11), WeaponUnFold (12), CancelJMPlan (13)
Actions with list candidates: Shoot (2), GetOn (3), GetOff (4), ChangeState (6), JMPlan (8), GuideShoot (9)

**Key insight**: The Move action in SDK valid_actions is a **binary availability indicator** (key present == can move, key absent == cannot move), not a candidate list. The `null` value does NOT mean "empty candidate list" -- it means "no candidate structure for this action type."

---

### Path B: Agent to Policy candidate filtering (Move path)

```
_generate_action() Move branch
  |
  |  SDK candidate (=None) is ONLY checked via missing_move_candidate
  |  The candidate list is NOT used for route planning
  |
  v
get cities from observation  -- agent.py:756
  cities = self.observation.get("cities", [])
  |
  |  The SDK candidate is NEVER passed to the policy for Move
  |  Instead, ALL cities from the observation are considered as potential targets
  |
  v
Route planning to ALL cities with ALL move types  -- agent.py:776-787
  |
  |  for try_move_type in [move_type, ...other move types]:
  |      for city in cities:
  |          route = self.map.gen_move_route(cur_hex, city_hex, try_move_type)
  |          if route:  # A* found a path
  |              record in best_route_by_city
  |
  v
reachable_city_routes  -- agent.py:788-790
  |
  |  Only cities with non-empty A* routes (ANY move type)
  |
  v
policy.choose_city(bop, [cities with routes], observation, map)  -- agent.py:863-868
  |                                                                    policy.py:668-761
  |  Policy scores each reachable city using:
  |  - situation_score (counterfactual model)
  |  - distance_penalty
  |  - threat_penalty (from visible enemies)
  |  - tactical_score (tactical model)
  |  - objective_score (city value, ownership, proximity)
  |  - intent_bias, occupy_bias, occupy_proximity_bonus
  |
  |  Policy CANNOT "filter candidates to empty" because:
  |  - Cities come from observation, not from SDK candidates
  |  - choose_city() scores ALL provided cities and picks the best
  |  - choose_city() returns None only when no models loaded (policy.py:675-676)
  |  - Even with no models, the agent falls through to the non-optimized path
  |
  v
Route lookup for selected city  -- agent.py:871-874
  route = route_by_city.get(best_city, [])
  |
  |  route_by_city is built from reachable_city_routes which only has non-empty routes
  |  So route should always be non-empty here (defensive if route is empty case exists)
```

---

### Path C: Route planner to empty path versus success

```
Map.gen_move_route(begin, end, mode)  -- agent.py:117-149
  |
  |  Dijkstra/A* algorithm over hex cost map
  |  Uses self.cost[mode][row][col] for edge costs by move type
  |
  +-- Returns [] (empty path) when:  -- agent.py:118-124
  |    1. begin hex is invalid (out of bounds) -- agent.py:119
  |    2. end hex is invalid (out of bounds) -- agent.py:119
  |    3. mode >= len(self.cost) (move type not supported) -- agent.py:121
  |    4. begin == end (already at destination) -- agent.py:122
  |    5. No path exists (A* exhausts frontier without reaching end) -- agent.py:142-148
  |
  +-- Returns path (non-empty list) when:
        A* reaches end hex from begin hex via cost map
        Path is constructed by walking came_from backwards
        Returned in order from begin to end

reachable_city_routes check  -- agent.py:794
  |
  +-- if reachable_city_routes: (at least one city reachable)
  |     |
  |     +-- choose_city() succeeds -> route found -> SUCCESS  -- agent.py:869-895
  |     +-- choose_city() returns None -> return None (no action) -- agent.py:918
  |
  +-- if not reachable_city_routes: (no city reachable by any move type)
        |
        +-- Fallback neighbor attempt  -- agent.py:795-836
        |     |
        |     +-- neighbor brings operator closer to nearest city -> FALLBACK SUCCESS
        |     +-- no neighbor helps -> COMPLETE FAILURE -> record_move_route(failure)
        |
        +-- COMPLETE FAILURE path  -- agent.py:837-862
              record_move_route with:
                "no_valid_move_candidate" if missing_move_candidate else "empty_route"
              missing_move_candidate is ALWAYS True for Move
              So this ALWAYS records "no_valid_move_candidate"

"empty_route" alternate path  -- agent.py:896-912
  |
  |  Only triggered when choose_city() succeeds but route lookup returns empty
  |  This is defensive code -- should NOT be reachable in practice
  |  Because route_by_city is built from the same reachable_city_routes
  |  that were passed to choose_city()
```

---

### Path D: Fallback and no-op paths

```
No-op paths (return None from _generate_action for Move):
  |
  +-- No operator found (bop is None)  -- agent.py:753
  +-- Operator is UGV (sub_type == 3)  -- agent.py:753
  +-- Operator on own faction city     -- agent.py:764-771
  +-- No cities or no map loaded       -- agent.py:757-758
  +-- No reachable city, no fallback   -- agent.py:862
  +-- choose_city() returns None       -- agent.py:918
  |
  When None is returned: agent tries next action type in priority list
  If no action types succeed: operator does nothing this step

Fallback neighbor path  -- agent.py:795-836
  |
  |  Triggered only when reachable_city_routes is empty
  |
  v
  Find nearest city by hex distance  -- agent.py:795-799
  Check all neighbor hexes for one closer to nearest city  -- agent.py:803-811
  |
  +-- Closer neighbor found:  -- agent.py:812-836
  |     Generate simple_route = [neighbor_hex]
  |     record_move_route(fallback_neighbor, route_empty=False)
  |     note_action("move", used_learned=False)
  |     Return Move action with single-step path
  |
  +-- No closer neighbor:  -- agent.py:837-862
        record_move_route(no_valid_move_candidate, route_empty=True)
        note_action("move", used_learned=False, reason=no_valid_move_candidate)
        Return None (no move action generated)

Rule-based fallback path (non-optimized mode)  -- agent.py:990-1040
  |
  |  This is a SEPARATE code branch from the AI-optimized path
  |  Uses A* planner to random city or direction-based target
  |  record_move_route(rule_fallback)
  |  Does NOT use the policy at all
```

---

## Key Metrics (from code analysis)

### Raw SDK Move candidate count
- SDK always returns `None` (not a list) for Move type valid_actions
- Documentation: observation_note.json:94 -- `"1-机动": null`
- Key presence = Move available; key absence = Move NOT available
- **No candidate list exists for Move actions** -- the SDK's valid_actions for Move is a binary flag

### Route-eligible candidate count
- "Candidates" for Move are city hexes from `observation["cities"]`
- ALL cities are considered, no filtering by the SDK
- Route-eligible cities = cities where `gen_move_route` returns a non-empty path for ANY move type
- This count is recorded as `reachable_city_count` in `record_move_route()` diagnostics

### What causes `no_valid_move_candidate`
- **Root cause**: The A* route planner (Map.gen_move_route) cannot find a path to ANY city on the map with ANY move type, AND the fallback neighbor search also fails to find a closer hex
- The check `missing_move_candidate = not candidate` (agent.py:755) is ALWAYS `True` for Move because candidate is always `None`
- The diagnostic label appears only in the complete failure path (agent.py:837-862) which is reached after route planning AND fallback both fail
- **The label is a misnomer**: it does NOT mean "SDK returned empty candidate list" -- it means "route planning+fallback completely failed"

### Route planner invocation conditions
- Called at agent.py:782 for EACH city x EACH move type combination
- Up to `num_cities * 4` calls to `gen_move_route` per Move action attempt
- **Always called** when Move action is attempted with AI optimization (before `no_valid_move_candidate` is recorded)

### Route planner empty path conditions
- `Map.gen_move_route` returns `[]` when (agent.py:118-124):
  1. Invalid begin or end hex coordinates
  2. Move type mode out of `self.cost` range
  3. begin == end
  4. A* exploration reveals no path (exhausts all reachable hexes)

### How fallback is recorded
- `target_source: "fallback_neighbor"` -- agent.py:821
- `route_empty: False` -- agent.py:827
- `route_failure_reason: "none"` -- agent.py:828
- `selected_by_learned: False` -- agent.py:829
- Stored in `move_route_aggregate` as a non-empty route

### How no-op/illegal is recorded
- No-op (return None): NOT recorded as a move attempt in `record_move_route()`
- `note_action("move", used_learned=False, reason=...)` IS called for failures (agent.py:854-861)
- The no-op case at agent.py:753-754 (UGV or dead operator) does NOT call `record_move_route()` or `note_action()`

---

## Distinctions

### 1. SDK candidate list empty BEFORE policy filtering?
- **Partially misleading**: For Move, there IS no candidate list from the SDK. The SDK Move entry is `None` (null), which is a binary availability flag. The code at agent.py:755 treats `None` as "empty candidate" via `not candidate`, making `missing_move_candidate` always `True` for Move.
- **What actually happens**: When Move is SDK-valid, the agent proceeds to route planning. The "no_valid_move_candidate" label fires only AFTER route planning and fallback both fail.
- **Separable**: Yes, if we distinguish between "SDK Move key absent" (action not available, route not attempted) vs. "SDK Move key present, candidate is None" (normal case).
- **Recording**: Currently BOTH cases record "no_valid_move_candidate" when they reach the failure path, because `missing_move_candidate = not None = True`.

### 2. Policy filters candidates to empty?
- **No**: The policy receives reachable cities (from route planning), not SDK candidates. It scores all provided cities and selects the best. It cannot "filter to empty" because it returns None only when no models are loaded (policy.py:675-676), and even then the agent continues to the non-optimized path.
- **Recording**: Would not be tracked as a route failure episode.

### 3. Route planner called but returns empty path?
- **Yes**: This IS happening, but the diagnostic label is `no_valid_move_candidate`, not `empty_route`.
- The `empty_route` label (agent.py:910) is in a separate code path (city selected but route lookup fails) that should be unreachable in practice.
- **Recording**: The route planner failure is combined with the fallback failure under `no_valid_move_candidate`.
- **Separable**: YES, with additional instrumentation. The current code records `reachable_city_count` but does not separately record "no city reachable by any route" vs "some cities reachable but route to selected city is empty."

### 4. Fallback selected due to no candidate?
- **Yes**: Fallback is selected when NO city is reachable via A* with ANY move type.
- This is the SECOND attempt (after route planning to all cities fails).
- **Recording**: Clearly distinguishable via `target_source: "fallback_neighbor"` and `selected_by_learned: False`.
- **Separable**: Yes, current diagnostics separate fallback from other outcomes.

### 5. Illegal/no-op selected?
- **Yes**: `_generate_action()` returns `None` in several cases (no cities, on own city, UGV, complete failure, choose_city returns None).
- **Recording**: Only the complete failure path (agent.py:837-862) calls `record_move_route()`. Other no-op paths do not record route diagnostics.
- **Separable**: Partially -- the complete failure case IS recorded as `no_valid_move_candidate`. Other no-op cases (UGV, on own city) are not recorded at all.

### 6. Non-move actions incorrectly counted as route failures?
- **No**: `record_move_route()` is only called from the Move action branch (agent.py:751-918). All non-Move actions (Shoot, Occupy, etc.) do not call this function.
- Confirmed by existing test `test_record_move_route_denominator_is_move_only` (test_v3f_route_reliability.py:24-53).

### 7. Deployment-phase actions contaminating route metrics?
- **No**: During deployment (stage 1), the code at agent.py:634-652 generates only deployment actions (types 303, 333) and returns immediately without calling `_generate_action` for Move. The route diagnostics are never reached.

---

## Instrumentation Gaps

### Gap 1: `missing_move_candidate` is always True for Move
The check `missing_move_candidate = not candidate` at agent.py:755 was clearly designed for action types that have actual candidate lists (like Shoot). For Move, candidate is always `None`, making this check meaningless. The `no_valid_move_candidate` label is always selected in the failure path regardless of actual SDK behavior.

**Impact**: 100% of empty route events are attributed to "no_valid_move_candidate" even though the real cause is "route planning + fallback failed."

### Gap 2: Route planner failure format is not separable from candidate availability
The diagnostic records `reachable_city_count` and `total_cities`, but there is no flag for "route planner completely failed vs. SDK Move unavailable." The `no_valid_move_candidate` label conflates these two cases.

### Gap 3: The `empty_route` failure reason at agent.py:910 is likely dead code
The `empty_route` recording at agent.py:896-912 fires only if `choose_city()` returns a city but `route_by_city.get(best_city, [])` returns empty. Since `route_by_city` is built from the same `reachable_city_routes` that was passed to `choose_city()`, this path should be unreachable. Without runtime evidence, it should be considered dead code for the V3F/V3G analysis.

### Gap 4: No distinction between "no path to ANY city" and "path exists but not selected"
The diagnostic groups all complete failures under `no_valid_move_candidate`. Cases where A* found paths but the policy chose not to act (choose_city returns None) are not recorded with a distinct failure reason.

### Gap 5: Non-optimized path route failures are recorded differently
The rule-based Move branch (agent.py:990-1040) records `route_failure_reason: "empty_route"` when `gen_move_route` returns empty. This uses the SAME `empty_route` string as the theoretically unreachable path at agent.py:910, creating potential confusion between the two different code paths.

### Gap 6: `note_action()` trace_samples cap at 12
The policy's trace_samples list is capped at 12 entries (policy.py:306). In games with many move actions, the trace samples may not include representative failure examples.

---

## Recommendations for V3G Evaluator Instrumentation

### Recommendation 1: Separate the "SDK Move available" check from the "route planning failed" check
Add an additional diagnostic field that explicitly records whether the SDK Move key was present:
```
"sdk_move_available": true/false  (key "1" present in action_map)
```
This would let the evaluator distinguish "SDK said operator could not move" from "SDK said operator could move but route planning failed."

### Recommendation 2: Record route planner failure mode explicitly
Add a field that captures WHY the route planner failed:
```
"route_planner_result": "success" | "no_path_to_any_city" | "some_paths_but_not_selected"
```
Or more granularly:
```
"route_planner": {
    "cities_attempted": N,
    "cities_reachable": M,
    "best_move_type_success": null | "Maneuver" | "March" | "Walk" | "Fly",
    "fallback_attempted": true/false,
    "fallback_succeeded": true/false,
    "failure_reason": "no_valid_candidate" | "route_planner_empty" | "fallback_empty" | "policy_no_selection" | "route_lookup_empty"
}
```

### Recommendation 3: Add the SDK's `not candidate` check to the evaluator
Since the code already computes `missing_move_candidate` (agent.py:755), expose this value (and whether candidate was None vs []) in the move route diagnostics. This would allow the evaluator to determine:
- `missing_move_candidate = True, candidate is None`: Normal Move case (always true)
- `missing_move_candidate = True, candidate is []`: Edge case -- SDK returned empty list (if this ever happens)
- `missing_move_candidate = False, candidate is [...]`: SDK returned actual candidates

### Recommendation 4: Add an explicit "no fallback available" failure reason
Currently, "no_valid_move_candidate" conflates:
1. SDK returned no candidates
2. Route planner found no reachable cities
3. Fallback neighbor search failed

Add a separate "fallback_failed" or "no_reachable_city" reason so that the evaluator can distinguish between these layers of failure.

### Recommendation 5: Increase trace_samples cap or add periodic sampling
The current cap of 12 for `trace_samples` and 50 for `move_route_samples` limits diagnostic coverage in large games. Either increase the cap (to 100-200) or implement periodic sampling (every Nth event).

### Recommendation 6: Instrument the rule-based Move path with distinct failure reason
The rule-based path at agent.py:1031 uses `"empty_route"` which is the same string as the defensive path at agent.py:910. Use a distinct label like `"rule_based_empty_route"` to avoid confusion.

---

## Summary

**The critical finding of this diagnostic is that the `no_valid_move_candidate` label is fundamentally misleading.** 

For Move actions, the SDK valid_actions entry is `null` (a binary availability indicator, not a candidate list). The code at agent.py:755 (`not candidate`) always computes to `True` because `candidate` is always `None`. The label fires in the complete failure path (agent.py:837-862) which is reached only AFTER:
1. A* route planning to ALL cities with ALL move types has failed
2. Fallback neighbor search has also failed

The correct root cause is: **the A* route planner cannot find a path to any city, and no fallback neighbor hex is closer to the nearest city.** This is a route planning failure, not a candidate generation failure.

The V3F statement that "SDK candidate generation constraints account for 100% of empty route events" is based on a misnamed diagnostic label. The instrumented data is correct (these ARE real move failures), but the attribution to "candidate generation" is inaccurate.
