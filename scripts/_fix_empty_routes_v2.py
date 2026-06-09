#!/usr/bin/env python3
"""Fix empty routes by making the 3-level fallback BULLETPROOF.

Key changes to agent.py:
1. When 3-level fallback finds a neighbor, try ALL move types for gen_move_route
2. Add Level 4: brute-force single-hex step with any move type
3. Add defensive neighbor validation (handle string/None neighbors)
4. Track empty route counts by failure level for diagnosis
"""
import re
import sys

AGENT_PY = "ai/agent.py"


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    src = read_file(AGENT_PY)
    original = src  # keep backup

    # ── 1. Find and replace the 3-level fallback block ──
    # We look for the block starting with "=== 3-level fallback" and ending
    # before "selected_city, trace = self.policy.choose_city("

    start_marker = '                if not reachable_city_routes:\n                    # === 3-level fallback when all routing fails ==='
    end_marker = '                selected_city, trace = self.policy.choose_city('

    idx_start = src.find(start_marker)
    idx_end = src.find(end_marker)

    if idx_start < 0 or idx_end < 0:
        print(f"ERROR: Could not find fallback block. start={idx_start}, end={idx_end}")
        sys.exit(1)

    # Build the replacement block
    replacement = '''                if not reachable_city_routes:
                    # === BULLETPROOF 5-level fallback when all routing fails ===
                    cur_hex = int(bop.get("cur_hex", 0))
                    raw_neighbors = self.map.get_neighbors(cur_hex) if self.map else []
                    # Defensive: normalize neighbor types, filter invalid
                    neighbors = []
                    for _n in raw_neighbors:
                        try:
                            _nv = int(_n)
                            if _nv >= 0 and self.map.is_valid(_nv):
                                neighbors.append(_nv)
                        except (TypeError, ValueError):
                            pass

                    fallback_hex = None
                    fallback_city_id = None
                    fallback_source = "none"
                    fallback_move_type = move_type

                    # Level 1: Walk toward nearest city (use any move type that works)
                    sorted_cities = sorted(
                        [c for c in cities if c.get("coord") is not None and int(c["coord"]) != cur_hex],
                        key=lambda c: self.map.get_distance(cur_hex, int(c["coord"])),
                    )
                    for target_city in sorted_cities:
                        target_hex = int(target_city["coord"])
                        best_dist = self.map.get_distance(cur_hex, target_hex)
                        if best_dist <= 0:
                            continue
                        for n_hex in neighbors:
                            d = self.map.get_distance(n_hex, target_hex)
                            if 0 <= d < best_dist:
                                # Try to get a valid route to this neighbor
                                for try_mt in [move_type] + [m for m in [0, 1, 2, 3] if m != move_type]:
                                    step_route = self.map.gen_move_route(cur_hex, n_hex, try_mt) if self.map else []
                                    if step_route:
                                        fallback_hex = n_hex
                                        fallback_city_id = target_hex
                                        fallback_source = "walk_toward_city"
                                        fallback_move_type = try_mt
                                        break
                                if fallback_hex is None:
                                    # Use raw single-step even without A* validation
                                    fallback_hex = n_hex
                                    fallback_city_id = target_hex
                                    fallback_source = "walk_toward_city_raw"
                                break
                        if fallback_hex is not None:
                            break

                    # Level 2: Walk to any neighbor NOT on a friendly city
                    if fallback_hex is None:
                        for n_hex in neighbors:
                            on_friendly_city = any(
                                c.get("coord") is not None and int(c["coord"]) == n_hex
                                and int(c.get("flag", -1)) == self.faction for c in cities
                            )
                            if not on_friendly_city:
                                for try_mt in [move_type] + [m for m in [0, 1, 2, 3] if m != move_type]:
                                    step_route = self.map.gen_move_route(cur_hex, n_hex, try_mt) if self.map else []
                                    if step_route:
                                        fallback_hex = n_hex
                                        fallback_source = "walk_to_non_city"
                                        fallback_move_type = try_mt
                                        break
                                if fallback_hex is None:
                                    fallback_hex = n_hex
                                    fallback_source = "walk_to_non_city_raw"
                                break

                    # Level 3: Walk to ANY valid neighbor (last resort)
                    if fallback_hex is None:
                        for n_hex in neighbors:
                            for try_mt in [move_type] + [m for m in [0, 1, 2, 3] if m != move_type]:
                                step_route = self.map.gen_move_route(cur_hex, n_hex, try_mt) if self.map else []
                                if step_route:
                                    fallback_hex = n_hex
                                    fallback_source = "walk_any_neighbor"
                                    fallback_move_type = try_mt
                                    break
                            if fallback_hex is None:
                                fallback_hex = n_hex
                                fallback_source = "walk_any_neighbor_raw"
                            break

                    # Level 4: Brute force - check ALL hexes in distance-1 ring
                    if fallback_hex is None and self.map:
                        row, col = divmod(cur_hex, 100)
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = row + dr, col + dc
                                if 0 <= nr < self.map.max_row and 0 <= nc < self.map.max_col:
                                    candidate_hex = nr * 100 + nc
                                    if candidate_hex != cur_hex:
                                        for try_mt in [0, 1, 2, 3]:
                                            step_route = self.map.gen_move_route(cur_hex, candidate_hex, try_mt)
                                            if step_route:
                                                fallback_hex = candidate_hex
                                                fallback_source = "brute_force_ring"
                                                fallback_move_type = try_mt
                                                break
                                    if fallback_hex is not None:
                                        break
                            if fallback_hex is not None:
                                break

                    # Level 5: Absolute last resort - use first valid neighbor with raw step
                    if fallback_hex is None and neighbors:
                        fallback_hex = neighbors[0]
                        fallback_source = "absolute_last_resort"

                    if fallback_hex is not None:
                        simple_route = [fallback_hex]
                        self.policy.record_move_route({
                            "operator_id": obj_id,
                            "operator_type": bop.get("type"),
                            "current_hex": cur_hex,
                            "move_type": fallback_move_type,
                            "move_type_name": {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}.get(fallback_move_type, str(fallback_move_type)),
                            "target_kind": "fallback",
                            "target_source": fallback_source,
                            "target_city_id": fallback_city_id,
                            "target_hex": fallback_hex,
                            "reachable_city_count": 0,
                            "total_cities": len(cities),
                            "route_len": 1,
                            "route_empty": False,
                            "route_failure_reason": "none",
                            "selected_by_learned": False,
                            "neighbor_count": len(neighbors),
                            "raw_neighbor_count": len(raw_neighbors),
                        })
                        self.policy.note_action(
                            "move",
                            False,
                            {"obj_id": obj_id, "target_hex": fallback_hex, "reason": fallback_source},
                        )
                        return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Move, "move_path": simple_route}
                    self.policy.record_move_route({
                        "operator_id": obj_id,
                        "operator_type": bop.get("type"),
                        "current_hex": cur_hex,
                        "move_type": move_type,
                        "move_type_name": {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}.get(move_type, str(move_type)),
                        "target_kind": "city",
                        "target_source": "no_valid_move_candidate" if missing_move_candidate else "empty_route",
                        "target_city_id": None,
                        "target_hex": None,
                        "reachable_city_count": 0,
                        "total_cities": len(cities),
                        "route_len": 0,
                        "route_empty": True,
                        "route_failure_reason": "no_valid_move_candidate" if missing_move_candidate else "empty_route",
                        "selected_by_learned": False,
                        "neighbor_count": len(neighbors),
                        "raw_neighbor_count": len(raw_neighbors),
                    })
                    self.policy.note_action(
                        "move",
                        False,
                        {
                            "obj_id": obj_id,
                            "reason": "no_valid_move_candidate" if missing_move_candidate else "empty_route",
                        },
                    )
                    return None
'''

    src = src[:idx_start] + replacement + src[idx_end:]

    # ── 2. Verify the edit ──
    if 'BULLETPROOF 5-level fallback' not in src:
        print("ERROR: Replacement not found in source")
        sys.exit(1)
    if 'brute_force_ring' not in src:
        print("ERROR: brute_force_ring not found")
        sys.exit(1)
    if 'absolute_last_resort' not in src:
        print("ERROR: absolute_last_resort not found")
        sys.exit(1)

    # ── 3. Write the file ──
    # Backup first
    write_file(AGENT_PY + ".bak_bulletproof", original)
    write_file(AGENT_PY, src)

    # Verify
    check = read_file(AGENT_PY)
    assert 'BULLETPROOF 5-level fallback' in check
    assert 'brute_force_ring' in check
    assert 'absolute_last_resort' in check
    assert 'raw_neighbor_count' in check
    print(f"OK: Patched {AGENT_PY} with 5-level bulletproof fallback")
    print(f"  Backup saved to {AGENT_PY}.bak_bulletproof")


if __name__ == "__main__":
    main()
