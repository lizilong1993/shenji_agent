#!/usr/bin/env python3
"""Fix empty route rate for Maneuver/Fly operators.

Root cause: When all routing methods fail (A*, BFS, neighbor-walk-to-nearest-city),
the agent gives up on Move. This happens especially when:
- Operator is ON a friendly city (nearest_city == current position)
- Neighbor walk can't find a hex closer to the current (same) city
- A* cost data for Maneuver/Fly modes has sparse/disconnected edges

Fix: Add 3-level fallback:
1. Try next-nearest city (skip current position)
2. Walk toward ANY reachable city (not just nearest)
3. Move to ANY neighbor as exploration step
"""
import re
from pathlib import Path

agent_path = Path("ai/agent.py")
text = agent_path.read_text(encoding="utf-8")

# Find the nearest-city fallback block and replace it with the enhanced version
OLD_BLOCK = '''                if not reachable_city_routes:
                    nearest_city = min(
                        (city for city in cities if city.get("coord") is not None),
                        key=lambda c: self.map.get_distance(bop["cur_hex"], int(c["coord"])),
                        default=None,
                    )
                    fallback_hex = None
                    if nearest_city is not None:
                        target_city_hex = int(nearest_city["coord"])
                        neighbors = self.map.get_neighbors(bop["cur_hex"])
                        best_dist = self.map.get_distance(bop["cur_hex"], target_city_hex)
                        for n_hex in neighbors:
                            if n_hex < 0:
                                continue
                            dist = self.map.get_distance(n_hex, target_city_hex)
                            if dist >= 0 and dist < best_dist:
                                best_dist = dist
                                fallback_hex = n_hex
                    if fallback_hex is not None:
                        simple_route = [fallback_hex]
                        self.policy.record_move_route({
                            "operator_id": obj_id,
                            "operator_type": bop.get("type"),
                            "current_hex": bop.get("cur_hex"),
                            "move_type": move_type,
                            "move_type_name": {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}.get(move_type, str(move_type)),
                            "target_kind": "nearest_city_neighbor",
                            "target_source": "fallback_neighbor",
                            "target_city_id": int(nearest_city["coord"]),
                            "target_hex": fallback_hex,
                            "reachable_city_count": 0,
                            "total_cities": len(cities),
                            "route_len": 1,
                            "route_empty": False,
                            "route_failure_reason": "none",
                            "selected_by_learned": False,
                        })
                        self.policy.note_action(
                            "move",
                            False,
                            {"obj_id": obj_id, "target_hex": fallback_hex, "reason": "fallback_neighbor"},
                        )
                        return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Move, "move_path": simple_route}
                    self.policy.record_move_route({
                        "operator_id": obj_id,
                        "operator_type": bop.get("type"),
                        "current_hex": bop.get("cur_hex"),
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
                    })
                    self.policy.note_action(
                        "move",
                        False,
                        {
                            "obj_id": obj_id,
                            "reason": "no_valid_move_candidate" if missing_move_candidate else "empty_route",
                        },
                    )
                    return None'''

NEW_BLOCK = '''                if not reachable_city_routes:
                    # === 3-level fallback when all routing fails ===
                    cur_hex = int(bop.get("cur_hex", 0))
                    neighbors = self.map.get_neighbors(cur_hex) if self.map else []
                    fallback_hex = None
                    fallback_city_id = None
                    fallback_source = "none"

                    # Level 1: Walk toward nearest city that is NOT current position
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
                            if n_hex < 0:
                                continue
                            d = self.map.get_distance(n_hex, target_hex)
                            if 0 <= d < best_dist:
                                fallback_hex = n_hex
                                fallback_city_id = target_hex
                                fallback_source = "walk_toward_city"
                                break
                        if fallback_hex is not None:
                            break

                    # Level 2: Walk to any neighbor not on a friendly city
                    if fallback_hex is None and neighbors:
                        for n_hex in neighbors:
                            if n_hex < 0:
                                continue
                            on_friendly_city = any(
                                c.get("coord") is not None
                                and int(c["coord"]) == n_hex
                                and int(c.get("flag", -1)) == self.faction
                                for c in cities
                            )
                            if not on_friendly_city:
                                fallback_hex = n_hex
                                fallback_source = "walk_to_non_city"
                                break

                    # Level 3: Walk to ANY neighbor (last resort exploration)
                    if fallback_hex is None and neighbors:
                        for n_hex in neighbors:
                            if n_hex >= 0:
                                fallback_hex = n_hex
                                fallback_source = "walk_any_neighbor"
                                break

                    if fallback_hex is not None:
                        simple_route = [fallback_hex]
                        self.policy.record_move_route({
                            "operator_id": obj_id,
                            "operator_type": bop.get("type"),
                            "current_hex": cur_hex,
                            "move_type": move_type,
                            "move_type_name": {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}.get(move_type, str(move_type)),
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
                    })
                    self.policy.note_action(
                        "move",
                        False,
                        {
                            "obj_id": obj_id,
                            "reason": "no_valid_move_candidate" if missing_move_candidate else "empty_route",
                        },
                    )
                    return None'''

if OLD_BLOCK in text:
    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    agent_path.write_text(text, encoding="utf-8")
    print("OK: Replaced nearest-city fallback with 3-level fallback")
else:
    print("ERROR: Could not find old fallback block")
    # Try to show context
    idx = text.find("if not reachable_city_routes:")
    if idx >= 0:
        # Count occurrences
        count = text.count("if not reachable_city_routes:")
        print(f"Found {count} occurrences of 'if not reachable_city_routes:'")
        # Find the one in the learned path (second occurrence, after BFS fallback)
        first = text.find("if not reachable_city_routes:")
        second = text.find("if not reachable_city_routes:", first + 1)
        if second >= 0:
            print(f"Second occurrence at char {second}")
            print("Context:", text[second:second+200])
