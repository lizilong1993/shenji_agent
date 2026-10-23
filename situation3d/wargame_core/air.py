"""Air movement and adjudication helpers for the joint wargame core.

Synthetic demonstration model.  Air units move in the same hex space as the
other domains; ``altitude_band`` drives detection and which weapons may engage.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import engagement, hexmap, sensors

# Relative speed by altitude band (ground speed is unaffected by band in this
# MVP; the values are here to make low-level ingress slower).
BAND_SPEED = {"超低空": 0.75, "低空": 0.9, "中空": 1.0, "高空": 1.05}
BAND_ALT_M = {"超低空": 100.0, "低空": 1000.0, "中空": 5000.0, "高空": 10000.0}


def air_speed_kmh(unit) -> float:
    base = float(getattr(unit, "speed_kmh", 800.0))
    band = getattr(unit, "altitude_band", "中空")
    base *= BAND_SPEED.get(band, 1.0)
    if getattr(unit, "fuel", 1.0) <= 0.0:
        return 0.0
    return max(0.0, base * float(getattr(unit, "readiness", 0.9)))


def fuel_per_tick(unit, tick_seconds: float) -> float:
    """Fraction of full fuel consumed per tick (synthetic endurance)."""
    endurance_h = float(getattr(unit, "raw", {}).get("endurance_h", 6.0))
    if endurance_h <= 0:
        return 0.0
    return tick_seconds / 3600.0 / endurance_h


def climb(unit, band: str) -> bool:
    """Change altitude band.  One step at a time, no fuel check here."""
    order = ["超低空", "低空", "中空", "高空"]
    if band not in order:
        return False
    cur = getattr(unit, "altitude_band", "中空")
    if cur not in order:
        unit.altitude_band = band
        return True
    if abs(order.index(band) - order.index(cur)) > 1:
        return False
    unit.altitude_band = band
    return True


def air_path(hmap: hexmap.HexMap, start: int, goal: int) -> List[int]:
    """Air units ignore terrain; straight-ish A* over all cells."""
    return _a_star(hmap, start, goal, lambda a, b: 1.0)


def _a_star(hmap, start, goal, edge_cost):
    import heapq
    if start == goal:
        return [start]
    if not hmap.valid(start) or not hmap.valid(goal):
        return []
    open_heap = [(float(hexmap.distance(start, goal)), 0.0, 0, start)]
    counter = 0
    came: Dict[int, int] = {}
    gscore = {start: 0.0}
    closed = set()
    while open_heap:
        _f, g, _n, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        for nb in hmap.neighbors(cur):
            if nb < 0 or nb in closed:
                continue
            cost = edge_cost(cur, nb)
            if cost == float("inf"):
                continue
            tentative = g + cost
            if tentative < gscore.get(nb, float("inf")):
                gscore[nb] = tentative
                came[nb] = cur
                counter += 1
                heapq.heappush(open_heap,
                               (tentative + float(hexmap.distance(nb, goal)),
                                tentative, counter, nb))
    return []


def advance_air(unit, path, tick_seconds: float, hmap: hexmap.HexMap) -> List[int]:
    if not path:
        return []
    speed = air_speed_kmh(unit)
    metres = speed * 1000.0 / 3600.0 * tick_seconds
    budget = max(0.05, metres / max(1.0, hmap.scale_m))
    if (not getattr(unit, "path", None) or unit.path[-1] != path[-1]
            or unit.pos not in unit.path):
        unit.path = list(path)
        unit.path_index = 0
    budget += float(getattr(unit, "move_progress", 0.0) or 0.0)
    cur = int(unit.pos)
    idx = max(0, int(getattr(unit, "path_index", 0)))
    entered: List[int] = []
    while idx + 1 < len(unit.path) and budget > 0.0:
        nxt = int(unit.path[idx + 1])
        if hexmap.distance(cur, nxt) != 1:
            break
        cur = nxt
        idx += 1
        budget -= 1.0
        entered.append(cur)
    unit.pos = cur
    unit.path_index = idx
    unit.move_progress = max(0.0, budget) if idx + 1 < len(unit.path) else 0.0
    return entered


def can_fire_air(attacker, target, state) -> bool:
    if not getattr(attacker, "alive", True) or not getattr(target, "alive", True):
        return False
    if attacker.color == target.color:
        return False
    if getattr(attacker, "suppressed", False):
        return False
    if getattr(attacker, "fuel", 1.0) <= 0.0:
        return False
    return engagement.can_engage(attacker, target)


def resolve_air_fire(attacker, target, state) -> Optional[dict]:
    if not can_fire_air(attacker, target, state):
        return None
    return engagement.resolve_sea_air(attacker, target, state)


def update_fuel(unit, tick_seconds: float) -> None:
    if getattr(unit, "domain", "") == "air":
        unit.fuel = max(0.0, unit.fuel - fuel_per_tick(unit, tick_seconds))
