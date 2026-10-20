"""Naval movement and adjudication helpers for the joint wargame core.

Synthetic demonstration model, not real performance data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import engagement, hexmap, movement


SEA_MOVE_COST = 1.0
SUBMARINE_SURFACED_COST = 1.5


def move_mode(unit) -> str:
    if getattr(unit, "domain", "") == "sub":
        return "sea" if getattr(unit, "depth", "surface") == "surface" else "sub"
    return "sea"


def sea_speed_kmh(unit) -> float:
    """Effective speed for sea movement.

    A submarine under the surface is slower and a damaged ship is slower.
    """
    base = float(getattr(unit, "speed_kmh", 25.0))
    if getattr(unit, "domain", "") == "sub":
        depth = getattr(unit, "depth", "surface")
        base *= 1.0 if depth == "surface" else 0.7 if depth == "periscope" else 0.55
    if not getattr(unit, "alive", True):
        return 0.0
    return max(0.0, base * float(getattr(unit, "readiness", 0.85)))


def naval_edge_cost(hmap: hexmap.HexMap, a: int, b: int, unit) -> float:
    """Cost in hexes.  Land hexes are impassable for surface ships."""
    if not hmap.valid(b):
        return float("inf")
    cond = hmap.cond(b)
    if getattr(unit, "domain", "") in ("sea", "sub"):
        # In the theatre map, terrain 4 (large river / sea) is water, and the
        # generated campaign map marks sea as terrain 4 too.
        if cond not in (4,):
            return float("inf")
        if getattr(unit, "domain", "") == "sub":
            return SUBMARINE_SURFACED_COST if getattr(unit, "depth", "surface") == "surface" else 1.4
        return SEA_MOVE_COST
    return hmap.move_cost(a, b, "vehicle")


def advance_naval(unit, path, tick_seconds: float, hmap: hexmap.HexMap) -> List[int]:
    """Move a surface ship/submarine along ``path`` for one tick."""
    if not path:
        return []
    # Reuse the generic advance with a nav-specific cost function by temporarily
    # wrapping the map is not worth it; implement the small loop here.
    speed = sea_speed_kmh(unit)
    metres = speed * 1000.0 / 3600.0 * tick_seconds
    budget = max(0.05, metres / max(1.0, hmap.scale_m))
    if (not getattr(unit, "path", None) or unit.path[-1] != path[-1]
            or unit.pos not in unit.path):
        unit.path = list(path)
        unit.path_index = 0
    cur = int(unit.pos)
    idx = max(0, int(getattr(unit, "path_index", 0)))
    entered: List[int] = []
    while idx + 1 < len(unit.path) and budget > 0.0:
        nxt = int(unit.path[idx + 1])
        if hexmap.distance(cur, nxt) != 1:
            break
        cost = naval_edge_cost(hmap, cur, nxt, unit)
        if cost == float("inf") or cost > budget:
            break
        cur = nxt
        idx += 1
        budget -= cost
        entered.append(cur)
    unit.pos = cur
    unit.path_index = idx
    return entered


def naval_path(hmap: hexmap.HexMap, start: int, goal: int, unit,
               max_expansions: int = 6000) -> List[int]:
    """A* over water cells for a surface ship/submarine."""
    import heapq
    if start == goal:
        return [start]
    if not hmap.valid(start) or not hmap.valid(goal):
        return []
    open_heap = [(float(hexmap.distance(start, goal)), 0.0, 0, start)]
    counter = 0
    came: Dict[int, int] = {}
    gscore: Dict[int, float] = {start: 0.0}
    closed = set()
    while open_heap and len(closed) < max_expansions:
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
            cost = naval_edge_cost(hmap, cur, nb, unit)
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


def can_fire_naval(attacker, target, state) -> bool:
    if not getattr(attacker, "alive", True) or not getattr(target, "alive", True):
        return False
    if attacker.color == target.color:
        return False
    if getattr(attacker, "suppressed", False):
        return False
    return engagement.can_engage(attacker, target)


def resolve_naval_fire(attacker, target, state) -> Optional[dict]:
    if not can_fire_naval(attacker, target, state):
        return None
    return engagement.resolve_sea_air(attacker, target, state)
