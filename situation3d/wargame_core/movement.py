"""Movement and pathfinding for the clean-room land wargame core."""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

from . import hexmap


def find_path(hmap: hexmap.HexMap, start: int, goal: int,
              mode: str = "vehicle",
              max_expansions: int = 50000) -> List[int]:
    """A* over the hex grid.  Returns a list of hexes, or ``[]`` if unreachable."""
    if start == goal:
        return [start]
    if not hmap.valid(start) or not hmap.valid(goal):
        return []
    if not hmap.passable(goal, mode):
        return []

    open_heap: List[Tuple[float, float, int, int]] = []
    counter = 0
    heapq.heappush(open_heap, (float(hexmap.distance(start, goal)), 0.0, counter, start))
    came: Dict[int, int] = {}
    gscore: Dict[int, float] = {start: 0.0}
    closed = set()
    expansions = 0

    while open_heap and expansions < max_expansions:
        _f, g, _n, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expansions += 1
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
            step = hmap.move_cost(cur, nb, mode)
            if step == float("inf"):
                continue
            tentative = g + step
            if tentative < gscore.get(nb, float("inf")):
                gscore[nb] = tentative
                came[nb] = cur
                counter += 1
                f = tentative + float(hexmap.distance(nb, goal))
                heapq.heappush(open_heap, (f, tentative, counter, nb))
    return []


def hex_budget(unit, tick_seconds: float, hmap: hexmap.HexMap) -> float:
    """How many normal-speed hexes the unit can cover in one tick."""
    speed_kmh = max(0.1, float(getattr(unit, "speed_kmh", 30.0)))
    metres = speed_kmh * 1000.0 / 3600.0 * tick_seconds
    return max(0.05, metres / max(1.0, hmap.scale_m))


def advance(unit, path: Sequence[int], tick_seconds: float,
            hmap: hexmap.HexMap, mode: str = "vehicle",
            can_enter=None) -> List[int]:
    """Move ``unit`` along ``path`` for one tick.  Returns the hexes entered.

    ``can_enter(pos)`` is an optional stacking/passability callback; returning
    ``False`` stops the unit before that hex.
    """
    if not path:
        return []
    if (not getattr(unit, "path", None)
            or unit.path[-1] != path[-1]
            or unit.pos not in unit.path):
        unit.path = list(path)
        unit.path_index = 0
    entered: List[int] = []
    budget = hex_budget(unit, tick_seconds, hmap)
    idx = max(0, int(getattr(unit, "path_index", 0)))
    cur = int(unit.pos)
    while idx + 1 < len(unit.path) and budget > 0.0:
        nxt = int(unit.path[idx + 1])
        if hexmap.distance(cur, nxt) != 1:
            # Path is stale or malformed; stop rather than teleport.
            break
        if can_enter is not None and not can_enter(nxt):
            break
        cost = hmap.move_cost(cur, nxt, mode)
        if cost == float("inf") or cost > budget:
            break
        cur = nxt
        idx += 1
        budget -= cost
        entered.append(cur)
    unit.pos = cur
    unit.path_index = idx
    return entered


def path_length_hex(path: Sequence[int]) -> int:
    return max(0, len(path) - 1)
