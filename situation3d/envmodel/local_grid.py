"""Small 3 m local grid planner for route repairs.

The global 30 m field is fast but not footprint-aware. This local planner is
only used around a blocked route window; it evaluates a lightweight exact
obstacle predicate (building, water depth, slope) on a 3 m grid and returns a
short detour path.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sim.entity import Pose, Route


def _grid_bbox(start, goal, margin_m: float) -> Tuple[float, float, float, float]:
    lon0 = min(start[0], goal[0])
    lon1 = max(start[0], goal[0])
    lat0 = min(start[1], goal[1])
    lat1 = max(start[1], goal[1])
    kx = 111320.0 * math.cos(math.radians((lat0 + lat1) / 2.0))
    ky = 110540.0
    dlon = margin_m / kx
    dlat = margin_m / ky
    return (lon0 - dlon, lat0 - dlat, lon1 + dlon, lat1 + dlat)


def _domain(profile) -> str:
    raw = str(getattr(profile, "domain", "land"))
    return {"maritime_surface": "surface",
            "maritime_subsurface": "subsurface"}.get(raw, raw)


def _cell_ok(query, profile, lon: float, lat: float) -> bool:
    env = query.env
    half = min(10.0, max(0.5, float(getattr(profile, "width_m", 1.0) or 1.0) / 2.0))
    kx = 111320.0 * math.cos(math.radians(lat))
    ky = 110540.0
    # Inflate building footprint tests by roughly half the vehicle width.
    for dlon, dlat in ((0.0, 0.0), (half / kx, 0.0),
                       (-half / kx, 0.0), (0.0, half / ky),
                       (0.0, -half / ky)):
        if env.building_at(lon + dlon, lat + dlat) is not None:
            return False
    try:
        is_water = bool(query._is_water(lon, lat))
    except Exception:
        is_water = bool(env.is_water(lon, lat))
    if is_water:
        depth = env.water_depth(lon, lat)
        if depth is None:
            return False
        if _domain(profile) in ("surface", "subsurface"):
            need = float(getattr(profile, "draft_m", 0.0)) + float(
                getattr(profile, "ukc_m", 0.0)
            )
            if depth < need:
                return False
        elif _domain(profile) == "amphibious":
            if depth > float(getattr(profile, "swim_depth_m", 0.0)):
                return False
        elif depth > float(getattr(profile, "wading_depth_m", 0.0)):
            return False
    slope = float(env.slope_deg(lon, lat))
    limit = float(getattr(profile, "max_gradient_deg", 0.0) or 0.0)
    if limit > 0.0 and slope > limit:
        return False
    return True


def plan_local(query, profile, start, goal, res_m: float = 3.0,
               margin_m: float = 40.0, max_cells: int = 40000
               ) -> Optional[Route]:
    """A* on a small 3 m grid; returns a short route or None."""
    lon0, lat0, lon1, lat1 = _grid_bbox(start, goal, margin_m)
    kx = 111320.0 * math.cos(math.radians((lat0 + lat1) / 2.0))
    ky = 110540.0
    width_m = (lon1 - lon0) * kx
    height_m = (lat1 - lat0) * ky
    cols = max(2, int(math.ceil(width_m / res_m)))
    rows = max(2, int(math.ceil(height_m / res_m)))
    if cols * rows > max_cells:
        return None

    def center(c: int, r: int) -> Tuple[float, float]:
        return (lon0 + (c + 0.5) * (lon1 - lon0) / cols,
                lat0 + (r + 0.5) * (lat1 - lat0) / rows)

    def index(lon: float, lat: float) -> Tuple[int, int]:
        c = int((lon - lon0) / max(1e-12, lon1 - lon0) * cols)
        r = int((lat - lat0) / max(1e-12, lat1 - lat0) * rows)
        return max(0, min(cols - 1, c)), max(0, min(rows - 1, r))

    start_c = index(start[0], start[1])
    goal_c = index(goal[0], goal[1])
    cache: Dict[Tuple[int, int], bool] = {}

    def ok(c: int, r: int) -> bool:
        key = (c, r)
        if key not in cache:
            lon, lat = center(c, r)
            cache[key] = _cell_ok(query, profile, lon, lat)
        return cache[key]

    if not ok(*start_c) or not ok(*goal_c):
        return None
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1))
    open_heap = []
    seq = 0
    g_score = {start_c: 0.0}
    came = {start_c: None}
    heapq.heappush(open_heap, (0.0, 0.0, seq, start_c))
    closed = set()
    found = False
    while open_heap:
        _f, g, _seq, cell = heapq.heappop(open_heap)
        if cell in closed:
            continue
        closed.add(cell)
        if cell == goal_c:
            found = True
            break
        for dc, dr in neighbors:
            nc, nr = cell[0] + dc, cell[1] + dr
            if not (0 <= nc < cols and 0 <= nr < rows):
                continue
            if not ok(nc, nr):
                continue
            step = res_m * (math.sqrt(2.0) if dc and dr else 1.0)
            ng = g + step
            ncell = (nc, nr)
            if ng + 1e-9 < g_score.get(ncell, float("inf")):
                g_score[ncell] = ng
                came[ncell] = cell
                seq += 1
                heuristic = math.hypot((nc - goal_c[0]) * res_m,
                                       (nr - goal_c[1]) * res_m)
                heapq.heappush(open_heap, (ng + heuristic, ng, seq, ncell))
    if not found:
        return None
    cells = [goal_c]
    while cells[-1] != start_c:
        parent = came.get(cells[-1])
        if parent is None:
            return None
        cells.append(parent)
    cells.reverse()
    points: List[Pose] = []
    for c, r in cells:
        lon, lat = center(c, r)
        points.append(Pose(lon, lat, query.env.elevation(lon, lat)))
    if len(points) < 2:
        return None
    speed = float(getattr(profile, "max_speed_ms", 5.0) or 5.0)
    return Route(points=points,
                 speed_limits=[speed] * (len(points) - 1),
                 modes=["offroad"] * (len(points) - 1))
