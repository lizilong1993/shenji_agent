"""Grid A* planner producing continuous ``Route`` polylines (P3).

Planning happens on the 200 m rule grid; the resulting polyline is consumed
by ``sim.motion`` which integrates along it with 50 m substeps and re-checks
passability, so a coarse plan cannot tunnel through a thin obstacle.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

from geo.geodesy import haversine_m
from mobility.passability import effective_speed_ms, passable
from mobility.profile import PlatformProfile
from sim.entity import Pose, Route

DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]


class GridPlanner(object):
    def __init__(self, env, grid, max_nodes: int = 200000):
        self.env = env
        self.grid = grid
        self.max_nodes = int(max_nodes)

    def _alt(self, profile: PlatformProfile, lon: float, lat: float) -> float:
        return self.env.elevation(lon, lat) + (profile.agl_m if profile.domain == "air" else 0.0)

    def plan(self, prof: PlatformProfile, start: Tuple[float, float],
             goal: Tuple[float, float],
             mode: Optional[str] = None) -> Optional[Route]:
        start_cell = self.grid.cell_of(start[0], start[1])
        goal_cell = self.grid.cell_of(goal[0], goal[1])
        if not self.grid.in_bounds(start_cell) or not self.grid.in_bounds(goal_cell):
            return None
        if not passable(prof, goal[0], goal[1], self.env,
                        self._alt(prof, goal[0], goal[1])):
            return None

        open_heap = []
        g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        came: Dict[Tuple[int, int], Tuple[int, int]] = {}
        seq = 0
        start_ll = self.grid.cell_center_lonlat(*start_cell)
        goal_ll = self.grid.cell_center_lonlat(*goal_cell)
        heapq.heappush(open_heap, (haversine_m(start_ll, goal_ll) / max(0.1, prof.max_speed_ms),
                                   0.0, seq, start_cell))
        closed = set()
        expanded = 0

        while open_heap and expanded < self.max_nodes:
            _, g, _, cell = heapq.heappop(open_heap)
            if cell in closed:
                continue
            closed.add(cell)
            expanded += 1
            if cell == goal_cell:
                return self._build_route(prof, start, goal, came, cell, mode)
            row, col = cell
            cx, cy = self.grid.cell_center_lonlat(row, col)
            for dr, dc in DIRECTIONS:
                nb = (row + dr, col + dc)
                if not self.grid.in_bounds(nb) or nb in closed:
                    continue
                nlon, nlat = self.grid.cell_center_lonlat(*nb)
                alt = self._alt(prof, nlon, nlat)
                if not passable(prof, nlon, nlat, self.env, alt):
                    continue
                step_m = haversine_m((cx, cy), (nlon, nlat))
                speed = effective_speed_ms(prof, nlon, nlat, self.env, alt)
                if speed <= 0.0:
                    continue
                if prof.road_bound and not self.env.on_road(nlon, nlat):
                    continue
                tent = g + step_m / speed
                if tent < g_score.get(nb, float("inf")) - 1e-9:
                    g_score[nb] = tent
                    came[nb] = cell
                    h = haversine_m((nlon, nlat), goal_ll) / max(0.1, prof.max_speed_ms)
                    seq += 1
                    heapq.heappush(open_heap, (tent + h, tent, seq, nb))
        return None

    def _build_route(self, prof: PlatformProfile, start, goal, came, end_cell, mode):
        cells = [end_cell]
        while cells[-1] in came:
            cells.append(came[cells[-1]])
        cells.reverse()
        points: List[Pose] = []
        # Start exactly where the entity is, then cell centres, then goal.
        points.append(Pose(start[0], start[1], self.env.elevation(*start) +
                           (prof.agl_m if prof.domain == "air" else 0.0)))
        for (row, col) in cells[1:]:
            lon, lat = self.grid.cell_center_lonlat(row, col)
            points.append(Pose(lon, lat, self.env.elevation(lon, lat) +
                               (prof.agl_m if prof.domain == "air" else 0.0)))
        points.append(Pose(goal[0], goal[1], self.env.elevation(*goal) +
                           (prof.agl_m if prof.domain == "air" else 0.0)))
        # Remove duplicate consecutive points.
        dedup = [points[0]]
        for p in points[1:]:
            if haversine_m(dedup[-1].as_tuple(), p.as_tuple()) > 0.5:
                dedup.append(p)
        if len(dedup) < 2:
            dedup.append(Pose(goal[0] + 1e-7, goal[1], points[-1].alt_m))
        speeds = []
        for i in range(1, len(dedup)):
            a, b = dedup[i - 1], dedup[i]
            mid = ((a.lon + b.lon) / 2.0, (a.lat + b.lat) / 2.0)
            speeds.append(max(0.1, effective_speed_ms(
                prof, mid[0], mid[1], self.env,
                self._alt(prof, mid[0], mid[1]))))
        route_mode = mode or {"land": "offroad", "amphibious": "offroad",
                              "surface": "water", "subsurface": "water",
                              "air": "air"}.get(prof.domain, "offroad")
        return Route(points=dedup, speed_limits=speeds,
                     modes=[route_mode] * (len(dedup) - 1))
