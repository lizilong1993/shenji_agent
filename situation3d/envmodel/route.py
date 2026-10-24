"""Route post-validation and small local point repairs.

The 30 m A* mask is not footprint-aware: a smoothed route can clip a building
or a water polygon between cell centres. This module validates routes against
the exact gate contract and tries small perpendicular point shifts when a
route fails. It never returns a route that fails validation.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence

from geo.geodesy import initial_bearing_deg
from sim.entity import Pose, Route


def validate_route(query: Any, profile: Any, route: Route,
                   step_m: float = 10.0) -> bool:
    """Sample a route and require every point to pass ``passability``."""
    total = float(route.total_m)
    s = 0.0
    while s <= total:
        pose = route.pose_at(s)
        if not query.passability(profile, pose.lon, pose.lat).value["allowed"]:
            return False
        s += step_m
    if total > 0.0:
        pose = route.pose_at(total)
        if not query.passability(profile, pose.lon, pose.lat).value["allowed"]:
            return False
    return True


def _first_blocked_s(query: Any, profile: Any, route: Route,
                     step_m: float = 5.0) -> Optional[float]:
    total = float(route.total_m)
    s = 0.0
    while s <= total:
        pose = route.pose_at(s)
        if not query.passability(profile, pose.lon, pose.lat).value["allowed"]:
            return s
        s += step_m
    return None


def _nearest_index(route: Route, s: float) -> int:
    best = 0
    best_d = None
    acc = 0.0
    for i in range(len(route.points)):
        if i:
            a, b = route.points[i - 1], route.points[i]
            acc += math.hypot((b.lon - a.lon) * 111320.0
                              * math.cos(math.radians(a.lat)),
                              (b.lat - a.lat) * 110540.0)
        d = abs(acc - s)
        if best_d is None or d < best_d:
            best = i
            best_d = d
    return best


def _detour_point(route: Route, idx: int, offset_m: float) -> Optional[Pose]:
    if idx <= 0 or idx >= len(route.points) - 1:
        return None
    a, b = route.points[idx - 1], route.points[idx + 1]
    bearing = initial_bearing_deg((a.lon, a.lat), (b.lon, b.lat))
    angle = math.radians(bearing + 90.0)
    kx = 111320.0 * math.cos(math.radians(route.points[idx].lat))
    ky = 110540.0
    p = route.points[idx]
    return Pose(
        p.lon + offset_m * math.sin(angle) / kx,
        p.lat + offset_m * math.cos(angle) / ky,
        p.alt_m,
    )


def repair_route(query: Any, profile: Any, route: Route,
                 step_m: float = 15.0,
                 offsets: Sequence[float] = (10.0, 30.0)
                 ) -> Optional[Route]:
    """Try shifting the first blocked route vertex sideways.

    Candidate validation uses a coarser step (15 m) than the final conformance
    check; the caller should still re-check with its own step.
    """
    blocked_s = _first_blocked_s(query, profile, route, step_m=5.0)
    if blocked_s is None:
        return route
    idx = _nearest_index(route, blocked_s)
    for offset in offsets:
        for sign in (-1.0, 1.0):
            p = _detour_point(route, idx, sign * offset)
            if p is None:
                continue
            if not query.passability(profile, p.lon, p.lat).value["allowed"]:
                continue
            points: List[Pose] = list(route.points)
            points[idx] = p
            candidate = Route(
                points=points,
                speed_limits=list(route.speed_limits),
                modes=list(route.modes),
            )
            if validate_route(query, profile, candidate, step_m=step_m):
                return candidate
    return None


def plan_valid_route(nav: Any, query: Any, profile: Any, start, goal,
                     blocked=None, step_m: float = 10.0) -> Optional[Route]:
    """Plan with the 30 m field, then validate and try a local point repair."""
    route = nav.plan(profile, start, goal, blocked=blocked,
                     safe_fallback=True)
    if route is None:
        return None
    if validate_route(query, profile, route, step_m=step_m):
        return route
    return repair_route(query, profile, route, step_m=step_m)
