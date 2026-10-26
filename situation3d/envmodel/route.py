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

from .local_grid import plan_local


def _footprint_clear(query: Any, profile: Any, lon: float, lat: float) -> bool:
    """Half-width building clearance, same margin the local planner uses."""
    if str(getattr(profile, "domain", "land")) not in ("land", "amphibious"):
        return True
    env = getattr(query, "env", None)
    footprint = getattr(env, "building_footprint_at", None)
    if footprint is None:
        return True
    half = min(10.0, max(0.5, float(getattr(profile, "width_m", 1.0) or 1.0) / 2.0))
    kx = 111320.0 * math.cos(math.radians(lat))
    ky = 110540.0
    dx, dy = half / kx, half / ky
    for dlon, dlat in ((0.0, 0.0), (dx, 0.0), (-dx, 0.0), (0.0, dy),
                       (0.0, -dy), (dx, dy), (dx, -dy), (-dx, dy), (-dx, -dy)):
        if footprint(lon + dlon, lat + dlat) is not None:
            return False
    return True


def validate_route(query: Any, profile: Any, route: Route,
                   step_m: float = 3.0) -> bool:
    """Sample a route and require every point to pass ``passability``."""
    total = float(route.total_m)
    s = 0.0
    while s <= total:
        pose = route.pose_at(s)
        if not query.passability(profile, pose.lon, pose.lat).value["allowed"]:
            return False
        if not _footprint_clear(query, profile, pose.lon, pose.lat):
            return False
        s += step_m
    if total > 0.0:
        pose = route.pose_at(total)
        if not query.passability(profile, pose.lon, pose.lat).value["allowed"]:
            return False
        if not _footprint_clear(query, profile, pose.lon, pose.lat):
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
        if not _footprint_clear(query, profile, pose.lon, pose.lat):
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
                 step_m: float = 3.0,
                 offsets: Sequence[float] = (10.0, 30.0)
                 ) -> Optional[Route]:
    """Try shifting the first blocked route vertex sideways.

    Candidate validation uses the same fine step (3 m by default) as the final
    conformance check, so thin water or footprint strips cannot slip through.
    """
    blocked_s = _first_blocked_s(query, profile, route,
                                 step_m=min(5.0, step_m))
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

    # A single point shift is not enough for a block spanning several cells:
    # plan a short 3 m detour around the blocked window and splice it in.
    total = float(route.total_m)
    s0 = max(0.0, blocked_s - 120.0)
    s1 = min(total, blocked_s + 120.0)
    a = route.pose_at(s0)
    b = route.pose_at(s1)
    local = plan_local(query, profile, (a.lon, a.lat), (b.lon, b.lat),
                       res_m=4.0, margin_m=50.0)
    if local is not None:
        i0 = _nearest_index(route, s0)
        i1 = _nearest_index(route, s1)
        if i0 < i1:
            points = list(route.points[:i0 + 1]) + list(local.points) + list(route.points[i1:])
            speed = float(getattr(profile, "max_speed_ms", 5.0) or 5.0)
            combined = Route(
                points=points,
                speed_limits=[speed] * (len(points) - 1),
                modes=["offroad"] * (len(points) - 1),
            )
            if validate_route(query, profile, combined, step_m=step_m):
                return combined
    return None


def plan_valid_route(nav: Any, query: Any, profile: Any, start, goal,
                     blocked=None, step_m: float = 3.0) -> Optional[Route]:
    """Plan with the 30 m field, then validate and try a local point repair."""
    route = nav.plan(profile, start, goal, blocked=blocked,
                     safe_fallback=True)
    if route is None:
        return None
    if validate_route(query, profile, route, step_m=step_m):
        return route
    repaired = repair_route(query, profile, route, step_m=step_m)
    if repaired is None:
        return None
    return repaired if validate_route(query, profile, repaired,
                                      step_m=step_m) else None
