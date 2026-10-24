"""Route post-validation against the exact gate contract.

The 30 m A* mask is not footprint-aware: a smoothed route can clip a building
or a water polygon between cell centres. This module is the safety layer that
rejects such routes before they are accepted by a caller.
"""

from __future__ import annotations

from typing import Any, Optional

from sim.entity import Route


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


def plan_valid_route(nav: Any, query: Any, profile: Any, start, goal,
                     blocked=None, step_m: float = 10.0) -> Optional[Route]:
    """Plan with the 30 m field, then reject routes that fail exact gates."""
    route = nav.plan(profile, start, goal, blocked=blocked,
                     safe_fallback=True)
    if route is None:
        return None
    if not validate_route(query, profile, route, step_m=step_m):
        return None
    return route
