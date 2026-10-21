"""Continuous route integration with substep collision checks (P3).

Positions advance by arc length, never by fractional hex progress.  Fast
entities take 50 m substeps so a 300 m/s aircraft cannot tunnel through a
thin obstacle between ticks.
"""

from __future__ import annotations

import math
from typing import Optional

from mobility.passability import passable, speed_factor
from mobility.profile import profile_for, PlatformProfile


def _approach(current: float, target: float, delta: float) -> float:
    if current < target:
        return min(target, current + delta)
    return max(target, current - delta)


def entity_profile(entity) -> PlatformProfile:
    prof = getattr(entity, "profile", None)
    if prof is not None:
        return prof
    return profile_for(entity.unit_type, entity.domain)


def integrate(entity, dt_s: float, env, cell_check=None) -> Optional[str]:
    """Advance one entity by ``dt_s``.  Returns a reason string when blocked."""
    prof = entity_profile(entity)
    if not entity.alive:
        return None
    if entity.route is None:
        entity.velocity_ms = _approach(entity.velocity_ms, 0.0, prof.decel_ms2 * dt_s)
        if entity.velocity_ms <= 0.01:
            entity.velocity_ms = 0.0
            entity.moving = False
        return None

    p = entity.route.pose_at(entity.route_s)
    v_target = entity.route.speed_at(entity.route_s)
    v_target *= speed_factor(prof, p.lon, p.lat, env, p.alt_m)
    entity.velocity_ms = _approach(entity.velocity_ms, v_target,
                                   prof.accel_ms2 * dt_s)
    entity.moving = entity.velocity_ms > 0.05
    if entity.moving:
        entity.concealed = False

    ds = entity.velocity_ms * dt_s
    n = max(1, int(math.ceil(ds / 50.0))) if ds > 0 else 1
    reason = None
    for _ in range(n):
        s_next = min(entity.route.total_m, entity.route_s + ds / n)
        p_next = entity.route.pose_at(s_next)
        blocked = not passable(prof, p_next.lon, p_next.lat, env, p_next.alt_m)
        if not blocked and cell_check is not None:
            blocked = not cell_check(p_next.lon, p_next.lat)
        if blocked:
            entity.pose = entity.route.pose_at(entity.route_s)
            entity.velocity_ms = 0.0
            entity.moving = False
            entity.route = None
            reason = "blocked"
            break
        entity.route_s = s_next
        if entity.route_s >= entity.route.total_m:
            break

    if entity.route is not None:
        entity.pose = entity.route.pose_at(entity.route_s)
        if entity.route_s >= entity.route.total_m - 1e-6:
            entity.route = None
            entity.route_s = 0.0
            entity.velocity_ms = 0.0
            entity.moving = False
    return reason
