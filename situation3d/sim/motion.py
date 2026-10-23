"""Continuous route integration with kinematic and collision checks (P0.2).

The route provides per-segment speed limits; `mobility.kinematics` turns
those into a braking-aware speed profile and a yaw-rate-limited heading.
Substeps are shorter than the platform footprint for ground units so OBB
checks cannot tunnel, and blocked-by-traffic is reported as ``yield``
(keep the route, wait) while blocked-by-geometry clears the route.
"""

from __future__ import annotations

import math
from typing import Optional

from mobility import kinematics
from mobility.passability import passable
from mobility.profile import profile_for, PlatformProfile
from sim.collision import footprint_samples


def _approach(current: float, target: float, delta: float) -> float:
    if current < target:
        return min(target, current + delta)
    return max(target, current - delta)


def entity_profile(entity) -> PlatformProfile:
    prof = getattr(entity, "profile", None)
    if prof is not None:
        return prof
    return profile_for(entity.unit_type, entity.domain)


def _footprint_ok(nav, prof, lon: float, lat: float, heading: float,
                  env) -> bool:
    """Footprint samples against the pack authority (base map outside).

    Building-only raster blocks are enforced by the centre-cell check (cell
    granularity); water and other impassable samples reject the footprint.
    """
    for slon, slat in footprint_samples(prof, lon, lat, heading):
        inside = nav.passable(prof, slon, slat)
        if inside is None:
            if not passable(prof, slon, slat, env, env.elevation(slon, slat)):
                return False
            continue
        if inside:
            continue
        if nav.building_only_block(prof, slon, slat):
            continue
        return False
    return True


def integrate(entity, dt_s: float, env, cell_check=None, nav=None,
              traffic_check=None) -> Optional[str]:
    """Advance one entity by ``dt_s``.

    Returns ``"blocked"`` for geometry (route cleared) or ``"yield"`` for a
    traffic conflict (route kept, unit waits).
    """
    prof = entity_profile(entity)
    if not entity.alive:
        return None
    if entity.route is None:
        entity.velocity_ms = _approach(entity.velocity_ms, 0.0,
                                       prof.decel_ms2 * dt_s)
        if entity.velocity_ms <= 0.01:
            entity.velocity_ms = 0.0
            entity.moving = False
        return None

    route = entity.route
    caps = kinematics.speed_caps(route, prof)
    ground = entity.domain in ("land", "amphibious")
    nav_domain = nav is not None and ground

    p = route.pose_at(entity.route_s)
    nav_here = nav_domain and nav.covers(p.lon, p.lat)
    escaping = (nav_here and not nav.motion_point_ok(prof, p.lon, p.lat))
    is_air = entity.domain == "air"

    v_target = kinematics.cap_at(route, caps, entity.route_s, prof.decel_ms2)
    # Discrete guard: the speed at the end of this tick must be brakeable
    # for the envelope there (avoids cap overshoot at large dt).
    look_s = min(route.total_m,
                 entity.route_s + max(0.0, entity.velocity_ms) * dt_s)
    look_cap = kinematics.cap_at(route, caps, look_s, prof.decel_ms2)
    v_target = min(v_target, math.sqrt(
        look_cap * look_cap
        + 2.0 * prof.decel_ms2 * max(0.0, entity.velocity_ms) * dt_s))
    # Reserve the distance covered during the final decel tick: without it a
    # coarse dt lands on the route end while still >1 m/s, forcing a stop
    # spike.  Braking one tick earlier makes the terminal residual tiny.
    remaining_end = max(0.0, route.total_m - entity.route_s)
    stop_margin = 0.5 * max(0.0, entity.velocity_ms) * dt_s
    v_target = min(v_target, math.sqrt(max(
        0.0, 2.0 * prof.decel_ms2 * (remaining_end - stop_margin))))
    if escaping:
        v_target = max(v_target, prof.max_speed_ms * 0.2)
    if is_air:
        # Vertical dynamics (P2): climb/descent rate limited, not a pose jump.
        if entity.route_s <= 1e-6 and entity.pose.alt_m < p.alt_m:
            entity.pose.alt_m = p.alt_m  # airborne at the planned altitude
        target_alt = p.alt_m
        error = target_alt - entity.pose.alt_m
        if abs(error) < 0.2:
            desired_rate = 0.0
        elif error > 0.0:
            desired_rate = prof.climb_rate_ms
        else:
            desired_rate = -prof.descent_rate_ms
        entity.vertical_ms = _approach(entity.vertical_ms, desired_rate,
                                       prof.accel_ms2 * dt_s)
        delta = entity.vertical_ms * dt_s
        if error > 0.0:
            entity.pose.alt_m = min(target_alt,
                                    entity.pose.alt_m + max(0.0, delta))
        elif error < 0.0:
            entity.pose.alt_m = max(target_alt,
                                    entity.pose.alt_m + min(0.0, delta))
        if entity.vertical_ms > 0.2:
            v_target *= max(0.6, 1.0 - 0.25 * entity.vertical_ms
                            / max(0.1, prof.climb_rate_ms))
    if entity.domain == "surface":
        try:
            sea_state = int(getattr(env.weather(p.lon, p.lat), "sea_state", 0)
                            or 0)
        except Exception:
            sea_state = 0
        v_target *= max(0.5, 1.0 - 0.04 * sea_state)
    try:
        raw_mobility = float((entity.components or {}).get("mobility", 1.0))
    except (AttributeError, TypeError, ValueError):
        raw_mobility = 1.0
    mobility = 0.0 if raw_mobility <= 0.05 else max(0.2, raw_mobility)
    v_target *= mobility
    try:
        pace = max(0.3, min(1.0, float(getattr(entity, "pace_scale", 1.0))))
    except (TypeError, ValueError):
        pace = 1.0
    # Pace is a fraction of the allowed cap: it may slow the unit but never
    # overrides terrain/curvature/braking limits.
    v_target = min(v_target, v_target * pace)
    rate = prof.accel_ms2 if v_target > entity.velocity_ms else prof.decel_ms2
    entity.velocity_ms = _approach(entity.velocity_ms, v_target, rate * dt_s)
    entity.moving = entity.velocity_ms > 0.05
    if entity.moving:
        entity.concealed = False

    ds = entity.velocity_ms * dt_s
    substep_m = 50.0
    if ground:
        substep_m = min(substep_m, max(2.0, prof.length_m))
    if nav_domain:
        substep_m = min(substep_m, max(5.0, nav.res_m * 0.7))
    n = max(1, int(math.ceil(ds / substep_m))) if ds > 0 else 1
    dt_seg = dt_s / float(n)

    heading = entity.pose.heading_deg   # last committed heading
    reason = None
    for _ in range(n):
        s_next = min(route.total_m, entity.route_s + ds / n)
        p_next = route.pose_at(s_next)
        proposed = kinematics.turn_towards(
            heading, p_next.heading_deg,
            math.degrees(kinematics.max_yaw_rate(prof, entity.velocity_ms)) * dt_seg)

        blocked = False
        if nav_domain and nav.covers(p_next.lon, p_next.lat):
            if not escaping:
                blocked = not nav.motion_point_ok(prof, p_next.lon, p_next.lat)
                if not blocked:
                    blocked = not _footprint_ok(nav, prof, p_next.lon,
                                                p_next.lat, proposed, env)
        else:
            # Air routes are planned at terrain+AGL; the actual altitude lags
            # for presentation/energy but does not hard-block the sortie.
            blocked = not passable(prof, p_next.lon, p_next.lat, env,
                                   p_next.alt_m)
        if not blocked and cell_check is not None:
            blocked = not cell_check(p_next.lon, p_next.lat)
        if blocked:
            keep_alt = entity.pose.alt_m if is_air else None
            entity.pose = route.pose_at(entity.route_s)
            if is_air:
                entity.pose.alt_m = keep_alt
            entity.pose.heading_deg = heading
            entity.velocity_ms = 0.0
            entity.moving = False
            entity.route = None
            reason = "blocked"
            break
        if (traffic_check is not None
                and not traffic_check(p_next.lon, p_next.lat, proposed)):
            # Rejected move: keep the last committed heading (an un-checked
            # in-place rotation could itself cause an overlap).
            keep_alt = entity.pose.alt_m if is_air else None
            entity.pose = route.pose_at(entity.route_s)
            if is_air:
                entity.pose.alt_m = keep_alt
            entity.pose.heading_deg = heading
            entity.velocity_ms = 0.0
            entity.moving = False
            reason = "yield"
            break
        entity.route_s = s_next
        heading = proposed
        if nav_domain and nav.covers(p_next.lon, p_next.lat):
            escaping = not nav.motion_point_ok(prof, p_next.lon, p_next.lat)
        if entity.route_s >= route.total_m:
            break

    if entity.route is not None:
        remaining = route.total_m - entity.route_s
        if reason is None and (remaining <= 1e-6
                               or (remaining <= 0.5
                                   and entity.velocity_ms <= 1.0)):
            keep_alt = entity.pose.alt_m if is_air else None
            entity.pose = route.pose_at(route.total_m)
            if is_air:
                entity.pose.alt_m = keep_alt
            entity.pose.heading_deg = heading
            entity.route = None
            entity.route_s = 0.0
            entity.velocity_ms = 0.0
            entity.moving = False
            entity.pace_scale = 1.0
        else:
            keep_alt = entity.pose.alt_m if is_air else None
            entity.pose = route.pose_at(entity.route_s)
            if is_air:
                entity.pose.alt_m = keep_alt
            entity.pose.heading_deg = heading
    return reason
