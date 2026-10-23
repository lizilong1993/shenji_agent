"""Kinematic constraints (P0.2): yaw rate, corner speed and braking.

Motion is generated from a speed profile instead of instant heading changes:

* every route vertex gets a speed cap from path curvature -- the vehicle
  needs a yaw rate ``v * kappa`` no larger than ``turn_rate_dps``;
* a backward pass limits approach speed so the platform can brake for any
  upcoming cap (corner or route end) within ``decel_ms2``;
* heading chases the route tangent at no more than ``max_yaw_rate``; wheeled
  platforms are limited by their minimum turn radius and stop-and-turn
  (modelled multi-point) at low speed instead of pivoting on the spot.
"""

from __future__ import annotations

import bisect
import math

from geo.geodesy import initial_bearing_deg
from mobility.profile import PlatformProfile


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def turn_towards(current: float, target: float, max_delta: float) -> float:
    delta = wrap180(target - current)
    if abs(delta) <= max_delta:
        return target % 360.0
    return (current + math.copysign(max_delta, delta)) % 360.0


def max_yaw_rate(prof: PlatformProfile, speed_ms: float) -> float:
    """Maximum heading rate (rad/s) at the given speed."""
    base = math.radians(max(0.0, prof.turn_rate_dps))
    if prof.pivot_turn:
        return base
    if prof.min_turn_radius_m > 0.0:
        # Low speed: modelled multi-point turn with wheels turning.
        return max(base * 0.25,
                   min(base, max(0.0, speed_ms) / prof.min_turn_radius_m))
    return base


def speed_caps(route, prof: PlatformProfile):
    """Per-vertex speed caps (m/s) with braking-distance propagation."""
    limits = tuple(float(v) for v in (route.speed_limits or ()))
    key = (prof.key, len(route.points), round(route.cum_m[-1], 6), limits)
    cached = getattr(route, "_caps_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    n = len(route.points)
    caps = [0.0] * n
    if n >= 2:
        for i in range(n - 1):
            lim = (limits[min(i, len(limits) - 1)] if limits
                   else prof.max_speed_ms)
            caps[i] = max(0.0, min(lim, prof.max_speed_ms))
    omega = math.radians(max(0.0, prof.turn_rate_dps))
    for i in range(1, n - 1):
        prev, cur, nxt = route.points[i - 1], route.points[i], route.points[i + 1]
        span = min(route.cum_m[i] - route.cum_m[i - 1],
                   route.cum_m[i + 1] - route.cum_m[i])
        if span <= 1e-6:
            continue  # zero-length legs cannot define a real corner
        h_in = initial_bearing_deg((prev.lon, prev.lat), (cur.lon, cur.lat))
        h_out = initial_bearing_deg((cur.lon, cur.lat), (nxt.lon, nxt.lat))
        dpsi = abs(wrap180(h_out - h_in))
        if dpsi < 1e-3 or omega <= 0.0:
            continue
        kappa = math.radians(dpsi) / max(5.0, span)
        if kappa <= 0.0:
            continue
        # Fixed point: the yaw rate itself depends on speed for non-pivot
        # platforms (v/R), so solve v = max_yaw_rate(v) / kappa.
        v = omega / kappa
        for _ in range(4):
            v_next = max_yaw_rate(prof, v) / kappa
            if abs(v_next - v) < 0.05:
                v = v_next
                break
            v = v_next
        caps[i] = min(caps[i], v)
    caps[n - 1] = 0.0  # stop at the end of the route
    for i in range(n - 2, -1, -1):
        seg = max(1e-6, route.cum_m[i + 1] - route.cum_m[i])
        caps[i] = min(caps[i], math.sqrt(
            caps[i + 1] * caps[i + 1] + 2.0 * prof.decel_ms2 * seg))
    route._caps_cache = (key, caps)
    return caps


def cap_at(route, caps, s: float, decel_ms2: float = 2.0) -> float:
    """Speed cap at arc length ``s`` using the braking envelope.

    Between vertices the envelope is ``min(cap[i], sqrt(cap[i+1]^2 +
    2*decel*(cum[i+1]-s)))``; linear chord interpolation would sit well
    below it and stretch braking over three times the distance.
    """
    if not caps or len(caps) != len(route.points):
        return 0.0
    if not math.isfinite(s):
        return caps[0]
    total = route.cum_m[-1]
    if s <= 0.0:
        return caps[0]
    if s >= total:
        return caps[-1]
    i = bisect.bisect_right(route.cum_m, s) - 1
    i = max(0, min(i, len(caps) - 2))
    remain = max(0.0, route.cum_m[i + 1] - s)
    envelope = math.sqrt(caps[i + 1] * caps[i + 1]
                         + 2.0 * max(0.0, decel_ms2) * remain)
    return min(caps[i], envelope)
