"""Geodesic helpers for the situation3d engine.

All coordinates are WGS84 degrees. Distances are metres unless stated otherwise.
Implemented with the WGS84 mean radius so results are consistent with the
great-circle paths that CesiumJS draws between sampled positions.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

# WGS84 mean radius (metres).
EARTH_RADIUS_M = 6371008.8

# Nautical mile in metres, for equipment speed conversions.
NM_IN_M = 1852.0

Coord = Tuple[float, float]  # (lat_deg, lon_deg)


def haversine_m(a: Coord, b: Coord) -> float:
    """Great-circle distance between two coordinates in metres."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def initial_bearing_deg(a: Coord, b: Coord) -> float:
    """Initial true bearing from a to b, in degrees clockwise from north."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def interpolate(a: Coord, b: Coord, fraction: float) -> Coord:
    """Spherical linear interpolation between two coordinates.

    ``fraction`` is clamped to [0, 1]. Interpolating on the sphere rather than in
    lat/lon space keeps long legs from bending towards the poles.
    """
    if fraction <= 0.0:
        return a
    if fraction >= 1.0:
        return b

    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])

    d = haversine_m(a, b) / EARTH_RADIUS_M
    if d < 1e-12:
        return a

    sin_d = math.sin(d)
    w1 = math.sin((1.0 - fraction) * d) / sin_d
    w2 = math.sin(fraction * d) / sin_d

    x = w1 * math.cos(lat1) * math.cos(lon1) + w2 * math.cos(lat2) * math.cos(lon2)
    y = w1 * math.cos(lat1) * math.sin(lon1) + w2 * math.cos(lat2) * math.sin(lon2)
    z = w1 * math.sin(lat1) + w2 * math.sin(lat2)

    lat = math.atan2(z, math.hypot(x, y))
    lon = math.atan2(y, x)
    return (math.degrees(lat), math.degrees(lon))


def path_length_m(points: Sequence[Coord]) -> float:
    """Total great-circle length of a polyline in metres."""
    return sum(haversine_m(points[i], points[i + 1]) for i in range(len(points) - 1))


def knots_to_mps(knots: float) -> float:
    """Convert knots to metres per second."""
    return knots * NM_IN_M / 3600.0


def kmh_to_knots(kmh: float) -> float:
    """Convert kilometres per hour to knots.

    Ground units are specified in km/h because that is how they are quoted in
    open sources, while the frame schema stores a single canonical speed unit.

    ``1 kn = 1.852 km/h``. The previous implementation divided by 3600 as well,
    turning 45 km/h into 0.0067 kn and freezing every caller (R04/R05/B03 and
    the dense-transfer target definitions).
    """
    return kmh * 1000.0 / NM_IN_M


def sample_polyline(points: Sequence[Coord], distance_m: float) -> Tuple[Coord, float]:
    """Return the position at ``distance_m`` along a polyline and its true bearing.

    If ``distance_m`` exceeds the polyline length the final vertex is returned
    with the bearing of the last leg.
    """
    if len(points) < 2:
        raise ValueError("polyline needs at least two points")
    if distance_m <= 0.0:
        return points[0], initial_bearing_deg(points[0], points[1])

    remaining = distance_m
    for i in range(len(points) - 1):
        leg = haversine_m(points[i], points[i + 1])
        if remaining <= leg or i == len(points) - 2:
            fraction = 1.0 if leg <= 0.0 else min(1.0, remaining / leg)
            return interpolate(points[i], points[i + 1], fraction), initial_bearing_deg(
                points[i], points[i + 1]
            )
        remaining -= leg

    return points[-1], initial_bearing_deg(points[-2], points[-1])


def densify(points: Iterable[Coord], max_segment_m: float) -> list:
    """Insert intermediate vertices so no leg exceeds ``max_segment_m``."""
    pts = list(points)
    if len(pts) < 2 or max_segment_m <= 0:
        return pts

    out = [pts[0]]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        leg = haversine_m(a, b)
        steps = max(1, int(math.ceil(leg / max_segment_m)))
        for s in range(1, steps + 1):
            out.append(interpolate(a, b, s / steps))
    return out
