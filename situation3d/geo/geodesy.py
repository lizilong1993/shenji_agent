"""Geodesy helpers shared by the geo engine.

Coordinates in the geo engine are ``(lon, lat)`` pairs in WGS84 degrees.
``engine.geo`` uses ``(lat, lon)`` for historical reasons, so this module
wraps it and converts argument order once here instead of at every call site.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from engine.geo import EARTH_RADIUS_M, NM_IN_M

Coord = Tuple[float, float]  # (lon_deg, lat_deg)


def haversine_m(a: Coord, b: Coord) -> float:
    """Great-circle distance in metres between two ``(lon, lat)`` points."""
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def initial_bearing_deg(a: Coord, b: Coord) -> float:
    """Initial true bearing from a to b, degrees clockwise from north."""
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def interpolate(a: Coord, b: Coord, fraction: float) -> Coord:
    """Spherical interpolation between two ``(lon, lat)`` points."""
    if fraction <= 0.0:
        return a
    if fraction >= 1.0:
        return b
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
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
    return (math.degrees(lon), math.degrees(lat))


def path_length_m(points: Sequence[Coord]) -> float:
    return sum(haversine_m(points[i], points[i + 1]) for i in range(len(points) - 1))


def knots_to_mps(knots: float) -> float:
    return knots * NM_IN_M / 3600.0


__all__ = [
    "EARTH_RADIUS_M",
    "NM_IN_M",
    "Coord",
    "haversine_m",
    "initial_bearing_deg",
    "interpolate",
    "knots_to_mps",
    "path_length_m",
]
