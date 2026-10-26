"""adsb.lol aircraft snapshot connector (global, 250 nm point queries)."""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

from ..normalize import normalize_adsb

API_ROOT = "https://api.adsb.lol"
MAX_RADIUS_NM = 250
DEFAULT_RADIUS_NM = 250
KM_PER_NM = 1.852
KM_PER_DEG_LAT = 111.32


def build_url(lat: float, lon: float,
              radius_nm: int = DEFAULT_RADIUS_NM) -> str:
    radius = max(1, min(int(radius_nm), MAX_RADIUS_NM))
    return "%s/v2/point/%.6f/%.6f/%d" % (
        API_ROOT, float(lat), float(lon), radius)


def sample_points(bbox: Sequence[float],
                  radius_nm: int = DEFAULT_RADIUS_NM,
                  max_points: int = 9) -> List[Tuple[float, float]]:
    """Cover ``bbox`` with at most ``max_points`` 250 nm circles."""
    if isinstance(bbox, str) or len(bbox) != 4:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
    west, south, east, north = (float(value) for value in bbox)
    center_lat = (south + north) / 2.0
    width_km = abs(east - west) * KM_PER_DEG_LAT * max(
        0.05, math.cos(math.radians(center_lat)))
    height_km = abs(north - south) * KM_PER_DEG_LAT
    radius_km = max(1, min(int(radius_nm), MAX_RADIUS_NM)) * KM_PER_NM
    half_diagonal = math.sqrt(width_km ** 2 + height_km ** 2) / 2.0
    if half_diagonal <= radius_km:
        return [((south + north) / 2.0, (west + east) / 2.0)]
    nx = max(1, int(math.ceil(width_km / radius_km)))
    ny = max(1, int(math.ceil(height_km / radius_km)))
    if nx * ny > max_points:
        raise ValueError(
            "bbox %.0fx%.0f km needs %d ADS-B queries (limit %d)"
            % (width_km, height_km, nx * ny, max_points))
    points = []
    for ix in range(nx):
        lon = west + (ix + 0.5) * (east - west) / nx
        for iy in range(ny):
            lat = south + (iy + 0.5) * (north - south) / ny
            points.append((lat, lon))
    return points


def parse_payload(payload: dict, fetched_at: Any, start_seq: int = 0):
    return normalize_adsb(payload, fetched_at, start_seq=start_seq)
