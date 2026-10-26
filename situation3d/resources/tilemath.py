"""Web-Mercator (XYZ) tile maths shared by acquisition and processing."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

TILE_SIZE = 256
MERCATOR_R = 6378137.0


def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[float, float]:
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(max(min(lat, 85.05112878), -85.05112878))
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_range(bbox: Sequence[float], z: int) -> Tuple[int, int, int, int]:
    w, s, e, n = [float(v) for v in bbox]
    x0f, y0f = lonlat_to_tile(w, n, z)
    x1f, y1f = lonlat_to_tile(e, s, z)
    side = 2 ** z
    x0 = max(0, min(side - 1, int(math.floor(x0f))))
    x1 = max(0, min(side - 1, int(math.floor(x1f))))
    y0 = max(0, min(side - 1, int(math.floor(y0f))))
    y1 = max(0, min(side - 1, int(math.floor(y1f))))
    return x0, x1, y0, y1


def tile_mercator_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    side = 2.0 ** z
    world = 2.0 * math.pi * MERCATOR_R
    size = world / side
    minx = -world / 2.0 + x * size
    maxy = world / 2.0 - y * size
    return minx, maxy - size, minx + size, maxy


def tiles_for_bbox(bbox: Sequence[float], min_z: int, max_z: int
                   ) -> Iterable[Tuple[int, int, int]]:
    for z in range(int(min_z), int(max_z) + 1):
        x0, x1, y0, y1 = tile_range(bbox, z)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                yield z, x, y


def tile_count(bbox: Sequence[float], min_z: int, max_z: int) -> int:
    return sum(1 for _ in tiles_for_bbox(bbox, min_z, max_z))
