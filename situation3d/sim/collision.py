"""OBB footprints, SAT overlap and a coarse spatial index (P0.2).

Ground movement uses each platform's physical footprint instead of the old
"four units per rule cell" stop rule: planning and motion stay unchanged,
but a candidate pose is rejected when the OBB would overlap another ground
platform, which produces queueing at bottlenecks.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

M_PER_DEG_LON = 111320.0
M_PER_DEG_LAT = 110540.0


def _kx(lat: float) -> float:
    return M_PER_DEG_LON * math.cos(math.radians(lat))


def footprint_samples(prof, lon: float, lat: float,
                      heading_deg: float) -> List[Tuple[float, float]]:
    """Nine OBB samples in lon/lat: centre, corners and edge midpoints."""
    h = math.radians(heading_deg)
    sin_h, cos_h = math.sin(h), math.cos(h)
    hl = max(0.6, float(prof.length_m)) / 2.0
    hw = max(0.4, float(prof.width_m)) / 2.0
    kx, ky = _kx(lat), M_PER_DEG_LAT
    out = []
    for sf in (-1.0, 0.0, 1.0):
        for sr in (-1.0, 0.0, 1.0):
            east = sf * hl * sin_h + sr * hw * cos_h
            north = sf * hl * cos_h - sr * hw * sin_h
            out.append((lon + east / kx, lat + north / ky))
    return out


def _corners(prof, heading_deg: float, ox: float, oy: float,
             inflate: float) -> List[Tuple[float, float]]:
    h = math.radians(heading_deg)
    sin_h, cos_h = math.sin(h), math.cos(h)
    hl = max(0.6, float(prof.length_m)) / 2.0 + inflate
    hw = max(0.4, float(prof.width_m)) / 2.0 + inflate
    pts = []
    for sf in (-1.0, 1.0):
        for sr in (-1.0, 1.0):
            pts.append((ox + sf * hl * sin_h + sr * hw * cos_h,
                        oy + sf * hl * cos_h - sr * hw * sin_h))
    return pts


def _project(points, axis) -> Tuple[float, float]:
    lo = hi = points[0][0] * axis[0] + points[0][1] * axis[1]
    for x, y in points[1:]:
        v = x * axis[0] + y * axis[1]
        if v < lo:
            lo = v
        elif v > hi:
            hi = v
    return lo, hi


def obbs_overlap(prof_a, lon_a: float, lat_a: float, heading_a: float,
                 prof_b, lon_b: float, lat_b: float, heading_b: float,
                 margin: float = 0.25) -> bool:
    kx, ky = _kx((lat_a + lat_b) / 2.0), M_PER_DEG_LAT
    ax, ay = 0.0, 0.0
    bx, by = (lon_b - lon_a) * kx, (lat_b - lat_a) * ky
    corners_a = _corners(prof_a, heading_a, ax, ay, margin)
    corners_b = _corners(prof_b, heading_b, bx, by, margin)
    for heading in (heading_a, heading_b):
        h = math.radians(heading)
        for axis in ((math.cos(h), -math.sin(h)), (math.sin(h), math.cos(h))):
            lo_a, hi_a = _project(corners_a, axis)
            lo_b, hi_b = _project(corners_b, axis)
            if hi_a < lo_b or hi_b < lo_a:
                return False
    return True


class CollisionIndex(object):
    """Coarse lon/lat bucket index over ground entities."""

    def __init__(self, cell_m: float = 100.0):
        self.cell_deg = float(cell_m) / M_PER_DEG_LAT
        self.buckets: Dict[Tuple[int, int], List[object]] = {}
        self.keys: Dict[int, Tuple[int, int]] = {}

    def _key(self, lon: float, lat: float) -> Tuple[int, int]:
        return (int(math.floor(lon / self.cell_deg)),
                int(math.floor(lat / self.cell_deg)))

    def rebuild(self, entities: Iterable) -> None:
        self.buckets = {}
        self.keys = {}
        for entity in entities:
            self.insert(entity)

    def insert(self, entity) -> None:
        if entity.uid in self.keys:
            return
        key = self._key(entity.pose.lon, entity.pose.lat)
        self.keys[entity.uid] = key
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = self.buckets[key] = []
        bucket.append(entity)

    def refresh(self, entity) -> None:
        old = self.keys.pop(entity.uid, None)
        key = self._key(entity.pose.lon, entity.pose.lat)
        if old == key:
            self.keys[entity.uid] = key
            return
        if old is not None:
            bucket = self.buckets.get(old)
            if bucket is not None:
                self.buckets[old] = [e for e in bucket if e.uid != entity.uid]
        self.keys[entity.uid] = key
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = self.buckets[key] = []
        bucket.append(entity)

    def near(self, lon: float, lat: float):
        cx, cy = self._key(lon, lat)
        seen = set()
        for ix in (cx - 1, cx, cx + 1):
            for iy in (cy - 1, cy, cy + 1):
                for entity in self.buckets.get((ix, iy), ()):  # type: ignore[arg-type]
                    if entity.uid in seen:
                        continue
                    seen.add(entity.uid)
                    yield entity
