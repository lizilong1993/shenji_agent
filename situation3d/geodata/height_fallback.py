"""Building-height fallback from a local polygon GeoJSON (e.g. GBA tiles).

OSM heights are sparse in many areas: only ``height``/``building:levels``
tags provide real values.  This module fills the remaining footprints by
matching each building centroid against polygons that carry a height
attribute (``properties.height`` or ``properties.h``), the same convention
as the tw_lod1 GBA tiles.  Inputs may be WGS84 or Web-Mercator (detected by
coordinate magnitude).
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

DEFAULT_HEIGHT = 6.0
MAX_PLAUSIBLE_M = 150.0   # GBA satellite heights have long-lived spikes


def _mercator_to_wgs(lon: float, lat: float):
    r = 6378137.0
    return (math.degrees(lon / r), math.degrees(math.atan(math.sinh(lat / r))))


def _ring_points(geometry: dict) -> List[List[float]]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if gtype == "Polygon":
        return coords[0] if coords else []
    if gtype == "MultiPolygon":
        return coords[0][0] if coords and coords[0] else []
    return []


def _contains(ring: List[List[float]], x: float, y: float) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < xi + (y - yi) * (xj - xi) / (yj - yi):
            inside = not inside
        j = i
    return inside


class HeightFallback(object):
    def __init__(self, path: str, cell_deg: float = 0.01):
        self.path = path
        self.cell = float(cell_deg)
        self._index = None
        self._count = 0

    def _load(self):
        if self._index is not None:
            return
        with open(self.path, encoding="utf-8") as fh:
            doc = json.load(fh)
        feats = doc.get("features") or []
        index: Dict[tuple, list] = {}
        for feat in feats:
            ring = _ring_points(feat.get("geometry") or {})
            if len(ring) < 3:
                continue
            props = feat.get("properties") or {}
            height = props.get("height", props.get("h"))
            try:
                height = float(height)
            except (TypeError, ValueError):
                continue
            if not (0 < height < 1000):
                continue
            if abs(ring[0][0]) > 1000.0:  # Web Mercator metres
                ring = [list(_mercator_to_wgs(p[0], p[1])) for p in ring]
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            rec = (ring, height, min(xs), min(ys), max(xs), max(ys))
            for ix in range(int(math.floor(min(xs) / self.cell)),
                            int(math.floor(max(xs) / self.cell)) + 1):
                for iy in range(int(math.floor(min(ys) / self.cell)),
                                int(math.floor(max(ys) / self.cell)) + 1):
                    index.setdefault((ix, iy), []).append(rec)
        self._index = index
        self._count = len(feats)

    def height_at(self, lon: float, lat: float) -> Optional[float]:
        self._load()
        key = (int(math.floor(lon / self.cell)), int(math.floor(lat / self.cell)))
        for ring, height, x0, y0, x1, y1 in self._index.get(key, ()):
            if x0 <= lon <= x1 and y0 <= lat <= y1 and _contains(ring, lon, lat):
                return height
        return None

    def fill(self, buildings: List[dict]) -> Dict[str, int]:
        """Fill ``height_m`` for buildings whose ``height_source`` is default."""
        self._load()
        filled = 0
        spikes = 0
        for item in buildings:
            if item.get("height_source") not in (None, "default"):
                continue
            ring = item.get("ring") or []
            if len(ring) < 3:
                continue
            lon = sum(p[0] for p in ring) / len(ring)
            lat = sum(p[1] for p in ring) / len(ring)
            height = self.height_at(lon, lat)
            if height is None:
                continue
            if height > MAX_PLAUSIBLE_M:
                spikes += 1
                continue
            item["height_m"] = round(height, 1)
            item["height_source"] = "fallback"
            filled += 1
        return {"filled": filled, "spikes": spikes, "total": len(buildings)}
