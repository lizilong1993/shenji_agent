"""Building/obstacle facade over the existing envpack spatial index.

The existing ``EnvPackEnvironment`` already stores building footprints in a
spatial index and performs segment/polygon + height checks.  This facade makes
that fact explicit so LOS/passability callers do not read the private index or
re-implement geometry.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from .manifest import V3Manifest
from mobility.passability import VEHICLE_BRIDGE_CLASSES


def _point_segment_mt(lon: float, lat: float, a, b) -> Tuple[float, float]:
    kx = 111320.0 * math.cos(math.radians(lat))
    ky = 110540.0
    px, py = lon * kx, lat * ky
    ax, ay = a[0] * kx, a[1] * ky
    bx, by = b[0] * kx, b[1] * ky
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    tc = max(0.0, min(1.0, t))
    dist = math.hypot(px - (ax + tc * dx), py - (ay + tc * dy))
    return dist, t


def _point_segment_m(lon: float, lat: float, a, b) -> float:
    return _point_segment_mt(lon, lat, a, b)[0]


class ObstacleSet:
    def __init__(self, env: Any, manifest: V3Manifest):
        self.env = env
        self.manifest = manifest

    def __len__(self) -> int:
        return len(getattr(self.env, "_buildings", []) or [])

    def at(self, lon: float, lat: float) -> Optional[Dict[str, Any]]:
        item = None
        polygon_at = getattr(self.env, "_polygon_at", None)
        index = getattr(self.env, "_building_index", None)
        if polygon_at is not None and index is not None:
            item = polygon_at(index, lon, lat)
        if item is not None:
            return {
                "footprint": item.get("ring") or [],
                "height_m": float(item.get("height_m") or 6.0),
                "height_source": item.get("height_source"),
                "levels": item.get("levels"),
                "kind": "building",
            }
        building = self.env.building_at(lon, lat)
        if building is None:
            return None
        return {
            "footprint": building.footprint,
            "height_m": float(building.height_m),
            "height_source": None,
            "levels": None,
            "kind": "building",
        }

    def blocked(self, a: Tuple[float, float], h0: float,
                b: Tuple[float, float], h1: float) -> bool:
        """True when a building prism intersects the 3D segment a->b."""
        return bool(self.env.building_intersects(a, h0, b, h1))

    def cover_height(self, lon: float, lat: float) -> float:
        return float(self.env.cover_height(lon, lat))

    def summary(self) -> Dict[str, Any]:
        meta = self.manifest.layer("obstacles")
        return {
            "count": len(self),
            "source": meta.get("source"),
            "confidence": meta.get("confidence"),
            "format": meta.get("format"),
        }


class BridgeSet:
    """Bridge decks derived from OSM road edges.

    The legacy pack has no bridge deck geometry, so this is a low-confidence
    placeholder: deck height = max(endpoint terrain) + 8 m.  It exists to make
    the water gate distinguish "on a bridge deck" from "in the water below".
    """

    def __init__(self, env: Any, manifest: V3Manifest, tolerance_m: float = 25.0):
        self.env = env
        self.manifest = manifest
        self.tolerance_m = float(tolerance_m)
        self._items = []
        nodes, edges = env.road_graph()
        for edge in edges:
            if not edge.get("bridge"):
                continue
            if str(edge.get("class")) not in VEHICLE_BRIDGE_CLASSES:
                # Footways/cycleways/paths are bridges too, but no vehicle
                # deck; treating them as such would waive the water gate.
                continue
            pa = nodes.get(edge.get("a"))
            pb = nodes.get(edge.get("b"))
            if not pa or not pb:
                continue
            deck = max(self.env.elevation(pa[0], pa[1]),
                       self.env.elevation(pb[0], pb[1])) + 8.0
            self._items.append({
                "edge": edge,
                "a": (float(pa[0]), float(pa[1])),
                "b": (float(pb[0]), float(pb[1])),
                "deck_m": deck,
                "clearance_m": 8.0,
                "confidence": "low",
            })

    def __len__(self) -> int:
        return len(self._items)

    def at(self, lon: float, lat: float) -> Optional[Dict[str, Any]]:
        best = None
        best_m = self.tolerance_m
        for item in self._items:
            dist, t = _point_segment_mt(lon, lat, item["a"], item["b"])
            # Require the projection to fall on the bridge segment, not the
            # 25 m capsule beyond either end.
            if -1e-9 <= t <= 1.0 + 1e-9 and dist <= best_m:
                best = item
                best_m = dist
        return best

    def near(self, lon: float, lat: float) -> bool:
        """Capsule test including bridge approaches; used by sampling filters."""
        for item in self._items:
            if _point_segment_m(lon, lat, item["a"], item["b"]) <= self.tolerance_m:
                return True
        return False

    def summary(self) -> Dict[str, Any]:
        return {
            "count": len(self),
            "tolerance_m": self.tolerance_m,
            "confidence": "low",
            "note": "deck height placeholder = terrain + 8 m",
        }
