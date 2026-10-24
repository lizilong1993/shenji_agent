"""Building/obstacle facade over the existing envpack spatial index.

The existing ``EnvPackEnvironment`` already stores building footprints in a
spatial index and performs segment/polygon + height checks.  This facade makes
that fact explicit so LOS/passability callers do not read the private index or
re-implement geometry.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .manifest import V3Manifest


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
