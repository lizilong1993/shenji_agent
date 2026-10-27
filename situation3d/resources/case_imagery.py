"""Per-case imagery recipes (Taiwan, India-China) for reuse without re-download.

The recipes point at the already cached Sentinel-2 scenes/tiles in
``<store>/var/resources``; ``ensure_case_imagery`` is an idempotent call:
cache hit -> no network, cache miss -> normal Sentinel acquisition/fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .adapters import acquire_sentinel_tiles
from .processing import TilesetSpec


@dataclass(frozen=True)
class CaseImagery:
    key: str
    name: str
    bbox: Tuple[float, float, float, float]
    date: str
    window_days: int
    max_cloud: float
    resolution_m: float
    min_zoom: int
    max_zoom: int
    mosaic: bool = False

    def tileset(self) -> TilesetSpec:
        return TilesetSpec(name=self.name, bbox=self.bbox,
                           min_zoom=self.min_zoom, max_zoom=self.max_zoom,
                           layer="imagery")


CASES: Dict[str, CaseImagery] = {
    "taiwan": CaseImagery(
        key="taiwan", name="taiwan-anping-s2",
        bbox=(120.055, 22.915, 120.235, 23.025),
        date="2026-10-26", window_days=180, max_cloud=30.0,
        resolution_m=20.0, min_zoom=10, max_zoom=13),
    "india_china": CaseImagery(
        key="india_china", name="india-china-s2",
        bbox=(78.7, 33.35, 79.5, 34.15),
        date="2025-12-31", window_days=365, max_cloud=30.0,
        resolution_m=30.0, min_zoom=10, max_zoom=12, mosaic=True),
}


def case_keys():
    return sorted(CASES)


def ensure_case_imagery(store, key: str, cache_root=None,
                        force: bool = False):
    try:
        recipe = CASES[key]
    except KeyError:
        raise KeyError("unknown case imagery %r (known: %s)"
                       % (key, ", ".join(case_keys())))
    return acquire_sentinel_tiles(
        store, recipe.tileset(), cache_root,
        target_date=recipe.date, window_days=recipe.window_days,
        max_cloud=recipe.max_cloud, resolution_m=recipe.resolution_m,
        force=force, mosaic=recipe.mosaic)
