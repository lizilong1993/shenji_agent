"""Enrich envpack building heights from a building-footprint resource.

This is the practical part of the VoxCity idea we want in our engine: use a
global building-height dataset (GBA/Overture) to fill the coarse ``default``
heights of the envpack buildings by footprint overlap, producing a new,
registered envpack version (the old one and its runs stay untouched).
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .model import slugify
from .scanners import scan_envpack
from .store import ResourceStore


class EnrichmentError(RuntimeError):
    pass


def load_features(path: str):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc.get("features") or []


def match_building_heights(env_buildings, features,
                           only_sources=("default",)) -> Tuple[Dict[int, float], Dict[str, Any]]:
    try:
        from shapely.geometry import Polygon, shape
        from shapely.strtree import STRtree
    except ImportError as exc:  # pragma: no cover
        raise EnrichmentError("shapely is required for enrichment: %s" % exc)
    candidates = []
    geoms = []
    heights = []
    for feature in features:
        props = feature.get("properties") or {}
        try:
            height = float(props.get("height_m"))
        except (TypeError, ValueError):
            continue
        if not (height > 0):
            continue
        try:
            geom = shape(feature.get("geometry"))
        except Exception:
            continue
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        candidates.append(len(geoms))
        geoms.append(geom)
        heights.append(height)
    if not geoms:
        return {}, {"candidates": 0, "matched": 0, "filled": 0}
    tree = STRtree(geoms)
    updates = {}
    matched = 0
    for index, item in enumerate(env_buildings):
        source = str(item.get("height_source") or "default")
        if only_sources and source not in only_sources:
            continue
        try:
            ring = item.get("ring") or []
            poly = Polygon(ring)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
        except Exception:
            continue
        point = poly.representative_point()
        hits = tree.query(point)
        best = None
        best_area = 0.0
        for hit in hits:
            geom = geoms[int(hit)]
            if not geom.contains(point) and not geom.intersects(poly):
                continue
            try:
                area = geom.intersection(poly).area
            except Exception:
                area = 0.0
            if area >= best_area:
                best_area = area
                best = int(hit)
        if best is not None:
            matched += 1
            updates[index] = heights[best]
    return updates, {"candidates": len(geoms), "matched": matched,
                     "filled": len(updates)}


def enrich_envpack(store: ResourceStore, envpack_entry, buildings_entry,
                   name: Optional[str] = None,
                   cache_root: Optional[str] = None) -> Dict[str, Any]:
    if envpack_entry.kind != "envpack":
        raise EnrichmentError("envpack entry required")
    if buildings_entry.kind != "building_footprints":
        raise EnrichmentError("building_footprints entry required")
    src = envpack_entry.path
    buildings_path = os.path.join(src, "buildings", "buildings.json")
    if not os.path.isfile(buildings_path):
        raise EnrichmentError("envpack has no buildings layer: %s" % buildings_path)
    feature_path = os.path.join(buildings_entry.path, "features.geojson")
    if not os.path.isfile(feature_path):
        raise EnrichmentError("building resource has no features.geojson")
    with open(buildings_path, encoding="utf-8") as fh:
        buildings = json.load(fh)
    features = load_features(feature_path)
    updates, stats = match_building_heights(buildings, features)
    for index, height in updates.items():
        buildings[index]["height_m"] = round(float(height), 2)
        buildings[index]["height_source"] = "gbd_overlap"
    cache_root = cache_root or os.path.join(str(store.root), "cache", "envpacks")
    os.makedirs(cache_root, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = name or (envpack_entry.entry_id.split(":", 1)[-1] + "-gbd")
    out = os.path.join(cache_root, "%s@%s" % (base, stamp))
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    with open(os.path.join(tmp, "buildings", "buildings.json"), "w",
              encoding="utf-8") as fh:
        json.dump(buildings, fh, ensure_ascii=False)
    manifest_path = os.path.join(tmp, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["built"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    manifest["as_of"] = manifest["built"]
    provenance = manifest.setdefault("provenance", {})
    provenance["building_heights"] = {
        "source": buildings_entry.entry_id,
        "matched": stats["matched"],
        "filled": stats["filled"],
        "candidates": stats["candidates"],
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, out)
    new_entry = scan_envpack(out)
    entry_id = ("envpack:" + slugify(name)) if name else new_entry.entry_id
    new_entry = type(new_entry)(**dict(new_entry.as_dict(), bbox=new_entry.bbox,
                                       entry_id=entry_id,
                                       provider="enrichment",
                                       source="%s + %s" % (
                                           envpack_entry.entry_id,
                                           buildings_entry.entry_id),
                                       as_of=new_entry.as_of,
                                       notes=dict(new_entry.notes, enriched_from=envpack_entry.entry_id,
                                                  buildings_source=buildings_entry.entry_id,
                                                  height_fill=stats)))
    store.register(new_entry, replace=True)
    return {"entry": new_entry, "stats": stats}
