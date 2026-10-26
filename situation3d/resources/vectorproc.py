"""Normalise vector/raster datasets into registered resources.

Vector output (buildings):

    <root>/features.geojson      EPSG:4326 FeatureCollection
    <root>/manifest.json         s3d.vector.v1

Raster output (canopy / land cover):

    <root>/<name>.tif
    <root>/manifest.json         s3d.raster.v1

The same ResourceEntry / manifest / cache conventions as tile acquisition are
used, so these datasets participate in the online/offline resolver and can be
consumed by envpack enrichment or a future city voxel builder.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .model import (ResourceEntry, canonical_json, hash_path, sha256_bytes,
                    slugify, utc_now)


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    bbox: Tuple[float, float, float, float]
    kind: str = "building_footprints"
    layer: str = ""
    resolution_m: float = 0.0

    def validate(self):
        if not self.name:
            raise ValueError("dataset name is required")
        if len(self.bbox) != 4:
            raise ValueError("bbox must be west,south,east,north")
        w, s, e, n = [float(v) for v in self.bbox]
        if not (w < e and s < n):
            raise ValueError("bbox must be west<south<east<north")
        if not all(math.isfinite(v) for v in (w, s, e, n)):
            raise ValueError("bbox must be finite")
        return self

    def as_dict(self):
        return {
            "schema": "s3d.dataset.spec.v1",
            "name": self.name,
            "bbox": [float(v) for v in self.bbox],
            "kind": self.kind,
            "layer": self.layer,
            "resolution_m": float(self.resolution_m),
        }

    def hash(self) -> str:
        return sha256_bytes(canonical_json(self.as_dict()).encode("utf-8"))

    @property
    def slug(self):
        return slugify(self.name)


def write_vector(root: str, features: Sequence[Dict[str, Any]],
                 spec: DatasetSpec, source: Dict[str, Any],
                 stats: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    os.makedirs(root, exist_ok=True)
    collection = {"type": "FeatureCollection", "features": list(features)}
    path = os.path.join(root, "features.geojson")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, ensure_ascii=False)
    os.replace(tmp, path)
    manifest = {
        "schema": "s3d.vector.v1",
        "name": spec.name,
        "kind": spec.kind,
        "crs": "EPSG:4326",
        "bbox": [float(v) for v in spec.bbox],
        "spec_hash": spec.hash(),
        "built": utc_now(),
        "source": dict(source),
        "feature_count": stats.get("feature_count"),
        "height_coverage": stats.get("height_coverage"),
        "height_mean_m": stats.get("height_mean_m"),
        "feature_file": "features.geojson",
    }
    if extra:
        manifest.update(extra)
    mpath = os.path.join(root, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return manifest


def vector_stats(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    heights = []
    for feature in features:
        value = (feature.get("properties") or {}).get("height_m")
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            heights.append(value)
    count = len(features)
    return {
        "feature_count": count,
        "height_coverage": (len(heights) / float(count)) if count else 0.0,
        "height_mean_m": (sum(heights) / len(heights)) if heights else 0.0,
        "height_p90_m": (sorted(heights)[int(len(heights) * 0.9)]
                         if heights else 0.0),
    }


def write_raster_manifest(root: str, raster_name: str, spec: DatasetSpec,
                          source: Dict[str, Any], info: Dict[str, Any],
                          extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    manifest = {
        "schema": "s3d.raster.v1",
        "name": spec.name,
        "kind": spec.kind,
        "bbox": [float(v) for v in spec.bbox],
        "spec_hash": spec.hash(),
        "built": utc_now(),
        "source": dict(source),
        "raster_file": raster_name,
        "width": info.get("width"),
        "height": info.get("height"),
        "band_count": info.get("band_count"),
        "crs": info.get("crs"),
        "stats": info.get("stats") or {},
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(root, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return manifest


def register_dataset(store, spec: DatasetSpec, root: str,
                     source: Dict[str, Any], license_text: str,
                     quality: str = "measured", provider: str = "adapter",
                     as_of: str = "", notes: Optional[Dict[str, Any]] = None):
    digest, size = hash_path(root)
    entry = ResourceEntry(
        entry_id="%s:%s" % (spec.kind, spec.slug),
        kind=spec.kind,
        path=os.path.abspath(root),
        content_hash=digest,
        size=size,
        version=digest[:12],
        schema="s3d.vector.v1" if spec.kind == "building_footprints"
        else "s3d.raster.v1",
        provider=provider,
        source=source.get("label") or provider,
        fetched_at=utc_now(),
        as_of=as_of,
        bbox=spec.bbox,
        license=license_text,
        quality=quality,
        notes=dict(notes or {}, spec_hash=spec.hash(),
                   manifest="manifest.json"),
    ).validate()
    store.register(entry, replace=True)
    return entry
