"""Validation and access for ``msgeo.envpack.v3`` manifests.

The existing ``EnvPackEnvironment`` reads the layer files by path; this module
adds the v3 contract checks (identity, bbox, required layers, per-layer
source/license/confidence) without changing the runtime loader.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

REQUIRED_LAYERS = (
    "terrain",
    "hydrography",
    "landcover",
    "buildings",
    "roads",
    "obstacles",
)
LAYER_REQUIRED_FIELDS = ("format", "source", "license", "confidence")


class ManifestError(ValueError):
    """Raised when an envpack manifest does not satisfy the v3 contract."""


@dataclass(frozen=True)
class V3Manifest:
    root: str
    data: Dict[str, Any]

    @property
    def schema(self) -> str:
        return str(self.data.get("schema") or "")

    @property
    def pack_id(self) -> str:
        return str(self.data.get("pack_id") or "")

    @property
    def version(self) -> str:
        return str(self.data.get("version") or "")

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        values = self.data.get("bbox") or []
        return tuple(float(v) for v in values)  # type: ignore[return-value]

    @property
    def crs(self) -> str:
        return str(self.data.get("crs") or "EPSG:4326")

    @property
    def layers(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.data.get("layers") or {})

    def layer(self, name: str) -> Dict[str, Any]:
        return dict(self.layers.get(name) or {})

    def layer_path(self, name: str) -> Optional[str]:
        meta = self.layer(name)
        path = meta.get("path")
        if not path:
            return None
        candidate = os.path.realpath(os.path.join(self.root, str(path)))
        root = os.path.realpath(self.root)
        try:
            if os.path.commonpath([root, candidate]) != root:
                raise ManifestError("layer path escapes pack root: %r" % path)
        except ValueError:
            raise ManifestError("layer path escapes pack root: %r" % path)
        return candidate


def load_manifest(root: str) -> V3Manifest:
    """Load and validate ``<root>/manifest.json`` as v3.

    Raises:
        ManifestError: missing file, wrong schema, or missing required fields.
    """
    path = os.path.join(root, "manifest.json")
    if not os.path.isfile(path):
        raise ManifestError("manifest.json not found under %s" % root)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")

    if data.get("schema") != "msgeo.envpack.v3":
        raise ManifestError(
            "unsupported schema %r (expected msgeo.envpack.v3)"
            % (data.get("schema"),)
        )

    for key in ("pack_id", "version", "bbox", "layers", "license"):
        if key not in data:
            raise ManifestError("manifest missing %r" % key)

    bbox = data.get("bbox") or []
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ManifestError("bbox must be [lon_min, lat_min, lon_max, lat_max]")
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v))
               for v in bbox):
        raise ManifestError("bbox values must be finite numbers")

    layers = data.get("layers")
    if not isinstance(layers, dict):
        raise ManifestError("layers must be an object")
    if "hydrography" in layers and "water" not in layers:
        # Legacy EnvPackEnvironment reads the "water" key; v3 standardizes on
        # "hydrography". Keep one compatibility alias in memory.
        layers["water"] = dict(layers["hydrography"])
    for name in REQUIRED_LAYERS:
        if name not in layers:
            raise ManifestError("manifest missing layer %r" % name)
    for name, meta in layers.items():
        if not isinstance(meta, dict):
            raise ManifestError("layer %r must be an object" % name)
        for field in LAYER_REQUIRED_FIELDS:
            if not meta.get(field):
                raise ManifestError(
                    "layer %r missing %r" % (name, field)
                )
    return V3Manifest(root=root, data=data)
