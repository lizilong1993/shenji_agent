"""Environment pack contract (msgeo.envpack.v1).

A pack is a directory with a ``manifest.json`` and layer files.  This module
validates the manifest and resolves layer paths; the actual COG/FlatGeobuf
readers belong to P1's data pipeline.  ``load_basic_map`` is the bridge used
by the current grid-backed environment (a pack may carry the 庙算
``basic.json`` as a fallback layer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

REQUIRED_LAYERS = ("terrain", "landcover")


class EnvPackError(ValueError):
    pass


class EnvPack(object):
    def __init__(self, root: str, manifest: dict):
        self.root = Path(root)
        self.manifest = manifest
        self._validate()

    def _validate(self) -> None:
        m = self.manifest
        if m.get("schema") not in ("msgeo.envpack.v1", "msgeo.envpack.v2"):
            raise EnvPackError("unsupported schema %r" % (m.get("schema"),))
        for key in ("theatre", "bbox", "tile_size_m", "layers"):
            if key not in m:
                raise EnvPackError("manifest missing %r" % key)
        if len(m["bbox"]) != 4:
            raise EnvPackError("bbox must be [lon_min, lat_min, lon_max, lat_max]")
        for layer in REQUIRED_LAYERS:
            if layer not in m["layers"]:
                raise EnvPackError("manifest missing layer %r" % layer)

    @classmethod
    def load(cls, root: str) -> "EnvPack":
        path = Path(root) / "manifest.json"
        if not path.is_file():
            raise EnvPackError("manifest.json not found under %s" % root)
        return cls(root, json.loads(path.read_text(encoding="utf-8")))

    def layer_path(self, name: str) -> Path:
        return self.root / name

    def tile_path(self, layer: str, z: int, x: int, y: int) -> Path:
        meta = self.manifest.get("layers", {}).get(layer, {})
        if meta.get("format") == "json":
            return self.root / layer / ("%d_%d.json" % (x, y))
        return self.root / layer / str(z) / str(x) / ("%s.tif" % y)

    @property
    def bbox(self):
        return tuple(float(v) for v in self.manifest["bbox"])

    @property
    def tile_size_m(self) -> float:
        return float(self.manifest["tile_size_m"])

    def basic_map_path(self) -> Optional[Path]:
        fallback = self.manifest.get("fallback_basic_json")
        if not fallback:
            return None
        p = self.root / fallback
        return p if p.is_file() else None

    def load_basic_map(self) -> dict:
        p = self.basic_map_path()
        if p is None:
            raise EnvPackError("pack has no fallback basic.json")
        return json.loads(p.read_text(encoding="utf-8"))
