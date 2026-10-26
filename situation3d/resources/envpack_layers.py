"""Convert dataset rasters into engine-readable envpack layers.

The engine runtime deliberately has no GDAL/rasterio dependency, so rasters
are sampled at build time into the same regular lon/lat grid JSON used by the
terrain layer.  The first consumer is canopy height (CHMv2): it feeds
``EnvPackEnvironment.cover_height`` and therefore line-of-sight/cover.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .model import slugify
from .scanners import scan_envpack
from .store import ResourceStore


class LayerError(RuntimeError):
    pass


def _run(cmd, timeout_s=240):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout_s)
    if proc.returncode != 0:
        raise LayerError("%s failed: %s" % (
            cmd[0], proc.stderr.decode("utf-8", "replace")[-800:]))
    return proc


WORLDCOVER_TO_ENGINE = {
    10: "forest", 20: "open", 30: "open", 40: "soft", 50: "urban",
    60: "open", 70: "rock", 80: "water", 90: "soft", 95: "forest",
    100: "open",
}


def raster_to_grid(raster_path: str, bbox, resolution_m: float = 30.0,
                   timeout_s: int = 300, nearest: bool = False,
                   class_map: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """Warp a raster to EPSG:4326 and sample it into an envpack grid dict."""
    if shutil.which("gdalwarp") is None:
        raise LayerError("gdalwarp not found")
    w, s, e, n = [float(v) for v in bbox]
    midlat = (s + n) / 2.0
    dx = float(resolution_m) / (111320.0 * max(0.05, math.cos(math.radians(midlat))))
    dy = float(resolution_m) / 110540.0
    # AAIGrid only carries one cellsize header in GDAL 3.0; use a square
    # degree cell (slight anisotropy at mid-latitudes is acceptable for cover).
    cell = max(dx, dy)
    dx = dy = cell
    fd, warped = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    fd, asc = tempfile.mkstemp(suffix=".asc")
    os.close(fd)
    try:
        _run(["gdalwarp", "-q", "-overwrite", "-t_srs", "EPSG:4326",
              "-te", "%.8f" % w, "%.8f" % s, "%.8f" % e, "%.8f" % n,
              "-tr", "%.10f" % dx, "%.10f" % dy,
              "-r", "near" if nearest else "bilinear",
              "-of", "GTiff", raster_path, warped], timeout_s=timeout_s)
        # GDAL 3.0 gdalwarp only writes GTiff/VRT directly; convert to ASCII.
        _run(["gdal_translate", "-q", "-of", "AAIGrid", warped, asc],
             timeout_s=timeout_s)
        with open(asc, encoding="utf-8") as fh:
            text = fh.read()
    finally:
        for path in (warped, asc):
            try:
                os.unlink(path)
            except OSError:
                pass
    from geodata.aaigrid import parse_aaigrid
    grid = parse_aaigrid(text)
    rows = grid["rows"]
    cols = grid["cols"]
    # AAIGrid is north-up; envpack grids are south-up (values[0] near lat0).
    values = list(reversed(grid["values"]))
    nodata = grid.get("nodata")
    clean = []
    for row in values:
        out = []
        for value in row:
            if nodata is not None and value == nodata:
                value = -1
            if class_map is not None:
                out.append(class_map.get(int(round(float(value))), None))
                continue
            if value >= 250 or value < 0:  # CHMv2 no-canopy sentinel / bad cells
                value = 0.0
            out.append(round(float(value), 2))
        clean.append(out)
    return {
        "lon0": grid["xll"],
        "lat0": grid["yll"],
        "lon1": grid["xll"] + cols * grid["cell"],
        "lat1": grid["yll"] + rows * grid["cell"],
        "cols": cols,
        "rows": rows,
        "values": clean,
    }


def add_landcover_layer(store: ResourceStore, envpack_entry, landcover_entry,
                        name: Optional[str] = None, resolution_m: float = 10.0,
                        cache_root: Optional[str] = None) -> Dict[str, Any]:
    if envpack_entry.kind != "envpack":
        raise LayerError("envpack entry required")
    if landcover_entry.kind != "landcover":
        raise LayerError("landcover entry required")
    manifest_path = os.path.join(landcover_entry.path, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        lc_manifest = json.load(fh)
    raster_file = lc_manifest.get("raster_file") or "landcover.tif"
    raster = os.path.join(landcover_entry.path, raster_file)
    if not os.path.isfile(raster):
        raise LayerError("landcover raster missing: %s" % raster)
    with open(os.path.join(envpack_entry.path, "manifest.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    bbox = manifest.get("bbox") or envpack_entry.bbox
    grid = raster_to_grid(raster, bbox, resolution_m=resolution_m,
                          nearest=True, class_map=WORLDCOVER_TO_ENGINE)
    cache_root = cache_root or os.path.join(str(store.root), "cache", "envpacks")
    os.makedirs(cache_root, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = name or (envpack_entry.entry_id.split(":", 1)[-1] + "-landcover")
    out = os.path.join(cache_root, "%s@%s" % (base, stamp))
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(envpack_entry.path, tmp)
    layer_dir = os.path.join(tmp, "landcover_raster")
    os.makedirs(layer_dir, exist_ok=True)
    with open(os.path.join(layer_dir, "grid.json"), "w", encoding="utf-8") as fh:
        json.dump(grid, fh, ensure_ascii=False)
    manifest["built"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    manifest["as_of"] = manifest["built"]
    manifest.setdefault("layers", {})["landcover_raster"] = {
        "format": "grid-json", "path": "landcover_raster/grid.json",
        "source": "WorldCover via %s" % landcover_entry.entry_id,
        "license": landcover_entry.license,
        "resolution_m": resolution_m,
        "class_map": {str(k): v for k, v in WORLDCOVER_TO_ENGINE.items()},
        "as_of": lc_manifest.get("source", {}).get("as_of", ""),
    }
    manifest.setdefault("provenance", {})["landcover_raster"] = {
        "source": landcover_entry.entry_id, "resolution_m": resolution_m}
    with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, out)
    scanned = scan_envpack(out)
    entry_id = "envpack:" + slugify(base) if name else scanned.entry_id
    entry = type(scanned)(**dict(scanned.as_dict(), bbox=scanned.bbox,
                                 entry_id=entry_id, provider="enrichment",
                                 source="%s + %s" % (
                                     envpack_entry.entry_id,
                                     landcover_entry.entry_id),
                                 as_of=scanned.as_of,
                                 notes=dict(scanned.notes,
                                            landcover_source=landcover_entry.entry_id,
                                            resolution_m=resolution_m)))
    store.register(entry, replace=True)
    return {"entry": entry, "grid": {"cols": grid["cols"], "rows": grid["rows"],
                                     "lon0": grid["lon0"], "lat0": grid["lat0"]}}


def add_canopy_layer(store: ResourceStore, envpack_entry, canopy_entry,
                     name: Optional[str] = None, resolution_m: float = 30.0,
                     cache_root: Optional[str] = None) -> Dict[str, Any]:
    if envpack_entry.kind != "envpack":
        raise LayerError("envpack entry required")
    if canopy_entry.kind != "canopy_height":
        raise LayerError("canopy_height entry required")
    manifest_path = os.path.join(canopy_entry.path, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        canopy_manifest = json.load(fh)
    raster_file = canopy_manifest.get("raster_file") or "canopy_height.tif"
    raster = os.path.join(canopy_entry.path, raster_file)
    if not os.path.isfile(raster):
        raise LayerError("canopy raster missing: %s" % raster)
    with open(os.path.join(envpack_entry.path, "manifest.json"),
              encoding="utf-8") as fh:
        manifest = json.load(fh)
    bbox = manifest.get("bbox") or envpack_entry.bbox
    grid = raster_to_grid(raster, bbox, resolution_m=resolution_m)
    cache_root = cache_root or os.path.join(str(store.root), "cache", "envpacks")
    os.makedirs(cache_root, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = name or (envpack_entry.entry_id.split(":", 1)[-1] + "-canopy")
    out = os.path.join(cache_root, "%s@%s" % (base, stamp))
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(envpack_entry.path, tmp)
    canopy_dir = os.path.join(tmp, "canopy")
    os.makedirs(canopy_dir, exist_ok=True)
    with open(os.path.join(canopy_dir, "grid.json"), "w", encoding="utf-8") as fh:
        json.dump(grid, fh, ensure_ascii=False)
    manifest["built"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    manifest["as_of"] = manifest["built"]
    manifest.setdefault("layers", {})["canopy"] = {
        "format": "grid-json", "path": "canopy/grid.json",
        "source": "CHMv2 via %s" % canopy_entry.entry_id,
        "license": canopy_entry.license,
        "resolution_m": resolution_m,
        "as_of": canopy_manifest.get("source", {}).get("as_of", ""),
    }
    manifest.setdefault("provenance", {})["canopy"] = {
        "source": canopy_entry.entry_id,
        "resolution_m": resolution_m,
    }
    with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, out)
    scanned = scan_envpack(out)
    entry_id = "envpack:" + slugify(base) if name else scanned.entry_id
    entry = type(scanned)(**dict(scanned.as_dict(), bbox=scanned.bbox,
                                 entry_id=entry_id, provider="enrichment",
                                 source="%s + %s" % (
                                     envpack_entry.entry_id,
                                     canopy_entry.entry_id),
                                 as_of=scanned.as_of,
                                 notes=dict(scanned.notes,
                                            canopy_source=canopy_entry.entry_id,
                                            resolution_m=resolution_m)))
    store.register(entry, replace=True)
    return {"entry": entry, "grid": {"cols": grid["cols"], "rows": grid["rows"],
                                     "lon0": grid["lon0"], "lat0": grid["lat0"]}}
