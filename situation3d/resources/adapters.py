"""Formal source adapters for China-wide datasets.

Buildings:
  * GBA (Global Building Atlas LOD1) -- 5-degree GeoParquet tiles, global.
  * Overture Maps buildings -- query via the optional ``overturemaps`` package.

Canopy:
  * Meta/WRI CHMv2 -- zoom-10 quadkey COGs on AWS Open Data, read by GDAL
    ``/vsicurl`` windows (no account).

Adapters are optional-dependency friendly: pyarrow/shapely/overturemaps are
imported only inside the functions, and the resource CLI reports a clear
error when they are missing.
"""

from __future__ import annotations

import json
import math
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

from .processing import crop_raster_window, download_file, raster_info
from .tilemath import lonlat_to_tile
from .vectorproc import (DatasetError, DatasetSpec, register_dataset,
                         vector_stats, write_raster_manifest, write_vector)
from .store import ResourceStore

GBA_BASE = "https://data.source.coop/tge-labs/globalbuildingatlas-lod1"
GBA_LICENSE = "Global Building Atlas terms (check the dataset page before redistribution)"
GBA_AS_OF = "2024-01-01"

CHMV2_BASE = ("https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/"
              "dinov3_global_chm_v2_ml3")
CHMV2_ZOOM = 10
CHMV2_LICENSE = "CC BY 4.0 (Meta/WRI CHMv2)"
CHMV2_AS_OF = "2024-01-01"

OVERTURE_LICENSE = "Overture Maps data license (per-layer; check CDLA/ODbL terms)"


def _lon_tag(lon: int) -> str:
    return "%s%03d" % ("e" if lon >= 0 else "w", abs(int(lon)))


def _lat_tag(lat: int) -> str:
    return "%s%02d" % ("n" if lat >= 0 else "s", abs(int(lat)))


def _gba_tiles(bbox, step: int = 5) -> List[Tuple[int, int, int, int]]:
    w, s, e, n = [float(v) for v in bbox]
    out = []
    lat0 = int(math.floor(s / step) * step)
    lon0 = int(math.floor(w / step) * step)
    while lon0 <= e:
        lat = lat0
        while lat <= n:
            out.append((lon0, lat, lon0 + step, lat + step))
            lat += step
        lon0 += step
    return out


def _gba_url(west: int, south: int, east: int, north: int) -> str:
    return "%s/%s_%s_%s_%s.parquet" % (
        GBA_BASE, _lon_tag(west), _lat_tag(north),
        _lon_tag(east), _lat_tag(south))


def fetch_gba(store: ResourceStore, spec: DatasetSpec, cache_root: str,
              mode: str = "auto", timeout_s: int = 300,
              max_features: int = 500000) -> Any:
    spec.validate()
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        from shapely import wkb
        from shapely.geometry import box, mapping, shape
    except ImportError as exc:
        raise DatasetError("GBA adapter needs pyarrow+shapely: %s" % exc)
    cache_dir = os.path.join(cache_root, "vector", "gba")
    os.makedirs(cache_dir, exist_ok=True)
    bbox = spec.bbox
    features = []
    seen = set()
    sources = []
    for west, south, east, north in _gba_tiles(bbox):
        url = _gba_url(west, south, east, north)
        name = os.path.basename(url)
        path = os.path.join(cache_dir, name)
        if not os.path.isfile(path):
            if mode == "offline":
                raise DatasetError("GBA tile not cached and mode=offline: %s" % url)
            try:
                download_file(url, path, timeout_s=timeout_s)
            except Exception as exc:  # noqa: BLE001
                raise DatasetError("GBA tile download failed %s: %s" % (url, exc))
        sources.append(url)
        table = pq.read_table(path, columns=["id", "height", "var", "region",
                                             "bbox", "geometry"])
        b = table["bbox"]
        xmin = pc.struct_field(b, "xmin")
        xmax = pc.struct_field(b, "xmax")
        ymin = pc.struct_field(b, "ymin")
        ymax = pc.struct_field(b, "ymax")
        mask = pc.and_(
            pc.and_(pc.greater_equal(xmax, bbox[0]),
                    pc.less_equal(xmin, bbox[2])),
            pc.and_(pc.greater_equal(ymax, bbox[1]),
                    pc.less_equal(ymin, bbox[3])))
        table = table.filter(mask)
        ids = table["id"].to_pylist()
        heights = table["height"].to_pylist()
        variances = table["var"].to_pylist()
        regions = table["region"].to_pylist()
        geoms = table["geometry"].to_pylist()
        clip = box(bbox[0], bbox[1], bbox[2], bbox[3])
        for fid, height, var, region, raw in zip(ids, heights, variances,
                                                  regions, geoms):
            if fid in seen:
                continue
            seen.add(fid)
            try:
                geom = wkb.loads(raw)
            except Exception:
                continue
            if geom.is_empty or not geom.intersects(clip):
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "id": str(fid),
                    "height_m": float(height) if height is not None else None,
                    "height_var_m": float(var) if var is not None else None,
                    "region": region,
                    "source": "gba",
                },
            })
            if len(features) > max_features:
                raise DatasetError("GBA feature count exceeds %d; narrow the bbox"
                                   % max_features)
    if not features:
        raise DatasetError("GBA returned no buildings for %s" % (bbox,))
    source = {"label": "Global Building Atlas LOD1", "tiles": sources,
              "license": GBA_LICENSE, "as_of": GBA_AS_OF}
    return features, source


def fetch_overture(store: ResourceStore, spec: DatasetSpec,
                   release: Optional[str] = None,
                   max_features: int = 500000) -> Any:
    spec.validate()
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:  # noqa: BLE001
        pass
    try:
        from overturemaps import core
        from shapely import wkb
        from shapely.geometry import box, mapping
    except ImportError as exc:
        raise DatasetError(
            "Overture adapter needs overturemaps+shapely: %s" % exc)
    try:
        reader = core.record_batch_reader("building", bbox=spec.bbox,
                                          release=release)
    except Exception as exc:  # noqa: BLE001
        raise DatasetError("Overture query failed: %s" % exc)
    if reader is None:
        raise DatasetError("Overture returned no reader")
    clip = box(spec.bbox[0], spec.bbox[1], spec.bbox[2], spec.bbox[3])
    features = []
    seen = set()
    for batch in reader:
        cols = batch.column_names
        ids = batch.column(cols.index("id")).to_pylist()
        geoms = batch.column(cols.index("geometry")).to_pylist()
        heights = (batch.column(cols.index("height")).to_pylist()
                   if "height" in cols else [None] * len(ids))
        floors = (batch.column(cols.index("num_floors")).to_pylist()
                  if "num_floors" in cols else [None] * len(ids))
        classes = (batch.column(cols.index("class")).to_pylist()
                   if "class" in cols else [None] * len(ids))
        for fid, raw, height, floor, klass in zip(ids, geoms, heights, floors,
                                                  classes):
            if fid in seen:
                continue
            seen.add(fid)
            geom = wkb.loads(raw)
            if geom.is_empty or not geom.intersects(clip):
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            h = height
            if h is None and floor:
                h = float(floor) * 3.0
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"id": str(fid),
                               "height_m": float(h) if h else None,
                               "num_floors": floor, "class": klass,
                               "source": "overture"},
            })
            if len(features) > max_features:
                raise DatasetError("Overture feature count exceeds %d" % max_features)
    if not features:
        raise DatasetError("Overture returned no buildings for %s" % (spec.bbox,))
    source = {"label": "Overture Maps buildings", "release": release,
              "license": OVERTURE_LICENSE, "as_of": ""}
    return features, source


def _quadkey(x: int, y: int, zoom: int) -> str:
    digits = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        digits.append(str(digit))
    return "".join(digits)


def _quadkeys(bbox, zoom: int = CHMV2_ZOOM) -> List[str]:
    w, s, e, n = [float(v) for v in bbox]
    n = max(-85.05112878, min(85.05112878, n))
    s = max(-85.05112878, min(85.05112878, s))
    last = 2 ** zoom - 1
    x0, y0 = lonlat_to_tile(w, n, zoom)
    x1, y1 = lonlat_to_tile(e, s, zoom)
    x0, x1 = max(0, min(last, int(x0))), max(0, min(last, int(x1)))
    y0, y1 = max(0, min(last, int(y0))), max(0, min(last, int(y1)))
    return sorted({_quadkey(x, y, zoom)
                   for x in range(min(x0, x1), max(x0, x1) + 1)
                   for y in range(min(y0, y1), max(y0, y1) + 1)})


def fetch_chmv2(store: ResourceStore, spec: DatasetSpec, cache_root: str,
                mode: str = "auto", resolution_m: float = 1.2,
                timeout_s: int = 240) -> Any:
    spec.validate()
    keys = _quadkeys(spec.bbox)
    if not keys:
        raise DatasetError("CHMv2: no tiles cover %s" % (spec.bbox,))
    urls = ["%s/chm/%s.tif" % (CHMV2_BASE, qk) for qk in keys]
    work = os.path.join(cache_root, "raster", "chmv2")
    os.makedirs(work, exist_ok=True)
    dest = os.path.join(work, "%s.tif" % spec.slug)
    # GDAL /vsicurl handles the COG windows; pass every covering tile.
    if len(urls) == 1:
        crop_raster_window(urls[0], spec.bbox, dest, timeout_s=timeout_s)
    else:
        # gdalwarp accepts multiple sources; build the command directly.
        import subprocess
        import shutil as _shutil
        if _shutil.which("gdalwarp") is None:
            raise DatasetError("gdalwarp not found")
        w, s, e, n = spec.bbox
        env = dict(os.environ)
        env.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        env.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
        env.setdefault("GDAL_HTTP_MAX_RETRY", "2")
        cmd = ["gdalwarp", "-q", "-te_srs", "EPSG:4326",
               "-te", "%.8f" % w, "%.8f" % s, "%.8f" % e, "%.8f" % n,
               "-of", "GTiff"] + ["/vsicurl/" + u for u in urls] + [dest]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout_s,
                              env=env)
        if proc.returncode != 0:
            raise DatasetError("CHMv2 warp failed: %s"
                               % proc.stderr.decode("utf-8", "replace")[-500:])
    info = raster_info(dest)
    source = {"label": "Meta/WRI CHMv2 canopy height", "tiles": urls,
              "license": CHMV2_LICENSE, "as_of": CHMV2_AS_OF}
    return dest, info, source


def acquire_canopy(store: ResourceStore, spec: DatasetSpec,
                   cache_root: str, mode: str = "auto",
                   resolution_m: float = 1.2) -> Any:
    spec.validate()
    dest, info, source = fetch_chmv2(store, spec, cache_root, mode,
                                     resolution_m=resolution_m)
    root = os.path.join(str(store.root), "cache", "datasets",
                        spec.kind, spec.slug, spec.hash()[:12])
    os.makedirs(root, exist_ok=True)
    import shutil
    target = os.path.join(root, "canopy_height.tif")
    if os.path.abspath(dest) != os.path.abspath(target):
        shutil.copyfile(dest, target)
    manifest = write_raster_manifest(root, "canopy_height.tif", spec, source,
                                     info, extra={"adapter": "chmv2"})
    entry = register_dataset(store, spec, root, source, CHMV2_LICENSE,
                             quality="measured", provider="adapter:chmv2",
                             as_of=CHMV2_AS_OF,
                             notes={"adapter": "chmv2",
                                    "resolution_m": resolution_m})
    return {"entry": entry, "manifest": manifest, "source": source}


def acquire_buildings(store: ResourceStore, spec: DatasetSpec,
                      sources: Sequence[str], cache_root: str,
                      mode: str = "auto") -> Any:
    """Try building adapters in order; first non-empty wins."""
    errors = []
    for name in sources:
        try:
            if name == "gba":
                features, source = fetch_gba(store, spec, cache_root, mode)
            elif name == "overture":
                features, source = fetch_overture(store, spec)
            else:
                raise DatasetError("unknown building source %r" % name)
            stats = vector_stats(features)
            root = os.path.join(str(store.root), "cache", "datasets",
                                spec.kind, spec.slug, spec.hash()[:12])
            os.makedirs(root, exist_ok=True)
            manifest = write_vector(root, features, spec, source, stats,
                                    extra={"adapter": name})
            quality = "measured" if stats["height_coverage"] >= 0.5 else "derived"
            entry = register_dataset(store, spec, root, source, GBA_LICENSE
                                     if name == "gba" else OVERTURE_LICENSE,
                                     quality=quality, provider="adapter:" + name,
                                     as_of=source.get("as_of", ""),
                                     notes={"adapter": name,
                                            "feature_count": stats["feature_count"],
                                            "height_coverage": stats["height_coverage"]})
            return {"entry": entry, "manifest": manifest, "stats": stats,
                    "source": source, "errors": errors}
        except Exception as exc:  # noqa: BLE001
            errors.append("%s: %s" % (name, exc))
    raise DatasetError("all building sources failed: %s" % "; ".join(errors))
