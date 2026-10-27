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

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
SENTINEL2_COLLECTION = "sentinel-2-l2a"
SENTINEL2_LICENSE = "Copernicus Sentinel data terms (free, open, attribution)"

WORLDCOVER_BASE = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                   "v200/2021/map")
WORLDCOVER_LICENSE = "CC BY 4.0 (ESA WorldCover 2021 v200)"
WORLDCOVER_AS_OF = "2021-01-01"


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


def _tag_upper(lon: int, lat: int) -> str:
    return "%s%02d%s%03d" % ("N" if lat >= 0 else "S", abs(int(lat)),
                             "E" if lon >= 0 else "W", abs(int(lon)))


def _worldcover_tiles(bbox, step: int = 3):
    w, s, e, n = [float(v) for v in bbox]
    lat0 = int(math.floor(s / step) * step)
    lon0 = int(math.floor(w / step) * step)
    out = []
    while lon0 <= e:
        lat = lat0
        while lat <= n:
            out.append((lon0, lat))
            lat += step
        lon0 += step
    return out


def _worldcover_url(lon0: int, lat0: int) -> str:
    return "%s/ESA_WorldCover_10m_2021_v200_%s_Map.tif" % (
        WORLDCOVER_BASE, _tag_upper(lon0, lat0))


def _http_json(url: str, payload=None, timeout_s: float = 30.0) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(url, data=data, headers={
        "Content-Type": "application/json",
        "User-Agent": "situation3d-geo-engine/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_day(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise DatasetError("date must be ISO 8601: %r" % (value,))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def search_sentinel2(bbox, target_date=None, window_days: int = 60,
                     max_cloud: float = 30.0, limit: int = 40) -> Any:
    target = _parse_day(target_date)
    start = target - __import__("datetime").timedelta(days=int(window_days))
    payload = {
        "collections": [SENTINEL2_COLLECTION],
        "bbox": [float(v) for v in bbox],
        "datetime": "%s/%s" % (start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                               target.strftime("%Y-%m-%dT%H:%M:%SZ")),
        "limit": int(limit),
    }
    try:
        doc = _http_json(EARTH_SEARCH_URL, payload)
    except Exception as exc:  # noqa: BLE001
        raise DatasetError("Sentinel-2 STAC search failed: %s" % exc)
    features = doc.get("features") or []
    candidates = []
    for feature in features:
        props = feature.get("properties") or {}
        assets = feature.get("assets") or {}
        href = None
        for key in ("visual", "TCI", "red"):
            asset = assets.get(key) or {}
            href = asset.get("href")
            if href:
                break
        if not href:
            continue
        try:
            cloud = float(props.get("eo:cloud_cover"))
        except (TypeError, ValueError):
            cloud = 100.0
        try:
            when = _parse_day(props.get("datetime"))
        except DatasetError:
            continue
        candidates.append({"id": feature.get("id"), "href": href,
                           "datetime": when, "cloud": cloud,
                           "bbox": feature.get("bbox"),
                           "assets": sorted(assets)})
    if not candidates:
        raise DatasetError("no Sentinel-2 scene covers %s in %s..%s"
                           % (bbox, start.date(), target.date()))
    clear = [c for c in candidates if c["cloud"] <= max_cloud]
    pool = clear or candidates
    pool.sort(key=lambda c: (c["cloud"],
                             abs((target - c["datetime"]).total_seconds())))
    chosen = pool[0]
    return chosen, {"requested": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "window_days": window_days,
                    "max_cloud": max_cloud, "found": len(candidates),
                    "clear": len(clear),
                    "cloud_cover": chosen["cloud"],
                    "scene_datetime": chosen["datetime"].strftime(
                        "%Y-%m-%dT%H:%M:%SZ")}


def fetch_sentinel2(store: ResourceStore, spec: DatasetSpec,
                    cache_root: str, target_date=None,
                    window_days: int = 60, max_cloud: float = 30.0,
                    resolution_m: float = 10.0,
                    timeout_s: int = 240) -> Any:
    spec.validate()
    chosen, query = search_sentinel2(spec.bbox, target_date, window_days,
                                     max_cloud)
    work = os.path.join(cache_root, "raster", "sentinel2")
    os.makedirs(work, exist_ok=True)
    dest = os.path.join(work, "%s.tif" % spec.slug)
    _warp_sources([chosen["href"]], spec.bbox, dest, timeout_s=timeout_s,
                  resolution_m=resolution_m, resampling="bilinear")
    info = raster_info(dest)
    source = {"label": "Sentinel-2 L2A true colour (Earth Search AWS)",
              "scene_id": chosen["id"], "scene_datetime": query["scene_datetime"],
              "cloud_cover": query["cloud_cover"], "query": query,
              "license": SENTINEL2_LICENSE,
              "as_of": query["scene_datetime"][:10]}
    return dest, info, source, chosen


def acquire_sentinel(store: ResourceStore, spec: DatasetSpec,
                     cache_root: str, target_date=None,
                     window_days: int = 60, max_cloud: float = 30.0,
                     resolution_m: float = 10.0,
                     force: bool = False) -> Any:
    spec.validate()
    if not force:
        cached = _cached_dataset(store, spec)
        if cached is not None:
            return cached
    dest, info, source, chosen = fetch_sentinel2(
        store, spec, cache_root, target_date, window_days, max_cloud,
        resolution_m)
    root = os.path.join(str(store.root), "cache", "datasets",
                        spec.kind, spec.slug, spec.hash()[:12])
    os.makedirs(root, exist_ok=True)
    import shutil
    target = os.path.join(root, "sentinel2.tif")
    if os.path.abspath(dest) != os.path.abspath(target):
        shutil.copyfile(dest, target)
    manifest = write_raster_manifest(root, "sentinel2.tif", spec, source,
                                     info, extra={"adapter": "sentinel2"})
    quality = "measured" if source["cloud_cover"] <= max_cloud else "derived"
    entry = register_dataset(store, spec, root, source, SENTINEL2_LICENSE,
                             quality=quality, provider="adapter:sentinel2",
                             as_of=source["as_of"],
                             notes={"adapter": "sentinel2",
                                    "scene_id": chosen["id"],
                                    "cloud_cover": chosen["cloud"]})
    return {"entry": entry, "manifest": manifest, "source": source,
            "chosen": chosen, "cached": False}


def acquire_sentinel_tiles(store: ResourceStore, tile_spec, cache_root: str,
                           target_date=None, window_days: int = 60,
                           max_cloud: float = 30.0,
                           resolution_m: float = 10.0,
                           fallback_sources=("eox-s2", "esri"),
                           force: bool = False) -> Any:
    """Acquire a Sentinel scene raster, then render/register XYZ tiles.

    On any Sentinel failure the function falls back to the existing XYZ
    imagery providers (EOX cloudless / Esri), so map-layer acquisition still
    succeeds when the scene chain is unavailable.
    """
    from .acquire import AcquireError, acquire_tiles
    from .sources import raster_candidate
    from .vectorproc import DatasetSpec
    errors = []
    sentinel = None
    try:
        scene_spec = DatasetSpec(
            name=tile_spec.name, bbox=tile_spec.bbox,
            kind="satellite_imagery", resolution_m=resolution_m,
            query={"date": target_date, "window_days": window_days,
                   "max_cloud": max_cloud})
        sentinel = acquire_sentinel(
            store, scene_spec, cache_root, target_date=target_date,
            window_days=window_days, max_cloud=max_cloud,
            resolution_m=resolution_m, force=force)
        entry = sentinel["entry"]
        with open(os.path.join(entry.path, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        raster = os.path.join(entry.path,
                              manifest.get("raster_file") or "sentinel2.tif")
        if not os.path.isfile(raster):
            raise DatasetError("Sentinel raster missing: %s" % raster)
        tiles = acquire_tiles(
            store, tile_spec,
            candidates=[raster_candidate("file://" + raster,
                                         source_id="sentinel2",
                                         license_text=SENTINEL2_LICENSE)],
            force=force)
        return {"sentinel": sentinel, "tiles": tiles, "fallback": False,
                "errors": errors}
    except (DatasetError, AcquireError, Exception) as exc:  # noqa: BLE001
        errors.append("%s: %s" % (type(exc).__name__, exc))
    try:
        tiles = acquire_tiles(store, tile_spec,
                              source_ids=list(fallback_sources), force=force)
        return {"sentinel": sentinel, "tiles": tiles, "fallback": True,
                "errors": errors}
    except Exception as exc:  # noqa: BLE001
        errors.append("fallback: %s: %s" % (type(exc).__name__, exc))
        raise DatasetError("Sentinel and fallback tile acquisition failed: %s"
                           % "; ".join(errors))


def _warp_sources(urls, bbox, dest: str, timeout_s: int = 240,
                  resolution_m: float = 0.0,
                  resampling: str = "bilinear") -> str:
    import shutil
    import subprocess
    if shutil.which("gdalwarp") is None:
        raise DatasetError("gdalwarp not found")
    w, s, e, n = [float(v) for v in bbox]
    env = dict(os.environ)
    env.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    env.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
    env.setdefault("GDAL_HTTP_MAX_RETRY", "2")
    env.setdefault("GDAL_HTTP_TIMEOUT", "60")
    env.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "10")
    cmd = ["gdalwarp", "-q", "-overwrite", "-wm", "256", "-te_srs", "EPSG:4326",
           "-te", "%.8f" % w, "%.8f" % s, "%.8f" % e, "%.8f" % n]
    if resolution_m > 0:
        deg = float(resolution_m) / 111320.0
        cmd += ["-t_srs", "EPSG:4326", "-tr", "%.10f" % deg, "%.10f" % deg,
                "-r", resampling]
    cmd += ["-of", "GTiff"] + ["/vsicurl/" + u for u in urls] + [dest]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout_s, env=env)
    if proc.returncode != 0:
        raise DatasetError("raster warp failed: %s"
                           % proc.stderr.decode("utf-8", "replace")[-500:])
    return dest


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
    _warp_sources(urls, spec.bbox, dest, timeout_s=timeout_s,
                  resolution_m=resolution_m, resampling="bilinear")
    info = raster_info(dest)
    source = {"label": "Meta/WRI CHMv2 canopy height", "tiles": urls,
              "license": CHMV2_LICENSE, "as_of": CHMV2_AS_OF}
    return dest, info, source


def fetch_worldcover(store: ResourceStore, spec: DatasetSpec,
                     cache_root: str, mode: str = "auto",
                     timeout_s: int = 240) -> Any:
    spec.validate()
    tiles = _worldcover_tiles(spec.bbox)
    urls = [_worldcover_url(lon, lat) for lon, lat in tiles]
    work = os.path.join(cache_root, "raster", "worldcover")
    os.makedirs(work, exist_ok=True)
    dest = os.path.join(work, "%s.tif" % spec.slug)
    _warp_sources(urls, spec.bbox, dest, timeout_s=timeout_s)
    info = raster_info(dest)
    source = {"label": "ESA WorldCover 10m 2021 v200", "tiles": urls,
              "license": WORLDCOVER_LICENSE, "as_of": WORLDCOVER_AS_OF}
    return dest, info, source


def _cached_dataset(store: ResourceStore, spec: DatasetSpec):
    """Return a previously acquired dataset for the same spec, if valid."""
    entry_id = "%s:%s" % (spec.kind, spec.slug)
    try:
        entry = store.get(entry_id)
    except Exception:  # noqa: BLE001
        return None
    if not os.path.isdir(entry.path):
        return None
    manifest_path = os.path.join(entry.path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError):
        return None
    if manifest.get("spec_hash") != spec.hash():
        return None
    if not store.verify(entry_id)["ok"]:
        return None
    stats = {"feature_count": manifest.get("feature_count"),
             "height_coverage": manifest.get("height_coverage"),
             "height_mean_m": manifest.get("height_mean_m")}
    return {"entry": entry, "manifest": manifest, "stats": stats,
            "source": manifest.get("source") or {}, "errors": [],
            "cached": True}


def acquire_landcover(store: ResourceStore, spec: DatasetSpec,
                      cache_root: str, mode: str = "auto",
                      force: bool = False) -> Any:
    spec.validate()
    if not force:
        cached = _cached_dataset(store, spec)
        if cached is not None:
            return cached
    if mode == "offline":
        raise DatasetError("offline and no cached landcover for %s" % spec.name)
    dest, info, source = fetch_worldcover(store, spec, cache_root, mode)
    root = os.path.join(str(store.root), "cache", "datasets",
                        spec.kind, spec.slug, spec.hash()[:12])
    os.makedirs(root, exist_ok=True)
    import shutil
    target = os.path.join(root, "landcover.tif")
    if os.path.abspath(dest) != os.path.abspath(target):
        shutil.copyfile(dest, target)
    manifest = write_raster_manifest(root, "landcover.tif", spec, source,
                                     info, extra={"adapter": "worldcover"})
    entry = register_dataset(store, spec, root, source, WORLDCOVER_LICENSE,
                             quality="measured", provider="adapter:worldcover",
                             as_of=WORLDCOVER_AS_OF,
                             notes={"adapter": "worldcover",
                                    "classes": "ESA WorldCover 2021"})
    return {"entry": entry, "manifest": manifest, "source": source,
            "stats": {}, "errors": [], "cached": False}


def acquire_canopy(store: ResourceStore, spec: DatasetSpec,
                   cache_root: str, mode: str = "auto",
                   resolution_m: float = 1.2, force: bool = False) -> Any:
    spec.validate()
    if not force:
        cached = _cached_dataset(store, spec)
        if cached is not None:
            return cached
    if mode == "offline":
        raise DatasetError("offline and no cached canopy for %s" % spec.name)
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
    return {"entry": entry, "manifest": manifest, "source": source,
            "stats": {}, "errors": [], "cached": False}


def acquire_buildings(store: ResourceStore, spec: DatasetSpec,
                      sources: Sequence[str], cache_root: str,
                      mode: str = "auto", force: bool = False) -> Any:
    """Try building adapters in order; first non-empty wins."""
    spec.validate()
    if not force:
        cached = _cached_dataset(store, spec)
        if cached is not None:
            return cached
    if mode == "offline":
        raise DatasetError("offline and no cached building dataset for %s"
                           % spec.name)
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
                    "source": source, "errors": errors, "cached": False}
        except Exception as exc:  # noqa: BLE001
            errors.append("%s: %s" % (name, exc))
    raise DatasetError("all building sources failed: %s" % "; ".join(errors))
