#!/usr/bin/env python3
"""Download the offline asset set for situation3d.

Everything is stored as XYZ (Web Mercator) tiles so the three families line up
with each other and with Cesium's WebMercatorTilingScheme:

  imagery   satellite / aerial raster basemap
  terrain   DEM in Terrarium encoding, consumed by CustomHeightmapTerrainProvider
  wsf3d     World Settlement Footprint 3D building height / volume, via DLR WMS

Standard library only: the target container has no pip and no system packages.

Usage
-----
    # everything, default AOI (Taiwan Strait) and zoom ranges
    python3 tools/fetch_assets.py all

    # one family
    python3 tools/fetch_assets.py imagery --source esri --max-zoom 11
    python3 tools/fetch_assets.py terrain --max-zoom 10
    python3 tools/fetch_assets.py wsf3d   --max-zoom 10

    # a different area
    python3 tools/fetch_assets.py all --bbox 118,21,123,26

Run `--dry-run` first if you want the tile count and an estimated size before
committing to a long download.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

# Tile URL templates use {z}/{x}/{y} unless noted. `flip_y` marks services
# that take {z}/{y}/{x} in that order.
IMAGERY_SOURCES: Dict[str, dict] = {
    "esri": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "ext": "jpg",
        "max_zoom": 19,
        "attribution": "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        "licence": "Esri terms of use -- free with attribution; review before operational use.",
    },
    "eox-s2": {
        "url": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg",
        "ext": "jpg",
        "max_zoom": 14,
        "attribution": "Sentinel-2 cloudless 2020 by EOX IT Services GmbH",
        "licence": "CC BY-NC-SA 4.0 -- non-commercial only.",
    },
    "gibs-modis": {
        "url": (
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            "MODIS_Terra_CorrectedReflectance_TrueColor/default/2023-06-01/"
            "GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
        ),
        "ext": "jpg",
        "max_zoom": 9,
        "attribution": "NASA EOSDIS GIBS / MODIS Terra",
        "licence": "Public domain (US Government work). 250 m, includes cloud cover.",
    },
}

TERRAIN_SOURCES: Dict[str, dict] = {
    "terrarium": {
        "url": "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        "ext": "png",
        "max_zoom": 15,
        "attribution": "Elevation tiles by Mapzen, hosted on AWS Open Data (SRTM and others)",
        "licence": "Public domain / CC0.",
        # Decoded by the front end as (R*256 + G + B/256) - 32768 metres.
    },
}

# DLR GeoService WMS. Layer names must be upper case -- lower case returns a
# ServiceException, and WMS 1.3.0 with EPSG:4326 would need lat/lon axis order,
# so EPSG:3857 is used to stay in unambiguous XY metres.
WSF3D_LAYERS: Dict[str, dict] = {
    "buildingheight": {
        "layer": "WSF3D_V02_BUILDINGHEIGHT",
        "title": "WSF3D 建筑高度 (m)",
    },
    "buildingvolume": {
        "layer": "WSF3D_V02_BUILDINGVOLUME",
        "title": "WSF3D 建筑体积 (m³)",
    },
    "buildingfraction": {
        "layer": "WSF3D_V02_BUILDINGFRACTION",
        "title": "WSF3D 建筑占比",
    },
}
WSF3D_WMS = "https://geoservice.dlr.de/eoc/land/wms"
WSF3D_ATTRIBUTION = (
    "World Settlement Footprint 3D (WSF3D) v02, DLR EOC -- "
    "https://geoservice.dlr.de/web/maps/eoc:wsf3d"
)
WSF3D_LICENCE = "DLR GeoService terms -- attribution required, see the EOC terms of use."

DEFAULT_BBOX = (115.0, 18.0, 125.0, 29.0)  # Taiwan Strait and its approaches
DEFAULT_MAX_ZOOM = {"imagery": 11, "terrain": 10, "wsf3d": 10}

TILE_SIZE = 256
MERCATOR_R = 6378137.0


# --------------------------------------------------------------------------
# tiling maths
# --------------------------------------------------------------------------


def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[float, float]:
    """Fractional XYZ tile coordinates."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(max(min(lat, 85.05112878), -85.05112878))
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_range(bbox: Sequence[float], z: int) -> Tuple[int, int, int, int]:
    """Inclusive (x0, x1, y0, y1) tile indices covering bbox at zoom z."""
    w, s, e, n = bbox
    x0f, y0f = lonlat_to_tile(w, n, z)          # north-west corner
    x1f, y1f = lonlat_to_tile(e, s, z)          # south-east corner
    n_side = 2 ** z
    x0 = max(0, min(n_side - 1, int(math.floor(x0f))))
    x1 = max(0, min(n_side - 1, int(math.floor(x1f))))
    y0 = max(0, min(n_side - 1, int(math.floor(y0f))))
    y1 = max(0, min(n_side - 1, int(math.floor(y1f))))
    return x0, x1, y0, y1


def tile_mercator_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """EPSG:3857 (minx, miny, maxx, maxy) of a tile."""
    n = 2.0 ** z
    world = 2.0 * math.pi * MERCATOR_R
    size = world / n
    minx = -world / 2.0 + x * size
    maxy = world / 2.0 - y * size
    return minx, maxy - size, minx + size, maxy


def tiles_for_bbox(bbox: Sequence[float], min_z: int, max_z: int) -> Iterable[Tuple[int, int, int]]:
    for z in range(min_z, max_z + 1):
        x0, x1, y0, y1 = tile_range(bbox, z)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                yield z, x, y


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def fetch_bytes(url: str, timeout: int = 45, retries: int = 3) -> Optional[bytes]:
    last = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "situation3d-asset-fetcher/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except HTTPError as exc:
            # 404 means "no tile here" (ocean, empty settlement) -- not retryable.
            if exc.code == 404:
                return None
            last = exc
        except (URLError, TimeoutError, OSError) as exc:
            last = exc
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed: {url}: {last}")


def looks_like_image(data: Optional[bytes], ext: str) -> bool:
    if not data or len(data) < 100:
        return False
    if ext == "jpg":
        return data[:2] == b"\xff\xd8"
    if ext == "png":
        return data[:8] == b"\x89PNG\r\n\x1a\n"
    return True


def write_tile(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def run_batch(
    jobs: List[Tuple[str, str]],
    workers: int,
    label: str,
) -> Tuple[int, int, int, List[str]]:
    """jobs: list of (url, destination path). Returns (ok, skipped, failed, errors)."""
    ok = skipped = failed = 0
    errors: List[str] = []
    total = len(jobs)
    if not total:
        return 0, 0, 0, []

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for url, dest in jobs:
            if os.path.exists(dest) and os.path.getsize(dest) > 100:
                skipped += 1
                continue
            futures[pool.submit(fetch_bytes, url)] = (url, dest)

        done = 0
        for fut in as_completed(futures):
            url, dest = futures[fut]
            done += 1
            try:
                data = fut.result()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failed += 1
                if len(errors) < 10:
                    errors.append(str(exc))
                continue
            if data is None:
                # Service has no tile here. Record the absence so a re-run does
                # not keep asking for it.
                write_tile(dest + ".absent", b"")
                skipped += 1
                continue
            write_tile(dest, data)
            ok += 1

            if done % 250 == 0 or done == len(futures):
                rate = done / max(1e-6, time.time() - started)
                print(
                    f"  {label}: {done}/{len(futures)} fetched "
                    f"({rate:.0f}/s, {ok} new, {skipped} skipped, {failed} failed)",
                    flush=True,
                )
    return ok, skipped, failed, errors


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------


def fetch_imagery(out_dir: str, bbox, min_z: int, max_z: int, source: str, workers: int):
    cfg = IMAGERY_SOURCES[source]
    max_z = min(max_z, cfg["max_zoom"])
    ext = cfg["ext"]
    jobs = []
    for z, x, y in tiles_for_bbox(bbox, min_z, max_z):
        url = cfg["url"].format(z=z, x=x, y=y)
        jobs.append((url, os.path.join(out_dir, "imagery", str(z), str(x), f"{y}.{ext}")))
    print(f"imagery [{source}] z{min_z}-{max_z}: {len(jobs)} tiles")
    return run_batch(jobs, workers, "imagery"), {
        "source": source,
        "url_template": cfg["url"],
        "attribution": cfg["attribution"],
        "licence": cfg["licence"],
        "ext": ext,
        "min_zoom": min_z,
        "max_zoom": max_z,
    }


def fetch_terrain(out_dir: str, bbox, min_z: int, max_z: int, source: str, workers: int):
    cfg = TERRAIN_SOURCES[source]
    max_z = min(max_z, cfg["max_zoom"])
    jobs = []
    for z, x, y in tiles_for_bbox(bbox, min_z, max_z):
        url = cfg["url"].format(z=z, x=x, y=y)
        jobs.append((url, os.path.join(out_dir, "terrain", str(z), str(x), f"{y}.png")))
    print(f"terrain [{source}] z{min_z}-{max_z}: {len(jobs)} tiles")
    return run_batch(jobs, workers, "terrain"), {
        "source": source,
        "url_template": cfg["url"],
        "attribution": cfg["attribution"],
        "licence": cfg["licence"],
        "encoding": "terrarium",
        "decode": "height_m = (R*256 + G + B/256) - 32768",
        "min_zoom": min_z,
        "max_zoom": max_z,
    }


def fetch_wsf3d(out_dir: str, bbox, min_z: int, max_z: int, layers: Sequence[str], workers: int):
    metas = {}
    results = []
    for name in layers:
        cfg = WSF3D_LAYERS[name]
        jobs = []
        for z, x, y in tiles_for_bbox(bbox, min_z, max_z):
            minx, miny, maxx, maxy = tile_mercator_bounds(z, x, y)
            url = (
                f"{WSF3D_WMS}?service=WMS&version=1.3.0&request=GetMap"
                f"&layers={cfg['layer']}&styles=&crs=EPSG:3857"
                f"&bbox={minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f}"
                f"&width={TILE_SIZE}&height={TILE_SIZE}&format=image/png&transparent=true"
            )
            jobs.append((url, os.path.join(out_dir, "wsf3d", name, str(z), str(x), f"{y}.png")))
        print(f"wsf3d [{name}] z{min_z}-{max_z}: {len(jobs)} tiles")
        results.append(run_batch(jobs, workers, f"wsf3d:{name}"))
        metas[name] = {
            "wms_layer": cfg["layer"],
            "title": cfg["title"],
            "min_zoom": min_z,
            "max_zoom": max_z,
        }
    return results, {
        "source": "DLR EOC GeoService WMS",
        "wms_url": WSF3D_WMS,
        "attribution": WSF3D_ATTRIBUTION,
        "licence": WSF3D_LICENCE,
        "min_zoom": min_z,
        "max_zoom": max_z,
        "layers": metas,
    }


# --------------------------------------------------------------------------


def count_tiles(root: str) -> Tuple[int, int]:
    n = 0
    size = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".part") or f.endswith(".absent"):
                continue
            n += 1
            size += os.path.getsize(os.path.join(dirpath, f))
    return n, size


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download offline tiles for situation3d")
    ap.add_argument("what", choices=["imagery", "terrain", "wsf3d", "all"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "tiles"))
    ap.add_argument("--bbox", default=",".join(str(v) for v in DEFAULT_BBOX),
                    help="west,south,east,north (default: Taiwan Strait)")
    ap.add_argument("--min-zoom", type=int, default=0)
    ap.add_argument("--max-zoom", type=int, default=None)
    ap.add_argument("--source", default="esri", choices=sorted(IMAGERY_SOURCES))
    ap.add_argument("--terrain-source", default="terrarium", choices=sorted(TERRAIN_SOURCES))
    ap.add_argument("--wsf3d-layers", default="buildingheight,buildingvolume")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    # Secondary fetches (the urban AOIs) must not clobber the primary manifest:
    # they cover a tiny box and a narrow zoom band, and writing that over the
    # theatre-wide entry would tell the front end the basemap only exists over
    # one city.
    ap.add_argument("--no-manifest", action="store_true",
                    help="fetch tiles without touching manifest.json")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        print("ERROR: --bbox needs west,south,east,north", file=sys.stderr)
        return 2

    families = ["imagery", "terrain", "wsf3d"] if args.what == "all" else [args.what]

    if args.dry_run:
        for fam in families:
            mz = args.max_zoom if args.max_zoom is not None else DEFAULT_MAX_ZOOM[fam]
            n = sum(1 for _ in tiles_for_bbox(bbox, args.min_zoom, mz))
            print(f"{fam:8s} z{args.min_zoom}-{mz}: {n} tiles")
        return 0

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = {"bbox": list(bbox), "generated_utc": datetime.now(timezone.utc).isoformat()}
    if args.no_manifest:
        manifest = None  # type: ignore[assignment]
    if manifest is not None and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest.update(json.load(fh))
        except (OSError, ValueError):
            pass

    failures = 0
    for fam in families:
        mz = args.max_zoom if args.max_zoom is not None else DEFAULT_MAX_ZOOM[fam]
        if fam == "imagery":
            (ok, skip, bad, errs), meta = fetch_imagery(out_dir, bbox, args.min_zoom, mz, args.source, args.workers)
            if manifest is not None:
                manifest["imagery"] = meta
        elif fam == "terrain":
            (ok, skip, bad, errs), meta = fetch_terrain(out_dir, bbox, args.min_zoom, mz, args.terrain_source, args.workers)
            if manifest is not None:
                manifest["terrain"] = meta
        else:
            layers = [s.strip() for s in args.wsf3d_layers.split(",") if s.strip()]
            results, meta = fetch_wsf3d(out_dir, bbox, args.min_zoom, mz, layers, args.workers)
            ok = sum(r[0] for r in results)
            skip = sum(r[1] for r in results)
            bad = sum(r[2] for r in results)
            errs = [e for r in results for e in r[3]]
            if manifest is not None:
                manifest["wsf3d"] = meta
        print(f"{fam}: {ok} fetched, {skip} skipped, {bad} failed")
        for e in errs[:5]:
            print(f"  error: {e}")
        failures += bad

    n, size = count_tiles(out_dir)
    print(f"total: {n} tiles, {human(size)}")
    if manifest is None:
        print("manifest: skipped (--no-manifest)")
        return 1 if failures else 0

    manifest["tile_count"] = n
    manifest["total_bytes"] = size
    manifest["zoom_range"] = {"min": args.min_zoom, "max": args.max_zoom}
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
