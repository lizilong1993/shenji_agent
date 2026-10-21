#!/usr/bin/env python3
"""Build an ``msgeo.envpack.v1`` from OSM/DEM with online/offline/auto modes.

Examples:
    # online: fetch OSM (Overpass) + Terrarium DEM, write cache + pack
    python3 scripts/build_envpack.py --basic map_9001/basic.json \
        --out envpacks/taiwan --mode online --cache data_cache

    # offline: replay the same cache, no network access
    python3 scripts/build_envpack.py --basic map_9001/basic.json \
        --out envpacks/taiwan_offline --mode offline --cache data_cache

    # auto: prefer cache, fetch only missing entries, fall back to basic.json
    python3 scripts/build_envpack.py --basic map_9001/basic.json \
        --out envpacks/taiwan --mode auto

The online and offline builds share normalisation, so with a complete cache
the layer files are identical (manifest provenance differs by mode only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from compat.legacy import LegacyHexGrid  # noqa: E402
from geodata.cache import CacheStore  # noqa: E402
from geodata.pipeline import build_envpack  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basic", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--theatre", default="demo")
    ap.add_argument("--mode", choices=("online", "offline", "auto"), default="auto")
    ap.add_argument("--cache", default=None,
                    help="cache dir shared by online and offline builds")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--resolution-m", type=float, default=200.0)
    ap.add_argument("--dem-zoom", type=int, default=10)
    ap.add_argument("--no-osm", action="store_true")
    ap.add_argument("--no-dem", action="store_true")
    ap.add_argument("--no-bath", action="store_true",
                    help="skip GEBCO bathymetry sampling")
    ap.add_argument("--osm-pbf", default=None,
                    help="local .osm.pbf extract; replaces Overpass")
    ap.add_argument("--dem-source", choices=("terrarium", "copernicus"),
                    default="terrarium")
    ap.add_argument("--height-geojson", default=None,
                    help="polygon GeoJSON with height/h for building fallback")
    args = ap.parse_args(argv)

    with open(args.basic, encoding="utf-8") as fh:
        basic = json.load(fh)
    if not basic.get("grid"):
        print("ERROR: basic.json has no grid metadata; cannot georeference",
              file=sys.stderr)
        return 2
    grid = LegacyHexGrid.from_map_doc(basic)
    cache = CacheStore(args.cache or os.path.join(args.out, "_cache"),
                       mode=args.mode)
    osm_source = None
    if args.osm_pbf:
        from geodata.osm_pbf import PbfOsmSource
        osm_source = PbfOsmSource(cache, args.osm_pbf)
    dem_source = None
    if args.dem_source == "copernicus":
        from geodata.dem_copernicus import CopernicusDemSource
        dem_source = CopernicusDemSource(cache)
    manifest = build_envpack(
        args.out, basic, grid, theatre=args.theatre, mode=args.mode,
        cache_dir=args.cache, resolution_m=args.resolution_m,
        dem_zoom=args.dem_zoom, use_osm=not args.no_osm,
        use_dem=not args.no_dem, use_bathymetry=not args.no_bath,
        height_geojson=args.height_geojson,
        osm_source=osm_source, dem_source=dem_source, bbox=args.bbox)
    print(json.dumps({"out": args.out, "mode": args.mode,
                      "provenance": manifest["provenance"],
                      "bbox": manifest["bbox"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
