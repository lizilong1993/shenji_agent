#!/usr/bin/env python3
"""Build a minimal ``msgeo.envpack.v1`` pack from a 庙算 ``basic.json``.

This is the P1 fallback pipeline: it preserves the source map (terrain,
``cond`` landcover, road adjacency, rivers) as 5 km tiles plus a
``base_map.json``, and writes a manifest with bbox/provenance.  It does not
yet ingest OSM vectors, Copernicus DEM or GEBCO bathymetry; the manifest's
``layers`` block declares the fallback source so a future full pipeline can
replace individual layers without changing the runtime contract.

Usage:
    python3 scripts/build_envpack.py --basic map_9001/basic.json \
        --out envpacks/taiwan_strait --theatre taiwan_strait
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from compat.legacy import LegacyHexGrid  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basic", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--theatre", default="demo")
    ap.add_argument("--tile-m", type=float, default=5000.0)
    args = ap.parse_args(argv)

    with open(args.basic, encoding="utf-8") as fh:
        basic = json.load(fh)
    if not basic.get("grid"):
        print("ERROR: basic.json has no grid metadata; cannot georeference", file=sys.stderr)
        return 2
    grid = LegacyHexGrid.from_map_doc(basic)
    os.makedirs(args.out, exist_ok=True)

    corners = [grid.cell_center_lonlat(0, 0),
               grid.cell_center_lonlat(0, grid.cols - 1),
               grid.cell_center_lonlat(grid.rows - 1, 0),
               grid.cell_center_lonlat(grid.rows - 1, grid.cols - 1)]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    # Tile cells by their centre in 5 km bins.
    tiles = {}
    for r, row in enumerate(basic["map_data"]):
        for c, cell in enumerate(row):
            lon, lat = grid.cell_center_lonlat(r, c)
            tx = int(math.floor((lon - bbox[0]) * 111320.0 * math.cos(math.radians(grid.center_lat)) / args.tile_m))
            ty = int(math.floor((lat - bbox[1]) * 110540.0 / args.tile_m))
            tiles.setdefault((tx, ty), []).append(cell)
    tile_dir = os.path.join(args.out, "terrain")
    os.makedirs(tile_dir, exist_ok=True)
    for (tx, ty), cells in sorted(tiles.items()):
        with open(os.path.join(tile_dir, "%d_%d.json" % (tx, ty)), "w",
                  encoding="utf-8") as fh:
            json.dump({"cells": cells}, fh, ensure_ascii=False)

    with open(os.path.join(args.out, "base_map.json"), "w", encoding="utf-8") as fh:
        json.dump(basic, fh, ensure_ascii=False)

    manifest = {
        "schema": "msgeo.envpack.v1",
        "theatre": args.theatre,
        "bbox": bbox,
        "origin": {"lon": grid.center_lon, "lat": grid.center_lat, "alt": 0.0},
        "tile_size_m": args.tile_m,
        "layers": {
            "terrain": {"source": "basic.json (fallback)", "resolution_m": grid.cell_m},
            "landcover": {"source": "basic.json cond", "classes":
                          ["open", "forest", "urban", "soft", "water", "rock"]},
            "water": {"source": "basic.json cond==4", "depth": "synthetic 20 m"},
            "roads": {"source": "basic.json adjacency", "schema": "legacy.cell"},
        },
        "fallback_basic_json": "base_map.json",
        "provenance": {"source": args.basic, "built": "2026-10-21",
                       "note": "fallback pack: OSM/DEM/GEBCO ingestion pending"},
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print("wrote", os.path.join(args.out, "manifest.json"),
          "tiles", len(tiles), "cells", sum(len(v) for v in tiles.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
