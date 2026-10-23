#!/usr/bin/env python3
"""Build an envpack from GeoJSON exported by GDAL from a local OSM PBF.

This is the offline path used for the TW2 AOI, where pyosmium is not available
but GDAL's OSM driver is.  Export the ``lines`` and ``multipolygons`` layers
with ``ogr2ogr -f GeoJSON ... -spat W S E N`` first, then run this tool.

The normal Overpass pipeline does not fetch the open sea; pass ``--sea-box`` to
add a conservative water rectangle (well west of the coastline) so ships and
land passability see water.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from compat.legacy import LegacyHexGrid  # noqa: E402
from geodata.osm import normalize  # noqa: E402
from geodata.pipeline import build_envpack  # noqa: E402

TAG_KEYS = ("highway", "natural", "waterway", "landuse", "building", "bridge",
            "tunnel", "oneway", "surface", "width", "maxheight", "maxweight",
            "name", "building:levels", "height")


def _parse_other(value):
    out = {}
    if not value:
        return out
    for key, val in re.findall(r'"([^"]+)"=>"([^"]*)"', str(value)):
        out[key] = val
    return out


def _tags(props):
    tags = {}
    for key in TAG_KEYS:
        value = props.get(key)
        if value not in (None, ""):
            tags[key] = str(value)
    tags.update(_parse_other(props.get("other_tags")))
    return tags


def _add_way(elements, osm_id, coords, tags):
    if len(coords) < 2:
        return
    elements.append({"type": "way", "id": osm_id, "tags": tags,
                     "geometry": [{"lon": c[0], "lat": c[1]} for c in coords]})


def collect(lines_path, poly_path):
    elements = []
    lines = json.load(open(lines_path, encoding="utf-8"))
    for feature in lines.get("features") or []:
        geom = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        tags = _tags(props)
        if not any(k in tags for k in ("highway", "waterway", "natural", "landuse", "building")):
            continue
        coords = geom.get("coordinates") or []
        if geom.get("type") == "LineString":
            _add_way(elements, props.get("osm_id"), coords, tags)
        elif geom.get("type") == "MultiLineString":
            for i, part in enumerate(coords):
                _add_way(elements, "%s-%d" % (props.get("osm_id"), i), part, tags)
    polys = json.load(open(poly_path, encoding="utf-8"))
    for feature in polys.get("features") or []:
        geom = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        tags = _tags(props)
        if not any(k in tags for k in ("building", "natural", "landuse", "waterway")):
            continue
        polygons = ([geom.get("coordinates") or []] if geom.get("type") == "Polygon"
                    else (geom.get("coordinates") or []))
        for i, polygon in enumerate(polygons):
            if not polygon:
                continue
            ring = polygon[0]
            if len(ring) >= 4:
                _add_way(elements, "%s-%d" % (props.get("osm_id"), i), ring, tags)
    return normalize({"elements": elements})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--polygons", required=True)
    ap.add_argument("--basic", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox", nargs=4, type=float, required=True)
    ap.add_argument("--sea-box", nargs=4, type=float, default=None,
                    metavar=("W", "S", "E", "N"))
    ap.add_argument("--resolution-m", type=float, default=100.0)
    ap.add_argument("--theatre", default="taiwan_strait")
    args = ap.parse_args(argv)

    features = collect(args.lines, args.polygons)
    if args.sea_box:
        w, s, e, n = args.sea_box
        features["water"].append({"ring": [[w, s], [e, s], [e, n], [w, n], [w, s]],
                                  "name": "示意海面"})
    print("features:", {k: len(v) for k, v in features.items()}, flush=True)
    basic = json.load(open(args.basic, encoding="utf-8"))
    grid = LegacyHexGrid.from_map_doc(basic)

    class Source(object):
        provenance_label = "osm-pbf-gdal"
        fetched_network = False

        def features(self, bbox):
            return features

    manifest = build_envpack(args.out, basic, grid, theatre=args.theatre,
                             mode="offline", resolution_m=args.resolution_m,
                             use_osm=True, use_dem=False, use_bathymetry=False,
                             osm_source=Source(), bbox=list(args.bbox))
    # The pack geometry is authoritative inside the AOI; a coarse base-map
    # water/land fallback produced false water at inland points in TW2.
    manifest["fallback_basic_json"] = None
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"out": args.out, "bbox": manifest["bbox"],
                      "provenance": manifest["provenance"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
