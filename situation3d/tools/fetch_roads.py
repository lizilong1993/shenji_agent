#!/usr/bin/env python3
"""Fetch and slim OSM road networks, so land units can be constrained to roads.

Why this exists
---------------
Land platforms do not drive across open country: they follow roads, and in a
high-altitude theatre like Ladakh the road network is the *only* way armour and
artillery move at all. Routings that ignore this produce tracks nobody believes,
and — more importantly for this project — they let the intent layer judge
`REDEPLOY`/`ASSEMBLE` on a platform that physically could not have manoeuvred
there.

The network is fetched once and stored, matching the offline-first pattern used
for imagery, terrain, tiles and vector data.

Output
------
    engine/data/roads/<theatre>.json

    {"schema": "situation3d.roads.v1",
     "theatre": "...", "bbox": [...],
     "osm_base_utc": "...",          # from the mirror, NOT the fetch time
     "classes": ["primary", ...],
     "nodes": [[lon, lat], ...],      # positional, index == node id
     "ways": [{"id": ..., "class": "primary", "ref": "G219",
               "oneway": false, "nodes": [i0, i1, ...]}]}

Nodes are emitted as a positional array rather than an id map: it roughly halves
the file, and the graph builder only ever needs indices. OSM node ids are not
preserved — a deliberate trade, documented here so nobody later expects them.

Mirrors and staleness
---------------------
`overpass-api.de` currently answers **406 from Apache** (not an Overpass error) for
every query, including ones that worked previously, so the mirror list is ordered
with `overpass.kumi.systems` first and the script reports the mirror's own
`osm_base` timestamp. That timestamp is the age of the *data*, which is what
matters: a mirror can be up while serving months-old OSM.

Usage
-----
    python3 tools/fetch_roads.py --list
    python3 tools/fetch_roads.py --theatre taiwan_strait
    python3 tools/fetch_roads.py --theatre india_china --bbox 78,32,81,36
    python3 tools/fetch_roads.py --all --dry-run

Standard library only: the target container has no pip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "engine", "data", "roads")

UA = "situation3d-roads-fetch/1.0 (offline situational awareness build)"
TIMEOUT_S = 300

# Ordered by observed availability. overpass-api.de is listed but expected to
# fail (406 at the Apache layer) until it recovers.
MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# Road classes worth routing on. Motorway/trunk carry the strategic movement;
# primary/secondary/tertiary are what actually exists in the border theatre.
ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary"]

# Theatre bounding boxes, west,south,east,north. Kept in sync with
# engine/scenario.py THEATRES.
#
# `route_bbox` and `classes` are deliberately narrower/coarser than the theatre
# box. Measured against the kumi mirror, the full Taiwan Strait box including
# tertiary roads is **113256 ways / 1327684 nodes / 56.9 MB** -- a routing graph
# nobody needs, and Dijkstra over it would dominate case generation. The land
# phase of a landing operation happens on the west-coast approaches, so the graph
# is clipped to that corridor and tertiary tracks are dropped:
# **19907 ways / 215871 nodes / 9.2 MB**. Ladakh is sparse enough
# (1296 ways / 2.7 MB) to keep in full. Both figures are recorded here because
# they are the reason for the narrowing, not an arbitrary preference.
THEATRES = {
    "taiwan_strait": {
        "bbox": (117.5, 22.0, 122.0, 26.5),
        "route_bbox": (118.0, 22.5, 120.5, 26.0),
        "classes": ["trunk", "primary", "secondary"],
        "note": "台海：海峡与两岸沿海路网（陆军阶段在西岸，故路由图只取西岸走廊）",
    },
    "india_china": {
        "bbox": (78.0, 32.0, 81.0, 36.0),
        "route_bbox": (78.0, 32.0, 81.0, 36.0),
        "classes": ["trunk", "primary", "secondary", "tertiary"],
        "note": "中印边境西段：拉达克/加勒万—班公湖（路网稀疏，全部保留）",
    },
}

# Typical free-flow speeds per class, km/h. Used by engine/roads.py for travel
# time. These are road-design speeds, not observed traffic speeds; the scenario
# applies its own derating for altitude and convoy behaviour.
CLASS_SPEED_KMH = {
    "motorway": 100.0,
    "trunk": 80.0,
    "primary": 60.0,
    "secondary": 45.0,
    "tertiary": 35.0,
}


def http_post(url: str, body: str, timeout: int = TIMEOUT_S) -> bytes:
    req = Request(url, data=body.encode("utf-8"),
                  headers={"User-Agent": UA, "Content-Type": "text/plain"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def build_query(bbox, classes) -> str:
    w, s, e, n = bbox
    pat = "|".join(classes)
    # `out body; >; out skel qt;` is the standard two-stage form: ways first,
    # then the nodes they reference, which avoids downloading whole city nodesets.
    return ("[out:json][timeout:%d];"
            "(way[\"highway\"~\"^(%s)$\"](%s,%s,%s,%s););"
            "out body; >; out skel qt;" % (TIMEOUT_S - 60, pat, s, w, n, e))


def fetch_raw(bbox, classes):
    """Try each mirror; return (payload_bytes, mirror, osm_base_utc)."""
    query = build_query(bbox, classes)
    errors = []
    for url in MIRRORS:
        try:
            raw = http_post(url, query)
            doc = json.loads(raw.decode("utf-8", "replace"))
        except (HTTPError, URLError, OSError, ValueError) as ex:
            errors.append("%s: %s" % (url, ex))
            continue
        if "elements" not in doc:
            errors.append("%s: no 'elements' in response" % url)
            continue
        osm_base = ((doc.get("osm3s") or {}).get("timestamp_osm_base") or "")
        return doc, url, osm_base
    raise RuntimeError("all Overpass mirrors failed:\n  " + "\n  ".join(errors))


def slim(doc, bbox, classes, theatre, mirror, osm_base) -> dict:
    """Turn an Overpass response into positional node array + way list."""
    way_nodes = {}          # way id -> [node id, ...]
    way_meta = {}
    needed = set()
    for el in doc.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        klass = tags.get("highway")
        if klass not in classes:
            continue
        refs = el.get("nodes") or []
        if len(refs) < 2:
            continue
        way_nodes[el["id"]] = refs
        way_meta[el["id"]] = {
            "class": klass,
            "ref": tags.get("ref") or "",
            # `oneway=yes|-1` matters for routing direction; anything else is
            # treated as bidirectional.
            "oneway": tags.get("oneway") in ("yes", "true", "1", "-1"),
            "reverse": tags.get("oneway") == "-1",
            "name": tags.get("name") or "",
        }
        needed.update(refs)

    coords = {}
    for el in doc.get("elements", []):
        if el.get("type") == "node" and el.get("id") in needed:
            lon, lat = el.get("lon"), el.get("lat")
            if lon is None or lat is None:
                continue
            # Keep only nodes inside the requested box: Overpass returns
            # referenced nodes that can sit well outside it for long ways.
            if not (bbox[0] - 0.05 <= lon <= bbox[2] + 0.05
                    and bbox[1] - 0.05 <= lat <= bbox[3] + 0.05):
                continue
            coords[el["id"]] = (round(lon, 6), round(lat, 6))

    index = {}
    nodes = []
    for nid, ll in coords.items():
        index[nid] = len(nodes)
        nodes.append([ll[0], ll[1]])

    ways = []
    for wid, refs in way_nodes.items():
        idx = [index[r] for r in refs if r in index]
        if len(idx) < 2:
            continue
        meta = way_meta[wid]
        ways.append({
            "id": wid,
            "class": meta["class"],
            "ref": meta["ref"],
            "name": meta["name"],
            "oneway": bool(meta["oneway"]),
            "reverse": bool(meta["reverse"]),
            "nodes": idx,
        })

    counts = {}
    for w in ways:
        counts[w["class"]] = counts.get(w["class"], 0) + 1

    return {
        "schema": "situation3d.roads.v1",
        "theatre": theatre,
        "bbox": list(bbox),
        "osm_base_utc": osm_base,
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mirror": mirror,
        "classes": list(classes),
        "class_speed_kmh": {k: CLASS_SPEED_KMH[k] for k in classes},
        "class_counts": counts,
        "node_count": len(nodes),
        "way_count": len(ways),
        "nodes": nodes,
        "ways": ways,
    }


def write_json(path: str, payload: dict) -> int:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return os.path.getsize(path)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="Fetch/slim OSM road networks")
    ap.add_argument("--theatre", default=None, help="theatre name from --list")
    ap.add_argument("--all", action="store_true", help="fetch every theatre")
    ap.add_argument("--bbox", default=None, help="override west,south,east,north")
    # default=None (not the full class list!) so each theatre's own `classes`
    # applies. Leaving a default here silently overrode every per-theatre set:
    # the Taiwan graph came back with 45105 ways / 18.7 MB including tertiary and
    # motorway instead of the intended 19907 / 9.2 MB.
    ap.add_argument("--classes", default=None,
                    help="comma list overriding every theatre's own class set")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report counts without writing")
    args = ap.parse_args(argv)

    if args.list:
        for name, spec in sorted(THEATRES.items()):
            print("%-16s %-28s %s" % (name, spec["bbox"], spec["note"]))
        return 0

    names = sorted(THEATRES) if args.all else ([args.theatre] if args.theatre else [])
    if not names:
        print("nothing to do: pass --theatre NAME, --all, or --list", file=sys.stderr)
        return 2
    for n in names:
        if n not in THEATRES:
            print("unknown theatre %r (see --list)" % n, file=sys.stderr)
            return 2

    # `--classes` defaults to None so each theatre can use its own set; an
    # explicit value still overrides every theatre.
    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        unknown = [c for c in classes if c not in CLASS_SPEED_KMH]
        if unknown:
            print("unknown road class(es): %s" % unknown, file=sys.stderr)
            return 2

    bbox_override = None
    if args.bbox:
        try:
            bbox_override = tuple(float(v) for v in args.bbox.split(","))
        except ValueError:
            print("--bbox must be west,south,east,north", file=sys.stderr)
            return 2
        if len(bbox_override) != 4:
            print("--bbox must have 4 values", file=sys.stderr)
            return 2

    os.makedirs(args.out, exist_ok=True)
    failures = []
    for name in names:
        spec = THEATRES[name]
        # Route graph, not theatre extent: see the note on THEATRES.
        bbox = bbox_override or spec.get("route_bbox") or spec["bbox"]
        th_classes = classes or spec.get("classes") or ROAD_CLASSES

        t0 = time.time()
        try:
            doc, mirror, osm_base = fetch_raw(bbox, th_classes)
            roads = slim(doc, bbox, th_classes, name, mirror, osm_base)
        except Exception as e:  # noqa: BLE001 - report, keep other theatres
            print("%-14s FAILED: %s" % (name, e), file=sys.stderr)
            failures.append(name)
            continue

        if roads["way_count"] == 0:
            print("%-14s FAILED: 0 ways in bbox %s" % (name, bbox), file=sys.stderr)
            failures.append(name)
            continue

        if args.dry_run:
            print("%-14s %5d ways  %7d nodes  %.1f MB (dry-run)"
                  % (name, roads["way_count"], roads["node_count"],
                     len(json.dumps(roads)) / 1048576.0))
            continue

        size = write_json(os.path.join(args.out, "%s.json" % name), roads)
        # Staleness is reported, not hidden: a mirror can be up while serving
        # months-old OSM, and route planning should know that.
        age = "unknown"
        if osm_base:
            try:
                dt = datetime.strptime(osm_base, "%Y-%m-%dT%H:%M:%SZ")
                age = "%.1f d" % ((datetime.utcnow() - dt).total_seconds() / 86400.0)
            except ValueError:
                pass
        print("%-14s %5d ways  %7d nodes  %6.1f MB  %.1fs  osm_base=%s (age %s)  %s"
              % (name, roads["way_count"], roads["node_count"], size / 1048576.0,
                 time.time() - t0, osm_base or "n/a", age, mirror))
        print("               classes: %s" % roads["class_counts"])

    if failures:
        print("FAILED theatres: %s" % ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))