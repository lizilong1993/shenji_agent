#!/usr/bin/env python3
"""Fetch and slim the open vector datasets, for offline use via WFS.

Why download-and-store
----------------------
The deployment must work without internet, and these sources are open-licensed
so storing them is legitimate (unlike the Google/ion tiles, which are explicitly
not redistributable -- see the note in web/src/buildings3d.js). So each dataset
is fetched once, trimmed to the theatre bounding box, stripped of fields nobody
renders, and committed. `engine/wfs.py` then serves them over OGC WFS.

Datasets
--------
  military  OSM military installations (Overpass). ODbL. Names/labels are kept
            because the point of the layer is *which* facility is where.
  cables    TeleGeography submarine cables + landing points. CC BY-NC-SA 3.0.
  places    Reference admin/boundary points, if a source is reachable.

Sources are keyed by name in `SOURCES` below with their licence, so the licence
travels with the data -- `engine/wfs.py` publishes it in GetCapabilities, which is
where ODbL attribution is expected to live.

Usage
-----
    python3 tools/fetch_vector.py --list
    python3 tools/fetch_vector.py                 # all sources
    python3 tools/fetch_vector.py --source military
    python3 tools/fetch_vector.py --dry-run       # report bbox extents only

Standard library only: the target container has no pip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "engine", "data", "vector")

# Theatre bounding box, WGS84 degrees: (west, south, east, north).
BBOX = (115.0, 18.0, 125.0, 29.0)

TIMEOUT_S = 180

# Overpass rejects requests without a UA and asks for a descriptive one.
UA = "situation3d-vector-fetch/1.0 (offline situational awareness build)"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# OSM tags that mark a military installation. Kept explicit rather than a single
# "military" key so the layer can say what kind of facility each point is.
MILITARY_FILTER = """
[out:json][timeout:{timeout}];
(
  node["military"]({s},{w},{n},{e});
  way["military"]({s},{w},{n},{e});
  relation["military"]({s},{w},{n},{e});
);
out center tags;
"""

SOURCES = {
    "military": {
        "kind": "osm-overpass",
        "licence": "ODbL 1.0 (OpenStreetMap contributors)",
        "licence_url": "https://www.openstreetmap.org/copyright",
        "attribution": "© OpenStreetMap contributors",
        "title": "军事设施（OSM military=*）",
        "description": "OpenStreetMap 中标注 military=* 的设施点位，裁剪到战区范围。",
    },
    "cables": {
        "kind": "telegeography-mirror",
        "licence": "CC BY-NC-SA 3.0 (TeleGeography)",
        "licence_url": "https://www.telegeography.com/",
        "attribution": "TeleGeography submarine cable map",
        "title": "海底光缆",
        "description": "TeleGeography 海底光缆走向，裁剪到战区范围。",
    },
}


def http_get(url: str, data: bytes = None, headers: dict = None,
             timeout: int = TIMEOUT_S) -> bytes:
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    req = Request(url, data=data, headers=h)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def in_bbox(lon: float, lat: float, pad: float = 0.0) -> bool:
    w, s, e, n = BBOX
    return (w - pad) <= lon <= (e + pad) and (s - pad) <= lat <= (n + pad)


def bbox_intersects(coords, pad: float = 0.5) -> bool:
    """True if any coordinate of a (possibly nested) geometry list is in the box."""
    if not coords:
        return False
    first = coords[0]
    if isinstance(first, (int, float)):
        return len(coords) >= 2 and in_bbox(coords[0], coords[1], pad)
    return any(bbox_intersects(c, pad) for c in coords)


# ---------------------------------------------------------------------------
# OSM military installations (Overpass)
# ---------------------------------------------------------------------------

def fetch_military() -> dict:
    w, s, e, n = BBOX
    query = MILITARY_FILTER.format(timeout=TIMEOUT_S - 40, s=s, w=w, n=n, e=e)
    last_err = None
    payload = None
    for url in OVERPASS_ENDPOINTS:
        try:
            # Overpass requires POST for a query body; GET returns 406 on these
            # mirrors, which is why this is not a plain urlopen(url).
            payload = http_get(url, data=query.encode("utf-8"),
                               headers={"Content-Type": "text/plain"})
            break
        except (HTTPError, URLError, OSError) as ex:
            last_err = "%s: %s" % (url, ex)
            continue
    if payload is None:
        raise RuntimeError("all Overpass mirrors failed (%s)" % last_err)

    raw = json.loads(payload.decode("utf-8", "replace"))
    features = []
    kinds = {}
    for el in raw.get("elements", []):
        tags = el.get("tags") or {}
        kind = tags.get("military")
        if not kind:
            continue
        if el["type"] == "node":
            lon, lat = el.get("lon"), el.get("lat")
        else:
            c = el.get("center") or {}
            lon, lat = c.get("lon"), c.get("lat")
        if lon is None or lat is None or not in_bbox(lon, lat):
            continue
        name = tags.get("name") or tags.get("name:zh") or tags.get("name:en") or ""
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            # Only the fields the layer renders. Dropping the rest is the
            # difference between ~150 KB and several MB, and the raw tags are
            # re-fetchable.
            "properties": {
                "name": name,
                "kind": kind,
                "osm_type": el["type"],
                "osm_id": el.get("id"),
                "operator": tags.get("operator") or "",
                "landuse": tags.get("landuse") or "",
            },
        })
        kinds[kind] = kinds.get(kind, 0) + 1

    return {
        "type": "FeatureCollection",
        "features": features,
        "_stats": {"kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1]))},
    }


# ---------------------------------------------------------------------------
# TeleGeography submarine cables
# ---------------------------------------------------------------------------

CABLE_ARCHIVE_URL = ("https://codeload.github.com/tbotnz/submarine-cables-geojson/"
                     "tar.gz/refs/heads/main")
CABLE_MEMBER = "cables.json"


def _slim_telegeography(raw: bytes, keep_props, geometry_types=None) -> dict:
    doc = json.loads(raw.decode("utf-8", "replace"))
    out = []
    for f in doc.get("features", []):
        geom = f.get("geometry") or {}
        if geometry_types and geom.get("type") not in geometry_types:
            continue
        if not bbox_intersects(geom.get("coordinates")):
            continue
        props = f.get("properties") or {}
        out.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {k: props.get(k) for k in keep_props if props.get(k) is not None},
        })
    return out


def fetch_cables(archive_path: str = None) -> dict:
    """Submarine cables from the mirror, trimmed to the bbox.

    `archive_path` lets a caller supply an already-downloaded tarball; without it
    the download is attempted directly (works from a machine that can reach
    GitHub, fails cleanly otherwise).
    """
    if archive_path:
        with open(archive_path, "rb") as fh:
            raw = fh.read()
    else:
        raw = http_get(CABLE_ARCHIVE_URL)

    import io
    import tarfile
    member = None
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and os.path.basename(m.name) == CABLE_MEMBER:
                member = m
                break
        if member is None:
            names = [m.name for m in tf.getmembers() if m.isfile()][:10]
            raise RuntimeError("no %s in archive (members: %s)" % (CABLE_MEMBER, names))
        data = tf.extractfile(member).read()

    features = _slim_telegeography(data, ("id", "name", "color"),
                                   ("MultiLineString", "LineString", "Point"))
    for f in features:
        f["properties"]["geom"] = f["geometry"]["type"]
    return {
        "type": "FeatureCollection",
        "features": features,
        "_stats": {"total": len(features)},
    }


# ---------------------------------------------------------------------------

FETCHERS = {
    "military": fetch_military,
    "cables": fetch_cables,
}


def write_json(path: str, payload: dict) -> int:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return os.path.getsize(path)


def write_manifest(out_dir: str, results: dict) -> str:
    """Write the WFS-facing manifest: one entry per feature type, with licence."""
    types = {}
    for name, res in results.items():
        src = SOURCES[name]
        types[name] = {
            "title": src["title"],
            "description": src["description"],
            "licence": src["licence"],
            "licence_url": src["licence_url"],
            "attribution": src["attribution"],
            "file": "%s.geojson" % name,
            "features": len(res.get("features", [])),
            "bytes": os.path.getsize(os.path.join(out_dir, "%s.geojson" % name)),
            "stats": res.get("_stats") or {},
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    manifest = {
        "schema": "situation3d.vector.v1",
        "bbox": list(BBOX),
        "crs": "EPSG:4326",
        "types": types,
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="Fetch/slim open vector datasets")
    ap.add_argument("--source", action="append", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report stats without writing")
    ap.add_argument("--cables-archive", default=None,
                    help="path to a pre-downloaded submarine-cables tarball "
                         "(for hosts that cannot reach GitHub)")
    args = ap.parse_args(argv)

    if args.list:
        for name, src in sorted(SOURCES.items()):
            print("%-10s %-14s %s" % (name, src["kind"], src["licence"]))
        return 0

    names = args.source or sorted(SOURCES)
    for n in names:
        if n not in SOURCES:
            print("unknown source %r (see --list)" % n, file=sys.stderr)
            return 2

    os.makedirs(args.out, exist_ok=True)
    results = {}
    failures = []
    for name in names:
        t0 = time.time()
        try:
            # TeleGeography's mirror lives on GitHub, which the target container
            # cannot reach; --cables-archive lets the tarball be fetched elsewhere
            # and handed in.
            if name == "cables":
                res = FETCHERS[name](archive_path=args.cables_archive)
            else:
                res = FETCHERS[name]()
        except Exception as e:  # noqa: BLE001 - report, keep other sources
            print("%-10s FAILED: %s" % (name, e), file=sys.stderr)
            failures.append(name)
            continue
        n = len(res.get("features", []))
        if not n:
            print("%-10s FAILED: 0 features in bbox %s" % (name, BBOX), file=sys.stderr)
            failures.append(name)
            continue
        if args.dry_run:
            print("%-10s %5d features  %.1f KB (dry-run, not written)"
                  % (name, n, len(json.dumps(res)) / 1024.0))
            continue
        size = write_json(os.path.join(args.out, "%s.geojson" % name), res)
        results[name] = res
        print("%-10s %5d features  %6.1f KB  %.1fs  %s"
              % (name, n, size / 1024.0, time.time() - t0, res.get("_stats")))

    if results and not args.dry_run:
        # Merge with any previously written types so re-fetching one source does
        # not drop the others from the manifest.
        merged = {}
        for name in sorted(SOURCES):
            p = os.path.join(args.out, "%s.geojson" % name)
            if name in results:
                merged[name] = results[name]
            elif os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        merged[name] = json.load(fh)
                except (OSError, ValueError):
                    continue
        mp = write_manifest(args.out, merged)
        print("manifest  : %s (%d types)" % (mp, len(merged)))

    if failures:
        print("FAILED sources: %s" % ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))