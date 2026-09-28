#!/usr/bin/env python3
"""Normalise web/tiles/manifest.json after all fetches have run.

The tile set is built by more than one command -- the theatre-wide fetch, then
the urban AOI fetch -- and each writes only the part it knows about. This step
merges them into the single shape the front end reads, so `assets.js` never has
to guess a zoom range or fall back to a hard-coded bounding box.

Ownership is explicit, because getting it wrong is not obvious downstream:
`bbox`, `imagery`, `terrain` and `wsf3d` describe the *theatre-wide* fetch and
are written authoritatively here. `urban_*` describes the urban AOI fetch and is
preserved. A secondary fetch that overwrote the theatre keys would leave the
basemap provider claiming to cover one city, which reads as "the map is broken"
rather than "the manifest is wrong".

It is idempotent: running it twice changes nothing.

Usage (from the repository root, on the container):
    python3 situation3d/tools/finalize_manifest.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TILES = os.path.join(ROOT, "web", "tiles")
MANIFEST = os.path.join(TILES, "manifest.json")

# Matches DEFAULT_BBOX in fetch_assets.py: the theatre-wide coverage.
DEFAULT_BBOX = [115.0, 18.0, 125.0, 29.0]
THEATRE_MAX_ZOOM = {"imagery": 11, "terrain": 10, "wsf3d": 10}
URBAN = {"wsf3d_zoom": 13, "imagery_min": 12, "imagery_max": 14}


def count_tiles(root, family):
    """Number of tiles actually present for one family."""
    base = os.path.join(root, family)
    if not os.path.isdir(base):
        return 0
    n = 0
    for _dirpath, _dirnames, filenames in os.walk(base):
        for f in filenames:
            if f.endswith(".part") or f.endswith(".absent"):
                continue
            n += 1
    return n


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    # Walking ~10k tiles over NFS takes minutes, so counting is opt-in and the
    # previous count is carried forward when it is not requested.
    ap.add_argument("--count", action="store_true",
                    help="recount tiles (slow on the NFS-backed remote)")
    ap.add_argument("--bbox", default=",".join(str(v) for v in DEFAULT_BBOX),
                    help="theatre-wide west,south,east,north")
    ap.add_argument("--max-zoom", type=int, default=None,
                    help="override the theatre-wide imagery/terrain/wsf3d max zoom")
    ap.add_argument("--theatre", default=None,
                    help="record this bbox under `theatres[NAME]` so the front end "
                         "can switch coverage per case (e.g. taiwan_strait, "
                         "india_china)")
    args = ap.parse_args(argv[1:])
    args.bbox = [float(v) for v in args.bbox.split(",")]
    if len(args.bbox) != 4:
        print("ERROR: --bbox needs west,south,east,north", file=sys.stderr)
        return 2

    if not os.path.exists(MANIFEST):
        print("ERROR: %s not found -- run tools/fetch_assets.py first" % MANIFEST,
              file=sys.stderr)
        return 1

    with open(MANIFEST, "r", encoding="utf-8") as fh:
        man = json.load(fh)

    before = json.dumps(man, sort_keys=True)

    # Theatre-wide coverage is authoritative: whatever a previous run left here
    # is replaced, not merged.
    man["bbox"] = list(args.bbox)
    for family, max_zoom in THEATRE_MAX_ZOOM.items():
        entry = man.setdefault(family, {})
        entry["min_zoom"] = 0
        entry["max_zoom"] = args.max_zoom if args.max_zoom is not None else max_zoom

    # Urban coverage is produced by tools/fetch_urban_wsf3d.sh, which runs with
    # --no-manifest so it cannot overwrite the theatre-wide entries above.
    man.setdefault("urban_wsf3d", {
        "min_zoom": URBAN["wsf3d_zoom"],
        "max_zoom": URBAN["wsf3d_zoom"],
    })
    man.setdefault("urban_imagery", {
        "min_zoom": URBAN["imagery_min"],
        "max_zoom": URBAN["imagery_max"],
    })

    if args.count:
        man["tile_counts"] = {
            "imagery": count_tiles(TILES, "imagery"),
            "terrain": count_tiles(TILES, "terrain"),
            "wsf3d": count_tiles(TILES, os.path.join("wsf3d", "buildingheight")),
        }
        man["tile_count"] = sum(man["tile_counts"].values())

    # Per-theatre coverage. The two theatres are disjoint (Taiwan Strait and the
    # India-China western sector), so a single `bbox` cannot describe both: one
    # box spanning them covers most of Asia, and the front end would then declare
    # regional imagery across a rectangle with no tiles and chase 404s.
    # `--theatre NAME` records this run's box under that name; the front end reads
    # `theatres` and falls back to a built-in table when it is absent.
    if args.theatre:
        theatres = man.setdefault("theatres", {})
        prev_t = theatres.get(args.theatre) or {}
        cov = dict(prev_t.get("coverage") or {})
        for family in ("imagery", "terrain", "wsf3d"):
            if family in man:
                cov[family] = {"min_zoom": man[family].get("min_zoom", 0),
                               "max_zoom": man[family].get("max_zoom")}
        theatres[args.theatre] = {"bbox": list(args.bbox), "coverage": cov}
        man["coverage_note"] = (
            (man.get("coverage_note") or "")
            + " 战区 %s bbox=%s。" % (args.theatre, list(args.bbox)))
    man["coverage_note"] = (
        "影像 z0-%d 覆盖 bbox；高程 z0-%d；WSF3D z0-%d。城区（URBAN_AOI）"
        "另有影像 z%d-%d 与 WSF3D z%d。前端 provider 的 rectangle 取自这里，"
        "声明之外不再发请求。"
        % (man["imagery"]["max_zoom"], man["terrain"]["max_zoom"],
           man["wsf3d"]["max_zoom"], man["urban_imagery"]["min_zoom"],
           man["urban_imagery"]["max_zoom"], man["urban_wsf3d"]["max_zoom"])
    )

    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)

    changed = json.dumps(man, sort_keys=True) != before
    print("bbox          : %s" % man["bbox"])
    print("imagery       : z%s-%s" % (man["imagery"]["min_zoom"], man["imagery"]["max_zoom"]))
    print("terrain       : z%s-%s" % (man["terrain"]["min_zoom"], man["terrain"]["max_zoom"]))
    print("wsf3d         : z%s-%s" % (man["wsf3d"]["min_zoom"], man["wsf3d"]["max_zoom"]))
    print("urban imagery : z%s-%s" % (man["urban_imagery"]["min_zoom"], man["urban_imagery"]["max_zoom"]))
    print("urban wsf3d   : z%s-%s" % (man["urban_wsf3d"]["min_zoom"], man["urban_wsf3d"]["max_zoom"]))
    if "tile_counts" in man:
        print("tile counts   : %s (total %d)"
              % (man["tile_counts"], man["tile_count"]))
    print("written       : %s%s" % (MANIFEST, "" if changed else "  (already normalised)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
