#!/usr/bin/env python3
"""Upgrade a legacy envpack directory to ``msgeo.envpack.v3``.

This is a P1 vertical-slice tool: it does not change the underlying layer
files, it copies them and writes a v3 manifest with source/license/confidence
metadata.  It exists so the new ``envmodel`` contract can be exercised on the
existing Taiwan pack before the real COG/GeoParquet pipeline is available.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone


def _copy_tree(src: str, dst: str) -> None:
    if os.path.isdir(src):
        # Manual walk/copyfile avoids NFS metadata errors from copytree.
        for root, _dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            target_dir = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(target_dir, exist_ok=True)
            for name in files:
                shutil.copyfile(
                    os.path.join(root, name), os.path.join(target_dir, name)
                )
    elif os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)


def upgrade(src: str, dst: str, force: bool = False) -> dict:
    src_real = os.path.realpath(src)
    dst_real = os.path.realpath(dst)
    if src_real == dst_real or dst_real.startswith(src_real + os.sep):
        raise SystemExit("refusing to write into the source directory")
    src_manifest_path = os.path.join(src, "manifest.json")
    if not os.path.isfile(src_manifest_path):
        raise SystemExit("source manifest not found: %s" % src_manifest_path)
    with open(src_manifest_path, encoding="utf-8") as fh:
        old = json.load(fh)

    if os.path.exists(dst) and not force:
        raise SystemExit("destination exists; pass --force to replace: %s" % dst)
    if force and os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    for name in ("terrain", "landcover", "water", "buildings", "roads"):
        _copy_tree(os.path.join(src, name), os.path.join(dst, name))
    bathy_rel = (old.get("layers") or {}).get("bathymetry", {}).get("path")
    has_bathy = bool(bathy_rel and os.path.isfile(os.path.join(src, bathy_rel)))
    if has_bathy:
        _copy_tree(
            os.path.join(src, bathy_rel), os.path.join(dst, bathy_rel)
        )
    basic = old.get("fallback_basic_json")
    if basic and os.path.isfile(os.path.join(src, basic)):
        _copy_tree(os.path.join(src, basic), os.path.join(dst, basic))

    resolution = float(old.get("resolution_m") or 100.0)
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "schema": "msgeo.envpack.v3",
        "pack_id": str(old.get("theatre") or "legacy-pack"),
        "version": "3.0-p1-slice",
        "theatre": old.get("theatre"),
        "bbox": old.get("bbox"),
        "crs": "EPSG:4326",
        "vertical_datum": "unknown",
        "tile_size_m": old.get("tile_size_m"),
        "resolution_m": resolution,
        "ele_grade": old.get("ele_grade"),
        "origin": old.get("origin"),
        "grid": old.get("grid"),
        "fallback_basic_json": basic,
        "layers": {
            "terrain": {
                "format": "grid-json",
                "path": "terrain/grid.json",
                "source": "legacy-basic",
                "license": "mixed",
                "license_id": "MIXED",
                "confidence": "low",
                "resolution_m": resolution,
                "notes": "P1 slice: legacy layer copied without re-projection",
            },
            "hydrography": {
                "format": "json",
                "path": "water/polygons.json",
                "source": "osm-water",
                "license": "ODbL-1.0",
                "license_id": "ODbL-1.0",
                "confidence": "low",
                "depth_kind": "synthetic" if has_bathy else "unknown",
                "notes": "depth from bathymetry grid when present" if has_bathy else "no depth field in legacy pack",
            },
            # Compatibility alias: EnvPackEnvironment reads the "water" key.
            "water": {
                "format": "json",
                "path": "water/polygons.json",
                "source": "osm-water",
                "license": "ODbL-1.0",
                "license_id": "ODbL-1.0",
                "confidence": "low",
                "depth_kind": "synthetic" if has_bathy else "unknown",
            },
            "landcover": {
                "format": "json",
                "path": "landcover/polygons.json",
                "source": "osm-landuse",
                "license": "ODbL-1.0",
                "license_id": "ODbL-1.0",
                "confidence": "medium",
            },
            "buildings": {
                "format": "json",
                "path": "buildings/buildings.json",
                "source": "osm-buildings",
                "license": "ODbL-1.0",
                "license_id": "ODbL-1.0",
                "confidence": "low",
                "height_kind": "default",
            },
            "roads": {
                "format": "json",
                "path": "roads/graph.json",
                "source": "osm-roads",
                "license": "ODbL-1.0",
                "license_id": "ODbL-1.0",
                "confidence": "medium",
            },
            "obstacles": {
                "format": "json",
                "path": "buildings/buildings.json",
                "source": "derived-from-buildings",
                "license": "ODbL-1.0",
                "license_id": "ODbL-1.0",
                "confidence": "low",
            },
        },
        "nav": {
            "global_res_m": 30.0,
            "local_res_m": 2.0,
            "graph_version": "p1-slice",
        },
        "gates": {
            "building": {"unknown_policy": "deny"},
            "water_depth": {"unknown_policy": "deny"},
            "slope": {"unknown_policy": "deny"},
            "road_limit": {"unknown_policy": "deny"},
            "bridge_mlc": {"unknown_policy": "advisory", "capacity_kind": "maxweight"},
            "tunnel_clearance": {"unknown_policy": "advisory", "capacity_kind": "maxheight"},
        },
        "provenance": {
            "built_at": built_at,
            "tool_versions": {"upgrade_envpack_v3": "0.1"},
            "source_hashes": {},
            "notes": "P1 vertical slice; not a production data package",
        },
        "license": {
            "spdx": "MIXED",
            "redistribute": False,
            "commercial_use": False,
            "attribution": "OpenStreetMap contributors / legacy pack",
        },
        "licenses": [
            {
                "spdx": "ODbL-1.0",
                "terms_url": "https://www.openstreetmap.org/copyright",
                "redistribute": True,
                "commercial_use": True,
                "attribution": "OpenStreetMap contributors",
            },
            {
                "spdx": "MIXED",
                "redistribute": False,
                "commercial_use": False,
                "attribution": "legacy pack",
            },
        ],
    }
    if has_bathy:
        manifest["layers"]["bathymetry"] = {
            "format": "grid-json",
            "path": bathy_rel,
            "source": "legacy-bathymetry",
            "license": "mixed",
            "license_id": "MIXED",
            "confidence": "low",
            "depth_kind": "synthetic",
            "notes": "GEBCO/legacy bathymetry grid; not navigation-grade",
        }
    if (old.get("provenance") or {}).get("building_heights"):
        manifest["layers"]["buildings"]["height_kind"] = "derived"
        manifest["layers"]["buildings"]["notes"] = (
            "height_source is per-feature: tag/levels/fallback/default"
        )
    with open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return {
        "out": dst,
        "pack_id": manifest["pack_id"],
        "layers": sorted(manifest["layers"]),
        "built_at": built_at,
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(upgrade(args.src, args.out, args.force), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
