#!/usr/bin/env python3
"""Smoke-check a v3 pack through the envmodel contract.

This is not a routing conformance test; it reports sample coverage, water
depth/quality, building height quality and LOS visibility counts.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envmodel import open_pack


def _centroid(ring):
    return (
        sum(p[0] for p in ring) / len(ring),
        sum(p[1] for p in ring) / len(ring),
    )


def check(pack: str, samples: int = 200, los_pairs: int = 200,
          seed: int = 20261024) -> dict:
    query = open_pack(pack)
    bbox = query.manifest.bbox
    rng = random.Random(seed)
    water = building = road = 0
    depths = []
    for _ in range(samples):
        lon = rng.uniform(bbox[0], bbox[2])
        lat = rng.uniform(bbox[1], bbox[3])
        w = query.water(lon, lat)
        if w.value["is_water"]:
            water += 1
            if w.value["depth_m"] is not None:
                depths.append(w.value["depth_m"])
        if query.building(lon, lat).value is not None:
            building += 1
        if query.road(lon, lat).value is not None:
            road += 1

    # Building height quality at footprint centroids.
    quality = collections.Counter()
    path = query.manifest.layer_path("buildings")
    if path:
        with open(path, encoding="utf-8") as fh:
            items = json.load(fh)
        for item in items:
            ring = item.get("ring") or []
            if len(ring) < 3:
                continue
            lon, lat = _centroid(ring)
            env = query.building(lon, lat)
            if env.value is not None:
                quality[env.quality] += 1

    visible_2 = visible_30 = 0
    for _ in range(los_pairs):
        a = (rng.uniform(bbox[0], bbox[2]), rng.uniform(bbox[1], bbox[3]))
        b = (rng.uniform(bbox[0], bbox[2]), rng.uniform(bbox[1], bbox[3]))
        if query.los(a, 2.0, b, 2.0).value["visible"]:
            visible_2 += 1
        if query.los(a, 30.0, b, 30.0).value["visible"]:
            visible_30 += 1

    return {
        "pack": query.manifest.pack_id,
        "version": query.manifest.version,
        "layers": sorted(query.manifest.layers),
        "samples": samples,
        "water_points": water,
        "water_depths": {
            "count": len(depths),
            "min": min(depths) if depths else None,
            "max": max(depths) if depths else None,
        },
        "building_points": building,
        "road_points": road,
        "building_height_quality": quality.most_common(),
        "los_pairs": los_pairs,
        "los_visible_2m": visible_2,
        "los_visible_30m": visible_30,
        "obstacles": query.obstacles().value.summary(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--los-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    print(json.dumps(
        check(args.pack, args.samples, args.los_pairs, args.seed),
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
