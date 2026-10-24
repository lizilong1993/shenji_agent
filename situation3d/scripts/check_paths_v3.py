#!/usr/bin/env python3
"""Path conformance smoke test for a v3 pack.

Samples passable start/goal pairs, plans with the existing 30 m NavField, checks
route_ok, and samples route points through the gate contract.  This is a
conformance harness, not a replacement for the future 10k-route regression.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envmodel import open_pack
from mobility.nav_field import NavField
from mobility.profile import profile_for


def check(pack: str, routes: int, seed: int, sample_m: float = 50.0) -> dict:
    query = open_pack(pack)
    bbox = query.manifest.bbox
    profile = profile_for("vehicle", "land")
    t0 = time.time()
    nav = NavField(query.env, res_m=30.0)
    build_s = time.time() - t0
    rng = random.Random(seed)
    nodes, _edges = query.env.road_graph()
    node_points = [
        (float(v[0]), float(v[1])) for v in nodes.values()
    ]
    rng.shuffle(node_points)
    passable_nodes = []
    for point in node_points[:4000]:
        if query.passability(profile, *point).value["allowed"]:
            passable_nodes.append(point)
        if len(passable_nodes) >= 500:
            break

    planned = 0
    no_path = 0
    route_ok_n = 0
    gate_denied_routes = 0
    gate_denied_points = 0
    gate_denial_reasons = collections.Counter()
    sampled_points = 0
    total_m = 0.0
    plan_times = []
    attempts = 0
    max_attempts = max(routes * 10, 100)

    while planned + no_path < routes and attempts < max_attempts:
        attempts += 1
        if len(passable_nodes) >= 2:
            start, goal = rng.sample(passable_nodes, 2)
        else:
            start = (rng.uniform(bbox[0], bbox[2]), rng.uniform(bbox[1], bbox[3]))
            goal = (rng.uniform(bbox[0], bbox[2]), rng.uniform(bbox[1], bbox[3]))
            if not query.passability(profile, *start).value["allowed"]:
                continue
            if not query.passability(profile, *goal).value["allowed"]:
                continue
        t0 = time.time()
        route = nav.plan(profile, start, goal, safe_fallback=True)
        plan_times.append(time.time() - t0)
        if route is None:
            no_path += 1
            continue
        planned += 1
        total_m += route.total_m
        ok = bool(nav.route_ok(profile, route))
        route_ok_n += 1 if ok else 0
        denied = 0
        s = 0.0
        while s <= route.total_m:
            pose = route.pose_at(s)
            sampled_points += 1
            gate = query.passability(profile, pose.lon, pose.lat)
            if not gate.value["allowed"]:
                denied += 1
                gate_denial_reasons.update(gate.value["reason_codes"])
            s += sample_m
        gate_denied_points += denied
        if denied:
            gate_denied_routes += 1

    plan_times.sort()
    p95 = plan_times[int(0.95 * (len(plan_times) - 1))] if plan_times else None
    return {
        "pack": query.manifest.pack_id,
        "nav_build_s": round(build_s, 3),
        "requested_routes": routes,
        "attempts": attempts,
        "planned": planned,
        "no_path": no_path,
        "route_ok": route_ok_n,
        "route_ok_rate": round(route_ok_n / planned, 4) if planned else None,
        "gate_denied_routes": gate_denied_routes,
        "gate_denied_points": gate_denied_points,
        "sampled_points": sampled_points,
        "gate_denied_rate": (
            round(gate_denied_points / sampled_points, 6) if sampled_points else None
        ),
        "gate_denial_reasons": gate_denial_reasons.most_common(),
        "mean_length_m": round(total_m / planned, 1) if planned else None,
        "plan_p95_s": round(p95, 4) if p95 is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True)
    parser.add_argument("--routes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--sample-m", type=float, default=50.0)
    args = parser.parse_args()
    print(json.dumps(
        check(args.pack, args.routes, args.seed, args.sample_m),
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
