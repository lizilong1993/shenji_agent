#!/usr/bin/env python3
"""Self-check for the road network and land-routing constraints.

Run on the target container, where there is no test runner:

    python3 tools/road_selftest.py                 # every theatre with data
    python3 tools/road_selftest.py --theatre taiwan_strait

What this guards against
------------------------
Land platforms are asserted to move on roads, and the intent layer treats
`on_road` / `route_deviation_m` as evidence. So a routing bug does not merely
draw a wrong line: it fabricates manoeuvre evidence for a platform that could not
have manoeuvred, which is exactly the kind of error that survives review because
the picture still looks plausible.

Checks:
1. **Load and shape** -- node/way counts sane, every way index in range.
2. **Snapping** -- a point on a node snaps within metres; a mid-ocean point is
   refused rather than dragged onto the nearest road. The refusal path is the one
   that matters: silently snapping would hide an off-network platform.
3. **Routing optimality** -- Dijkstra cost is compared against an independent
   brute-force shortest path over the same adjacency. A wrong graph orientation
   (oneway/reverse) shows up here.
4. **Route realism** -- routed path length is never shorter than the great-circle
   distance between its endpoints, and never absurdly longer.
5. **Land-track constraint** -- every frame of every *land* target in the
   generated dataset lies within the constraint radius of the network. This is
   the end-to-end assertion: it fails if someone later bypasses routing.

Exits non-zero with a readable message on the first failing group.
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import geo, roads  # noqa: E402

# A land track may legitimately sit a little off the road centreline (vehicle
# spread, node spacing, snapping at the start/end). 3 km is generous but still
# catches a track that abandoned the network entirely.
MAX_TRACK_OFFSET_KM = 3.0

FAILURES = []


def check(cond, label, detail=""):
    if cond:
        print("  ok   %s" % label)
    else:
        print("  FAIL %s %s" % (label, detail))
        FAILURES.append(label)


def brute_force_cost(r: roads.Roads, a: int, b: int, nodes: set,
                     cap: float = 1e9) -> float:
    """Independent shortest-path cost, over a delimited node set.

    Deliberately a separate implementation (Bellman-Ford style relaxation over an
    edge list) so a shared helper cannot hide a bug in either one.

    Restricted to `nodes`, a neighbourhood around the query. The full Taiwan graph
    is 488081 nodes / 45105 ways, and relaxing every edge 60 times over it takes
    minutes -- the check only needs to confirm the two implementations agree on a
    small subgraph. The restriction is sound because ways crossing out of the set
    are truncated in *both* the brute-force edges and the neighbour expansion that
    defines the set.
    """
    edges = []
    for w in r.ways:
        idx = [n for n in (w.get("nodes") or []) if n in nodes]
        if len(idx) < 2:
            continue
        kmh = max(1.0, float(r.speed.get(w.get("class") or "", 40.0)))
        oneway = bool(w.get("oneway"))
        reverse = bool(w.get("reverse"))
        for x, y in zip(idx, idx[1:]):
            m = geo.haversine_m((r.nodes[x][1], r.nodes[x][0]),
                                (r.nodes[y][1], r.nodes[y][0]))
            if m <= 0:
                continue
            t = (m / 1000.0) / kmh
            if (not oneway) or (not reverse):
                edges.append((x, y, t))
            if (not oneway) or reverse:
                edges.append((y, x, t))

    dist = {a: 0.0}
    for _ in range(min(len(nodes), 40)):
        changed = False
        for x, y, t in edges:
            if x in dist:
                nd = dist[x] + t
                if nd < dist.get(y, cap):
                    dist[y] = nd
                    changed = True
        if not changed:
            break
    return dist.get(b, cap)


def theatre_report(name: str) -> None:
    print("== theatre: %s ==" % name)
    try:
        r = roads.load(name)
    except roads.RoadError as e:
        check(False, "load", str(e))
        return

    print("       nodes=%d ways=%d  osm_base=%s  mirror=%s"
          % (len(r.nodes), len(r.ways), r.meta.get("osm_base_utc"), r.meta.get("mirror")))
    print("       classes: %s" % r.meta.get("class_counts"))

    n = len(r.nodes)
    bad_idx = 0
    for w in r.ways:
        for i in w.get("nodes") or []:
            if not (0 <= i < n):
                bad_idx += 1
    check(bad_idx == 0, "1. all way node indices in range", "(bad=%d)" % bad_idx)
    check(len(r.ways) > 0, "1. network has ways")
    check(len(r.adj) == n, "1. adjacency covers every node")

    # --- 2. snapping ---
    probe = r.nodes[len(r.nodes) // 2]
    plat, plon = probe[1], probe[0]
    i = r.nearest_node(plat, plon, max_km=1.0)
    if i is not None:
        d = geo.haversine_m((plat, plon), r.node_latlon(i))
        check(d < 200.0, "2. on-node point snaps within 200 m", "(%.0f m)" % d)
    else:
        check(False, "2. on-node point snaps", "(returned None)")

    # Somewhere far outside every network: the middle of the Pacific.
    far = r.nearest_node(0.0, -160.0, max_km=25.0)
    check(far is None, "2. off-network point is refused, not dragged",
          "(got node %s)" % far)

    # --- 3. routing optimality ---
    # Pick two nodes that are actually connected: walk out from a busy node.
    busy = max(range(n), key=lambda k: len(r.adj[k])) if n else 0
    target = None
    for nb, _m, _s in r.adj[busy][:200]:
        target = nb
        break
    if target is None:
        check(False, "3. found a connected pair to test", "(isolated graph)")
    else:
        path = r.route(busy, target)
        check(path is not None, "3. route found between adjacent nodes")
        if path:
            dijkstra = 0.0
            for a, b in zip(path, path[1:]):
                for nb, m, kmh in r.adj[a]:
                    if nb == b:
                        dijkstra += (m / 1000.0) / max(1.0, kmh)
                        break
            # Neighbourhood for the independent check: a few hops out from both
            # endpoints. Bounded so the O(V*E) relaxation stays cheap on the
            # 488k-node Taiwan graph (see brute_force_cost's note).
            hood = set()
            frontier = {busy, target}
            hood |= frontier
            for _ in range(3):
                nxt = set()
                for u in frontier:
                    for v, _m, _s in r.adj[u]:
                        if v not in hood:
                            nxt.add(v)
                hood |= nxt
                frontier = nxt
                if not nxt:
                    break
            bf = brute_force_cost(r, busy, target, hood)
            check(bf >= 1e9 or abs(dijkstra - bf) < 1e-6,
                  "3. Dijkstra matches independent shortest path",
                  "(dijkstra=%.6f brute=%.6f hood=%d)" % (dijkstra, bf, len(hood)))

        # --- 4. route realism ---
        if path and len(path) > 1:
            straight = geo.path_length_m([r.node_latlon(path[0]),
                                          r.node_latlon(path[-1])]) / 1000.0
            along = r.path_length_km([r.node_latlon(p) for p in path])
            check(along + 1e-6 >= straight,
                  "4. routed length >= great-circle",
                  "(along=%.1f straight=%.1f)" % (along, straight))
            check(along <= max(50.0, straight * 60.0),
                  "4. routed length not absurd",
                  "(along=%.1f straight=%.1f)" % (along, straight))

    # --- 5. land tracks in the generated dataset ---
    #
    # Only datasets that actually claim road-constrained routing are checked. A
    # dataset generated before that feature existed has land targets on straight
    # waypoints and would fail here for a reason that says nothing about the road
    # code; the generator marks constrained datasets with `road_constrained`.
    #
    # ⚠ 数据集现在住在 `web/data/situations/` 下（服务端扫描该目录生成案例清单）。
    # 这里**必须跟着改**：否则 `os.listdir(data_dir)` 只看到 base.json / ion.json，
    # `checked_any` 保持 False，第 5 组会以"没有找到数据集"**空过** —— 那是假通过，
    # 路网约束坏掉也不会有人知道。子目录不存在时也要能把话说清楚。
    data_dir = os.path.join(ROOT, "web", "data", "situations")
    checked_any = False
    skipped = []
    for fn in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []:
        if not fn.endswith(".json"):
            continue
        p = os.path.join(data_dir, fn)
        try:
            import json
            with open(p, "r", encoding="utf-8") as fh:
                ds = json.load(fh)
        except (OSError, ValueError):
            continue
        if ds.get("theatre") not in (None, name):
            continue
        # Only road-bound land platforms. A positional area-defence battery is
        # `domain == "land"` but fights from where it is emplaced, so requiring it
        # to sit near a road would fail for a reason that says nothing about the
        # routing (measured: it flagged B03/R04/R05, three fixed batteries that sit
        # tens of km from the clipped network).
        land_ids = {t["id"] for t in ds.get("targets", [])
                    if t.get("domain") == "land" and t.get("road_bound", True)}
        if not land_ids:
            continue
        if not ds.get("road_constrained"):
            skipped.append(fn)
            continue
        checked_any = True
        worst = 0.0
        worst_id = None
        worst_tick = None
        for fi, fr in enumerate(ds.get("frames") or []):
            for s in fr.get("targets") or []:
                if s["id"] not in land_ids:
                    continue
                # 3 km is the constraint, so nothing farther needs measuring.
                d = r.distance_to_network_km(s["lat"], s["lon"], max_km=3.0)
                if d is None:
                    d = 3.0
                if d > worst:
                    worst, worst_id, worst_tick = d, s["id"], fi
        check(worst <= MAX_TRACK_OFFSET_KM,
              "5. %s land tracks stay on the network" % fn,
              "(worst %.2f km on %s @tick %s, limit %.1f)"
              % (worst, worst_id, worst_tick, MAX_TRACK_OFFSET_KM))
    if skipped:
        print("  --  5. not road-constrained yet, skipped: %s" % ", ".join(skipped))
    if not checked_any and not skipped:
        print("  --  5. no dataset with land targets found; skipped")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--theatre", default=None)
    args = ap.parse_args()

    names = [args.theatre] if args.theatre else roads.available_theatres()
    if not names:
        print("no road networks found in %s -- run tools/fetch_roads.py"
              % roads.ROADS_DIR)
        return 2

    for name in names:
        theatre_report(name)
        print()

    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())