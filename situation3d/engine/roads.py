"""Road-network routing for land platforms, standard library only.

Why this is not just a convenience
----------------------------------
Land platforms follow roads. In the Ladakh theatre the network is sparse enough
(120 primary ways across the whole sector) that "off-road" movement is not a
simplification but a fiction. A track that ignores the network also corrupts the
intent judgement downstream: `REDEPLOY` / `ASSEMBLE` / `DISPERSE` are only
credible if the platform could physically have got there.

So the network does two jobs: it shapes the generated tracks, and it feeds the
intent feature layer (`on_road`, `route_deviation_m`, `road_class_code`) so a
manoeuvre intent is never asserted for a platform that never left the road.

Data model
----------
`tools/fetch_roads.py` writes `engine/data/roads/<theatre>.json` with nodes as a
**positional [lon, lat] array** and ways referencing node indices. This module
builds an adjacency list over those indices.

Coordinate convention, stated once and then never ambiguous: the JSON stores
`[lon, lat]` (GeoJSON order), while this engine's `geo` helpers and every
`(lat, lon)` tuple in `scenario.py` use **lat first**. `nearest_node()` returns
`(lat, lon)`; `node_ll()` returns `[lon, lat]`. Mixing these up silently mirrors a
route across the equator, so the conversion is done in exactly two places.

Only the standard library is used: the target container has python3.8 and no pip.
"""

from __future__ import annotations

import heapq
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import geo

HERE = os.path.dirname(os.path.abspath(__file__))
ROADS_DIR = os.path.join(HERE, "data", "roads")

# Snap radius. Beyond this the caller is told the point is not on the network
# rather than being silently dragged onto a road kilometres away.
MAX_SNAP_KM = 25.0

# Default spacing for densified routes: one vertex roughly every 3 km, which at
# 30 s ticks and 60 km/h is under one tick of travel and keeps tracks smooth
# without exploding the frame payload.
DEFAULT_VERTEX_SPACING_M = 3000.0

# Sentinel for the distance cache, so a cached `None` ("no road within max_km")
# is distinguishable from a cache miss.
_MISS = object()


class RoadError(ValueError):
    """Road data missing, malformed, or a route that cannot be satisfied."""


def network_path(theatre: str) -> str:
    return os.path.join(ROADS_DIR, "%s.json" % theatre)


def available_theatres() -> List[str]:
    if not os.path.isdir(ROADS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(ROADS_DIR)
                  if f.endswith(".json") and not f.endswith(".tmp.json"))


class Roads(object):
    """A loaded road network with routing."""

    __slots__ = ("theatre", "nodes", "ways", "adj", "speed", "meta",
                 "_grid", "_cell", "comp", "comp_sizes", "_by_comp", "_dist_cache")

    def __init__(self, doc: dict, grid_deg: float = 0.25):
        if doc.get("schema") != "situation3d.roads.v1":
            raise RoadError("unexpected road schema: %r" % doc.get("schema"))
        self.theatre = doc.get("theatre") or "?"
        self.nodes = doc.get("nodes") or []      # [[lon, lat], ...]
        self.ways = doc.get("ways") or []
        self.meta = {
            "bbox": doc.get("bbox"),
            "osm_base_utc": doc.get("osm_base_utc"),
            "fetched_utc": doc.get("fetched_utc"),
            "mirror": doc.get("mirror"),
            "class_counts": doc.get("class_counts") or {},
        }
        self.speed = doc.get("class_speed_kmh") or {}
        if not self.nodes:
            raise RoadError("road network for %s has no nodes" % self.theatre)

        # Adjacency: node index -> [(neighbour index, metres, km/h), ...]
        adj = [[] for _ in range(len(self.nodes))]
        for w in self.ways:
            idx = w.get("nodes") or []
            kmh = float(self.speed.get(w.get("class") or "", 40.0))
            oneway = bool(w.get("oneway"))
            reverse = bool(w.get("reverse"))
            for a, b in zip(idx, idx[1:]):
                m = _dist_m(self.nodes[a], self.nodes[b])
                if m <= 0.0:
                    continue
                # oneway=-1 means the way runs against its node order, so the
                # traversable direction is reversed.
                fwd = (not oneway) or (not reverse)
                bwd = (not oneway) or reverse
                if fwd:
                    adj[a].append((b, m, kmh))
                if bwd:
                    adj[b].append((a, m, kmh))
        self.adj = adj

        # Weakly-connected components. Road networks along a disputed border are
        # frequently *not* contiguous in OSM: the India-China graph splits into 52
        # components whose largest holds only 35.5% of nodes, with Leh (comp 13),
        # Chushul/Galwan (comp 16) and Depsang/DBO (comp 43) in three separate
        # ones even though they are physically joined on the ground. Routing
        # therefore has to know which component a node belongs to, or a
        # geometrically nearest snap can land in a component with no path to the
        # destination and the whole leg fails for a reason that looks like a data
        # gap but is really a snapping choice.
        comp = [-1] * len(self.nodes)
        cid = 0
        for s in range(len(self.nodes)):
            if comp[s] != -1:
                continue
            comp[s] = cid
            stack = [s]
            while stack:
                u = stack.pop()
                for v, _m, _k in adj[u]:
                    if comp[v] == -1:
                        comp[v] = cid
                        stack.append(v)
            cid += 1
        self.comp = comp
        self.comp_sizes = {}
        for c in comp:
            self.comp_sizes[c] = self.comp_sizes.get(c, 0) + 1
        # component -> member node indices, for snapping into a given component
        self._by_comp = {}
        for i, c in enumerate(comp):
            self._by_comp.setdefault(c, []).append(i)
        # Cache for `distance_to_network_km`. Feature extraction asks the same
        # question for every (target, frame) and again when predicting, and each
        # answer walks thousands of grid nodes; on the India graph an uncached
        # sampling pass took 223 s. Coordinates are rounded to ~11 m, which is far
        # below the tolerance these distances are used at.
        self._dist_cache = {}

        # Uniform grid for nearest-node lookup; a linear scan over 10^5 nodes per
        # query would dominate case generation.
        self._cell = grid_deg
        grid: Dict[Tuple[int, int], List[int]] = {}
        for i, ll in enumerate(self.nodes):
            grid.setdefault(self._key(ll[1], ll[0]), []).append(i)
        self._grid = grid

    # -- geometry helpers -------------------------------------------------

    def _key(self, lat: float, lon: float) -> Tuple[int, int]:
        return (int(math.floor(lat / self._cell)), int(math.floor(lon / self._cell)))

    def node_ll(self, i: int) -> List[float]:
        """Node as [lon, lat] (GeoJSON order)."""
        return self.nodes[i]

    def node_latlon(self, i: int) -> Tuple[float, float]:
        """Node as (lat, lon) — the convention the rest of the engine uses."""
        n = self.nodes[i]
        return (n[1], n[0])

    def nearest_node(self, lat: float, lon: float,
                     max_km: float = MAX_SNAP_KM) -> Optional[int]:
        """Index of the nearest node, searching outward through grid cells.

        Returns None when the nearest node is farther than `max_km`. That
        distinction matters: a caller that gets a node knows the point was
        snapped onto the network, and one that gets None knows the point is
        genuinely off-network rather than being quietly moved tens of km.
        """
        k = self._key(lat, lon)
        best = None
        best_m = None
        # Search radius is bounded up front by the snap limit rather than
        # expanding ring by ring, so the work is (2*ceil(max_km/cell)+1)^2 cells
        # no matter how dense the network is.
        rings = int(math.ceil((max_km / 111.0) / self._cell)) + 1
        limit_m = max_km * 1000.0
        for dlat in range(-rings, rings + 1):
            for dlon in range(-rings, rings + 1):
                for i in self._grid.get((k[0] + dlat, k[1] + dlon), ()):
                    m = _dist_m([lon, lat], self.nodes[i])
                    if m < limit_m and (best_m is None or m < best_m):
                        best, best_m = i, m
        return best

    def distance_to_network_km(self, lat: float, lon: float,
                               max_km: float = 200.0) -> Optional[float]:
        """Distance to the nearest node in km, or None if nothing within `max_km`.

        `max_km` genuinely bounds the search, because it is passed straight to
        `nearest_node`. That matters when the caller only wants to know whether a
        point is inside a known threshold: the Taiwan graph has 488081 nodes and a
        200 km search inspects far more of them than a 3 km one, which is the
        difference between minutes and seconds on a whole-dataset sweep.
        """
        key = (round(lat, 4), round(lon, 4), max_km)
        hit = self._dist_cache.get(key, _MISS)
        if hit is not _MISS:
            return hit
        i = self.nearest_node(lat, lon, max_km=max_km)
        if i is None:
            out = None
        else:
            out = _dist_m([lon, lat], self.nodes[i]) / 1000.0
        self._dist_cache[key] = out
        return out

    # -- routing ----------------------------------------------------------

    def route(self, a: int, b: int) -> Optional[List[int]]:
        """Shortest-time node path from a to b (Dijkstra), or None if unreachable."""
        if a == b:
            return [a]
        # Cost is travel time so a fast trunk road beats a slightly shorter
        # secondary; the scenario moves at road speeds, so time is the right metric.
        dist = {a: 0.0}
        prev = {}
        pq = [(0.0, a)]
        seen = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == b:
                break
            for v, m, kmh in self.adj[u]:
                if v in seen:
                    continue
                kmh = kmh if kmh > 1.0 else 1.0
                nd = d + (m / 1000.0) / kmh
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if b not in dist:
            return None
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def route_latlon(self, waypoints: Sequence[Tuple[float, float]],
                     max_snap_km: float = MAX_SNAP_KM) -> List[Tuple[float, float]]:
        """Snap and route a whole waypoint list, returning (lat, lon) vertices.

        Raises RoadError when a waypoint cannot be snapped or a leg is
        unreachable — a silently-failed route would put a land platform at sea.
        """
        if not waypoints:
            return []
        # First snap freely, then resolve components. Snapping each waypoint
        # independently to "the nearest node" is not enough on a fragmented
        # graph: the nearest node to a point on the Indus valley road can belong
        # to a component that never connects to the next waypoint's component,
        # and the leg then fails as if there were no road at all.
        free = []
        for (lat, lon) in waypoints:
            i = self.nearest_node(lat, lon, max_km=max_snap_km)
            if i is None:
                raise RoadError(
                    "waypoint (%.4f, %.4f) is more than %.0f km from any road in "
                    "the %s network" % (lat, lon, max_snap_km, self.theatre))
            free.append(i)

        # Pick the component that covers the most waypoints; that is the network
        # the route actually travels on. Requiring a single component for *every*
        # waypoint would reject a route whose start merely sits nearer a stray
        # side road.
        counts: Dict[int, int] = {}
        for i in free:
            c = self.comp[i]
            counts[c] = counts.get(c, 0) + 1
        best_comp = max(counts, key=lambda c: (counts[c], self.comp_sizes.get(c, 0)))

        nodes = []
        for (lat, lon) in waypoints:
            i = self.nearest_node_in_component(lat, lon, best_comp, max_km=max_snap_km)
            if i is None:
                # The waypoint itself may sit closer to a stray component; retry
                # with its own snap so the failure reports the real reason.
                own = self.comp[free[len(nodes)]]
                raise RoadError(
                    "component mismatch in the %s network: no road within %.0f km of "
                    "(%.4f, %.4f) in component %d (route is on component %d, %d nodes)"
                    % (self.theatre, max_snap_km, lat, lon, best_comp, best_comp,
                       self.comp_sizes.get(best_comp, 0))
                    + (" [nearest node is in component %d]" % own
                       if own != best_comp else ""))
            nodes.append(i)

        out = [self.node_latlon(nodes[0])]
        for a, b in zip(nodes, nodes[1:]):
            if a == b:
                continue
            path = self.route(a, b)
            if path is None:
                raise RoadError(
                    "no road route from %s to %s in the %s network"
                    % (self.node_latlon(a), self.node_latlon(b), self.theatre))
            for i in path[1:]:
                out.append(self.node_latlon(i))
        return out

    def nearest_node_in_component(self, lat: float, lon: float, comp_id: int,
                                  max_km: float = MAX_SNAP_KM) -> Optional[int]:
        """Nearest node belonging to a specific component, else None."""
        members = self._by_comp.get(comp_id)
        if not members:
            return None
        best = None
        best_m = None
        limit_m = max_km * 1000.0
        for i in members:
            m = _dist_m([lon, lat], self.nodes[i])
            if m < limit_m and (best_m is None or m < best_m):
                best, best_m = i, m
        return best

    def path_length_km(self, path: Sequence[Tuple[float, float]]) -> float:
        return geo.path_length_m(path) / 1000.0

    def path_speed_kmh(self, path: Sequence[Tuple[float, float]]) -> float:
        """Mean free-flow speed implied by the classes along a path.

        Reported for the dataset so the UI can show *why* a land unit is slower
        than a ship, rather than presenting an unexplained number.
        """
        if len(path) < 2:
            return 0.0
        total_m = 0.0
        total_t = 0.0
        for a, b in zip(path, path[1:]):
            m = _dist_m([a[1], a[0]], [b[1], b[0]])
            if m <= 0:
                continue
            total_m += m
            total_t += (m / 1000.0) / 40.0
        if total_t <= 0:
            return 0.0
        return (total_m / 1000.0) / total_t

    def densify(self, path: Sequence[Tuple[float, float]],
                spacing_m: float = DEFAULT_VERTEX_SPACING_M
                ) -> List[Tuple[float, float]]:
        """Resample a path to roughly even spacing, keeping endpoints.

        Routes are produced from OSM nodes, which cluster at junctions and can
        leave 20 km straight runs between them. Frames are emitted per tick, so
        uneven vertices would make apparent speed jump by an order of magnitude
        and the `closing_rate` / displacement features meaningless.
        """
        return list(geo.densify(path, max_segment_m=spacing_m))


def load(theatre: str) -> Roads:
    """Load a theatre's road network, with a cached instance per theatre."""
    cached = _CACHE.get(theatre)
    if cached is not None:
        return cached
    p = network_path(theatre)
    if not os.path.isfile(p):
        raise RoadError(
            "no road network for theatre %r at %s -- run tools/fetch_roads.py "
            "--theatre %s" % (theatre, p, theatre))
    with open(p, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    r = Roads(doc)
    _CACHE[theatre] = r
    return r


_CACHE: Dict[str, Roads] = {}


def invalidate(theatre: Optional[str] = None) -> None:
    if theatre is None:
        _CACHE.clear()
    else:
        _CACHE.pop(theatre, None)


def _dist_m(a_ll: Sequence[float], b_ll: Sequence[float]) -> float:
    """Metres between two [lon, lat] pairs."""
    return geo.haversine_m((a_ll[1], a_ll[0]), (b_ll[1], b_ll[0]))