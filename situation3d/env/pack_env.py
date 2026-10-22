"""EnvironmentService backed by an ``msgeo.envpack.v1`` directory.

Layers consumed: terrain grid (bilinear), landcover/water/building polygons
(point-in-polygon), road graph (nearest edge) and the pack's fallback
``basic.json`` for anything the pack does not cover.  All geometry is JSON;
no GDAL/Shapely dependency is required for this fallback pack format.
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Tuple

from env.service import Building, LandCover, Weather
from geo.geodesy import haversine_m
from geo.rule_grid import HexGrid
from compat.legacy import LegacyHexGrid


def _point_in_ring(x: float, y: float, ring: List[List[float]]) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y):
            xcross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < xcross:
                inside = not inside
        j = i
    return inside


def _point_segment_m(lon, lat, pa, pb) -> float:
    """Approximate point-to-segment distance in metres (local plane)."""
    kx = 111320.0 * math.cos(math.radians(lat))
    ky = 110540.0
    px, py = lon * kx, lat * ky
    ax, ay = pa[0] * kx, pa[1] * ky
    bx, by = pb[0] * kx, pb[1] * ky
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _bbox(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


class _SpatialIndex(object):
    """Coarse lon/lat bucket index for polygons/edges (stdlib only)."""

    def __init__(self, cell_deg: float = 0.01):
        self.cell = float(cell_deg)
        self.buckets = {}

    def _range(self, x0, y0, x1, y1):
        ix0, ix1 = int(math.floor(x0 / self.cell)), int(math.floor(x1 / self.cell))
        iy0, iy1 = int(math.floor(y0 / self.cell)), int(math.floor(y1 / self.cell))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                yield (ix, iy)

    def add(self, item, bbox):
        for key in self._range(*bbox):
            self.buckets.setdefault(key, []).append(item)

    def query(self, lon, lat):
        return self.buckets.get((int(math.floor(lon / self.cell)),
                                 int(math.floor(lat / self.cell))), ())

    def query_bbox(self, x0, y0, x1, y1):
        seen = set()
        out = []
        for key in self._range(x0, y0, x1, y1):
            for item in self.buckets.get(key, ()):
                if id(item) in seen:
                    continue
                seen.add(id(item))
                out.append(item)
        return out


class EnvPackEnvironment(object):
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.layers = self.manifest.get("layers") or {}
        self.grid = LegacyHexGrid.from_grid_meta(self.manifest["grid"])
        self._terrain = self._load_json("terrain")
        self._bathymetry = self._load_json("bathymetry")
        self._landcover = self._load_json("landcover") or []
        self._water = self._load_json("water") or []
        self._buildings = self._load_json("buildings") or []
        self._roads = self._load_json("roads") or {"nodes": {}, "edges": []}
        for layer in (self._landcover, self._water, self._buildings):
            kept = []
            for item in layer:
                ring = item.get("ring") or []
                if len(ring) >= 4:
                    item["_bbox"] = _bbox(ring)
                    kept.append(item)
            layer[:] = kept
        self._landcover_index = _SpatialIndex()
        self._water_index = _SpatialIndex()
        self._building_index = _SpatialIndex()
        for item in self._landcover:
            self._landcover_index.add(item, item["_bbox"])
        for item in self._water:
            self._water_index.add(item, item["_bbox"])
        for item in self._buildings:
            self._building_index.add(item, item["_bbox"])
        self._road_index = _SpatialIndex(0.02)
        nodes = self._road_nodes = (self._roads.get("nodes") or {})
        for edge in self._roads.get("edges") or []:
            pa, pb = nodes.get(edge["a"]), nodes.get(edge["b"])
            if not pa or not pb:
                continue
            edge["_bbox"] = (min(pa[0], pb[0]), min(pa[1], pb[1]),
                             max(pa[0], pb[0]), max(pa[1], pb[1]))
            self._road_index.add(edge, edge["_bbox"])
        self._basic = None
        fallback = self.manifest.get("fallback_basic_json")
        if fallback and os.path.isfile(os.path.join(root, fallback)):
            with open(os.path.join(root, fallback), encoding="utf-8") as fh:
                self._basic = json.load(fh)
            from env.grid import MapGridEnvironment
            self._basic_env = MapGridEnvironment(self._basic, self.grid)
        else:
            self._basic_env = None
        self._weather = Weather()
        self._building_bbox = None
        if self._buildings:
            xs = [item["_bbox"][0] for item in self._buildings]
            ys = [item["_bbox"][1] for item in self._buildings]
            xs2 = [item["_bbox"][2] for item in self._buildings]
            ys2 = [item["_bbox"][3] for item in self._buildings]
            self._building_bbox = (min(xs), min(ys), max(xs2), max(ys2))
        self._landcover_cache = {}
        self._max_building_height = max(
            (float(b.get("height_m") or 6.0) for b in self._buildings),
            default=0.0)
        max_elev = 0.0
        if self._terrain:
            max_elev = max(v for row in self._terrain["values"] for v in row
                           if v is not None)
        self._max_terrain_elev = float(max_elev)
        self._water_cache = {}
        self._los_cache = {}
        self._min_elev = None

    def road_edge_at(self, lon: float, lat: float,
                     tolerance_m: float = 25.0):
        """Return the nearest road edge within ``tolerance_m`` (or None)."""
        best = None
        best_m = None
        for edge in self._road_index.query(lon, lat):
            pa = self._road_nodes.get(edge["a"])
            pb = self._road_nodes.get(edge["b"])
            if not pa or not pb:
                continue
            d = _point_segment_m(lon, lat, pa, pb)
            if d <= tolerance_m and (best_m is None or d < best_m):
                best, best_m = edge, d
        return best

    def _load_json(self, layer: str):
        meta = self.layers.get(layer)
        if not meta or not meta.get("path"):
            return None
        path = os.path.join(self.root, meta["path"])
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # -- terrain -----------------------------------------------------------
    def elevation(self, lon: float, lat: float) -> float:
        t = self._terrain
        if not t:
            return self._basic_env.elevation(lon, lat) if self._basic_env else 0.0
        fx = (lon - t["lon0"]) / max(1e-12, t["lon1"] - t["lon0"])
        fy = (lat - t["lat0"]) / max(1e-12, t["lat1"] - t["lat0"])
        fx = min(1.0, max(0.0, fx)) * (t["cols"] - 1)
        fy = min(1.0, max(0.0, fy)) * (t["rows"] - 1)
        x0, y0 = int(fx), int(fy)
        x1, y1 = min(t["cols"] - 1, x0 + 1), min(t["rows"] - 1, y0 + 1)
        tx, ty = fx - x0, fy - y0
        v = (t["values"][y0][x0] * (1 - tx) * (1 - ty)
             + t["values"][y0][x1] * tx * (1 - ty)
             + t["values"][y1][x0] * (1 - tx) * ty
             + t["values"][y1][x1] * tx * ty)
        return float(v)

    @staticmethod
    def _grid_value(grid, lon: float, lat: float):
        """Bilinear sample of a regular lon/lat grid (None when off-grid)."""
        if not grid:
            return None
        fx = (lon - grid["lon0"]) / max(1e-12, grid["lon1"] - grid["lon0"])
        fy = (lat - grid["lat0"]) / max(1e-12, grid["lat1"] - grid["lat0"])
        if fx < 0 or fy < 0 or fx > 1 or fy > 1:
            return None
        x0, y0 = int(fx * (grid["cols"] - 1)), int(fy * (grid["rows"] - 1))
        x0 = min(x0, grid["cols"] - 2) if grid["cols"] > 1 else 0
        y0 = min(y0, grid["rows"] - 2) if grid["rows"] > 1 else 0
        x1 = min(x0 + 1, grid["cols"] - 1)
        y1 = min(y0 + 1, grid["rows"] - 1)
        tx = fx * (grid["cols"] - 1) - x0
        ty = fy * (grid["rows"] - 1) - y0
        v = grid["values"]
        corners = (v[y0][x0], v[y0][x1], v[y1][x0], v[y1][x1])
        if any(c is None for c in corners):
            return None
        top = corners[0] * (1 - tx) + corners[1] * tx
        bottom = corners[2] * (1 - tx) + corners[3] * tx
        return top * (1 - ty) + bottom * ty

    def slope_deg(self, lon: float, lat: float, bearing_deg: float = 0.0) -> float:
        br = math.radians(bearing_deg)
        d = max(50.0, self.grid.cell_m)
        m_lon = 111320.0 * math.cos(math.radians(lat))
        a = (lon - d * math.sin(br) / m_lon, lat - d * math.cos(br) / 110540.0)
        b = (lon + d * math.sin(br) / m_lon, lat + d * math.cos(br) / 110540.0)
        dh = self.elevation(*b) - self.elevation(*a)
        return math.degrees(math.atan2(abs(dh), 2.0 * d))

    def min_elevation(self) -> float:
        if self._min_elev is None:
            t = self._terrain
            if t and t.get("values"):
                self._min_elev = min(min(row) for row in t["values"])
            elif self._basic_env is not None:
                self._min_elev = self._basic_env.min_elevation()
            else:
                self._min_elev = 0.0
        return self._min_elev

    def elevation_grade(self, lon: float, lat: float) -> int:
        return int(round((self.elevation(lon, lat) - self.min_elevation()) /
                         max(1.0, self.ele_grade)))

    # -- polygons ----------------------------------------------------------
    def _polygon_at(self, items, lon, lat):
        if isinstance(items, _SpatialIndex):
            candidates = items.query(lon, lat)
        else:
            candidates = items
        for item in candidates:
            x0, y0, x1, y1 = item.get("_bbox") or _bbox(item.get("ring") or [[0, 0]])
            if lon < x0 or lon > x1 or lat < y0 or lat > y1:
                continue
            if _point_in_ring(lon, lat, item.get("ring") or []):
                return item
        return None

    def landcover(self, lon: float, lat: float) -> str:
        key = (round(lon, 5), round(lat, 5))
        if key in self._landcover_cache:
            return self._landcover_cache[key]
        if self._polygon_at(self._water_index, lon, lat) is not None:
            value = LandCover.WATER
        else:
            item = self._polygon_at(self._landcover_index, lon, lat)
            if item is not None:
                value = item.get("class") or LandCover.OPEN
            elif self._basic_env is not None:
                value = self._basic_env.landcover(lon, lat)
            else:
                value = LandCover.OPEN
        self._landcover_cache[key] = value
        return value

    def water_depth(self, lon: float, lat: float) -> Optional[float]:
        key = (round(lon, 5), round(lat, 5))
        if key in self._water_cache:
            return self._water_cache[key]
        item = self._polygon_at(self._water_index, lon, lat)
        if item is not None:
            depth = self.layers.get("water", {}).get("depth_m", 20.0)
            value = float(item.get("depth_m") or depth)
        elif self._basic_env is not None:
            value = self._basic_env.water_depth(lon, lat)
        else:
            value = None
        if value is not None:
            sound = self._grid_value(self._bathymetry, lon, lat)
            if sound is not None and sound < 0.0:
                value = round(-float(sound), 1)
        self._water_cache[key] = value
        return value

    def is_water(self, lon: float, lat: float) -> bool:
        return self.water_depth(lon, lat) is not None

    def building_at(self, lon: float, lat: float) -> Optional[Building]:
        item = self._polygon_at(self._building_index, lon, lat)
        if item is not None:
            return Building(item.get("ring") or [], float(item.get("height_m") or 6.0))
        if self._basic_env is not None:
            return self._basic_env.building_at(lon, lat)
        return None

    def cover_height(self, lon: float, lat: float) -> float:
        item = self._polygon_at(self._building_index, lon, lat)
        if item is not None:
            return float(item.get("height_m") or 6.0)
        cover = self.landcover(lon, lat)
        return {LandCover.FOREST: 15.0, LandCover.URBAN: 10.0}.get(cover, 0.0)

    def building_intersects(self, a, h0, b, h1) -> bool:
        if not self._buildings or self._building_bbox is None:
            return False
        bb = self._building_bbox
        if (max(a[0], b[0]) < bb[0] or min(a[0], b[0]) > bb[2]
                or max(a[1], b[1]) < bb[1] or min(a[1], b[1]) > bb[3]):
            return False
        if (min(h0, h1) > self._max_terrain_elev
                + self._max_building_height + 1.0):
            return False
        candidates = self._candidates_along(a, b)
        for item in candidates:
            ring = item.get("ring") or []
            x0, y0, x1, y1 = item.get("_bbox")
            if (max(a[0], b[0]) < x0 or min(a[0], b[0]) > x1
                    or max(a[1], b[1]) < y0 or min(a[1], b[1]) > y1):
                continue
            if not _segment_hits_aabb(a, b, x0, y0, x1, y1):
                continue
            if not _segment_intersects_ring(a, b, ring):
                continue
            f = _first_inside_fraction(a, b, ring)
            lon = a[0] + (b[0] - a[0]) * f
            lat = a[1] + (b[1] - a[1]) * f
            top = self.elevation(lon, lat) + float(item.get("height_m") or 6.0)
            if top > h0 + (h1 - h0) * f:
                return True
        return False

    def _candidates_along(self, a, b):
        """Buildings in every index cell the segment crosses (exact DDA)."""
        cell = self._building_index.cell
        x0, y0 = a[0], a[1]
        x1, y1 = b[0], b[1]
        ix, iy = int(math.floor(x0 / cell)), int(math.floor(y0 / cell))
        ex, ey = int(math.floor(x1 / cell)), int(math.floor(y1 / cell))
        dx, dy = x1 - x0, y1 - y0
        stepx = 1 if dx > 0 else -1
        stepy = 1 if dy > 0 else -1
        inf = float("inf")
        if abs(dx) < 1e-15:
            tmax_x, tdelta_x = inf, inf
        else:
            boundary = (ix + (1 if dx > 0 else 0)) * cell
            tmax_x = (boundary - x0) / dx
            tdelta_x = cell / abs(dx)
        if abs(dy) < 1e-15:
            tmax_y, tdelta_y = inf, inf
        else:
            boundary = (iy + (1 if dy > 0 else 0)) * cell
            tmax_y = (boundary - y0) / dy
            tdelta_y = cell / abs(dy)
        seen = set()
        out = []
        guard = 0
        while guard < 100000:
            guard += 1
            for item in self._building_index.query(
                    (ix + 0.5) * cell, (iy + 0.5) * cell):
                key = id(item)
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
            if ix == ex and iy == ey:
                break
            if tmax_x < tmax_y:
                ix += stepx
                tmax_x += tdelta_x
            else:
                iy += stepy
                tmax_y += tdelta_y
        return out

    def weather(self, lon: float, lat: float) -> Weather:
        return self._weather

    def sample_line(self, a, b, step_m: float = 30.0):
        total = haversine_m(a, b)
        n = max(1, int(math.ceil(total / step_m)))
        for i in range(n + 1):
            f = i / float(n)
            lon = a[0] + (b[0] - a[0]) * f
            lat = a[1] + (b[1] - a[1]) * f
            yield {"lon": lon, "lat": lat, "fraction": f,
                   "elevation": self.elevation(lon, lat),
                   "cover_height": self.cover_height(lon, lat),
                   "landcover": self.landcover(lon, lat)}

    def cell_line_of_sight(self, ca, cb) -> bool:
        key = (ca, cb) if ca <= cb else (cb, ca)
        if key in self._los_cache:
            return self._los_cache[key]
        a = self.grid.cell_center_lonlat(*ca)
        b = self.grid.cell_center_lonlat(*cb)
        ea, eb = self.elevation(*a), self.elevation(*b)
        h0 = ea + 2.0 + (self.ele_grade if ea > eb else 0.0)
        h1 = eb + 1.5
        result = True
        total = haversine_m(a, b)
        n = max(1, int(math.ceil(total / max(100.0, self.grid.cell_m / 4.0))))
        for i in range(1, n):
            f = i / float(n)
            lon = a[0] + (b[0] - a[0]) * f
            lat = a[1] + (b[1] - a[1]) * f
            sight = h0 + (h1 - h0) * f
            if self.elevation(lon, lat) + self.cover_height(lon, lat) > sight + 1e-6:
                result = False
                break
        self._los_cache[key] = result
        return result

    def line_of_sight(self, a, ha, b, hb, step_m: float = 30.0) -> bool:
        ea, eb = self.elevation(*a), self.elevation(*b)
        h0 = ha + (self.ele_grade if ea > eb else 0.0)
        h1 = hb
        if self.building_intersects(a, h0, b, h1):
            return False
        for p in self.sample_line(a, b, step_m):
            if p["fraction"] <= 0.0 or p["fraction"] >= 1.0:
                continue
            sight = h0 + (h1 - h0) * p["fraction"]
            if p["elevation"] + p["cover_height"] > sight + 1e-6:
                return False
        return True

    @property
    def ele_grade(self) -> float:
        return float(self.manifest.get("ele_grade") or 10.0)

    # -- roads -------------------------------------------------------------
    def on_road(self, lon: float, lat: float, tolerance_m: float = 30.0) -> bool:
        return self.road_class(lon, lat, tolerance_m) > 0

    def road_class(self, lon: float, lat: float, tolerance_m: float = 30.0) -> int:
        nodes = self._roads.get("nodes") or {}
        cell = self._road_index.cell
        d_deg = tolerance_m / 100000.0
        candidates = self._road_index.query_bbox(lon - d_deg, lat - d_deg,
                                                 lon + d_deg, lat + d_deg)
        for edge in candidates:
            pa = nodes.get(edge["a"])
            pb = nodes.get(edge["b"])
            if not pa or not pb:
                continue
            if _point_segment_distance(lon, lat, pa, pb) <= tolerance_m:
                return _CLASS_RANK.get(str(edge.get("class")), 0)
        return 0

    def nearest_road(self, lon: float, lat: float, max_m: float = 3000.0):
        nodes = self._roads.get("nodes") or {}
        best = None
        best_d = None
        d_deg = max_m / 100000.0
        candidates = self._road_index.query_bbox(lon - d_deg, lat - d_deg,
                                                 lon + d_deg, lat + d_deg)
        for edge in candidates:
            pa = nodes.get(edge["a"])
            pb = nodes.get(edge["b"])
            if not pa or not pb:
                continue
            point, d = _segment_nearest_point(lon, lat, pa, pb)
            if d <= max_m and (best_d is None or d < best_d):
                best_d = d
                best = point
        return best


_CLASS_RANK = {
    "motorway": 3, "trunk": 3, "primary": 2, "secondary": 2,
    "tertiary": 1, "residential": 1, "unclassified": 1,
    "service": 1, "track": 1, "legacy": 1,
}
_NON_DRIVABLE = {"footway", "path", "cycleway", "steps", "pedestrian",
                 "platform", "corridor", "bridleway", "construction", "proposed"}


def _segment_nearest_point(lon, lat, a, b):
    """Return (nearest lon/lat point on segment, distance metres)."""
    m_lon = 111320.0 * math.cos(math.radians(lat))
    m_lat = 110540.0
    px, py = (lon - a[0]) * m_lon, (lat - a[1]) * m_lat
    bx, by = (b[0] - a[0]) * m_lon, (b[1] - a[1]) * m_lat
    denom = bx * bx + by * by
    t = 0.0 if denom <= 1e-9 else max(0.0, min(1.0, (px * bx + py * by) / denom))
    qx, qy = t * bx, t * by
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])), math.hypot(px - qx, py - qy)


def _point_segment_distance(lon, lat, a, b) -> float:
    return _segment_nearest_point(lon, lat, a, b)[1]


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _segment_hits_aabb(a, b, x0, y0, x1, y1) -> bool:
    """Liang-Barsky segment/AABB overlap (cheap prefilter)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                 (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-15:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            if t > t0:
                t0 = t
        else:
            if t < t0:
                return False
            if t < t1:
                t1 = t
    return True


def _segment_intersects_ring(a, b, ring) -> bool:
    if _point_in_ring(a[0], a[1], ring) or _point_in_ring(b[0], b[1], ring):
        return True
    for i in range(len(ring) - 1):
        if _segments_cross(a, b, ring[i], ring[i + 1]):
            return True
    return False


def _first_inside_fraction(a, b, ring) -> float:
    n = 20
    for i in range(n + 1):
        f = i / float(n)
        lon = a[0] + (b[0] - a[0]) * f
        lat = a[1] + (b[1] - a[1]) * f
        if _point_in_ring(lon, lat, ring):
            return f
    return 0.5


def load_envpack_environment(root: str):
    """Return ``(environment, hex grid)`` for a pack directory."""
    env = EnvPackEnvironment(root)
    return env, env.grid
