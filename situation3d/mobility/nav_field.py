"""Fine navigation field built from an envpack's own geometry (P0.1).

The field is the single spatial authority *inside the pack bbox*: planning
(coarse route refinement), motion substeps and speed factors all read the
same raster, so a plan can no longer be blocked by sub-cell water or
building geometry that the coarse planner never sampled.

Raster layers (regular lon/lat lattice, default 30 m):
    elevation, slope, water depth, landcover class, real building
    footprints (height > 0 = blocked), road mask with bridge/tunnel limits.

Out-of-pack planning stays on the base-map ``GridPlanner``; ``plan_hybrid``
keeps the coarse route outside and replaces the in-pack portion with a fine
A* path, so a route crossing the pack boundary remains continuous.

Stdlib only, like the rest of the engine core.
"""

from __future__ import annotations

import heapq
import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

from env.service import LandCover
from geo.geodesy import haversine_m, initial_bearing_deg
from mobility.passability import effective_speed_ms, passable
from mobility.profile import PlatformProfile
from sim.entity import Pose, Route

RES_M = 30.0
MAX_CELLS = 250000
CACHE_VERSION = 6
_MAX_SNAP_CELLS = 5      # start snap radius; route still begins at the entity pose
_MAX_GOAL_SNAP_CELLS = 2  # goal snap radius keeps objectives in their rule cell

_COVER_CODES = {
    LandCover.OPEN: 0,
    LandCover.FOREST: 1,
    LandCover.URBAN: 2,
    LandCover.SOFT: 3,
    LandCover.WATER: 4,
    LandCover.BEACH: 5,
    LandCover.ROCK: 6,
}
_COVER_NAMES = [None] * len(_COVER_CODES)
for _name, _code in _COVER_CODES.items():
    _COVER_NAMES[_code] = _name

_ROAD_RANK = {
    "motorway": 3, "trunk": 3, "primary": 2, "secondary": 2,
    "tertiary": 1, "living_street": 1, "residential": 1, "unclassified": 1,
    "service": 1, "track": 1, "legacy": 1,
}

DIRECTIONS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
]


def _measure(value) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().lower().replace(",", ".")
    num = ""
    for ch in text:
        if ch.isdigit() or ch in (".", "-"):
            num += ch
        elif num:
            break
    try:
        return float(num)
    except ValueError:
        return 0.0


class NavField(object):
    """Navigation raster + fine planner for one envpack.

    ``env`` must provide ``nav_bbox()``, ``elevation``, ``water_depth``,
    ``landcover``, ``building_footprint_at`` and ``road_graph``.
    """

    def __init__(self, env, res_m: float = RES_M, use_cache: bool = True):
        self.env = env
        self.lon0, self.lat0, self.lon1, self.lat1 = [
            float(v) for v in env.nav_bbox()]
        self.res_m = float(res_m)
        self._fit_lattice()
        self._masks: Dict[Tuple[str, str], Tuple[bytearray, List[float]]] = {}
        self._cache_hit = False
        if not (use_cache and self._load_cache()):
            self._build()
            if use_cache:
                self._save_cache()

    # -- lattice -----------------------------------------------------------
    def _fit_lattice(self) -> None:
        self.res_m = max(5.0, float(self.res_m))
        while True:
            lat_mid = (self.lat0 + self.lat1) / 2.0
            self.m_lon = 111320.0 * math.cos(math.radians(lat_mid))
            self.m_lat = 110540.0
            self.d_lon = self.res_m / self.m_lon
            self.d_lat = self.res_m / self.m_lat
            # floor keeps every cell centre inside the pack rectangle
            self.cols = int(math.floor((self.lon1 - self.lon0) / self.d_lon)) + 1
            self.rows = int(math.floor((self.lat1 - self.lat0) / self.d_lat)) + 1
            if self.rows * self.cols <= MAX_CELLS:
                break
            self.res_m *= 1.25

    def _center(self, row: int, col: int) -> Tuple[float, float]:
        return self.lon0 + col * self.d_lon, self.lat0 + row * self.d_lat

    def cell_index(self, lon: float, lat: float) -> Tuple[int, int]:
        col = int(round((lon - self.lon0) / self.d_lon))
        row = int(round((lat - self.lat0) / self.d_lat))
        return row, col

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def covers(self, lon: float, lat: float) -> bool:
        return self._index_of(lon, lat) is not None

    def _index_of(self, lon: float, lat: float) -> Optional[int]:
        # Coverage is the pack rectangle, not the pad of the rounded lattice;
        # points just outside must fall back to the base-map authority.
        if not (self.lon0 <= lon <= self.lon1
                and self.lat0 <= lat <= self.lat1):
            return None
        row, col = self.cell_index(lon, lat)
        if not self.in_bounds(row, col):
            return None
        return row * self.cols + col

    # -- build -------------------------------------------------------------
    def _build(self) -> None:
        env = self.env
        n = self.rows * self.cols
        self.elev = [0.0] * n
        self.depth: List[Optional[float]] = [None] * n
        self.cover = bytearray(n)
        self.building_h = [0.0] * n
        footprint = getattr(env, "building_footprint_at", None)
        fine_cover_at = getattr(env, "landcover_polygon_at", None)
        for row in range(self.rows):
            lat = self.lat0 + row * self.d_lat
            base = row * self.cols
            for col in range(self.cols):
                lon = self.lon0 + col * self.d_lon
                i = base + col
                self.elev[i] = env.elevation(lon, lat)
                cover_code = _COVER_CODES.get(env.landcover(lon, lat), 0)
                self.cover[i] = cover_code
                depth = env.water_depth(lon, lat)
                fine_cover = (fine_cover_at(lon, lat)
                              if fine_cover_at is not None else None)
                if depth is not None and fine_cover is not None:
                    if fine_cover != LandCover.WATER:
                        # OSM landcover polygon says land: the 2 km base-map
                        # water fallback must not make it impassable.
                        depth = None
                self.depth[i] = depth
                if footprint is not None:
                    item = footprint(lon, lat)
                    if item is not None:
                        self.building_h[i] = float(item.height_m or 6.0)
        self._build_slope()
        self._build_roads()

    def _build_slope(self) -> None:
        rows, cols = self.rows, self.cols
        elev = self.elev
        res = self.res_m
        slope = [0.0] * (rows * cols)
        for row in range(rows):
            r0 = max(0, row - 1)
            r1 = min(rows - 1, row + 1)
            dy = (r1 - r0) * res
            for col in range(cols):
                c0 = max(0, col - 1)
                c1 = min(cols - 1, col + 1)
                dx = (c1 - c0) * res
                if dx <= 0.0 or dy <= 0.0:
                    continue
                dzdx = (elev[row * cols + c1] - elev[row * cols + c0]) / dx
                dzdy = (elev[r1 * cols + col] - elev[r0 * cols + col]) / dy
                slope[row * cols + col] = math.degrees(
                    math.atan(math.hypot(dzdx, dzdy)))
        self.slope = slope

    def _build_roads(self) -> None:
        n = self.rows * self.cols
        self.road = bytearray(n)          # 0 none, 1..3 class rank
        self.bridge = bytearray(n)        # 1 = bridge/tunnel deck
        # 0 = no one-way info; 1..255 = edge bearing (a->b) quantised.
        self.oneway = bytearray(n)
        self.limits: Dict[int, dict] = {}
        graph = getattr(self.env, "road_graph", None)
        if graph is None:
            return
        nodes, edges = graph()
        step = self.res_m * 0.7
        for edge in edges:
            pa, pb = nodes.get(edge.get("a")), nodes.get(edge.get("b"))
            if not pa or not pb:
                continue
            rank = _ROAD_RANK.get(str(edge.get("class")), 0)
            deck = rank > 0 and bool(edge.get("bridge") or edge.get("tunnel"))
            limits = {}
            max_w = _measure(edge.get("maxweight"))
            max_h = _measure(edge.get("maxheight"))
            width = _measure(edge.get("width"))
            if max_w > 0:
                limits["maxweight"] = max_w
            if max_h > 0:
                limits["maxheight"] = max_h
            if width > 0:
                limits["width"] = width
            length = haversine_m(pa, pb)
            bearing = initial_bearing_deg(pa, pb)
            dir_code = int(round(bearing / 360.0 * 254.0)) + 1
            oneway = rank > 0 and bool(edge.get("oneway"))
            steps = max(1, int(math.ceil(length / step)))
            for k in range(steps + 1):
                f = k / float(steps)
                lon = pa[0] + (pb[0] - pa[0]) * f
                lat = pa[1] + (pb[1] - pa[1]) * f
                i = self._index_of(lon, lat)
                if i is None:
                    continue
                if rank > self.road[i]:
                    self.road[i] = rank
                    self.oneway[i] = dir_code if oneway else 0
                elif rank == self.road[i] and rank > 0:
                    if not oneway:
                        self.oneway[i] = 0  # bidirectional road wins
                    elif self.oneway[i] == 0:
                        self.oneway[i] = dir_code
                    elif self.oneway[i] != 255:
                        prev_deg = (self.oneway[i] - 1) / 254.0 * 360.0
                        diff = abs((bearing - prev_deg + 180.0) % 360.0
                                   - 180.0)
                        if diff > 90.0:
                            self.oneway[i] = 255  # junction: no enforcement
                if deck:
                    # Thicken to a plus pad so diagonal bridges are not
                    # severed by the A* no-corner-cutting rule.
                    r0, c0 = divmod(i, self.cols)
                    self.bridge[i] = 1
                    for rr, cc in ((r0 - 1, c0), (r0 + 1, c0),
                                   (r0, c0 - 1), (r0, c0 + 1)):
                        if 0 <= rr < self.rows and 0 <= cc < self.cols:
                            self.bridge[rr * self.cols + cc] = 1
                if limits and rank > 0:
                    merged = self.limits.setdefault(i, {})
                    for key, value in limits.items():
                        old = merged.get(key, 0.0)
                        merged[key] = value if old <= 0 else min(old, value)

    # -- cache -------------------------------------------------------------
    def _signature(self) -> Tuple:
        root = getattr(self.env, "root", None)
        if not root:
            return (CACHE_VERSION, round(self.res_m, 3))
        stamps = [CACHE_VERSION, round(self.res_m, 3)]
        try:
            def _stat(path):
                st = os.stat(path)
                return (path, st.st_mtime_ns, st.st_size)

            stamps.append(_stat(os.path.join(root, "manifest.json")))
            for meta in (getattr(self.env, "layers", None) or {}).values():
                path = os.path.join(root, meta.get("path") or "")
                if meta.get("path") and os.path.isfile(path):
                    stamps.append(_stat(path))
            fallback = (getattr(self.env, "manifest", None) or {}).get(
                "fallback_basic_json")
            if fallback:
                path = os.path.join(root, fallback)
                if os.path.isfile(path):
                    stamps.append(_stat(path))
        except OSError:
            # Never reuse a cache we cannot prove current.
            return (CACHE_VERSION, round(self.res_m, 3), "unstable", id(self))
        return tuple(stamps)

    def _cache_path(self) -> Optional[str]:
        root = getattr(self.env, "root", None)
        if not root:
            return None
        return os.path.join(root, "nav_cache_%sm.pkl"
                            % ("%g" % round(self.res_m, 1)))

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if not path or not os.path.isfile(path):
            return False
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            if data.get("signature") != self._signature():
                return False
            rows, cols = int(data["rows"]), int(data["cols"])
            size = rows * cols
            elev = data["elev"]
            if (rows <= 0 or cols <= 0 or len(elev) != size
                    or len(data["slope"]) != size
                    or len(data["depth"]) != size
                    or len(data["cover"]) != size
                    or len(data["building_h"]) != size
                    or len(data["road"]) != size
                    or len(data["bridge"]) != size
                    or len(data["oneway"]) != size):
                return False
        except (OSError, EOFError, ImportError, AttributeError, KeyError,
                TypeError, ValueError, pickle.PickleError):
            return False
        self.rows = rows
        self.cols = cols
        self.res_m = data["res_m"]
        self.d_lon = data["d_lon"]
        self.d_lat = data["d_lat"]
        self.m_lon = data["m_lon"]
        self.m_lat = data["m_lat"]
        self.elev = elev
        self.slope = data["slope"]
        self.depth = data["depth"]
        self.cover = data["cover"]
        self.building_h = data["building_h"]
        self.road = data["road"]
        self.bridge = data["bridge"]
        self.oneway = data["oneway"]
        self.limits = data["limits"]
        self._cache_hit = True
        return True

    def _save_cache(self) -> None:
        path = self._cache_path()
        if not path:
            return
        data = {
            "signature": self._signature(),
            "rows": self.rows, "cols": self.cols, "res_m": self.res_m,
            "d_lon": self.d_lon, "d_lat": self.d_lat,
            "m_lon": self.m_lon, "m_lat": self.m_lat,
            "elev": self.elev, "slope": self.slope, "depth": self.depth,
            "cover": self.cover, "building_h": self.building_h,
            "road": self.road, "bridge": self.bridge, "oneway": self.oneway,
            "limits": self.limits,
        }
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(data, fh, protocol=2)
            os.replace(tmp, path)
        except OSError:
            pass

    # -- per-profile masks -------------------------------------------------
    def _limits_ok(self, prof: PlatformProfile, i: int) -> bool:
        limits = self.limits.get(i)
        if not limits:
            return True
        if prof.weight_t > 0 and 0 < limits.get("maxweight", 0.0) < prof.weight_t:
            return False
        if (prof.vehicle_height_m > 0
                and 0 < limits.get("maxheight", 0.0) < prof.vehicle_height_m):
            return False
        if prof.beam_m > 0 and 0 < limits.get("width", 0.0) < prof.beam_m:
            return False
        return True

    def _cell_passable(self, prof: PlatformProfile, i: int,
                       factors: List[float]) -> bool:
        if prof.domain in ("land", "amphibious") and self.bridge[i]:
            # Bridge/tunnel deck: water depth, slope and footprints below it
            # do not apply; weight/height/width limits still do.
            return self._limits_ok(prof, i)
        if self.building_h[i] > 0.0:
            return False
        depth = self.depth[i]
        if prof.domain == "air":
            return True
        if prof.domain in ("surface", "subsurface"):
            if depth is None:
                return False
            if depth < prof.draft_m + prof.ukc_m:
                return False
            if (prof.domain == "subsurface" and prof.max_operating_depth_m > 0
                    and depth > prof.max_operating_depth_m):
                return False
            return True
        if prof.domain == "amphibious":
            if depth is not None:
                return depth <= prof.swim_depth_m
            return self._land_ok(prof, i, factors)
        # land
        if depth is not None:
            return (depth <= prof.wading_depth_m
                    and self.slope[i] <= prof.max_gradient_deg
                    and self._limits_ok(prof, i))
        return self._land_ok(prof, i, factors)

    def _land_ok(self, prof: PlatformProfile, i: int,
                 factors: List[float]) -> bool:
        # Optional per-scenario strict mode (TW2 convoy demo): ground platforms
        # may only move on road cells. Default off keeps P6 cases unchanged.
        if (getattr(self, "strict_roads", False)
                and prof.domain in ("land", "amphibious")
                and self.road[i] == 0):
            return False
        factor = factors[self.cover[i]]
        if factor <= 0.0:
            return prof.road_bound and self.road[i] > 0
        if self.slope[i] > prof.max_gradient_deg:
            return False
        if (self.cover[i] == _COVER_CODES[LandCover.URBAN]
                and prof.road_bound and self.road[i] == 0):
            return False
        return self._limits_ok(prof, i)

    def _speed_at_cell(self, prof: PlatformProfile, i: int,
                       factors: List[float]) -> float:
        if prof.domain in ("land", "amphibious") and self.bridge[i]:
            return prof.max_speed_ms
        depth = self.depth[i]
        if depth is not None:
            if prof.domain == "amphibious":
                f = 0.5 if depth <= prof.swim_depth_m else 0.0
            else:
                f = 0.35 if depth <= prof.wading_depth_m else 0.0
        else:
            f = factors[self.cover[i]]
            if prof.max_gradient_deg > 0:
                f *= max(0.1, 1.0 - 0.5 * min(
                    1.0, self.slope[i] / prof.max_gradient_deg))
            if prof.road_bound and self.road[i] > 0:
                f = 1.0
            elif self.road[i] > 0 and prof.domain == "land":
                f = min(1.0, f * 1.25)
            if f <= 0.0:
                # Unreachable for mask cells; keeps the planner finite.
                f = 0.05
        # Route speeds already carry the terrain factor (motion multiplies
        # again, as with GridPlanner routes).
        return max(0.05, prof.max_speed_ms * f)

    def profile_mask(self, prof: PlatformProfile) -> Tuple[bytearray, List[float]]:
        key = (prof.key, prof.domain)
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        factors = [0.0] * len(_COVER_NAMES)
        for code, name in enumerate(_COVER_NAMES):
            factors[code] = prof.terrain_factor(name)
        mask = bytearray(self.rows * self.cols)
        speed = [0.0] * (self.rows * self.cols)
        if prof.domain == "air":
            return (mask, speed)
        for i in range(len(mask)):
            if self._cell_passable(prof, i, factors):
                mask[i] = 1
                speed[i] = self._speed_at_cell(prof, i, factors)
        self._masks[key] = (mask, speed)
        return mask, speed

    # -- point queries -----------------------------------------------------
    def passable(self, prof: PlatformProfile, lon: float, lat: float) -> Optional[bool]:
        """True/False inside the pack; ``None`` outside (caller uses env)."""
        i = self._index_of(lon, lat)
        if i is None:
            return None
        return bool(self.profile_mask(prof)[0][i])

    def speed_factor(self, prof: PlatformProfile, lon: float, lat: float) -> Optional[float]:
        i = self._index_of(lon, lat)
        if i is None:
            return None
        if prof.domain in ("air", "surface", "subsurface"):
            return 1.0
        speed = self.profile_mask(prof)[1][i]
        if prof.max_speed_ms <= 0:
            return 0.0
        return speed / prof.max_speed_ms

    def building_only_block(self, prof: PlatformProfile, lon: float,
                            lat: float) -> bool:
        """True when the in-pack block comes *only* from a footprint raster.

        OBB footprint checks ignore these samples: building obstacles are
        enforced at cell granularity by the centre-cell check, while water,
        slope, landcover and bridge limits still reject the whole footprint.
        """
        i = self._index_of(lon, lat)
        if i is None or self.building_h[i] <= 0.0 or self.depth[i] is not None:
            return False
        if self.slope[i] > prof.max_gradient_deg:
            return False
        factor = prof.terrain_factor(_COVER_NAMES[self.cover[i]])
        if factor <= 0.0 and not (prof.road_bound and self.road[i] > 0):
            return False
        if (self.cover[i] == _COVER_CODES[LandCover.URBAN]
                and prof.road_bound and self.road[i] == 0):
            return False
        return self._limits_ok(prof, i)

    def nearest_passable(self, prof: PlatformProfile, lon: float, lat: float,
                         radius: int = 6) -> Optional[Tuple[float, float]]:
        mask = self.profile_mask(prof)[0]
        start_row, start_col = self.cell_index(lon, lat)
        best = None
        for dr in range(-radius, radius + 1):
            row = start_row + dr
            if not 0 <= row < self.rows:
                continue
            for dc in range(-radius, radius + 1):
                col = start_col + dc
                if not 0 <= col < self.cols:
                    continue
                if not mask[row * self.cols + col]:
                    continue
                cx, cy = self._center(row, col)
                d = (cx - lon) ** 2 * self.m_lon ** 2 + (cy - lat) ** 2 * self.m_lat ** 2
                ring = max(abs(dr), abs(dc))
                key = (ring, d, row, col)
                if best is None or key < best[0]:
                    best = (key, cx, cy)
        if best is None:
            return None
        return best[1], best[2]

    # -- planning ----------------------------------------------------------
    def _snap_cell(self, mask, prof, lon: float, lat: float,
                   radius: int) -> Optional[int]:
        row, col = self.cell_index(lon, lat)
        if self.in_bounds(row, col) and mask[row * self.cols + col]:
            return row * self.cols + col
        point = self.nearest_passable(prof, lon, lat, radius)
        if point is None:
            return None
        return self._index_of(point[0], point[1])

    def plan(self, prof: PlatformProfile, start: Tuple[float, float],
             goal: Tuple[float, float], blocked=None,
             start_snap: int = _MAX_SNAP_CELLS,
             goal_snap: int = _MAX_GOAL_SNAP_CELLS,
             max_nodes: int = 300000,
             safe_fallback: bool = False) -> Optional[Route]:
        mask, speed = self.profile_mask(prof)
        start_i = self._snap_cell(mask, prof, start[0], start[1], start_snap)
        goal_i = self._snap_cell(mask, prof, goal[0], goal[1], goal_snap)
        if start_i is None or goal_i is None:
            return None
        if start_i == goal_i:
            row, col = divmod(start_i, self.cols)
            cx, cy = self._center(row, col)
            first = Pose(start[0], start[1],
                         self.env.elevation(start[0], start[1]))
            goal_own = self._index_of(goal[0], goal[1])
            if goal_own is not None and mask[goal_own]:
                end = Pose(goal[0], goal[1],
                           self.env.elevation(goal[0], goal[1]))
            else:
                end = Pose(cx, cy, self.env.elevation(cx, cy))
            if haversine_m((first.lon, first.lat),
                           (end.lon, end.lat)) < 0.5:
                end = Pose(end.lon + 1e-7, end.lat, end.alt_m)
            if self._segment_clear(first, end, mask, blocked):
                return self._route(prof, [first, end])
            mid = Pose(cx, cy, self.env.elevation(cx, cy))
            if (self._segment_clear(first, mid, mask, blocked)
                    and self._segment_clear(mid, end, mask, blocked)):
                return self._route(prof, [first, mid, end])
            return None
        if blocked is not None:
            gr, gc = divmod(goal_i, self.cols)
            if blocked(*self._center(gr, gc)):
                return None
        cols, rows = self.cols, self.rows
        start_row, start_col = divmod(start_i, cols)
        goal_row, goal_col = divmod(goal_i, cols)
        straight = haversine_m(self._center(start_row, start_col),
                               self._center(goal_row, goal_col))
        max_speed = max(0.1, prof.max_speed_ms)
        open_heap = [(straight / max_speed, 0.0, 0, start_i)]
        g_score = {start_i: 0.0}
        came: Dict[int, int] = {}
        closed = bytearray(rows * cols)
        seq = 0
        expanded = 0
        found = False
        while open_heap and expanded < max_nodes:
            _, g, _, i = heapq.heappop(open_heap)
            if closed[i]:
                continue
            closed[i] = 1
            expanded += 1
            if i == goal_i:
                found = True
                break
            row, col = divmod(i, cols)
            base = row * cols
            for dr, dc, cost in DIRECTIONS:
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                ni = nr * cols + nc
                if not mask[ni]:
                    continue
                if dr and dc:
                    # No corner cutting through diagonal gaps.
                    if not mask[base + nc] or not mask[nr * cols + col]:
                        continue
                step = self.res_m * cost
                oneway_factor = self._oneway_factor(prof, i, ni)
                if oneway_factor <= 0.0:
                    continue
                # Road-bound convoys should not shortcut through open ground.
                # A hard block is too slow at 30 m resolution, so pay a strong
                # detour penalty instead (still finite if a road is missing).
                offroad = (4.0 if (prof.road_bound and self.road[ni] == 0)
                           else 1.0)
                tentative = g + step / speed[ni] * oneway_factor * offroad
                if tentative >= g_score.get(ni, float("inf")) - 1e-9:
                    continue
                if blocked is not None and ni != goal_i:
                    if blocked(*self._center(nr, nc)):
                        continue
                g_score[ni] = tentative
                came[ni] = i
                h = haversine_m(self._center(nr, nc),
                                self._center(goal_row, goal_col)) / max_speed
                seq += 1
                heapq.heappush(open_heap, (tentative + h, tentative, seq, ni))
        if not found:
            return None
        cells = [goal_i]
        while cells[-1] != start_i:
            cells.append(came[cells[-1]])
        cells.reverse()
        points: List[Pose] = []
        first = Pose(start[0], start[1],
                     self.env.elevation(start[0], start[1]))
        points.append(first)
        for i in cells:
            r, c = divmod(i, cols)
            lon, lat = self._center(r, c)
            # Keep every A* vertex: decimation can shortcut a blocked cell.
            if haversine_m((points[-1].lon, points[-1].lat), (lon, lat)) > 0.5:
                points.append(Pose(lon, lat, self.env.elevation(lon, lat)))
        goal_own = self._index_of(goal[0], goal[1])
        if goal_own is not None and mask[goal_own]:
            last_lon, last_lat = goal
        else:
            last_lon, last_lat = self._center(goal_row, goal_col)
        if haversine_m((points[-1].lon, points[-1].lat),
                       (last_lon, last_lat)) > 0.5:
            points.append(Pose(last_lon, last_lat,
                               self.env.elevation(last_lon, last_lat)))
        if len(points) < 2:
            return None
        raw_points = list(points)
        points = self._smooth(points, prof, mask, blocked)
        points = self._fillet(prof, points, mask, blocked)
        if prof.road_bound and not self._points_oneway_ok(prof, points):
            # Smoothing/fillets must not bypass a one-way restriction.
            if self._points_oneway_ok(prof, raw_points):
                points = raw_points
            else:
                return None
        if safe_fallback and not self._points_ok(prof, points):
            # Smoothing/fillets can shave a blocked cell at 30 m resolution,
            # and the exact start/goal may sit outside the 30 m mask cell.
            # Fall back to raw A* vertices with snapped cell-center endpoints.
            start_row, start_col = divmod(start_i, self.cols)
            goal_row, goal_col = divmod(goal_i, self.cols)
            sx, sy = self._center(start_row, start_col)
            gx, gy = self._center(goal_row, goal_col)
            fixed = [Pose(sx, sy, self.env.elevation(sx, sy))]
            fixed.extend(raw_points[1:-1])
            fixed.append(Pose(gx, gy, self.env.elevation(gx, gy)))
            if self._points_ok(prof, fixed):
                points = fixed
            elif self._points_ok(prof, raw_points):
                points = raw_points
            else:
                # Both candidates are invalid; do not return a route that
                # route_ok would reject.
                return None
        return self._route(prof, points)

    def _segment_clear(self, a: Pose, b: Pose, mask, blocked) -> bool:
        """Exact supercover test of the cells the segment touches.

        Cells outside the lattice belong to the base-map authority and are
        skipped; every in-lattice cell must be passable.
        """
        x0 = (a.lon - self.lon0) / self.d_lon + 0.5
        y0 = (a.lat - self.lat0) / self.d_lat + 0.5
        x1 = (b.lon - self.lon0) / self.d_lon + 0.5
        y1 = (b.lat - self.lat0) / self.d_lat + 0.5
        col, row = int(math.floor(x0)), int(math.floor(y0))
        end_col, end_row = int(math.floor(x1)), int(math.floor(y1))
        if not self._cell_clear(row, col, mask, blocked):
            return False
        if not self._cell_clear(end_row, end_col, mask, blocked):
            return False
        dx, dy = x1 - x0, y1 - y0
        step_c = 1 if dx > 0 else -1
        step_r = 1 if dy > 0 else -1
        inf = float("inf")
        if dx == 0:
            t_max_c, t_delta_c = inf, inf
        else:
            boundary = col + (1 if dx > 0 else 0)
            t_max_c = (boundary - x0) / dx
            t_delta_c = 1.0 / abs(dx)
        if dy == 0:
            t_max_r, t_delta_r = inf, inf
        else:
            boundary = row + (1 if dy > 0 else 0)
            t_max_r = (boundary - y0) / dy
            t_delta_r = 1.0 / abs(dy)
        cur_r, cur_c = row, col
        limit = abs(end_col - col) + abs(end_row - row) + 2
        for _ in range(limit):
            if cur_r == end_row and cur_c == end_col:
                return True
            if t_max_c < t_max_r:
                cur_c += step_c
                t_max_c += t_delta_c
            else:
                cur_r += step_r
                t_max_r += t_delta_r
            if not self._cell_clear(cur_r, cur_c, mask, blocked):
                return False
        return False

    def _cell_clear(self, row: int, col: int, mask, blocked) -> bool:
        if not self.in_bounds(row, col):
            return True
        if not mask[row * self.cols + col]:
            return False
        if blocked is not None:
            clon, clat = self._center(row, col)
            if blocked(clon, clat):
                return False
        return True

    def _points_oneway_ok(self, prof, points: List[Pose]) -> bool:
        if not prof.road_bound:
            return True
        for i in range(1, len(points)):
            a, b = points[i - 1], points[i]
            total = haversine_m((a.lon, a.lat), (b.lon, b.lat))
            n = max(1, int(math.ceil(total / (self.res_m * 0.5))))
            prev_cell = None
            for k in range(n + 1):
                f = k / float(n)
                cell = self._index_of(a.lon + (b.lon - a.lon) * f,
                                      a.lat + (b.lat - a.lat) * f)
                if cell is None:
                    continue
                if prev_cell is not None and cell != prev_cell:
                    if self._oneway_factor(prof, prev_cell, cell) <= 0.0:
                        return False
                prev_cell = cell
        return True

    def _oneway_factor(self, prof: PlatformProfile, from_i: int,
                       to_i: int) -> float:
        """Cost multiplier for travelling against a one-way road.

        Road-bound platforms are blocked; others pay a heavy detour penalty
        (they may still cross kerbs/off-road, but prefer legal direction).
        """
        a = self.oneway[from_i]
        b = self.oneway[to_i]
        if (not a and not b) or a == 255 or b == 255:
            return 1.0
        r0, c0 = divmod(from_i, self.cols)
        r1, c1 = divmod(to_i, self.cols)
        bearing = initial_bearing_deg(self._center(r0, c0),
                                      self._center(r1, c1))
        ref_code = a if a not in (0, 255) else b
        if ref_code in (0, 255):
            return 1.0
        ref = float(ref_code - 1) / 254.0 * 360.0
        delta = abs((bearing - ref + 180.0) % 360.0 - 180.0)
        if delta > 90.0:
            return 0.0 if prof.road_bound else 3.0
        return 1.0

    def _smooth(self, points: List[Pose], prof, mask, blocked) -> List[Pose]:
        out = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self._segment_clear(
                    points[i], points[j], mask, blocked):
                j -= 1
            out.append(points[j])
            i = j
        return out

    def _segment_clear_wide(self, a: Pose, b: Pose, prof, mask,
                            blocked) -> bool:
        """Centreline plus ±half-width samples (planner/motion agreement)."""
        if not self._segment_clear(a, b, mask, blocked):
            return False
        half = max(1.0, float(getattr(prof, "width_m", 2.5)) / 2.0)
        total = haversine_m((a.lon, a.lat), (b.lon, b.lat))
        if total < 1.0:
            return True
        bearing = math.radians(initial_bearing_deg((a.lon, a.lat),
                                                   (b.lon, b.lat)))
        east = math.sin(bearing)
        north = math.cos(bearing)
        kx = 111320.0 * math.cos(math.radians((a.lat + b.lat) / 2.0))
        ky = 110540.0
        n = max(1, int(math.ceil(total / (self.res_m * 0.5))))
        for k in range(n + 1):
            f = k / float(n)
            lon = a.lon + (b.lon - a.lon) * f
            lat = a.lat + (b.lat - a.lat) * f
            for side in (-1.0, 1.0):
                olon = lon + side * half * (-north) / kx
                olat = lat + side * half * east / ky
                i = self._index_of(olon, olat)
                if i is None:
                    continue
                if not mask[i] and not self.building_only_block(prof, olon, olat):
                    return False
                if blocked is not None and blocked(olon, olat):
                    return False
        return True

    def _fillet(self, prof, points: List[Pose], mask, blocked) -> List[Pose]:
        """Replace sharp corners with circular arcs of radius R_min.

        Only non-pivot platforms need this; tracked/infantry keep their
        vertices (they can turn in place or at high yaw rate).  Every arc is
        validated against the same raster/collision authority, falling back
        to the original vertex when the fillet would leave passable space.
        """
        radius = float(getattr(prof, "min_turn_radius_m", 0.0) or 0.0)
        if getattr(prof, "pivot_turn", False) or radius <= 0.0 or len(points) < 3:
            return points
        out = [points[0]]
        for i in range(1, len(points) - 1):
            a, b, c = points[i - 1], points[i], points[i + 1]
            kx = 111320.0 * math.cos(math.radians(b.lat))
            ky = 110540.0
            bx, by = b.lon * kx, b.lat * ky
            ax, ay = a.lon * kx, a.lat * ky
            cx, cy = c.lon * kx, c.lat * ky
            d1 = math.hypot(ax - bx, ay - by)
            d2 = math.hypot(cx - bx, cy - by)
            if d1 < 1.0 or d2 < 1.0:
                out.append(b)
                continue
            u1 = ((ax - bx) / d1, (ay - by) / d1)
            u2 = ((cx - bx) / d2, (cy - by) / d2)
            dot = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
            interior = math.acos(dot)          # straight = pi
            deflection = math.pi - interior    # straight = 0
            if (deflection < math.radians(15.0)
                    or deflection > math.radians(150.0)):
                out.append(b)  # gentle corner or U-turn (multi-point turn)
                continue
            theta = interior
            # Circular fillet: tangent length t = r / tan(interior/2)
            # (equivalent to r*tan(deflection/2)).
            cot_half = 1.0 / math.tan(theta / 2.0)
            r = min(radius, 0.45 * d1, 0.45 * d2)
            t = r * cot_half
            if t > 0.45 * d1 or t > 0.45 * d2:
                r = min(r, 0.45 * d1 / max(1e-6, cot_half),
                        0.45 * d2 / max(1e-6, cot_half))
                t = r * cot_half
            if r < 1.0:
                out.append(b)
                continue
            p1 = Pose(b.lon + u1[0] * t / kx, b.lat + u1[1] * t / ky,
                      b.alt_m)
            p2 = Pose(b.lon + u2[0] * t / kx, b.lat + u2[1] * t / ky,
                      b.alt_m)
            bis_raw = (u1[0] + u2[0], u1[1] + u2[1])
            bl = math.hypot(bis_raw[0], bis_raw[1])
            if bl < 1e-6:
                out.append(b)
                continue
            bis = (bis_raw[0] / bl, bis_raw[1] / bl)
            dist = r / math.sin(theta / 2.0)
            center = (bx + bis[0] * dist, by + bis[1] * dist)
            v1 = (p1.lon * kx - center[0], p1.lat * ky - center[1])
            v2 = (p2.lon * kx - center[0], p2.lat * ky - center[1])
            a1 = math.atan2(v1[1], v1[0])
            a2 = math.atan2(v2[1], v2[0])
            delta = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
            samples = max(3, int(math.ceil(abs(delta) / math.radians(30.0))))
            candidate = [p1]
            for k in range(1, samples):
                ang = a1 + delta * k / samples
                candidate.append(Pose(
                    (center[0] + r * math.cos(ang)) / kx,
                    (center[1] + r * math.sin(ang)) / ky, b.alt_m))
            candidate.append(p2)
            prev = out[-1]
            ok = True
            for p in candidate + [c]:
                if not self._segment_clear_wide(prev, p, prof, mask, blocked):
                    ok = False
                    break
                prev = p
            if ok:
                out.extend(candidate)
            else:
                out.append(b)
        out.append(points[-1])
        return out

    def _segment_speed(self, prof: PlatformProfile, a: Pose, b: Pose) -> float:
        mid_lon = (a.lon + b.lon) / 2.0
        mid_lat = (a.lat + b.lat) / 2.0
        nav_speed = self.speed_factor(prof, mid_lon, mid_lat)
        if nav_speed is None:
            return max(0.1, effective_speed_ms(
                prof, mid_lon, mid_lat, self.env, a.alt_m))
        return max(0.1, prof.max_speed_ms * nav_speed)

    def _route(self, prof: PlatformProfile, points: List[Pose]) -> Route:
        dedup = [points[0]]
        for p in points[1:]:
            if haversine_m((dedup[-1].lon, dedup[-1].lat),
                           (p.lon, p.lat)) > 0.5:
                dedup.append(p)
        if len(dedup) < 2:
            dedup.append(Pose(dedup[0].lon + 1e-7, dedup[0].lat,
                              dedup[0].alt_m))
        speeds = [self._segment_speed(prof, dedup[i - 1], dedup[i])
                  for i in range(1, len(dedup))]
        return Route(points=dedup, speed_limits=speeds,
                     modes=["offroad"] * (len(dedup) - 1))

    # -- hybrid planning ---------------------------------------------------
    def plan_hybrid(self, coarse_planner, prof: PlatformProfile,
                    start: Tuple[float, float],
                    goal: Tuple[float, float], blocked=None) -> Optional[Route]:
        """Fine plan when both ends are inside; otherwise refine the coarse
        route's in-pack portion and keep the rest.  A coarse fallback is only
        returned when the pack authority confirms its in-pack part."""
        if self.covers(*start) and self.covers(*goal):
            return self.plan(prof, start, goal, blocked=blocked)
        coarse = coarse_planner.plan(prof, start, goal, blocked=blocked)
        if coarse is None:
            return None
        refined = self._refine(coarse, prof, blocked)
        if refined is not None:
            return refined
        return coarse if self.route_ok(prof, coarse) else None

    def _clip_inside(self, a, b) -> Optional[Tuple[float, float]]:
        """Segment/rect intersection parameters (Liang-Barsky)."""
        t0, t1 = 0.0, 1.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        for p, q in ((-dx, a[0] - self.lon0), (dx, self.lon1 - a[0]),
                     (-dy, a[1] - self.lat0), (dy, self.lat1 - a[1])):
            if abs(p) < 1e-15:
                if q < 0:
                    return None
                continue
            t = q / p
            if p < 0:
                if t > t1:
                    return None
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return None
                if t < t1:
                    t1 = t
        if t0 >= t1:
            return None
        return t0, t1

    def _coarse_slice(self, route: Route, s0: float, s1: float) -> List[Pose]:
        pts = [route.pose_at(s0)]
        for i in range(1, len(route.points)):
            if s0 < route.cum_m[i] < s1:
                pts.append(route.points[i])
        pts.append(route.pose_at(s1))
        return pts

    def _refine(self, route: Route, prof: PlatformProfile, blocked=None) -> Optional[Route]:
        intervals = []
        for i in range(1, len(route.points)):
            a = route.points[i - 1].as_tuple()
            b = route.points[i].as_tuple()
            clip = self._clip_inside(a, b)
            if clip is None:
                continue
            s0 = route.cum_m[i - 1] + (route.cum_m[i] - route.cum_m[i - 1]) * clip[0]
            s1 = route.cum_m[i - 1] + (route.cum_m[i] - route.cum_m[i - 1]) * clip[1]
            if intervals and s0 <= intervals[-1][1] + 1.0:
                intervals[-1][1] = max(intervals[-1][1], s1)
            else:
                intervals.append([s0, s1])
        if not intervals:
            return None
        pieces = []   # ("coarse", s0, s1) or ("nav", [Pose])
        cursor = 0.0
        for s0, s1 in intervals:
            if s1 - s0 < 2.0 * self.res_m:
                continue
            a = route.pose_at(s0)
            b = route.pose_at(s1)
            if (not self._point_ok(prof, a.lon, a.lat)
                    or not self._point_ok(prof, b.lon, b.lat)):
                # The coarse route enters/exits the pack through geometry the
                # pack itself forbids; no sound refinement is possible.
                continue
            fine = self.plan(prof, (a.lon, a.lat), (b.lon, b.lat),
                             blocked=blocked, start_snap=1, goal_snap=1)
            if fine is None:
                continue
            if cursor < s0:
                pieces.append(("coarse", cursor, s0))
            pieces.append(("nav", fine.points))
            cursor = s1
        if cursor < route.total_m:
            pieces.append(("coarse", cursor, route.total_m))
        if not any(piece[0] == "nav" for piece in pieces):
            return None
        points: List[Pose] = []
        for piece in pieces:
            if piece[0] == "coarse":
                _, s0, s1 = piece
                pts = self._coarse_slice(route, s0, s1)
            else:
                pts = piece[1]
            for p in pts:
                if points and haversine_m((points[-1].lon, points[-1].lat),
                                          (p.lon, p.lat)) <= 0.5:
                    continue
                points.append(p)
        if len(points) < 2:
            return None
        if not self._points_ok(prof, points):
            return None
        if self._route_length(points) > 2.5 * route.total_m:
            return None
        return self._route(prof, points)

    @staticmethod
    def _route_length(points: List[Pose]) -> float:
        return sum(haversine_m((points[i - 1].lon, points[i - 1].lat),
                               (points[i].lon, points[i].lat))
                   for i in range(1, len(points)))

    # -- motion support ----------------------------------------------------
    def _point_ok(self, prof: PlatformProfile, lon: float, lat: float) -> bool:
        # ``None`` (a sub-cell sliver at the pack border) is base-map
        # authority and therefore acceptable; only explicit blocks are not.
        return self.passable(prof, lon, lat) is not False

    def _points_ok(self, prof: PlatformProfile, points: List[Pose]) -> bool:
        """Exact supercover test of every in-pack portion of a polyline."""
        mask = self.profile_mask(prof)[0]
        for i in range(1, len(points)):
            a, b = points[i - 1], points[i]
            clip = self._clip_inside(a.as_tuple(), b.as_tuple())
            if clip is None:
                continue
            pa = Pose(a.lon + (b.lon - a.lon) * clip[0],
                      a.lat + (b.lat - a.lat) * clip[0])
            pb = Pose(a.lon + (b.lon - a.lon) * clip[1],
                      a.lat + (b.lat - a.lat) * clip[1])
            if not self._segment_clear(pa, pb, mask, None):
                return False
        return True

    def route_ok(self, prof: PlatformProfile, route: Route) -> bool:
        return self._points_ok(prof, route.points)

    def motion_point_ok(self, prof: PlatformProfile, lon: float, lat: float) -> bool:
        """Substep check used by ``sim.motion`` inside coverage."""
        i = self._index_of(lon, lat)
        if i is None:
            return passable(prof, lon, lat, self.env, self.env.elevation(lon, lat))
        return bool(self.profile_mask(prof)[0][i])
