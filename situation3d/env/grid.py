"""Grid-backed environment service for the geo engine.

P1's full envpack (DEM/landcover/buildings/water/road graph) is not built yet;
this adapter exposes the same ``EnvironmentService`` contract using a 庙算
``basic.json`` map plus its hex grid metadata.  It is intentionally honest
about its limits:

* elevation comes from the map cell (nearest sample), not a 30 m DEM;
* landcover is the map ``cond`` code;
* water is ``cond == 4`` with a synthetic depth;
* buildings do not exist (urban cells are treated as 10 m cover prisms);
* roads are cell-level adjacency, not an OSM graph.

The interface matches ``env.service.EnvironmentService`` so a real envpack can
replace it without touching mobility/sim/rules.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

from env.service import Building, LandCover, Weather
from geo.geodesy import haversine_m, interpolate
from geo.rule_grid import HexGrid

COND_LANDCOVER = {
    0: LandCover.OPEN,
    1: LandCover.FOREST,
    2: LandCover.URBAN,
    3: LandCover.SOFT,
    4: LandCover.WATER,
    5: LandCover.ROCK,   # roadblock; treated as impassable rock in mobility
}

COVER_HEIGHT_M = {
    LandCover.FOREST: 15.0,
    LandCover.URBAN: 10.0,
}


def _offset_lonlat(lon: float, lat: float, east_m: float, north_m: float) -> Tuple[float, float]:
    m_per_lon = 111320.0 * math.cos(math.radians(lat))
    return lon + east_m / m_per_lon, lat + north_m / 110540.0


class MapGridEnvironment(object):
    def __init__(self, basic: dict, grid: HexGrid,
                 default_water_depth_m: float = 20.0,
                 weather: Optional[Weather] = None):
        self.basic = basic
        self.grid = grid
        self.rows = int(basic.get("max_row") or len(basic.get("map_data") or []))
        self.cells = basic.get("map_data") or []
        if self.cells:
            self.rows = len(self.cells)
            self.cols = len(self.cells[0])
        else:
            self.cols = int(basic.get("max_column") or grid.cols)
        self.ele_grade = float(basic.get("ele_grade") or 10.0)
        self.start_elev = float(basic.get("start_elev") or 0.0)
        self.default_water_depth = float(default_water_depth_m)
        self._weather = weather or Weather()
        self._min_elev = None
        self._los_cache = {}

    # -- cell access -------------------------------------------------------
    def cell_at(self, lon: float, lat: float) -> dict:
        row, col = self.grid.cell_of(lon, lat)
        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))
        return self.cells[row][col]

    def elevation(self, lon: float, lat: float) -> float:
        return self.start_elev + float(self.cell_at(lon, lat).get("elev", 0) or 0)

    def min_elevation(self) -> float:
        if self._min_elev is None:
            self._min_elev = self.start_elev + min(
                float(c.get("elev", 0) or 0) for row in self.cells for c in row)
        return self._min_elev

    def elevation_grade(self, lon: float, lat: float) -> int:
        return int(round((self.elevation(lon, lat) - self.min_elevation()) / max(1.0, self.ele_grade)))

    def slope_deg(self, lon: float, lat: float, bearing_deg: float = 0.0) -> float:
        br = math.radians(bearing_deg)
        d = max(50.0, self.grid.cell_m)
        a = _offset_lonlat(lon, lat, -d * math.sin(br), -d * math.cos(br))
        b = _offset_lonlat(lon, lat, d * math.sin(br), d * math.cos(br))
        dh = self.elevation(b[0], b[1]) - self.elevation(a[0], a[1])
        return math.degrees(math.atan2(abs(dh), 2.0 * d))

    def landcover(self, lon: float, lat: float) -> str:
        return COND_LANDCOVER.get(int(self.cell_at(lon, lat).get("cond", 0) or 0),
                                  LandCover.OPEN)

    def water_depth(self, lon: float, lat: float) -> Optional[float]:
        if self.landcover(lon, lat) == LandCover.WATER:
            return self.default_water_depth
        return None

    def is_water(self, lon: float, lat: float) -> bool:
        return self.landcover(lon, lat) == LandCover.WATER

    def building_at(self, lon: float, lat: float) -> Optional[Building]:
        if self.landcover(lon, lat) == LandCover.URBAN:
            return Building(footprint=[], height_m=COVER_HEIGHT_M[LandCover.URBAN])
        return None

    def cover_height(self, lon: float, lat: float) -> float:
        return COVER_HEIGHT_M.get(self.landcover(lon, lat), 0.0)

    def building_intersects(self, a, h0, b, h1) -> bool:
        # No footprints: treat urban cover as sampled cover height only.
        return False

    def weather(self, lon: float, lat: float) -> Weather:
        return self._weather

    # -- roads -------------------------------------------------------------
    def road_class(self, lon: float, lat: float) -> int:
        roads = self.cell_at(lon, lat).get("roads") or []
        return max([int(x or 0) for x in roads] or [0])

    def on_road(self, lon: float, lat: float) -> bool:
        return self.road_class(lon, lat) > 0

    def nearest_road(self, lon: float, lat: float,
                     max_m: float = 3000.0) -> Optional[Tuple[float, float]]:
        """Nearest road-cell centre within ``max_m`` (demo-quality snap)."""
        row0, col0 = self.grid.cell_of(lon, lat)
        radius = max(1, int(math.ceil(max_m / max(1.0, self.grid.cell_m))))
        best = None
        best_d = None
        for r in range(max(0, row0 - radius), min(self.rows, row0 + radius + 1)):
            for c in range(max(0, col0 - radius), min(self.cols, col0 + radius + 1)):
                cell = self.cells[r][c]
                if not any(int(x or 0) > 0 for x in (cell.get("roads") or [])):
                    continue
                plat, plon = self.grid.cell_center_lonlat(r, c)
                d = haversine_m((lon, lat), (plon, plat))
                if d <= max_m and (best_d is None or d < best_d):
                    best_d = d
                    best = (plon, plat)
        return best

    # -- sampling / LOS ----------------------------------------------------
    def sample_line(self, a: Tuple[float, float], b: Tuple[float, float],
                    step_m: float = 30.0,
                    include_endpoints: bool = True) -> Iterator[dict]:
        total = haversine_m(a, b)
        n = max(1, int(math.ceil(total / step_m)))
        start = 0 if include_endpoints else 1
        end = n + 1 if include_endpoints else n
        for i in range(start, end):
            f = i / float(n)
            lon, lat = interpolate(a, b, f)
            yield {"lon": lon, "lat": lat, "fraction": f,
                   "elevation": self.elevation(lon, lat),
                   "cover_height": self.cover_height(lon, lat),
                   "landcover": self.landcover(lon, lat)}

    def line_of_sight(self, a: Tuple[float, float], ha: float,
                      b: Tuple[float, float], hb: float,
                      step_m: float = 100.0) -> bool:
        """Rule-text LOS on the map grid: intermediate cells only.

        ``step_m`` is accepted for interface compatibility; the check walks
        the hex cell line (centre-to-centre), which is the rule definition.
        """
        ca = self.grid.cell_of(a[0], a[1])
        cb = self.grid.cell_of(b[0], b[1])
        ea = self._cell_elev(ca)
        eb = self._cell_elev(cb)
        h0 = ha + (self.ele_grade if ea > eb else 0.0)
        h1 = hb
        cells = self.grid.line(ca, cb)
        n = max(1, len(cells) - 1)
        for i, (r, c) in enumerate(cells[1:-1], start=1):
            sight = h0 + (h1 - h0) * (i / float(n))
            elev, cover = self._cell_obstacle(r, c)
            if elev + cover > sight + 1e-6:
                return False
        return True

    def cell_line_of_sight(self, ca: Tuple[int, int], cb: Tuple[int, int]) -> bool:
        """Cached ground-level LOS between two cells (static terrain)."""
        key = (ca, cb) if ca <= cb else (cb, ca)
        if key in self._los_cache:
            return self._los_cache[key]
        ea = self._cell_elev(ca)
        eb = self._cell_elev(cb)
        h0 = ea + 2.0 + (self.ele_grade if ea > eb else 0.0)
        h1 = eb + 1.5
        cells = self.grid.line(ca, cb)
        n = max(1, len(cells) - 1)
        result = True
        for i, (r, c) in enumerate(cells[1:-1], start=1):
            sight = h0 + (h1 - h0) * (i / float(n))
            elev, cover = self._cell_obstacle(r, c)
            if elev + cover > sight + 1e-6:
                result = False
                break
        self._los_cache[key] = result
        return result

    def _cell_elev(self, cell: Tuple[int, int]) -> float:
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return self.start_elev
        return self.start_elev + float(self.cells[r][c].get("elev", 0) or 0)

    def _cell_obstacle(self, r: int, c: int):
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return self.start_elev, 0.0
        cell = self.cells[r][c]
        elev = self.start_elev + float(cell.get("elev", 0) or 0)
        cond = int(cell.get("cond", 0) or 0)
        cover = COVER_HEIGHT_M.get(COND_LANDCOVER.get(cond, LandCover.OPEN), 0.0)
        return elev, cover
        for p in self.sample_line(a, b, step_m, include_endpoints=False):
            sight = h0 + (h1 - h0) * p["fraction"]
            if p["elevation"] + p["cover_height"] > sight + 1e-6:
                return False
        return True
