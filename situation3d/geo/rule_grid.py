"""Odd-r hex grids anchored in WGS84.

Two grid types, kept deliberately separate (plan ADR-001 / ADR-006):

* :class:`RuleGrid` -- fixed 200 m cells (庙算 standard).  Used for rule
  distances (weapon/observation tables), stacking, objective capture and
  same-cell engagement.
* :class:`LegacyHexGrid` (``compat.legacy``) -- source-map cell size
  (200/1500/4000 m) used only to map old ``row/col`` coordinates.

Coordinate conventions mirror ``build_campaign_map.py`` and the legacy
``hexmap.HexMap``: flat-top odd-r layout, ``pos = row*100 + col``, neighbour
order right, upper-right, upper-left, left, lower-left, lower-right.

This module is dependency-free (stdlib only) so it runs on Python 3.8.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

# Metres per degree at the equator / meridian used by the map builder.
M_PER_DEG_LON = 111320.0
M_PER_DEG_LAT = 110540.0


def encode(row: int, col: int) -> int:
    return int(row) * 100 + int(col)


def decode(pos: int) -> Tuple[int, int]:
    return divmod(int(pos), 100)


def offset_to_cube(row: int, col: int) -> Tuple[int, int, int]:
    x = col - (row - (row & 1)) // 2
    z = row
    return x, -x - z, z


def cube_to_offset(x: int, y: int, z: int) -> Tuple[int, int]:
    row = z
    col = x + (row - (row & 1)) // 2
    return row, col


def cube_round(x: float, y: float, z: float) -> Tuple[int, int, int]:
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(ry), int(rz)


def cube_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    ax, ay, az = offset_to_cube(a[0], a[1])
    bx, by, bz = offset_to_cube(b[0], b[1])
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def neighbor_offsets(row: int) -> List[Tuple[int, int]]:
    if row & 1:
        return [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, 0), (1, 1)]
    return [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]


class HexGrid(object):
    """Flat-top odd-r grid with a pixel-space origin at the grid centre.

    ``center_lon`` / ``center_lat`` are the centre of the whole grid (the
    ``grid`` block written by ``build_campaign_map.py``), not a corner.
    """

    def __init__(self, center_lon: float, center_lat: float,
                 cols: int, rows: int, cell_m: float):
        self.center_lon = float(center_lon)
        self.center_lat = float(center_lat)
        self.cols = int(cols)
        self.rows = int(rows)
        self.cell_m = float(cell_m)
        self.width_m = self.cell_m
        self.height_m = self.cell_m * math.sqrt(3.0) / 2.0
        self._m_per_deg_lon = M_PER_DEG_LON * math.cos(math.radians(self.center_lat))
        # Pixel-space centre of the grid (same formula as build_campaign_map).
        self._cx = ((self.cols - 1) * self.width_m + self.width_m * 0.5) / 2.0
        self._cy = (self.rows - 1) * self.height_m / 2.0

    # -- row/col <-> metres ------------------------------------------------
    def cell_center_px(self, row: int, col: int) -> Tuple[float, float]:
        px = (col + 0.5 * (row & 1)) * self.width_m
        py = row * self.height_m
        return px, py

    def cell_center_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        px, py = self.cell_center_px(row, col)
        lon = self.center_lon + (px - self._cx) / self._m_per_deg_lon
        lat = self.center_lat - (py - self._cy) / M_PER_DEG_LAT
        return lon, lat

    def cell_center_m(self, row: int, col: int) -> Tuple[float, float]:
        """Cell centre in local metres relative to the grid centre."""
        px, py = self.cell_center_px(row, col)
        return px - self._cx, py - self._cy

    def lonlat_to_cube(self, lon: float, lat: float) -> Tuple[int, int, int]:
        px = self._cx + (lon - self.center_lon) * self._m_per_deg_lon
        py = self._cy - (lat - self.center_lat) * M_PER_DEG_LAT
        # Pixel -> fractional axial (odd-r) -> cube: x = px/w - r/2, z = r.
        r = py / self.height_m
        x = px / self.width_m - r / 2.0
        z = r
        y = -x - z
        return cube_round(x, y, z)

    def cell_of(self, lon: float, lat: float) -> Tuple[int, int]:
        """Nearest cell for a WGS84 position (no bounds clamping)."""
        x, y, z = self.lonlat_to_cube(lon, lat)
        return cube_to_offset(x, y, z)

    def in_bounds(self, cell: Tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def contains(self, lon: float, lat: float) -> bool:
        return self.in_bounds(self.cell_of(lon, lat))

    # -- topology ------------------------------------------------------------
    def neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        out = []
        for dr, dc in neighbor_offsets(cell[0]):
            nb = (cell[0] + dr, cell[1] + dc)
            if self.in_bounds(nb):
                out.append(nb)
        return out

    @staticmethod
    def distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return cube_distance(a, b)

    def distance_m(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        ax, ay = self.cell_center_m(*a)
        bx, by = self.cell_center_m(*b)
        return math.hypot(bx - ax, by - ay)

    @classmethod
    def from_grid_meta(cls, grid: dict) -> "HexGrid":
        width = float(grid["width_m"])
        return cls(
            center_lon=float(grid["center_lon"]),
            center_lat=float(grid["center_lat"]),
            cols=int(grid["cols"]),
            rows=int(grid["rows"]),
            cell_m=width,
        )


class RuleGrid(HexGrid):
    """Rule-adjudication grid: fixed 200 m cells (庙算 standard).

    Rule distances come from this grid's cube distance -- never from
    Euclidean metres divided by 200, which disagrees on non-collinear
    directions (two adjacent cell centres are 346 m apart).
    """

    RULE_CELL_M = 200.0

    def __init__(self, center_lon: float, center_lat: float,
                 cols: int, rows: int):
        super(RuleGrid, self).__init__(center_lon, center_lat, cols, rows,
                                       self.RULE_CELL_M)

    @classmethod
    def covering_bbox(cls, lon_min: float, lat_min: float,
                      lon_max: float, lat_max: float) -> "RuleGrid":
        """Smallest 200 m grid covering ``bbox`` (approximate ENU extent)."""
        lat0 = (lat_min + lat_max) / 2.0
        m_per_lon = M_PER_DEG_LON * math.cos(math.radians(lat0))
        width_m = (lon_max - lon_min) * m_per_lon
        height_m = (lat_max - lat_min) * M_PER_DEG_LAT
        cols = int(math.ceil(width_m / cls.RULE_CELL_M)) + 2
        rows = int(math.ceil(height_m / (cls.RULE_CELL_M * math.sqrt(3.0) / 2.0))) + 2
        return cls((lon_min + lon_max) / 2.0, lat0, cols, rows)

    @classmethod
    def from_grid_meta(cls, grid: dict) -> "RuleGrid":
        """Build from ``basic.json`` ``grid`` metadata (must be a 200 m map)."""
        width = float(grid["width_m"])
        if abs(width - cls.RULE_CELL_M) > 1e-6:
            raise ValueError(
                "grid width_m=%s is not the 200 m 庙算 rule scale; use "
                "LegacyHexGrid for old campaign maps" % width)
        return cls(
            center_lon=float(grid["center_lon"]),
            center_lat=float(grid["center_lat"]),
            cols=int(grid["cols"]),
            rows=int(grid["rows"]),
        )


def rule_distance(a: Sequence[float], b: Sequence[float],
                  grid: RuleGrid) -> int:
    """Hex distance between two WGS84 positions on the 200 m rule grid."""
    return grid.distance(grid.cell_of(a[0], a[1]), grid.cell_of(b[0], b[1]))
