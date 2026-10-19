"""Hex-grid map primitives for the clean-room land wargame core.

Coordinate convention
---------------------
* ``pos = row * 100 + col`` (four decimal digits), matching the public 庙算
  map/ scenario data contract.
* Rows and columns are zero-based.  ``max_row`` / ``max_column`` come from
  ``basic.json``; this module supports up to 100 x 100 for the compatible
  format, while the rest of the core can use ``(row, col)`` directly for
  larger maps.
* Neighbour order is the public data order: right, upper-right, upper-left,
  left, lower-left, lower-right (clockwise from east).

This module is deliberately dependency-free (Python 3.8 standard library).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

# Terrain codes from the public map contract.
OPEN = 0
FOREST = 1
URBAN = 2
SOFT = 3
RIVER = 4
ROADBLOCK = 5

TERRAIN_NAMES = {
    OPEN: "开阔地",
    FOREST: "丛林地",
    URBAN: "居民地",
    SOFT: "松软地",
    RIVER: "大河流",
    ROADBLOCK: "路障",
}

# Base movement multipliers for ground vehicles.  Roads and slopes are
# applied separately.  These are the public rule-of-thumb values; the spec
# records them as approximations where the public document gives no exact
# number.
VEHICLE_TERRAIN_COST = {
    OPEN: 1.0,
    FOREST: 2.0,
    URBAN: 3.0,
    SOFT: 4.0,
    RIVER: 4.0,
    ROADBLOCK: float("inf"),
}

INFANTRY_TERRAIN_COST = {
    OPEN: 1.0,
    FOREST: 1.0,
    URBAN: 1.0,
    SOFT: 1.0,
    RIVER: 1.0,
    ROADBLOCK: 1.0,  # personnel can cross a roadblock per the public rules
}

# Road classes in ``basic.json``: 0 none, 1 black, 2 red, 3 yellow.
ROAD_SPEED_KMH = {0: 0.0, 1: 40.0, 2: 60.0, 3: 90.0}


def encode(row: int, col: int) -> int:
    return int(row) * 100 + int(col)


def decode(pos: int) -> Tuple[int, int]:
    return divmod(int(pos), 100)


def in_bounds(row: int, col: int, rows: int, cols: int) -> bool:
    return 0 <= row < rows and 0 <= col < cols


def neighbor_offsets(row: int) -> List[Tuple[int, int]]:
    """Odd-r offset neighbours, clockwise from east."""
    if row & 1:
        return [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, 0), (1, 1)]
    return [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]


def neighbors(row: int, col: int, rows: int, cols: int) -> List[int]:
    out: List[int] = []
    for dr, dc in neighbor_offsets(row):
        nr, nc = row + dr, col + dc
        out.append(encode(nr, nc) if in_bounds(nr, nc, rows, cols) else -1)
    return out


def offset_to_cube(row: int, col: int) -> Tuple[int, int, int]:
    x = col - (row - (row & 1)) // 2
    z = row
    y = -x - z
    return x, y, z


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
    return rx, ry, rz


def distance(a: int, b: int) -> int:
    ar, ac = decode(a)
    br, bc = decode(b)
    ax, ay, az = offset_to_cube(ar, ac)
    bx, by, bz = offset_to_cube(br, bc)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def line(a: int, b: int) -> List[int]:
    """All hexes on the straight centre-to-centre line, inclusive."""
    if a == b:
        return [a]
    ar, ac = decode(a)
    br, bc = decode(b)
    ax, ay, az = offset_to_cube(ar, ac)
    bx, by, bz = offset_to_cube(br, bc)
    n = distance(a, b)
    out: List[int] = []
    for i in range(n + 1):
        t = i / float(n) if n else 0.0
        x, y, z = cube_round(ax + (bx - ax) * t,
                             ay + (by - ay) * t,
                             az + (bz - az) * t)
        r, c = cube_to_offset(x, y, z)
        p = encode(r, c)
        if not out or out[-1] != p:
            out.append(p)
    return out


class HexMap(object):
    """A ``basic.json``-compatible map with movement/LOS helpers."""

    def __init__(self, doc: dict):
        self.doc = doc
        self.map_id = doc.get("map_id")
        self.rows = int(doc.get("max_row") or len(doc.get("map_data") or []))
        self.cols = int(doc.get("max_column") or
                       (len(doc["map_data"][0]) if doc.get("map_data") else 0))
        self.scale_m = float(doc.get("scale") or 200.0)
        self.ele_grade = float(doc.get("ele_grade") or 10.0)
        self.start_elev = doc.get("start_elev")
        self.upper_left_lon = doc.get("upper_left_longitude")
        self.upper_left_lat = doc.get("upper_left_latitude")
        self.coordinate_system = doc.get("coordinate_system") or "oddr"
        self._cells: List[List[dict]] = doc.get("map_data") or []
        if len(self._cells) != self.rows:
            self.rows = len(self._cells)
        if self._cells:
            self.cols = len(self._cells[0])

    # -- cell access -------------------------------------------------------
    def valid(self, pos: int) -> bool:
        r, c = decode(pos)
        return in_bounds(r, c, self.rows, self.cols)

    def cell(self, pos: int) -> dict:
        r, c = decode(pos)
        if not in_bounds(r, c, self.rows, self.cols):
            raise KeyError("hex out of map: %s" % pos)
        return self._cells[r][c]

    def elev(self, pos: int) -> float:
        return float(self.cell(pos).get("elev", self.cell(pos).get("e", 0)) or 0)

    def cond(self, pos: int) -> int:
        return int(self.cell(pos).get("cond", 0) or 0)

    def roads(self, pos: int) -> Sequence[int]:
        return self.cell(pos).get("roads") or self.cell(pos).get("rd") or [0] * 6

    def rivers(self, pos: int) -> Sequence[int]:
        return self.cell(pos).get("rivers") or self.cell(pos).get("rv") or [0] * 6

    def neighbors(self, pos: int) -> List[int]:
        cell = self.cell(pos)
        nb = cell.get("neighbors") or cell.get("nb")
        if nb and len(nb) == 6:
            return [int(x) for x in nb]
        r, c = decode(pos)
        return neighbors(r, c, self.rows, self.cols)

    def direction_to(self, a: int, b: int) -> int:
        nb = self.neighbors(a)
        try:
            return nb.index(b)
        except ValueError:
            return -1

    # -- movement ----------------------------------------------------------
    def passable(self, pos: int, mode: str = "vehicle") -> bool:
        cond = self.cond(pos)
        if mode in ("air", "naval"):
            return True
        if mode == "infantry":
            return True
        # vehicle / march
        if cond == ROADBLOCK:
            return False
        if cond == RIVER:
            # A vehicle can cross a large river only on a road.
            return any(int(x) != 0 for x in self.roads(pos))
        return True

    def move_cost(self, a: int, b: int, mode: str = "vehicle") -> float:
        """Cost in ``normal-speed hexes`` for moving a -> b.

        A value of ``inf`` means the edge is impassable.  The returned cost is
        multiplied by the unit's base speed to get elapsed time.
        """
        if not self.valid(b):
            return float("inf")
        if not self.passable(b, mode):
            return float("inf")
        cond = self.cond(b)
        direction = self.direction_to(a, b)
        road_class = int(self.roads(b)[direction]) if direction >= 0 else 0

        if mode == "air" or mode == "naval":
            return 1.0
        if mode == "infantry":
            cost = INFANTRY_TERRAIN_COST.get(cond, 1.0)
            slope = abs(self.elev(b) - self.elev(a))
            if slope > 60.0:
                cost *= 2.0
            return cost
        if mode == "march":
            if road_class <= 0:
                return float("inf")
            return 1.0

        # vehicle
        cost = VEHICLE_TERRAIN_COST.get(cond, 1.0)
        if road_class > 0:
            return 1.0
        slope_level = abs(self.elev(b) - self.elev(a)) / max(1.0, self.ele_grade)
        if slope_level > 5.0:
            return float("inf")
        if slope_level >= 1.0:
            cost *= 1.0 + 0.5 * (slope_level - 1.0)
        return cost

    def terrain_cover(self, pos: int) -> int:
        cond = self.cond(pos)
        if cond == URBAN:
            return 1
        if cond == FOREST:
            return 2
        return 0

    def los_bonus(self, pos: int) -> float:
        """Extra elevation added to urban/forest cells for LOS blocking."""
        cond = self.cond(pos)
        if cond in (URBAN, FOREST):
            return self.ele_grade
        return 0.0
