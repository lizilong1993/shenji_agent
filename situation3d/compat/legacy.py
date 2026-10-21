"""Legacy coordinate adapter for pre-geo scenarios and replays.

``LegacyHexGrid`` maps the old ``row/col`` contract to WGS84 using the source
map's own cell size (200 m for 庙算 maps, 1500/4000 m for the old campaign
maps).  It never participates in rule adjudication -- that is ``RuleGrid``'s
job (ADR-001 / ADR-006).

The adapter needs explicit grid metadata (``basic.json`` ``grid`` block):
bbox+scale alone cannot recover the origin, so we refuse to guess.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from geo.rule_grid import HexGrid, decode, encode

REQUIRED_GRID_KEYS = ("center_lon", "center_lat", "cols", "rows", "width_m")


class LegacyHexGrid(HexGrid):
    """Old-map coordinate adapter (cell size = source map scale)."""

    def __init__(self, center_lon: float, center_lat: float,
                 cols: int, rows: int, cell_m: float):
        super(LegacyHexGrid, self).__init__(center_lon, center_lat, cols, rows,
                                            cell_m)

    def to_lonlat(self, pos: int) -> Tuple[float, float]:
        row, col = decode(pos)
        return self.cell_center_lonlat(row, col)

    def from_lonlat(self, lon: float, lat: float) -> int:
        row, col = self.cell_of(lon, lat)
        return encode(row, col)

    def to_pose(self, pos: int, alt_m: float = 0.0) -> dict:
        lon, lat = self.to_lonlat(pos)
        return {"lon": lon, "lat": lat, "alt_m": alt_m,
                "heading_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}

    @classmethod
    def from_grid_meta(cls, grid: dict) -> "LegacyHexGrid":
        missing = [k for k in REQUIRED_GRID_KEYS if k not in grid]
        if missing:
            raise ValueError("legacy grid metadata missing: %s" % ", ".join(missing))
        return cls(
            center_lon=float(grid["center_lon"]),
            center_lat=float(grid["center_lat"]),
            cols=int(grid["cols"]),
            rows=int(grid["rows"]),
            cell_m=float(grid["width_m"]),
        )

    @classmethod
    def from_map_doc(cls, basic: dict) -> "LegacyHexGrid":
        """Build from a ``basic.json`` document that carries a ``grid`` block."""
        grid = basic.get("grid")
        if not grid:
            raise ValueError(
                "map has no 'grid' metadata; legacy row/col cannot be "
                "georeferenced without an explicit centre/scale/origin")
        return cls.from_grid_meta(grid)


def legacy_scenario_positions(scenario: dict,
                              grid: LegacyHexGrid) -> Dict[int, Tuple[float, float]]:
    """Map every operator's ``cur_hex`` to WGS84 (debug/observation helper)."""
    out = {}
    for op in scenario.get("operators", []):
        if "cur_hex" not in op:
            continue
        out[int(op["obj_id"])] = grid.to_lonlat(int(op["cur_hex"]))
    return out


def move_path_to_lonlat(move_path: List[int],
                        grid: LegacyHexGrid) -> List[Tuple[float, float]]:
    """Old ``move_path`` (hex ids) -> centre coordinates for Route building."""
    return [grid.to_lonlat(int(p)) for p in move_path]


def legacy_observation_fields(pos: int, grid: LegacyHexGrid,
                              alt_m: float = 0.0) -> dict:
    """Legacy observation subset derived from a continuous pose (P5 helper)."""
    lon, lat = grid.to_lonlat(pos)
    row, col = decode(pos)
    return {
        "hex": pos,
        "row": row,
        "col": col,
        "pos": {"lon": lon, "lat": lat, "alt": alt_m},
    }
