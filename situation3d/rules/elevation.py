"""Elevation attack-level modifier (``ele_rect_to_car/peo``).

Semantics confirmed by black-box runs (2026-10-21): the 8x12 table is
indexed by ``[delta_grade - 1][distance_hex - 1]`` where ``delta_grade`` is
``|attacker_elev - target_elev| / ele_grade`` clamped to 1..8, and the
modifier is applied **only when the attacker is lower than the target**
(firing uphill).  Firing downhill gives no bonus (0), and distances beyond
12 hexes give 0.  Values are <= 0.

Examples from the probe (weapon 36 vs vehicle, map ele_grade=20 m):
  attacker 2 grades lower, distance 1 -> row 2, col 1 = -2 (base 10 -> 8)
  attacker 2 grades lower, distance 5 -> row 2, col 5 = -1 (base 9 -> 8)
  attacker 1 grade lower,  distance 8 -> row 1, col 8 = 0  (base 9 -> 9)
and the same row/col rule applies to the personnel table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

DEFAULT_ELEVATION = Path(__file__).resolve().parent / "data" / "elevation.json"

MAX_GRADE = 8
MAX_DISTANCE = 12


def load_elevation(path: Optional[str] = None) -> dict:
    p = Path(path) if path else DEFAULT_ELEVATION
    return json.loads(p.read_text(encoding="utf-8"))


def elevation_modifier(doc: dict, attacker_higher: bool,
                       delta_grade: int, distance_hex: int,
                       target_kind: str = "vehicle") -> int:
    """Attack-level modifier from the 8x12 table (0 when not uphill)."""
    if attacker_higher:
        return 0
    if delta_grade <= 0 or distance_hex <= 0:
        return 0
    if distance_hex > MAX_DISTANCE:
        return 0
    row = min(int(delta_grade), MAX_GRADE) - 1
    col = int(distance_hex) - 1
    table = doc["peo"] if target_kind == "infantry" else doc["car"]
    return int(table[row][col])
