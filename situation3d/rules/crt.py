"""CRT and correction-table lookups from the authoritative 裁决表格.

Vehicle CRT (sheet 对车辆单位战斗结果):
  * rows 2..6 map (attacker vehicles/squads 1..5, attack level) -> column;
  * rows 8..18 map random1 (2d6, 2..12) -> raw damage in that column.
Personnel CRT (sheet 直瞄武器对人员and步兵轻武器对车辆战斗结果表):
  * rows 2..12 random1, columns 1..10 attack level; "压" = suppression.
Vehicle damage correction (sheet 车辆战损结果修正):
  * random2 + state modifiers -> random band -> armour column -> rect value.

Armour classes: 0 无装甲, 1 轻型, 2 中型, 3 重型, 4 复合.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_CRT = Path(__file__).resolve().parent / "data" / "crt.json"
DEFAULT_ELEVATION = Path(__file__).resolve().parent / "data" / "elevation.json"

ARMOR_COLUMNS = {4: 3, 3: 4, 2: 5, 1: 6, 0: 7}  # raw grid column index

VEHICLE_SHOOTER_MODS = {"suppressed": -1, "moving": -1, "water": -1}
VEHICLE_TERRAIN_MODS = {"urban": -1, "forest": -2, "water": -2}
VEHICLE_TARGET_MODS = {"concealed": -2, "moving": -2, "stacked": 2, "marching": 4}

PERSONNEL_MOD_BANDS = [(-1, 0), (7, 0), (99, 1)]


def load_crt(path: Optional[str] = None) -> dict:
    p = Path(path) if path else DEFAULT_CRT
    return json.loads(p.read_text(encoding="utf-8"))


class CRT(object):
    def __init__(self, doc: dict):
        sheets = doc["sheets"]
        self.vehicle_grid = sheets["对车辆单位战斗结果"]
        self.personnel_grid = sheets["直瞄武器对人员and步兵轻武器对车辆战斗结果表"]
        self.correction_grid = sheets["车辆战损结果修正"]
        self.vehicle_columns: Dict[Tuple[int, int], int] = {}
        for blood in range(1, 6):
            row = self.vehicle_grid[blood]        # sheet rows 2..6 -> index 1..5
            for col in range(1, len(row)):
                level = row[col]
                if isinstance(level, int) and level >= 1:
                    self.vehicle_columns[(blood, level)] = col
        self.vehicle_rows: Dict[int, List] = {}
        for r in range(7, len(self.vehicle_grid)):
            random = self.vehicle_grid[r][0]
            if isinstance(random, int):
                self.vehicle_rows[random] = self.vehicle_grid[r]
        self.personnel_rows: Dict[int, List] = {}
        for r in range(1, len(self.personnel_grid)):
            random = self.personnel_grid[r][0]
            if isinstance(random, int):
                self.personnel_rows[random] = self.personnel_grid[r]
        self._bands = self._parse_bands()

    def _parse_bands(self):
        rows = []
        for r in range(2, len(self.correction_grid)):
            label = self.correction_grid[r][2]
            if label in (None, ""):
                continue
            rows.append((str(label), self.correction_grid[r]))
        return rows

    # -- vehicle CRT -------------------------------------------------------
    def vehicle_raw(self, attacks_blood: int, attack_level: int, random1: int) -> int:
        col = self.vehicle_columns.get((int(attacks_blood), int(attack_level)))
        row = self.vehicle_rows.get(int(random1))
        if col is None or row is None:
            return 0
        v = row[col] if col < len(row) else 0
        return int(v) if isinstance(v, int) else 0

    def vehicle_correction(self, random2: int, armor_class: int,
                           shooter_state: str = "normal",
                           target_terrain: str = "open",
                           target_state: str = "normal") -> int:
        mod = 0
        mod += VEHICLE_SHOOTER_MODS.get(shooter_state, 0)
        mod += VEHICLE_TERRAIN_MODS.get(target_terrain, 0)
        mod += VEHICLE_TARGET_MODS.get(target_state, 0)
        band = self._band_for(int(random2) + mod)
        if band is None:
            return 0
        col = ARMOR_COLUMNS.get(int(armor_class), 7)
        v = band[col] if col < len(band) else 0
        return int(v) if isinstance(v, int) else 0

    def _band_for(self, value: int):
        for label, row in self._bands:
            if _band_contains(label, value):
                return row
        return None

    # -- personnel CRT -----------------------------------------------------
    def personnel_result(self, attack_level: int, random1: int) -> Tuple[str, int]:
        """Return ('damage'|'suppressed'|'none', raw_damage)."""
        row = self.personnel_rows.get(int(random1))
        if row is None:
            return "none", 0
        col = int(attack_level)  # columns 1..10
        if col < 1 or col >= len(row):
            return "none", 0
        v = row[col]
        if v is None or v == "":
            return "none", 0
        if isinstance(v, str) and v.startswith("压"):
            return "suppressed", 0
        if isinstance(v, int):
            return ("damage", v) if v > 0 else ("none", 0)
        return "none", 0

    def personnel_correction(self, random2: int, shooter_state: str = "normal",
                             target_terrain: str = "open",
                             target_state: str = "normal") -> int:
        mod = 0
        mod += VEHICLE_SHOOTER_MODS.get(shooter_state, 0)
        mod += {"urban": -1, "forest": -2, "fort": -1}.get(target_terrain, 0)
        mod += {"concealed": -2, "moving": -2, "stacked": 2, "marching": 4}.get(target_state, 0)
        value = int(random2) + mod
        if value <= 0:
            return -1
        if value <= 7:
            return 0
        return 1


def _band_contains(label: str, value: int) -> bool:
    label = label.replace("，", ",")
    if label.startswith("<="):
        return value <= int(label[2:])
    if label.startswith(">="):
        return value >= int(label[2:])
    if "," in label:
        parts = [int(x) for x in label.split(",") if x.strip().lstrip("-").isdigit()]
        return value in parts
    if "-" in label[1:]:
        lo, hi = label.split("-")
        return int(lo) <= value <= int(hi)
    return int(label) == value
