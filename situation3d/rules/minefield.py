"""Minefield adjudication from the authoritative 雷场 tables (P4).

``rules/data/minefield.json`` stores the 庙算 ``minefield_*.csv`` tables:

* ``personnel`` / ``vehicle``: rows keyed by minefield type, columns are the
  random number (2d6, 2..12) -> damage;
* ``vehicle_calibration``: columns are 2d6, rows are armour classes
  (复合/重型/中型/轻型/无装甲) -> damage correction.

A trigger rolls 2d6 once per entity per field; the result is the base damage
plus (vehicles only) the armour correction, never below zero.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional

DEFAULT_MINEFIELD = Path(__file__).resolve().parent / "data" / "minefield.json"

ARMOR_LABELS = {4: "复合装甲", 3: "重型装甲", 2: "中型装甲",
                1: "轻型装甲", 0: "无装甲"}


class MinefieldError(ValueError):
    pass


def load_minefield(path: Optional[str] = None) -> dict:
    p = Path(path) if path else DEFAULT_MINEFIELD
    return json.loads(p.read_text(encoding="utf-8"))


def _table_rows(sheet):
    header = [str(v) for v in sheet[0]]
    out = {}
    for row in sheet[1:]:
        name = str(row[0])
        values = {}
        for i in range(1, len(row)):
            try:
                roll = int(float(header[i]))
            except ValueError:
                continue
            if roll < 1 or roll > 12:
                continue
            try:
                values[roll] = int(float(row[i]))
            except (TypeError, ValueError):
                continue
        out[name] = values
    return out


class MinefieldTables(object):
    def __init__(self, doc: Optional[dict] = None):
        doc = doc or load_minefield()
        self.personnel = _table_rows(doc["personnel"])
        self.vehicle = _table_rows(doc["vehicle"])
        self.calibration: Dict[str, Dict[int, int]] = {}
        cal = doc.get("vehicle_calibration") or []
        if cal:
            header = [str(v) for v in cal[0]]
            for row in cal[1:]:
                label = str(row[0])
                values = {}
                for i in range(1, len(row)):
                    try:
                        roll = int(float(header[i]))
                    except ValueError:
                        continue
                    try:
                        values[roll] = int(float(row[i]))
                    except (TypeError, ValueError):
                        continue
                self.calibration[label] = values

    def resolve(self, kind: str, armor_class: int, rng: random.Random,
                field_type: str = "临时雷场") -> dict:
        table = self.personnel if kind == "infantry" else self.vehicle
        if field_type not in table:
            field_type = sorted(table)[0] if table else None
        roll = rng.randint(1, 6) + rng.randint(1, 6)
        base = int((table.get(field_type) or {}).get(roll, 0))
        rect = 0
        label = None
        if kind != "infantry":
            label = ARMOR_LABELS.get(int(armor_class))
            if label is not None:
                rect = int((self.calibration.get(label) or {}).get(roll, 0))
        damage = base + rect
        if damage < 0:
            damage = 0
        return {"roll": roll, "field_type": field_type, "base": base,
                "armor": label, "rect": rect, "damage": damage}


_TABLES = None


def tables() -> MinefieldTables:
    global _TABLES
    if _TABLES is None:
        _TABLES = MinefieldTables()
    return _TABLES
