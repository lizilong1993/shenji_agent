"""Authoritative weapon tables (庙算 ``WpData``) and table lookups.

Semantics verified by black-box runs against train_env 4.1.0
(``tools/geo_p0/probe_env.py``, 2026-10-21):

======  ==========================================
TarType meaning
======  ==========================================
0       间瞄对地 (indirect fire, ground targets)
1       对人员 (personnel); ``ObjBlood`` = the **attacker's** vehicle/squad
        count (1..5), not the target's blood
2       对车辆 (vehicles; ObjBlood = 0)
3       对空   (air targets; ObjBlood = 0)
======  ==========================================

Evidence: firing weapon 36 from attackers with blood 1/2/3/4/5 at an
infantry target produced attack levels 2/3/5/7/9 -- the TarType=1 rows for
ObjBlood 1..5 -- while the target's blood stayed 3.  A recorded ``judge_info``
shot of weapon 36 against a vehicle at 14/15/16/18 hexes produced 6/5/4/2,
matching the TarType=2 row value-for-value.

The old data dictionary had 1/2 swapped, and the current MVP
``wargame_core.weapons`` stores only two flat lists per weapon and truncates
them at ``att_max``; this module keys by ``(weapon_id, tar_type, obj_blood)``
and rejects malformed tables instead of silently clamping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rules.context import WeaponTable

TAR_TYPE_INDIRECT = 0
TAR_TYPE_PERSONNEL = 1
TAR_TYPE_VEHICLE = 2
TAR_TYPE_AIR = 3

TAR_TYPE_BY_KIND = {
    "infantry": TAR_TYPE_PERSONNEL,
    "vehicle": TAR_TYPE_VEHICLE,
    "air": TAR_TYPE_AIR,
}

DEFAULT_WPDATA = Path(__file__).resolve().parent / "data" / "wpdata.json"


class WeaponDataError(ValueError):
    pass


def load_wpdata(path: Optional[str] = None) -> dict:
    p = Path(path) if path else DEFAULT_WPDATA
    return json.loads(p.read_text(encoding="utf-8"))


def _row_table(row: dict) -> WeaponTable:
    att_min = int(row["att_min"])
    att_max = int(row["att_max"])
    if att_min < 0 or att_max < att_min:
        raise WeaponDataError(
            "bad range for %s: att_min=%s att_max=%s" % (
                row.get("name"), att_min, att_max))
    raw = row.get("values") or {}
    missing = [d for d in range(0, att_max + 1) if str(d) not in raw]
    if missing:
        raise WeaponDataError(
            "table truncated for %s tar=%s blood=%s: missing distances %s" % (
                row.get("name"), row.get("tar_type"), row.get("obj_blood"),
                missing[:8]))
    values = [int(raw[str(d)]) for d in range(0, att_max + 1)]
    return WeaponTable(att_min=att_min, att_max=att_max, values=tuple(values))


class WeaponTables(object):
    """Indexed view over ``wpdata.json``."""

    def __init__(self, doc: dict):
        self.rows = doc["rows"]
        self._by_key: Dict[Tuple[int, int, int], dict] = {}
        self._by_weapon: Dict[int, List[dict]] = {}
        self.issues: List[str] = []
        for row in self.rows:
            wid_text = str(row["weapon_id"]).strip()
            if not wid_text:
                self.issues.append("row %s: empty weapon_id" % row.get("combo_id"))
                continue
            for token in wid_text.replace("，", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    wid = int(token)
                except ValueError:
                    self.issues.append("row %s: bad weapon_id %r" % (row.get("combo_id"), token))
                    continue
                key = (wid, int(row["tar_type"]), int(row["obj_blood"]))
                if key in self._by_key:
                    self.issues.append("duplicate key %s for row %s" % (key, row.get("combo_id")))
                self._by_key[key] = row
                self._by_weapon.setdefault(wid, []).append(row)

    def weapon_ids(self) -> List[int]:
        return sorted(self._by_weapon)

    def row(self, weapon_id: int, tar_type: int, obj_blood: int) -> dict:
        key = (int(weapon_id), int(tar_type), int(obj_blood))
        if key not in self._by_key and int(tar_type) in (TAR_TYPE_VEHICLE, TAR_TYPE_AIR):
            # WpData stores vehicle/air rows with ObjBlood=0; callers must not
            # have to know that.  Personnel/indirect keep the caller's blood.
            key = (int(weapon_id), int(tar_type), 0)
        if key not in self._by_key:
            raise WeaponDataError("no WpData row for %s" % (key,))
        return self._by_key[key]

    def table(self, weapon_id: int, tar_type: int, obj_blood: int) -> WeaponTable:
        return _row_table(self.row(weapon_id, tar_type, obj_blood))

    def indirect_table(self, weapon_id: int, obj_blood: int = 0) -> WeaponTable:
        """Indirect-fire (TarType=0) table for artillery (ids 72/88/89)."""
        return self.table(weapon_id, TAR_TYPE_INDIRECT, obj_blood)

    def table_for_target(self, weapon_id: int, target_kind: str,
                         stack_count: int) -> WeaponTable:
        """Table for a shot.

        ``stack_count`` is the attacker's vehicle/squad count; it only
        matters for ``target_kind == "infantry"`` (TarType=1).  Vehicle and
        air rows ignore it and are normalised to ObjBlood=0.
        """
        tar_type = TAR_TYPE_BY_KIND.get(target_kind)
        if tar_type is None:
            raise WeaponDataError("unknown target kind %r" % (target_kind,))
        return self.table(weapon_id, tar_type, stack_count)

    def tables_for_weapon(self, weapon_id: int) -> Dict[Tuple[int, int], WeaponTable]:
        out = {}
        for row in self._by_weapon.get(int(weapon_id), []):
            out[(int(row["tar_type"]), int(row["obj_blood"]))] = _row_table(row)
        return out

    def validate_all(self) -> List[str]:
        """Return one message per malformed table (empty means all good)."""
        problems = []
        for (wid, tar, blood), row in sorted(self._by_key.items()):
            try:
                _row_table(row)
            except WeaponDataError as exc:
                problems.append(str(exc))
        return problems


def attack_level(table: WeaponTable, distance_hex: int) -> Optional[int]:
    """Attack level at a rule-grid distance; ``None`` means out of range."""
    return table.level_at(int(distance_hex))


def attack_level_for(tables: WeaponTables, weapon_id: int, target_kind: str,
                     stack_count: int, distance_hex: int) -> Optional[int]:
    return attack_level(tables.table_for_target(weapon_id, target_kind,
                                                stack_count), distance_hex)
