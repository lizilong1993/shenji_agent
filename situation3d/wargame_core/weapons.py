"""Weapon profiles and direct-fire attack levels.

The attack-level tables below are transcribed from the public 庙算 adjudication
tables (``docs/裁决表格.xlsx``, sheets "对人员攻击等级表" and "对车辆攻击等级表").
Only the first row (one vehicle/squad) is used in this MVP; a fuller model can
scale by the shooter's stack size later.

These are *rule data*, not code copied from the closed engine.  Where the
public table does not give a value (e.g. beyond a weapon's range), the entry is
zero and the shot is invalid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class WeaponProfile:
    key: str
    name: str
    max_range: int
    vs_personnel: List[int]
    vs_vehicle: List[int]
    kind: str = "direct"       # direct | indirect | mg
    ammo: int = 10
    # Synthetic effective range in km for campaign maps.  On the public
    # 200 m/hex maps ``max_range`` (hexes) is used instead.
    range_km: float = 2.0

    def attack_level(self, distance: int, target_kind: str) -> int:
        if distance < 0 or distance > self.max_range:
            return 0
        table = self.vs_personnel if target_kind in ("infantry",) else self.vs_vehicle
        if not table:
            return 0
        idx = min(distance, len(table) - 1)
        return int(table[idx])

    def attack_level_at(self, distance_hex: int, scale_m: float,
                        target_kind: str) -> int:
        """Attack level for a real hex distance under either scale regime.

        Campaign maps rescale the documented hex table by the ratio of the
        real distance to the weapon's synthetic range, so the table keeps its
        "closer = stronger" shape without implying 50 km tank shots.
        """
        if scale_m <= 500.0:
            return self.attack_level(distance_hex, target_kind)
        dist_km = distance_hex * scale_m / 1000.0
        if dist_km > self.range_km:
            return 0
        table = self.vs_personnel if target_kind in ("infantry",) else self.vs_vehicle
        if not table:
            return 0
        frac = 0.0 if self.range_km <= 0 else dist_km / self.range_km
        idx = min(len(table) - 1, max(0, int(round(frac * (self.max_range - 1)))))
        return int(table[idx])


PROFILES: Dict[str, WeaponProfile] = {
    "cannon_large": WeaponProfile(
        "cannon_large", "大号直瞄炮", 18,
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [10, 10, 10, 10, 10, 9, 9, 9, 9, 8],
        ammo=8, range_km=4.0),
    "cannon_medium": WeaponProfile(
        "cannon_medium", "中号直瞄炮", 15,
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [10, 10, 10, 9, 9, 9, 8, 8, 7, 7],
        ammo=10, range_km=3.0),
    "cannon_small": WeaponProfile(
        "cannon_small", "小号直瞄炮", 13,
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [10, 10, 9, 9, 8, 8, 7, 6, 6, 5],
        ammo=12, range_km=2.5),
    "autocannon": WeaponProfile(
        "autocannon", "速射炮", 10,
        [4, 4, 4, 4, 4, 3, 3, 3, 3, 3],
        [5, 5, 4, 4, 3, 3, 3, 2, 2, 2],
        ammo=16, range_km=2.0),
    "mg": WeaponProfile(
        "mg", "车载轻武器", 10,
        [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        kind="mg", ammo=30, range_km=1.2),
    "small_arms": WeaponProfile(
        "small_arms", "步兵轻武器", 3,
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ammo=30, range_km=0.6),
    "atgm": WeaponProfile(
        "atgm", "重型导弹", 20,
        [4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [0, 0, 5, 5, 6, 8, 8, 8, 8, 8],
        ammo=4, range_km=4.5),
    "portable_atgm": WeaponProfile(
        "portable_atgm", "便携导弹", 10,
        [4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
        ammo=3, range_km=3.0),
    "aa_gun": WeaponProfile(
        "aa_gun", "防空高炮", 10,
        [4, 4, 4, 4, 4, 3, 3, 3, 3, 3],
        [5, 5, 4, 4, 3, 3, 3, 2, 2, 2],
        ammo=20, range_km=4.0),
    "artillery": WeaponProfile(
        "artillery", "间瞄火炮", 20,
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        kind="indirect", ammo=12, range_km=25.0),
}


# Public platform names -> weapon keys.  This is intentionally a small,
# explicit mapping so a reader can see which profile each platform uses.
NAME_WEAPONS = {
    "重型坦克": ["cannon_large", "mg"],
    "中型坦克": ["cannon_medium", "mg"],
    "重型战车": ["autocannon", "atgm"],
    "中型战车": ["autocannon", "atgm"],
    "轻型战车": ["autocannon", "mg"],
    "步兵": ["small_arms", "portable_atgm"],
    "步兵小队": ["small_arms", "portable_atgm"],
    "炮兵": ["artillery"],
    "无人战车": ["autocannon"],
    "侦察型战车": ["autocannon", "mg"],
    "防空导弹排": ["portable_atgm"],
    "防空高炮": ["aa_gun"],
    "火箭布雷车": ["mg"],
    "扫雷车": ["mg"],
    "运输机": [],
    "无人机": [],
    "武装直升机": ["cannon_small", "atgm"],
    "巡飞弹": ["cannon_small"],
    "人员工事": ["small_arms"],
    "车辆工事": ["autocannon"],
}


def weapons_for(name: str, op_type: Optional[int] = None,
                sub_type: Optional[int] = None) -> List[WeaponProfile]:
    keys = NAME_WEAPONS.get(name)
    if keys is None:
        # Fall back by operator type: infantry / vehicle / aircraft / fort.
        if op_type == 1:
            keys = ["small_arms", "portable_atgm"]
        elif op_type == 2:
            keys = ["autocannon", "mg"]
        elif op_type == 4:
            keys = ["autocannon"]
        else:
            keys = []
    return [PROFILES[k] for k in keys if k in PROFILES]
