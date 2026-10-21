"""Rule/physics boundary: protocols only (P2 skeleton).

``rules/`` must not import ``env/``, ``sim/`` or ``geo/``.  The physics side
implements these protocols; the rules side consumes them.  This keeps the
adjudication logic independently testable and comparable to 庙算 data
(ADR-005).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Protocol, Tuple


class Cover(IntEnum):
    NONE = 0
    URBAN = 1        # attack level -1
    FOREST = 2       # attack level -2
    BUILDING = 2     # target inside a building: stronger of urban/forest


@dataclass(frozen=True)
class WeaponTable(object):
    """One ``(WeaponID, TarType, ObjBlood)`` attack-level row.

    ``values[d]`` is the attack level at rule-grid distance ``d``; the table
    must cover ``0..att_max`` exactly (P0 diff report showed the MVP tables
    are truncated, which silently pins far shots to near values).
    """
    att_min: int
    att_max: int
    values: Tuple[int, ...]

    def level_at(self, distance_hex: int) -> Optional[int]:
        if distance_hex < self.att_min or distance_hex > self.att_max:
            return None
        return self.values[distance_hex]


@dataclass(frozen=True)
class WeaponView(object):
    key: str
    kind: str                       # direct | indirect | mg | sam | ...
    shoot_on_move: bool = False
    cooldown_s: float = 75.0
    ammo: int = 0
    tables: Dict[Tuple[int, int], WeaponTable] = field(default_factory=dict)


class RuleEntityView(Protocol):
    uid: int
    name: str
    affiliation: int
    domain: str                     # land | air | surface | subsurface
    unit_type: str                  # infantry | vehicle | artillery | aa | ...
    unit_kind: str                  # infantry | vehicle | recon | aa | helicopter | ...
    alive: bool
    armor: int
    blood: int
    max_blood: int
    concealed: bool
    moving: bool
    stacked: bool
    marching: bool
    suppressed: bool
    readiness: float
    value: int
    ammo: Dict[str, int]

    def ammo_count(self, key: str) -> int: ...


class RuleContext(Protocol):
    """Every spatial/state input the rules are allowed to see."""

    now_s: float

    # spatial inputs
    def distance_m(self, a: RuleEntityView, b: RuleEntityView) -> float: ...
    def rule_distance(self, a: RuleEntityView, b: RuleEntityView) -> int: ...
    def elevation_delta_m(self, a: RuleEntityView, b: RuleEntityView) -> float: ...
    def elevation_grade(self, e: RuleEntityView) -> int: ...
    def ele_grade(self) -> float: ...
    def line_of_sight(self, a: RuleEntityView, b: RuleEntityView) -> bool: ...
    def cover_at(self, e: RuleEntityView) -> Cover: ...
    def target_kind(self, e: RuleEntityView) -> str: ...      # infantry | vehicle | air
    def target_blood(self, e: RuleEntityView) -> int: ...

    # state / timers
    def can_detect(self, observer: RuleEntityView, target: RuleEntityView) -> bool: ...
    def entities(self, side: Optional[int] = None) -> List[RuleEntityView]: ...
    def units_in_cell(self, cell: Tuple[int, int]) -> List[RuleEntityView]: ...
    def side_detects(self, target: RuleEntityView) -> bool: ...
    def apply_damage(self, uid: int, damage: int) -> None: ...
    def apply_suppression(self, uid: int, seconds: float) -> None: ...
    def side_jam(self, affiliation: int) -> float: ...
    def weapon_ready(self, uid: int, key: str) -> bool: ...
    def weapon_state(self, uid: int, key: str) -> str: ...
    def mark_fired(self, uid: int, key: str, cooldown_s: float) -> None: ...
    def consume_ammo(self, uid: int, key: str, n: int = 1) -> bool: ...
