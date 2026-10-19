"""Observation/detection rules for the clean-room land wargame core.

The public observation table is small and explicit; it is reproduced here as
data, not copied code.  Terrain concealment and the "high ground sees over
concealment" exception follow the public rule text.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from . import hexmap
from .los import LineOfSight


# Observer -> target observation distance in hexes (public table).
OBSERVE_TABLE: Dict[str, Dict[str, int]] = {
    "infantry": {"infantry": 10, "vehicle": 25, "helicopter": 25, "uav": 1},
    "vehicle": {"infantry": 10, "vehicle": 25, "helicopter": 25, "uav": 1},
    "helicopter": {"infantry": 10, "vehicle": 25, "helicopter": 25, "uav": 1},
    "uav": {"infantry": 2, "vehicle": 2, "helicopter": 0, "uav": 0},
}


def _kind(unit) -> str:
    kind = str(getattr(unit, "unit_type", "vehicle"))
    if kind in ("infantry", "vehicle", "helicopter", "uav"):
        return kind
    if kind in ("artillery", "aa", "recon", "engineering"):
        return "vehicle"
    if kind in ("aircraft",):
        return "helicopter"
    return "vehicle"


def observe_range(observer, target, hmap: hexmap.HexMap) -> int:
    """Effective observation range in hexes after terrain concealment."""
    table = OBSERVE_TABLE.get(_kind(observer), OBSERVE_TABLE["vehicle"])
    base = table.get(_kind(target), 25)
    if base <= 0:
        return 0
    # Concealed units are observed at half range, unless the observer is on
    # higher ground (public exception).
    if getattr(target, "concealed", False):
        if hmap.elev(observer.pos) <= hmap.elev(target.pos):
            base = max(1, base // 2)
    # Urban/forest target hex halves observation range when visible.
    if hmap.terrain_cover(target.pos) > 0:
        base = max(1, base // 2)
    return base


def can_observe(observer, target, hmap: hexmap.HexMap,
                los: Optional[LineOfSight] = None) -> bool:
    if observer is target:
        return False
    if getattr(observer, "alive", True) is False or getattr(target, "alive", True) is False:
        return False
    rng = observe_range(observer, target, hmap)
    if rng <= 0:
        return False
    d = hexmap.distance(observer.pos, target.pos)
    if d > rng:
        return False
    los = los or LineOfSight(hmap)
    return los.visible(observer.pos, target.pos)


def detected_by(state, target) -> List[object]:
    """All living enemy units that can currently see ``target``."""
    out = []
    for u in state.units.values():
        if not getattr(u, "alive", True) or u.color == target.color:
            continue
        if can_observe(u, target, state.map, state.los):
            out.append(u)
    return out


def visible_enemies(state, observer) -> List[object]:
    out = []
    for u in state.units.values():
        if not getattr(u, "alive", True) or u.color == observer.color:
            continue
        if can_observe(observer, u, state.map, state.los):
            out.append(u)
    return out
