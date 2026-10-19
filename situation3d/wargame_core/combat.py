"""Direct-fire adjudication for the clean-room land wargame core.

The model is intentionally transparent:

    attack_level = weapon table(distance, target kind) + modifiers
    roll         = 2d6
    score        = attack_level + (roll - 7) - armour
    result       = damage / suppression / no effect

The public tables give attack levels and a combat-result matrix.  This MVP
uses a compact score mapping derived from that matrix; the spec marks it as an
approximation.  A later version can replace ``_result`` with the full matrix
without changing the rest of the engine.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import detection, hexmap, weapons


def _elevation_modifier(attacker, target, hmap: hexmap.HexMap) -> int:
    dh = hmap.elev(attacker.pos) - hmap.elev(target.pos)
    if dh == 0:
        return 0
    if dh < 0:
        # firing uphill is penalised; public table is negative
        return -min(5, int(abs(dh) / max(1.0, hmap.ele_grade) / 2.0) + 1)
    return min(2, int(dh / max(1.0, hmap.ele_grade) / 2.0) + 1)


def attack_modifiers(attacker, target, hmap: hexmap.HexMap) -> Dict[str, int]:
    mods: Dict[str, int] = {}
    mods["elevation"] = _elevation_modifier(attacker, target, hmap)
    cover = hmap.terrain_cover(target.pos)
    if cover:
        mods["cover"] = -cover
    if getattr(target, "concealed", False):
        mods["concealed"] = -1
    if getattr(target, "moving", False):
        mods["target_moving"] = -1
    if getattr(target, "stacked", False):
        mods["target_stacked"] = 2
    if getattr(target, "marching", False):
        mods["target_marching"] = 4
    if getattr(attacker, "moving", False):
        mods["shooter_moving"] = -1
    if getattr(attacker, "suppressed", False):
        mods["shooter_suppressed"] = -1
    return mods


def _target_kind(unit) -> str:
    return "infantry" if getattr(unit, "unit_type", "") == "infantry" else "vehicle"


def _result(attack_level: int, roll: int, target, target_kind: str) -> Tuple[str, int]:
    """Return (result, damage).  ``result`` is damage/suppressed/none."""
    armour = int(getattr(target, "armor", 0) or 0)
    score = attack_level + (roll - 7) - armour
    if target_kind == "infantry":
        # Personnel have no armour in this MVP; blood is squads.
        if score >= 6:
            return "damage", 1
        if score >= 3:
            return "suppressed", 0
        return "none", 0
    if score >= 9:
        return "damage", 2
    if score >= 6:
        return "damage", 1
    if score >= 3:
        return "suppressed", 0
    return "none", 0


def select_weapon(attacker, target, hmap: hexmap.HexMap) -> Optional[weapons.WeaponProfile]:
    d = hexmap.distance(attacker.pos, target.pos)
    best = None
    best_level = 0
    for w in getattr(attacker, "weapons", []):
        if w.kind == "indirect":
            continue
        if getattr(attacker, "ammo", {}).get(w.key, 0) <= 0:
            continue
        level = w.attack_level(d, _target_kind(target))
        if level > best_level:
            best_level = level
            best = w
    return best


def can_fire(attacker, target, state) -> bool:
    if not getattr(attacker, "alive", True) or not getattr(target, "alive", True):
        return False
    if attacker.color == target.color:
        return False
    if getattr(attacker, "suppressed", False):
        return False
    if getattr(attacker, "moving", False) and getattr(attacker, "unit_type", "") not in ("vehicle",):
        # Only vehicle-mounted direct-fire weapons may shoot on the move in
        # this MVP; infantry/artillery must stop first.
        return False
    w = select_weapon(attacker, target, state.map)
    if w is None:
        return False
    if not detection.can_observe(attacker, target, state.map, state.los):
        return False
    return True


def resolve_direct_fire(attacker, target, state) -> Optional[dict]:
    """Resolve one direct-fire shot.  Returns an event dict or ``None``."""
    if not can_fire(attacker, target, state):
        return None
    w = select_weapon(attacker, target, state.map)
    if w is None:
        return None
    d = hexmap.distance(attacker.pos, target.pos)
    base = w.attack_level(d, _target_kind(target))
    mods = attack_modifiers(attacker, target, state.map)
    effective = max(0, base + sum(mods.values()))
    roll = state.rng.randint(2, 12)
    result, damage = _result(effective, roll, target, _target_kind(target))

    attacker.ammo[w.key] = max(0, attacker.ammo.get(w.key, 0) - 1)
    if damage > 0:
        target.blood = max(0, target.blood - damage)
        if target.blood <= 0:
            target.alive = False
    if result == "suppressed" and target.alive:
        target.suppressed = True
        target.suppressed_ticks = max(target.suppressed_ticks, 2)

    event = {
        "id": "EV-%06d" % (state.next_event_id(),),
        "tick": state.tick,
        "type": "direct_fire",
        "source": "engine",
        "level": "warn" if damage else "info",
        "actors": [attacker.uid, target.uid],
        "location": list(hexmap.decode(target.pos)),
        "weapon": w.name,
        "attack_level": base,
        "effective_attack_level": effective,
        "modifiers": mods,
        "roll": roll,
        "result": result,
        "damage": damage,
        "target_blood": target.blood,
        "description": "%s 对 %s 使用%s射击：%s%s" % (
            attacker.name, target.name, w.name,
            {"damage": "毁伤 %d" % damage,
             "suppressed": "压制",
             "none": "无效果"}[result],
            "，目标被摧毁" if not target.alive else ""),
        "raw": {"distance": d},
    }
    state.events.append(event)
    return event


def best_target(attacker, state) -> Optional[object]:
    """Nearest visible enemy within any direct-fire weapon range."""
    best = None
    best_score = None
    for u in state.units.values():
        if not getattr(u, "alive", True) or u.color == attacker.color:
            continue
        w = select_weapon(attacker, u, state.map)
        if w is None:
            continue
        if not detection.can_observe(attacker, u, state.map, state.los):
            continue
        d = hexmap.distance(attacker.pos, u.pos)
        score = (d, -int(getattr(u, "value", 0) or 0))
        if best_score is None or score < best_score:
            best_score = score
            best = u
    return best
