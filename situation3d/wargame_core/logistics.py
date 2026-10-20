"""Logistics for the joint wargame core (synthetic).

Ship/port/airbase resupply, ground-unit fuel/ammo recovery, and transport
interdiction.  All rates are demonstration values.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import hexmap


# How close a supply source must be (km) for a unit to replenish.
PORT_SUPPLY_KM = 25.0
AIRBASE_SUPPLY_KM = 40.0
SHIP_SUPPLY_KM = 15.0
GROUND_SUPPLY_KM = 10.0

RATE_PER_HOUR = {
    "ammo": 0.15,
    "fuel": 0.25,
    "readiness": 0.05,
}


def _km(state, a: int, b: int) -> float:
    return hexmap.distance(a, b) * state.map.scale_m / 1000.0


def supply_sources(state) -> Dict[str, List[int]]:
    """Return {side: [hexes]} of friendly ports/airbases.

    Campaign scenarios can declare ``state.supply_points`` as a list of
    ``{color, pos, kind}``; this helper normalises that.
    """
    out: Dict[str, List[int]] = {}
    for sp in getattr(state, "supply_points", []) or []:
        color = sp.get("color")
        if color not in (0, 1):
            continue
        out.setdefault(str(color), []).append(int(sp.get("pos")))
    return out


def resupply(state, tick_seconds: float) -> List[dict]:
    """Apply per-tick resupply to units near a source.  Returns events."""
    events: List[dict] = []
    sources = supply_sources(state)
    hours = tick_seconds / 3600.0
    for u in state.living():
        if not getattr(u, "alive", True):
            continue
        color = str(u.color)
        hexes = sources.get(color) or []
        if not hexes:
            continue
        if u.domain in ("sea", "sub"):
            radius = PORT_SUPPLY_KM
        elif u.domain == "air":
            radius = AIRBASE_SUPPLY_KM
        else:
            radius = GROUND_SUPPLY_KM
        near = [p for p in hexes if _km(state, u.pos, p) <= radius]
        if not near:
            continue
        before = (dict(u.ammo), u.fuel, u.readiness)
        if u.unit_type in ("artillery", "aa") or u.domain == "land":
            # Ground units replenish slowly and only when stationary.
            if getattr(u, "moving", False):
                continue
            u.fuel = min(1.0, u.fuel + RATE_PER_HOUR["fuel"] * hours)
            u.readiness = min(1.0, u.readiness + RATE_PER_HOUR["readiness"] * hours)
        else:
            u.fuel = min(1.0, u.fuel + RATE_PER_HOUR["fuel"] * hours * 2.0)
            u.readiness = min(1.0, u.readiness + RATE_PER_HOUR["readiness"] * hours * 2.0)
            for k in list(u.ammo):
                base = _base_ammo(u, k)
                if u.ammo[k] < base:
                    u.ammo[k] = min(base, u.ammo[k] + max(1, int(base * 0.05 * hours * 60)))
        changed = (before[0] != u.ammo or abs(before[1] - u.fuel) > 1e-6
                   or abs(before[2] - u.readiness) > 1e-6)
        if changed:
            events.append({
                "tick": state.tick,
                "type": "resupply",
                "source": "engine",
                "actors": [u.uid],
                "location": list(hexmap.decode(u.pos)),
                "description": "%s 在补给点完成补给" % u.name,
                "raw": {"fuel": round(u.fuel, 2), "readiness": round(u.readiness, 2)},
            })
    return events


def _base_ammo(unit, key: str) -> int:
    for w in getattr(unit, "sea_air_weapons", []):
        if getattr(w, "key", "") == key:
            return int(getattr(w, "ammo", 8))
    for w in getattr(unit, "weapons", []):
        if getattr(w, "key", "") == key:
            return int(getattr(w, "ammo", 10))
    return 8


def consume_fuel(state, tick_seconds: float) -> None:
    """Drain fuel for moving units; a dry unit stops."""
    for u in state.living():
        if getattr(u, "domain", "") != "air":
            continue
        rate = 1.0 / max(1.0, float(u.raw.get("endurance_h", 6.0)))
        u.fuel = max(0.0, u.fuel - tick_seconds / 3600.0 * rate)


def interdiction_event(state, target, attacker) -> dict:
    """Record a logistics-interdiction event (transport cut off)."""
    return {
        "tick": state.tick,
        "type": "logistics_interdiction",
        "source": "engine",
        "level": "warn",
        "actors": [attacker.uid, target.uid],
        "location": list(hexmap.decode(target.pos)),
        "description": "%s 的运输队被 %s 拦截，前沿补给受影响" % (target.name, attacker.name),
        "raw": {"target_fuel": round(target.fuel, 2)},
    }
