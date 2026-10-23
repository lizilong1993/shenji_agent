"""Missile flight segments and terminal defence (P2).

Sea/air engagements no longer apply damage at the launch tick: the launch
event carries a time of flight, the impact is scheduled in the world and a
terminal-defence roll (target SAM/CIWS/AAA) can defeat the round.  The Pk
from ``rules.engagement`` remains the planned hit probability; this module
adds kinematics and intercept only.
"""

from __future__ import annotations

import math

from rules.context import RuleContext, RuleEntityView

# Cruise speeds (km/h) for the sea/air weapon kinds.
MISSILE_SPEED_KMH = {
    "asm": 1000.0,
    "asm_light": 950.0,
    "coastal_asm": 1000.0,
    "land_attack": 900.0,
    "sam": 3000.0,
    "sam_long": 3000.0,
    "sam_medium": 2500.0,
    "bvr": 4000.0,
    "wvr": 2500.0,
    "torpedo": 60.0,
    "depth_charge": 30.0,
    "asroc": 900.0,
    "naval_gun": 1200.0,
    "air_ground": 900.0,
    "aaa": 1500.0,
    "ciws": 1500.0,
}

# Terminal-defence kinds that can engage an incoming round.
_INTERCEPT_KINDS = ("sam", "sam_long", "sam_medium", "ciws", "aaa")


def tof_s(weapon_kind: str, distance_km: float) -> float:
    """Time of flight in seconds (minimum 5 s for presentation)."""
    speed = max(30.0, float(MISSILE_SPEED_KMH.get(weapon_kind, 900.0)))
    return max(5.0, float(distance_km) / speed * 3600.0)


def intercept_probability(target: RuleEntityView, threat_kind: str,
                          distance_km: float, ctx: RuleContext) -> float:
    """Combined terminal-defence probability against one incoming round."""
    if threat_kind not in ("asm", "asm_light", "coastal_asm", "bvr", "wvr",
                           "air_ground", "land_attack", "asroc"):
        return 0.0
    p_miss = 1.0
    readiness = float(getattr(target, "readiness", 1.0))
    if readiness is None:
        readiness = 1.0
    for weapon in getattr(target, "sea_air_weapons", []) or []:
        kind = str(getattr(weapon, "kind", ""))
        if kind not in _INTERCEPT_KINDS:
            continue
        max_range = float(getattr(weapon, "max_range_km", 0.0))
        min_range = float(getattr(weapon, "min_range_km", 0.0))
        # Terminal defence engages the incoming round inside its own
        # envelope, not at the attacker's launch range.
        terminal = min(float(distance_km), max_range)
        if terminal > max_range or terminal < min_range:
            continue
        if getattr(target, "ammo_count", lambda key: 0)(weapon.key) <= 0:
            continue
        if not ctx.weapon_ready(target.uid, weapon.key):
            continue
        p = float(getattr(weapon, "base_pk", 0.0)) * readiness
        p = max(0.0, min(0.85, p))
        p_miss *= (1.0 - p)
    chance = 1.0 - p_miss
    return max(0.0, min(0.9, chance))


def terminal_roll(ctx: RuleContext, intercept_pk: float) -> bool:
    """True when the missile is intercepted before impact."""
    if intercept_pk <= 0.0:
        return False
    return ctx.rng.random() < intercept_pk
