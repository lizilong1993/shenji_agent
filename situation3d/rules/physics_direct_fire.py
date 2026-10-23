"""Physical direct-fire adjudication (optional ``direct_fire_model=physics``).

Ported (algorithm only) from Claude-of-Tanks ``src/sim/ballistics.ts`` and
``src/sim/damage.ts`` (MIT, Copyright © 2026 Kevin B. Liu).  The original
resolves a physical shell against a plate-level tank model; this port keeps
the penetration/angle/ricochet rules but uses the geo engine's five armour
classes (or an explicit ``armor_profile``) as RHAe thickness.

The default adjudication remains the authoritative 庙算 CRT in
``rules/combat.py``.  This module is only reached when a scenario asks for
``direct_fire_model: physics`` (or ``shadow``); it never changes the default
path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

from rules.context import RuleContext, RuleEntityView

DEFAULT_BALLISTICS = Path(__file__).resolve().parent / "data" / "ballistics.json"
_DOC = None

MIN_PEN_FRACTION = 0.75
PEN_ROLL_SPAN = 0.5

# Claude-of-Tanks shell behaviour table (armor doc §2–§5, §11).
SHELL_BEHAVIOR = {
    "AP": {"kind": "KE", "norm_deg": 5.0, "ricochet_deg": 70.0, "slope_exp": 1.4},
    "APCR": {"kind": "KE", "norm_deg": 2.0, "ricochet_deg": 70.0, "slope_exp": 1.4},
    "APFSDS": {"kind": "KE", "norm_deg": 2.0, "ricochet_deg": 78.0, "slope_exp": 1.0},
    "HEAT": {"kind": "CE", "norm_deg": 0.0, "ricochet_deg": 85.0, "slope_exp": 1.0},
    "HE": {"kind": "HE", "norm_deg": 0.0, "ricochet_deg": 180.0, "slope_exp": 1.0},
    "HESH": {"kind": "HE", "norm_deg": 0.0, "ricochet_deg": 180.0, "slope_exp": 1.0},
}

SIZE_RADIUS_M = {"small": 1.5, "medium": 3.0, "large": 5.0}
ARMOR_FACES = ("front", "side", "rear", "top")


def ballistics_doc() -> dict:
    global _DOC
    if _DOC is None:
        _DOC = json.loads(DEFAULT_BALLISTICS.read_text(encoding="utf-8"))
    return _DOC


def _num(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def profile_for_weapon(weapon) -> Optional[dict]:
    table = ballistics_doc().get("by_weapon_id", {})
    key = str(getattr(weapon, "weapon_id", "") or "")
    return table.get(key)


def has_profile(attacker: RuleEntityView) -> bool:
    for weapon in getattr(attacker, "weapons", []) or []:
        if getattr(weapon, "kind", "") == "direct" and profile_for_weapon(weapon) is not None:
            return True
    return False


def can_engage(attacker: RuleEntityView, target: RuleEntityView,
               ctx: RuleContext) -> bool:
    if not attacker.alive or not target.alive:
        return False
    if attacker.affiliation == target.affiliation:
        return False
    if ctx.target_kind(target) not in ("infantry", "vehicle"):
        return False
    distance_m = float(ctx.distance_m(attacker, target))
    return _select_weapon(attacker, target, ctx, distance_m) is not None


def armor_for(target: RuleEntityView) -> Dict[str, float]:
    table = ballistics_doc().get("armor_class_rha_mm", {})
    try:
        armor_class = max(0, min(4, int(getattr(target, "armor", 0))))
    except (TypeError, ValueError):
        armor_class = 0
    row = table.get(str(armor_class)) or table.get("0") or {}
    armor = {face: float(row.get(face, 0.0) or 0.0) for face in ARMOR_FACES}
    explicit = getattr(target, "armor_profile", None) or {}
    for face in ARMOR_FACES:
        if face in explicit and explicit[face] is not None:
            armor[face] = max(0.0, _num(explicit[face], armor[face]))
    return armor


def pen_at_distance(profile: dict, distance_m: float) -> float:
    p100 = _num(profile.get("pen_100m_mm"), 0.0)
    p1000 = _num(profile.get("pen_1000m_mm"), p100)
    p2000 = _num(profile.get("pen_2000m_mm"), 0.0)
    if distance_m > 1000.0 and p2000 > 0.0:
        f = min(1.0, (distance_m - 1000.0) / 1000.0)
        return p1000 + (p2000 - p1000) * f
    f = min(1.0, max(0.0, (distance_m - 100.0) / 900.0))
    return p100 + (p1000 - p100) * f


def _overmatch_caliber(profile: dict) -> float:
    explicit = profile.get("overmatch_caliber_mm")
    if explicit is not None:
        return max(0.0, _num(explicit, 0.0))
    caliber = max(0.0, _num(profile.get("caliber_mm"), 0.0))
    if str(profile.get("shell_type", "AP")) == "APFSDS":
        return caliber * 0.6
    return caliber


def _effective_thickness(profile: dict, thickness_mm: float,
                         plate_angle_deg: float) -> Tuple[float, float]:
    behavior = SHELL_BEHAVIOR.get(str(profile.get("shell_type", "AP")), SHELL_BEHAVIOR["AP"])
    norm = float(behavior["norm_deg"])
    caliber = _overmatch_caliber(profile)
    if thickness_mm > 0.0 and caliber >= 2.0 * thickness_mm:
        norm = norm * 1.4 * (caliber / thickness_mm)
    effective_angle = max(0.0, float(plate_angle_deg) - norm)
    clamped = min(effective_angle, 89.0)
    effective = thickness_mm / (math.cos(math.radians(clamped)) ** float(behavior["slope_exp"]))
    return effective, effective_angle


def _would_ricochet(profile: dict, thickness_mm: float, plate_angle_deg: float) -> bool:
    if thickness_mm <= 0.0:
        return False
    behavior = SHELL_BEHAVIOR.get(str(profile.get("shell_type", "AP")), SHELL_BEHAVIOR["AP"])
    caliber = _overmatch_caliber(profile)
    if behavior["kind"] == "KE" and caliber >= 3.0 * thickness_mm:
        return False
    return float(plate_angle_deg) > float(behavior["ricochet_deg"])


def _face_for_angle(angle_deg: float) -> str:
    if angle_deg <= 60.0:
        return "front"
    if angle_deg <= 120.0:
        return "side"
    return "rear"


def _plate_angle(face: str, hull_angle_deg: float) -> float:
    if face == "side":
        return abs(hull_angle_deg - 90.0)
    if face == "rear":
        return abs(180.0 - hull_angle_deg)
    return hull_angle_deg


def _aim_error_m(profile: dict, distance_m: float, rng) -> float:
    dispersion_mrad = max(0.0, _num(profile.get("dispersion_mrad"), 1.0))
    sigma_m = max(0.1, dispersion_mrad / 1000.0 * distance_m)
    return math.hypot(rng.gauss(0.0, sigma_m), rng.gauss(0.0, sigma_m))


def _select_weapon(attacker: RuleEntityView, target: RuleEntityView,
                   ctx: RuleContext, distance_m: float, weapon=None):
    try:
        firepower = float((getattr(attacker, "components", {}) or {}).get(
            "firepower", 1.0))
    except (TypeError, ValueError):
        firepower = 1.0
    if firepower <= 0.05:
        return None  # fire-control/firepower destroyed
    if weapon is not None:
        profile = profile_for_weapon(weapon)
        if profile is None:
            return None
        if not _ground_capable(weapon):
            return None
        max_range = _num(profile.get("max_range_m"), 3000.0)
        return (weapon, profile) if distance_m <= max_range else None
    best = None
    best_pen = -1.0
    for candidate in getattr(attacker, "weapons", []) or []:
        if candidate.kind != "direct":
            continue
        if not _ground_capable(candidate):
            continue
        if attacker.moving and not getattr(candidate, "shoot_on_move", False):
            continue
        if attacker.ammo_count(candidate.key) <= 0:
            continue
        if not ctx.weapon_ready(attacker.uid, candidate.key):
            continue
        profile = profile_for_weapon(candidate)
        if profile is None:
            continue
        max_range = _num(profile.get("max_range_m"), 3000.0)
        if distance_m > max_range:
            continue
        pen = pen_at_distance(profile, distance_m)
        if pen > best_pen:
            best_pen = pen
            best = (candidate, profile)
    return best


def _ground_capable(weapon) -> bool:
    """AA-only weapons (WpData without ground TarType) cannot fire direct."""
    tables = getattr(weapon, "tables", None)
    if not tables:
        return True
    target_types = set()
    for key in tables.keys():
        if isinstance(key, tuple) and key:
            target_types.add(key[0])
    return bool(target_types & {1, 2})


def resolve_physics_direct_fire(attacker: RuleEntityView, target: RuleEntityView,
                                ctx: RuleContext, dry_run: bool = False,
                                weapon=None, rng=None) -> Optional[dict]:
    if not attacker.alive or not target.alive:
        return None
    if attacker.affiliation == target.affiliation:
        return None
    if ctx.target_kind(target) not in ("infantry", "vehicle"):
        return None
    distance_m = float(ctx.distance_m(attacker, target))
    picked = _select_weapon(attacker, target, ctx, distance_m, weapon=weapon)
    if picked is None:
        return None
    weapon, profile = picked
    shell_type = str(profile.get("shell_type", "AP"))
    behavior = SHELL_BEHAVIOR.get(shell_type, SHELL_BEHAVIOR["AP"])

    rng = rng or ctx.rng
    target_radius = SIZE_RADIUS_M.get(str(getattr(target, "size_class", "medium")), 3.0)
    try:
        from rules.context import Cover
        cover = ctx.cover_at(target)
        target_radius *= {Cover.URBAN: 0.7, Cover.FOREST: 0.6,
                          Cover.BUILDING: 0.5}.get(cover, 1.0)
    except Exception:
        pass
    aim_error = _aim_error_m(profile, distance_m, rng)
    hit = aim_error <= target_radius
    velocity = max(1.0, _num(profile.get("velocity_mps"), 800.0))
    tof_s = distance_m / velocity

    heading = float(getattr(target, "heading_deg", 0.0) or 0.0)
    bearing_to_shooter = float(ctx.bearing_deg(target, attacker))
    hull_angle = abs((bearing_to_shooter - heading + 180.0) % 360.0 - 180.0)
    face = _face_for_angle(hull_angle)
    plate_angle = _plate_angle(face, hull_angle)
    armor = armor_for(target)
    thickness = float(armor.get(face, 0.0) or 0.0)
    effective_mm, effective_angle = _effective_thickness(profile, thickness, plate_angle)
    ricochet = hit and _would_ricochet(profile, thickness, plate_angle)
    pen_base = pen_at_distance(profile, distance_m)
    pen_roll = MIN_PEN_FRACTION + PEN_ROLL_SPAN * rng.random()
    pen_mm = pen_base * pen_roll

    penetrated = hit and not ricochet and (thickness <= 0.0 or pen_mm > effective_mm)
    damage = 0
    component = None
    component_health = None
    result = "miss"
    if hit:
        if ricochet:
            result = "ricochet"
        elif thickness <= 0.0:
            result = "penetration"
        elif pen_mm <= effective_mm:
            result = "no_penetration"
        else:
            result = "penetration"
    if penetrated:
        firepower = 1.0
        try:
            firepower = max(0.2, float((attacker.components or {}).get(
                "firepower", 1.0)))
        except (AttributeError, TypeError, ValueError):
            firepower = 1.0
        base_damage = max(0.0, _num(profile.get("blood_damage"), 1.0))
        ratio = pen_mm / max(1.0, effective_mm)
        damage = max(1, int(round(
            base_damage * firepower * min(2.0, ratio))))
        # Component hit (P1.5): mobility / firepower / sensors / structural.
        roll = rng.random()
        if roll < 0.35:
            component = "mobility"
        elif roll < 0.65:
            component = "firepower"
        elif roll < 0.80:
            component = "sensors"
        apply = getattr(ctx, "apply_component_damage", None)
        if component is not None and apply is not None and not dry_run:
            component_health = apply(target.uid, component,
                                     min(1.0, 0.35 * damage))

    if not dry_run:
        ctx.consume_ammo(attacker.uid, weapon.key, 1)
        ctx.mark_fired(attacker.uid, weapon.key, getattr(weapon, "cooldown_s", 75.0))
        if damage > 0:
            ctx.apply_damage(target.uid, damage)
        if behavior["kind"] == "HE" and hit and not penetrated:
            ctx.apply_suppression(target.uid, 30.0)

    return {
        "type": "direct_fire",
        "model": "physics",
        "actors": [attacker.uid, target.uid],
        "weapon": weapon.key,
        "distance_m": round(distance_m, 1),
        "tof_s": round(tof_s, 3),
        "aim_error_m": round(aim_error, 2),
        "target_radius_m": target_radius,
        "impact_angle_deg": round(hull_angle, 1),
        "plate_angle_deg": round(plate_angle, 1),
        "armor_face": face,
        "armor_mm": round(thickness, 1),
        "effective_angle_deg": round(effective_angle, 1),
        "effective_mm": round(effective_mm, 1),
        "pen_mm": round(pen_mm, 1),
        "pen_roll": round(pen_roll, 3),
        "shell_type": shell_type,
        "ricochet": ricochet,
        "result": result,
        "damage": damage,
        "component": component,
        "component_health": (None if component_health is None
                             else round(float(component_health), 3)),
        "suppressed": behavior["kind"] == "HE" and hit and not penetrated,
        "target_blood": getattr(target, "blood", 0),
        "dry_run": bool(dry_run),
    }
