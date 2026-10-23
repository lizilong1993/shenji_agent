"""Direct-fire adjudication using the authoritative 庙算 tables (P4).

Pipeline per shot:
    attack_level = weapon table[(TarType, ObjBlood)] at rule distance
                   + elevation modifier (uphill only)
                   - cover (urban 1 / forest 2)
    random1      = 2d6 (attack roll)
    random2      = 2d6 (damage-correction roll)
    raw damage   = vehicle CRT / personnel CRT at (stack, attack_level, random1)
    correction   = vehicle/personnel correction table at random2 (+state mods)
    damage       = raw + correction  (values < 0 mean no effect, reported as -1)
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from rules.context import Cover, RuleContext, RuleEntityView, WeaponView
from rules.crt import CRT, load_crt
from rules.elevation import elevation_modifier, load_elevation

_CRT = None
_ELEV = None


def crt() -> CRT:
    global _CRT
    if _CRT is None:
        _CRT = CRT(load_crt())
    return _CRT


def elevation_doc() -> dict:
    global _ELEV
    if _ELEV is None:
        _ELEV = load_elevation()
    return _ELEV


def target_cover(ctx: RuleContext, target: RuleEntityView) -> int:
    cover = ctx.cover_at(target)
    if cover == Cover.URBAN:
        return 1
    if cover in (Cover.FOREST, Cover.BUILDING):
        return 2
    return 0


def select_weapon(attacker: RuleEntityView, target: RuleEntityView,
                  ctx: RuleContext) -> Optional[Tuple[WeaponView, object]]:
    distance = ctx.rule_distance(attacker, target)
    kind = ctx.target_kind(target)
    stack = ctx.stack_count(attacker)
    best = None
    best_level = None
    for weapon in getattr(attacker, "weapons", []) or []:
        if weapon.kind != "direct":
            continue
        if attacker.moving and not getattr(weapon, "shoot_on_move", False):
            continue
        if attacker.ammo_count(weapon.key) <= 0:
            continue
        if not ctx.weapon_ready(attacker.uid, weapon.key):
            continue
        table = (weapon.tables or {}).get(
            (1 if kind == "infantry" else 2 if kind == "vehicle" else 3, stack))
        if table is None:
            # vehicle/air tables are stored with ObjBlood=0
            table = (weapon.tables or {}).get(
                (1 if kind == "infantry" else 2 if kind == "vehicle" else 3, 0))
        if table is None:
            continue
        level = table.level_at(distance)
        if level is None:
            continue
        if best_level is None or level > best_level:
            best_level = level
            best = (weapon, table)
    return best


def resolve_direct_fire(attacker: RuleEntityView, target: RuleEntityView,
                        ctx: RuleContext) -> Optional[dict]:
    if not attacker.alive or not target.alive:
        return None
    if attacker.affiliation == target.affiliation:
        return None
    if ctx.target_kind(target) not in ("infantry", "vehicle"):
        return None
    model = getattr(ctx, "direct_fire_model", "crt")
    physics_direct_fire = None
    if model in ("physics", "shadow"):
        from rules import physics_direct_fire as _physics
        physics_direct_fire = _physics
    if model == "physics":
        physical = None
        try:
            physical = physics_direct_fire.resolve_physics_direct_fire(
                attacker, target, ctx)
        except Exception:
            physical = None
        if physical is not None:
            return physical
        # A unit with ballistic data but no physical firing solution is out of
        # physical range; do not silently fall back to the CRT table.
        if physics_direct_fire.has_profile(attacker):
            return None
    picked = select_weapon(attacker, target, ctx)
    if picked is None:
        return None
    weapon, table = picked
    shadow_event = None
    if model == "shadow" and physics_direct_fire.has_profile(attacker):
        # Counterfactual on the pre-shot state, using a copy of the RNG state so
        # the shadow never advances the gameplay stream.
        shadow_rng = random.Random()
        shadow_rng.setstate(ctx.rng.getstate())
        try:
            shadow_event = physics_direct_fire.resolve_physics_direct_fire(
                attacker, target, ctx, dry_run=True, weapon=weapon,
                rng=shadow_rng)
        except Exception as exc:
            shadow_event = {"error": str(exc)}
    distance = ctx.rule_distance(attacker, target)
    base = table.level_at(distance)
    idx = base if base is not None else 0
    elev = ctx.elevation_modifier(attacker, target)
    level = idx + elev  # terrain cover is applied once, in the correction table
    event = resolve_crt_shot(attacker, target, ctx, level, weapon,
                             distance=distance)
    event["type"] = "direct_fire"
    event["ele_diff"] = abs(ctx.elevation_grade(attacker) -
                            ctx.elevation_grade(target))
    if model == "shadow":
        event["model"] = "shadow"
    if shadow_event is not None:
        event["shadow_physics"] = shadow_event
    return event


def resolve_crt_shot(attacker, target, ctx, level, weapon,
                     distance=None, consume=True) -> dict:
    """CRT + correction adjudication shared by direct and indirect fire."""
    distance = (ctx.rule_distance(attacker, target)
                if distance is None else int(distance))
    mod = 0
    if attacker.suppressed:
        mod -= 1
    if attacker.moving:
        mod -= 1
    if ctx.cover_at(target) == Cover.URBAN:
        mod -= 1
    elif ctx.cover_at(target) in (Cover.FOREST, Cover.BUILDING):
        mod -= 2
    if target.concealed:
        mod -= 2
    if target.moving:
        mod -= 2
    if target.stacked:
        mod += 2
    if target.marching:
        mod += 4

    random1 = ctx.rng.randint(1, 6) + ctx.rng.randint(1, 6)
    random2 = ctx.rng.randint(1, 6) + ctx.rng.randint(1, 6)

    if ctx.target_kind(target) == "infantry":
        result, raw = crt().personnel_result(level, random1)
        rect = crt().personnel_correction(random2, mod)
        damage = raw + rect if result == "damage" else 0
        if result == "damage" and damage < 0:
            damage = -1
        suppressed = result == "suppressed"
    else:
        raw = crt().vehicle_raw(ctx.stack_count(attacker), level, random1)
        rect = crt().vehicle_correction(random2, target.armor, mod)
        damage = raw + rect
        if damage < 0:
            damage = -1
        suppressed = False

    if consume:
        ctx.consume_ammo(attacker.uid, weapon.key, 1)
        ctx.mark_fired(attacker.uid, weapon.key,
                       getattr(weapon, "cooldown_s", 75.0))
    if damage > 0:
        ctx.apply_damage(target.uid, damage)
    if suppressed:
        ctx.apply_suppression(target.uid, 60.0)

    return {
        "type": "crt_shot",
        "actors": [attacker.uid, target.uid],
        "weapon": weapon.key,
        "distance": distance,
        "att_level": level,
        "random1": random1,
        "random2": random2,
        "ori_damage": raw,
        "random2_rect": rect,
        "rect_damage": rect,
        "damage": damage,
        "suppressed": suppressed,
        "target_blood": getattr(target, "blood", 0),
    }


def best_target(attacker: RuleEntityView, entities: List[RuleEntityView],
                ctx: RuleContext) -> Optional[RuleEntityView]:
    best = None
    best_key = None
    for target in entities:
        if not target.alive or target.affiliation == attacker.affiliation:
            continue
        if ctx.target_kind(target) not in ("infantry", "vehicle"):
            continue
        if select_weapon(attacker, target, ctx) is None:
            continue
        if not ctx.can_detect(attacker, target):
            continue
        key = (ctx.rule_distance(attacker, target), -target.value)
        if best_key is None or key < best_key:
            best_key = key
            best = target
    return best
