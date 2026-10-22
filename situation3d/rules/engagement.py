"""Sea/air engagement matrix and Pk resolution (ported from the MVP).

The matrix and synthetic weapon parameters are unchanged from
``wargame_core.engagement`` (marked ``approximated``); only distance and
timers now come from ``RuleContext`` (metres/seconds instead of hexes/ticks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rules.context import RuleContext, RuleEntityView

MATRIX: Dict[Tuple[str, str], List[str]] = {
    ("land", "land"): ["direct"],
    ("land", "sea"): ["coastal_asm"],
    ("land", "air"): ["sam", "aaa"],
    ("sea", "land"): ["naval_gun", "land_attack"],
    ("sea", "sea"): ["asm", "naval_gun"],
    ("sea", "air"): ["sam", "ciws"],
    ("air", "land"): ["air_ground"],
    ("air", "sea"): ["asm", "air_ground"],
    ("air", "air"): ["bvr", "wvr"],
    ("sub", "sea"): ["torpedo", "asm"],
    ("sub", "sub"): ["torpedo"],
    ("sea", "sub"): ["torpedo", "depth_charge", "asroc"],
}


@dataclass(frozen=True)
class SeaAirWeapon:
    key: str
    name: str
    kind: str
    max_range_km: float
    base_pk: float
    damage: int
    ammo: int = 8
    min_range_km: float = 0.0
    anti: Tuple[str, ...] = ("sea",)
    cooldown_s: float = 180.0


SEA_AIR_PROFILES: Dict[str, SeaAirWeapon] = {
    "asm": SeaAirWeapon("asm", "反舰导弹", "asm", 180.0, 0.176, 1, 8, 15.0, ("sea",), 720.0),
    "asm_light": SeaAirWeapon("asm_light", "轻型反舰导弹", "asm", 90.0, 0.154, 1, 8, 10.0, ("sea",), 600.0),
    "sam_long": SeaAirWeapon("sam_long", "远程舰空导弹", "sam", 120.0, 0.193, 1, 12, 5.0, ("air",), 480.0),
    "sam_medium": SeaAirWeapon("sam_medium", "中程舰空导弹", "sam", 50.0, 0.165, 1, 16, 3.0, ("air",), 360.0),
    "ciws": SeaAirWeapon("ciws", "近防炮", "ciws", 8.0, 0.138, 1, 30, 0.0, ("air",), 120.0),
    "naval_gun": SeaAirWeapon("naval_gun", "舰炮", "naval_gun", 25.0, 0.121, 1, 40, 1.0, ("sea", "land"), 240.0),
    "land_attack": SeaAirWeapon("land_attack", "对地巡航导弹", "land_attack", 200.0, 0.220, 1, 6, 20.0, ("land",), 2400.0),
    "torpedo": SeaAirWeapon("torpedo", "重型鱼雷", "torpedo", 40.0, 0.220, 2, 6, 2.0, ("sea", "sub"), 1200.0),
    "depth_charge": SeaAirWeapon("depth_charge", "深水炸弹", "depth_charge", 10.0, 0.2, 1, 20, 0.0, ("sub",), 300.0),
    "asroc": SeaAirWeapon("asroc", "反潜火箭", "asroc", 30.0, 0.25, 1, 10, 2.0, ("sub",), 360.0),
    "bvr": SeaAirWeapon("bvr", "中距空空导弹", "bvr", 100.0, 0.165, 1, 8, 10.0, ("air",), 600.0),
    "wvr": SeaAirWeapon("wvr", "近距空空导弹", "wvr", 15.0, 0.138, 1, 6, 0.0, ("air",), 360.0),
    "air_ground": SeaAirWeapon("air_ground", "航空制导炸弹", "air_ground", 25.0, 0.165, 1, 8, 0.0, ("land",), 720.0),
    "coastal_asm": SeaAirWeapon("coastal_asm", "岸舰导弹", "coastal_asm", 160.0, 0.165, 1, 8, 20.0, ("sea",), 1800.0),
    "sam": SeaAirWeapon("sam", "地空导弹", "sam", 60.0, 0.176, 1, 16, 3.0, ("air",), 600.0),
    "aaa": SeaAirWeapon("aaa", "高炮", "aaa", 10.0, 0.110, 1, 30, 0.0, ("air",), 180.0),
}


def _norm(domain: str) -> str:
    return {"surface": "sea", "subsurface": "sub",
            "amphibious": "land"}.get(domain, domain)


def allowed_kinds(attacker_domain: str, target_domain: str) -> List[str]:
    return list(MATRIX.get((_norm(attacker_domain), _norm(target_domain)), []))


def weapons_for(name: str, domain: str) -> List[SeaAirWeapon]:
    out: List[SeaAirWeapon] = []

    def add(*keys):
        for k in keys:
            if k in SEA_AIR_PROFILES:
                out.append(SEA_AIR_PROFILES[k])

    if _norm(domain) == "sea":
        if "驱逐舰" in name:
            add("asm", "sam_long", "naval_gun", "asroc")
        elif "护卫舰" in name:
            add("asm_light", "sam_medium", "naval_gun", "asroc")
        elif "巡逻" in name or "导弹艇" in name:
            add("asm_light", "naval_gun")
        elif "补给" in name:
            pass
        elif "两栖" in name or "登陆" in name:
            add("ciws")
        else:
            add("naval_gun")
    elif _norm(domain) == "sub":
        add("torpedo", "asm_light")
    elif domain == "air":
        if "无人机" in name:
            add("air_ground")
        elif "直升机" in name:
            add("air_ground", "wvr")
        elif "运输" in name:
            pass
        else:
            add("bvr", "wvr", "air_ground", "asm_light")
    elif domain == "land":
        if "岸舰" in name:
            add("coastal_asm")
        elif "防空" in name:
            add("sam", "aaa")
    return out


def resolve_sea_air(attacker: RuleEntityView, target: RuleEntityView,
                    ctx: RuleContext, flight_model: str = "instant",
                    now_s: Optional[float] = None) -> Optional[dict]:
    kinds = allowed_kinds(attacker.domain, target.domain)
    if not kinds or attacker.affiliation == target.affiliation:
        return None
    distance_km = ctx.distance_m(attacker, target) / 1000.0
    best = None
    for w in getattr(attacker, "sea_air_weapons", []) or []:
        if w.kind not in kinds or attacker.ammo_count(w.key) <= 0:
            continue
        if distance_km < w.min_range_km or distance_km > w.max_range_km:
            continue
        if best is None or w.base_pk > best.base_pk:
            best = w
    if best is None or not ctx.weapon_ready(attacker.uid, best.key):
        return None
    if not ctx.can_detect(attacker, target):
        return None
    pk = best.base_pk * max(0.0, 1.0 - getattr(ctx, "jam", 0.0)) * attacker.readiness
    roll = ctx.rng.random()
    hit = roll <= pk
    damage = best.damage if hit else 0
    ctx.consume_ammo(attacker.uid, best.key, 1)
    ctx.mark_fired(attacker.uid, best.key, best.cooldown_s)
    if flight_model == "flight":
        if now_s is None:
            raise ValueError("flight_model='flight' requires now_s")
        from rules import missile
        tof = missile.tof_s(best.kind, distance_km)
        intercept_pk = missile.intercept_probability(
            target, best.kind, distance_km, ctx)
        intercepted = missile.terminal_roll(ctx, intercept_pk)
        final_hit = hit and not intercepted
        return {
            "type": "sea_air_fire",
            "model": "flight",
            "actors": [attacker.uid, target.uid],
            "domain": "%s->%s" % (_norm(attacker.domain), _norm(target.domain)),
            "weapon": best.key,
            "weapon_kind": best.kind,
            "range_km": round(distance_km, 1),
            "pk": round(pk, 3),
            "roll": round(roll, 3),
            "result": "launched",
            "planned_hit": bool(hit),
            "intercepted": bool(intercepted),
            "intercept_pk": round(intercept_pk, 3),
            "tof_s": round(tof, 1),
            "impact_at_s": (None if now_s is None else round(now_s + tof, 1)),
            "planned_damage": int(damage if final_hit else 0),
            "suppress_s": 60.0 if final_hit and best.kind in (
                "asm", "torpedo", "air_ground") else 0.0,
            "damage": 0,
            "target_blood": getattr(target, "blood", 0),
        }
    if hit:
        ctx.apply_damage(target.uid, damage)
        if best.kind in ("asm", "torpedo", "air_ground"):
            ctx.apply_suppression(target.uid, 60.0)
    return {
        "type": "sea_air_fire",
        "model": "instant",
        "actors": [attacker.uid, target.uid],
        "domain": "%s->%s" % (_norm(attacker.domain), _norm(target.domain)),
        "weapon": best.key,
        "weapon_kind": best.kind,
        "range_km": round(distance_km, 1),
        "pk": round(pk, 3),
        "roll": round(roll, 3),
        "result": "hit" if hit else "miss",
        "damage": damage,
        "target_blood": getattr(target, "blood", 0),
    }
