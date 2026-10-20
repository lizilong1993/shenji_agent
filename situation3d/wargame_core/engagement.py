"""Unified engagement matrix for the joint wargame core.

The matrix says *who can engage whom with which weapon kind*.  It exists so
that implausible interactions (a tank firing at a warship, an artillery piece
engaging an aircraft) are rejected explicitly instead of being accidentally
allowed by a generic weapon list.

All sea/air weapon parameters are synthetic demonstration values and are
exported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import hexmap, weapons


# (attacker domain, target domain) -> allowed weapon kinds
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
}

# Submarine-specific pairs are handled by the naval module; it declares the
# submarine domain as "sub" and maps it here.
MATRIX.update({
    ("sub", "sea"): ["torpedo", "asm"],
    ("sub", "sub"): ["torpedo"],
    ("sea", "sub"): ["torpedo", "depth_charge", "asroc"],
})


@dataclass(frozen=True)
class SeaAirWeapon:
    key: str
    name: str
    kind: str               # asm | sam | torpedo | bvr | wvr | ciws | naval_gun | ...
    max_range_km: float
    base_pk: float          # per-shot probability of a damaging hit
    damage: int             # strength/blood points on a hit
    ammo: int = 8
    min_range_km: float = 0.0
    anti: Tuple[str, ...] = ("sea",)   # target domains this weapon can engage
    # Reload/cooldown in ticks (30 s each).  Without it every platform fires
    # every tick and the whole force is destroyed inside the first 15 minutes.
    cooldown_ticks: int = 6

    def to_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "kind": self.kind,
                "max_range_km": self.max_range_km, "base_pk": self.base_pk,
                "damage": self.damage, "ammo": self.ammo,
                "anti": list(self.anti), "approximated": True}


# Synthetic sea/air weapon profiles.  Numbers are chosen to make a 12-hour
# demonstration legible, not to model real systems.
SEA_AIR_PROFILES: Dict[str, SeaAirWeapon] = {
    "asm": SeaAirWeapon("asm", "反舰导弹", "asm", 180.0, 0.176, 1, 8, 15.0, ("sea",), 24),
    "asm_light": SeaAirWeapon("asm_light", "轻型反舰导弹", "asm", 90.0, 0.154, 1, 8, 10.0, ("sea",), 20),
    "sam_long": SeaAirWeapon("sam_long", "远程舰空导弹", "sam", 120.0, 0.193, 1, 12, 5.0, ("air",), 16),
    "sam_medium": SeaAirWeapon("sam_medium", "中程舰空导弹", "sam", 50.0, 0.165, 1, 16, 3.0, ("air",), 12),
    "ciws": SeaAirWeapon("ciws", "近防炮", "ciws", 8.0, 0.138, 1, 30, 0.0, ("air",), 4),
    "naval_gun": SeaAirWeapon("naval_gun", "舰炮", "naval_gun", 25.0, 0.121, 1, 40, 1.0, ("sea", "land"), 8),
    "land_attack": SeaAirWeapon("land_attack", "对地巡航导弹", "land_attack", 200.0, 0.220, 1, 6, 20.0, ("land",), 80),
    "torpedo": SeaAirWeapon("torpedo", "重型鱼雷", "torpedo", 40.0, 0.220, 2, 6, 2.0, ("sea", "sub"), 40),
    "depth_charge": SeaAirWeapon("depth_charge", "深水炸弹", "depth_charge", 10.0, 0.2, 1, 20, 0.0, ("sub",), 10),
    "asroc": SeaAirWeapon("asroc", "反潜火箭", "asroc", 30.0, 0.25, 1, 10, 2.0, ("sub",), 12),
    "bvr": SeaAirWeapon("bvr", "中距空空导弹", "bvr", 100.0, 0.165, 1, 8, 10.0, ("air",), 20),
    "wvr": SeaAirWeapon("wvr", "近距空空导弹", "wvr", 15.0, 0.138, 1, 6, 0.0, ("air",), 12),
    "air_ground": SeaAirWeapon("air_ground", "航空制导炸弹", "air_ground", 25.0, 0.165, 1, 8, 0.0, ("land",), 24),
    "coastal_asm": SeaAirWeapon("coastal_asm", "岸舰导弹", "coastal_asm", 160.0, 0.165, 1, 8, 20.0, ("sea",), 60),
    "sam": SeaAirWeapon("sam", "地空导弹", "sam", 60.0, 0.176, 1, 16, 3.0, ("air",), 20),
    "aaa": SeaAirWeapon("aaa", "高炮", "aaa", 10.0, 0.110, 1, 30, 0.0, ("air",), 6),
}


def sea_air_weapons_for(name: str, domain: str) -> List[SeaAirWeapon]:
    """Synthetic loadout by platform name / domain."""
    out: List[SeaAirWeapon] = []
    def add(*keys: str) -> None:
        for k in keys:
            if k in SEA_AIR_PROFILES:
                out.append(SEA_AIR_PROFILES[k])
    if domain == "sea":
        if "驱逐舰" in name:
            add("asm", "sam_long", "naval_gun", "asroc")
        elif "护卫舰" in name:
            add("asm_light", "sam_medium", "naval_gun", "asroc")
        elif "巡逻舰" in name or "导弹艇" in name:
            add("asm_light", "naval_gun")
        elif "补给舰" in name:
            pass
        elif "两栖" in name or "登陆舰" in name:
            add("ciws")
        else:
            add("naval_gun")
    elif domain == "sub":
        add("torpedo", "asm_light")
    elif domain == "air":
        if "预警" in name or "电子战" in name:
            pass
        elif "无人机" in name:
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
        else:
            add("direct")
    return out


def allowed_kinds(attacker_domain: str, target_domain: str) -> List[str]:
    td = "sub" if target_domain == "sub" else target_domain
    ad = "sub" if attacker_domain == "sub" else attacker_domain
    return list(MATRIX.get((ad, td), []))


def can_engage(attacker, target) -> bool:
    ad = getattr(attacker, "domain", "land")
    td = getattr(target, "domain", "land")
    # Land-vs-land is handled by the direct-fire tables in combat.py.
    if ad == "land" and td == "land":
        return False
    kinds = allowed_kinds(ad, td)
    if not kinds:
        return False
    return any(w.kind in kinds for w in getattr(attacker, "sea_air_weapons", []))


def resolve_sea_air(attacker, target, state) -> Optional[dict]:
    """One sea/air shot.  Returns an event dict or ``None`` if not allowed."""
    ad = getattr(attacker, "domain", "land")
    td = getattr(target, "domain", "land")
    kinds = allowed_kinds(ad, td)
    if not kinds:
        return None
    d_km = hexmap.distance(attacker.pos, target.pos) * state.map.scale_m / 1000.0
    best = None
    for w in getattr(attacker, "sea_air_weapons", []):
        if w.kind not in kinds:
            continue
        if attacker.ammo.get(w.key, 0) <= 0:
            continue
        if d_km > w.max_range_km or d_km < w.min_range_km:
            continue
        if best is None or w.base_pk > best.base_pk:
            best = w
    if best is None:
        return None
    if not _sea_air_detected(attacker, target, state, best):
        return None
    ready = state.weapon_ready_at(attacker.uid, best.key)
    if state.tick < ready:
        return None

    jam = 1.0 - float(getattr(state, "jam", 0.0) or 0.0)
    readiness = float(getattr(attacker, "readiness", 0.85))
    pk = best.base_pk * jam * readiness
    roll = state.rng.random()
    attacker.ammo[best.key] = max(0, attacker.ammo.get(best.key, 0) - 1)
    state.mark_weapon_fired(attacker.uid, best.key, best.cooldown_ticks)
    hit = roll <= pk
    damage = best.damage if hit else 0
    if hit:
        target.blood = max(0, target.blood - damage)
        if target.blood <= 0:
            target.alive = False
    if hit and target.alive and best.kind in ("asm", "torpedo", "air_ground"):
        target.suppressed = True
        target.suppressed_ticks = max(target.suppressed_ticks, 2)

    ev = {
        "id": "EV-%06d" % state.next_event_id(),
        "tick": state.tick,
        "type": "sea_air_fire",
        "source": "engine",
        "level": "warn" if hit else "info",
        "actors": [attacker.uid, target.uid],
        "location": list(hexmap.decode(target.pos)),
        "domain": "%s->%s" % (ad, td),
        "weapon": best.name,
        "range_km": round(d_km, 1),
        "pk": round(pk, 3),
        "roll": round(roll, 3),
        "result": "hit" if hit else "miss",
        "damage": damage,
        "target_blood": target.blood,
        "description": "%s 对 %s 发射%s：%s" % (
            attacker.name, target.name, best.name,
            "命中，毁伤 %d%s" % (damage, "，目标被摧毁" if not target.alive else "")
            if hit else "未命中"),
        "raw": {"approximated": True},
    }
    state.events.append(ev)
    return ev


def _sea_air_detected(attacker, target, state, weapon) -> bool:
    from . import sensors
    jam = float(getattr(state, "jam", 0.0) or 0.0)
    for s in getattr(attacker, "sensors", []):
        if s.kind in ("visual",):
            continue
        if sensors.detects(s, attacker, target, state.map, state.los, jam):
            return True
    # A weapon with its own seeker can engage a target already reported by the
    # side's shared picture.  For the MVP, accept if any friendly sensor holds
    # the target and the weapon has enough range.
    for u in state.units.values():
        if not getattr(u, "alive", True) or u.color != attacker.color:
            continue
        for s in getattr(u, "sensors", []):
            if sensors.detects(s, u, target, state.map, state.los, jam):
                return True
    return False


def best_sea_air_target(attacker, state) -> Optional[object]:
    best = None
    best_score = None
    for u in state.units.values():
        if not getattr(u, "alive", True) or u.color == attacker.color:
            continue
        if not state.map.valid(u.pos) or not state.map.valid(attacker.pos):
            continue
        if not can_engage(attacker, u):
            continue
        d_km = hexmap.distance(attacker.pos, u.pos) * state.map.scale_m / 1000.0
        in_range = False
        for w in getattr(attacker, "sea_air_weapons", []):
            if (w.kind in allowed_kinds(attacker.domain, u.domain)
                    and w.min_range_km <= d_km <= w.max_range_km
                    and attacker.ammo.get(w.key, 0) > 0
                    and state.tick >= state.weapon_ready_at(attacker.uid, w.key)):
                in_range = True
                break
        if not in_range:
            continue
        if not _sea_air_detected(attacker, u, state, None):
            continue
        score = (d_km, -int(getattr(u, "value", 0) or 0))
        if best_score is None or score < best_score:
            best_score = score
            best = u
    return best
