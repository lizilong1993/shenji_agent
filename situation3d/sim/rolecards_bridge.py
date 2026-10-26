"""Bridge between role cards and the geo engine (P2).

The public 庙算 fixtures carry only ``type/sub_type/name``; scenarios can opt in
to cards per operator (``subclass_id``) or for the whole run
(``engine.use_role_cards = true``).  When a card is used, profile, sensors and
weapons come from it; otherwise the legacy name-based path remains untouched.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from engine.rolecards import load_registry
from mobility.card_profile import profile_from_card
from rules.context import WeaponView
from rules.weapons import WeaponTables, load_wpdata
from sim.sensors import Sensor

_REGISTRY = None
_TABLES = None
_CATALOG = None

DOMAIN_KEYS = {
    "land": "land",
    "installation": "land",
    "sea": "surface",
    "surface": "surface",
    "subsurface": "subsurface",
    "air": "air",
    "amphibious": "land",
}

# Longest/most specific needles first.
NAME_RULES: List[Tuple[str, str]] = [
    ("岸舰", "land.installation.coastal_battery"),
    ("防空导弹排", "land.infantry.manpads"),
    ("防空高炮", "land.vehicle.aaa"),
    ("防空导弹", "land.vehicle.sam"),
    ("防空", "land.vehicle.sam"),
    ("轻型坦克", "land.vehicle.light_tank"),
    ("重型坦克", "land.vehicle.mbt"),
    ("中型坦克", "land.vehicle.mbt"),
    ("坦克", "land.vehicle.mbt"),
    ("两栖战车", "land.vehicle.amphibious"),
    ("两栖突击", "land.vehicle.amphibious"),
    ("步战", "land.vehicle.ifv"),
    ("战车", "land.vehicle.ifv"),
    ("火箭炮", "land.vehicle.mlrs"),
    ("炮兵", "land.vehicle.spg"),
    ("无人战车", "land.vehicle.ugv"),
    ("巡飞", "air.uav.loitering"),
    ("无人机", "air.uav.tactical"),
    ("直升机", "air.helicopter.attack"),
    ("运输机", "air.aircraft.transport"),
    ("侦察", "land.vehicle.recon"),
    ("校射雷达", "land.vehicle.radar"),
    ("雷达", "land.vehicle.radar"),
    ("扫雷", "land.vehicle.engineer"),
    ("布雷", "land.vehicle.engineer"),
    ("工程", "land.vehicle.engineer"),
    ("步兵", "land.infantry.rifle"),
    ("驱逐舰", "sea.surface.destroyer"),
    ("护卫舰", "sea.surface.frigate"),
    ("巡逻舰", "sea.surface.corvette"),
    ("补给舰", "sea.surface.replenishment"),
    ("登陆舰", "sea.surface.landing_ship"),
    ("两栖舰", "sea.surface.landing_ship"),
    ("潜艇", "sea.subsurface.attack"),
]


def registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def capability_catalog():
    global _CATALOG
    if _CATALOG is None:
        from capabilities import CapabilityCatalog
        _CATALOG = CapabilityCatalog(registry())
    return _CATALOG


def entity_type_for(subclass_id: str):
    return capability_catalog().get(subclass_id)


def weapon_tables() -> WeaponTables:
    global _TABLES
    if _TABLES is None:
        _TABLES = WeaponTables(load_wpdata())
    return _TABLES


def subclass_for_operator(op: dict, domain: Optional[str] = None,
                          infer: bool = False) -> Optional[str]:
    """Resolve an operator to a subclass id.

    Explicit ``subclass_id``/``role_card_type`` always wins.  Inference from
    name / 庙算 ``type.sub_type`` only runs when ``infer`` is true or a subclass
    was declared, so legacy fixtures keep their old behaviour by default.
    """
    explicit = (op.get("subclass_id") or op.get("role_card_type")
                or op.get("type_key"))
    if explicit:
        resolved = registry().resolve_type(str(explicit))
        if resolved:
            return resolved
    if not infer:
        return None

    name = str(op.get("name") or "")
    for needle, subclass_id in NAME_RULES:
        if needle in name:
            return subclass_id

    op_type = int(op.get("type") or 2)
    sub_type = op.get("sub_type")
    if sub_type is not None:
        resolved = registry().resolve_type("%d.%d" % (op_type, int(sub_type)))
        if resolved:
            return resolved

    if op_type == 1:
        return "land.infantry.unknown"
    if op_type == 3:
        return "air.aircraft.unknown"
    if op_type == 4:
        return "land.installation.unknown"
    return {
        "surface": "sea.surface.unknown",
        "subsurface": "sea.subsurface.unknown",
        "air": "air.aircraft.unknown",
    }.get(DOMAIN_KEYS.get(str(domain or ""), "land"), "land.vehicle.unknown")


def card_for_operator(op: dict, domain: Optional[str] = None,
                      infer: bool = False) -> Tuple[Optional[str], Optional[dict]]:
    subclass_id = subclass_for_operator(op, domain, infer=infer)
    if not subclass_id:
        return None, None
    return subclass_id, registry().effective(subclass_id)


def domain_from_card(card: dict) -> str:
    raw = str((card.get("identity") or {}).get("primary_domain") or "land")
    return DOMAIN_KEYS.get(raw, "land")


def unit_type_from_card(subclass_id: str) -> str:
    sid = subclass_id
    if sid.startswith("land.infantry"):
        return "infantry"
    if sid.startswith("land.installation"):
        return "fort"
    if sid.startswith("sea.surface"):
        return "ship"
    if sid.startswith("sea.subsurface"):
        return "sub"
    if sid.startswith("air.helicopter"):
        return "helicopter"
    if sid.startswith("air.uav"):
        return "uav"
    if sid.startswith("air."):
        return "aircraft"
    if any(k in sid for k in ("artillery", "spg", "mlrs", "towed")):
        return "artillery"
    if any(k in sid for k in (".sam", ".aaa")):
        return "aa"
    if ".recon" in sid:
        return "recon"
    if ".engineer" in sid:
        return "engineering"
    return "vehicle"


def unit_kind_from_card(subclass_id: str) -> str:
    if subclass_id.startswith("land.infantry"):
        return "infantry"
    if subclass_id.startswith("sea.surface"):
        return "ship"
    if subclass_id.startswith("sea.subsurface"):
        return "sub"
    if subclass_id.startswith("air.helicopter"):
        return "helicopter"
    if subclass_id.startswith("air.uav"):
        return "uav"
    if subclass_id.startswith("air."):
        return "helicopter"  # observation table has no fixed-wing row
    if ".aaa" in subclass_id or ".sam" in subclass_id:
        return "aa"
    if ".recon" in subclass_id:
        return "recon"
    return "vehicle"


def sensors_from_card(card: dict) -> List[Sensor]:
    out = []
    for item in card.get("sensors") or []:
        spec = registry().equipment.get("sensors", {}).get(item.get("ref"))
        if not spec:
            continue
        out.append(Sensor(kind=str(spec.get("kind") or "radar"),
                          range_km=float(spec.get("range_km") or 0.0),
                          quality=float(spec.get("quality") or 1.0),
                          passive=bool(spec.get("passive"))))
    return out


def weapons_from_card(card: dict) -> Tuple[List[WeaponView], Dict[str, int]]:
    tables = weapon_tables()
    views: List[WeaponView] = []
    ammo: Dict[str, int] = {}
    for item in card.get("weapons") or []:
        spec = registry().equipment.get("weapons", {}).get(item.get("ref")) or {}
        sdk = spec.get("sdk") or {}
        wid = sdk.get("wpdata_id")
        try:
            wid = int(wid) if wid is not None else 0
        except (TypeError, ValueError):
            wid = 0
        wtab = tables.tables_for_weapon(wid) if wid else {}
        key = str(wid) if wid else str(item.get("ref"))
        category = str(spec.get("category") or "direct")
        kind = "indirect" if category in ("howitzer", "mlrs") else "direct"
        shoot_on_move = bool(item.get("shoot_on_move",
                                      spec.get("shoot_on_move", False)))
        ammo_count = int(item.get("ammo_default", item.get("ammo_max", 0)) or 0)
        views.append(WeaponView(
            key=key, weapon_id=wid, name=str(spec.get("label") or key),
            kind=kind, shoot_on_move=shoot_on_move,
            cooldown_s=float(spec.get("cooldown_s") or 75.0),
            ammo=ammo_count, tables=wtab))
        ammo[key] = ammo_count
    return views, ammo


__all__ = [
    "capability_catalog",
    "card_for_operator",
    "domain_from_card",
    "entity_type_for",
    "profile_from_card",
    "sensors_from_card",
    "subclass_for_operator",
    "unit_kind_from_card",
    "unit_type_from_card",
    "weapons_from_card",
]
