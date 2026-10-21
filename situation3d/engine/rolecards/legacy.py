"""Adapters from the pre-role-card ``Target`` model to role-card snapshots.

P0 contract: this module does not change any behaviour path.  It only derives a
role-card snapshot for an existing target so the dataset can carry both the old
flat fields and the new card, and so a later migration can compare the two.

``speed_kn`` is treated as the *cruise* speed and converted with the same
constant as ``engine.geo.knots_to_mps`` (1852/3600).  No rounding is applied to
the speed override: rounding 8 kn to 4 decimals already drifts ~2 m over a
12-hour track and would break frame parity when P1 starts reading the card.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .registry import RoleCardRegistry, load_registry

KNOT_TO_MS = 1852.0 / 3600.0
KNOT_TO_KMH = 1.852
STATIC_SPEED_KN = 0.1  # legacy positional batteries sit at ~0.005 kn


def _get(target: Any, key: str, default: Any = None) -> Any:
    if isinstance(target, dict):
        return target.get(key, default)
    return getattr(target, key, default)


def _text(target: Any) -> str:
    parts = [
        str(_get(target, "name", "") or ""),
        str(_get(target, "platform_class", "") or ""),
        str(_get(target, "model", "") or ""),
        str(_get(target, "role", "") or ""),
    ]
    return " ".join(parts).lower()


def legacy_view(target: Any) -> Dict[str, Any]:
    """Serialise the fields the old pipeline actually used."""
    speed_kn = _get(target, "speed_kn", 0.0) or 0.0
    domain = _get(target, "domain", "")
    target_id = _get(target, "target_id")
    if target_id is None:
        target_id = _get(target, "id")
    return {
        "id": target_id,
        "name": _get(target, "name", ""),
        "platform_class": _get(target, "platform_class", ""),
        "domain": domain,
        "affiliation": _get(target, "affiliation", ""),
        "role": _get(target, "role", ""),
        "model": _get(target, "model", ""),
        "model_note": _get(target, "model_note", ""),
        "icon": _get(target, "icon", "unknown"),
        "speed_kn": speed_kn,
        "speed_kmh": round(speed_kn * KNOT_TO_KMH, 1),
        "speed_unit": "km/h" if domain == "land" else "kn",
        "altitude_m": _get(target, "altitude_m", 0.0),
        "sensor_name": _get(target, "sensor_name", ""),
        "sensor_range_km": _get(target, "sensor_range_km", 0.0),
        "influence_km": _get(target, "influence_km", 0.0),
        "engage_km": _get(target, "engage_km", 0.0),
        "loop": bool(_get(target, "loop", False)),
        "road_bound": bool(_get(target, "road_bound", True)),
        "waypoints": [list(p) for p in (_get(target, "waypoints", []) or [])],
        "attributes": dict(_get(target, "attributes", {}) or {}),
    }


def _subclass_for(target: Any, registry: RoleCardRegistry) -> str:
    explicit = _get(target, "subclass_id") or _get(target, "type_key")
    if explicit:
        resolved = registry.resolve_type(str(explicit))
        if resolved:
            return resolved

    icon = str(_get(target, "icon", "unknown"))
    domain = str(_get(target, "domain", "land"))
    text = _text(target)
    speed_kn = float(_get(target, "speed_kn", 0.0) or 0.0)

    # Refine the coarse icon buckets with the model/platform hints that already
    # exist in the legacy data.  This is still an import-time mapping, not a
    # runtime name heuristic in the simulation path.
    if icon == "armor":
        if any(k in text for k in ("两栖", "ztd", "zbd-05", "05式", "05 式")):
            return "land.vehicle.amphibious"
        if any(k in text for k in ("轻型坦克", "15式", "15 式", "ztq")):
            return "land.vehicle.light_tank"
        if any(k in text for k in ("步战", "ifv", "04a", "zbd-04")):
            return "land.vehicle.ifv"
        if any(k in text for k in ("自行榴弹", "自行加榴", "plz", "spg")):
            return "land.vehicle.spg"
        if any(k in text for k in ("工程", "engineer")):
            return "land.vehicle.engineer"
        if any(k in text for k in ("运输", "卡车", "eq240", "truck")):
            return "land.vehicle.truck"
        return "land.vehicle.mbt"
    if icon == "light_vehicle":
        if any(k in text for k in ("两栖", "ztd", "05式", "05 式")):
            return "land.vehicle.amphibious"
        if any(k in text for k in ("04a", "步战", "ifv")):
            return "land.vehicle.ifv"
        return "land.vehicle.ifv"
    if icon == "artillery":
        if any(k in text for k in ("火箭炮", "phl", "mlrs")):
            return "land.vehicle.mlrs"
        if any(k in text for k in ("牵引", "m777", "pll", "towed")):
            return "land.vehicle.towed_artillery"
        return "land.vehicle.spg"
    if icon == "replenishment":
        if any(k in text for k in ("登陆", "两栖", "071", "船坞")):
            return "sea.surface.landing_ship"
        return "sea.surface.replenishment"
    if icon == "transport":
        if domain == "sea":
            return "sea.surface.transport_ship"
        return "land.vehicle.truck"
    if icon == "sam_battalion":
        if speed_kn > STATIC_SPEED_KN:
            return "land.vehicle.sam"
        return "land.installation.sam_site"
    if icon == "coastal_missile":
        if speed_kn > STATIC_SPEED_KN:
            return "land.vehicle.coastal_launcher"
        return "land.installation.coastal_battery"
    if icon == "fighter":
        if any(k in text for k in ("隐形", "隐身", "j-20", "歼-20", "歼20")):
            return "air.aircraft.stealth_fighter"
        if any(k in text for k in ("歼轰", "攻击机", "strike")):
            return "air.aircraft.strike"
        return "air.aircraft.fighter"
    if icon == "helicopter":
        return "air.helicopter.attack"
    if icon == "uav":
        if any(k in text for k in ("巡飞", "loiter")):
            return "air.uav.loitering"
        return "air.uav.male"
    if icon == "outpost":
        if speed_kn > STATIC_SPEED_KN:
            # A manned outpost detachment that actually marches is infantry,
            # not an immobile fortification.
            return "land.infantry.rifle"
        return "land.installation.outpost"

    return registry.subclass_for_legacy(icon, domain)


def _max_sensor_km(card: dict, registry: RoleCardRegistry) -> float:
    best = 0.0
    for item in card.get("sensors") or []:
        spec = registry.equipment.get("sensors", {}).get(item.get("ref"))
        if spec:
            best = max(best, float(spec.get("range_km") or 0.0))
    return best


def _max_weapon_km(card: dict, registry: RoleCardRegistry) -> float:
    best = 0.0
    for item in card.get("weapons") or []:
        spec = registry.equipment.get("weapons", {}).get(item.get("ref"))
        if spec:
            best = max(best, float(spec.get("range_km") or 0.0))
    return best


def _legacy_or(value: Any, fallback: float) -> float:
    """Use the legacy number when it carries information (non-null, >0)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def project_legacy(card: dict, registry: RoleCardRegistry,
                   legacy: Optional[dict] = None) -> Dict[str, Any]:
    """Project an effective card back onto the old flat-field semantics.

    When a legacy field is present and non-zero it wins, because this function
    exists to reproduce the pre-card dataset while P1 migrates module by module.
    Capability-derived values are only fallbacks for targets without legacy
    numbers.
    """
    legacy = legacy or {}
    motion = card.get("motion") or {}
    cruise_ms = motion.get("cruise_speed_ms") or 0.0
    domain = (card.get("identity") or {}).get("primary_domain") or legacy.get("domain", "")
    sensor_km = _legacy_or(legacy.get("sensor_range_km"), _max_sensor_km(card, registry))
    weapon_km = _legacy_or(legacy.get("engage_km"), _max_weapon_km(card, registry))
    influence_km = _legacy_or(legacy.get("influence_km"), max(sensor_km, weapon_km))
    return {
        "speed_kn": cruise_ms / KNOT_TO_MS if cruise_ms else 0.0,
        "speed_kmh": round(cruise_ms * 3.6, 1),
        "speed_unit": "km/h" if domain == "land" else "kn",
        "sensor_range_km": round(sensor_km, 1),
        "engage_km": round(weapon_km, 1),
        "influence_km": round(influence_km, 1),
        "road_bound": bool(legacy.get("road_bound", (motion.get("terrain") or {}).get("road_bound", False))),
        "icon": legacy.get("icon"),
    }


def _display_speed(speed_ms: float, domain: str) -> str:
    if not speed_ms:
        return "0"
    if domain == "land":
        return "%.0f km/h" % (speed_ms * 3.6)
    return "%.0f kn" % (speed_ms / KNOT_TO_MS)


def _card_summary(cls: dict, sub: dict, card: dict,
                  registry: RoleCardRegistry) -> Dict[str, Any]:
    """Compact, browser-friendly projection for the target detail panel.

    The dataset already carries the full card; this block avoids shipping the
    equipment catalogue to the browser just to render labels and ranges.
    """
    motion = card.get("motion") or {}
    protection = card.get("protection") or {}
    identity = card.get("identity") or {}
    domain = identity.get("primary_domain") or cls.get("domain") or ""
    sensors = []
    for item in card.get("sensors") or []:
        spec = registry.equipment.get("sensors", {}).get(item.get("ref")) or {}
        if spec:
            sensors.append({"label": spec.get("label"), "kind": spec.get("kind"),
                            "range_km": spec.get("range_km")})
    weapons = []
    for item in card.get("weapons") or []:
        spec = registry.equipment.get("weapons", {}).get(item.get("ref")) or {}
        if spec:
            weapons.append({"label": spec.get("label"), "category": spec.get("category"),
                            "range_km": spec.get("range_km"),
                            "ammo_max": item.get("ammo_max")})
    return {
        "class_label": cls.get("label"),
        "subclass_label": sub.get("label"),
        "confidence": "legacy",
        "unknown": sub.get("confidence") == "unknown",
        "max_speed_ms": motion.get("max_speed_ms"),
        "cruise_speed_ms": motion.get("cruise_speed_ms"),
        "max_speed_display": _display_speed(float(motion.get("max_speed_ms") or 0.0), domain),
        "cruise_speed_display": _display_speed(float(motion.get("cruise_speed_ms") or 0.0), domain),
        "armor_class": protection.get("armor_class"),
        "max_blood": protection.get("max_blood"),
        "size_class": protection.get("size_class"),
        "sensors": sensors,
        "weapons": weapons,
        "capacity": dict(card.get("capacity") or {}),
    }


def card_from_legacy(target: Any, registry: Optional[RoleCardRegistry] = None,
                     include_effective: bool = True) -> Dict[str, Any]:
    """Build a frozen role-card snapshot for an existing target."""
    registry = registry or load_registry()
    subclass_id = _subclass_for(target, registry)
    cls = registry.class_of(subclass_id)
    base = registry.effective(subclass_id)

    overrides: Dict[str, Any] = {}
    legacy = legacy_view(target)
    speed_kn = float(legacy.get("speed_kn") or 0.0)
    base_motion = base.get("motion") or {}
    if not base_motion.get("immobile"):
        cruise_ms = speed_kn * KNOT_TO_MS
        base_max = float(base_motion.get("max_speed_ms") or 0.0)
        overrides["motion"] = {
            "cruise_speed_ms": cruise_ms,
            "max_speed_ms": max(base_max, cruise_ms),
        }
    if legacy.get("domain") == "land" and not base_motion.get("immobile"):
        overrides.setdefault("motion", {})["terrain"] = {
            "road_bound": bool(legacy.get("road_bound", True)),
        }

    model = legacy.get("model")
    if model:
        overrides.setdefault("identity", {})["model"] = {"reference": model}
    overrides.setdefault("identity", {})["symbol"] = {"legacy_icon": legacy.get("icon")}

    effective = registry.effective(subclass_id, overrides)
    # Preserve the base asset/dimensions; mark the asset approximate whenever
    # the legacy reference model differs from the class default.
    base_identity = base.get("identity") or {}
    eff_identity = effective.setdefault("identity", {})
    base_model = dict(base_identity.get("model") or {})
    eff_model = dict(eff_identity.get("model") or {})
    for key in ("asset", "asset_approximation"):
        if key in base_model and key not in eff_model:
            eff_model[key] = base_model[key]
    if model and eff_model.get("asset"):
        eff_model["asset_approximation"] = (model != base_model.get("reference"))
    eff_identity["model"] = eff_model
    if "dimensions" not in eff_identity and base_identity.get("dimensions"):
        eff_identity["dimensions"] = base_identity["dimensions"]

    snapshot = {
        "class_id": cls["class_id"],
        "subclass_id": subclass_id,
        "variant_id": None,
        "confidence": "legacy",
        "overrides": overrides,
        "legacy": legacy,
        "projection": project_legacy(effective, registry, legacy),
        "summary": _card_summary(cls, registry.subclasses[subclass_id], effective, registry),
    }
    if include_effective:
        snapshot["effective"] = effective
    return snapshot
