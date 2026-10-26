"""Project a role-card motion/protection block onto PlatformProfile (P2).

The geographic engine keeps ``PlatformProfile`` as its internal contract; this
adapter is the single place where a role card becomes that profile.  Callers
must keep the legacy ``profile_for(unit_type, domain)`` fallback for scenarios
that do not declare a subclass yet.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mobility.profile import PlatformProfile

DOMAIN_MAP = {
    "land": "land",
    "installation": "land",
    "sea": "surface",
    "surface": "surface",
    "subsurface": "subsurface",
    "air": "air",
}

TURN_RATE_DEFAULT = {
    "land": 40.0,
    "surface": 3.0,
    "subsurface": 2.0,
    "air": 5.0,
}

# P0.2 physical footprint defaults per domain (cards may override via motion).
FOOTPRINT_DEFAULT = {
    "land": (6.0, 2.8, 8.0, False),
    "surface": (120.0, 15.0, 150.0, False),
    "subsurface": (60.0, 8.0, 120.0, False),
    "air": (15.0, 5.0, 600.0, False),
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _typed_quantities(entity) -> Dict[str, float]:
    """Known values from a capability EntityType (empty when not given)."""
    if entity is None:
        return {}
    out: Dict[str, float] = {}
    for key, quantity in getattr(entity, "quantities", {}).items():
        if getattr(quantity, "known", False):
            out[key] = float(quantity.value)
    return out


def profile_from_card(card: dict, key: Optional[str] = None,
                      entity=None) -> PlatformProfile:
    """Build a ``PlatformProfile`` from an effective role card."""
    identity = card.get("identity") or {}
    motion = card.get("motion") or {}
    protection = card.get("protection") or {}
    domain_raw = str(identity.get("primary_domain") or "land")
    domain = DOMAIN_MAP.get(domain_raw, "land")
    terrain = motion.get("terrain") or {}
    sea = motion.get("sea") or {}
    air = motion.get("air") or {}
    size = str(protection.get("size_class") or "medium")
    length_d, width_d, turn_d, pivot_d = FOOTPRINT_DEFAULT.get(
        domain, FOOTPRINT_DEFAULT["land"])
    typed = _typed_quantities(entity)
    length_m = typed.get("dim.length_m",
                         _f(motion.get("length_m"), length_d))
    width_m = typed.get("dim.width_m",
                        _f(motion.get("width_m"), width_d))
    height_m = typed.get("dim.height_m", 0.0)
    weight_t = typed.get("mass.t", 0.0)
    if entity is not None and domain in ("land", "surface"):
        beam_m = typed.get("sea.beam_m", width_m)
    else:
        beam_m = _f(sea.get("beam_m"))
    return PlatformProfile(
        key=key or str(identity.get("subclass_id") or "card"),
        domain=domain,
        max_speed_ms=max(0.0, _f(motion.get("max_speed_ms"))),
        accel_ms2=_f(motion.get("accel_ms2"), 1.0),
        decel_ms2=_f(motion.get("decel_ms2"), 2.0),
        turn_rate_dps=_f(motion.get("turn_rate_dps"), TURN_RATE_DEFAULT.get(domain, 30.0)),
        terrain_factors={str(k): _f(v) for k, v in (terrain.get("factors") or {}).items()},
        max_gradient_deg=_f(terrain.get("max_gradient_deg"), 30.0),
        max_side_slope_deg=_f(terrain.get("max_side_slope_deg"), 17.0),
        wading_depth_m=_f(terrain.get("wading_depth_m"), 1.2),
        swim_depth_m=_f(terrain.get("swim_depth_m"), 0.0),
        amphibious=bool(terrain.get("amphibious")),
        road_bound=bool(terrain.get("road_bound")),
        draft_m=_f(sea.get("draft_m")),
        ukc_m=_f(sea.get("ukc_m"), 0.5),
        beam_m=beam_m,
        max_operating_depth_m=_f(sea.get("max_operating_depth_m")),
        min_clearance_m=_f(air.get("min_clearance_m")),
        service_ceiling_m=_f(air.get("service_ceiling_m")),
        agl_m=_f(air.get("agl_m")),
        climb_rate_ms=_f(air.get("climb_rate_ms"), 5.0),
        descent_rate_ms=_f(air.get("descent_rate_ms"), 5.0),
        armor_class=int(protection.get("armor_class") or 0),
        size_class=size if size in ("small", "medium", "large") else "medium",
        weight_t=weight_t,
        vehicle_height_m=height_m,
        length_m=length_m,
        width_m=width_m,
        min_turn_radius_m=_f(motion.get("min_turn_radius_m"), turn_d),
        pivot_turn=bool(motion.get("pivot_turn", pivot_d)),
    )
