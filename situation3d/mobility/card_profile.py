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


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def profile_from_card(card: dict, key: Optional[str] = None) -> PlatformProfile:
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
        beam_m=_f(sea.get("beam_m")),
        max_operating_depth_m=_f(sea.get("max_operating_depth_m")),
        min_clearance_m=_f(air.get("min_clearance_m")),
        service_ceiling_m=_f(air.get("service_ceiling_m")),
        agl_m=_f(air.get("agl_m")),
        climb_rate_ms=_f(air.get("climb_rate_ms"), 5.0),
        descent_rate_ms=_f(air.get("descent_rate_ms"), 5.0),
        armor_class=int(protection.get("armor_class") or 0),
        size_class=size if size in ("small", "medium", "large") else "medium",
    )
