"""Typed gate evaluation for the P1 environment slice.

Gates are evaluated in a deterministic order: building collision, medium
(water/air/land), slope, then road weight/height/width limits. Unknown values
fail closed unless an approved policy explicitly chooses advisory/penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from mobility.passability import passable as _passable
from mobility.passability import road_edge_passable
from env.service import LandCover

DOMAIN_ALIASES = {
    "maritime_surface": "surface",
    "maritime_subsurface": "subsurface",
}
UNKNOWN_POLICIES = ("deny", "advisory", "allow_with_penalty")


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason_codes: Tuple[str, ...] = ()
    gate_ids: Tuple[str, ...] = ()
    unknown: bool = False
    quality: str = "derived"


def _policy_value(policy: Optional[Dict[str, str]], key: str,
                  default: str = "deny") -> str:
    if not policy:
        return default
    value = policy.get(key)
    if isinstance(value, dict):
        value = value.get("unknown_policy")
    return str(value or default)


def _unknown_gate(policy: Optional[Dict[str, str]], key: str,
                  gate_id: str, default: str = "deny") -> GateResult:
    decision = _policy_value(policy, key, default)
    if decision not in UNKNOWN_POLICIES:
        decision = "deny"
    if decision == "deny":
        return GateResult(False, (key + "_unknown",), (gate_id,), True, "unknown")
    if decision == "advisory":
        return GateResult(
            True, (key + "_unknown_advisory",), (gate_id,), True, "unknown"
        )
    return GateResult(
        True, (key + "_unknown_penalty",), (gate_id,), True, "unknown"
    )


def _number(value) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().lower().replace(",", ".")
    num = ""
    for ch in text:
        if ch.isdigit() or ch in (".", "-"):
            num += ch
        elif num:
            break
    try:
        return float(num)
    except ValueError:
        return 0.0


def _domain(profile) -> str:
    raw = str(getattr(profile, "domain", "land"))
    return DOMAIN_ALIASES.get(raw, raw)


def evaluate_gates(profile, lon: float, lat: float, env,
                   depth_kind: str = "unknown",
                   policy: Optional[Dict[str, str]] = None,
                   is_water: Optional[bool] = None,
                   on_bridge: bool = False,
                   altitude_m: Optional[float] = None) -> GateResult:
    """Evaluate hard gates; unknown values are conservative.

    ``on_bridge`` is only meaningful for land/amphibious platforms: it skips
    the water-depth gate (the deck is above the water) while bridge weight and
    clearance limits are still checked. Air platforms are delegated to the
    existing altitude-aware passability predicate.
    """
    domain = _domain(profile)
    if domain == "air":
        if not _passable(profile, lon, lat, env, altitude_m):
            return GateResult(False, ("airspace",), ("airspace",))
        return GateResult(True)

    if env.building_at(lon, lat) is not None:
        return GateResult(False, ("building",), ("building",))

    depth = None if depth_kind == "unknown" else env.water_depth(lon, lat)
    if is_water is None:
        is_water = bool(env.is_water(lon, lat))
    on_bridge = bool(on_bridge and domain in ("land", "amphibious"))

    if not on_bridge:
        if domain in ("surface", "subsurface"):
            if not is_water:
                return GateResult(False, ("not_water",), ("medium",))
            need = float(getattr(profile, "draft_m", 0.0)) + float(
                getattr(profile, "ukc_m", 0.0)
            )
            if depth is None:
                return _unknown_gate(policy, "water_depth", "water_depth")
            if depth < need:
                return GateResult(False, ("water_depth",), ("water_depth",))
        else:
            wading = float(getattr(profile, "wading_depth_m", 0.0))
            if is_water and depth is None:
                return _unknown_gate(policy, "water_depth", "water_depth")
            if is_water and depth is not None and depth > wading:
                return GateResult(False, ("water_depth",), ("water_depth",))

    if not on_bridge and domain not in ("surface", "subsurface"):
        if not is_water or depth is None:
            # Same landcover rule as the navigation mask, so the gate and the
            # route authority cannot disagree on forest/soft/water cells.
            cover = env.landcover(lon, lat)
            factor = float(profile.terrain_factor(cover))
            road_bound = bool(getattr(profile, "road_bound", False))
            on_road = bool(env.on_road(lon, lat))
            if factor <= 0.0 and not (road_bound and on_road):
                return GateResult(False, ("landcover",), ("landcover",))
            if cover == LandCover.URBAN and road_bound and not on_road:
                return GateResult(False, ("road_bound",), ("landcover",))

    slope = float(env.slope_deg(lon, lat))
    max_slope = float(getattr(profile, "max_gradient_deg", 0.0) or 0.0)
    if max_slope > 0.0 and slope > max_slope:
        return GateResult(False, ("slope",), ("slope",))

    getter = getattr(env, "road_edge_at", None)
    if getter is not None:
        edge = getter(lon, lat, 25.0)
        if edge:
            reasons = []
            max_weight = _number(edge.get("maxweight"))
            if max_weight > 0 and float(
                getattr(profile, "weight_t", 0.0)
            ) > max_weight:
                reasons.append("road_maxweight")
            max_height = _number(edge.get("maxheight"))
            if max_height > 0 and float(
                getattr(profile, "vehicle_height_m", 0.0)
            ) > max_height:
                reasons.append("road_maxheight")
            width = _number(edge.get("width"))
            if width > 0 and float(getattr(profile, "beam_m", 0.0)) > width:
                reasons.append("road_width")
            if reasons:
                return GateResult(False, tuple(reasons), ("road_limit",))

            if edge.get("bridge") or edge.get("tunnel"):
                missing = [
                    key for key in ("maxweight", "maxheight")
                    if _number(edge.get(key)) <= 0.0
                ]
                if missing:
                    key = "bridge_mlc" if edge.get("bridge") else "tunnel_clearance"
                    return _unknown_gate(policy, key, "road_limit", "advisory")
            if not road_edge_passable(profile, edge):
                return GateResult(False, ("road_limit",), ("road_limit",))

    return GateResult(True)
