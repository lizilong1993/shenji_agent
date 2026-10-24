"""Minimal typed gate evaluation for the P1/P2 environment slice.

The full P0 contract has typed gates with unknown policies.  This module covers
the gates the existing legacy layers can actually evaluate: building collision,
water depth, slope and road weight/height/width limits.  It returns structured
reasons instead of a bare boolean so callers can audit why a move was denied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from mobility.passability import road_edge_passable


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


def evaluate_gates(profile, lon: float, lat: float, env,
                   depth_kind: str = "unknown",
                   policy: Optional[Dict[str, str]] = None,
                   is_water: Optional[bool] = None,
                   on_bridge: bool = False) -> GateResult:
    """Evaluate hard gates in a deterministic order.

    Order: building collision -> water/depth -> slope -> road limits.
    ``depth_kind='unknown'`` treats the legacy default water depth as unknown
    rather than pretending it is surveyed. ``policy`` maps gate names to
    ``unknown_policy`` values (``deny``/``advisory``/``allow_with_penalty``).
    ``is_water`` overrides the legacy fallback water predicate when supplied.
    ``on_bridge`` means the point is on a bridge deck, so water below is not a
    blocking condition; bridge limits are still checked in the road section.
    """
    if env.building_at(lon, lat) is not None:
        return GateResult(False, ("building",), ("building",))

    depth = None if depth_kind == "unknown" else env.water_depth(lon, lat)
    if is_water is None:
        is_water = bool(env.is_water(lon, lat))
    domain = str(getattr(profile, "domain", "land"))
    if not on_bridge:
        if domain in ("surface", "subsurface"):
            need = float(getattr(profile, "draft_m", 0.0)) + float(
                getattr(profile, "ukc_m", 0.0)
            )
            if depth is None and is_water:
                return _unknown_gate(policy, "water_depth", "water_depth")
            if depth is not None and depth < need:
                return GateResult(False, ("water_depth",), ("water_depth",))
        else:
            wading = float(getattr(profile, "wading_depth_m", 0.0))
            if depth is None and is_water:
                return _unknown_gate(policy, "water_depth", "water_depth")
            if depth is not None and depth > wading:
                return GateResult(False, ("water_depth",), ("water_depth",))

    slope = float(env.slope_deg(lon, lat))
    max_slope = float(getattr(profile, "max_gradient_deg", 0.0) or 0.0)
    if max_slope > 0.0 and slope > max_slope:
        return GateResult(False, ("slope",), ("slope",))

    getter = getattr(env, "road_edge_at", None)
    if getter is not None:
        edge = getter(lon, lat, 25.0)
        if edge and (edge.get("bridge") or edge.get("tunnel")):
            missing = [
                key for key in ("maxweight", "maxheight")
                if _number(edge.get(key)) <= 0.0
            ]
            if missing:
                key = "bridge_mlc" if edge.get("bridge") else "tunnel_clearance"
                return _unknown_gate(policy, key, "road_limit", "advisory")
        if edge and not road_edge_passable(profile, edge):
            reasons = []
            max_weight = _number(edge.get("maxweight"))
            if max_weight > 0 and float(getattr(profile, "weight_t", 0.0)) > max_weight:
                reasons.append("road_maxweight")
            max_height = _number(edge.get("maxheight"))
            if max_height > 0 and float(
                getattr(profile, "vehicle_height_m", 0.0)
            ) > max_height:
                reasons.append("road_maxheight")
            width = _number(edge.get("width"))
            if width > 0 and float(getattr(profile, "beam_m", 0.0)) > width:
                reasons.append("road_width")
            return GateResult(
                False, tuple(reasons or ("road_limit",)), ("road_limit",)
            )

    return GateResult(True)
