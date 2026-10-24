"""Physical capability gates (P2 slice).

Each gate compares one environment measurement against one or more entity
quantities.  A gate never guesses: when the card lacks the quantity it returns
``unknown`` and defers to the entity's ``unknown_policy`` (deny by default).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .model import EntityType, GateResult, normalize_policy

GATES = ("slope", "wading", "bridge", "sea_state", "ukc", "runway")


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unknown(gate: str, reason: str, policy: str) -> GateResult:
    return GateResult(gate, "unknown", reason, unknown_policy=policy)


def _compare(gate: str, actual: Optional[float], quantity, policy: str,
             label: str) -> Optional[GateResult]:
    """Deny when ``actual`` exceeds ``quantity``; pass/None when it does not."""
    if actual is None:
        return _unknown(gate, "%s not supplied" % label, policy)
    if quantity is None or not quantity.known:
        return _unknown(gate, "entity %s unknown" % label, policy)
    if actual > quantity.value:
        return GateResult(gate, "deny",
                          "%s %.2f exceeds %.2f" % (label, actual, quantity.value),
                          limit=quantity.value, actual=actual,
                          unknown_policy=policy)
    return None


def _require(gate: str, requirement, available: Optional[float], policy: str,
             label: str) -> Optional[GateResult]:
    """Deny when the entity requirement exceeds what the environment offers."""
    if available is None:
        return _unknown(gate, "%s not supplied" % label, policy)
    if requirement is None or not requirement.known:
        return _unknown(gate, "entity %s unknown" % label, policy)
    if requirement.value > available:
        return GateResult(gate, "deny",
                          "entity %s %.2f exceeds available %.2f"
                          % (label, requirement.value, available),
                          limit=available, actual=requirement.value,
                          unknown_policy=policy)
    return None


def _pass(gate: str, actual: float, limit: float, policy: str) -> GateResult:
    return GateResult(gate, "pass", "within limit", limit=limit, actual=actual,
                      unknown_policy=policy)


def _aggregate(results):
    """Pick the worst check result: deny > unknown > pass."""
    for result in results:
        if result.decision == "deny":
            return result
    for result in results:
        if result.decision == "unknown":
            return result
    return results[-1]


def _slope(entity: EntityType, env: Mapping[str, Any], policy: str) -> GateResult:
    checks = (("gradient_deg", "terrain.gradient_max_deg", "gradient"),
              ("side_slope_deg", "terrain.side_slope_max_deg", "side slope"))
    results = []
    for env_key, q_key, label in checks:
        actual = _num(env.get(env_key))
        if actual is None:
            continue
        quantity = entity.quantities.get(q_key)
        result = _compare("slope", actual, quantity, policy, label)
        if result is None:
            result = _pass("slope", actual, quantity.value, policy)
        results.append(result)
    if not results:
        return _unknown("slope", "no slope inputs supplied", policy)
    return _aggregate(results)


def _wading(entity: EntityType, env: Mapping[str, Any], policy: str) -> GateResult:
    actual = _num(env.get("water_depth_m"))
    wading = entity.quantities.get("terrain.wading_m")
    swim = entity.quantities.get("terrain.swim_m")
    limit = swim if (swim is not None and swim.known) else wading
    result = _compare("wading", actual, limit, policy, "water depth")
    return result or _pass("wading", actual, limit.value, policy)


def _bridge(entity: EntityType, env: Mapping[str, Any], policy: str) -> GateResult:
    checks = (
        ("bridge_max_weight_t", "mass.t", "weight"),
        ("bridge_max_height_m", "dim.height_m", "height"),
        ("bridge_max_width_m", "dim.width_m", "width"),
    )
    checked = 0
    results = []
    for env_key, q_key, label in checks:
        available = _num(env.get(env_key))
        if available is None:
            continue
        checked += 1
        requirement = entity.quantities.get(q_key)
        result = _require("bridge", requirement, available, policy, label)
        if result is None:
            result = _pass("bridge", requirement.value, available, policy)
        results.append(result)
    if checked == 0:
        return _unknown("bridge", "no bridge limits supplied", policy)
    return _aggregate(results)


def _sea_state(entity: EntityType, env: Mapping[str, Any], policy: str) -> GateResult:
    actual = _num(env.get("sea_state"))
    limit = entity.quantities.get("sea.sea_state_max")
    result = _compare("sea_state", actual, limit, policy, "sea state")
    return result or _pass("sea_state", actual, limit.value, policy)


def _ukc(entity: EntityType, env: Mapping[str, Any], policy: str) -> GateResult:
    depth = _num(env.get("water_depth_m"))
    draft = entity.quantities.get("sea.draft_m")
    ukc = entity.quantities.get("sea.ukc_m")
    depth_max = entity.quantities.get("sea.depth_max_m")
    if depth is None:
        return _unknown("ukc", "water depth not supplied", policy)
    if draft is None or not draft.known or ukc is None or not ukc.known:
        return _unknown("ukc", "entity draft/UKC unknown", policy)
    required = draft.value + ukc.value
    if depth < required:
        return GateResult("ukc", "deny",
                          "depth %.2f below draft+UKC %.2f" % (depth, required),
                          limit=required, actual=depth, unknown_policy=policy)
    if depth_max is not None and depth_max.known and depth > depth_max.value:
        return GateResult("ukc", "deny",
                          "depth %.2f exceeds operating limit %.2f"
                          % (depth, depth_max.value),
                          limit=depth_max.value, actual=depth,
                          unknown_policy=policy)
    return _pass("ukc", depth, required, policy)


def _runway(entity: EntityType, env: Mapping[str, Any], policy: str) -> GateResult:
    available = _num(env.get("runway_length_m"))
    required = entity.quantities.get("air.runway_m")
    if available is None:
        return _unknown("runway", "runway length not supplied", policy)
    result = _require("runway", required, available, policy, "runway")
    if result is not None:
        return result
    return _pass("runway", required.value, available, policy)


_FUNCS = {
    "slope": _slope,
    "wading": _wading,
    "bridge": _bridge,
    "sea_state": _sea_state,
    "ukc": _ukc,
    "runway": _runway,
}


def evaluate(entity: EntityType, gate: str, env: Mapping[str, Any],
             unknown_policy: Optional[str] = None) -> GateResult:
    """Evaluate one gate against one environment sample."""
    try:
        func = _FUNCS[gate]
    except KeyError:
        raise ValueError("unsupported gate: %r" % gate) from None
    policy = normalize_policy(unknown_policy if unknown_policy is not None
                              else entity.unknown_policy)
    return func(entity, env or {}, policy)


def evaluate_all(entity: EntityType, env: Mapping[str, Any],
                 gates=GATES, unknown_policy: Optional[str] = None
                 ) -> Dict[str, GateResult]:
    """Evaluate several gates; callers decide how to combine the results."""
    return {gate: evaluate(entity, gate, env, unknown_policy) for gate in gates}
