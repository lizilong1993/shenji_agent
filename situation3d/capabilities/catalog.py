"""Build EntityType views from the existing role-card registry (P2 slice).

The catalog is a bypass layer: role cards remain the single data source, and
``catalog_mode`` selects between the legacy effective-card dict and the typed
EntityType view.  Nothing here mutates the registry.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Tuple

from engine.rolecards import load_registry

from .model import EntityType, Quantity, SourceRef, normalize_policy, unknown

_DIM = "m"


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(block: Mapping[str, Any], *paths: str) -> Optional[float]:
    for path in paths:
        cur: Any = block
        for part in path.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                cur = None
                break
            cur = cur[part]
        value = _num(cur)
        if value is not None:
            return value
    return None


def _source_of(sub: Mapping[str, Any]) -> SourceRef:
    raw = sub.get("source") or {}
    refs = tuple(str(r) for r in (raw.get("refs") or []) if r)
    return SourceRef(refs=refs, note=str(raw.get("note") or ""),
                     confidence=str(sub.get("confidence") or "medium"))


def _q(card: Mapping[str, Any], paths: Tuple[str, ...], unit: str,
       source: SourceRef) -> Quantity:
    value = _first(card, *paths)
    return Quantity(value, unit, source) if value is not None else unknown(unit, source)


def build_entity_type(registry, type_id: str) -> EntityType:
    """Project one subclass card onto an EntityType without altering it."""
    try:
        sub = registry.subclasses[type_id]
    except KeyError:
        raise KeyError("unknown entity type: %s" % type_id) from None
    card = registry.effective(type_id)
    source = _source_of(sub)
    identity = card.get("identity") or {}
    motion = card.get("motion") or {}
    protection = card.get("protection") or {}

    quantities: Dict[str, Quantity] = {
        "dim.length_m": _q(card, ("identity.dimensions.length_m",
                                  "motion.length_m", "motion.sea.length_m"), _DIM, source),
        "dim.width_m": _q(card, ("identity.dimensions.width_m",
                                 "motion.width_m", "motion.sea.beam_m"), _DIM, source),
        "dim.height_m": _q(card, ("identity.dimensions.height_m",
                                  "motion.height_m"), _DIM, source),
        "speed.max_ms": _q(card, ("motion.max_speed_ms",), "m/s", source),
        "speed.cruise_ms": _q(card, ("motion.cruise_speed_ms",), "m/s", source),
        "motion.accel_ms2": _q(card, ("motion.accel_ms2",), "m/s2", source),
        "motion.decel_ms2": _q(card, ("motion.decel_ms2",), "m/s2", source),
        "terrain.gradient_max_deg": _q(card, ("motion.terrain.max_gradient_deg",),
                                       "deg", source),
        "terrain.side_slope_max_deg": _q(card, ("motion.terrain.max_side_slope_deg",),
                                         "deg", source),
        "terrain.wading_m": _q(card, ("motion.terrain.wading_depth_m",), _DIM, source),
        "terrain.swim_m": _q(card, ("motion.terrain.swim_depth_m",), _DIM, source),
        "sea.draft_m": _q(card, ("motion.sea.draft_m",), _DIM, source),
        "sea.ukc_m": _q(card, ("motion.sea.ukc_m",), _DIM, source),
        "sea.beam_m": _q(card, ("motion.sea.beam_m",), _DIM, source),
        "sea.sea_state_max": _q(card, ("motion.sea.sea_state_max",), "level", source),
        "sea.depth_max_m": _q(card, ("motion.sea.max_operating_depth_m",), _DIM, source),
        "air.ceiling_m": _q(card, ("motion.air.service_ceiling_m",), _DIM, source),
        "air.runway_m": _q(card, ("motion.air.runway_required_m",), _DIM, source),
        "air.climb_ms": _q(card, ("motion.air.climb_rate_ms",), "m/s", source),
        "air.descent_ms": _q(card, ("motion.air.descent_rate_ms",), "m/s", source),
        "air.agl_m": _q(card, ("motion.air.agl_m",), _DIM, source),
        "air.min_clearance_m": _q(card, ("motion.air.min_clearance_m",), _DIM, source),
        "protection.armor_class": _q(card, ("protection.armor_class",), "class", source),
        "protection.max_blood": _q(card, ("protection.max_blood",), "hp", source),
        "mass.t": _q(card, ("motion.weight_t", "logistics.weight_t",
                            "identity.weight_t", "identity.dimensions.weight_t"),
                     "t", source),
    }

    gate_cfg = card.get("gates") or {}
    policy = normalize_policy(gate_cfg.get("unknown_policy"))

    terrain = motion.get("terrain") or {}
    flags = {
        "amphibious": bool(terrain.get("amphibious")),
        "road_bound": bool(terrain.get("road_bound")),
    }
    factors = {str(k): float(v) for k, v in (terrain.get("factors") or {}).items()
               if _num(v) is not None}

    return EntityType(
        type_id=str(type_id),
        class_id=str(sub.get("class_id") or ""),
        label=str(sub.get("label") or ""),
        domain=str(identity.get("primary_domain") or "land"),
        modes=tuple(str(m) for m in (identity.get("mobility_modes") or [])),
        quantities=quantities,
        flags=flags,
        terrain_factors=factors,
        unknown_policy=policy,
        sdk=copy.deepcopy(dict(sub.get("sdk") or {})),
    )


class CapabilityCatalog:
    """All entity types projected from one role-card registry instance."""

    def __init__(self, registry=None):
        self.registry = registry or load_registry()
        self._types = {
            type_id: build_entity_type(self.registry, type_id)
            for type_id in sorted(self.registry.subclasses)
        }

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._types)

    def get(self, type_id: str) -> EntityType:
        try:
            return self._types[type_id]
        except KeyError:
            raise KeyError("unknown entity type: %s" % type_id) from None

    def resolve(self, key: str) -> Optional[EntityType]:
        type_id = self.registry.resolve_type(key)
        return self.get(type_id) if type_id else None

    def __len__(self) -> int:
        return len(self._types)


def resolve_type(key: str, mode: str = "legacy", registry=None):
    """Dual read: legacy returns the effective card dict, new an EntityType."""
    reg = registry or load_registry()
    type_id = reg.resolve_type(key)
    if type_id is None:
        return None
    if mode == "legacy":
        return reg.effective(type_id)
    if mode == "new":
        return CapabilityCatalog(reg).get(type_id)
    raise ValueError("unsupported catalog mode: %r" % mode)
