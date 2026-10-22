"""Platform data v1 (P0.3): JSON records -> PlatformProfile.

Every ``platforms/*.json`` file describes one platform class (dimensions,
mass/power, speed curve, turning, terrain/sea/air limits, armour faces,
signature, sensors, weapons).  The loader validates the schema with stdlib
code and rebuilds the ``PlatformProfile`` used by the mobility/motion layer.

The built-in constants in ``mobility.profile.PROFILES`` remain the fallback
when a key is missing or the data directory is absent.  Values in the v1
files mirror those constants exactly, so loading them reproduces the
previous behaviour; extra physical metadata (mass, armour faces, signature,
sensors, weapons) is carried for later phases.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional

from mobility.profile import PlatformProfile

SCHEMA = "msplatform.v1"
DOMAINS = ("land", "amphibious", "surface", "subsurface", "air")
SIZE_CLASSES = ("small", "medium", "large")
PLATFORM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "platforms")


class PlatformDataError(ValueError):
    """Raised when a platform record fails schema validation."""


def _is_num(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _block(errors: List[str], doc: Dict[str, Any], name: str,
           required: bool = False) -> Optional[Dict[str, Any]]:
    node = doc.get(name)
    if node is None:
        if required:
            errors.append("missing required field %s" % name)
        return None
    if not isinstance(node, dict):
        errors.append("%s must be an object" % name)
        return None
    return node


def _check_num(errors: List[str], doc: Dict[str, Any], path: str,
               minimum: Optional[float] = None,
               maximum: Optional[float] = None) -> None:
    parts = path.split(".")
    node: Any = doc
    for part in parts[:-1]:
        if not isinstance(node, dict):
            return  # a non-object parent is reported by _block
        if part not in node:
            return
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return
    value = node[parts[-1]]
    if not _is_num(value):
        errors.append("%s must be a finite number" % path)
        return
    if minimum is not None and value < minimum:
        errors.append("%s must be >= %s" % (path, minimum))
    if maximum is not None and value > maximum:
        errors.append("%s must be <= %s" % (path, maximum))


def _require(errors: List[str], doc: Dict[str, Any], path: str) -> None:
    node: Any = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            errors.append("missing required field %s" % path)
            return
        node = node[part]


def validate_platform(doc: Any) -> None:
    """Validate one platform record; raises ``PlatformDataError``."""
    if not isinstance(doc, dict):
        raise PlatformDataError("platform record must be a JSON object")
    errors: List[str] = []
    if doc.get("schema") != SCHEMA:
        errors.append("schema must be %r" % SCHEMA)
    if not isinstance(doc.get("key"), str) or not doc.get("key"):
        errors.append("key must be a non-empty string")
    identity = _block(errors, doc, "identity", required=True) or {}
    motion = _block(errors, doc, "motion", required=True) or {}
    terrain = _block(errors, doc, "terrain", required=True) or {}
    dimensions = _block(errors, doc, "dimensions", required=True) or {}
    _block(errors, doc, "mass")
    _block(errors, doc, "power")
    _block(errors, doc, "sea")
    _block(errors, doc, "air")
    _block(errors, doc, "road")
    armor = _block(errors, doc, "armor")
    signature = _block(errors, doc, "signature")
    provenance = _block(errors, doc, "provenance", required=True) or {}
    if provenance:
        for field in ("source", "confidence"):
            value = provenance.get(field)
            if not isinstance(value, str) or not value:
                errors.append("provenance.%s must be a non-empty string"
                              % field)
    for path in ("identity.domain", "identity.size_class",
                 "dimensions.length_m", "dimensions.width_m",
                 "motion.max_speed_ms", "motion.accel_ms2",
                 "motion.decel_ms2", "motion.turn_rate_dps"):
        _require(errors, doc, path)
    domain = identity.get("domain")
    if domain is not None and domain not in DOMAINS:
        errors.append("identity.domain must be one of %s" % (DOMAINS,))
    size = identity.get("size_class")
    if size is not None and size not in SIZE_CLASSES:
        errors.append("identity.size_class must be one of %s" % (SIZE_CLASSES,))
    _check_num(errors, doc, "dimensions.length_m", minimum=0.05)
    _check_num(errors, doc, "dimensions.width_m", minimum=0.05)
    _check_num(errors, doc, "dimensions.height_m", minimum=0.0)
    _check_num(errors, doc, "mass.combat_t", minimum=0.0)
    _check_num(errors, doc, "power.power_hp", minimum=0.0)
    _check_num(errors, doc, "motion.max_speed_ms", minimum=0.01)
    _check_num(errors, doc, "motion.accel_ms2", minimum=0.0)
    _check_num(errors, doc, "motion.decel_ms2", minimum=0.0)
    _check_num(errors, doc, "motion.turn_rate_dps", minimum=0.1,
               maximum=720.0)
    _check_num(errors, doc, "motion.min_turn_radius_m", minimum=0.0)
    pivot = motion.get("pivot_turn")
    if pivot is not None and not isinstance(pivot, bool):
        errors.append("motion.pivot_turn must be a boolean")
    factors = motion.get("terrain_factors")
    if not isinstance(factors, dict) or not factors:
        errors.append("motion.terrain_factors must be a non-empty object")
    else:
        for name, value in factors.items():
            if not _is_num(value) or value < 0.0 or value > 1.0:
                errors.append("motion.terrain_factors.%s must be in [0, 1]"
                              % name)
    curve = motion.get("speed_curve")
    if curve is not None:
        if not isinstance(curve, list) or len(curve) < 2:
            errors.append("motion.speed_curve must have at least two points")
        else:
            last_speed = -1.0
            for idx, point in enumerate(curve):
                if (not isinstance(point, list) or len(point) != 2
                        or not _is_num(point[0]) or not _is_num(point[1])):
                    errors.append("motion.speed_curve[%d] must be [speed, "
                                  "factor]" % idx)
                    continue
                if point[0] < last_speed:
                    errors.append("motion.speed_curve speeds must be "
                                  "non-decreasing")
                if point[1] < 0.0 or point[1] > 1.5:
                    errors.append("motion.speed_curve factors must be in "
                                  "[0, 1.5]")
                last_speed = point[0]
    _check_num(errors, doc, "terrain.max_gradient_deg", minimum=0.0,
               maximum=90.0)
    _check_num(errors, doc, "terrain.max_side_slope_deg", minimum=0.0,
               maximum=90.0)
    for path in ("terrain.wading_depth_m", "terrain.swim_depth_m",
                 "sea.draft_m", "sea.ukc_m", "sea.max_operating_depth_m",
                 "air.min_clearance_m", "air.service_ceiling_m",
                 "air.agl_m", "air.climb_rate_ms", "air.descent_rate_ms",
                 "road.weight_t", "road.height_m", "road.beam_m"):
        _check_num(errors, doc, path, minimum=0.0)
    for name in ("amphibious", "road_bound"):
        value = terrain.get(name)
        if value is not None and not isinstance(value, bool):
            errors.append("terrain.%s must be a boolean" % name)
    if armor:
        cls = armor.get("class")
        if cls is not None and (not isinstance(cls, int)
                                or isinstance(cls, bool)
                                or not 0 <= cls <= 10):
            errors.append("armor.class must be an integer in [0, 10]")
        faces = armor.get("faces_mm_rha")
        if faces is not None and not isinstance(faces, dict):
            errors.append("armor.faces_mm_rha must be an object")
        elif isinstance(faces, dict):
            for name, value in faces.items():
                if not _is_num(value) or value < 0.0:
                    errors.append("armor.faces_mm_rha.%s must be >= 0" % name)
    if signature is not None:
        for name, value in signature.items():
            if not _is_num(value) or value < 0.0:
                errors.append("signature.%s must be >= 0" % name)
    for name in ("sensors", "weapons"):
        items = doc.get(name)
        if items is not None and not isinstance(items, list):
            errors.append("%s must be a list" % name)
        elif isinstance(items, list):
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append("%s[%d] must be an object" % (name, idx))
    if errors:
        raise PlatformDataError("%s: %s" % (doc.get("key") or "<record>",
                                            "; ".join(errors)))


def profile_from_record(doc: Dict[str, Any]) -> PlatformProfile:
    """Rebuild the mobility ``PlatformProfile`` from a validated record."""
    identity = doc["identity"]
    motion = doc.get("motion") or {}
    terrain = doc.get("terrain") or {}
    sea = doc.get("sea") or {}
    air = doc.get("air") or {}
    road = doc.get("road") or {}
    dims = doc.get("dimensions") or {}
    armor = doc.get("armor") or {}

    return PlatformProfile(
        key=str(doc["key"]),
        domain=str(identity["domain"]),
        max_speed_ms=float(motion["max_speed_ms"]),
        accel_ms2=float(motion.get("accel_ms2", 1.0)),
        decel_ms2=float(motion.get("decel_ms2", 2.0)),
        turn_rate_dps=float(motion.get("turn_rate_dps", 30.0)),
        terrain_factors={str(k): float(v)
                         for k, v in (motion.get("terrain_factors") or {}).items()},
        max_gradient_deg=float(terrain.get("max_gradient_deg", 30.0)),
        max_side_slope_deg=float(terrain.get("max_side_slope_deg", 17.0)),
        wading_depth_m=float(terrain.get("wading_depth_m", 1.2)),
        swim_depth_m=float(terrain.get("swim_depth_m", 2.5)),
        amphibious=bool(terrain.get("amphibious", False)),
        road_bound=bool(terrain.get("road_bound", False)),
        draft_m=float(sea.get("draft_m", 0.0)),
        ukc_m=float(sea.get("ukc_m", 0.5)),
        beam_m=float(road.get("beam_m", 0.0)),
        max_operating_depth_m=float(sea.get("max_operating_depth_m", 0.0)),
        min_clearance_m=float(air.get("min_clearance_m", 0.0)),
        service_ceiling_m=float(air.get("service_ceiling_m", 0.0)),
        agl_m=float(air.get("agl_m", 0.0)),
        climb_rate_ms=float(air.get("climb_rate_ms", 5.0)),
        descent_rate_ms=float(air.get("descent_rate_ms", 5.0)),
        weight_t=float(road.get("weight_t", 0.0)),
        vehicle_height_m=float(dims.get("height_m", 0.0)),
        armor_class=int(armor.get("class", 0)),
        size_class=str(identity.get("size_class", "medium")),
        length_m=float(dims.get("length_m", 5.0)),
        width_m=float(dims.get("width_m", 2.5)),
        min_turn_radius_m=float(motion.get("min_turn_radius_m", 0.0)),
        pivot_turn=bool(motion.get("pivot_turn", True)),
    )


def load_platforms(directory: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load and validate every ``*.json`` record in ``directory``.

    Returns ``{key: {"doc": record, "profile": PlatformProfile}}``.  A
    missing directory yields an empty registry (constants remain the
    fallback); a malformed record raises ``PlatformDataError``.
    """
    root = directory or PLATFORM_DIR
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(root):
        return out
    names = sorted(n for n in os.listdir(root) if n.endswith(".json"))
    if not names:
        return out
    for name in names:
        path = os.path.join(root, name)
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            raise PlatformDataError("%s: %s" % (name, exc)) from exc
        try:
            validate_platform(doc)
            key = str(doc["key"])
            record = {"doc": doc, "profile": profile_from_record(doc),
                      "path": path}
        except PlatformDataError as exc:
            raise PlatformDataError("%s: %s" % (name, exc)) from exc
        if key in out:
            raise PlatformDataError("duplicate platform key %r (%s)"
                                    % (key, name))
        out[key] = record
    return out


_REGISTRY: Optional[Dict[str, Dict[str, Any]]] = None


def registry() -> Dict[str, Dict[str, Any]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_platforms()
    return _REGISTRY


def reload_registry(directory: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    global _REGISTRY
    _REGISTRY = load_platforms(directory)
    return _REGISTRY


def _safe_registry() -> Dict[str, Dict[str, Any]]:
    """Registry with constant fallback on invalid data (engine robustness)."""
    try:
        return registry()
    except PlatformDataError as exc:
        warnings.warn("platform data invalid, falling back to built-in "
                      "profiles: %s" % exc, RuntimeWarning, stacklevel=3)
        return {}


def profile_by_key(key: str) -> Optional[PlatformProfile]:
    record = _safe_registry().get(key)
    return record["profile"] if record else None


def record_by_key(key: str) -> Optional[Dict[str, Any]]:
    record = _safe_registry().get(key)
    return record["doc"] if record else None
