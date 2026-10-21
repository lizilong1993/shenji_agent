"""Structural and cross-field validation for the role-card registry.

The checks are intentionally dependency-free (no ``jsonschema`` package).  The
JSON Schema files under ``schema/`` document the contract; this module is the
runtime enforcement used by tests and the dataset generator.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional

CONFIDENCE_LEVELS = {"high", "medium", "low", "unknown", "legacy"}
DOMAINS = {"land", "sea", "air", "subsurface", "installation"}
ARMOR_CLASSES = {0, 1, 2, 3, 4}

# Class cards must not carry numeric capability values; field names appearing in
# ``required_values`` are fine, actual values under these keys are not.
FORBIDDEN_CLASS_VALUE_KEYS = {
    "max_speed_ms", "cruise_speed_ms", "accel_ms2", "decel_ms2",
    "armor_class", "max_blood", "signature", "concealment",
    "sensors", "weapons", "terrain", "sea", "air",
}

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _get_path(doc: Any, path: str) -> Any:
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _walk_any(doc: Any, path: str = "") -> Iterable[tuple]:
    """Walk dicts and lists; class cards are checked in full, not just shared."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            here = path + "." + key if path else key
            yield here, key, value
            yield from _walk_any(value, here)
    elif isinstance(doc, list):
        for idx, value in enumerate(doc):
            yield from _walk_any(value, "%s[%d]" % (path, idx))


def validate_registry(classes: Dict[str, dict], subclasses: Dict[str, dict],
                      equipment: Dict[str, dict], types: dict) -> List[str]:
    errors: List[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    # -- classes -----------------------------------------------------------
    for class_id, cls in classes.items():
        if cls.get("schema") != "situation3d.rolecard-class.v1":
            err("class %s: bad schema %r" % (class_id, cls.get("schema")))
        if cls.get("class_id") != class_id:
            err("class %s: class_id mismatch %r" % (class_id, cls.get("class_id")))
        for key in ("label", "domain", "shared", "required_values",
                    "equipment_policy", "unknown_subclass", "subclasses"):
            if key not in cls:
                err("class %s: missing %s" % (class_id, key))
        if cls.get("domain") not in DOMAINS:
            err("class %s: bad domain %r" % (class_id, cls.get("domain")))
        for path, key, value in _walk_any(cls):
            if key in FORBIDDEN_CLASS_VALUE_KEYS:
                err("class %s: carries subclass capability value %s" % (class_id, path))
        if cls.get("unknown_subclass") not in (cls.get("subclasses") or []):
            err("class %s: unknown_subclass %r not listed"
                % (class_id, cls.get("unknown_subclass")))

    # -- subclasses --------------------------------------------------------
    listed = {}
    for class_id, cls in classes.items():
        for sid in cls.get("subclasses") or []:
            listed.setdefault(sid, []).append(class_id)
    for sid, owners in listed.items():
        if len(set(owners)) > 1:
            err("subclass %s listed under multiple classes: %s" % (sid, sorted(set(owners))))
    for subclass_id, sub in subclasses.items():
        if sub.get("schema") != "situation3d.rolecard-subclass.v1":
            err("subclass %s: bad schema %r" % (subclass_id, sub.get("schema")))
        if sub.get("subclass_id") != subclass_id:
            err("subclass %s: subclass_id mismatch %r" % (subclass_id, sub.get("subclass_id")))
        class_id = sub.get("class_id")
        if class_id not in classes:
            err("subclass %s: unknown class %r" % (subclass_id, class_id))
            continue
        if class_id not in listed.get(subclass_id, []):
            err("subclass %s: not registered under class %s" % (subclass_id, class_id))
        if sub.get("confidence") not in CONFIDENCE_LEVELS:
            err("subclass %s: bad confidence %r" % (subclass_id, sub.get("confidence")))
        for key in ("label", "sdk", "values"):
            if key not in sub:
                err("subclass %s: missing %s" % (subclass_id, key))
        values = sub.get("values")
        if not isinstance(values, dict):
            err("subclass %s: values must be an object" % subclass_id)
            continue
        for key in ("identity", "motion", "protection", "sensors", "weapons", "capacity"):
            if key not in values:
                err("subclass %s: values.%s missing" % (subclass_id, key))

        cls = classes[class_id]
        merged = _deep_merge(cls.get("shared") or {}, values)
        for path in cls.get("required_values") or []:
            if _get_path(merged, path) is None:
                err("subclass %s: required value %s missing after merge"
                    % (subclass_id, path))

        identity = values.get("identity")
        if not isinstance(identity, dict):
            err("subclass %s: values.identity must be an object" % subclass_id)
            identity = {}
        symbol = identity.get("symbol")
        if not isinstance(symbol, dict) or not symbol.get("icon"):
            err("subclass %s: identity.symbol.icon missing" % subclass_id)
        if identity.get("primary_domain") and identity.get("primary_domain") != cls.get("domain"):
            err("subclass %s: primary_domain %r != class domain %r"
                % (subclass_id, identity.get("primary_domain"), cls.get("domain")))

        protection = values.get("protection")
        if not isinstance(protection, dict):
            err("subclass %s: values.protection must be an object" % subclass_id)
            protection = {}
        armor = protection.get("armor_class")
        if not isinstance(armor, int) or armor not in ARMOR_CLASSES:
            err("subclass %s: armor_class %r not in 0..4" % (subclass_id, armor))
        blood = protection.get("max_blood")
        if not isinstance(blood, int) or blood < 1:
            err("subclass %s: max_blood must be a positive int" % subclass_id)
        if protection.get("size_class") not in ("small", "medium", "large"):
            err("subclass %s: size_class must be small/medium/large" % subclass_id)
        sig = protection.get("signature") or {}
        for kind, factor in sig.items():
            if not isinstance(factor, (int, float)) or factor < 0:
                err("subclass %s: signature.%s must be >=0" % (subclass_id, kind))

        motion = values.get("motion")
        if not isinstance(motion, dict):
            err("subclass %s: values.motion must be an object" % subclass_id)
            motion = {}
        max_ms = motion.get("max_speed_ms")
        cruise_ms = motion.get("cruise_speed_ms")
        if max_ms is None:
            err("subclass %s: motion.max_speed_ms missing" % subclass_id)
        elif not isinstance(max_ms, (int, float)) or isinstance(max_ms, bool):
            err("subclass %s: max_speed_ms must be a number" % subclass_id)
        elif max_ms < 0:
            err("subclass %s: max_speed_ms must be >=0" % subclass_id)
        elif max_ms == 0:
            if not motion.get("immobile"):
                err("subclass %s: zero max speed must set motion.immobile=true" % subclass_id)
        else:
            if not isinstance(cruise_ms, (int, float)) or isinstance(cruise_ms, bool) or cruise_ms <= 0:
                err("subclass %s: cruise_speed_ms must be >0" % subclass_id)
            elif cruise_ms > max_ms:
                err("subclass %s: cruise_speed_ms > max_speed_ms" % subclass_id)
        terrain = motion.get("terrain") or {}
        if not isinstance(terrain, dict):
            err("subclass %s: motion.terrain must be an object" % subclass_id)
            terrain = {}
        if terrain.get("amphibious"):
            if motion.get("swim_speed_ms") is None and terrain.get("swim_speed_ms") is None:
                err("subclass %s: amphibious without swim_speed_ms" % subclass_id)
            if terrain.get("swim_depth_m") is None:
                err("subclass %s: amphibious without swim_depth_m" % subclass_id)
        sea_motion = motion.get("sea") or {}
        air_motion = motion.get("air") or {}
        if not isinstance(sea_motion, dict):
            err("subclass %s: motion.sea must be an object" % subclass_id)
            sea_motion = {}
        if not isinstance(air_motion, dict):
            err("subclass %s: motion.air must be an object" % subclass_id)
            air_motion = {}
        domain = values.get("identity", {}).get("primary_domain") or cls.get("domain")
        if domain == "sea" and not sea_motion.get("draft_m"):
            err("subclass %s: sea platform without motion.sea.draft_m" % subclass_id)
        if domain == "subsurface" and not sea_motion.get("max_operating_depth_m"):
            err("subclass %s: submarine without max_operating_depth_m" % subclass_id)
        if domain == "air" and not air_motion.get("service_ceiling_m"):
            err("subclass %s: air platform without service_ceiling_m" % subclass_id)

        policy = cls.get("equipment_policy") or {}
        allowed_weapons = set(policy.get("weapon_categories") or [])
        allowed_sensors = set(policy.get("sensor_kinds") or [])
        for item in values.get("weapons") or []:
            ref = item.get("ref")
            spec = (equipment.get("weapons") or {}).get(ref)
            if spec is None:
                err("subclass %s: unknown weapon ref %r" % (subclass_id, ref))
                continue
            if spec.get("category") not in allowed_weapons:
                err("subclass %s: weapon %s category %s not allowed by class"
                    % (subclass_id, ref, spec.get("category")))
            ammo = item.get("ammo_max")
            default = item.get("ammo_default", ammo)
            if not isinstance(ammo, int) or isinstance(ammo, bool) or ammo < 0:
                err("subclass %s: ammo_max must be a non-negative int for %s"
                    % (subclass_id, ref))
            elif not isinstance(default, int) or isinstance(default, bool) or default < 0:
                err("subclass %s: ammo_default must be a non-negative int for %s"
                    % (subclass_id, ref))
            elif default > ammo:
                err("subclass %s: ammo_default > ammo_max for %s" % (subclass_id, ref))
        for item in values.get("sensors") or []:
            ref = item.get("ref")
            spec = (equipment.get("sensors") or {}).get(ref)
            if spec is None:
                err("subclass %s: unknown sensor ref %r" % (subclass_id, ref))
                continue
            if spec.get("kind") not in allowed_sensors:
                err("subclass %s: sensor %s kind %s not allowed by class"
                    % (subclass_id, ref, spec.get("kind")))
        for item in values.get("comms") or []:
            ref = item.get("ref")
            if ref not in (equipment.get("comms") or {}):
                err("subclass %s: unknown comm ref %r" % (subclass_id, ref))

        if sub.get("confidence") == "unknown":
            policy = sub.get("unknown_policy") or {}
            if policy.get("engage") is not False:
                err("subclass %s: unknown card must set unknown_policy.engage=false"
                    % subclass_id)
            if values.get("weapons"):
                err("subclass %s: unknown card must not carry weapons" % subclass_id)

    for subclass_id in listed:
        if subclass_id not in subclasses:
            err("class registry lists missing subclass %s" % subclass_id)

    # -- types -------------------------------------------------------------
    for key, value in (types.get("aliases") or {}).items():
        if value not in subclasses:
            err("alias %s -> unknown subclass %s" % (key, value))
    for key, value in (types.get("sdk_map") or {}).items():
        if value not in subclasses:
            err("sdk_map %s -> unknown subclass %s" % (key, value))

    return errors


def raise_if_errors(errors: List[str]) -> None:
    if errors:
        raise ValueError("role-card validation failed:\n- " + "\n- ".join(errors))
