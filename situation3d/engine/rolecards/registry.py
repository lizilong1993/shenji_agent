"""Role-card registry for situation3d.

Data lives in ``engine/rolecards/data``:
  classes.json        -- 大类角色卡（字段契约与共性，不含小类数值）
  subclasses/*.json   -- 小类角色卡（具体数值/装备/载员）
  equipment.json      -- 武器/传感器/通信目录
  types.json          -- 类型别名与庙算 type/sub_type 映射

The module is deliberately dependency-free (stdlib only) and does not import
``engine`` or the web code, so the same registry can back the offline dataset
generator, the geo engine, and the front end export.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import validate as _validate

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

CLASS_SCHEMA = "situation3d.rolecard-class.v1"
SUBCLASS_SCHEMA = "situation3d.rolecard-subclass.v1"
EQUIPMENT_SCHEMA = "situation3d.equipment.v2"
TYPES_SCHEMA = "situation3d.types.v2"

GENERIC_BY_DOMAIN = {
    "land": "icon.generic_land",
    "sea": "icon.generic_sea",
    "air": "icon.generic_air",
    "subsurface": "icon.generic_sea",
    "installation": "icon.generic_land",
}


class RoleCardError(ValueError):
    """Raised when the registry is missing data or fails validation."""


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise RoleCardError("role-card data file missing: %s" % path) from None
    except json.JSONDecodeError as exc:
        raise RoleCardError("invalid JSON in %s: %s" % (path, exc)) from None


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    - dicts merge key by key;
    - lists and scalars replace the base value;
    - ``None`` explicitly clears the field.
    """
    out: Dict[str, Any] = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class RoleCardRegistry(object):
    """Loaded and validated role-card registry."""

    def __init__(self, data_dir: Optional[str] = None, validate: bool = True):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.classes: Dict[str, dict] = {}
        self.subclasses: Dict[str, dict] = {}
        self.equipment: Dict[str, dict] = {"weapons": {}, "sensors": {}, "comms": {}}
        self.types: dict = {}
        self.aliases: Dict[str, str] = {}
        self.sdk_map: Dict[str, str] = {}
        self._load()
        if validate:
            errors = self.validate()
            if errors:
                raise RoleCardError("role-card registry invalid:\n- " + "\n- ".join(errors))

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        classes_doc = _load_json(self.data_dir / "classes.json")
        if classes_doc.get("schema") != "situation3d.rolecard-classes.v1":
            raise RoleCardError("unsupported classes schema: %r" % classes_doc.get("schema"))
        for card in classes_doc.get("classes") or []:
            if card["class_id"] in self.classes:
                raise RoleCardError("duplicate class id: %s" % card["class_id"])
            self.classes[card["class_id"]] = card

        sub_dir = self.data_dir / "subclasses"
        if not sub_dir.is_dir():
            raise RoleCardError("missing subclasses directory: %s" % sub_dir)
        for path in sorted(sub_dir.glob("*.json")):
            doc = _load_json(path)
            if doc.get("schema") != "situation3d.rolecard-subclasses.v1":
                raise RoleCardError("unsupported subclass schema in %s" % path)
            for card in doc.get("subclasses") or []:
                if card["subclass_id"] in self.subclasses:
                    raise RoleCardError("duplicate subclass id: %s" % card["subclass_id"])
                self.subclasses[card["subclass_id"]] = card

        equipment_doc = _load_json(self.data_dir / "equipment.json")
        if equipment_doc.get("schema") != EQUIPMENT_SCHEMA:
            raise RoleCardError("unsupported equipment schema: %r" % equipment_doc.get("schema"))
        for kind in ("weapons", "sensors", "comms"):
            self.equipment[kind] = equipment_doc.get(kind) or {}

        self.types = _load_json(self.data_dir / "types.json")
        if self.types.get("schema") != TYPES_SCHEMA:
            raise RoleCardError("unsupported types schema: %r" % self.types.get("schema"))
        self.aliases = dict(self.types.get("aliases") or {})
        self.sdk_map = dict(self.types.get("sdk_map") or {})

    # -- lookups -----------------------------------------------------------
    def class_of(self, subclass_id: str) -> dict:
        try:
            sub = self.subclasses[subclass_id]
        except KeyError:
            raise RoleCardError("unknown subclass: %s" % subclass_id) from None
        try:
            return self.classes[sub["class_id"]]
        except KeyError:
            raise RoleCardError("subclass %s references unknown class %s"
                                % (subclass_id, sub["class_id"])) from None

    def unknown_for(self, class_id: str) -> str:
        try:
            return self.classes[class_id]["unknown_subclass"]
        except KeyError:
            raise RoleCardError("unknown class: %s" % class_id) from None

    def list_classes(self) -> List[dict]:
        return [self.classes[k] for k in sorted(self.classes)]

    def list_subclasses(self, class_id: Optional[str] = None) -> List[dict]:
        if class_id is None:
            return [self.subclasses[k] for k in sorted(self.subclasses)]
        return [self.subclasses[k] for k in self.classes[class_id].get("subclasses", [])]

    def resolve_type(self, key: str) -> Optional[str]:
        """Resolve a subclass id, ``icon.<key>`` alias, raw icon or SDK key.

        Returns ``None`` when nothing matches; callers decide whether to fall
        back to ``<class>.unknown``.
        """
        if not key or not isinstance(key, str):
            return None
        if key in self.subclasses:
            return key
        for candidate in (key, "icon." + key):
            if candidate in self.aliases:
                return self.aliases[candidate]
        if key in self.sdk_map:
            return self.sdk_map[key]
        return None

    def subclass_for_legacy(self, icon: str, domain: str) -> str:
        """Best-effort mapping from a legacy target to a subclass id.

        The alias table is the single mapping; an unmapped icon falls back to
        the domain's ``generic_*`` alias (and therefore to ``<class>.unknown``).
        """
        for candidate in ("icon." + (icon or ""), GENERIC_BY_DOMAIN.get(domain, "")):
            if candidate and candidate in self.aliases:
                return self.aliases[candidate]
        raise RoleCardError("cannot map legacy icon %r/domain %r to a subclass" % (icon, domain))

    # -- effective cards ---------------------------------------------------
    def effective(self, subclass_id: str, overrides: Optional[dict] = None) -> dict:
        """Merge class shared values, subclass values and instance overrides."""
        cls = self.class_of(subclass_id)
        sub = self.subclasses[subclass_id]
        merged = deep_merge(cls.get("shared") or {}, sub.get("values") or {})
        if overrides:
            merged = deep_merge(merged, overrides)
        return merged

    def equipment_item(self, kind: str, ref: str) -> dict:
        try:
            return self.equipment[kind][ref]
        except KeyError:
            raise RoleCardError("unknown %s ref: %s" % (kind, ref)) from None

    # -- validation --------------------------------------------------------
    def validate(self) -> List[str]:
        return _validate.validate_registry(
            self.classes, self.subclasses, self.equipment, self.types)

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise RoleCardError("role-card registry invalid:\n- " + "\n- ".join(errors))


def load_registry(data_dir: Optional[str] = None, validate: bool = True) -> RoleCardRegistry:
    return RoleCardRegistry(data_dir=data_dir, validate=validate)
