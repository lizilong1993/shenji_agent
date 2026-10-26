"""User-managed intelligence source registry.

Built-in connectors are seeded here so the UI can list, search and disable them;
custom entries are metadata (RSS/social/web/etc.) that can be attached to manual
intel or a future generic connector.  Deleting a built-in source is refused.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .taxonomy import normalize_domains, normalize_source_kind

SOURCE_VERSION = 1
_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("false", "0", "no", "off", ""):
        return False
    if text in ("true", "1", "yes", "on"):
        return True
    return bool(value)


def _tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text[:40])
    return out[:12]


def _builtin_entries() -> List[dict]:
    try:
        from .connectors.registry import SOURCES
    except Exception:  # pragma: no cover - a partial deployment still lists manual
        SOURCES = {}
    entries = [{
        "source_id": "manual",
        "name": "人工录入",
        "kind": "manual",
        "domains": [],
        "url": "",
        "description": "作战/分析人员结构化录入的人工情报",
        "tags": ["manual", "人工"],
        "enabled": True,
        "builtin": True,
        "license": "",
        "attribution": "人工录入",
        "created_at": None,
        "updated_at": None,
    }]
    for source_id, spec in sorted(SOURCES.items()):
        domains = {
            "usgs": ["hazard"], "gdacs": ["hazard"], "firms": ["hazard"],
            "eonet": ["hazard"], "adsb": ["aviation"],
            "gdelt": ["news", "conflict", "society", "selfmedia"],
        }.get(source_id, [])
        entries.append({
            "source_id": source_id,
            "name": source_id.upper(),
            "kind": "auto",
            "domains": domains,
            "url": "https://%s" % (spec.host,),
            "description": spec.attribution,
            "tags": [source_id],
            "enabled": True,
            "builtin": True,
            "license": spec.license,
            "attribution": spec.attribution,
            "created_at": None,
            "updated_at": None,
        })
    return entries


class SourceStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()

    def _load_unlocked(self) -> dict:
        if not os.path.exists(self.path):
            return {"version": SOURCE_VERSION, "items": _builtin_entries()}
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError("source store is malformed")
        known = {item.get("source_id") for item in data["items"]
                 if isinstance(item, dict)}
        for entry in _builtin_entries():
            if entry["source_id"] not in known:
                data["items"].append(entry)
            else:
                for item in data["items"]:
                    if (item.get("source_id") == entry["source_id"]
                            and not item.get("domains")):
                        item["domains"] = entry["domains"]
        return data

    def _save_unlocked(self, data: dict) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        temporary = self.path + ".tmp-" + uuid.uuid4().hex
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1,
                      sort_keys=True)
            handle.write("\n")
        os.replace(temporary, self.path)

    def list(self, q: Optional[str] = None, kind: Optional[str] = None,
             enabled: Optional[bool] = None) -> List[dict]:
        text = str(q or "").strip().lower()
        with self._lock:
            items = [copy.deepcopy(item)
                     for item in self._load_unlocked()["items"]
                     if isinstance(item, dict)]
        out = []
        for item in items:
            if kind and item.get("kind") != kind:
                continue
            if enabled is not None and bool(item.get("enabled")) != enabled:
                continue
            haystack = " ".join([
                str(item.get("source_id") or ""),
                str(item.get("name") or ""),
                str(item.get("description") or ""),
                " ".join(item.get("tags") or []),
                " ".join(item.get("domains") or []),
            ]).lower()
            if text and text not in haystack:
                continue
            out.append(item)
        out.sort(key=lambda item: (not bool(item.get("builtin")),
                                   str(item.get("name") or "").lower()))
        return out

    def get(self, source_id: str) -> dict:
        with self._lock:
            for item in self._load_unlocked()["items"]:
                if item.get("source_id") == source_id:
                    return copy.deepcopy(item)
        raise KeyError("source not found")

    def add(self, payload: Dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        source_id = str(payload.get("source_id") or "").strip().lower()
        if not _SOURCE_ID.match(source_id):
            raise ValueError(
                "source_id must be 2-32 lowercase letters/digits/_/-")
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        stamp = _now()
        item = {
            "source_id": source_id,
            "name": name[:80],
            "kind": normalize_source_kind(payload.get("kind") or "other"),
            "domains": list(normalize_domains(payload.get("domains"))),
            "url": str(payload.get("url") or "").strip()[:500],
            "description": str(payload.get("description") or "").strip()[:500],
            "tags": _tags(payload.get("tags")),
            "enabled": _as_bool(payload.get("enabled", True), True),
            "builtin": False,
            "license": str(payload.get("license") or "").strip()[:200],
            "attribution": str(payload.get("attribution") or name).strip()[:200],
            "created_at": stamp,
            "updated_at": stamp,
        }
        with self._lock:
            data = self._load_unlocked()
            if any(item.get("source_id") == source_id for item in data["items"]):
                raise ValueError("source_id already exists")
            data["items"].append(item)
            self._save_unlocked(data)
        return copy.deepcopy(item)

    def update(self, source_id: str, payload: Dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        with self._lock:
            data = self._load_unlocked()
            if payload.get("name") is not None and not str(
                    payload.get("name")).strip():
                raise ValueError("name cannot be empty")
            for item in data["items"]:
                if item.get("source_id") != source_id:
                    continue
                if item.get("builtin"):
                    # Built-ins are fixed identities; only presentation and
                    # enabled state may change.
                    for key in ("name", "description", "tags", "enabled"):
                        if payload.get(key) is not None:
                            item[key] = (_as_bool(payload[key])
                                         if key == "enabled"
                                         else (str(payload[key]).strip()[:500]
                                               if key != "tags"
                                               else _tags(payload[key])))
                else:
                    for key in ("name", "kind", "url", "description",
                                "tags", "enabled", "license", "attribution"):
                        if payload.get(key) is None:
                            continue
                        if key == "kind":
                            item[key] = normalize_source_kind(payload[key])
                        elif key == "tags":
                            item[key] = _tags(payload[key])
                        elif key == "enabled":
                            item[key] = _as_bool(payload[key])
                        else:
                            item[key] = str(payload[key]).strip()[:500]
                    if payload.get("domains") is not None:
                        item["domains"] = list(normalize_domains(
                            payload["domains"]))
                item["updated_at"] = _now()
                self._save_unlocked(data)
                return copy.deepcopy(item)
        raise KeyError("source not found")

    def delete(self, source_id: str) -> dict:
        with self._lock:
            data = self._load_unlocked()
            for index, item in enumerate(data["items"]):
                if item.get("source_id") != source_id:
                    continue
                if item.get("builtin"):
                    raise ValueError("built-in sources cannot be deleted")
                removed = data["items"].pop(index)
                self._save_unlocked(data)
                return copy.deepcopy(removed)
        raise KeyError("source not found")

    def revision(self) -> str:
        with self._lock:
            items = self._load_unlocked()["items"]
            parts = ["%s:%s" % (item.get("source_id"), item.get("updated_at"))
                     for item in items if isinstance(item, dict)]
            return "|".join(sorted(parts))
