"""Persistent structured manual intelligence entries."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .taxonomy import (DOMAINS, normalize_type, normalize_verification)


def _domain(value: Any) -> str:
    raw = str(value or "other").strip().lower()
    if raw not in DOMAINS:
        raise ValueError("unknown domain %r" % (value,))
    return raw

MANUAL_VERSION = 1
STATUSES = ("active", "revoked")
_CASE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("event_time is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("event_time must be ISO 8601") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: Any, name: str, maximum: int = 2000,
          required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError("%s is required" % (name,))
    if len(text) > maximum:
        raise ValueError("%s exceeds %d characters" % (name, maximum))
    return text


def _coord(value: Any, name: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be numeric" % (name,)) from None
    if not math.isfinite(number) or number < low or number > high:
        raise ValueError("%s out of range" % (name,))
    return round(number, 6)


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


def _case_id(value: Any) -> str:
    text = str(value or "").strip()
    if not _CASE_ID.match(text):
        raise ValueError("case_id is required and must be a safe identifier")
    return text


class ManualIntelStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()

    def _load_unlocked(self) -> dict:
        if not os.path.exists(self.path):
            return {"version": MANUAL_VERSION, "items": []}
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError("manual intel store is malformed")
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

    def list(self, case_id: Optional[str] = None,
             include_revoked: bool = True) -> List[dict]:
        with self._lock:
            data = self._load_unlocked()
            out = []
            for item in data["items"]:
                if not isinstance(item, dict):
                    continue
                if case_id and item.get("case_id") != case_id:
                    continue
                if not include_revoked and item.get("status") != "active":
                    continue
                out.append(copy.deepcopy(item))
            out.sort(key=lambda item: (str(item.get("event_time") or ""),
                                       str(item.get("intel_id") or "")))
            return out

    def get(self, intel_id: str) -> dict:
        with self._lock:
            for item in self._load_unlocked()["items"]:
                if item.get("intel_id") == intel_id:
                    return copy.deepcopy(item)
        raise KeyError("manual intel not found")

    def create(self, payload: Dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        stamp = _now()
        case_id = _case_id(payload.get("case_id"))
        intel_type = normalize_type(payload.get("intel_type")
                                    or payload.get("type"))
        item = {
            "intel_id": "manual:" + uuid.uuid4().hex,
            "case_id": case_id,
            "title": _text(payload.get("title"), "title", 200),
            "domain": _domain(payload.get("domain")),
            "intel_type": intel_type,
            "event_time": _time(payload.get("event_time")),
            "lat": _coord(payload.get("lat"), "lat", -90.0, 90.0),
            "lon": _coord(payload.get("lon"), "lon", -180.0, 180.0),
            "place": _text(payload.get("place"), "place", 200,
                           required=False),
            "source_id": _text(payload.get("source_id") or "manual",
                               "source_id", 80),
            "source_name": _text(payload.get("source_name")
                                 or "人工录入", "source_name", 120),
            "content": _text(payload.get("content"), "content", 4000),
            "verification": normalize_verification(
                payload.get("verification")),
            "priority": "manual",
            "status": "active",
            "tags": _tags(payload.get("tags")),
            "created_by": _text(payload.get("created_by"), "created_by",
                                80, required=False),
            "created_at": stamp,
            "updated_at": stamp,
            "revision": 1,
        }
        with self._lock:
            data = self._load_unlocked()
            data["items"].append(item)
            self._save_unlocked(data)
        return copy.deepcopy(item)

    def update(self, intel_id: str, payload: Dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        with self._lock:
            data = self._load_unlocked()
            for item in data["items"]:
                if item.get("intel_id") != intel_id:
                    continue
                if payload.get("title") is not None:
                    item["title"] = _text(payload["title"], "title", 200)
                if payload.get("domain") is not None:
                    item["domain"] = _domain(payload["domain"])
                if payload.get("intel_type") is not None or payload.get("type") is not None:
                    item["intel_type"] = normalize_type(
                        payload.get("intel_type") or payload.get("type"))
                if payload.get("event_time") is not None:
                    item["event_time"] = _time(payload["event_time"])
                if payload.get("lat") is not None:
                    item["lat"] = _coord(payload["lat"], "lat", -90.0, 90.0)
                if payload.get("lon") is not None:
                    item["lon"] = _coord(payload["lon"], "lon", -180.0, 180.0)
                if payload.get("place") is not None:
                    item["place"] = _text(payload["place"], "place", 200,
                                          required=False)
                if payload.get("source_id") is not None:
                    item["source_id"] = _text(payload["source_id"], "source_id", 80)
                if payload.get("source_name") is not None:
                    item["source_name"] = _text(payload["source_name"],
                                                "source_name", 120)
                if payload.get("content") is not None:
                    item["content"] = _text(payload["content"], "content", 4000)
                if payload.get("verification") is not None:
                    item["verification"] = normalize_verification(
                        payload["verification"])
                if payload.get("tags") is not None:
                    item["tags"] = _tags(payload["tags"])
                if payload.get("status") is not None:
                    status = str(payload["status"]).lower()
                    if status not in STATUSES:
                        raise ValueError("status must be active or revoked")
                    item["status"] = status
                item["updated_at"] = _now()
                item["revision"] = int(item.get("revision") or 1) + 1
                self._save_unlocked(data)
                return copy.deepcopy(item)
        raise KeyError("manual intel not found")

    def revoke(self, intel_id: str) -> dict:
        with self._lock:
            data = self._load_unlocked()
            for item in data["items"]:
                if item.get("intel_id") != intel_id:
                    continue
                item["status"] = "revoked"
                item["updated_at"] = _now()
                item["revision"] = int(item.get("revision") or 1) + 1
                self._save_unlocked(data)
                return copy.deepcopy(item)
        raise KeyError("manual intel not found")

    def revision(self) -> str:
        with self._lock:
            data = self._load_unlocked()
            parts = ["%s:%s" % (item.get("intel_id"), item.get("revision"))
                     for item in data["items"] if isinstance(item, dict)]
            return "|".join(sorted(parts))
