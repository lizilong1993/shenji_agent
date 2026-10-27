"""AI suggestion store with analyst review (no automatic verification).

AI workers write *suggestions*; only an analyst review changes their status.
The original claims, evidence and source text are immutable so a rejected or
corrected suggestion can always be audited.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

AI_VERSION = 1
AI_TASKS = ("extract", "translate", "summarize", "geolocate",
            "change_detect", "associate", "other")
AI_STATUSES = ("suggested", "accepted", "rejected", "needs_review")
_TASK = re.compile(r"^[a-z_]{2,32}$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value[:4000]]
    if not isinstance(value, (list, tuple)):
        return []
    return [item if isinstance(item, (dict, list)) else str(item)[:4000]
            for item in value][:100]


class AISuggestionStore:
    """Per-case JSON store for AI suggestions and analyst decisions."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()

    def _load_unlocked(self) -> dict:
        if not os.path.exists(self.path):
            return {"version": AI_VERSION, "items": []}
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError("AI suggestion store is malformed")
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

    def add(self, payload: Dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        task = _text(payload.get("task"), 32).lower() or "other"
        if not _TASK.match(task):
            raise ValueError("invalid task")
        if task not in AI_TASKS:
            task = "other"
        model = _text(payload.get("model"), 120)
        prompt_version = _text(payload.get("prompt_version"), 80)
        if not model or not prompt_version:
            raise ValueError("model and prompt_version are required")
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            raise ValueError("confidence must be a number") from None
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        claims = payload.get("claims")
        if claims is None:
            claims = {}
        if not isinstance(claims, (dict, list)):
            raise ValueError("claims must be an object or list")
        evidence = _list(payload.get("evidence"))
        if not evidence:
            raise ValueError("evidence is required")
        item = {
            "id": "ai:" + uuid.uuid4().hex,
            "status": "suggested",
            "task": task,
            "title": _text(payload.get("title"), 200),
            "observation_id": _text(payload.get("observation_id"), 200),
            "claims": copy.deepcopy(claims),
            "evidence": evidence,
            "confidence": round(confidence, 3),
            "model": model,
            "prompt_version": prompt_version,
            "source_text": _text(payload.get("source_text"), 20000),
            "created_at": _now(),
            "reviewed_at": None,
            "reviewed_by": "",
            "review_note": "",
        }
        with self._lock:
            data = self._load_unlocked()
            data["items"].append(item)
            self._save_unlocked(data)
        return copy.deepcopy(item)

    def list(self, status: Optional[str] = None) -> List[dict]:
        with self._lock:
            items = [copy.deepcopy(item)
                     for item in self._load_unlocked()["items"]
                     if isinstance(item, dict)]
        if status:
            items = [item for item in items if item.get("status") == status]
        items.sort(key=lambda item: str(item.get("created_at") or ""),
                   reverse=True)
        return items

    def review(self, suggestion_id: str, status: str, note: str = "",
               reviewer: str = "") -> dict:
        status = _text(status, 32).lower()
        if status not in AI_STATUSES:
            raise ValueError("invalid review status")
        with self._lock:
            data = self._load_unlocked()
            for item in data["items"]:
                if item.get("id") != suggestion_id:
                    continue
                # Only review metadata changes; claims/evidence stay intact.
                item["status"] = status
                item["reviewed_at"] = _now()
                item["review_note"] = _text(note, 1000)
                item["reviewed_by"] = _text(reviewer, 80)
                self._save_unlocked(data)
                return copy.deepcopy(item)
        raise KeyError("AI suggestion not found")

    def counts(self) -> dict:
        items = self.list()
        return {
            "total": len(items),
            "suggested": sum(1 for item in items
                             if item.get("status") == "suggested"),
            "accepted": sum(1 for item in items
                            if item.get("status") == "accepted"),
            "rejected": sum(1 for item in items
                            if item.get("status") == "rejected"),
            "needs_review": sum(1 for item in items
                                if item.get("status") == "needs_review"),
        }
