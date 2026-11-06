"""Analyst verdicts on cross-source correlations.

``association.link_items()`` produces pairwise *links* from geometry and text.
A link is a hypothesis, not a fact: two observations that are 50 km and two
hours apart may be the same target, or may be unrelated.  This store lets an
analyst confirm or reject a specific pair, and the confirmed/rejected verdict
then overrides the algorithm's kind/score for that pair.

Only the verdict is stored.  The underlying observations are never rewritten —
the same discipline as ``manual.py`` and ``ai.py``: a decision changes how the
picture is read, never what the sources said.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

CORRELATION_VERSION = 1
VERDICTS = ("confirm", "reject")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: Any, name: str, maximum: int = 200, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError("%s is required" % (name,))
    if len(text) > maximum:
        raise ValueError("%s exceeds %d characters" % (name, maximum))
    return text


def pair_key(left: Any, right: Any) -> str:
    """Order-independent key: (a, b) and (b, a) are the same pair."""
    a, b = sorted([_text(left, "a_id"), _text(right, "b_id")])
    return a + "|" + b


class CorrelationStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()

    def _load_unlocked(self) -> dict:
        if not os.path.exists(self.path):
            return {"version": CORRELATION_VERSION, "items": []}
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError("correlation store is malformed")
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

    def revision(self) -> str:
        """Cheap change token for the overview cache key."""
        with self._lock:
            data = self._load_unlocked()
            return "%d:%d" % (len(data["items"]),
                              max((int(item.get("revision") or 0)
                                   for item in data["items"]
                                   if isinstance(item, dict)), default=0))

    def list(self, case_id: Optional[str] = None) -> List[dict]:
        with self._lock:
            data = self._load_unlocked()
            out = [copy.deepcopy(item) for item in data["items"]
                   if isinstance(item, dict)
                   and (not case_id or item.get("case_id") == case_id)]
        out.sort(key=lambda item: (str(item.get("updated_at") or ""),
                                   str(item.get("a_id") or "")))
        return out

    def verdicts(self, case_id: Optional[str] = None) -> Dict[str, dict]:
        """``pair_key -> verdict``, for overriding links in one pass."""
        return {pair_key(item.get("a_id"), item.get("b_id")): item
                for item in self.list(case_id)}

    def set(self, payload: Dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        case_id = _text(payload.get("case_id"), "case_id", 80)
        a_id = _text(payload.get("a_id"), "a_id")
        b_id = _text(payload.get("b_id"), "b_id")
        if a_id == b_id:
            raise ValueError("a_id and b_id must differ")
        verdict = _text(payload.get("verdict"), "verdict", 16).lower()
        if verdict not in VERDICTS:
            raise ValueError("verdict must be confirm or reject")
        note = _text(payload.get("note"), "note", 2000, required=False)
        actor = _text(payload.get("actor"), "actor", 120, required=False)
        with self._lock:
            data = self._load_unlocked()
            key = pair_key(a_id, b_id)
            existing = None
            highest = 0
            for item in data["items"]:
                if not isinstance(item, dict):
                    continue
                highest = max(highest, int(item.get("revision") or 0))
                if (item.get("case_id") == case_id
                        and pair_key(item.get("a_id"), item.get("b_id")) == key):
                    existing = item
            stamp = _now()
            if existing is not None:
                existing.update({
                    "verdict": verdict, "note": note,
                    "updated_at": stamp,
                    "revision": highest + 1,
                })
                if actor:
                    existing["actor"] = actor
                result = existing
            else:
                result = {
                    "correlation_id": "corr:" + uuid.uuid4().hex[:16],
                    "case_id": case_id,
                    "a_id": a_id,
                    "b_id": b_id,
                    "verdict": verdict,
                    "note": note,
                    "actor": actor,
                    "created_at": stamp,
                    "updated_at": stamp,
                    "revision": highest + 1,
                }
                data["items"].append(result)
            self._save_unlocked(data)
            return copy.deepcopy(result)

    def clear(self, correlation_id: str) -> bool:
        """Drop a verdict — the pair falls back to the algorithm's reading."""
        with self._lock:
            data = self._load_unlocked()
            kept = [item for item in data["items"]
                    if not (isinstance(item, dict)
                            and item.get("correlation_id") == correlation_id)]
            if len(kept) == len(data["items"]):
                return False
            data["items"] = kept
            self._save_unlocked(data)
            return True
