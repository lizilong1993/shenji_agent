"""Target lifecycle state machine (five doctrinal stages).

A target moves through a fixed sequence: identified → prioritized → in
coordination → in execution → complete.  This module owns that sequence and
the audit trail; it deliberately knows nothing about how a target looks on the
map or who is allowed to move it.

Two rules the code enforces rather than documents:

* Only the listed transitions are legal.  A drag that would skip a stage is
  rejected with a readable error instead of silently recording a jump — an
  audit trail with holes is worse than no trail.
* Every transition is appended, never overwritten.  ``history`` is the record
  an analyst reconstructs a decision from; current state is derived from it.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LIFECYCLE_VERSION = 1

STAGES = (
    "identified",
    "prioritized",
    "in_coordination",
    "in_execution",
    "complete",
)
STAGE_LABELS = {
    "identified": "已识别",
    "prioritized": "已排序",
    "in_coordination": "协同中",
    "in_execution": "执行中",
    "complete": "已完成",
}
# 允许的迁移。complete 是终态；允许从 complete 撤销回 identified 是因为
# 目标误判在实践中确实发生，把它锁死只会导致人绕过系统另开一条记录。
TRANSITIONS = {
    "identified": ("prioritized",),
    "prioritized": ("in_coordination",),
    "in_coordination": ("in_execution", "prioritized"),
    "in_execution": ("complete", "in_coordination"),
    "complete": ("identified",),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(value: Any, name: str, maximum: int = 200, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError("%s is required" % (name,))
    if len(text) > maximum:
        raise ValueError("%s exceeds %d characters" % (name, maximum))
    return text


def can_transition(source: str, target: str) -> bool:
    return target in TRANSITIONS.get(source, ())


class LifecycleStore:
    """Per-case target state plus append-only history."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()

    def _load_unlocked(self) -> dict:
        if not os.path.exists(self.path):
            return {"version": LIFECYCLE_VERSION, "targets": {}, "history": []}
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("targets"), dict):
            raise ValueError("lifecycle store is malformed")
        data.setdefault("history", [])
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
        with self._lock:
            data = self._load_unlocked()
            return "%d" % len(data["history"])

    def get(self, target_id: str) -> Optional[dict]:
        with self._lock:
            row = self._load_unlocked()["targets"].get(str(target_id))
            return copy.deepcopy(row) if isinstance(row, dict) else None

    def board(self) -> Dict[str, List[dict]]:
        """五列视图：列名 → 该列的目标（按最后变更时间倒序）。"""
        with self._lock:
            targets = self._load_unlocked()["targets"]
            columns = {stage: [] for stage in STAGES}
            for row in targets.values():
                if not isinstance(row, dict):
                    continue
                stage = row.get("stage")
                if stage not in columns:
                    continue
                columns[stage].append(copy.deepcopy(row))
        for stage in columns:
            columns[stage].sort(key=lambda row: (str(row.get("updated_at") or ""),
                                                str(row.get("target_id") or "")),
                                reverse=True)
        return columns

    def history(self, target_id: Optional[str] = None) -> List[dict]:
        with self._lock:
            rows = copy.deepcopy(self._load_unlocked()["history"])
        if target_id:
            rows = [row for row in rows if row.get("target_id") == target_id]
        rows.sort(key=lambda row: (str(row.get("at") or ""),
                                   str(row.get("seq") or "")))
        return rows

    def transit(self, payload: Dict[str, Any]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        target_id = _text(payload.get("target_id"), "target_id")
        to_stage = _text(payload.get("to_stage"), "to_stage", 32).lower()
        if to_stage not in STAGES:
            raise ValueError("to_stage must be one of " + ", ".join(STAGES))
        actor = _text(payload.get("actor"), "actor", 120, required=False)
        note = _text(payload.get("note"), "note", 1000, required=False)
        with self._lock:
            data = self._load_unlocked()
            row = data["targets"].get(target_id)
            if row is None:
                from_stage = _text(payload.get("from_stage") or "identified",
                                   "from_stage", 32).lower()
                if from_stage not in STAGES:
                    raise ValueError("from_stage must be one of "
                                     + ", ".join(STAGES))
                if from_stage != to_stage:
                    # 首次登记只能落在当前阶段，不能借"新建"跳步。
                    raise ValueError(
                        "new target must start at its own stage: "
                        "%s -> %s" % (from_stage, to_stage))
                row = {
                    "target_id": target_id,
                    "stage": to_stage,
                    "title": _text(payload.get("title"), "title", 200,
                                   required=False),
                    "created_at": _now(),
                }
            else:
                from_stage = str(row.get("stage") or "identified")
                if not can_transition(from_stage, to_stage):
                    raise ValueError(
                        "illegal transition: %s -> %s (allowed: %s)"
                        % (from_stage, to_stage,
                           ", ".join(TRANSITIONS.get(from_stage, ())) or "none"))
            stamp = _now()
            row["stage"] = to_stage
            row["updated_at"] = stamp
            data["targets"][target_id] = row
            event = {
                "seq": len(data["history"]) + 1,
                "target_id": target_id,
                "from": from_stage,
                "to": to_stage,
                "at": stamp,
                "actor": actor,
                "note": note,
            }
            data["history"].append(event)
            self._save_unlocked(data)
            return copy.deepcopy(event)
