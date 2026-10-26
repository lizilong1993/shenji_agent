"""Resource registry contracts (P0).

A resource entry is a *pinned reference* to an artifact already present on
disk (an envpack directory, a role-card data directory, a road network file).
The registry records enough provenance to answer three questions later:

* what exactly was used        -- ``content_hash`` + ``size`` + ``schema``
* where it came from           -- ``provider`` / ``source`` / ``license``
* when it describes the world  -- ``as_of`` (data vintage) vs ``fetched_at``

P0 only freezes the contracts and indexes existing artifacts.  Fetching,
resolution, criticality decisions and engine wiring are later phases; see
``docs/engine_task_resource_plan_2026-10-26.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

INDEX_SCHEMA = "s3d.resources.index.v1"
PLAN_SCHEMA = "s3d.resources.plan.v1"

CRITICALITIES = ("critical", "advisory", "cosmetic")
QUALITIES = ("measured", "derived", "default", "unknown")
DECISION_ACTIONS = ("use_cached_stale", "use_fallback", "fetch_online",
                    "provide_file", "cancel")

_ENTRY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*:[A-Za-z0-9][A-Za-z0-9_.:@-]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_path(path: str) -> Tuple[str, int]:
    """Content hash and byte size for a file or a directory tree.

    Directories are hashed over ``(relative path, size, bytes)`` in sorted
    order, so rebuilding a layer with identical content keeps the same hash
    regardless of filesystem ordering or mtimes.
    """
    if os.path.isfile(path):
        h = hashlib.sha256()
        size = 0
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                size += len(chunk)
                h.update(chunk)
        return h.hexdigest(), size
    if not os.path.isdir(path):
        raise FileNotFoundError("resource path not found: %s" % path)
    h = hashlib.sha256()
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            size = os.path.getsize(full)
            total += size
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(str(size).encode("ascii"))
            h.update(b"\0")
            with open(full, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest(), total


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9_.-]+", "-", str(text).strip().lower())
    return value.strip("-.") or "unnamed"


def _bbox_list(value) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        raise ValueError("bbox must be four finite numbers")
    if len(values) != 4:
        raise ValueError("bbox must be four finite numbers")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("bbox must be four finite numbers")
    return values


@dataclass(frozen=True)
class ResourceEntry:
    """One indexed artifact.  Paths are absolute so entries survive a cwd change."""

    entry_id: str
    kind: str
    path: str
    content_hash: str
    size: int
    version: str = ""
    schema: str = ""
    provider: str = ""
    source: str = ""
    fetched_at: str = ""
    as_of: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None
    license: str = ""
    quality: str = "measured"
    pinned: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ResourceEntry":
        if not _ENTRY_ID_RE.match(str(self.entry_id or "")):
            raise ValueError("invalid entry_id %r (expected kind:name)" % (self.entry_id,))
        if not self.kind:
            raise ValueError("kind is required")
        if not self.path:
            raise ValueError("path is required")
        if not os.path.isabs(self.path):
            raise ValueError("path must be absolute: %r" % (self.path,))
        if not re.match(r"^[0-9a-f]{64}$", str(self.content_hash or "")):
            raise ValueError("content_hash must be a sha256 hex digest")
        if not isinstance(self.size, int) or self.size < 0:
            raise ValueError("size must be a non-negative integer")
        if self.quality not in QUALITIES:
            raise ValueError("quality must be one of %s" % (", ".join(QUALITIES),))
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError("bbox must have four values")
            if not all(math.isfinite(float(v)) for v in self.bbox):
                raise ValueError("bbox must be four finite numbers")
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "path": self.path,
            "content_hash": self.content_hash,
            "size": int(self.size),
            "version": self.version,
            "schema": self.schema,
            "provider": self.provider,
            "source": self.source,
            "fetched_at": self.fetched_at,
            "as_of": self.as_of,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "license": self.license,
            "quality": self.quality,
            "pinned": bool(self.pinned),
            "notes": dict(self.notes or {}),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceEntry":
        bbox = data.get("bbox")
        return cls(
            entry_id=str(data["entry_id"]),
            kind=str(data["kind"]),
            path=str(data["path"]),
            content_hash=str(data["content_hash"]),
            size=int(data["size"]),
            version=str(data.get("version") or ""),
            schema=str(data.get("schema") or ""),
            provider=str(data.get("provider") or ""),
            source=str(data.get("source") or ""),
            fetched_at=str(data.get("fetched_at") or ""),
            as_of=str(data.get("as_of") or ""),
            bbox=tuple(float(v) for v in _bbox_list(bbox)) if bbox is not None else None,
            license=str(data.get("license") or ""),
            quality=str(data.get("quality") or "measured"),
            pinned=bool(data.get("pinned") or False),
            notes=dict(data.get("notes") or {}),
        )


@dataclass(frozen=True)
class ResourceRequirement:
    """What one run needs, before a concrete entry is chosen."""

    kind: str
    criticality: str
    reason: str = ""
    scope: Dict[str, Any] = field(default_factory=dict)
    allow_stale: bool = False
    multi: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "criticality": self.criticality,
            "reason": self.reason,
            "scope": dict(self.scope or {}),
            "allow_stale": bool(self.allow_stale),
            "multi": bool(self.multi),
        }


@dataclass(frozen=True)
class ResourceGap:
    """A requirement the current plan cannot satisfy."""

    kind: str
    criticality: str
    impact: str = ""
    reason: str = ""
    recommendation: Dict[str, Any] = field(default_factory=dict)
    candidates: Tuple[Dict[str, Any], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "criticality": self.criticality,
            "impact": self.impact,
            "reason": self.reason,
            "recommendation": dict(self.recommendation or {}),
            "candidates": [dict(c) for c in self.candidates],
        }


@dataclass(frozen=True)
class ResourceDecision:
    """One user/operator answer to a gap (used by the P2 preflight gate)."""

    action: str
    entry_id: str = ""
    note: str = ""
    decided_by: str = ""
    decided_at: str = ""
    accept_stale: bool = False

    def validate(self) -> "ResourceDecision":
        if self.action not in DECISION_ACTIONS:
            raise ValueError("action must be one of %s"
                             % (", ".join(DECISION_ACTIONS),))
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "entry_id": self.entry_id,
            "note": self.note,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at or utc_now(),
            "accept_stale": bool(self.accept_stale),
        }


@dataclass(frozen=True)
class ResourcePlan:
    """A pinned set of entries for one run (P2 consumes it; P0 defines it)."""

    plan_id: str
    scenario_date: str
    entries: Tuple[ResourceEntry, ...] = ()
    gaps: Tuple[ResourceGap, ...] = ()
    decisions: Tuple[ResourceDecision, ...] = ()
    status: str = "ready"

    def payload(self) -> Dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "scenario_date": self.scenario_date,
            "status": self.status,
            "entries": [
                {"entry_id": e.entry_id, "kind": e.kind,
                 "content_hash": e.content_hash, "as_of": e.as_of}
                for e in sorted(self.entries, key=lambda item: item.entry_id)
            ],
            "gaps": [g.as_dict() for g in self.gaps],
            "decisions": [d.as_dict() for d in self.decisions],
        }

    def hash(self) -> str:
        return sha256_bytes(canonical_json(self.payload()).encode("utf-8"))

    def as_dict(self) -> Dict[str, Any]:
        payload = self.payload()
        payload["plan_hash"] = self.hash()
        payload["entries"] = [e.as_dict() for e in self.entries]
        return payload
