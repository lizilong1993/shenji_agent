"""Persistent resource index (P0).

The index lives in ``situation3d/var/resources/index.json`` (gitignored) and
references artifacts *in place*; it never copies large payloads.  Every write
is atomic and serialised by an advisory file lock so the test server and the
CLI can share one index safely.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # Linux server; Windows dev falls back to no cross-process lock.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from .catalog import kind_spec
from .model import (INDEX_SCHEMA, ResourceEntry, canonical_json, hash_path,
                    sha256_bytes, utc_now)


class ResourceError(Exception):
    """Base error for resource-store operations."""


class ResourceNotFound(ResourceError):
    pass


class ResourceConflict(ResourceError):
    pass


def default_root() -> str:
    env_root = os.environ.get("SITUATION_RESOURCES_DIR")
    if env_root:
        return env_root
    return str(Path(__file__).resolve().parent.parent / "var" / "resources")


class ResourceStore(object):
    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or default_root())
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / ".lock"

    # -- locking / persistence ---------------------------------------------

    @contextlib.contextmanager
    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "a+") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load(self) -> Dict[str, object]:
        if not self.index_path.is_file():
            return {"schema": INDEX_SCHEMA, "updated_at": "", "entries": {}}
        try:
            with open(self.index_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ResourceError("resource index unreadable: %s" % exc)
        if not isinstance(data, dict) or data.get("schema") != INDEX_SCHEMA:
            raise ResourceError("resource index has unsupported schema")
        if not isinstance(data.get("entries"), dict):
            raise ResourceError("resource index entries must be an object")
        return data

    def _save(self, data: Dict[str, object]) -> None:
        data["updated_at"] = utc_now()
        payload = json.dumps(data, ensure_ascii=False, indent=1,
                             sort_keys=True).encode("utf-8")
        fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, str(self.index_path))
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # -- queries -----------------------------------------------------------

    def entries(self, kind: Optional[str] = None) -> List[ResourceEntry]:
        raw = self._load()["entries"]
        out = [ResourceEntry.from_dict(item) for item in raw.values()]
        if kind:
            kind_spec(kind)
            out = [e for e in out if e.kind == kind]
        return sorted(out, key=lambda e: e.entry_id)

    def get(self, entry_id: str) -> ResourceEntry:
        raw = self._load()["entries"]
        item = raw.get(str(entry_id))
        if item is None:
            raise ResourceNotFound("resource entry not found: %s" % entry_id)
        return ResourceEntry.from_dict(item)

    def index_hash(self) -> str:
        raw = self._load()["entries"]
        payload = canonical_json({k: raw[k] for k in sorted(raw)})
        return sha256_bytes(payload.encode("utf-8"))

    # -- mutations ---------------------------------------------------------

    def register(self, entry: ResourceEntry, replace: bool = False) -> ResourceEntry:
        entry.validate()
        kind_spec(entry.kind)
        with self._locked():
            data = self._load()
            entries = data["entries"]
            old_item = entries.get(entry.entry_id)
            if old_item is not None:
                old = ResourceEntry.from_dict(old_item)
                if old.content_hash == entry.content_hash:
                    merged = self._merge(old, entry)
                elif not replace:
                    raise ResourceConflict(
                        "resource %s already registered with a different "
                        "content hash (%s != %s); pass replace=True"
                        % (entry.entry_id, old.content_hash[:12],
                           entry.content_hash[:12]))
                else:
                    merged = ResourceEntry(**dict(
                        entry.as_dict(), bbox=entry.bbox,
                        fetched_at=entry.fetched_at or utc_now(),
                        pinned=old.pinned or entry.pinned))
                entries[entry.entry_id] = merged.as_dict()
            else:
                if not entry.fetched_at:
                    entry = ResourceEntry(**dict(entry.as_dict(), bbox=entry.bbox,
                                                 fetched_at=utc_now()))
                entries[entry.entry_id] = entry.as_dict()
            self._save(data)
        return self.get(entry.entry_id)

    @staticmethod
    def _merge(old: ResourceEntry, new: ResourceEntry) -> ResourceEntry:
        notes = dict(old.notes)
        notes.update(new.notes)
        if old.notes.get("manual_override") \
                and not new.notes.get("manual_override"):
            notes.pop("manual_override", None)
        return ResourceEntry(
            entry_id=old.entry_id,
            kind=new.kind,
            path=new.path,
            content_hash=old.content_hash,
            size=old.size,
            version=new.version or old.version,
            schema=new.schema or old.schema,
            provider=new.provider or old.provider,
            source=new.source or old.source,
            fetched_at=new.fetched_at or old.fetched_at,
            as_of=new.as_of or old.as_of,
            bbox=new.bbox if new.bbox is not None else old.bbox,
            license=new.license or old.license,
            quality=new.quality or old.quality,
            pinned=old.pinned or new.pinned,
            notes=notes,
        )

    def set_pin(self, entry_id: str, pinned: bool = True) -> ResourceEntry:
        with self._locked():
            data = self._load()
            item = data["entries"].get(str(entry_id))
            if item is None:
                raise ResourceNotFound("resource entry not found: %s" % entry_id)
            item["pinned"] = bool(pinned)
            self._save(data)
        return self.get(entry_id)

    def remove(self, entry_id: str, force: bool = False) -> None:
        with self._locked():
            data = self._load()
            item = data["entries"].get(str(entry_id))
            if item is None:
                raise ResourceNotFound("resource entry not found: %s" % entry_id)
            if item.get("pinned") and not force:
                raise ResourceConflict(
                    "resource %s is pinned; unpin it first" % entry_id)
            del data["entries"][str(entry_id)]
            self._save(data)

    # -- verification ------------------------------------------------------

    def verify(self, entry_id: str) -> Dict[str, object]:
        entry = self.get(entry_id)
        if not os.path.exists(entry.path):
            return {"entry_id": entry.entry_id, "ok": False,
                    "status": "missing", "path": entry.path,
                    "detail": "path does not exist"}
        try:
            digest, size = hash_path(entry.path)
        except OSError as exc:
            return {"entry_id": entry.entry_id, "ok": False,
                    "status": "unreadable_incomplete", "path": entry.path,
                    "detail": str(exc)}
        if size != entry.size:
            return {"entry_id": entry.entry_id, "ok": False,
                    "status": "size_mismatch", "path": entry.path,
                    "expected_size": entry.size, "actual_size": size,
                    "detail": "size differs from the registered entry"}
        if digest != entry.content_hash:
            return {"entry_id": entry.entry_id, "ok": False,
                    "status": "hash_mismatch", "path": entry.path,
                    "expected_hash": entry.content_hash,
                    "actual_hash": digest,
                    "detail": "content changed after registration"}
        return {"entry_id": entry.entry_id, "ok": True, "status": "ok",
                "path": entry.path}

    def verify_all(self, kind: Optional[str] = None) -> Dict[str, object]:
        results = [self.verify(e.entry_id) for e in self.entries(kind)]
        return {
            "ok": all(item["ok"] for item in results),
            "checked": len(results),
            "results": results,
        }
