"""Persistent resource registry backed by SQLite (P0, upgraded 2026-10-27).

The registry lives in ``situation3d/var/resources/resources.db``.  It stores
only metadata and points at artifacts *in place*; it never copies payloads.
An existing ``index.json`` is imported once on first open, then kept as a
migration backup and no longer written.

SQLite is used instead of a server database because the CLI and the local
server share one directory on the same host.  Writes take an advisory file
lock in addition to SQLite's own transaction lock: the resource root lives on
NFS in production, where cross-host writer serialisation is worth the extra
lock.  Readers do not take the advisory lock; SQLite's journal makes each
transaction atomic.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Linux server; Windows dev falls back to no cross-process lock.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from .catalog import KINDS, kind_spec
from .model import (INDEX_SCHEMA, ResourceEntry, canonical_json, hash_path,
                    sha256_bytes, slugify, utc_now)


class ResourceError(Exception):
    """Base error for resource-store operations."""


class ResourceNotFound(ResourceError):
    pass


class ResourceConflict(ResourceError):
    pass


DB_SCHEMA_VERSION = 1
_DB_NAME = "resources.db"
_READY_DB_PATHS = set()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS resource_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resource_kinds (
    kind          TEXT PRIMARY KEY,
    description   TEXT NOT NULL DEFAULT '',
    criticality   TEXT NOT NULL DEFAULT 'advisory',
    ttl_days      REAL,
    schema        TEXT NOT NULL DEFAULT '',
    providers_json TEXT NOT NULL DEFAULT '[]',
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resources (
    entry_id      TEXT PRIMARY KEY,
    kind          TEXT NOT NULL REFERENCES resource_kinds(kind),
    path          TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    size          INTEGER NOT NULL CHECK (size >= 0),
    version       TEXT NOT NULL DEFAULT '',
    schema        TEXT NOT NULL DEFAULT '',
    provider      TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT '',
    fetched_at    TEXT NOT NULL DEFAULT '',
    as_of         TEXT NOT NULL DEFAULT '',
    bbox_west     REAL,
    bbox_south    REAL,
    bbox_east     REAL,
    bbox_north    REAL,
    license       TEXT NOT NULL DEFAULT '',
    quality       TEXT NOT NULL DEFAULT 'measured',
    pinned        INTEGER NOT NULL DEFAULT 0,
    notes_json    TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS resources_kind_idx ON resources(kind);
CREATE INDEX IF NOT EXISTS resources_asof_idx ON resources(as_of);
"""

_COLUMNS = (
    "entry_id", "kind", "path", "content_hash", "size", "version", "schema",
    "provider", "source", "fetched_at", "as_of", "bbox_west", "bbox_south",
    "bbox_east", "bbox_north", "license", "quality", "pinned", "notes_json",
    "created_at", "updated_at",
)

_UPDATABLE = {
    "path", "version", "schema", "provider", "source", "fetched_at", "as_of",
    "bbox", "license", "quality", "pinned", "notes",
}


def default_root() -> str:
    env_root = os.environ.get("SITUATION_RESOURCES_DIR")
    if env_root:
        return env_root
    return str(Path(__file__).resolve().parent.parent / "var" / "resources")


def _bbox_tuple(row: sqlite3.Row) -> Optional[tuple]:
    values = (row["bbox_west"], row["bbox_south"],
              row["bbox_east"], row["bbox_north"])
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)


def _row_to_entry(row: sqlite3.Row) -> ResourceEntry:
    try:
        notes = json.loads(row["notes_json"] or "{}")
    except (TypeError, ValueError):
        notes = {}
    if not isinstance(notes, dict):
        notes = {}
    return ResourceEntry(
        entry_id=row["entry_id"],
        kind=row["kind"],
        path=row["path"],
        content_hash=row["content_hash"],
        size=int(row["size"]),
        version=row["version"] or "",
        schema=row["schema"] or "",
        provider=row["provider"] or "",
        source=row["source"] or "",
        fetched_at=row["fetched_at"] or "",
        as_of=row["as_of"] or "",
        bbox=_bbox_tuple(row),
        license=row["license"] or "",
        quality=row["quality"] or "measured",
        pinned=bool(row["pinned"]),
        notes=notes,
    )


def _entry_values(entry: ResourceEntry, created_at: str) -> tuple:
    bbox = entry.bbox or (None, None, None, None)
    return (
        entry.entry_id, entry.kind, entry.path, entry.content_hash,
        int(entry.size), entry.version, entry.schema, entry.provider,
        entry.source, entry.fetched_at, entry.as_of,
        float(bbox[0]) if bbox[0] is not None else None,
        float(bbox[1]) if bbox[1] is not None else None,
        float(bbox[2]) if bbox[2] is not None else None,
        float(bbox[3]) if bbox[3] is not None else None,
        entry.license, entry.quality, 1 if entry.pinned else 0,
        json.dumps(entry.notes or {}, ensure_ascii=False, sort_keys=True),
        created_at, utc_now(),
    )


class ResourceStore(object):
    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or default_root())
        self.index_path = self.root / "index.json"
        self.db_path = self.root / _DB_NAME
        self.lock_path = self.root / ".lock"
        self._ready = False

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

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=15,
                               isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("PRAGMA synchronous = FULL")
        return conn

    def _ensure_schema(self) -> None:
        # Initialisation is once per database path per process.  A fresh
        # ResourceStore object is created for every HTTP request, so an
        # instance-level flag would take the advisory write lock (and run
        # DDL) on every read.
        key = str(self.db_path)
        if key in _READY_DB_PATHS:
            return
        with self._locked():
            if key in _READY_DB_PATHS:
                return
            conn = self._connect()
            try:
                # SQLite's default DELETE journal is the safest choice on NFS:
                # a WAL file needs shared-memory locking that NFS does not
                # reliably provide. Do not re-set the journal mode on every
                # request - it is persistent and can take a write lock.
                conn.executescript(_SCHEMA_SQL)
                # Kinds are code-owned; upsert once per process so a newly
                # added KINDS entry satisfies the foreign key on restart.
                self._seed_kinds(conn)
                self._migrate_index(conn)
                row = conn.execute(
                    "SELECT value FROM resource_meta WHERE key='schema_version'"
                ).fetchone()
                if row is None or row["value"] != str(DB_SCHEMA_VERSION):
                    conn.execute(
                        "INSERT INTO resource_meta(key, value) "
                        "VALUES('schema_version', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (str(DB_SCHEMA_VERSION),))
            finally:
                conn.close()
            _READY_DB_PATHS.add(key)

    @staticmethod
    def _seed_kinds(conn: sqlite3.Connection) -> None:
        now = utc_now()
        for spec in KINDS.values():
            conn.execute(
                "INSERT INTO resource_kinds"
                "(kind, description, criticality, ttl_days, schema, "
                " providers_json, updated_at) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(kind) DO UPDATE SET "
                " description=excluded.description, "
                " criticality=excluded.criticality, "
                " ttl_days=excluded.ttl_days, "
                " schema=excluded.schema, "
                " providers_json=excluded.providers_json, "
                " updated_at=excluded.updated_at",
                (spec.kind, spec.description, spec.criticality, spec.ttl_days,
                 spec.schema, json.dumps(list(spec.providers), ensure_ascii=False),
                 now))

    def _migrate_index(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT value FROM resource_meta WHERE key='migrated_from_index'"
        ).fetchone()
        if row is not None or not self.index_path.is_file():
            return
        try:
            with open(self.index_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            warnings.warn("resource index not migrated: %s" % exc)
            return
        if not isinstance(data, dict) or data.get("schema") != INDEX_SCHEMA:
            warnings.warn("resource index not migrated: unsupported schema")
            return
        entries = data.get("entries")
        if not isinstance(entries, dict):
            warnings.warn("resource index not migrated: entries is not an object")
            return
        skipped = 0
        for raw in entries.values():
            try:
                entry = ResourceEntry.from_dict(raw)
                entry.validate()
                kind_spec(entry.kind)
                created = str(raw.get("fetched_at") or utc_now())
                conn.execute(
                    "INSERT OR IGNORE INTO resources(" + ",".join(_COLUMNS) +
                    ") VALUES(" + ",".join("?" for _ in _COLUMNS) + ")",
                    _entry_values(entry, created))
            except Exception as exc:  # noqa: BLE001 - one bad row must not brick the store
                skipped += 1
                warnings.warn("resource index entry skipped: %s" % exc)
        conn.execute(
            "INSERT OR REPLACE INTO resource_meta(key, value) "
            "VALUES('migrated_from_index', ?)", (utc_now(),))
        if skipped:
            warnings.warn("resource index migrated with %d skipped entries"
                          % skipped)

    # -- queries -----------------------------------------------------------

    def entries(self, kind: Optional[str] = None) -> List[ResourceEntry]:
        self._ensure_schema()
        if kind:
            kind_spec(kind)
        conn = self._connect()
        try:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM resources WHERE kind=? ORDER BY entry_id",
                    (kind,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM resources ORDER BY entry_id").fetchall()
            return [_row_to_entry(row) for row in rows]
        finally:
            conn.close()

    def get(self, entry_id: str) -> ResourceEntry:
        self._ensure_schema()
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM resources WHERE entry_id=?",
                               (str(entry_id),)).fetchone()
            if row is None:
                raise ResourceNotFound("resource entry not found: %s" % entry_id)
            return _row_to_entry(row)
        finally:
            conn.close()

    def kinds(self) -> List[Dict[str, Any]]:
        self._ensure_schema()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM resource_kinds ORDER BY kind").fetchall()
        finally:
            conn.close()
        out = []
        for row in rows:
            try:
                providers = json.loads(row["providers_json"] or "[]")
            except (TypeError, ValueError):
                providers = []
            out.append({
                "kind": row["kind"],
                "description": row["description"],
                "criticality": row["criticality"],
                "ttl_days": row["ttl_days"],
                "schema": row["schema"],
                "providers": providers,
            })
        return out

    def stats(self) -> Dict[str, Dict[str, int]]:
        self._ensure_schema()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT kind, COUNT(*) AS count, COALESCE(SUM(size),0) AS bytes, "
                "COALESCE(SUM(pinned),0) AS pinned FROM resources GROUP BY kind"
            ).fetchall()
        finally:
            conn.close()
        return {row["kind"]: {"count": int(row["count"]),
                              "bytes": int(row["bytes"]),
                              "pinned": int(row["pinned"])}
                for row in rows}

    def index_hash(self) -> str:
        raw = {entry.entry_id: entry.as_dict()
               for entry in self.entries()}
        payload = canonical_json({key: raw[key] for key in sorted(raw)})
        return sha256_bytes(payload.encode("utf-8"))

    # -- mutations ---------------------------------------------------------

    def register(self, entry: ResourceEntry, replace: bool = False) -> ResourceEntry:
        entry.validate()
        kind_spec(entry.kind)
        self._ensure_schema()
        with self._locked():
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM resources WHERE entry_id=?",
                    (entry.entry_id,)).fetchone()
                if row is not None:
                    old = _row_to_entry(row)
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
                    self._update_row(conn, merged)
                else:
                    if not entry.fetched_at:
                        entry = ResourceEntry(**dict(
                            entry.as_dict(), bbox=entry.bbox,
                            fetched_at=utc_now()))
                    values = _entry_values(entry, utc_now())
                    conn.execute(
                        "INSERT INTO resources(" + ",".join(_COLUMNS) +
                        ") VALUES(" + ",".join("?" for _ in _COLUMNS) + ")",
                        values)
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return self.get(entry.entry_id)

    @staticmethod
    def _update_row(conn: sqlite3.Connection, entry: ResourceEntry) -> None:
        bbox = entry.bbox or (None, None, None, None)
        updates = {
            "kind": entry.kind,
            "path": entry.path,
            "content_hash": entry.content_hash,
            "size": int(entry.size),
            "version": entry.version,
            "schema": entry.schema,
            "provider": entry.provider,
            "source": entry.source,
            "fetched_at": entry.fetched_at,
            "as_of": entry.as_of,
            "bbox_west": float(bbox[0]) if bbox[0] is not None else None,
            "bbox_south": float(bbox[1]) if bbox[1] is not None else None,
            "bbox_east": float(bbox[2]) if bbox[2] is not None else None,
            "bbox_north": float(bbox[3]) if bbox[3] is not None else None,
            "license": entry.license,
            "quality": entry.quality,
            "pinned": 1 if entry.pinned else 0,
            "notes_json": json.dumps(entry.notes or {}, ensure_ascii=False,
                                     sort_keys=True),
            "updated_at": utc_now(),
        }
        assignments = ",".join("%s=?" % key for key in updates)
        conn.execute("UPDATE resources SET " + assignments +
                     " WHERE entry_id=?",
                     tuple(updates.values()) + (entry.entry_id,))

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

    def create_entry(self, kind: str, name: str, path: str,
                     **metadata: Any) -> ResourceEntry:
        """Register a new on-disk artifact, computing hash and size.

        ``name`` is only used to derive an entry id when the caller does not
        supply one explicitly.  The path must already exist; directory
        resources (envpacks, tilesets) are registered in place rather than
        copied.
        """
        kind_spec(kind)
        if not isinstance(path, str) or not os.path.isabs(path):
            raise ValueError("path must be an absolute path")
        if not os.path.exists(path):
            raise ValueError("path does not exist: %s" % path)
        replace = bool(metadata.pop("replace", False))
        entry_id = metadata.pop("entry_id", None)
        if not entry_id:
            slug = slugify(name)
            if slug == "unnamed":
                slug = "manual-%s-%s" % (utc_now().replace(":", "").replace("-", "")[:14],
                                          uuid.uuid4().hex[:6])
            entry_id = "%s:%s" % (kind, slug)
        digest, size = hash_path(path)
        data = {
            "entry_id": str(entry_id),
            "kind": kind,
            "path": os.path.abspath(path),
            "content_hash": digest,
            "size": size,
        }
        data.update(metadata)
        entry = ResourceEntry.from_dict(data)
        entry.validate()
        return self.register(entry, replace=replace)

    def update(self, entry_id: str, changes: Dict[str, Any]) -> ResourceEntry:
        """Update editable metadata; re-hash the artifact if the path changed."""
        if not isinstance(changes, dict):
            raise ValueError("changes must be an object")
        unknown = sorted(set(changes) - _UPDATABLE - {"entry_id", "kind"})
        if unknown:
            raise ValueError("unsupported fields: %s" % ", ".join(unknown))
        old_pre = self.get(entry_id)
        if "kind" in changes and changes["kind"] != old_pre.kind:
            raise ValueError("kind cannot be changed")
        if "entry_id" in changes and changes["entry_id"] != entry_id:
            raise ValueError("entry_id cannot be changed")
        # Hash outside the write lock: an envpack or tileset can take seconds
        # to walk, and read requests must not wait on that.
        prepared_hash = None
        if "path" in changes:
            new_path = changes["path"]
            if not isinstance(new_path, str) or not os.path.isabs(new_path):
                raise ValueError("path must be an absolute path")
            if os.path.abspath(new_path) != os.path.abspath(old_pre.path):
                if not os.path.exists(new_path):
                    raise ValueError("path does not exist: %s" % new_path)
                try:
                    prepared_hash = hash_path(new_path)
                except OSError as exc:
                    raise ValueError("path unreadable: %s" % exc)
        self._ensure_schema()
        with self._locked():
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM resources WHERE entry_id=?",
                    (str(entry_id),)).fetchone()
                if row is None:
                    raise ResourceNotFound(
                        "resource entry not found: %s" % entry_id)
                old = _row_to_entry(row)
                data = old.as_dict()
                for key in _UPDATABLE:
                    if key in changes:
                        data[key] = changes[key]
                if "notes" in changes and not isinstance(data.get("notes"), dict):
                    raise ValueError("notes must be an object")
                new_path = data.get("path")
                if not isinstance(new_path, str) or not os.path.isabs(new_path):
                    raise ValueError("path must be an absolute path")
                if os.path.abspath(new_path) != os.path.abspath(old.path):
                    data["path"] = os.path.abspath(new_path)
                    if (prepared_hash is not None
                            and os.path.abspath(old.path)
                            == os.path.abspath(old_pre.path)):
                        data["content_hash"], data["size"] = prepared_hash
                    else:
                        try:
                            data["content_hash"], data["size"] = hash_path(
                                data["path"])
                        except OSError as exc:
                            raise ValueError("path unreadable: %s" % exc)
                entry = ResourceEntry.from_dict(data)
                entry.validate()
                self._update_row(conn, entry)
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return self.get(entry_id)

    def set_pin(self, entry_id: str, pinned: bool = True) -> ResourceEntry:
        self._ensure_schema()
        with self._locked():
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM resources WHERE entry_id=?",
                    (str(entry_id),)).fetchone()
                if row is None:
                    raise ResourceNotFound(
                        "resource entry not found: %s" % entry_id)
                conn.execute(
                    "UPDATE resources SET pinned=?, updated_at=? WHERE entry_id=?",
                    (1 if pinned else 0, utc_now(), str(entry_id)))
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return self.get(entry_id)

    def remove(self, entry_id: str, force: bool = False) -> None:
        """Remove the database record; the artifact on disk is left untouched."""
        self._ensure_schema()
        with self._locked():
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT pinned FROM resources WHERE entry_id=?",
                    (str(entry_id),)).fetchone()
                if row is None:
                    raise ResourceNotFound(
                        "resource entry not found: %s" % entry_id)
                if row["pinned"] and not force:
                    raise ResourceConflict(
                        "resource %s is pinned; unpin it first" % entry_id)
                conn.execute("DELETE FROM resources WHERE entry_id=?",
                             (str(entry_id),))
                conn.execute("COMMIT")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

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
