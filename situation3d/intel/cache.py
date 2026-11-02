"""Rolling on-disk cache for normalized VGI observations (7-day retention).

The case pipeline used to re-fetch the full lookback window from every source
on each run.  ``IntelCache`` stores normalized observations keyed by
``observation_id`` and remembers, per (source, AOI), how far that source has
been fetched, so the next run only asks upstream for the delta.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from .model import Observation, iso, parse_time

RETENTION_DAYS = 7.0

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS observations (
        observation_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        event_ms INTEGER NOT NULL,
        fetched_at TEXT,
        record TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_obs_event ON observations(event_ms)",
    "CREATE INDEX IF NOT EXISTS idx_obs_source ON observations(source_id)",
    """
    CREATE TABLE IF NOT EXISTS source_state (
        state_key TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        bbox_key TEXT NOT NULL,
        cursor_until TEXT,
        last_fetch_at TEXT,
        last_error TEXT
    )
    """,
)


def cache_db_path(root: str) -> str:
    """SQLite path for the intel cache; ``SITUATION_INTEL_CACHE_DB`` overrides.

    The default lives next to the other intel stores.  Deployments that run
    the store on NFS should point the override at a local disk path: fsync and
    lock amplification on the shared server is a known operational hazard.
    """
    override = str(os.environ.get("SITUATION_INTEL_CACHE_DB") or "").strip()
    if override:
        return os.path.abspath(override)
    return os.path.join(os.path.abspath(root), "cache.db")


def bbox_key(bbox: Sequence[float]) -> str:
    return ",".join("%.3f" % float(value) for value in bbox)


def _observation_from_record(record: Dict[str, Any]) -> Observation:
    geometry = record.get("geometry") or {}
    coords = geometry.get("coordinates") or []
    lat = float(coords[1]) if len(coords) > 1 else 0.0
    lon = float(coords[0]) if coords else 0.0
    provenance = record.get("provenance") or {}
    try:
        confidence = float(record.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        ingest_seq = int(record.get("ingest_seq") or 0)
    except (TypeError, ValueError):
        ingest_seq = 0
    try:
        uncertainty = float(record.get("geo_uncertainty_m") or 0.0)
    except (TypeError, ValueError):
        uncertainty = 0.0
    return Observation(
        observation_id=str(record.get("observation_id") or ""),
        source_id=str(record.get("source_id") or "unknown"),
        source_native_id=str(record.get("source_native_id") or ""),
        domain=str(record.get("domain") or "other"),
        type=str(record.get("type") or "other"),
        lat=lat,
        lon=lon,
        event_time=str(record.get("event_time") or ""),
        ingest_time=str(record.get("ingest_time") or ""),
        ingest_seq=ingest_seq,
        confidence=confidence,
        source_url=str(record.get("source_url") or ""),
        title=str(record.get("title") or ""),
        geo_precision=str(record.get("geo_precision") or "exact"),
        geo_uncertainty_m=uncertainty,
        is_centroid=bool(record.get("is_centroid")),
        status=str(record.get("status") or "active"),
        properties=dict(record.get("properties") or {}),
        schema_version=str(record.get("schema_version")
                           or "situation3d.intel.v1"),
        license=str(record.get("license") or ""),
        attribution=str(record.get("attribution") or ""),
        parser_version=str(provenance.get("parser_version")
                           or "vgi-connectors-1"),
    )


class IntelCache:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                     timeout=10.0)
        # NFS-hostile by design: no WAL, bounded fsyncs.
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock, self._conn:
            for statement in _SCHEMA:
                self._conn.execute(statement)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- observations ---------------------------------------------------

    def upsert(self, observations: Sequence[Observation]) -> int:
        rows = [(obs.observation_id, obs.source_id, obs.event_ms(),
                 obs.ingest_time,
                 json.dumps(obs.as_dict(), ensure_ascii=False))
                for obs in observations]
        if not rows:
            return 0
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO observations"
                " (observation_id, source_id, event_ms, fetched_at, record)"
                " VALUES (?, ?, ?, ?, ?)", rows)
        return len(rows)

    def query(self, start: Any, end: Any,
              source_ids: Optional[Sequence[str]] = None) -> List[Observation]:
        start_ms = int(parse_time(start).timestamp() * 1000)
        end_ms = int(parse_time(end).timestamp() * 1000)
        sql = ("SELECT record FROM observations"
               " WHERE event_ms >= ? AND event_ms <= ?")
        args = [start_ms, end_ms]
        ids = tuple(str(item) for item in (source_ids or ()) if str(item))
        if ids:
            sql += " AND source_id IN (%s)" % ",".join("?" * len(ids))
            args.extend(ids)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_observation_from_record(json.loads(row[0])) for row in rows]

    def prune(self, cutoff: Any) -> int:
        cutoff_ms = int(parse_time(cutoff).timestamp() * 1000)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM observations WHERE event_ms < ?", (cutoff_ms,))
        return cursor.rowcount or 0

    # -- per-source cursor state ---------------------------------------

    def _state_key(self, source_id: str, bbox: Sequence[float]) -> str:
        return "%s:%s" % (source_id, bbox_key(bbox))

    def state(self, source_id: str,
              bbox: Sequence[float]) -> Dict[str, Any]:
        key = self._state_key(source_id, bbox)
        with self._lock:
            row = self._conn.execute(
                "SELECT cursor_until, last_fetch_at, last_error"
                " FROM source_state WHERE state_key = ?", (key,)).fetchone()
        if not row:
            return {}
        return {"cursor_until": row[0], "last_fetch_at": row[1],
                "last_error": row[2]}

    def mark(self, source_id: str, bbox: Sequence[float],
             cursor_until: Optional[str] = None,
             error: Optional[str] = None,
             fetched_at: Optional[str] = None) -> None:
        key = self._state_key(source_id, bbox)
        current = self.state(source_id, bbox)
        cursor = (cursor_until if cursor_until is not None
                  else current.get("cursor_until"))
        stamp = fetched_at or iso(datetime.now(timezone.utc))
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO source_state (state_key, source_id,"
                " bbox_key, cursor_until, last_fetch_at, last_error)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (key, source_id, bbox_key(bbox), cursor, stamp,
                 (error or "")[:300] or None))

    # -- reporting ------------------------------------------------------

    def stats(self, reference: Any = None) -> Dict[str, Any]:
        ref = (parse_time(reference) if reference is not None
               else datetime.now(timezone.utc))
        cutoff_ms = int((ref - timedelta(days=RETENTION_DAYS))
                        .timestamp() * 1000)
        with self._lock:
            counts = {row[0]: row[1] for row in self._conn.execute(
                "SELECT source_id, COUNT(*) FROM observations"
                " WHERE event_ms >= ? GROUP BY source_id", (cutoff_ms,))}
            states = self._conn.execute(
                "SELECT source_id, cursor_until, last_fetch_at, last_error"
                " FROM source_state").fetchall()
        sources: Dict[str, Dict[str, Any]] = {}
        for source_id, cursor, fetched, error in states:
            sources[source_id] = {
                "cursor_until": cursor,
                "last_fetch_at": fetched,
                "last_error": error or "",
                "observations": counts.get(source_id, 0),
            }
        for source_id, count in counts.items():
            sources.setdefault(source_id, {
                "cursor_until": None, "last_fetch_at": "",
                "last_error": "", "observations": count})
        return {
            "retention_days": RETENTION_DAYS,
            "db": os.path.basename(self.path),
            "observations_7d": sum(counts.values()),
            "sources": sources,
        }
