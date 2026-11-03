"""Incremental per-source intel collection backed by the rolling cache.

Each refresh asks every source only for the window after its stored cursor
(plus a small overlap for late-published records), writes the result into the
7-day cache, and finally assembles the case bundle from the cache.  Upstream
outages therefore no longer erase the case view: the cached observations still
serve the window.
"""

from __future__ import annotations

import dataclasses
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cache import IntelCache, RETENTION_DAYS, cache_db_path
from .case import CaseIntelProfile, expand_bbox
from .connectors.fetch import (SourceStatus, _in_bbox, _in_window,
                               build_bundle)
from .gateway import IntelBundle, SOURCE_PRIORITY
from .model import iso, parse_time

# Sources that only expose a live snapshot.  A historical window that ends
# before ``now`` is not fetchable upstream; it is served from the cache only.
LIVE_SOURCES = ("adsb", "open-meteo")
LIVE_TOLERANCE_S = 600.0
# Overlap re-requested before the stored cursor: late-published records are
# the norm for USGS revisions, GDACS status updates and multi-day events.
OVERLAP_H = {"usgs": 1.0, "gdacs": 24.0, "eonet": 24.0,
             "firms": 24.0, "gdelt": 1.0}
DEFAULT_OVERLAP_H = 1.0
FIRMS_MAX_DAYS = 5


@dataclass(frozen=True)
class RefreshResult:
    bundle: IntelBundle
    statuses: tuple
    archives: tuple
    errors: tuple
    warnings: tuple
    truncated: bool
    cache_report: dict


def _overlap(source_id: str) -> timedelta:
    return timedelta(hours=OVERLAP_H.get(source_id, DEFAULT_OVERLAP_H))


def _utc_midnight(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0)


def fetch_windows(source_id: str, start: datetime, end: datetime,
                  cursor_until: Optional[str], now: datetime,
                  live_allowed: bool) -> List[Tuple[datetime, datetime]]:
    """Windows the source still needs, oldest first; ``[]`` means cache-only."""
    if source_id in LIVE_SOURCES:
        if not live_allowed:
            return []
        if end < now - timedelta(seconds=LIVE_TOLERANCE_S):
            # Injected fetchers (tests/offline runs) replay the request as-is.
            return [(start, end)]
        return [(max(start, now - timedelta(seconds=LIVE_TOLERANCE_S)), end)]
    window_start = start
    if cursor_until:
        try:
            cursor = parse_time(cursor_until)
        except ValueError:
            cursor = None
        if cursor is not None:
            window_start = max(start, cursor - _overlap(source_id))
            if window_start >= end:
                return []
    if source_id == "firms":
        # FIRMS serves whole UTC calendar days (day_span is a day count from
        # the URL's date), so chunk boundaries must be UTC midnights. Cutting
        # at "start + 5 days" instead loses the tail after day 5's midnight.
        chunks: List[Tuple[datetime, datetime]] = []
        current = window_start
        while current < end:
            nxt = min(end, _utc_midnight(current)
                      + timedelta(days=FIRMS_MAX_DAYS))
            if nxt <= current:  # defensive: never loop without progress
                nxt = min(end, current + timedelta(days=1))
            chunks.append((current, nxt))
            current = nxt
        return chunks
    return [(window_start, end)]


def _merge_statuses(statuses: Sequence[SourceStatus]) -> List[SourceStatus]:
    """Collapse one status per source across chunked fetches (e.g. FIRMS)."""
    merged: Dict[str, SourceStatus] = {}
    order: List[str] = []
    for status in statuses:
        current = merged.get(status.source_id)
        if current is None:
            merged[status.source_id] = status
            order.append(status.source_id)
            continue
        ok = current.ok and status.ok
        if not ok:
            state = "unavailable"
        elif "degraded" in (current.state, status.state):
            state = "degraded"
        else:
            state = status.state
        merged[status.source_id] = dataclasses.replace(
            status,
            ok=ok,
            state=state,
            bytes=current.bytes + status.bytes,
            error="; ".join(value for value in (current.error, status.error)
                            if value)[:300],
            archives=tuple(current.archives) + tuple(status.archives),
        )
    return [merged[source_id] for source_id in order]


def refresh_bundle(case_id: str, profile: CaseIntelProfile, start: Any,
                   end: Any, sources: Sequence[str], root: str,
                   cache: Optional[IntelCache] = None,
                   custom_sources: Optional[Sequence[dict]] = None,
                   archive_root: Optional[str] = None,
                   fetcher=None,
                   firms_map_key: Optional[str] = None,
                   fetched_at: Any = None) -> RefreshResult:
    """Fetch deltas for every source, cache them, and build the case bundle."""
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    now_dt = (parse_time(fetched_at) if fetched_at is not None
              else datetime.now(timezone.utc))
    own_cache = cache is None
    cache = cache if cache is not None else IntelCache(cache_db_path(root))
    bbox = tuple(float(value) for value in profile.bbox)
    expanded = expand_bbox(bbox, profile.buffer_km)
    live_allowed = (fetcher is not None
                    or end_dt >= now_dt - timedelta(seconds=LIVE_TOLERANCE_S))
    custom = {str(entry.get("source_id")): entry
              for entry in (custom_sources or [])
              if entry.get("source_id")}

    statuses: List[SourceStatus] = []
    archives: List[Any] = []
    errors: List[dict] = []
    warnings: List[dict] = []
    windows_log: List[dict] = []
    fetched_count = 0
    try:
        for source_id in sources:
            # GDELT queries depend on the case keywords, so its cursor and
            # cached rows are namespaced per case. The buffer also belongs in
            # the cursor key: growing it exposes an unfetched ring.
            scope = case_id if source_id == "gdelt" else ""
            variant = "buf=%.3f" % float(profile.buffer_km)
            if scope:
                variant += ";case=" + scope
            state = cache.state(source_id, bbox, variant=variant)
            windows = fetch_windows(source_id, start_dt, end_dt,
                                    state.get("cursor_until"), now_dt,
                                    live_allowed)
            entry = custom.get(source_id)
            source_custom = [entry] if entry is not None else None
            source_ok = True
            if not windows:
                # A cursor past the window normally means "cache is complete".
                # If pruning already removed that window, fall back to a full
                # fetch instead of serving an empty bundle forever.
                probe = [obs for obs in cache.query(
                    start_dt, end_dt, (source_id,), scope=scope or None)
                    if _in_bbox(obs, expanded)
                    and _in_window(obs, start_dt, end_dt)]
                if probe or (source_id in LIVE_SOURCES and not live_allowed):
                    windows_log.append({
                        "source_id": source_id, "mode": "cached",
                        "cursor_until": state.get("cursor_until"),
                    })
                    statuses.append(SourceStatus(
                        source_id=source_id, state="cached", ok=True,
                        fetched_at=state.get("last_fetch_at") or "", count=0,
                        error=state.get("last_error") or ""))
                    continue
                windows = [(start_dt, end_dt)]
                windows_log.append({
                    "source_id": source_id, "mode": "cache-miss",
                    "cursor_until": state.get("cursor_until"),
                })
            source_statuses: List[SourceStatus] = []
            for window_start, window_end in windows:
                try:
                    result = build_bundle(
                        source_ids=(source_id,),
                        bbox=bbox,
                        start=window_start,
                        end=window_end,
                        fetched_at=now_dt,
                        scenario_id=case_id,
                        case_profile=profile,
                        custom_sources=source_custom,
                        archive_root=archive_root,
                        fetcher=fetcher,
                        firms_map_key=firms_map_key,
                        filter_observations=False,
                    )
                except Exception as exc:  # invalid args or unexpected errors
                    message = "%s: %s" % (type(exc).__name__, exc)
                    errors.append({"source_id": source_id,
                                   "error": message[:300]})
                    cache.mark(source_id, bbox, error=message,
                               fetched_at=iso(now_dt), variant=variant)
                    windows_log.append({
                        "source_id": source_id, "mode": "error",
                        "start": iso(window_start), "end": iso(window_end),
                        "error": message[:300],
                    })
                    source_ok = False
                    break
                status = result.statuses[0] if result.statuses else None
                if status is not None and not status.ok:
                    cache.mark(source_id, bbox,
                               error=status.error or "source unavailable",
                               fetched_at=iso(now_dt), variant=variant)
                    source_ok = False
                else:
                    fetched_count += cache.upsert(
                        result.bundle.observations, scope=scope)
                if status is not None:
                    source_statuses.append(status)
                archives.extend(result.archives)
                errors.extend(result.errors)
                warnings.extend(result.warnings)
                windows_log.append({
                    "source_id": source_id, "mode": "fetched",
                    "start": iso(window_start), "end": iso(window_end),
                    "ok": bool(status is not None and status.ok),
                    "observations": len(result.bundle.observations),
                })
                if not source_ok:
                    break
            degraded = any(status.state == "degraded"
                           for status in source_statuses)
            if source_ok and not degraded:
                cache.mark(source_id, bbox, cursor_until=iso(end_dt),
                           error=None, fetched_at=iso(now_dt),
                           variant=variant)
            elif source_ok:
                # Partial result (e.g. one FIRMS sensor down): keep the old
                # cursor so the gap is requested again on the next run.
                cache.mark(source_id, bbox, error="degraded fetch",
                           fetched_at=iso(now_dt), variant=variant)
            statuses.extend(_merge_statuses(source_statuses))

        # -- assemble the case bundle from the cache --------------------
        cached = cache.query(start_dt, end_dt, sources,
                             scope=case_id if "gdelt" in sources else None)
        cached = [obs for obs in cached
                  if _in_bbox(obs, expanded)
                  and _in_window(obs, start_dt, end_dt)]
        unique: Dict[str, Any] = {}
        for obs in cached:
            current = unique.get(obs.observation_id)
            if current is None or (
                    (float(obs.confidence), obs.content_hash())
                    > (float(current.confidence), current.content_hash())):
                unique[obs.observation_id] = obs
        observations = sorted(unique.values(), key=lambda obs: (
            obs.event_ms(), SOURCE_PRIORITY.get(obs.source_id, 9),
            obs.observation_id))
        truncated = False
        if len(observations) > profile.max_items:
            observations = observations[-profile.max_items:]
            truncated = True
        observations = [dataclasses.replace(obs, ingest_seq=index)
                        for index, obs in enumerate(observations)]
        counts = Counter(obs.source_id for obs in observations)
        statuses = [dataclasses.replace(status,
                                        count=counts.get(status.source_id, 0))
                    for status in statuses]
        bundle = IntelBundle(
            bundle_id="vgi-%s-%s" % (
                case_id, now_dt.strftime("%Y%m%dT%H%M%SZ")),
            scenario_id=case_id,
            observations=observations,
            scenario_start=iso(start_dt),
            synthetic=False,
            domain="multi",
            bbox=expanded,
            time_window=(iso(start_dt), iso(end_dt)),
            source_allowlist=tuple(sources),
        )
        # Retention is anchored at min(window end, now): a future end must not
        # prune recent data. Historical windows keep their own horizon.
        prune_anchor = min(end_dt, now_dt)
        pruned = cache.prune(prune_anchor - timedelta(days=RETENTION_DAYS))
        report = {
            "enabled": True,
            "db": os.path.basename(cache.path),
            "retention_days": RETENTION_DAYS,
            "fetched_observations": fetched_count,
            "served_observations": len(observations),
            "pruned": pruned,
            "windows": windows_log,
        }
        return RefreshResult(
            bundle=bundle, statuses=tuple(statuses),
            archives=tuple(archives), errors=tuple(errors),
            warnings=tuple(warnings), truncated=truncated,
            cache_report=report)
    finally:
        if own_cache:
            cache.close()
