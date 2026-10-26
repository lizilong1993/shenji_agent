"""Build an IntelBundle from live public sources (P0 external runtime).

Network access happens here, never inside ``sim``.  A fetch either produces an
immutable bundle (plus raw archive records) or reports a per-source failure; it
does not modify the world.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..gateway import IntelBundle, SOURCE_PRIORITY
from ..model import Observation, iso, parse_time
from . import firms, gdacs, usgs
from .archive import ArchiveRecord, archive_raw
from .http import FetchError, MissingCredential, fetch_bytes
from .registry import get_source

Fetcher = Callable[[str], bytes]


@dataclass(frozen=True)
class SourceStatus:
    source_id: str
    state: str
    ok: bool
    fetched_at: str
    count: int
    bytes: int = 0
    error: str = ""
    archives: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "state": self.state,
            "ok": self.ok,
            "fetched_at": self.fetched_at,
            "count": self.count,
            "bytes": self.bytes,
            "error": self.error,
            "archives": list(self.archives),
        }


@dataclass(frozen=True)
class FetchResult:
    bundle: IntelBundle
    statuses: Tuple[SourceStatus, ...]
    archives: Tuple[ArchiveRecord, ...]
    errors: Tuple[dict, ...]


def _normalize_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if isinstance(bbox, str):
        raise ValueError("bbox must be a numeric sequence, not a string")
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
    west, south, east, north = (float(v) for v in bbox)
    if west > east or south > north:
        raise ValueError("bbox min values must not exceed max values")
    if not -180.0 <= west <= 180.0 or not -180.0 <= east <= 180.0:
        raise ValueError("bbox longitude out of range")
    if not -90.0 <= south <= 90.0 or not -90.0 <= north <= 90.0:
        raise ValueError("bbox latitude out of range")
    return west, south, east, north


def _in_bbox(obs: Observation, bbox: Tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return west <= obs.lon <= east and south <= obs.lat <= north


def _in_window(obs: Observation, start: datetime, end: datetime) -> bool:
    if obs.source_id == "gdacs":
        from_time = obs.properties.get("from_time")
        to_time = obs.properties.get("to_time")
        if from_time and to_time:
            return (parse_time(from_time) <= end
                    and parse_time(to_time) >= start)
    event = obs.event_dt()
    return start <= event <= end


def _clamp_live_event(obs: Observation, start: datetime) -> Observation:
    """Make an already-active event cue-able at the bundle's scenario start."""
    if obs.event_dt() >= start:
        return obs
    properties = dict(obs.properties)
    properties.setdefault("event_start_time", obs.event_time)
    return dataclasses.replace(
        obs,
        event_time=iso(start),
        properties=properties,
    )


def _fetch(url: str, host: str, fetcher: Optional[Fetcher]) -> bytes:
    if fetcher is not None:
        return fetcher(url)
    return fetch_bytes(url, [host])


def _decode_json(raw: bytes) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FetchError("invalid JSON from source: %s" % (exc,)) from None
    if not isinstance(payload, dict):
        raise FetchError("source payload is not a JSON object")
    return payload


def _fetch_usgs(spec, start, end, bbox, fetcher, fetched_at):
    url = usgs.build_url(start, end, bbox)
    raw = _fetch(url, spec.host, fetcher)
    payload = _decode_json(raw)
    return [(spec.source_id, raw, "json")], usgs.normalize_payload(
        payload, fetched_at, start_seq=0)


def _fetch_gdacs(spec, start, end, bbox, fetcher, fetched_at):
    url = gdacs.build_url(start, end, bbox)
    raw = _fetch(url, spec.host, fetcher)
    payload = _decode_json(raw)
    return [(spec.source_id, raw, "json")], gdacs.normalize_payload(
        payload, fetched_at, start_seq=0)


def _fetch_firms(spec, start, end, bbox, fetcher, map_key, fetched_at):
    if not map_key:
        raise MissingCredential("%s is not set" % (spec.requires_env,))
    payloads = []
    observations = []
    errors = []
    start_utc = parse_time(start).astimezone(timezone.utc)
    end_utc = parse_time(end).astimezone(timezone.utc)
    span_days = (end_utc.date() - start_utc.date()).days + 1
    if span_days > 5:
        errors.append("window exceeds FIRMS 5-day limit; only 5 days requested")
    for sensor in firms.SENSORS:
        url = firms.build_url(map_key, sensor, bbox, start, end)
        try:
            raw = _fetch(url, spec.host, fetcher)
        except Exception as exc:  # one sensor may be unavailable; keep others
            errors.append("%s: %s" % (sensor, exc))
            continue
        text = raw.decode("utf-8-sig", "replace")
        try:
            rows = firms.parse_csv(text)
        except ValueError as exc:
            errors.append("%s: %s" % (sensor, exc))
            continue
        payloads.append(("firms_" + sensor, raw, "csv"))
        observations.extend(firms.normalize_payload(
            rows, fetched_at, start_seq=0))
    if not payloads:
        raise FetchError("; ".join(errors) or "no FIRMS sensor returned data")
    return payloads, observations, errors


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text[:300]


def build_bundle(source_ids: Sequence[str], bbox: Sequence[float],
                 start: Any, end: Any, fetched_at: Any = None,
                 bundle_id: Optional[str] = None,
                 scenario_id: str = "vgi_live",
                 archive_root: Optional[str] = None,
                 fetcher: Optional[Fetcher] = None,
                 firms_map_key: Optional[str] = None) -> FetchResult:
    """Fetch selected sources, archive raw bytes, and return one bundle.

    ``fetcher`` is injected for offline tests; production calls ``http``.
    """
    if isinstance(source_ids, str):
        raise ValueError("source_ids must be a sequence, not a string")
    sources = tuple(dict.fromkeys(
        str(item).strip().lower() for item in source_ids if str(item).strip()))
    if not sources:
        raise ValueError("at least one source is required")
    for source_id in sources:
        try:
            get_source(source_id)
        except KeyError:
            raise ValueError("unsupported source %r" % (source_id,))
    bbox_tuple = _normalize_bbox(bbox)
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    if end_dt < start_dt:
        raise ValueError("end must not be before start")
    fetched_dt = parse_time(fetched_at) if fetched_at is not None else datetime.now(timezone.utc)
    fetched_text = iso(fetched_dt)
    key = firms_map_key if firms_map_key is not None else os.environ.get("FIRMS_MAP_KEY")

    observations: List[Observation] = []
    statuses: List[SourceStatus] = []
    archives: List[ArchiveRecord] = []
    errors: List[dict] = []

    for source_id in sources:
        spec = get_source(source_id)
        try:
            sensor_errors = []
            if source_id == "usgs":
                payloads, found = _fetch_usgs(
                    spec, start_dt, end_dt, bbox_tuple, fetcher, fetched_dt)
            elif source_id == "gdacs":
                payloads, found = _fetch_gdacs(
                    spec, start_dt, end_dt, bbox_tuple, fetcher, fetched_dt)
            elif source_id == "firms":
                payloads, found, sensor_errors = _fetch_firms(
                    spec, start_dt, end_dt, bbox_tuple, fetcher, key,
                    fetched_dt)
            else:  # pragma: no cover - registry already rejects this
                raise FetchError("unsupported source %s" % (source_id,))

            kept = [obs for obs in found
                    if _in_bbox(obs, bbox_tuple)
                    and _in_window(obs, start_dt, end_dt)]
            kept = [
                _clamp_live_event(obs, start_dt) if obs.source_id == "gdacs"
                else obs
                for obs in kept
            ]
            kept = [
                dataclasses.replace(
                    obs,
                    license=spec.license,
                    attribution=spec.attribution,
                    parser_version=spec.parser_version,
                )
                for obs in kept
            ]

            records = []
            if archive_root:
                for label, raw, ext in payloads:
                    try:
                        records.append(archive_raw(
                            archive_root, label, raw, fetched_dt, ext=ext))
                    except Exception as exc:
                        sensor_errors.append("archive: %s" % (_safe_error(exc),))
            observations.extend(kept)
            archives.extend(records)
            state = "degraded" if sensor_errors else "nominal"
            statuses.append(SourceStatus(
                source_id=source_id,
                state=state,
                ok=True,
                fetched_at=fetched_text,
                count=len(kept),
                bytes=sum(len(raw) for _, raw, _ in payloads),
                error="; ".join(sensor_errors)[:300],
                archives=tuple(record.path for record in records),
            ))
        except Exception as exc:
            message = _safe_error(exc)
            statuses.append(SourceStatus(
                source_id=source_id,
                state="unavailable",
                ok=False,
                fetched_at=fetched_text,
                count=0,
                error=message,
            ))
            errors.append({"source_id": source_id, "error": message})

    unique: Dict[str, Observation] = {}
    for obs in observations:
        current = unique.get(obs.observation_id)
        if current is None or (
                (float(obs.confidence), obs.content_hash())
                > (float(current.confidence), current.content_hash())):
            unique[obs.observation_id] = obs
    observations = list(unique.values())
    observations.sort(key=lambda obs: (
        obs.event_ms(), SOURCE_PRIORITY.get(obs.source_id, 9),
        obs.observation_id))
    observations = [
        dataclasses.replace(obs, ingest_seq=index)
        for index, obs in enumerate(observations)
    ]
    counts = Counter(obs.source_id for obs in observations)
    statuses = [
        dataclasses.replace(status, count=counts.get(status.source_id, 0))
        for status in statuses
    ]
    bundle = IntelBundle(
        bundle_id=bundle_id or ("vgi-" + fetched_dt.strftime("%Y%m%dT%H%M%SZ")),
        scenario_id=scenario_id,
        observations=observations,
        scenario_start=iso(start_dt),
        synthetic=False,
        domain="hazard",
        bbox=bbox_tuple,
        time_window=(iso(start_dt), iso(end_dt)),
        source_allowlist=sources,
    )
    return FetchResult(
        bundle=bundle,
        statuses=tuple(statuses),
        archives=tuple(archives),
        errors=tuple(errors),
    )
