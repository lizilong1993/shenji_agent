"""Offline normalizers for USGS / GDACS / FIRMS payloads."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from .model import Observation, iso, parse_time
from . import trust


def _utc_now(value: Any) -> datetime:
    return parse_time(value) if value is not None else datetime.now(timezone.utc)


def _point(geometry: Dict[str, Any]):
    """Return (lon, lat) for Point/MultiPoint geometry, else None."""
    geometry = geometry or {}
    coords = geometry.get("coordinates") or []
    gtype = geometry.get("type") or "Point"
    if gtype == "MultiPoint" and coords:
        coords = coords[0]
    if len(coords) < 2:
        return None
    return float(coords[0]), float(coords[1])


def _url_value(value: Any) -> str:
    """GDACS publishes url as a dict of report/detail links."""
    if isinstance(value, dict):
        for key in ("report", "details", "geometry"):
            if value.get(key):
                return str(value[key])
        return ""
    return str(value or "")


def _as_float(value: Any):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def normalize_usgs(payload: Dict[str, Any], fetched_at: Any,
                   start_seq: int = 0) -> List[Observation]:
    now = _utc_now(fetched_at)
    out = []
    for i, feature in enumerate(payload.get("features") or []):
        try:
            props = feature.get("properties") or {}
            point = _point(feature.get("geometry") or {})
            if point is None or props.get("time") is None:
                continue
            event = datetime.fromtimestamp(float(props["time"]) / 1000.0,
                                           tz=timezone.utc)
            native = str(feature.get("id") or props.get("code") or i)
            event_type = str(props.get("type") or "earthquake").lower()
            event_type = event_type.replace(" ", "_")
            coords = (feature.get("geometry") or {}).get("coordinates") or []
            depth = coords[2] if len(coords) > 2 else None
            conf = trust.confidence("usgs", event, now, "hazard")
            out.append(Observation(
                observation_id="usgs:" + native,
                source_id="usgs",
                source_native_id=native,
                domain="hazard",
                type=event_type,
                lat=point[1],
                lon=point[0],
                event_time=iso(event),
                ingest_time=iso(now),
                ingest_seq=start_seq + len(out),
                confidence=conf,
                source_url=_url_value(props.get("url")),
                title=str(props.get("place") or "earthquake"),
                properties={
                    "magnitude": props.get("mag"),
                    "depth_km": depth,
                },
            ))
        except (TypeError, ValueError, KeyError, AttributeError,
                IndexError, OverflowError, OSError):
            continue
    return out


def normalize_gdacs(payload: Dict[str, Any], fetched_at: Any,
                    start_seq: int = 0) -> List[Observation]:
    now = _utc_now(fetched_at)
    out = []
    for i, feature in enumerate(payload.get("features") or []):
        try:
            props = feature.get("properties") or {}
            point = _point(feature.get("geometry") or {})
            if point is None:
                continue
            from_raw = (props.get("fromdate") or props.get("date")
                        or props.get("todate"))
            event = parse_time(from_raw or now)
            to_raw = props.get("todate") or props.get("datemodified")
            event_end = parse_time(to_raw) if to_raw else event
            updated_raw = (props.get("datemodified") or props.get("todate")
                           or props.get("fromdate"))
            updated = parse_time(updated_raw) if updated_raw else event_end
            native = str(props.get("eventid") or props.get("id") or i)
            event_type = str(props.get("eventtype") or "disaster").lower()
            conf = trust.confidence("gdacs", updated, now, "hazard")
            out.append(Observation(
                observation_id="gdacs:" + native,
                source_id="gdacs",
                source_native_id=native,
                domain="hazard",
                type=event_type,
                lat=point[1],
                lon=point[0],
                event_time=iso(event),
                ingest_time=iso(now),
                ingest_seq=start_seq + len(out),
                confidence=conf,
                source_url=_url_value(props.get("url")),
                title=str(props.get("eventname") or props.get("name")
                          or props.get("description") or event_type),
                properties={
                    "alert_level": props.get("alertlevel"),
                    "country": props.get("country"),
                    "from_time": iso(event),
                    "to_time": iso(event_end),
                },
            ))
        except (TypeError, ValueError, KeyError, AttributeError,
                IndexError, OverflowError, OSError):
            continue
    return out


def normalize_firms(payload: Iterable[Dict[str, Any]], fetched_at: Any,
                    start_seq: int = 0) -> List[Observation]:
    now = _utc_now(fetched_at)
    out = []
    for i, item in enumerate(payload or []):
        try:
            lat = _as_float(item.get("latitude", item.get("lat")))
            lon = _as_float(item.get("longitude", item.get("lon")))
            if lat is None or lon is None:
                continue
            event = parse_time(item.get("acq_time") or item.get("acq_date")
                               or now)
            satellite = str(item.get("satellite") or "")
            native = str(item.get("id") or "%s_%s_%s_%s" % (
                satellite, lat, lon, iso(event)))
            brightness = item.get(
                "brightness", item.get("bright_ti4", item.get("bright_ti5")))
            conf = trust.confidence("firms", event, now, "hazard")
            out.append(Observation(
                observation_id="firms:" + native,
                source_id="firms",
                source_native_id=native,
                domain="hazard",
                type="fire",
                lat=lat,
                lon=lon,
                event_time=iso(event),
                ingest_time=iso(now),
                ingest_seq=start_seq + len(out),
                confidence=conf,
                source_url=_url_value(item.get("url")),
                title="thermal anomaly",
                properties={
                    "brightness": brightness,
                    "bright_ti4": item.get("bright_ti4"),
                    "bright_ti5": item.get("bright_ti5"),
                    "frp": item.get("frp"),
                    "satellite": item.get("satellite") or None,
                },
            ))
        except (TypeError, ValueError, KeyError, AttributeError,
                IndexError, OverflowError, OSError):
            continue
    return out


def normalize(source: str, payload: Any, fetched_at: Any,
              start_seq: int = 0) -> List[Observation]:
    source = source.lower()
    if source == "usgs":
        return normalize_usgs(payload, fetched_at, start_seq)
    if source == "gdacs":
        return normalize_gdacs(payload, fetched_at, start_seq)
    if source == "firms":
        return normalize_firms(payload, fetched_at, start_seq)
    raise ValueError("unsupported source %r" % source)
