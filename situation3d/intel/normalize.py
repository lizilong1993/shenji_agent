"""Offline normalizers for USGS / GDACS / FIRMS fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from .model import Observation, iso, parse_time
from . import trust


def _utc_now(value: Any) -> datetime:
    return parse_time(value) if value is not None else datetime.now(timezone.utc)


def _point(geometry: Dict[str, Any]):
    coords = (geometry or {}).get("coordinates") or []
    if len(coords) < 2:
        return None
    return float(coords[0]), float(coords[1])


def normalize_usgs(payload: Dict[str, Any], fetched_at: Any,
                   start_seq: int = 0) -> List[Observation]:
    now = _utc_now(fetched_at)
    out = []
    for i, feature in enumerate(payload.get("features") or []):
        props = feature.get("properties") or {}
        point = _point(feature.get("geometry") or {})
        if point is None or props.get("time") is None:
            continue
        event = datetime.fromtimestamp(float(props["time"]) / 1000.0,
                                       tz=timezone.utc)
        native = str(feature.get("id") or props.get("code") or i)
        conf = trust.confidence("usgs", event, now, "hazard")
        out.append(Observation(
            observation_id="usgs:" + native,
            source_id="usgs",
            source_native_id=native,
            domain="hazard",
            type="earthquake",
            lat=point[1],
            lon=point[0],
            event_time=iso(event),
            ingest_time=iso(now),
            ingest_seq=start_seq + i,
            confidence=conf,
            source_url=str(props.get("url") or ""),
            title=str(props.get("place") or "earthquake"),
            properties={
                "magnitude": props.get("mag"),
                "depth_km": (feature.get("geometry") or {}).get("coordinates", [None, None, None])[2],
            },
        ))
    return out


def normalize_gdacs(payload: Dict[str, Any], fetched_at: Any,
                    start_seq: int = 0) -> List[Observation]:
    now = _utc_now(fetched_at)
    out = []
    for i, feature in enumerate(payload.get("features") or []):
        props = feature.get("properties") or {}
        point = _point(feature.get("geometry") or {})
        if point is None:
            continue
        event = parse_time(props.get("fromdate") or props.get("date")
                           or props.get("todate") or now)
        native = str(props.get("eventid") or props.get("id") or i)
        event_type = str(props.get("eventtype") or "disaster").lower()
        conf = trust.confidence("gdacs", event, now, "hazard")
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
            ingest_seq=start_seq + i,
            confidence=conf,
            source_url=str(props.get("url") or ""),
            title=str(props.get("name") or props.get("description") or event_type),
            properties={
                "alert_level": props.get("alertlevel"),
                "country": props.get("country"),
            },
        ))
    return out


def normalize_firms(payload: Iterable[Dict[str, Any]], fetched_at: Any,
                    start_seq: int = 0) -> List[Observation]:
    now = _utc_now(fetched_at)
    out = []
    for i, item in enumerate(payload or []):
        lat = item.get("latitude", item.get("lat"))
        lon = item.get("longitude", item.get("lon"))
        if lat is None or lon is None:
            continue
        event = parse_time(item.get("acq_time") or item.get("acq_date")
                           or now)
        native = str(item.get("id") or "%s_%s" % (lat, lon))
        conf = trust.confidence("firms", event, now, "hazard")
        out.append(Observation(
            observation_id="firms:" + native,
            source_id="firms",
            source_native_id=native,
            domain="hazard",
            type="fire",
            lat=float(lat),
            lon=float(lon),
            event_time=iso(event),
            ingest_time=iso(now),
            ingest_seq=start_seq + i,
            confidence=conf,
            source_url=str(item.get("url") or ""),
            title="thermal anomaly",
            properties={
                "brightness": item.get("brightness"),
                "frp": item.get("frp"),
                "satellite": item.get("satellite"),
            },
        ))
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
