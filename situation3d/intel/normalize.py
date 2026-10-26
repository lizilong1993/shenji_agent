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


def normalize_adsb(payload: Dict[str, Any], fetched_at: Any,
                   start_seq: int = 0) -> List[Observation]:
    """Normalize an adsb.lol /v2 snapshot; each aircraft is one observation."""
    now = _utc_now(fetched_at)
    aircraft = payload.get("ac") if isinstance(payload, dict) else payload
    out = []
    for item in aircraft or []:
        try:
            hex_code = str(item.get("hex") or "").strip().lower()
            lat = _as_float(item.get("lat"))
            lon = _as_float(item.get("lon"))
            if not hex_code or lat is None or lon is None:
                continue
            callsign = str(item.get("flight") or "").strip()
            registration = str(item.get("r") or "").strip()
            db_flags = int(item.get("dbFlags") or 0)
            conf = trust.confidence("adsb", now, now, "aviation")
            out.append(Observation(
                observation_id="adsb:" + hex_code,
                source_id="adsb",
                source_native_id=hex_code,
                domain="aviation",
                type="aircraft",
                lat=lat,
                lon=lon,
                event_time=iso(now),
                ingest_time=iso(now),
                ingest_seq=start_seq + len(out),
                confidence=conf,
                source_url="https://globe.adsb.lol/?icao=%s" % hex_code,
                title=callsign or registration or hex_code,
                properties={
                    "icao24": hex_code,
                    "callsign": callsign or None,
                    "registration": registration or None,
                    "aircraft_type": item.get("t"),
                    "altitude_baro_ft": item.get("alt_baro"),
                    "altitude_geom_ft": item.get("alt_geom"),
                    "ground_speed_kt": item.get("gs"),
                    "track_deg": item.get("track"),
                    "squawk": item.get("squawk"),
                    "category": item.get("category"),
                    "military": bool(db_flags & 1),
                    "seen_pos_s": item.get("seen_pos"),
                    "snapshot": True,
                },
            ))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
    return out


def normalize_eonet(payload: Dict[str, Any], fetched_at: Any,
                    start_seq: int = 0, bbox=None,
                    start: Any = None, end: Any = None) -> List[Observation]:
    """Normalize NASA EONET events; keep the latest point in bbox/window."""
    now = _utc_now(fetched_at)
    start_dt = parse_time(start) if start is not None else None
    end_dt = parse_time(end) if end is not None else None
    box = None
    if bbox is not None:
        west, south, east, north = (float(value) for value in bbox)
        box = (west, south, east, north)
    out = []
    for i, event in enumerate(payload.get("events") or []):
        try:
            selected = None
            for geometry in event.get("geometry") or []:
                if (geometry.get("type") or "Point") != "Point":
                    continue
                coords = geometry.get("coordinates") or []
                if len(coords) < 2:
                    continue
                lon, lat = float(coords[0]), float(coords[1])
                stamp = parse_time(geometry.get("date") or now)
                if start_dt is not None and stamp < start_dt:
                    continue
                if end_dt is not None and stamp > end_dt:
                    continue
                if box is not None and not (
                        box[0] <= lon <= box[2] and box[1] <= lat <= box[3]):
                    continue
                selected = (lat, lon, stamp, geometry)
            if selected is None:
                continue
            lat, lon, stamp, geometry = selected
            categories = event.get("categories") or []
            category = categories[0] if categories else {}
            event_type = str(category.get("id") or "natural_event").lower()
            sources = event.get("sources") or []
            source_url = str((sources[0] or {}).get("url") or event.get("link") or "")
            native = str(event.get("id") or i)
            conf = trust.confidence("eonet", stamp, now, "hazard")
            out.append(Observation(
                observation_id="eonet:" + native,
                source_id="eonet",
                source_native_id=native,
                domain="hazard",
                type=event_type,
                lat=lat,
                lon=lon,
                event_time=iso(stamp),
                ingest_time=iso(now),
                ingest_seq=start_seq + len(out),
                confidence=conf,
                source_url=source_url,
                title=str(event.get("title") or event_type),
                properties={
                    "category": category.get("title"),
                    "closed": event.get("closed"),
                    "description": event.get("description"),
                    "magnitude_value": geometry.get("magnitudeValue"),
                    "magnitude_unit": geometry.get("magnitudeUnit"),
                    "event_link": event.get("link"),
                    "geometry_count": len(event.get("geometry") or []),
                },
            ))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
    return out


def _gdelt_time(value: Any, fallback: datetime) -> datetime:
    text = str(value or "").strip()
    try:
        return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc)
    except ValueError:
        try:
            return parse_time(text)
        except ValueError:
            return fallback


def normalize_gdelt(payload: Dict[str, Any], fetched_at: Any, centroid,
                    start_seq: int = 0) -> List[Observation]:
    """Normalize GDELT DOC articles.

    GDELT DOC has no coordinates.  Articles are placed at the case centroid
    with ``geo_precision="unknown"`` and a low confidence; callers must not
    present them as located events.
    """
    now = _utc_now(fetched_at)
    if centroid is None:
        raise ValueError("GDELT normalization needs a case centroid")
    lat, lon = float(centroid[1]), float(centroid[0])
    out = []
    for i, article in enumerate(payload.get("articles") or []):
        try:
            title = str(article.get("title") or "").strip()
            url = str(article.get("url") or "").strip()
            if not title and not url:
                continue
            event = _gdelt_time(article.get("seendate"), now)
            native = url or "%s_%s" % (title, iso(event))
            conf = trust.confidence("gdelt", event, now, "other",
                                    geo_precision="unknown", is_centroid=True)
            out.append(Observation(
                observation_id="gdelt:" + native,
                source_id="gdelt",
                source_native_id=native,
                domain="other",
                type="news",
                lat=lat,
                lon=lon,
                event_time=iso(event),
                ingest_time=iso(now),
                ingest_seq=start_seq + len(out),
                confidence=conf,
                source_url=url,
                title=title or url,
                geo_precision="unknown",
                geo_uncertainty_m=50000.0,
                is_centroid=True,
                properties={
                    "article_domain": article.get("domain"),
                    "sourcecountry": article.get("sourcecountry"),
                    "language": article.get("language"),
                    "seendate": article.get("seendate"),
                    "location_basis": "case_centroid",
                },
            ))
        except (TypeError, ValueError, KeyError, AttributeError):
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
    if source == "adsb":
        return normalize_adsb(payload, fetched_at, start_seq)
    if source == "eonet":
        return normalize_eonet(payload, fetched_at, start_seq)
    raise ValueError("unsupported source %r" % source)
