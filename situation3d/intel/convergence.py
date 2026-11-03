"""Grid-cell convergence events across intelligence types (task F).

A *convergence* is the smallest unit that reads as an event rather than a
link: several different kinds of signal landing in one grid cell inside a
short window.  ``association.link_items`` already answers "which two items are
related"; this module answers "where is something building up".

Deterministic and explainable by construction: every event carries its member
ids and one reason line per member, so an analyst can trace it back to the
observations.  No new third-party dependency.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .association import canonical_type, _distance_m, _time

DEFAULT_CELL_DEG = 1.0
DEFAULT_WINDOW_H = 24.0
DEFAULT_MIN_TYPES = 3
# A cell smaller than this on either axis is widened to it: at 0.05 deg the
# nominal cell is ~5 km, below the 25-50 km radius the link layer uses, so
# signals that "converge" for association would never share a cell.
MIN_CELL_DEG = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_size_for(bbox: Sequence[float], divisions: int = 16) -> float:
    """Adaptive cell edge in degrees: bbox divided into ``divisions``."""
    try:
        west, south, east, north = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return DEFAULT_CELL_DEG
    span = max(abs(east - west), abs(north - south))
    if span <= 0.0:
        return DEFAULT_CELL_DEG
    return max(MIN_CELL_DEG, round(span / float(max(1, divisions)), 6))


def cell_of(lat: float, lon: float, size: float) -> Tuple[int, int]:
    """Stable integer cell index (floor, so cells never overlap)."""
    size = max(MIN_CELL_DEG, float(size))
    return (int(math.floor(lat / size)), int(math.floor(lon / size)))


def _cell_bounds(key: Tuple[int, int], size: float) -> Dict[str, float]:
    row, col = key
    return {
        "west": round(col * size, 6),
        "south": round(row * size, 6),
        "east": round((col + 1) * size, 6),
        "north": round((row + 1) * size, 6),
        "center_lon": round((col + 0.5) * size, 6),
        "center_lat": round((row + 0.5) * size, 6),
    }


def _event_id(key: Tuple[int, int], start: datetime, types: Sequence[str],
              size: float) -> str:
    payload = "|".join([
        "%.6f" % size, str(key[0]), str(key[1]),
        start.strftime("%Y-%m-%dT%H:%M:%S"), ",".join(sorted(types)),
    ])
    return "conv:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _active(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        if item.get("status") == "revoked":
            continue
        if item.get("effective") is False:
            continue
        lat, lon = _num(item.get("lat")), _num(item.get("lon"))
        stamp = _time(item.get("event_time"))
        if lat is None or lon is None or stamp is None:
            continue
        out.append(dict(item, _lat=lat, _lon=lon, _ts=stamp))
    out.sort(key=lambda row: (row["_ts"], str(row.get("id"))))
    return out


def _score(members: Sequence[Dict[str, Any]], types: Sequence[str],
           window_h: float) -> Tuple[float, Dict[str, float]]:
    """Diversity-dominant score; confidence and clustering modulate it."""
    strengths = []
    for member in members:
        value = _num(member.get("confidence"))
        strengths.append(0.5 if value is None else max(0.0, min(1.0, value)))
    confidence = sum(strengths) / float(len(strengths)) if strengths else 0.0
    diversity = min(1.0, len(types) / 4.0)
    span_s = (max(member["_ts"] for member in members)
              - min(member["_ts"] for member in members)).total_seconds()
    compactness = 1.0
    if window_h > 0:
        compactness = max(0.0, 1.0 - (span_s / 3600.0) / window_h)
    sources = {str(member.get("source_id") or "") for member in members}
    corroboration = min(1.0, len(sources) / 3.0)
    score = (0.40 * diversity + 0.25 * confidence
             + 0.20 * compactness + 0.15 * corroboration)
    parts = {
        "diversity": round(diversity, 3),
        "confidence": round(confidence, 3),
        "compactness": round(compactness, 3),
        "corroboration": round(corroboration, 3),
    }
    return round(min(0.99, score), 3), parts


def convergence_events(items: Sequence[Dict[str, Any]],
                       bbox: Optional[Sequence[float]] = None,
                       cell_deg: Optional[float] = None,
                       window_h: float = DEFAULT_WINDOW_H,
                       min_types: int = DEFAULT_MIN_TYPES,
                       max_events: int = 50) -> List[Dict[str, Any]]:
    """Group active items into cells and emit convergence events.

    ``items`` are the unified intel items ``IntelManager.overview`` returns
    (``id``/``lat``/``lon``/``event_time``/``type``/``domain``/``source_id``...).
    Items without usable coordinates or timestamps are ignored, as are revoked
    and superseded ones: a convergence built on data the analyst already
    overrode would be noise.
    """
    active = _active(items)
    if not active:
        return []
    size = float(cell_deg) if cell_deg else (
        cell_size_for(bbox) if bbox else DEFAULT_CELL_DEG)
    size = max(MIN_CELL_DEG, size)
    window = max(0.0, float(window_h))
    needed = max(2, int(min_types))

    cells: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for item in active:
        cells.setdefault(cell_of(item["_lat"], item["_lon"], size),
                         []).append(item)

    events: List[Dict[str, Any]] = []
    for key, members in cells.items():
        if len(members) < needed:
            continue
        # Sliding window over time-sorted members.  Tie handling is irrelevant
        # because members were sorted by (timestamp, id) once, up front.
        emitted: List[Tuple[int, int]] = []
        start = 0
        while start < len(members):
            end = start
            types = set()
            while end < len(members):
                elapsed = (members[end]["_ts"]
                           - members[start]["_ts"]).total_seconds() / 3600.0
                if elapsed > window:
                    break
                types.add(canonical_type(members[end].get("type")))
                end += 1
            if len(types) >= needed:
                # `start += 1`, not `start = end`: a window beginning inside
                # this one can still be the only cover for a later member
                # (A@0h B@23h C@24h D@25h: {A,B,C} and {B,C,D} are both
                # valid, and jumping to `end` would lose the second).
                # Sub-windows already covered by an emitted range are dropped
                # so one long burst still reads as one event.
                covered = any(begin <= start and finish >= end
                              for begin, finish in emitted)
                if not covered:
                    emitted.append((start, end))
                    events.append(_build(key, size, members[start:end],
                                         types, window))
            start += 1
    events.sort(key=lambda ev: (-ev["score"], ev["t_start"], ev["id"]))
    return events[:max(1, int(max_events))]


def _build(key: Tuple[int, int], size: float,
           members: Sequence[Dict[str, Any]], types: Sequence[str],
           window_h: float) -> Dict[str, Any]:
    start = min(member["_ts"] for member in members)
    end = max(member["_ts"] for member in members)
    score, parts = _score(members, types, window_h)
    bounds = _cell_bounds(key, size)
    center_lat, center_lon = bounds["center_lat"], bounds["center_lon"]
    reasons = ["同格 %d 类信号：%s" % (len(types), "、".join(sorted(types)))]
    member_rows = []
    for member in sorted(members, key=lambda row: (row["_ts"],
                                                   str(row.get("id")))):
        distance_km = _distance_m(center_lat, center_lon,
                                  member["_lat"], member["_lon"]) / 1000.0
        offset_h = (member["_ts"] - start).total_seconds() / 3600.0
        # Only the first few members get a reason line: a 40-member cell would
        # otherwise emit 40 lines nobody reads.
        if len(member_rows) < 3:
            reasons.append("%s（%s，距格中心 %.1f km，+%.1f h）"
                           % (member.get("title") or member.get("id"),
                              member.get("source_id") or "?",
                              distance_km, offset_h))
        member_rows.append({
            "id": member.get("id"),
            "source_id": member.get("source_id"),
            "source_name": member.get("source_name"),
            "type": member.get("type"),
            "domain": member.get("domain"),
            "title": member.get("title"),
            "event_time": member.get("event_time"),
            "lat": member["_lat"],
            "lon": member["_lon"],
            "distance_km": round(distance_km, 3),
            "offset_h": round(offset_h, 3),
        })
    source_ids = sorted({str(member.get("source_id") or "")
                         for member in members})
    return {
        "id": _event_id(key, start, sorted(types), size),
        "cell": {"row": key[0], "col": key[1], "cell_deg": size,
                 "center_lat": center_lat, "center_lon": center_lon,
                 "bounds": bounds},
        "t_start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "t_end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "types": sorted(types),
        "source_ids": source_ids,
        "member_count": len(members),
        "members": member_rows,
        "score": score,
        "score_parts": parts,
        "reasons": reasons,
    }


def summarize(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts for the overview payload (kept small: the UI shows one line)."""
    type_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    for event in events:
        for type_name in event.get("types") or []:
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        for source_id in event.get("source_ids") or []:
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
    return {
        "count": len(events),
        "by_type": dict(sorted(type_counts.items())),
        "by_source": dict(sorted(source_counts.items())),
        "generated_at": _now(),
    }
