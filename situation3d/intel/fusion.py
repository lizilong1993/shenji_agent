"""Deterministic cross-source fusion of observations (P1 slice).

Multiple sources report the same earthquake with different ids and slightly
different coordinates.  This module groups observations by domain/type inside a
space-time window around a fixed cluster anchor (the highest-confidence
member), and fuses confidence with a noisy-OR.  It does not rewrite the source
observations.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .model import Observation, parse_time

TYPE_SYNONYMS = {
    "eq": "earthquake",
    "earthquake": "earthquake",
    "tc": "cyclone",
    "tropical_cyclone": "cyclone",
    "cyclone": "cyclone",
    "fl": "flood",
    "flood": "flood",
    "wf": "fire",
    "wildfire": "fire",
    "fire": "fire",
    "dr": "drought",
    "drought": "drought",
    "vo": "volcano",
    "volcano": "volcano",
    "ts": "tsunami",
    "tsunami": "tsunami",
    # NASA EONET category ids
    "severestorms": "cyclone",
    "wildfires": "fire",
    "volcanoes": "volcano",
    "floods": "flood",
    "earthquakes": "earthquake",
    "dusthaze": "dust",
    "sealakeice": "ice",
    "temperatureextremes": "extreme_temperature",
    "manmade": "other",
    "watercolor": "other",
}

SOURCE_PRIORITY = {"usgs": 0, "gdacs": 1, "firms": 2,
                   "eonet": 3, "adsb": 4, "gdelt": 5}


def _canonical_type(value: str) -> str:
    text = str(value or "other").strip().lower().replace(" ", "_")
    return TYPE_SYNONYMS.get(text, text)


def _validate(obs: Observation) -> None:
    try:
        lat = float(obs.lat)
        lon = float(obs.lon)
        conf = float(obs.confidence)
    except (TypeError, ValueError):
        raise ValueError("observation %s has non-numeric fields"
                         % obs.observation_id) from None
    if not (math.isfinite(lat) and math.isfinite(lon)
            and math.isfinite(conf)):
        raise ValueError("observation %s has non-finite fields"
                         % obs.observation_id)
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError("observation %s has out-of-range coordinates"
                         % obs.observation_id)


def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = (math.sin(dlat / 2.0) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))


@dataclass(frozen=True)
class FusedEvent:
    cluster_id: str
    domain: str
    type: str
    lat: float
    lon: float
    event_time: str
    confidence: float
    representative_id: str
    member_ids: Tuple[str, ...]
    sources: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "domain": self.domain,
            "type": self.type,
            "lat": self.lat,
            "lon": self.lon,
            "event_time": self.event_time,
            "confidence": self.confidence,
            "representative_id": self.representative_id,
            "member_ids": list(self.member_ids),
            "sources": list(self.sources),
        }


def _cluster_id(members: Sequence[Observation]) -> str:
    payload = "|".join(sorted(m.content_hash() for m in members))
    return "cluster:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _fused_confidence(members: Sequence[Observation]) -> float:
    remain = 1.0
    for member in members:
        remain *= 1.0 - max(0.0, min(1.0, float(member.confidence)))
    return round(min(0.999, 1.0 - remain), 3)


def _dedupe(observations: Sequence[Observation]) -> List[Observation]:
    """Keep one observation per id (higher confidence, then content hash)."""
    best: Dict[str, Observation] = {}
    for obs in observations:
        _validate(obs)
        current = best.get(obs.observation_id)
        if current is None or (
                (float(obs.confidence), obs.content_hash())
                > (float(current.confidence), current.content_hash())):
            best[obs.observation_id] = obs
    return list(best.values())


def fuse(observations: Sequence[Observation], radius_m: float = 10000.0,
         window_s: float = 3600.0) -> List[FusedEvent]:
    """Group observations into deterministic space-time clusters.

    Observations are processed by confidence (highest first); each cluster's
    anchor is its first member and never drifts, so every member is within the
    radius/window of the anchor that admitted it.
    """
    ordered = sorted(
        _dedupe(observations),
        key=lambda o: (-float(o.confidence),
                       SOURCE_PRIORITY.get(o.source_id, 9),
                       o.observation_id),
    )
    clusters: List[List[Observation]] = []
    for obs in ordered:
        best = None
        best_key = None
        for idx, members in enumerate(clusters):
            anchor = members[0]
            if obs.domain != anchor.domain:
                continue
            if _canonical_type(obs.type) != _canonical_type(anchor.type):
                continue
            dt = abs(obs.event_ms() - anchor.event_ms()) / 1000.0
            if dt > window_s:
                continue
            dist = _haversine_m((obs.lon, obs.lat), (anchor.lon, anchor.lat))
            if dist > radius_m:
                continue
            key = (round(dist, 3), round(dt, 3), idx)
            if best_key is None or key < best_key:
                best_key = key
                best = idx
        if best is None:
            clusters.append([obs])
        else:
            clusters[best].append(obs)

    events = []
    for members in clusters:
        anchor = members[0]
        events.append(FusedEvent(
            cluster_id=_cluster_id(members),
            domain=anchor.domain,
            type=_canonical_type(anchor.type),
            lat=anchor.lat,
            lon=anchor.lon,
            event_time=anchor.event_time,
            confidence=_fused_confidence(members),
            representative_id=anchor.observation_id,
            member_ids=tuple(sorted(m.observation_id for m in members)),
            sources=tuple(sorted({m.source_id for m in members})),
        ))
    events.sort(key=lambda e: (parse_time(e.event_time).timestamp(),
                               e.cluster_id))
    return events
