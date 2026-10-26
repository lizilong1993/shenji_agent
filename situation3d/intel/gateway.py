"""Intel bundle and future-only cue gateway (P1 slice)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .fusion import fuse
from .model import Observation, parse_time


@dataclass(frozen=True)
class IntelCue:
    cue_id: str
    observation_id: str
    ingest_seq: int
    event_time_ms: int
    knowledge_time_ms: int
    source_priority: int
    cue_kind: str
    sim_time: float
    valid_from: float
    valid_to: float
    side: int
    geometry: Dict[str, Any]
    domain: str
    type: str
    confidence: float
    ttl_s: float
    provenance: Dict[str, Any]
    sensor_model: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "observation_id": self.observation_id,
            "ingest_seq": self.ingest_seq,
            "event_time_ms": self.event_time_ms,
            "knowledge_time_ms": self.knowledge_time_ms,
            "source_priority": self.source_priority,
            "cue_kind": self.cue_kind,
            "sim_time": self.sim_time,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "side": self.side,
            "geometry": self.geometry,
            "domain": self.domain,
            "type": self.type,
            "confidence": self.confidence,
            "ttl_s": self.ttl_s,
            "provenance": self.provenance,
            "sensor_model": self.sensor_model,
        }


@dataclass(frozen=True)
class CueBatch:
    cues: Tuple[IntelCue, ...]
    watermark: float
    late_policy: str
    log_hash: str


SOURCE_PRIORITY = {"usgs": 0, "gdacs": 1, "firms": 2,
                   "eonet": 3, "adsb": 4, "gdelt": 5}


class IntelBundle:
    def __init__(self, bundle_id: str, scenario_id: str,
                 observations: Sequence[Observation], scenario_start: str,
                 synthetic: bool = True, domain: str = "multi",
                 bbox: Optional[Sequence[float]] = None,
                 time_window: Optional[Sequence[str]] = None,
                 source_allowlist: Optional[Sequence[str]] = None):
        self.bundle_id = bundle_id
        self.scenario_id = scenario_id
        self.scenario_start = scenario_start
        self.synthetic = bool(synthetic)
        self.domain = str(domain or "multi")
        self.observations = sorted(
            observations,
            key=lambda o: (o.event_ms(), SOURCE_PRIORITY.get(o.source_id, 9),
                           o.observation_id),
        )
        if bbox is None:
            bbox = self._default_bbox()
        if isinstance(bbox, str):
            raise ValueError("bbox must be a numeric sequence")
        self.bbox = tuple(float(value) for value in bbox)
        if len(self.bbox) != 4:
            raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
        if time_window is None:
            time_window = (scenario_start, self._last_event_time())
        self.time_window = tuple(str(value) for value in time_window)
        if len(self.time_window) != 2:
            raise ValueError("time_window must be (start, end)")
        if source_allowlist is None:
            source_allowlist = sorted({obs.source_id for obs in self.observations})
        self.source_allowlist = tuple(str(value) for value in source_allowlist)

    def _default_bbox(self):
        if not self.observations:
            return (-180.0, -90.0, 180.0, 90.0)
        lons = [obs.lon for obs in self.observations]
        lats = [obs.lat for obs in self.observations]
        return (min(lons), min(lats), max(lons), max(lats))

    def _last_event_time(self):
        if not self.observations:
            return self.scenario_start
        return max(obs.event_time for obs in self.observations)

    @property
    def hash(self) -> str:
        payload = [o.content_hash() for o in self.observations]
        return hashlib.sha256("|".join(payload).encode("utf-8")).hexdigest()

    @property
    def envelope_hash(self) -> str:
        """Hash of the bundle envelope, not only its observations."""
        payload = {
            "bundle_id": self.bundle_id,
            "scenario_id": self.scenario_id,
            "domain": self.domain,
            "bbox": list(self.bbox),
            "time_window": list(self.time_window),
            "source_allowlist": list(self.source_allowlist),
            "scenario_start": self.scenario_start,
            "synthetic": self.synthetic,
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "scenario_id": self.scenario_id,
            "domain": self.domain,
            "bbox": list(self.bbox),
            "time_window": list(self.time_window),
            "source_allowlist": list(self.source_allowlist),
            "scenario_start": self.scenario_start,
            "synthetic": self.synthetic,
            "hash": self.hash,
            "envelope_hash": self.envelope_hash,
            "observations": [o.as_dict() for o in self.observations],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IntelBundle":
        """Rebuild a bundle written by ``fetch_intel.py``."""
        if not isinstance(payload, dict):
            raise ValueError("bundle payload must be an object")
        observations = []
        for index, item in enumerate(payload.get("observations") or []):
            if not isinstance(item, dict):
                continue
            geometry = item.get("geometry") or {}
            coords = geometry.get("coordinates") or []
            lat = item.get("lat", coords[1] if len(coords) > 1 else None)
            lon = item.get("lon", coords[0] if len(coords) > 0 else None)
            if lat is None or lon is None:
                continue
            provenance = item.get("provenance") or {}
            observations.append(Observation(
                observation_id=str(item.get("observation_id") or "unknown:%d" % index),
                source_id=str(item.get("source_id") or "unknown"),
                source_native_id=str(item.get("source_native_id") or ""),
                domain=str(item.get("domain") or "other"),
                type=str(item.get("type") or "unknown"),
                lat=float(lat),
                lon=float(lon),
                event_time=str(item.get("event_time") or payload.get("scenario_start") or ""),
                ingest_time=str(item.get("ingest_time") or payload.get("scenario_start") or ""),
                ingest_seq=int(item.get("ingest_seq", index)),
                confidence=float(item.get("confidence") or 0.0),
                source_url=str(item.get("source_url") or ""),
                title=str(item.get("title") or ""),
                geo_precision=str(item.get("geo_precision") or "exact"),
                geo_uncertainty_m=float(item.get("geo_uncertainty_m") or 0.0),
                is_centroid=bool(item.get("is_centroid")),
                status=str(item.get("status") or "active"),
                properties=dict(item.get("properties") or {}),
                schema_version=str(item.get("schema_version") or "situation3d.intel.v1"),
                license=str(item.get("license") or ""),
                attribution=str(item.get("attribution") or ""),
                parser_version=str(provenance.get("parser_version")
                                   or "vgi-connectors-1"),
            ))
        return cls(
            bundle_id=str(payload.get("bundle_id") or "bundle"),
            scenario_id=str(payload.get("scenario_id") or "scenario"),
            observations=observations,
            scenario_start=str(payload.get("scenario_start") or ""),
            synthetic=bool(payload.get("synthetic", True)),
            domain=str(payload.get("domain") or "multi"),
            bbox=payload.get("bbox") or None,
            time_window=payload.get("time_window") or None,
            source_allowlist=payload.get("source_allowlist") or None,
        )


class IntelGateway:
    """Consume observations as future-only cues, never as ground truth."""

    def __init__(self, bundle: IntelBundle, ttl_s: float = 3600.0,
                 late_policy: str = "discard_and_log", fusion=None,
                 case_filter=None):
        self.bundle = bundle
        self.ttl_s = float(ttl_s)
        self.late_policy = late_policy
        self.fusion = fusion
        self.case_filter = case_filter
        self._start_ms = int(parse_time(bundle.scenario_start).timestamp() * 1000)
        self._consumed = set()
        self.late = []

    def set_case_filter(self, case_filter):
        """Update the AOI/text filter without rebuilding the bundle."""
        self.case_filter = case_filter
        return self

    @property
    def filter_hash(self):
        return getattr(self.case_filter, "filter_hash", None)

    def _matches(self, observation: Observation) -> bool:
        if self.case_filter is None:
            return True
        matcher = getattr(self.case_filter, "matches", self.case_filter)
        return bool(matcher(observation))

    def consume(self, side: int, sim_time: float, watermark: float,
                lookahead: float) -> CueBatch:
        horizon = float(sim_time) + float(lookahead)
        if self.fusion is not None:
            return self._consume_fused(side, horizon, watermark)
        cues: List[IntelCue] = []
        for obs in self.bundle.observations:
            key = (int(side), obs.observation_id)
            if key in self._consumed:
                continue
            sim = (obs.event_ms() - self._start_ms) / 1000.0
            if sim > horizon:
                break
            obs_side = obs.properties.get("side")
            if obs_side is not None and int(obs_side) != int(side):
                continue
            if not self._matches(obs):
                continue
            if sim < watermark:
                self.late.append(obs.observation_id)
                self._consumed.add(key)
                continue
            cue = IntelCue(
                cue_id="cue:" + obs.observation_id,
                observation_id=obs.observation_id,
                ingest_seq=obs.ingest_seq,
                event_time_ms=obs.event_ms(),
                knowledge_time_ms=int(parse_time(obs.ingest_time).timestamp() * 1000),
                source_priority=SOURCE_PRIORITY.get(obs.source_id, 9),
                cue_kind="observation",
                sim_time=sim,
                valid_from=sim,
                valid_to=sim + self.ttl_s,
                side=int(side),
                geometry={"type": "Point", "coordinates": [obs.lon, obs.lat]},
                domain=obs.domain,
                type=obs.type,
                confidence=obs.confidence,
                ttl_s=self.ttl_s,
                provenance={
                    "source_id": obs.source_id,
                    "source_url": obs.source_url,
                    **{
                        key: obs.properties[key]
                        for key in ("event_start_time", "alert_level",
                                    "from_time", "to_time")
                        if key in obs.properties
                    },
                },
                sensor_model={},
            )
            cues.append(cue)
            self._consumed.add(key)
        cues.sort(key=lambda c: (c.event_time_ms, c.source_priority, c.cue_id))
        payload = [(c.cue_id, c.event_time_ms) for c in cues]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return CueBatch(tuple(cues), float(watermark), self.late_policy, digest)

    def _consume_fused(self, side: int, horizon: float,
                       watermark: float) -> CueBatch:
        if isinstance(self.fusion, dict):
            radius = float(self.fusion.get("radius_m", 10000.0))
            window = float(self.fusion.get("window_s", 3600.0))
        else:
            radius, window = (float(v) for v in self.fusion)

        # Member-level future-only and watermark gating: only observations
        # inside [watermark, horizon] are clustered, so a cue never contains
        # evidence that is not yet knowable or that was already dropped.
        by_id: Dict[str, Observation] = {}
        for obs in self.bundle.observations:
            key = (int(side), obs.observation_id)
            if key in self._consumed:
                continue
            obs_side = obs.properties.get("side")
            if obs_side is not None and int(obs_side) != int(side):
                continue
            if not self._matches(obs):
                continue
            sim = (obs.event_ms() - self._start_ms) / 1000.0
            if sim > horizon:
                continue
            if sim < watermark:
                self.late.append(obs.observation_id)
                self._consumed.add(key)
                continue
            current = by_id.get(obs.observation_id)
            if current is None or (
                    (float(obs.confidence), obs.content_hash())
                    > (float(current.confidence), current.content_hash())):
                by_id[obs.observation_id] = obs

        events = fuse(list(by_id.values()), radius_m=radius, window_s=window)
        cues: List[IntelCue] = []
        for event in events:
            members = [by_id[i] for i in event.member_ids if i in by_id]
            if not members:
                continue
            event_ms = int(parse_time(event.event_time).timestamp() * 1000)
            sim = (event_ms - self._start_ms) / 1000.0
            knowledge_ms = max(
                int(parse_time(m.ingest_time).timestamp() * 1000)
                for m in members
            )
            priority = min(
                (SOURCE_PRIORITY.get(m.source_id, 9) for m in members),
                default=9,
            )
            cue = IntelCue(
                cue_id="cue:" + event.cluster_id,
                observation_id=event.cluster_id,
                ingest_seq=max(m.ingest_seq for m in members),
                event_time_ms=event_ms,
                knowledge_time_ms=knowledge_ms,
                source_priority=priority,
                cue_kind="observation",
                sim_time=sim,
                valid_from=sim,
                valid_to=sim + self.ttl_s,
                side=int(side),
                geometry={"type": "Point",
                          "coordinates": [event.lon, event.lat]},
                domain=event.domain,
                type=event.type,
                confidence=event.confidence,
                ttl_s=self.ttl_s,
                provenance={
                    "cluster_id": event.cluster_id,
                    "member_ids": list(event.member_ids),
                    "sources": list(event.sources),
                    "representative_id": event.representative_id,
                    "fused": True,
                },
                sensor_model={},
            )
            cues.append(cue)
            for member in members:
                self._consumed.add((int(side), member.observation_id))
        cues.sort(key=lambda c: (c.event_time_ms, c.source_priority, c.cue_id))
        payload = [(c.cue_id, c.event_time_ms) for c in cues]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return CueBatch(tuple(cues), float(watermark), self.late_policy, digest)
