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


SOURCE_PRIORITY = {"usgs": 0, "gdacs": 1, "firms": 2}


class IntelBundle:
    def __init__(self, bundle_id: str, scenario_id: str,
                 observations: Sequence[Observation], scenario_start: str,
                 synthetic: bool = True):
        self.bundle_id = bundle_id
        self.scenario_id = scenario_id
        self.scenario_start = scenario_start
        self.synthetic = bool(synthetic)
        self.observations = sorted(
            observations,
            key=lambda o: (o.event_ms(), SOURCE_PRIORITY.get(o.source_id, 9),
                           o.observation_id),
        )

    @property
    def hash(self) -> str:
        payload = [o.content_hash() for o in self.observations]
        return hashlib.sha256("|".join(payload).encode("utf-8")).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "scenario_id": self.scenario_id,
            "scenario_start": self.scenario_start,
            "synthetic": self.synthetic,
            "hash": self.hash,
            "observations": [o.as_dict() for o in self.observations],
        }


class IntelGateway:
    """Consume observations as future-only cues, never as ground truth."""

    def __init__(self, bundle: IntelBundle, ttl_s: float = 3600.0,
                 late_policy: str = "discard_and_log", fusion=None):
        self.bundle = bundle
        self.ttl_s = float(ttl_s)
        self.late_policy = late_policy
        self.fusion = fusion
        self._fused: Dict[int, List[Any]] = {}
        self._start_ms = int(parse_time(bundle.scenario_start).timestamp() * 1000)
        self._consumed = set()
        self.late = []

    def _fused_events(self, side: int):
        if self.fusion is None:
            return None
        if side in self._fused:
            return self._fused[side]
        if isinstance(self.fusion, dict):
            radius = float(self.fusion.get("radius_m", 10000.0))
            window = float(self.fusion.get("window_s", 3600.0))
        else:
            radius, window = (float(v) for v in self.fusion)
        observations = [
            obs for obs in self.bundle.observations
            if obs.properties.get("side") is None
            or int(obs.properties["side"]) == int(side)
        ]
        events = fuse(observations, radius_m=radius, window_s=window)
        self._fused[side] = events
        return events

    def consume(self, side: int, sim_time: float, watermark: float,
                lookahead: float) -> CueBatch:
        horizon = float(sim_time) + float(lookahead)
        events = self._fused_events(side)
        if events is not None:
            return self._consume_fused(events, side, horizon, watermark)
        cues: List[IntelCue] = []
        for obs in self.bundle.observations:
            if obs.observation_id in self._consumed:
                continue
            sim = (obs.event_ms() - self._start_ms) / 1000.0
            if sim > horizon:
                break
            obs_side = obs.properties.get("side")
            if obs_side is not None and int(obs_side) != int(side):
                continue
            if sim < watermark:
                self.late.append(obs.observation_id)
                self._consumed.add(obs.observation_id)
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
                },
                sensor_model={},
            )
            cues.append(cue)
            self._consumed.add(obs.observation_id)
        cues.sort(key=lambda c: (c.event_time_ms, c.source_priority, c.cue_id))
        payload = [(c.cue_id, c.event_time_ms) for c in cues]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return CueBatch(tuple(cues), float(watermark), self.late_policy, digest)

    def _consume_fused(self, events, side: int, horizon: float,
                       watermark: float) -> CueBatch:
        by_id = {obs.observation_id: obs for obs in self.bundle.observations}
        cues: List[IntelCue] = []
        for event in events:
            if event.cluster_id in self._consumed:
                continue
            event_ms = int(parse_time(event.event_time).timestamp() * 1000)
            sim = (event_ms - self._start_ms) / 1000.0
            if sim > horizon:
                break
            if sim < watermark:
                self.late.append(event.cluster_id)
                self._consumed.add(event.cluster_id)
                continue
            members = [by_id[i] for i in event.member_ids if i in by_id]
            knowledge_ms = min(
                int(parse_time(m.ingest_time).timestamp() * 1000)
                for m in members
            ) if members else event_ms
            priority = min(
                (SOURCE_PRIORITY.get(m.source_id, 9) for m in members),
                default=9,
            )
            cue = IntelCue(
                cue_id="cue:" + event.cluster_id,
                observation_id=event.cluster_id,
                ingest_seq=min((m.ingest_seq for m in members), default=0),
                event_time_ms=event_ms,
                knowledge_time_ms=knowledge_ms,
                source_priority=priority,
                cue_kind="fused",
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
                },
                sensor_model={},
            )
            cues.append(cue)
            self._consumed.add(event.cluster_id)
        cues.sort(key=lambda c: (c.event_time_ms, c.source_priority, c.cue_id))
        payload = [(c.cue_id, c.event_time_ms) for c in cues]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return CueBatch(tuple(cues), float(watermark), self.late_policy, digest)
