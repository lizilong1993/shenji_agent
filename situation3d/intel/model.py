"""Canonical VGI observation model (P1 slice)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source_id: str
    source_native_id: str
    domain: str
    type: str
    lat: float
    lon: float
    event_time: str
    ingest_time: str
    ingest_seq: int
    confidence: float
    source_url: str
    title: str = ""
    geo_precision: str = "exact"
    geo_uncertainty_m: float = 0.0
    is_centroid: bool = False
    status: str = "active"
    properties: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "situation3d.intel.v1"
    license: str = ""
    attribution: str = ""
    parser_version: str = "vgi-connectors-1"

    def event_dt(self) -> datetime:
        return parse_time(self.event_time)

    def event_ms(self) -> int:
        return int(self.event_dt().timestamp() * 1000)

    def content_hash(self) -> str:
        payload = {
            "id": self.observation_id,
            "source": self.source_id,
            "native": self.source_native_id,
            "event_time": self.event_time,
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "type": self.type,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_native_id": self.source_native_id,
            "domain": self.domain,
            "type": self.type,
            "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]},
            "event_time": self.event_time,
            "ingest_time": self.ingest_time,
            "ingest_seq": self.ingest_seq,
            "confidence": self.confidence,
            "source_url": self.source_url,
            "title": self.title,
            "geo_precision": self.geo_precision,
            "geo_uncertainty_m": self.geo_uncertainty_m,
            "is_centroid": self.is_centroid,
            "status": self.status,
            "content_hash": self.content_hash(),
            "license": self.license,
            "attribution": self.attribution,
            "provenance": {
                "raw_id": self.source_native_id,
                "parser_version": self.parser_version,
                "license": self.license,
                "attribution": self.attribution,
            },
            "properties": dict(self.properties),
        }
