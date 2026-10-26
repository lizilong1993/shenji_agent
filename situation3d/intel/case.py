"""Case-scoped intelligence profiles and filtering.

A profile is deliberately data, not code: a case owns an AOI, a domain/source
allow-list and optional text/country terms.  The same object is used by the
fetch CLI (pre-filtering a bundle) and by the gateway (runtime filtering, so a
caller can move the AOI without rebuilding the bundle).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .model import Observation

DEFAULT_PROFILE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "case_profiles")


def _bbox(values: Sequence[float]) -> Tuple[float, float, float, float]:
    if isinstance(values, str) or len(values) != 4:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
    west, south, east, north = (float(value) for value in values)
    if west > east or south > north:
        raise ValueError("bbox min values must not exceed max values")
    if not -180.0 <= west <= 180.0 or not -180.0 <= east <= 180.0:
        raise ValueError("bbox longitude out of range")
    if not -90.0 <= south <= 90.0 or not -90.0 <= north <= 90.0:
        raise ValueError("bbox latitude out of range")
    return west, south, east, north


def _strings(values: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    return tuple(str(value).strip() for value in values if str(value).strip())


def _flatten(value: Any, depth: int = 0) -> str:
    if depth > 4 or value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_flatten(item, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(item, depth + 1) for item in value)
    return str(value)


@dataclass(frozen=True)
class CaseIntelProfile:
    case_id: str
    name: str
    bbox: Tuple[float, float, float, float]
    domains: Tuple[str, ...] = ()
    sources: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()
    countries: Tuple[str, ...] = ()
    buffer_km: float = 0.0
    lookback_hours: float = 24.0
    max_items: int = 5000

    def __post_init__(self):
        object.__setattr__(self, "case_id", str(self.case_id or "case"))
        object.__setattr__(self, "name", str(self.name or self.case_id))
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        object.__setattr__(self, "domains",
                           tuple(value.lower() for value in _strings(self.domains)))
        object.__setattr__(self, "sources",
                           tuple(value.lower() for value in _strings(self.sources)))
        object.__setattr__(self, "keywords", _strings(self.keywords))
        object.__setattr__(self, "countries", _strings(self.countries))
        object.__setattr__(self, "buffer_km", max(0.0, float(self.buffer_km)))
        object.__setattr__(self, "lookback_hours",
                           max(0.0, float(self.lookback_hours)))
        object.__setattr__(self, "max_items", max(1, int(self.max_items)))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CaseIntelProfile":
        if not isinstance(payload, dict):
            raise ValueError("case profile must be an object")
        return cls(
            case_id=payload.get("case_id") or payload.get("id") or "case",
            name=payload.get("name") or payload.get("case_id") or "",
            bbox=payload.get("bbox") or (-180.0, -90.0, 180.0, 90.0),
            domains=payload.get("domains") or (),
            sources=payload.get("sources") or (),
            keywords=payload.get("keywords") or (),
            countries=payload.get("countries") or (),
            buffer_km=payload.get("buffer_km") or 0.0,
            lookback_hours=payload.get("lookback_hours") or 24.0,
            max_items=payload.get("max_items") or 5000,
        )

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "name": self.name,
            "bbox": list(self.bbox),
            "domains": list(self.domains),
            "sources": list(self.sources),
            "keywords": list(self.keywords),
            "countries": list(self.countries),
            "buffer_km": self.buffer_km,
            "lookback_hours": self.lookback_hours,
            "max_items": self.max_items,
        }

    @property
    def center(self) -> Tuple[float, float]:
        west, south, east, north = self.bbox
        return ((west + east) / 2.0, (south + north) / 2.0)

    @property
    def expanded_bbox(self) -> Tuple[float, float, float, float]:
        west, south, east, north = self.bbox
        if self.buffer_km <= 0.0:
            return self.bbox
        lat_km = 111.32
        lat_delta = self.buffer_km / lat_km
        center_lat = (south + north) / 2.0
        cos_lat = max(0.05, math.cos(math.radians(center_lat)))
        lon_delta = self.buffer_km / (lat_km * cos_lat)
        return (max(-180.0, west - lon_delta),
                max(-90.0, south - lat_delta),
                min(180.0, east + lon_delta),
                min(90.0, north + lat_delta))

    @property
    def filter_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_overrides(self, **kwargs) -> "CaseIntelProfile":
        allowed = {"case_id", "name", "bbox", "domains", "sources",
                   "keywords", "countries", "buffer_km", "lookback_hours",
                   "max_items"}
        changes = {key: value for key, value in kwargs.items()
                   if key in allowed and value is not None}
        return replace(self, **changes) if changes else self

    def _haystack(self, observation: Observation) -> str:
        return " ".join((
            observation.title,
            observation.source_url,
            observation.type,
            observation.domain,
            _flatten(observation.properties),
        )).lower()

    def matches(self, observation: Observation) -> bool:
        west, south, east, north = self.expanded_bbox
        if not (west <= observation.lon <= east
                and south <= observation.lat <= north):
            return False
        if self.domains and observation.domain.lower() not in self.domains:
            return False
        if self.sources and observation.source_id.lower() not in self.sources:
            return False
        # Keywords/countries are for text leads whose location is a case
        # centroid (GDELT) or only admin-level.  A geolocated observation
        # inside the AOI is already case-relevant; requiring its callsign or
        # title to contain a keyword would drop every ADS-B/AIS track.
        coarse = (observation.geo_precision in ("unknown", "admin0")
                  or observation.is_centroid)
        text = None
        if self.keywords and coarse:
            text = self._haystack(observation)
            if not any(keyword.lower() in text for keyword in self.keywords):
                return False
        if self.countries and coarse:
            if text is None:
                text = self._haystack(observation)
            if not any(country.lower() in text for country in self.countries):
                return False
        return True

    # Alias used by gateway callers.
    def __call__(self, observation: Observation) -> bool:
        return self.matches(observation)


def load_case_profile(identifier: Optional[str] = None,
                      root: Optional[str] = None) -> CaseIntelProfile:
    """Load ``<identifier>.json`` from a case-profile directory or a path."""
    if not identifier:
        identifier = "global"
    text = str(identifier)
    candidates = []
    if os.path.isfile(text):
        candidates.append(text)
    base = root or DEFAULT_PROFILE_ROOT
    for name in (text + ".json", text + ".case.json"):
        candidates.append(os.path.join(base, name))
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                return CaseIntelProfile.from_dict(json.load(handle))
    available = list_case_profiles(root)
    raise FileNotFoundError(
        "case profile %r not found; available: %s"
        % (identifier, ", ".join(available) or "none"))


def list_case_profiles(root: Optional[str] = None) -> Tuple[str, ...]:
    directory = root or DEFAULT_PROFILE_ROOT
    if not os.path.isdir(directory):
        return ()
    names = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            names.append(name[:-5])
    return tuple(names)
