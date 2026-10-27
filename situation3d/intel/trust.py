"""Source reliability and recency-decayed confidence."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

SOURCE_RELIABILITY = {
    "usgs": 0.98,
    "gdacs": 0.95,
    "firms": 0.80,
    "eonet": 0.85,
    "adsb": 0.75,
    "gdelt": 0.60,
    "open-meteo": 0.80,
    "rss": 0.55,
}
HALF_LIFE_H = {
    "hazard": 6.0,
    "conflict": 24.0,
    "maritime": 6.0,
    "aviation": 6.0,
    "weather": 3.0,
    "news": 12.0,
    "geography": 168.0,
    "society": 24.0,
    "social": 6.0,
    "selfmedia": 12.0,
    "health": 72.0,
    "infrastructure": 12.0,
    "other": 24.0,
}
GEO_FACTOR = {
    "exact": 1.0,
    "street": 0.95,
    "locality": 0.85,
    "admin1": 0.70,
    "admin0": 0.55,
    "unknown": 0.40,
}


def recency(event_time: datetime, now: datetime, domain: str = "other") -> float:
    age_h = max(0.0, (now - event_time).total_seconds() / 3600.0)
    half_life = HALF_LIFE_H.get(domain, HALF_LIFE_H["other"])
    return max(0.2, math.pow(0.5, age_h / half_life))


def confidence(source_id: str, event_time: datetime, now: datetime,
               domain: str = "other", geo_precision: str = "exact",
               is_centroid: bool = False) -> float:
    reliability = SOURCE_RELIABILITY.get(source_id, 0.7)
    geo = GEO_FACTOR.get(geo_precision, 0.4)
    if is_centroid:
        geo *= 0.7
    value = reliability * recency(event_time, now, domain) * geo
    return round(max(0.0, min(1.0, value)), 3)


def components(source_id: str, event_time: datetime, now: datetime,
               domain: str = "other", geo_precision: str = "exact",
               is_centroid: bool = False) -> dict:
    reliability = SOURCE_RELIABILITY.get(source_id, 0.7)
    rec = recency(event_time, now, domain)
    geo = GEO_FACTOR.get(geo_precision, 0.4) * (0.7 if is_centroid else 1.0)
    return {
        "source_reliability": reliability,
        "recency": round(rec, 3),
        "geo_precision": round(geo, 3),
        "confidence": round(reliability * rec * geo, 3),
    }
