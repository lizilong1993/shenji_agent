"""Open-Meteo current-weather connector (global, keyless)."""

from __future__ import annotations

from typing import Any, Tuple
from urllib.parse import urlencode

from ..model import Observation, iso, parse_time
from .. import trust

API_URL = "https://api.open-meteo.com/v1/forecast"
FIELDS = ("temperature_2m", "relative_humidity_2m", "precipitation",
          "weather_code", "wind_speed_10m", "wind_direction_10m",
          "visibility")


def build_url(lat: float, lon: float) -> str:
    params = {
        "latitude": "%.6f" % float(lat),
        "longitude": "%.6f" % float(lon),
        "current": ",".join(FIELDS),
        "timezone": "UTC",
    }
    return API_URL + "?" + urlencode(params)


def normalize_payload(payload: dict, fetched_at: Any,
                      lat: float, lon: float, start_seq: int = 0):
    now = parse_time(fetched_at)
    current = (payload or {}).get("current")
    if not isinstance(current, dict) or not current.get("time"):
        raise ValueError("Open-Meteo payload has no current time")
    if all(current.get(key) is None for key in FIELDS):
        raise ValueError("Open-Meteo payload has no current fields")
    stamp = parse_time(current.get("time"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=now.tzinfo)
    event = stamp.astimezone(now.tzinfo)
    observation = Observation(
        observation_id="open-meteo:%.6f:%.6f:%s"
        % (float(lat), float(lon), event.strftime("%Y%m%d%H%M")),
        source_id="open-meteo",
        source_native_id="%.6f,%.6f" % (float(lat), float(lon)),
        domain="weather",
        type="weather",
        lat=round(float(lat), 6),
        lon=round(float(lon), 6),
        event_time=iso(event),
        ingest_time=iso(now),
        ingest_seq=start_seq,
        confidence=trust.confidence("open-meteo", event, now, "weather"),
        source_url="https://open-meteo.com/",
        title="Weather at %.4f, %.4f" % (float(lat), float(lon)),
        properties={
            "temperature_c": current.get("temperature_2m"),
            "relative_humidity_pct": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "weather_code": current.get("weather_code"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_direction_deg": current.get("wind_direction_10m"),
            "visibility_m": current.get("visibility"),
            "elevation_m": (payload or {}).get("elevation"),
            "interval_s": current.get("interval"),
            "location_basis": "case_center",
        },
    )
    return [observation]
