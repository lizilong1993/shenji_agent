"""NASA FIRMS active-fire connector: CSV parsing and row preparation."""

from __future__ import annotations

import csv
import io
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from ..model import parse_time
from ..normalize import normalize_firms

API_ROOT = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SENSORS = (
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
)


def day_range(hours: float) -> int:
    """FIRMS accepts 1..10 trailing days; round a hour window up."""
    return max(1, min(10, int(math.ceil(max(0.0, float(hours)) / 24.0))))


def build_url(map_key: str, sensor: str, bbox: Optional[Sequence[float]],
              days: int, end: Any) -> str:
    if not map_key:
        raise ValueError("FIRMS MAP_KEY is empty")
    if bbox is None:
        area = "world"
    else:
        if len(bbox) != 4:
            raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
        west, south, east, north = (float(v) for v in bbox)
        if west > east or south > north:
            raise ValueError("bbox min values must not exceed max values")
        area = ",".join("%.6f" % v for v in (west, south, east, north))
    date = parse_time(end).strftime("%Y-%m-%d")
    return "%s/%s/%s/%s/%d/%s" % (
        API_ROOT,
        quote(str(map_key), safe=""),
        quote(str(sensor), safe=""),
        quote(area, safe=","),
        int(days),
        date,
    )


def parse_csv(text: str) -> List[Dict[str, str]]:
    """Parse a FIRMS CSV response into a list of dicts."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    rows = []
    for row in reader:
        if not row:
            continue
        clean = {}
        for key, value in row.items():
            if key is None:
                continue
            clean[str(key).strip()] = value.strip() if isinstance(value, str) else value
        if clean:
            rows.append(clean)
    return rows


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _event_time(date_text: Any, time_text: Any) -> Optional[str]:
    """Combine FIRMS ``acq_date`` (YYYY-MM-DD) and ``acq_time`` (HHMM)."""
    time_text = str(time_text or "").strip()
    date_text = str(date_text or "").strip()
    if "T" in time_text:
        try:
            return parse_time(time_text).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
    if date_text and time_text:
        digits = time_text.replace(".0", "")
        if digits.isdigit() and len(digits) <= 4:
            digits = digits.zfill(4)
            try:
                stamp = datetime.strptime(date_text, "%Y-%m-%d")
            except ValueError:
                return None
            hour = int(digits[:2])
            minute = int(digits[2:4])
            if hour > 23 or minute > 59:
                return None
            return "%sT%02d:%02d:00Z" % (stamp.strftime("%Y-%m-%d"), hour, minute)
    if date_text:
        try:
            stamp = datetime.strptime(date_text, "%Y-%m-%d")
        except ValueError:
            return None
        return stamp.strftime("%Y-%m-%dT00:00:00Z")
    return None


def prepare_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw FIRMS rows to the shape expected by ``normalize_firms``."""
    out = []
    for row in rows:
        lat = _as_float(row.get("latitude"))
        lon = _as_float(row.get("longitude"))
        if lat is None or lon is None:
            continue
        event = _event_time(row.get("acq_date"), row.get("acq_time"))
        if not event:
            continue
        clean = dict(row)
        clean["latitude"] = lat
        clean["longitude"] = lon
        clean["acq_time"] = event
        out.append(clean)
    return out


def normalize_payload(rows: Sequence[Dict[str, Any]], fetched_at: Any,
                      start_seq: int = 0):
    """Normalize prepared FIRMS rows."""
    return normalize_firms(prepare_rows(rows), fetched_at, start_seq=start_seq)
