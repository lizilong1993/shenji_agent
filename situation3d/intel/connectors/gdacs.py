"""GDACS multi-hazard connector: URL building and payload normalization."""

from __future__ import annotations

from typing import Any, Optional, Sequence
from urllib.parse import urlencode

from ..model import parse_time
from ..normalize import normalize_gdacs

API_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
EVENT_TYPES = "EQ,TC,FL,VO,WF,DR"
ALERT_LEVELS = "Green;Orange;Red"


def build_url(start: Any, end: Any,
              bbox: Optional[Sequence[float]] = None) -> str:
    """Build a GDACS event-list query.

    The list endpoint has no bbox parameter; the caller filters observations by
    bbox after normalization.  ``bbox`` is accepted to keep the connector
    signature uniform.
    """
    if bbox is not None and len(bbox) != 4:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    params = {
        "eventlist": EVENT_TYPES,
        "fromdate": start_dt.strftime("%Y-%m-%d"),
        "todate": end_dt.strftime("%Y-%m-%d"),
        "alertlevel": ALERT_LEVELS,
    }
    return API_URL + "?" + urlencode(params, safe=",;")


def normalize_payload(payload: dict, fetched_at: Any,
                      start_seq: int = 0):
    """Normalize an already-fetched GDACS FeatureCollection."""
    if not isinstance(payload, dict):
        raise ValueError("GDACS payload must be a FeatureCollection")
    if not isinstance(payload.get("features"), list):
        raise ValueError("GDACS payload has no features list")
    return normalize_gdacs(payload, fetched_at, start_seq=start_seq)
