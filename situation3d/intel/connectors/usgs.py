"""USGS earthquake connector: URL building and payload normalization."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence, Tuple
from urllib.parse import urlencode

from ..model import iso, parse_time
from ..normalize import normalize_usgs

API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def _bbox(bbox: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
    west, south, east, north = (float(v) for v in bbox)
    if west > east or south > north:
        raise ValueError("bbox min values must not exceed max values")
    return (west, south, east, north)


def build_url(start: Any, end: Any, bbox: Optional[Sequence[float]] = None,
              limit: int = 2000) -> str:
    """Build a bounded FDSN GeoJSON query for a time window and bbox."""
    params = {
        "format": "geojson",
        "starttime": iso(parse_time(start)),
        "endtime": iso(parse_time(end)),
        "orderby": "time",
        "limit": int(limit),
    }
    box = _bbox(bbox)
    if box is not None:
        west, south, east, north = box
        params.update({
            "minlatitude": "%.6f" % south,
            "maxlatitude": "%.6f" % north,
            "minlongitude": "%.6f" % west,
            "maxlongitude": "%.6f" % east,
        })
    return API_URL + "?" + urlencode(params)


def normalize_payload(payload: dict, fetched_at: Any,
                      start_seq: int = 0):
    """Normalize an already-fetched USGS GeoJSON payload."""
    if not isinstance(payload, dict):
        raise ValueError("USGS payload must be a GeoJSON object")
    return normalize_usgs(payload, fetched_at, start_seq=start_seq)
