"""GDELT DOC 2.0 news connector.

GDELT DOC returns articles without coordinates.  Observations are placed at
the case centroid with ``geo_precision="unknown"``; they are text leads, not
located events.
"""

from __future__ import annotations

from typing import Any, Sequence
from urllib.parse import urlencode

from ..model import parse_time
from ..normalize import normalize_gdelt

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def build_url(keywords: Sequence[str], start: Any, end: Any,
              maxrecords: int = 75) -> str:
    terms = [str(term).strip() for term in keywords or [] if str(term).strip()]
    if not terms:
        raise ValueError("GDELT needs at least one case keyword")
    query = " OR ".join('"%s"' % term.replace('"', "") for term in terms[:20])
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max(1, min(int(maxrecords), 250)),
        "format": "json",
        "sort": "datedesc",
        "startdatetime": parse_time(start).strftime("%Y%m%d%H%M%S"),
        "enddatetime": parse_time(end).strftime("%Y%m%d%H%M%S"),
    }
    return API_URL + "?" + urlencode(params)


def parse_payload(payload: dict, fetched_at: Any, centroid,
                  start_seq: int = 0):
    return normalize_gdelt(payload, fetched_at, centroid,
                           start_seq=start_seq)
