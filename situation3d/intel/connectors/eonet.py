"""NASA EONET natural-event connector (global)."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence
from urllib.parse import urlencode

from ..model import parse_time
from ..normalize import normalize_eonet

API_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"


def build_url(start: Any, end: Any,
              bbox: Optional[Sequence[float]] = None) -> str:
    seconds = max(0.0, (parse_time(end) - parse_time(start)).total_seconds())
    days = max(1, min(60, int(math.ceil(seconds / 86400.0)) + 1))
    # Do not send bbox: EONET applies it to the latest geometry only, which
    # loses a storm whose latest point has left the AOI but whose track still
    # crossed it.  Filter points locally instead.
    params = {"status": "open", "days": days}
    return API_URL + "?" + urlencode(params)


def parse_payload(payload: dict, fetched_at: Any,
                  bbox: Optional[Sequence[float]] = None,
                  start: Any = None, end: Any = None, start_seq: int = 0):
    return normalize_eonet(payload, fetched_at, start_seq=start_seq,
                           bbox=bbox, start=start, end=end)
