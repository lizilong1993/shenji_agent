"""Generic RSS/Atom connector for user-registered news/self-media feeds."""

from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from ..model import Observation, iso, parse_time
from .. import trust

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def _local(tag: str) -> str:
    return str(tag or "").split("}")[-1].lower()


def _child(node, *names):
    for child in list(node):
        if _local(child.tag) in names:
            return child
    return None


def _text(node) -> str:
    if node is None or node.text is None:
        return ""
    return _SPACE.sub(" ", html.unescape(_TAG.sub(" ", node.text))).strip()


def _link(node) -> str:
    link = _child(node, "link")
    if link is None:
        return ""
    return str(link.get("href") or link.text or "").strip()


def _published(node) -> Optional[Any]:
    for name in ("pubdate", "published", "updated", "date"):
        text = _text(_child(node, name))
        if not text:
            continue
        try:
            return parsedate_to_datetime(text)
        except (TypeError, ValueError):
            try:
                return parse_time(text)
            except ValueError:
                continue
    return None


def parse_feed(raw: bytes, source: Dict[str, Any], fetched_at: Any,
               center: tuple, start: Any = None, end: Any = None,
               start_seq: int = 0) -> List[Observation]:
    # Python 3.8's expat does not cap entity expansion; reject declarations
    # instead of trying to parse a potential billion-laughs payload.
    upper = raw[:65536].upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("RSS/Atom DTD and entities are not allowed")
    root = ET.fromstring(raw)
    items = [node for node in root.iter()
             if _local(node.tag) in ("item", "entry")][:500]
    now = parse_time(fetched_at)
    start_dt = parse_time(start) if start is not None else None
    end_dt = parse_time(end) if end is not None else None
    lat, lon = float(center[1]), float(center[0])
    domains = source.get("domains") or []
    kind = str(source.get("kind") or "rss").lower()
    if kind in ("social", "selfmedia"):
        domain = kind
        item_type = kind
    else:
        domain = str(domains[0] if domains else "news")
        item_type = "news"
    out = []
    for item in items:
        title = _text(_child(item, "title"))
        link = _link(item)
        description = (_text(_child(item, "description", "summary", "content"))
                       or title)[:4000]
        if not title and not link:
            continue
        event = _published(item) or now
        if event.tzinfo is None:
            event = event.replace(tzinfo=now.tzinfo)
        event = event.astimezone(now.tzinfo)
        if start_dt is not None and event < start_dt:
            continue
        if end_dt is not None and event > end_dt:
            continue
        native = link or hashlib.sha1(
            (title + "|" + iso(event)).encode("utf-8")).hexdigest()
        source_id = str(source.get("source_id") or "rss")
        out.append(Observation(
            observation_id="rss:%s:%s" % (source_id, hashlib.sha1(
                native.encode("utf-8")).hexdigest()[:20]),
            source_id=source_id,
            source_native_id=native,
            domain=domain,
            type=item_type,
            lat=lat,
            lon=lon,
            event_time=iso(event),
            ingest_time=iso(now),
            ingest_seq=start_seq + len(out),
            confidence=trust.confidence("rss", event, now, domain,
                                        geo_precision="unknown",
                                        is_centroid=True),
            source_url=link or str(source.get("url") or ""),
            title=title or "(untitled)",
            geo_precision="unknown",
            geo_uncertainty_m=50000.0,
            is_centroid=True,
            properties={
                "content": description,
                "feed_url": source.get("url"),
                "article_domain": (urlparse(link).hostname
                                   if link else None),
                "location_basis": "case_centroid",
            },
        ))
    return out
