"""Cross-domain association and key-intel extraction (deterministic rules).

This is deliberately explainable: every link and every key item carries the
reasons that produced it.  It is a first-stage ranking layer, not a truth
machine; manual entries remain flagged unverified until an analyst says otherwise.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .fusion import TYPE_SYNONYMS

EARTH_RADIUS_M = 6371000.0


def canonical_type(value: Any) -> str:
    text = str(value or "other").strip().lower().replace(" ", "_")
    return TYPE_SYNONYMS.get(text, text)


def _time(value: Any) -> Optional[datetime]:
    try:
        text = str(value or "").strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _distance_m(lat1, lon1, lat2, lon2) -> float:
    lon1r, lat1r, lon2r, lat2r = map(
        math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    h = (math.sin(dlat / 2.0) ** 2
         + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def _text(item: Dict[str, Any]) -> str:
    record = item.get("record") or {}
    return " ".join(str(value or "") for value in (
        item.get("title"), item.get("content"), item.get("place"),
        record.get("title"), record.get("content"),
        record.get("description"),
    )).lower()


def _keywords(profile) -> Tuple[str, ...]:
    if profile is None:
        return ()
    return tuple(str(value).strip().lower()
                 for value in getattr(profile, "keywords", ())
                 if str(value).strip())


def _term_hit(term: str, text: str) -> bool:
    """Word-boundary matching for short Latin terms; substring for CJK."""
    if not term:
        return False
    if term.isascii() and term.isalpha():
        return re.search(r"(?<![a-z0-9])" + re.escape(term)
                         + r"(?![a-z0-9])", text) is not None
    return term in text


def _shared_keywords(left: Dict[str, Any], right: Dict[str, Any],
                     profile) -> List[str]:
    terms = _keywords(profile)
    if not terms:
        return []
    left_text, right_text = _text(left), _text(right)
    return [term for term in terms
            if _term_hit(term, left_text) and _term_hit(term, right_text)]


def link_items(items: Sequence[Dict[str, Any]], profile=None,
               radius_km: float = 50.0, window_h: float = 24.0,
               max_links: int = 300,
               max_items: int = 600) -> List[Dict[str, Any]]:
    """Find spatial/temporal or shared-topic links between active items."""
    active = [item for item in items
              if item.get("status") != "revoked"]
    if len(active) > max_items:
        def _rank(item):
            confidence = 0.0
            try:
                confidence = float(item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            stamp = _time(item.get("event_time"))
            return (0 if item.get("priority") == "manual" else 1,
                    -confidence,
                    -(stamp.timestamp() if stamp else 0.0))
        active = sorted(active, key=_rank)[:max_items]
    links = []
    for index, left in enumerate(active):
        for right in active[index + 1:]:
            if left.get("id") == right.get("id"):
                continue
            try:
                lat1, lon1 = float(left["lat"]), float(left["lon"])
                lat2, lon2 = float(right["lat"]), float(right["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            left_time, right_time = _time(left.get("event_time")), _time(
                right.get("event_time"))
            dt_h = (abs((left_time - right_time).total_seconds()) / 3600.0
                    if left_time and right_time else float("inf"))
            distance_km = _distance_m(lat1, lon1, lat2, lon2) / 1000.0
            spatial = distance_km <= radius_km and dt_h <= window_h
            shared = _shared_keywords(left, right, profile)
            if not spatial and not shared:
                continue
            same_source = left.get("source_id") == right.get("source_id")
            same_type = (canonical_type(left.get("type"))
                         == canonical_type(right.get("type")))
            if same_source and same_type and not shared:
                continue
            if left.get("domain") != right.get("domain"):
                kind = "cross_domain"
            elif shared:
                kind = "topic"
            elif same_type:
                kind = "corroboration"
            else:
                kind = "co_location"
            proximity = max(0.0, 1.0 - distance_km / max(1.0, radius_km))
            temporal = max(0.0, 1.0 - dt_h / max(1.0, window_h))
            topic = min(1.0, len(shared) / 3.0)
            priority = 0.15 if (left.get("priority") == "manual"
                                or right.get("priority") == "manual") else 0.0
            score = min(0.99, 0.34 * proximity + 0.26 * temporal
                        + 0.25 * topic + priority)
            reasons = []
            if spatial:
                reasons.append("距离 %.1f km / 时间差 %.1f h" % (distance_km, dt_h))
            if shared:
                reasons.append("共同关键词：" + "、".join(shared[:3]))
            if kind == "cross_domain":
                reasons.append("跨领域：%s ↔ %s"
                               % (left.get("domain"), right.get("domain")))
            links.append({
                "from_id": left.get("id"),
                "to_id": right.get("id"),
                "kind": kind,
                "score": round(score, 3),
                "reasons": reasons,
            })
            if len(links) >= max_links:
                break
        if len(links) >= max_links:
            break
    links.sort(key=lambda link: (-link["score"], link["from_id"], link["to_id"]))
    return links


def extract_key_items(items: Sequence[Dict[str, Any]],
                      links: Sequence[Dict[str, Any]], profile=None,
                      limit: int = 10) -> List[Dict[str, Any]]:
    """Rank active items by case relevance and cross-source connection."""
    active = [item for item in items
              if item.get("effective") is not False
              and item.get("status") != "revoked"]
    degree = {}
    for link in links:
        degree[link.get("from_id")] = degree.get(link.get("from_id"), 0) + 1
        degree[link.get("to_id")] = degree.get(link.get("to_id"), 0) + 1
    terms = _keywords(profile)
    focus_domains = set(getattr(profile, "domains", ()) or ()) if profile else set()
    now = datetime.now(timezone.utc)
    scored = []
    for item in active:
        text = _text(item)
        hits = [term for term in terms if _term_hit(term, text)]
        age_h = 0.0
        stamp = _time(item.get("event_time"))
        if stamp is not None:
            age_h = max(0.0, (now - stamp).total_seconds() / 3600.0)
        recency = 1.0 / (1.0 + age_h / 24.0)
        manual = 1.0 if item.get("priority") == "manual" else 0.0
        try:
            confidence = float(item.get("confidence") or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        focus = 1.0 if (not focus_domains
                        or item.get("domain") in focus_domains) else 0.3
        link_score = min(1.0, degree.get(item.get("id"), 0) / 4.0)
        score = (0.28 * manual + 0.22 * min(1.0, len(hits) / 2.0)
                 + 0.18 * confidence + 0.14 * link_score
                 + 0.10 * recency + 0.08 * focus)
        reasons = []
        if manual:
            reasons.append("人工情报优先"
                           + ("（未核实）" if item.get("verification")
                              == "unverified" else ""))
        if hits:
            reasons.append("命中案例关键词：" + "、".join(hits[:3]))
        if degree.get(item.get("id")):
            reasons.append("与 %d 条情报存在关联" % degree[item.get("id")])
        if item.get("source_name"):
            reasons.append("来源：" + str(item.get("source_name")))
        scored.append({
            "id": item.get("id"),
            "score": round(min(0.99, score), 3),
            "reasons": reasons,
        })
    scored.sort(key=lambda row: (-row["score"], str(row["id"])))
    return scored[:max(1, int(limit))]
