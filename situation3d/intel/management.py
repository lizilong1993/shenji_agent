"""Unified automatic + manual intelligence overview and conflict resolution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .association import (canonical_type, extract_key_items, link_items)
from .manual import ManualIntelStore
from .sources import SourceStore
from .case import load_case_profile

CONFLICT_RADIUS_KM = 25.0
CONFLICT_WINDOW_H = 6.0
_CASE_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    lon1r, lat1r, lon2r, lat2r = map(
        math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    h = (math.sin(dlat / 2.0) ** 2
         + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def _lat_lon(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat, lon = record.get("lat"), record.get("lon")
    geometry = record.get("geometry") or {}
    coords = geometry.get("coordinates") or []
    if lat is None and len(coords) > 1:
        lat = coords[1]
    if lon is None and len(coords) > 0:
        lon = coords[0]
    try:
        return round(float(lat), 6), round(float(lon), 6)
    except (TypeError, ValueError):
        return None, None


def _auto_item(record: Dict[str, Any], source_names: Dict[str, str]) -> Optional[dict]:
    lat, lon = _lat_lon(record)
    if lat is None or lon is None:
        return None
    source_id = str(record.get("source_id") or "unknown")
    geometry = record.get("geometry") or {}
    return {
        "id": str(record.get("observation_id") or ""),
        "priority": "auto",
        "source_id": source_id,
        "source_name": source_names.get(source_id, source_id),
        "domain": str(record.get("domain") or "other"),
        "type": str(record.get("type") or "other"),
        "event_time": str(record.get("event_time") or ""),
        "lat": lat,
        "lon": lon,
        "place": str((record.get("properties") or {}).get("place") or ""),
        "title": str(record.get("title") or record.get("type") or "自动情报"),
        "content": str(record.get("title") or ""),
        "verification": "automatic",
        "confidence": record.get("confidence"),
        "status": str(record.get("status") or "active"),
        "effective": True,
        "superseded_by": None,
        "conflicts": [],
        "license": record.get("license") or "",
        "attribution": record.get("attribution") or "",
        "geometry": geometry,
        "record": record,
    }


def _manual_item(record: Dict[str, Any]) -> dict:
    return {
        "id": str(record.get("intel_id") or ""),
        "priority": "manual",
        "source_id": str(record.get("source_id") or "manual"),
        "source_name": str(record.get("source_name") or "人工录入"),
        "domain": str(record.get("domain") or "other"),
        "type": str(record.get("intel_type") or record.get("type") or "other"),
        "event_time": str(record.get("event_time") or ""),
        "lat": record.get("lat"),
        "lon": record.get("lon"),
        "place": str(record.get("place") or ""),
        "title": str(record.get("title") or "人工情报"),
        "content": str(record.get("content") or ""),
        "verification": str(record.get("verification") or "unverified"),
        "confidence": None,
        "status": str(record.get("status") or "active"),
        "effective": str(record.get("status") or "active") == "active",
        "superseded_by": None,
        "conflicts": [],
        "license": "",
        "attribution": "人工录入",
        "geometry": {"type": "Point",
                     "coordinates": [record.get("lon"), record.get("lat")]},
        "revision": record.get("revision"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "record": record,
    }


class IntelManager:
    def __init__(self, root: str, profile_root: Optional[str] = None):
        self.root = root
        self.profile_root = profile_root
        self.manual = ManualIntelStore(os.path.join(root, "manual.json"))
        self.sources = SourceStore(os.path.join(root, "sources.json"))

    def _bundle(self, case_id: str) -> Optional[dict]:
        safe = _CASE_SAFE.sub("_", case_id or "default") or "default"
        path = os.path.join(self.root, safe, "bundle.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _profile(self, case_id: str):
        try:
            return load_case_profile(case_id, root=self.profile_root)
        except (OSError, ValueError):
            return None

    def overview(self, case_id: str) -> dict:
        case_id = str(case_id or "default")
        profile = self._profile(case_id)
        sources = self.sources.list()
        source_names = {entry.get("source_id"): entry.get("name")
                        for entry in sources}
        bundle = self._bundle(case_id)
        warnings = []
        if bundle is None:
            warnings.append("案例 %s 暂无自动情报 bundle（var/intel/%s/bundle.json）"
                            % (case_id, _CASE_SAFE.sub("_", case_id)))
            auto_records = []
        else:
            auto_records = [record for record in bundle.get("observations") or []
                            if isinstance(record, dict)
                            and record.get("status", "active") == "active"]
        auto_items = [item for item in (_auto_item(record, source_names)
                                        for record in auto_records)
                      if item is not None]
        manual_records = self.manual.list(case_id, include_revoked=True)
        manual_items = [_manual_item(record) for record in manual_records]
        active_manual = [item for item in manual_items
                         if item.get("status") == "active"]
        auto_by_id = {item["id"]: item for item in auto_items}
        manual_by_id = {item["id"]: item for item in manual_items}
        superseded = {}
        for manual in sorted(active_manual,
                             key=lambda item: str(item.get("updated_at") or "")):
            manual_time = _time(manual.get("event_time"))
            for auto in auto_items:
                if manual.get("domain") != auto.get("domain"):
                    continue
                manual_type = canonical_type(manual.get("type"))
                auto_type = canonical_type(auto.get("type"))
                if manual_type != auto_type and manual_type not in (
                        "target", "threat"):
                    continue
                auto_time = _time(auto.get("event_time"))
                if manual_time is None or auto_time is None:
                    continue
                if abs((manual_time - auto_time).total_seconds()) / 3600.0 > CONFLICT_WINDOW_H:
                    continue
                try:
                    distance = _haversine_km(
                        float(manual["lat"]), float(manual["lon"]),
                        float(auto["lat"]), float(auto["lon"]))
                except (TypeError, ValueError):
                    continue
                if distance > CONFLICT_RADIUS_KM:
                    continue
                manual["conflicts"].append(auto["id"])
                current = superseded.get(auto["id"])
                if current is None or str(manual.get("updated_at") or "") >= str(
                        manual_by_id.get(current, {}).get("updated_at") or ""):
                    superseded[auto["id"]] = manual["id"]
        for auto_id, manual_id in superseded.items():
            auto = auto_by_id.get(auto_id)
            if auto is not None:
                auto["effective"] = False
                auto["superseded_by"] = manual_id
        for manual in manual_items:
            manual["effective"] = manual.get("status") == "active"
        items = manual_items + auto_items
        items.sort(key=lambda item: (
            0 if item.get("priority") == "manual"
            and item.get("status") == "active" else
            1 if item.get("priority") == "auto" and item.get("effective") else
            2 if item.get("priority") == "manual" else 3,
            -(_time(item.get("event_time")) or datetime(1970, 1, 1,
                                                         tzinfo=timezone.utc)).timestamp(),
            str(item.get("id")),
        ))
        links = link_items(items, profile,
                           radius_km=max(CONFLICT_RADIUS_KM, 50.0),
                           window_h=max(CONFLICT_WINDOW_H, 24.0))
        key_items = extract_key_items(items, links, profile)
        counts = {
            "manual_active": len(active_manual),
            "manual_revoked": len(manual_items) - len(active_manual),
            "auto": len(auto_items),
            "effective": sum(1 for item in items if item.get("effective")),
            "superseded": len(superseded),
            "links": len(links),
            "key_items": len(key_items),
        }
        revision = hashlib.sha256("|".join([
            self.manual.revision(),
            self.sources.revision(),
            str((bundle or {}).get("hash") or ""),
            str((profile.to_dict() if profile else "")),
        ]).encode("utf-8")).hexdigest()[:16]
        return {
            "case_id": case_id,
            "generated_at": _now(),
            "revision": revision,
            "profile": profile.to_dict() if profile else None,
            "conflict_rule": {
                "radius_km": CONFLICT_RADIUS_KM,
                "window_h": CONFLICT_WINDOW_H,
                "rule": "同领域、同规范类型、时空邻近时人工情报覆盖自动情报",
            },
            "counts": counts,
            "items": items,
            "links": links,
            "key_items": key_items,
            "sources": sources,
            "warnings": warnings,
        }
