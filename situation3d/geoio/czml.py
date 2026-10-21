"""CZML export for Cesium replay (P5)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

SIDE_COLOR = {0: [200, 40, 40, 255], 1: [40, 90, 220, 255], -1: [160, 160, 160, 255]}
DEFAULT_START = "2026-10-21T00:00:00Z"


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _entity_packet(entity, start_iso: str, end_iso: str) -> dict:
    packet = {
        "id": "entity-%s" % entity.uid,
        "name": entity.name,
        "availability": "%s/%s" % (start_iso, end_iso),
        "position": {
            "epoch": start_iso,
            "cartographicDegrees": [0.0, entity.pose.lon, entity.pose.lat,
                                    entity.pose.alt_m],
        },
        "point": {"pixelSize": 8,
                  "color": {"rgba": SIDE_COLOR.get(entity.affiliation, SIDE_COLOR[-1])}},
        "label": {"text": entity.name, "font": "12px sans-serif",
                  "pixelOffset": {"cartesian2": [0, 14]},
                  "fillColor": {"rgba": SIDE_COLOR.get(entity.affiliation, SIDE_COLOR[-1])}},
    }
    if entity.route is not None:
        packet["polyline"] = {
            "positions": {"cartographicDegrees": _route_degrees(entity.route)},
            "width": 2,
            "material": {"polylineOutline": {
                "color": {"rgba": SIDE_COLOR.get(entity.affiliation, SIDE_COLOR[-1])},
                "outlineColor": {"rgba": [255, 255, 255, 180]}, "outlineWidth": 1}},
        }
    return packet


def _route_degrees(route):
    out = []
    for p in route.points:
        out.extend([p.lon, p.lat, p.alt_m])
    return out


def world_to_czml(world, start_iso: str = DEFAULT_START,
                  duration_s: float = 43200.0) -> dict:
    start = _dt(start_iso)
    end_iso = _iso(start + timedelta(seconds=duration_s))
    packets = [{
        "id": "document",
        "name": "geo-engine replay",
        "version": "1.0",
        "clock": {"interval": "%s/%s" % (start_iso, end_iso),
                  "currentTime": start_iso,
                  "multiplier": 10},
    }]
    for entity in world.sorted_entities():
        if entity.alive:
            packets.append(_entity_packet(entity, start_iso, end_iso))
    return {"packets": packets}


def write_czml(world, path: str, start_iso: str = DEFAULT_START,
               duration_s: float = 43200.0) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(world_to_czml(world, start_iso, duration_s), fh,
                  ensure_ascii=False, indent=1)
