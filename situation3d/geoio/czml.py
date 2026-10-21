"""CZML export for Cesium replay (P5 minimal)."""

from __future__ import annotations

import json
from typing import Optional

SIDE_COLOR = {0: [200, 40, 40, 255], 1: [40, 90, 220, 255], -1: [160, 160, 160, 255]}


def _entity_packet(entity, t0: float) -> dict:
    packet = {
        "id": "entity-%s" % entity.uid,
        "name": entity.name,
        "availability": "%s/PT0S" % _iso(t0),
        "position": {"cartographicDegrees": [t0, entity.pose.lon, entity.pose.lat,
                                             entity.pose.alt_m]},
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


def _iso(seconds: float) -> str:
    # CZML supports ISO strings; keep a deterministic epoch-relative form.
    total = int(round(seconds))
    return "PT%sS" % total


def world_to_czml(world, t0: float = 0.0) -> dict:
    packets = [{
        "id": "document",
        "name": "geo-engine replay",
        "version": "1.0",
        "clock": {"interval": "%s/PT1H" % _iso(t0), "currentTime": _iso(t0),
                  "multiplier": 10},
    }]
    for entity in world.sorted_entities():
        if entity.alive:
            packets.append(_entity_packet(entity, t0))
    return {"packets": packets}


def write_czml(world, path: str, t0: float = 0.0) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(world_to_czml(world, t0), fh, ensure_ascii=False, indent=1)
