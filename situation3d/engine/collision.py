"""2-D oriented-bounding-box (OBB) collision checks for moving targets.

The viewer only has *positions and poses*: Cesium does not give entities a
physics body.  This module supplies the missing geometric part: every target
carries a `collision_bbox` (metres, taken from the normalized GLB asset), and
two boxes overlap when their road-plane footprints overlap after rotation by
the current heading.

It intentionally does **not** change a track.  Lane change and navigation are
separate consumers of these collision pairs; keeping detection pure makes the
same code usable by the campaign generator, the validator and the web layer.

Coordinate convention is the engine convention: boxes are (lat, lon), heading
is a compass angle (0 = north, clockwise), and lengths are metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

EARTH_M_PER_DEG_LAT = 110540.0
EARTH_M_PER_DEG_LON = 111320.0


@dataclass
class Box:
    target_id: str
    lat: float
    lon: float
    heading_deg: float
    length_m: float
    width_m: float
    height_m: float = 0.0


def _xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    x = (lon - ref_lon) * EARTH_M_PER_DEG_LON * math.cos(math.radians(ref_lat))
    y = (lat - ref_lat) * EARTH_M_PER_DEG_LAT
    return x, y


def corners(box: Box, ref_lat: float, ref_lon: float) -> List[Tuple[float, float]]:
    """Four footprint corners in a local east/north metre frame."""
    x, y = _xy(box.lat, box.lon, ref_lat, ref_lon)
    h = math.radians(box.heading_deg)
    # Forward = (east, north); right = forward rotated clockwise.
    fx, fy = math.sin(h), math.cos(h)
    rx, ry = math.cos(h), -math.sin(h)
    hl = max(0.0, float(box.length_m)) * 0.5
    hw = max(0.0, float(box.width_m)) * 0.5
    pts = []
    for fl, fw in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        pts.append((x + fx * hl * fl + rx * hw * fw,
                    y + fy * hl * fl + ry * hw * fw))
    return pts


def _project(points: Sequence[Tuple[float, float]],
             axis: Tuple[float, float]) -> Tuple[float, float]:
    ax, ay = axis
    vals = [px * ax + py * ay for px, py in points]
    return min(vals), max(vals)


def overlap(a: Box, b: Box) -> Optional[float]:
    """Minimum penetration depth in metres, or None when separated.

    SAT on the two boxes' four local axes.  A positive depth is the shortest
    translation that would separate the footprints.
    """
    ref_lat = (a.lat + b.lat) * 0.5
    ref_lon = (a.lon + b.lon) * 0.5
    ca = corners(a, ref_lat, ref_lon)
    cb = corners(b, ref_lat, ref_lon)
    axes = []
    for box in (a, b):
        h = math.radians(box.heading_deg)
        axes.append((math.sin(h), math.cos(h)))
        axes.append((math.cos(h), -math.sin(h)))
    depth = float("inf")
    for axis in axes:
        amin, amax = _project(ca, axis)
        bmin, bmax = _project(cb, axis)
        if amax < bmin or bmax < amin:
            return None
        depth = min(depth, amax - bmin, bmax - amin)
    return round(depth, 3)


def box_from_row(row: Mapping, target: Mapping) -> Optional[Box]:
    bbox = target.get("collision_bbox") or {}
    try:
        length = float(bbox.get("length_m") or 0.0)
        width = float(bbox.get("width_m") or 0.0)
    except (TypeError, ValueError):
        return None
    if length <= 0 or width <= 0:
        return None
    try:
        return Box(
            target_id=str(row.get("id") or target.get("id") or ""),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            heading_deg=float(row.get("heading") or 0.0),
            length_m=length,
            width_m=width,
            height_m=float(bbox.get("height_m") or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def frame_collisions(frame: Mapping, targets: Mapping[str, Mapping],
                     domains: Iterable[str] = ("land",)) -> List[dict]:
    """All overlapping pairs in one frame.

    Only targets present in the frame with a collision bbox are considered.
    `mode=embarked` targets are skipped: they inherit the carrier's position
    and must not collide with the deck they are standing on.
    """
    domain_set = set(domains)
    boxes: List[Box] = []
    for row in frame.get("targets", []) or []:
        tid = row.get("id")
        target = targets.get(tid)
        if not target or target.get("domain") not in domain_set:
            continue
        if row.get("mode") == "embarked":
            continue
        box = box_from_row(row, target)
        if box is not None:
            boxes.append(box)
    out: List[dict] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            depth = overlap(boxes[i], boxes[j])
            if depth is not None:
                out.append({
                    "tick": int(frame.get("tick") or 0),
                    "a": boxes[i].target_id,
                    "b": boxes[j].target_id,
                    "depth_m": depth,
                })
    return out


def _lerp_row(a: Mapping, b: Mapping, alpha: float) -> dict:
    if alpha <= 0:
        return dict(a)
    if alpha >= 1:
        return dict(b)
    out = dict(a)
    out["lat"] = float(a["lat"]) + (float(b["lat"]) - float(a["lat"])) * alpha
    out["lon"] = float(a["lon"]) + (float(b["lon"]) - float(a["lon"])) * alpha
    h0 = float(a.get("heading") or 0.0)
    dh = (float(b.get("heading") or 0.0) - h0) % 360.0
    if dh > 180.0:
        dh -= 360.0
    out["heading"] = (h0 + dh * alpha) % 360.0
    return out


def swept_collisions(frames: Sequence[Mapping], targets: Mapping[str, Mapping],
                     substeps: int = 4, domains: Iterable[str] = ("land",)) -> List[dict]:
    """Collision pairs sampled between frames as well as at frame times.

    `substeps=4` samples 7.5 s apart at the campaign's 30 s tick.  The result
    is deduplicated per pair and tick interval, so a long overlap yields one
    entry with the deepest penetration rather than one entry per sample.
    """
    domains = tuple(domains)
    best: Dict[Tuple[str, str, int], dict] = {}
    for i in range(1, len(frames)):
        a = frames[i - 1]
        b = frames[i]
        rows_a = {r.get("id"): r for r in (a.get("targets") or [])}
        rows_b = {r.get("id"): r for r in (b.get("targets") or [])}
        for k in range(1, max(1, int(substeps)) + 1):
            alpha = k / float(max(1, int(substeps)))
            ids = set(rows_a) & set(rows_b)
            virtual = {"tick": a.get("tick"), "targets": []}
            for tid in ids:
                ra, rb = rows_a.get(tid), rows_b.get(tid)
                if not ra or not rb:
                    continue
                row = _lerp_row(ra, rb, alpha)
                row["id"] = tid
                row["mode"] = ra.get("mode")
                virtual["targets"].append(row)
            for hit in frame_collisions(virtual, targets, domains):
                key = (hit["a"], hit["b"], int(a.get("tick") or 0))
                prev = best.get(key)
                if prev is None or hit["depth_m"] > prev["depth_m"]:
                    hit["interval"] = [int(a.get("tick") or 0), int(b.get("tick") or 0)]
                    best[key] = hit
    return sorted(best.values(), key=lambda h: (h["interval"][0], h["a"], h["b"]))


def summarize(hits: Sequence[Mapping]) -> dict:
    by_pair: Dict[Tuple[str, str], int] = {}
    first: Dict[Tuple[str, str], int] = {}
    worst: Dict[Tuple[str, str], float] = {}
    for h in hits:
        pair = tuple(sorted((str(h["a"]), str(h["b"]))))
        by_pair[pair] = by_pair.get(pair, 0) + 1
        tick = int(h.get("tick") or 0)
        first[pair] = min(first.get(pair, tick), tick)
        worst[pair] = max(worst.get(pair, 0.0), float(h.get("depth_m") or 0.0))
    return {
        "pairs": len(by_pair),
        "entries": len(hits),
        "by_pair": [
            {"a": a, "b": b, "count": by_pair[(a, b)],
             "first_tick": first[(a, b)], "max_depth_m": round(worst[(a, b)], 3)}
            for a, b in sorted(by_pair, key=lambda p: (-by_pair[p], p))
        ],
    }
