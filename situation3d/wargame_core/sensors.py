"""Sensor model for the clean-room joint wargame core.

Design
------
Sensors are data, not special cases.  Each sensor has a kind, a nominal range,
a quality factor and (optionally) an arc.  The detection score is a product of

    nominal_range * quality * (1 - jam) * signature_factor * environment

and a target is detected when the distance in hexes is within that effective
range and the sensor's line of sight is clear.  Passive sensors (ESM) never
reveal the observer.

The sea/air parameters are **synthetic demonstration values** (exported in the
model block and marked ``approximated``); they are not real performance data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from . import hexmap
from .los import LineOfSight


SENSOR_KINDS = ("radar", "ir", "sonar", "esm", "visual")


def pos_outside(unit, hmap: hexmap.HexMap) -> bool:
    try:
        return not hmap.valid(int(getattr(unit, "pos", -1)))
    except (TypeError, ValueError):
        return True


@dataclass
class Sensor:
    kind: str
    range_km: float
    quality: float = 1.0
    passive: bool = False
    arc_deg: float = 360.0

    def effective_km(self, target, jam: float = 0.0) -> float:
        sig = signature_factor(self.kind, target)
        return max(0.0, self.range_km * self.quality * (1.0 - jam) * sig)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "range_km": self.range_km,
                "quality": self.quality, "passive": self.passive,
                "arc_deg": self.arc_deg}


def signature_factor(kind: str, target) -> float:
    """How visible a target is to a given sensor kind (0..1.2)."""
    domain = getattr(target, "domain", "land")
    size = getattr(target, "size_class", "medium")   # small | medium | large
    altitude = getattr(target, "altitude_band", "ground")
    if kind == "radar":
        base = {"small": 0.5, "medium": 1.0, "large": 1.4}.get(size, 1.0)
        if domain == "air":
            # Low-flying targets hide in ground clutter / below the radar horizon.
            base *= {"超低空": 0.25, "低空": 0.5, "中空": 0.8, "高空": 1.0}.get(altitude, 1.0)
        if domain == "sea":
            base *= 1.0
        return base
    if kind == "ir":
        return {"small": 0.6, "medium": 0.9, "large": 1.2}.get(size, 1.0)
    if kind == "sonar":
        if getattr(target, "domain", "") != "sea":
            return 0.0
        if getattr(target, "depth", "surface") == "surface":
            return 0.2
        return {"small": 0.6, "medium": 1.0, "large": 1.3}.get(size, 1.0)
    if kind == "esm":
        return 1.0 if getattr(target, "emitting", False) else 0.15
    return 1.0


def _arc_ok(observer, target, sensor: Sensor, hmap: hexmap.HexMap) -> bool:
    if sensor.arc_deg >= 360.0:
        return True
    heading = getattr(observer, "heading", None)
    if heading is None:
        return True
    bearing = hmap.bearing_deg(observer.pos, target.pos)
    delta = abs((bearing - heading + 180.0) % 360.0 - 180.0)
    return delta <= sensor.arc_deg / 2.0


def detects(sensor: Sensor, observer, target, hmap: hexmap.HexMap,
            los: Optional[LineOfSight] = None, jam: float = 0.0) -> bool:
    if observer is target:
        return False
    if not getattr(target, "alive", True):
        return False
    rng_km = sensor.effective_km(target, jam)
    if rng_km <= 0.0:
        return False
    if pos_outside(observer, hmap) or pos_outside(target, hmap):
        # Public fixtures occasionally place an operator on a coordinate that
        # is outside the 100x100 map; treat such units as unreachable rather
        # than crashing the turn loop.
        return False
    d_km = hexmap.distance(observer.pos, target.pos) * hmap.scale_m / 1000.0
    if d_km > rng_km:
        return False
    if not _arc_ok(observer, target, sensor, hmap):
        return False
    los = los or LineOfSight(hmap)
    return los.tactical_visible(observer.pos, target.pos,
                                getattr(target, "altitude_band", "ground"))


def detected_by(state, target) -> List[object]:
    """Living enemy units whose sensors currently hold ``target``."""
    out = []
    jam = float(getattr(state, "jam", 0.0) or 0.0)
    for u in state.units.values():
        if not getattr(u, "alive", True) or u.color == target.color:
            continue
        for s in getattr(u, "sensors", []):
            if detects(s, u, target, state.map, state.los, jam):
                out.append(u)
                break
    return out


def visible_enemies(state, observer) -> List[object]:
    out = []
    jam = float(getattr(state, "jam", 0.0) or 0.0)
    for u in state.units.values():
        if not getattr(u, "alive", True) or u.color == observer.color:
            continue
        for s in getattr(observer, "sensors", []):
            if detects(s, observer, u, state.map, state.los, jam):
                out.append(u)
                break
    return out


# -- default sensor fits -----------------------------------------------------

def sensors_for(name: str, domain: str) -> List[Sensor]:
    """Synthetic sensor fits by platform name.  Documented as approximations."""
    if domain == "sea":
        if "驱逐舰" in name:
            return [Sensor("radar", 320.0, 1.0), Sensor("sonar", 40.0, 0.9),
                    Sensor("esm", 500.0, 1.0, passive=True)]
        if "护卫舰" in name:
            return [Sensor("radar", 220.0, 0.9), Sensor("sonar", 30.0, 0.8)]
        if "补给舰" in name:
            return [Sensor("radar", 80.0, 0.5)]
        if "潜艇" in name:
            return [Sensor("sonar", 60.0, 1.0), Sensor("esm", 200.0, 0.8, passive=True)]
        return [Sensor("radar", 120.0, 0.7)]
    if domain == "air":
        if "预警" in name:
            return [Sensor("radar", 420.0, 1.2)]
        if "电子战" in name or "干扰" in name:
            return [Sensor("esm", 350.0, 1.0, passive=True), Sensor("radar", 150.0, 0.6)]
        if "无人机" in name or "UAV" in name:
            return [Sensor("ir", 120.0, 0.8)]
        if "直升机" in name:
            return [Sensor("radar", 60.0, 0.7), Sensor("ir", 25.0, 0.9)]
        return [Sensor("radar", 160.0, 0.9)]
    # land
    if "防空" in name:
        return [Sensor("radar", 250.0, 1.0)]
    if "侦察" in name or "无人" in name:
        return [Sensor("ir", 80.0, 0.8)]
    if "炮兵" in name:
        return [Sensor("visual", 30.0, 0.6)]
    return [Sensor("visual", 40.0, 0.6), Sensor("radar", 20.0, 0.5)]
