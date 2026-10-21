"""Sensor model for sea/air domains (ported, synthetic values).

Land-vs-land observation stays on the public 观察规则 table; radar/IR/sonar/
ESM are used when either side is at sea, underwater or in the air, where the
hex observation table does not apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from env.service import LandCover


@dataclass(frozen=True)
class Sensor:
    kind: str
    range_km: float
    quality: float = 1.0
    passive: bool = False

    def effective_km(self, target) -> float:
        return max(0.0, self.range_km * self.quality * signature_factor(self.kind, target))


def signature_factor(kind: str, target) -> float:
    domain = getattr(target, "domain", "land")
    size = getattr(target, "size_class", "medium")
    if kind == "radar":
        base = {"small": 0.5, "medium": 1.0, "large": 1.4}.get(size, 1.0)
        if domain == "air":
            base *= 0.8
        return base
    if kind == "ir":
        return {"small": 0.6, "medium": 0.9, "large": 1.2}.get(size, 1.0)
    if kind == "sonar":
        if domain not in ("surface", "subsurface"):
            return 0.0
        return 0.2 if getattr(target, "depth", "surface") == "surface" else 1.0
    if kind == "esm":
        return 1.0 if getattr(target, "emitting", True) else 0.15
    return 1.0


def sensors_for(name: str, domain: str) -> List[Sensor]:
    if domain == "amphibious":
        return []
    if "岸舰" in name:
        return [Sensor("radar", 200.0, 0.9)]
    if domain == "surface":
        if "驱逐舰" in name:
            return [Sensor("radar", 320.0), Sensor("sonar", 40.0, 0.9),
                    Sensor("esm", 500.0, 1.0, True)]
        if "护卫舰" in name:
            return [Sensor("radar", 220.0, 0.9), Sensor("sonar", 30.0, 0.8)]
        if "补给" in name:
            return [Sensor("radar", 80.0, 0.5)]
        if "潜艇" in name:
            return [Sensor("sonar", 60.0), Sensor("esm", 200.0, 0.8, True)]
        return [Sensor("radar", 120.0, 0.7)]
    if domain == "subsurface":
        return [Sensor("sonar", 60.0), Sensor("esm", 200.0, 0.8, True)]
    if domain == "air":
        if "预警" in name:
            return [Sensor("radar", 420.0, 1.2)]
        if "电子战" in name or "干扰" in name:
            return [Sensor("esm", 350.0, 1.0, True), Sensor("radar", 150.0, 0.6)]
        if "无人机" in name:
            return [Sensor("ir", 120.0, 0.8)]
        if "直升机" in name:
            return [Sensor("radar", 60.0, 0.7), Sensor("ir", 25.0, 0.9)]
        return [Sensor("radar", 160.0, 0.9)]
    if "防空" in name:
        return [Sensor("radar", 250.0)]
    if "侦察" in name or "无人" in name:
        return [Sensor("ir", 80.0, 0.8)]
    return []


def detects(sensor: Sensor, observer, target, env) -> bool:
    if not getattr(target, "alive", True):
        return False
    rng_km = sensor.effective_km(target)
    if rng_km <= 0.0:
        return False
    from geo.geodesy import haversine_m
    d_km = haversine_m((observer.pose.lon, observer.pose.lat),
                       (target.pose.lon, target.pose.lat)) / 1000.0
    if d_km > rng_km:
        return False
    ha = observer.pose.alt_m + 2.0
    hb = target.pose.alt_m + 1.5
    # Coarse LOS for long-range radar (at most ~20 samples); fine terrain
    # sampling is unnecessary at sea/air scales.
    step = max(1000.0, d_km * 1000.0 / 20.0)
    return env.line_of_sight((observer.pose.lon, observer.pose.lat), ha,
                             (target.pose.lon, target.pose.lat), hb,
                             step_m=step)
