"""Environment service contract and a flat stub (P2 skeleton).

P1 will back this protocol with envpack tiles (DEM, landcover, buildings,
water, road graph).  P2 ships the interface plus ``FlatEnvironment`` so the
skeleton and rule tests can run without data.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Protocol, Tuple


class LandCover(object):
    OPEN = "open"
    FOREST = "forest"
    URBAN = "urban"
    SOFT = "soft"
    WATER = "water"
    BEACH = "beach"
    ROCK = "rock"


class Weather(object):
    def __init__(self, visibility_m: float = 10000.0, wind_ms: float = 0.0,
                 sea_state: int = 0):
        self.visibility_m = visibility_m
        self.wind_ms = wind_ms
        self.sea_state = sea_state


class Building(object):
    def __init__(self, footprint: Optional[list] = None, height_m: float = 0.0):
        self.footprint = footprint or []
        self.height_m = height_m


class EnvironmentService(Protocol):
    def elevation(self, lon: float, lat: float) -> float: ...
    def slope_deg(self, lon: float, lat: float, bearing_deg: float = 0.0) -> float: ...
    def landcover(self, lon: float, lat: float) -> str: ...
    def water_depth(self, lon: float, lat: float) -> Optional[float]: ...
    def building_at(self, lon: float, lat: float) -> Optional[Building]: ...
    def cover_height(self, lon: float, lat: float) -> float: ...
    def building_intersects(self, a: Tuple[float, float], h0: float,
                            b: Tuple[float, float], h1: float) -> bool: ...
    def weather(self, lon: float, lat: float) -> Weather: ...
    def sample_line(self, a: Tuple[float, float], b: Tuple[float, float],
                    step_m: float = 30.0) -> list: ...


class FlatEnvironment(object):
    """Constant-terrain stub: elevation 0, open ground, no water/buildings."""

    def __init__(self, elevation_m: float = 0.0):
        self._elev = float(elevation_m)
        self._weather = Weather()

    def elevation(self, lon: float, lat: float) -> float:
        return self._elev

    def slope_deg(self, lon: float, lat: float, bearing_deg: float = 0.0) -> float:
        return 0.0

    def landcover(self, lon: float, lat: float) -> str:
        return LandCover.OPEN

    def water_depth(self, lon: float, lat: float) -> Optional[float]:
        return None

    def building_at(self, lon: float, lat: float) -> Optional[Building]:
        return None

    def cover_height(self, lon: float, lat: float) -> float:
        return 0.0

    def building_intersects(self, a: Tuple[float, float], h0: float,
                            b: Tuple[float, float], h1: float) -> bool:
        return False

    def weather(self, lon: float, lat: float) -> Weather:
        return self._weather

    def sample_line(self, a: Tuple[float, float], b: Tuple[float, float],
                    step_m: float = 30.0) -> list:
        from geo.geodesy import haversine_m, interpolate
        total = haversine_m(a, b)
        n = max(1, int(math.ceil(total / step_m)))
        out = []
        for i in range(n + 1):
            lon, lat = interpolate(a, b, i / float(n))
            out.append({"lon": lon, "lat": lat,
                        "elevation": self.elevation(lon, lat),
                        "cover_height": self.cover_height(lon, lat),
                        "fraction": i / float(n)})
        return out
