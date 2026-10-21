"""Entity/route model for the continuous geo engine (P2 skeleton).

Positions are WGS84 ``(lon, lat, alt_m)``; routes are polyline polylines with
per-segment speed limits.  Physical integration and collision land in P3; P2
only fixes the data contract so rules and adapters can be written against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from geo.geodesy import haversine_m, interpolate


@dataclass
class Pose(object):
    lon: float
    lat: float
    alt_m: float = 0.0
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def as_tuple(self):
        return (self.lon, self.lat)


@dataclass
class EdgeRef(object):
    edge_id: str
    mode: str = "road"


@dataclass
class Route(object):
    points: List[Pose]
    speed_limits: List[float]
    modes: List[str]
    edges: List[EdgeRef] = field(default_factory=list)
    cum_m: List[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError("Route needs at least two points")
        if not self.cum_m:
            self.cum_m = [0.0]
            for i in range(1, len(self.points)):
                a = self.points[i - 1].as_tuple()
                b = self.points[i].as_tuple()
                self.cum_m.append(self.cum_m[-1] + haversine_m(a, b))
        if not self.speed_limits:
            self.speed_limits = [10.0] * (len(self.points) - 1)
        if not self.modes:
            self.modes = ["offroad"] * (len(self.points) - 1)

    @property
    def total_m(self) -> float:
        return self.cum_m[-1]

    def _segment(self, s: float):
        s = max(0.0, min(s, self.total_m))
        for i in range(1, len(self.cum_m)):
            if s <= self.cum_m[i] or i == len(self.cum_m) - 1:
                seg_len = self.cum_m[i] - self.cum_m[i - 1]
                frac = 0.0 if seg_len <= 1e-9 else (s - self.cum_m[i - 1]) / seg_len
                return i - 1, min(max(frac, 0.0), 1.0)
        return 0, 0.0

    def pose_at(self, s: float) -> Pose:
        i, f = self._segment(s)
        a, b = self.points[i], self.points[i + 1]
        lon, lat = interpolate(a.as_tuple(), b.as_tuple(), f)
        alt = a.alt_m + (b.alt_m - a.alt_m) * f
        heading = math.degrees(
            math.atan2(b.lon - a.lon, b.lat - a.lat)) % 360.0 if a.lat != b.lat or a.lon != b.lon else a.heading_deg
        return Pose(lon=lon, lat=lat, alt_m=alt, heading_deg=heading)

    def speed_at(self, s: float) -> float:
        i, _ = self._segment(s)
        return float(self.speed_limits[i])

    def mode_at(self, s: float) -> str:
        i, _ = self._segment(s)
        return self.modes[i]


@dataclass
class Entity(object):
    uid: int
    name: str
    affiliation: int                 # 0 red, 1 blue, -1 neutral
    domain: str                      # land | air | surface | subsurface | amphibious
    profile_id: str
    pose: Pose
    unit_type: str = "vehicle"       # infantry|vehicle|artillery|aa|...
    unit_kind: str = "vehicle"       # infantry|vehicle|recon|aa|helicopter|...
    velocity_ms: float = 0.0
    vertical_ms: float = 0.0
    route: Optional[Route] = None
    route_s: float = 0.0
    blood: int = 1
    max_blood: int = 1
    armor: int = 0
    ammo: Dict[str, int] = field(default_factory=dict)
    fuel: float = 1.0
    suppressed: bool = False
    suppressed_s: float = 0.0
    concealed: bool = False
    moving: bool = False
    marching: bool = False
    mobility: float = 1.0
    value: int = 0
    readiness: float = 0.85
    emitting: bool = True
    altitude_band: str = "ground"
    depth: str = "surface"
    stacked: bool = False
    on_board: bool = False
    carrier: Optional[int] = None
    passengers: List[int] = field(default_factory=list)
    in_fort: Optional[int] = None
    alive: bool = True

    def to_rule_view(self):
        """Project to the read-only view consumed by ``rules`` (no dependency)."""
        return {
            "uid": self.uid,
            "name": self.name,
            "affiliation": self.affiliation,
            "domain": self.domain,
            "unit_type": self.unit_type,
            "unit_kind": self.unit_kind,
            "alive": self.alive,
            "armor": self.armor,
            "blood": self.blood,
            "max_blood": self.max_blood,
            "concealed": self.concealed,
            "moving": self.moving,
            "stacked": self.stacked,
            "marching": self.marching,
            "suppressed": self.suppressed,
            "readiness": self.readiness,
            "value": self.value,
            "ammo": dict(self.ammo),
        }
