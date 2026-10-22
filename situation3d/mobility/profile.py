"""Platform profiles for mobility predicates (demo values, P3).

Values are demonstration estimates, not real equipment data.  The plan's
``profiles/*.json`` registry can replace this module without changing the
mobility code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class PlatformProfile:
    key: str
    domain: str                      # land | amphibious | surface | subsurface | air
    max_speed_ms: float
    accel_ms2: float = 1.0
    decel_ms2: float = 2.0
    turn_rate_dps: float = 30.0
    # terrain -> speed multiplier (0 = impassable)
    terrain_factors: Dict[str, float] = field(default_factory=dict)
    max_gradient_deg: float = 30.0
    max_side_slope_deg: float = 17.0
    wading_depth_m: float = 1.2
    swim_depth_m: float = 2.5
    amphibious: bool = False
    road_bound: bool = False
    # water
    draft_m: float = 0.0
    ukc_m: float = 0.5
    beam_m: float = 0.0
    max_operating_depth_m: float = 0.0
    # air
    min_clearance_m: float = 0.0
    service_ceiling_m: float = 0.0
    agl_m: float = 0.0
    climb_rate_ms: float = 5.0
    descent_rate_ms: float = 5.0
    # road/bridge constraints (0 = unconstrained)
    weight_t: float = 0.0
    vehicle_height_m: float = 0.0
    # combat metadata
    armor_class: int = 0
    size_class: str = "medium"

    def terrain_factor(self, landcover: str) -> float:
        return float(self.terrain_factors.get(landcover, 0.0))


LAND = {
    "open": 1.0, "forest": 0.5, "urban": 0.33, "soft": 0.25,
    "water": 0.0, "rock": 0.0,
}

PROFILES: Dict[str, PlatformProfile] = {
    "infantry": PlatformProfile(
        "infantry", "land", 1.7, 2.0, 4.0, 360.0,
        {"open": 1.0, "forest": 0.8, "urban": 0.9, "soft": 0.7, "water": 0.0, "rock": 0.6},
        max_gradient_deg=45.0, max_side_slope_deg=35.0, wading_depth_m=1.2,
        size_class="small"),
    "tracked": PlatformProfile(
        "tracked", "land", 10.0, 1.2, 2.5, 45.0, dict(LAND),
        max_gradient_deg=30.0, max_side_slope_deg=17.0, wading_depth_m=1.2,
        weight_t=40.0, vehicle_height_m=2.8, beam_m=3.5,
        armor_class=3, size_class="large"),
    "wheeled": PlatformProfile(
        "wheeled", "land", 25.0, 2.0, 4.0, 40.0,
        {"open": 1.0, "forest": 0.2, "urban": 0.33, "soft": 0.1, "water": 0.0, "rock": 0.3},
        max_gradient_deg=20.0, max_side_slope_deg=15.0, wading_depth_m=0.6,
        weight_t=12.0, vehicle_height_m=2.2, beam_m=2.4,
        road_bound=False, armor_class=1, size_class="medium"),
    "artillery": PlatformProfile(
        "artillery", "land", 6.0, 0.8, 2.0, 30.0, dict(LAND),
        max_gradient_deg=20.0, wading_depth_m=0.8,
        weight_t=15.0, vehicle_height_m=2.5, beam_m=2.5, armor_class=0, size_class="large"),
    "amphibious": PlatformProfile(
        "amphibious", "amphibious", 8.0, 1.0, 2.0, 40.0, dict(LAND),
        max_gradient_deg=25.0, wading_depth_m=2.0, swim_depth_m=4.0,
        weight_t=25.0, vehicle_height_m=3.0, beam_m=3.2,
        amphibious=True, armor_class=2, size_class="large"),
    "ship": PlatformProfile(
        "ship", "surface", 15.0, 0.5, 1.0, 3.0,
        {"water": 1.0}, draft_m=5.0, ukc_m=0.5, beam_m=15.0,
        size_class="large"),
    "submarine": PlatformProfile(
        "submarine", "subsurface", 8.0, 0.3, 0.8, 2.0,
        {"water": 1.0}, draft_m=6.0, ukc_m=0.5, beam_m=8.0,
        max_operating_depth_m=300.0, size_class="large"),
    "helicopter": PlatformProfile(
        "helicopter", "air", 70.0, 2.0, 3.0, 20.0,
        {"open": 1.0, "forest": 1.0, "urban": 1.0, "soft": 1.0, "water": 1.0, "rock": 1.0},
        min_clearance_m=20.0, service_ceiling_m=3000.0, agl_m=50.0,
        climb_rate_ms=8.0, size_class="medium"),
    "uav": PlatformProfile(
        "uav", "air", 50.0, 1.5, 2.0, 15.0,
        {"open": 1.0, "forest": 1.0, "urban": 1.0, "soft": 1.0, "water": 1.0, "rock": 1.0},
        min_clearance_m=30.0, service_ceiling_m=4000.0, agl_m=200.0,
        climb_rate_ms=5.0, size_class="small"),
    "aircraft": PlatformProfile(
        "aircraft", "air", 250.0, 3.0, 4.0, 5.0,
        {"open": 1.0, "forest": 1.0, "urban": 1.0, "soft": 1.0, "water": 1.0, "rock": 1.0},
        min_clearance_m=150.0, service_ceiling_m=12000.0, agl_m=1000.0,
        climb_rate_ms=15.0, size_class="large"),
}


def profile_for(unit_type: str, domain: str) -> PlatformProfile:
    if unit_type in ("infantry",):
        return PROFILES["infantry"]
    if domain in ("sea", "surface"):
        return PROFILES["ship"]
    if domain in ("sub", "subsurface"):
        return PROFILES["submarine"]
    if domain == "air":
        if unit_type in ("helicopter",):
            return PROFILES["helicopter"]
        if unit_type in ("uav", "cruise_missile"):
            return PROFILES["uav"]
        return PROFILES["aircraft"]
    if unit_type == "artillery":
        return PROFILES["artillery"]
    if unit_type == "amphibious":
        return PROFILES["amphibious"]
    return PROFILES["tracked"]
