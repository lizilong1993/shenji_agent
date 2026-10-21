"""Hard passability predicates and soft speed factors (P3).

Two levels, per plan §3.6:
* ``passable`` -- can the platform be here at all (water depth, slope,
  buildings, air clearance, ship draft);
* ``speed_factor`` -- how fast can it move given terrain/slope/weather.
"""

from __future__ import annotations

import math
from typing import Optional

from env.service import LandCover
from mobility.profile import PlatformProfile

INF = float("inf")


def passable(profile: PlatformProfile, lon: float, lat: float, env,
             altitude_m: Optional[float] = None) -> bool:
    cover = env.landcover(lon, lat)
    depth = env.water_depth(lon, lat)

    if profile.domain == "air":
        alt = altitude_m if altitude_m is not None else env.elevation(lon, lat) + profile.agl_m
        if alt < env.elevation(lon, lat) + profile.min_clearance_m:
            return False
        if profile.service_ceiling_m > 0 and alt > profile.service_ceiling_m:
            return False
        return True

    if profile.domain in ("surface", "subsurface"):
        if depth is None:
            return False
        if depth < profile.draft_m + profile.ukc_m:
            return False
        if profile.domain == "subsurface" and depth > profile.max_operating_depth_m > 0:
            return False
        return True

    if profile.domain == "amphibious":
        if depth is not None:
            return depth <= profile.swim_depth_m
        return _land_ok(profile, lon, lat, env, cover)

    # land: shallow water may be waded even though its landcover is "water"
    if depth is not None:
        if depth > profile.wading_depth_m:
            return False
        if env.slope_deg(lon, lat) > profile.max_gradient_deg:
            return False
        return True
    return _land_ok(profile, lon, lat, env, cover)


def _land_ok(profile: PlatformProfile, lon: float, lat: float, env, cover: str) -> bool:
    factor = profile.terrain_factor(cover)
    if factor <= 0.0:
        # Road-bound platforms may enter a cell only if it carries a road.
        return profile.road_bound and env.on_road(lon, lat)
    if env.slope_deg(lon, lat) > profile.max_gradient_deg:
        return False
    if cover == LandCover.URBAN and profile.road_bound and not env.on_road(lon, lat):
        return False
    return True


def speed_factor(profile: PlatformProfile, lon: float, lat: float, env,
                 altitude_m: Optional[float] = None) -> float:
    if profile.domain == "air":
        return 1.0
    if profile.domain in ("surface", "subsurface"):
        return 1.0
    cover = env.landcover(lon, lat)
    depth = env.water_depth(lon, lat)
    if depth is not None:
        if profile.domain == "amphibious":
            return 0.5 if depth <= profile.swim_depth_m else 0.0
        if depth <= profile.wading_depth_m:
            return 0.35
        return 0.0
    f = profile.terrain_factor(cover)
    if f <= 0.0:
        return 0.0
    slope = env.slope_deg(lon, lat)
    if profile.max_gradient_deg > 0:
        f *= max(0.1, 1.0 - 0.5 * min(1.0, slope / profile.max_gradient_deg))
    if profile.road_bound and env.on_road(lon, lat):
        f = 1.0
    elif env.on_road(lon, lat) and profile.domain == "land":
        f = min(1.0, f * 1.25)
    return f


def effective_speed_ms(profile: PlatformProfile, lon: float, lat: float, env,
                       altitude_m: Optional[float] = None) -> float:
    return profile.max_speed_ms * speed_factor(profile, lon, lat, env, altitude_m)
