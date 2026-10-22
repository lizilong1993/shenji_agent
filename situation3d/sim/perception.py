"""Physical 3D perception (P1.4): geometry rays, sensor FoV, probability.

Replaces the observation-table/ad-hoc spotting chain when
``detection_model="physical"``:

* line of sight is a 3D ray against the terrain raster, real building
  footprints (``env.building_intersects``) and scenario obscurants (smoke);
* every sensor has range, azimuth/elevation FoV and a distance falloff;
* detection probability also depends on target signature (RCS/IR/acoustic
  from the platform data v1 records), visibility/sea state, cover and motion;
* a deterministic per-pair roll (stable hash of seed/bucket/uids) decides
  detection, with a short hold so contacts do not flicker every tick;
* detections feed simple per-side tracks (fusion by target, TTL, confidence).

The legacy table/spotting models stay untouched and remain the default.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from env.service import LandCover
from geo.geodesy import haversine_m, initial_bearing_deg

from mobility.kinematics import wrap180 as _wrap180


_KIND_ALIAS = {
    "optics": "optics", "optical": "optics", "eo": "optics", "eo_ir": "ir",
    "ir": "ir", "infrared": "ir", "radar": "radar", "sonar": "sonar",
    "esm": "esm",
}

_KINDS = {
    "optics": dict(fov_deg=60.0, elev_fov_deg=35.0, p0=0.90, scale_frac=0.8),
    "ir": dict(fov_deg=45.0, elev_fov_deg=30.0, p0=0.85, scale_frac=0.9),
    "radar": dict(fov_deg=360.0, elev_fov_deg=90.0, p0=0.95, scale_frac=1.0),
    "sonar": dict(fov_deg=360.0, elev_fov_deg=90.0, p0=0.80, scale_frac=0.9),
    "esm": dict(fov_deg=360.0, elev_fov_deg=90.0, p0=0.70, scale_frac=1.0),
}

_SMOKE_ATTENUATION = 0.02
_HOLD_S = 60.0
_TRACK_TTL_S = 300.0


@dataclass(frozen=True)
class SensorSpec(object):
    kind: str
    range_m: float
    fov_deg: float
    elev_fov_deg: float
    p0: float
    scale_m: float
    mount_deg: float = 0.0
    passive: bool = False


def _kind(name) -> str:
    return _KIND_ALIAS.get(str(name or "").strip().lower(), "optics")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _record_for(entity) -> Optional[dict]:
    profile = getattr(entity, "profile", None)
    key = getattr(profile, "key", None)
    if not key:
        return None
    return _record_for_key(str(key))


_RECORD_CACHE: Dict[str, Optional[dict]] = {}


def _record_for_key(key: str) -> Optional[dict]:
    if key in _RECORD_CACHE:
        return _RECORD_CACHE[key]
    record = None
    try:
        from mobility import platform_data
        record = platform_data.record_by_key(key)
    except Exception:
        record = None
    _RECORD_CACHE[key] = record
    return record


def specs_for(entity) -> List[SensorSpec]:
    """Sensor specs: entity sensors first, platform data v1 as fallback."""
    specs: List[SensorSpec] = []
    seen = set()
    try:
        sensor_health = float((getattr(entity, "components", {}) or {}).get(
            "sensors", 1.0))
    except (TypeError, ValueError):
        sensor_health = 1.0
    if sensor_health <= 0.05:
        return []
    sensor_health = max(0.2, sensor_health)

    def _key(kind, range_m):
        return (kind, round(float(range_m) / 100.0))

    for sensor in getattr(entity, "sensors", None) or []:
        kind = _kind(getattr(sensor, "kind", ""))
        range_km = float(getattr(sensor, "range_km", 0.0) or 0.0)
        quality = float(getattr(sensor, "quality", 1.0) or 1.0)
        range_m = range_km * 1000.0 * max(0.0, quality) * sensor_health
        key = _key(kind, range_m)
        if range_m <= 0.0 or key in seen:
            continue
        defaults = _KINDS[kind]
        specs.append(SensorSpec(
            kind=kind, range_m=range_m,
            fov_deg=float(getattr(sensor, "fov_deg", defaults["fov_deg"])),
            elev_fov_deg=float(defaults["elev_fov_deg"]),
            p0=float(defaults["p0"]),
            scale_m=float(range_m * defaults["scale_frac"]),
            mount_deg=float(getattr(sensor, "mount_deg", 0.0) or 0.0),
            passive=bool(getattr(sensor, "passive", False))))
        seen.add(key)
    if specs:
        return specs
    record = _record_for(entity) or {}
    for item in record.get("sensors") or []:
        kind = _kind(item.get("kind"))
        range_m = float(item.get("range_m") or 0.0)
        if range_m <= 0.0:
            range_m = float(item.get("range_km") or 0.0) * 1000.0
        range_m *= sensor_health
        key = _key(kind, range_m)
        if range_m <= 0.0 or key in seen:
            continue
        defaults = _KINDS[kind]
        fov = float(item.get("fov_deg", defaults["fov_deg"]))
        specs.append(SensorSpec(
            kind=kind, range_m=range_m, fov_deg=fov,
            elev_fov_deg=float(item.get("elev_fov_deg",
                                        defaults["elev_fov_deg"])),
            p0=float(item.get("p0", defaults["p0"])),
            scale_m=float(item.get("scale_m", range_m * defaults["scale_frac"])),
            mount_deg=float(item.get("mount_deg", 0.0)),
            passive=bool(item.get("passive", kind == "esm"))))
        seen.add(key)
    return specs


# -- geometry -------------------------------------------------------------
def los_blocked(env, a: Tuple[float, float], ha: float,
                b: Tuple[float, float], hb: float,
                step_m: float = 30.0, check_buildings: bool = True,
                skip_terrain: bool = False) -> bool:
    """3D terrain + building ray between two points."""
    total = haversine_m(a, b)
    if total <= 1.0:
        return False
    if not skip_terrain:
        step = max(30.0, float(step_m), total / 120.0)
        n = max(1, int(math.ceil(total / step)))
        for i in range(1, n):
            f = i / float(n)
            lon = a[0] + (b[0] - a[0]) * f
            lat = a[1] + (b[1] - a[1]) * f
            sight = ha + (hb - ha) * f
            if env.elevation(lon, lat) + 0.5 > sight:
                return True
    if not check_buildings:
        return False
    intersect = getattr(env, "building_intersects", None)
    if intersect is not None:
        try:
            if intersect((a[0], a[1]), ha, (b[0], b[1]), hb):
                return True
        except Exception:
            # Fail closed: a broken footprint test must not open the map.
            return True
    return False


def _smoke_factor(kind: str, a, ha, b, hb,
                  obscurants: Iterable[dict], now_s: float) -> float:
    if kind not in ("optics", "ir") or not obscurants:
        return 1.0
    kx = 111320.0 * math.cos(math.radians((a[1] + b[1]) / 2.0))
    ky = 110540.0
    ax, ay = 0.0, 0.0
    bx, by = (b[0] - a[0]) * kx, (b[1] - a[1]) * ky
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    factor = 1.0
    for smoke in obscurants:
        t_from = float(smoke.get("from_s", 0.0))
        t_to = float(smoke.get("to_s", 1e18))
        if now_s < t_from or now_s > t_to:
            continue
        cx = (float(smoke["lon"]) - a[0]) * kx
        cy = (float(smoke["lat"]) - a[1]) * ky
        if seg2 <= 1e-9:
            t = 0.0
        else:
            t = _clamp((cx * dx + cy * dy) / seg2, 0.0, 1.0)
        px, py = dx * t, dy * t
        if math.hypot(cx - px, cy - py) > float(smoke.get("radius_m", 0.0)):
            continue
        height = ha + (hb - ha) * t
        base = float(smoke.get("base_m", 0.0))
        top = float(smoke.get("top_m", 200.0))
        if base <= height <= top:
            factor *= _clamp(
                float(smoke.get("attenuation", _SMOKE_ATTENUATION)),
                0.0, 1.0)
    return factor


# -- probability ----------------------------------------------------------
def _signature_factor(kind: str, target) -> float:
    record = _record_for(target) or {}
    sig = record.get("signature") or {}
    size = str(getattr(target, "size_class", "medium"))
    if kind == "radar":
        rcs = float(sig.get("rcs_m2",
                            {"small": 1.0, "medium": 10.0,
                             "large": 100.0}.get(size, 10.0)))
        return _clamp(math.sqrt(max(0.01, rcs) / 10.0), 0.3, 2.5)
    if kind == "ir":
        return _clamp(0.4 + 1.2 * float(sig.get("ir", 1.0)), 0.4, 1.6)
    if kind == "sonar":
        return _clamp(float(sig.get("acoustic", 1.0)), 0.2, 1.5)
    if kind == "esm":
        return 1.0 if getattr(target, "emitting", True) else 0.1
    return {"small": 0.7, "medium": 1.0, "large": 1.3}.get(size, 1.0)


def _env_factor(kind: str, distance_m: float, env, observer) -> float:
    if kind == "esm":
        return 1.0  # electromagnetic emissions do not care about weather
    try:
        weather = env.weather(observer.pose.lon, observer.pose.lat)
    except Exception:
        return 1.0
    visibility = getattr(weather, "visibility_m", None)
    visibility = 10000.0 if visibility is None else float(visibility)
    if kind in ("optics", "ir"):
        if visibility <= 0.0:
            return 0.0
        if visibility >= distance_m:
            return 1.0
        return _clamp(visibility / max(1.0, distance_m), 0.05, 1.0)
    if kind == "sonar":
        sea = int(getattr(weather, "sea_state", 0) or 0)
        return _clamp(1.0 - 0.08 * sea, 0.5, 1.0)
    return 0.95


def _concealment_factor(kind: str, target, env) -> float:
    if kind in ("radar", "sonar", "esm"):
        return 1.0
    cover = 0.0
    if getattr(target, "concealed", False):
        cover += 0.4
    explicit = getattr(target, "concealment", None)
    if explicit is not None:
        try:
            cover += _clamp(float(explicit), 0.0, 0.9)
        except (TypeError, ValueError):
            pass
    try:
        landcover = env.landcover(target.pose.lon, target.pose.lat)
    except Exception:
        landcover = LandCover.OPEN
    cover += {LandCover.FOREST: 0.35, LandCover.URBAN: 0.25,
              LandCover.SOFT: 0.1}.get(landcover, 0.0)
    return _clamp(1.0 - 0.7 * cover, 0.15, 1.0)


def _fov_ok(spec: SensorSpec, observer, target,
            distance_m: float = 0.0) -> bool:
    if spec.fov_deg >= 359.9:
        return True
    bearing = initial_bearing_deg((observer.pose.lon, observer.pose.lat),
                                  (target.pose.lon, target.pose.lat))
    heading = float(getattr(observer.pose, "heading_deg", 0.0)) + spec.mount_deg
    if abs(_wrap180(bearing - heading)) > spec.fov_deg / 2.0:
        return False
    if spec.elev_fov_deg < 180.0:
        dist = max(1.0, float(distance_m) or haversine_m(
            (observer.pose.lon, observer.pose.lat),
            (target.pose.lon, target.pose.lat)))
        dz = (float(target.pose.alt_m) - float(observer.pose.alt_m))
        elev = math.degrees(math.atan2(dz, dist))
        if abs(elev) > spec.elev_fov_deg / 2.0:
            return False
    return True


def _sensor_height(entity) -> float:
    """Sensor height above the surface (ships carry their masts)."""
    alt = float(entity.pose.alt_m)
    if entity.domain in ("land", "amphibious"):
        return alt + 2.0
    if entity.domain == "surface":
        return max(0.0, alt) + 20.0
    if entity.domain == "subsurface":
        return max(0.0, alt) + 10.0
    return alt


def _target_height(entity) -> float:
    alt = float(entity.pose.alt_m)
    if entity.domain in ("land", "amphibious"):
        return alt + 1.5
    if entity.domain == "surface":
        return max(0.0, alt) + 10.0
    if entity.domain == "subsurface":
        return max(0.0, alt) + 1.0
    return alt


def _horizon_m(observer, target) -> float:
    """Radar/optical horizon (4/3 earth radius approximation).

    4.12 * (sqrt(h_obs) + sqrt(h_target)) km for heights in metres.
    """
    h1 = max(0.0, _sensor_height(observer))
    h2 = max(0.0, _target_height(target))
    return 4120.0 * (math.sqrt(h1) + math.sqrt(h2))


def _skip_terrain(observer, target, distance_m: float) -> bool:
    """Long sea/air engagements ignore terrain masking (horizon dominates)."""
    if distance_m <= 20000.0:
        return False
    return (observer.domain not in ("land", "amphibious")
            and target.domain not in ("land", "amphibious"))


def detection_probability(spec: SensorSpec, observer, target,
                          distance_m: float, env, now_s: float = 0.0,
                          obscurants: Iterable[dict] = (),
                          geometry_ok: Optional[bool] = None) -> float:
    if not getattr(target, "alive", True):
        return 0.0
    if distance_m < 0.0 or distance_m > spec.range_m:
        return 0.0
    target_domain = getattr(target, "domain", "land")
    if spec.kind == "sonar" and target_domain not in ("surface", "subsurface"):
        return 0.0
    if (spec.kind in ("radar", "optics", "ir")
            and target_domain == "subsurface"):
        return 0.0
    if not _fov_ok(spec, observer, target, distance_m):
        return 0.0
    a = (observer.pose.lon, observer.pose.lat)
    b = (target.pose.lon, target.pose.lat)
    ha = _sensor_height(observer)
    hb = _target_height(target)
    if spec.kind != "sonar":
        if distance_m > _horizon_m(observer, target):
            return 0.0
        if geometry_ok is None:
            geometry_ok = not los_blocked(
                env, a, ha, b, hb,
                check_buildings=distance_m <= 20000.0,
                skip_terrain=_skip_terrain(observer, target, distance_m))
        if not geometry_ok:
            return 0.0
    p = spec.p0 * math.exp(-max(1.0, distance_m) / max(1.0, spec.scale_m))
    p *= _signature_factor(spec.kind, target)
    p *= _env_factor(spec.kind, distance_m, env, observer)
    p *= _smoke_factor(spec.kind, a, ha, b, hb, obscurants, now_s)
    p *= _concealment_factor(spec.kind, target, env)
    if float(getattr(target, "velocity_ms", 0.0)) > 0.5:
        p *= 1.1
    return _clamp(p, 0.0, 1.0)


def detection_roll(seed, bucket: int, observer_uid: int, target_uid: int) -> float:
    payload = "%s:%s:%s:%s" % (seed, bucket, observer_uid, target_uid)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 18446744073709551616.0


# -- world integration ----------------------------------------------------
def compute_seen(world) -> Dict[int, set]:
    """Per-observer seen sets for the physical model (updates tracks)."""
    entities = [e for e in world.sorted_entities()
                if e.alive and not e.on_board]
    seen: Dict[int, set] = {}
    now = float(world.time_s)
    seed = getattr(world, "seed", 0)
    bucket = int(now // 10.0)
    last_seen: Dict[Tuple[int, int], float] = world._phys_last_seen
    tracks: Dict[int, dict] = world.tracks
    obscurants = getattr(world, "obscurants", ())
    for observer in entities:
        specs = specs_for(observer)
        if not specs:
            seen[observer.uid] = set()
            continue
        max_range = max(spec.range_m for spec in specs)
        visible = set()
        for target in entities:
            if target.uid == observer.uid:
                continue
            if target.affiliation == observer.affiliation:
                continue
            distance = haversine_m(
                (observer.pose.lon, observer.pose.lat),
                (target.pose.lon, target.pose.lat))
            if distance > max_range:
                continue
            geometry_ok = None
            if any(spec.kind != "sonar" for spec in specs):
                geometry_ok = (distance <= _horizon_m(observer, target)
                               and not los_blocked(
                                   world.env,
                                   (observer.pose.lon, observer.pose.lat),
                                   _sensor_height(observer),
                                   (target.pose.lon, target.pose.lat),
                                   _target_height(target),
                                   check_buildings=distance <= 20000.0,
                                   skip_terrain=_skip_terrain(
                                       observer, target, distance)))
            p_total = 1.0
            kinds = set()
            for spec in specs:
                p = detection_probability(
                    spec, observer, target, distance, world.env, now,
                    obscurants,
                    geometry_ok=geometry_ok if spec.kind != "sonar" else None)
                if p <= 0.0:
                    continue
                p_total *= (1.0 - p)
                if p >= 0.25:
                    kinds.add(spec.kind)
            p_total = 1.0 - p_total
            key = (observer.uid, target.uid)
            fresh = (p_total > 0.0
                     and detection_roll(seed, bucket, observer.uid,
                                        target.uid) < p_total)
            held = (p_total > 0.0 and last_seen.get(key) is not None
                    and now - last_seen[key] <= _HOLD_S)
            if not fresh and not held:
                continue
            visible.add(target.uid)
            if fresh:
                last_seen[key] = now
                _update_track(tracks, observer.affiliation, target, now,
                              p_total, kinds)
        seen[observer.uid] = visible
    for key in [k for k, ts in last_seen.items() if now - ts > _HOLD_S * 2]:
        del last_seen[key]
    _prune_tracks(tracks, now)
    return seen


def _update_track(tracks: Dict[int, dict], side: int, target, now: float,
                  confidence: float, kinds) -> None:
    side_tracks = tracks.setdefault(int(side), {})
    track = side_tracks.get(target.uid)
    if track is None:
        track = {
            "uid": target.uid,
            "name": target.name,
            "domain": target.domain,
            "affiliation": target.affiliation,
            "first_seen_s": now,
            "last_seen_s": now,
            "confidence": 0.0,
            "lon": target.pose.lon,
            "lat": target.pose.lat,
            "sensors": set(),
        }
        side_tracks[target.uid] = track
    track["last_seen_s"] = now
    track["lon"] = target.pose.lon
    track["lat"] = target.pose.lat
    track["confidence"] = max(float(track["confidence"]), float(confidence))
    if kinds:
        track["sensors"].update(kinds)


def _prune_tracks(tracks: Dict[int, dict], now: float) -> None:
    for side in list(tracks):
        side_tracks = tracks[side]
        for uid in [u for u, t in side_tracks.items()
                    if now - t["last_seen_s"] > _TRACK_TTL_S]:
            del side_tracks[uid]
        if not side_tracks:
            tracks.pop(side, None)
