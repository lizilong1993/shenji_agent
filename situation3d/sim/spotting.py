"""Claude-of-Tanks style spotting / concealment model for land combat.

Ported (algorithm only) from ``src/sim/spotting.ts`` in
``Kevin-Liu-01/Claude-of-Tanks`` (MIT, Copyright © 2026 Kevin B. Liu).
The original model is tuned for a 445 m tank battlefield; this port keeps the
same shape (view range minus concealment, firing bloom, cover bonus, radio
sharing) but scales ranges to the geo engine's 200 m rule cells.

The model is intentionally side-effect free: callers pass the environment,
the current time and a line-of-sight callable.  ``sim.world`` uses it to build
the per-observer ``seen`` sets when ``detection_model`` is ``spotting`` or
``shadow``.
"""

from __future__ import annotations

import math
from typing import Dict, Set, Tuple

from env.service import LandCover
from geo.geodesy import haversine_m
from geo.rule_grid import cube_distance

MIN_SPOT_RANGE_M = 50.0
MAX_SPOT_RANGE_M = 5000.0
MAX_VIEW_RANGE_M = 8000.0
FIRE_CAMO_LOSS = 0.82
FIRE_BLOOM_TAU_S = 1.7
BUSH_FIRE_TRANSPARENT_M = 15.0
RADIO_RANGE_M = 600.0

_COVER_BONUS = {
    LandCover.FOREST: 0.35,
    LandCover.URBAN: 0.25,
    LandCover.SOFT: 0.10,
}
_SIZE_BONUS = {"small": 0.10, "medium": 0.0, "large": -0.05}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _observation_row(observer) -> Dict[str, int]:
    from rules.detection import observation_doc
    table = observation_doc()["table"]
    kind = str(getattr(observer, "unit_kind", "vehicle"))
    if kind in ("infantry", "vehicle", "helicopter", "uav"):
        return table.get(kind) or {}
    if kind == "aircraft":
        return table.get("helicopter") or {}
    return table.get("vehicle") or {}


def _target_key(target) -> str:
    kind = str(getattr(target, "unit_kind", "vehicle"))
    unit_type = str(getattr(target, "unit_type", kind))
    if kind == "infantry" or unit_type == "infantry":
        return "infantry"
    if kind in ("uav", "cruise_missile"):
        return "uav"
    if kind == "helicopter":
        return "helicopter"
    return "vehicle"


def view_range_m(observer, cell_m: float = 200.0, target=None) -> float:
    """Effective view range in metres for a land observer.

    Preference order: explicit sensor ranges, the scenario's
    ``observe_distance`` (keyed by target kind, in hex cells), then the public
    observation table.  When ``target`` is given, its kind selects the row;
    otherwise the largest available row is used for prefiltering.
    """
    target_key = _target_key(target) if target is not None else None
    ranges = []
    for sensor in getattr(observer, "sensors", None) or []:
        kind = str(getattr(sensor, "kind", ""))
        if kind not in ("radar", "ir", "optical", "esm"):
            continue
        km = float(getattr(sensor, "range_km", 0.0) or 0.0)
        quality = float(getattr(sensor, "quality", 1.0) or 1.0)
        if km > 0.0:
            ranges.append(km * 1000.0 * quality)
    observe = getattr(observer, "observe_distance", None) or {}
    if target_key is not None and target_key in observe:
        try:
            ranges.append(float(observe[target_key]) * float(cell_m))
        except (TypeError, ValueError):
            pass
    elif target_key is None:
        for value in observe.values():
            try:
                cells = float(value)
            except (TypeError, ValueError):
                continue
            if cells > 0.0:
                ranges.append(cells * float(cell_m))
    row = _observation_row(observer)
    if target_key is not None and target_key in row:
        ranges.append(row[target_key] * float(cell_m))
    elif row:
        ranges.append(max(row.values()) * float(cell_m))
    return _clamp(max(ranges) if ranges else 2000.0, 500.0, MAX_VIEW_RANGE_M)


def base_concealment(target) -> float:
    value = getattr(target, "concealment", None)
    if value is not None:
        return _clamp(float(value), 0.0, 0.90)
    camo = 0.15
    if getattr(target, "concealed", False):
        camo += 0.25
    camo += _SIZE_BONUS.get(str(getattr(target, "size_class", "medium")), 0.0)
    return _clamp(camo, 0.0, 0.90)


def cover_bonus(env, target) -> float:
    try:
        cover = env.landcover(target.pose.lon, target.pose.lat)
    except AttributeError:
        return 0.0
    return _COVER_BONUS.get(cover, 0.0)


def camo_of(target, env, now_s: float, cell_m: float = 200.0) -> float:
    """Total concealment at ``now_s`` (0..0.9).

    Firing removes most of the target's *own* concealment for a few seconds
    (exponential bloom); nearby foliage also stops concealing while the
    muzzle flash is hot (the original 15 m rule, approximated by halving the
    cover bonus for the first bloom window).
    """
    own = base_concealment(target)
    cover = cover_bonus(env, target)
    if getattr(target, "velocity_ms", 0.0) > 0.5:
        own -= 0.10
    fired_at = getattr(target, "last_fired_s", -1.0)
    fired_at = -1.0 if fired_at is None else float(fired_at)
    bloom = 0.0
    if fired_at >= 0.0 and now_s >= fired_at:
        bloom = math.exp(-(now_s - fired_at) / FIRE_BLOOM_TAU_S)
    if bloom > 0.0:
        own *= max(0.0, 1.0 - FIRE_CAMO_LOSS * bloom)
        if bloom > 0.45:
            cover *= 0.5
    return _clamp(own + cover, 0.0, 0.90)


def _bloom_hot(target, now_s: float) -> bool:
    fired_at = getattr(target, "last_fired_s", -1.0)
    if fired_at is None or fired_at < 0.0 or now_s < fired_at:
        return False
    return math.exp(-(now_s - fired_at) / FIRE_BLOOM_TAU_S) >= 0.45


def spot_range_m(view_range: float, camo: float) -> float:
    if view_range <= 0.0:
        return 0.0
    value = view_range - (view_range - MIN_SPOT_RANGE_M) * camo
    high = min(max(view_range, MIN_SPOT_RANGE_M), MAX_SPOT_RANGE_M)
    low = min(MIN_SPOT_RANGE_M, high)
    return _clamp(value, low, high)


def can_spot(observer, target, env, now_s: float, cell_m: float = 200.0,
             line_of_sight=None) -> bool:
    if observer is target or getattr(observer, "uid", None) == getattr(target, "uid", None):
        return False
    if not getattr(target, "alive", True):
        return False
    if getattr(observer, "affiliation", None) == getattr(target, "affiliation", None):
        return False
    if (getattr(observer, "domain", "land") not in ("land", "amphibious")
            or getattr(target, "domain", "land") not in ("land", "amphibious")):
        return False
    distance = haversine_m((observer.pose.lon, observer.pose.lat),
                           (target.pose.lon, target.pose.lat))
    view = view_range_m(observer, cell_m, target)
    if distance > view:
        return False
    if distance <= MIN_SPOT_RANGE_M:
        return True
    camo = camo_of(target, env, now_s, cell_m)
    if distance > spot_range_m(view, camo):
        # Muzzle-flash reveal: a bloom-hot target in the open is visible even
        # when its own camo would still hide it.  Dense cover keeps concealing.
        if not (_bloom_hot(target, now_s) and cover_bonus(env, target) < 0.20):
            return False
    if line_of_sight is not None and not line_of_sight(observer, target):
        return False
    return True


def compute_seen(world, ctx) -> Dict[int, Set[int]]:
    """Per-observer seen sets for land targets (sea/air keep the sensor model)."""
    entities = [e for e in world.sorted_entities() if e.alive and not e.on_board]
    out: Dict[int, Set[int]] = {e.uid: set() for e in entities}
    cell_m = float(getattr(world.rule_grid, "cell_m", 200.0) or 200.0)
    cells = {e.uid: world.rule_cell(e) for e in entities}
    los_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], bool] = {}
    for observer in entities:
        if observer.domain not in ("land", "amphibious"):
            continue
        view = view_range_m(observer, cell_m)
        max_cells = int(math.ceil(view / cell_m)) + 1
        for target in entities:
            if target.domain not in ("land", "amphibious"):
                continue
            if cube_distance(cells[observer.uid], cells[target.uid]) > max_cells:
                continue
            dy = abs(observer.pose.lat - target.pose.lat) * 111320.0
            if dy > view:
                continue
            dx = abs(observer.pose.lon - target.pose.lon) * 111320.0 * math.cos(
                math.radians(observer.pose.lat))
            if dx > view:
                continue
            key = (cells[observer.uid], cells[target.uid])
            visible = los_cache.get(key)
            if visible is None:
                visible = ctx.line_of_sight(ctx.view(observer), ctx.view(target))
                los_cache[key] = bool(visible)
            if can_spot(observer, target, world.env, world.time_s, cell_m,
                        line_of_sight=lambda a, b: bool(visible)):
                out[observer.uid].add(target.uid)
    return out


def merge_seen(table_seen: Dict[int, Set[int]],
               spotting_seen: Dict[int, Set[int]], world) -> Dict[int, Set[int]]:
    """Use spotting for land-land pairs, the table/sensor model for the rest."""
    entities = [e for e in world.sorted_entities() if e.alive and not e.on_board]
    result: Dict[int, Set[int]] = {}
    for observer in entities:
        o_land = observer.domain in ("land", "amphibious")
        result[observer.uid] = set()
        for target in entities:
            if target.uid == observer.uid:
                continue
            t_land = target.domain in ("land", "amphibious")
            source = spotting_seen if (o_land and t_land) else table_seen
            if target.uid in source.get(observer.uid, set()):
                result[observer.uid].add(target.uid)
    return result


def radio_share(world, seen: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    """One-hop contact sharing between compatible units inside radio range."""
    entities = [e for e in world.entities.values() if e.alive and not e.on_board]
    snapshot = {uid: set(contacts) for uid, contacts in seen.items()}
    result = {uid: set(contacts) for uid, contacts in seen.items()}
    for observer in entities:
        contacts = snapshot.get(observer.uid) or set()
        if not contacts:
            continue
        comms = getattr(observer, "comms_range_m", None)
        comms = RADIO_RANGE_M if comms is None else float(comms)
        if comms <= 0.0:
            continue
        for mate in entities:
            if mate.uid == observer.uid or mate.affiliation != observer.affiliation:
                continue
            if (observer.domain in ("land", "amphibious")) != (
                    mate.domain in ("land", "amphibious")):
                continue
            distance = haversine_m((observer.pose.lon, observer.pose.lat),
                                   (mate.pose.lon, mate.pose.lat))
            if distance <= comms:
                result.setdefault(mate.uid, set()).update(contacts)
    return result


def shadow_diff(table_seen: Dict[int, Set[int]],
                spotting_seen: Dict[int, Set[int]]) -> dict:
    """Aggregate comparison between the observation table and spotting model."""
    table_pairs = sum(len(v) for v in table_seen.values())
    spot_pairs = sum(len(v) for v in spotting_seen.values())
    both = 0
    only_table = 0
    only_spot = 0
    samples = []
    for uid in sorted(set(table_seen) | set(spotting_seen)):
        a = set(table_seen.get(uid, set()))
        b = set(spotting_seen.get(uid, set()))
        both += len(a & b)
        diff_a = sorted(a - b)
        diff_b = sorted(b - a)
        only_table += len(diff_a)
        only_spot += len(diff_b)
        if (diff_a or diff_b) and len(samples) < 20:
            samples.append({"observer": uid, "only_table": diff_a,
                            "only_spotting": diff_b})
    return {
        "table_pairs": table_pairs,
        "spotting_pairs": spot_pairs,
        "both": both,
        "only_table": only_table,
        "only_spotting": only_spot,
        "samples": samples,
    }
