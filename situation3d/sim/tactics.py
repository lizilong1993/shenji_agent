"""Deterministic tactical controller for ground units.

This is a compact port of the *behaviour shape* of Claude-of-Tanks
``src/game/ai.ts`` (MIT, Copyright © 2026 Kevin B. Liu): objective first,
cover when engaged, fire only from a sensible position, retreat when badly
damaged.  It is not a line-by-line port of the 6k-line original; it gives the
geo engine a working tactical layer that can be enabled per scenario and
extended later.

The controller only issues movement/stop orders through ``World.order``; it
never fires or applies damage itself, so the existing engagement rules stay
authoritative.
"""

from __future__ import annotations

from env.service import LandCover
from geo.geodesy import haversine_m


def _is_ground(entity) -> bool:
    return getattr(entity, "domain", "land") in ("land", "amphibious")


def _in_cover(world, entity) -> bool:
    try:
        cover = world.env.landcover(entity.pose.lon, entity.pose.lat)
    except AttributeError:
        return False
    return cover in (LandCover.FOREST, LandCover.URBAN)


def _nearest_entity(world, entity, candidates):
    ranked = []
    for other in candidates:
        if other.uid == entity.uid or not other.alive:
            continue
        ranked.append((haversine_m((entity.pose.lon, entity.pose.lat),
                                   (other.pose.lon, other.pose.lat)), other))
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1] if ranked else None


def _nearest_objective(world, entity, owned: bool):
    ranked = []
    for cell, obj in world.objectives.items():
        if owned and obj.get("owner") != entity.affiliation:
            continue
        if not owned and obj.get("owner") == entity.affiliation:
            continue
        lon, lat = world.rule_grid.cell_center_lonlat(*cell)
        ranked.append((haversine_m((entity.pose.lon, entity.pose.lat), (lon, lat)),
                       lon, lat))
    ranked.sort(key=lambda item: item[0])
    return (ranked[0][1], ranked[0][2]) if ranked else None


class TacticalController(object):
    def __init__(self, interval_s: float = 5.0):
        self.interval_s = float(interval_s)
        self._next_s = 0.0

    def update(self, world, ctx) -> None:
        if world.time_s < self._next_s:
            return
        self._next_s = world.time_s + self.interval_s
        for entity in world.sorted_entities():
            if not entity.alive or entity.on_board or not _is_ground(entity):
                continue
            if entity.route is not None or entity.goal is not None:
                continue
            if entity.uid in world.orders:
                continue
            if getattr(world, "_commander_control", {}).get(entity.uid, 0.0) > world.time_s:
                continue
            self._plan_one(world, ctx, entity)

    def _plan_one(self, world, ctx, entity) -> None:
        hp = entity.blood / max(1.0, float(entity.max_blood))
        enemies = []
        for uid in world.seen.get(entity.uid, set()):
            other = world.entities.get(uid)
            if other is not None and other.alive and other.affiliation != entity.affiliation:
                enemies.append(other)
        nearest_enemy = _nearest_entity(world, entity, enemies)

        if hp <= 0.35:
            goal = _nearest_objective(world, entity, owned=True)
            if goal is not None:
                world.order(entity.uid, {"type": "move_to", "lon": goal[0], "lat": goal[1]})
            else:
                world.order(entity.uid, {"type": "stop"})
            return

        if nearest_enemy is not None and _in_cover(world, entity):
            # Hold the covered position and let the engagement rules fire.
            world.order(entity.uid, {"type": "stop"})
            return

        target = world._best_target(entity, ctx)
        if target is not None:
            # In weapon range: stop and engage rather than drive into the open.
            world.order(entity.uid, {"type": "stop"})
            return

        if nearest_enemy is not None:
            goal = (nearest_enemy.pose.lon, nearest_enemy.pose.lat)
        else:
            goal = _nearest_objective(world, entity, owned=False)
        if goal is not None:
            world.order(entity.uid, {"type": "move_to", "lon": goal[0], "lat": goal[1]})
