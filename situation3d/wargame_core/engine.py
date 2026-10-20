"""Turn loop for the clean-room joint wargame core (land + sea + air)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from . import (air, combat, detection, engagement, hexmap, logistics,
               movement, naval)
from .state import GameState, Unit


@dataclass
class Order:
    uid: int
    action: str                 # move | fire | hold
    path: Optional[List[int]] = None
    target: Optional[int] = None


def _move_mode(unit: Unit) -> str:
    if unit.domain == "air":
        return "air"
    if unit.domain == "sea":
        return "sea"
    if unit.domain == "sub":
        return "sub"
    if unit.unit_type == "infantry":
        return "infantry"
    return "vehicle"


def _enemy_goal(unit: Unit, state: GameState) -> Optional[int]:
    enemies = [u for u in state.living() if u.color != unit.color]
    if not enemies:
        return None
    return min(enemies, key=lambda u: hexmap.distance(unit.pos, u.pos)).pos


def _objective_goal(unit: Unit, state: GameState) -> Optional[int]:
    # Prefer an objective not owned by this side, then the nearest one.
    candidates = [o for o in state.objectives if o.flag != unit.color]
    if not candidates:
        candidates = list(state.objectives)
    if not candidates:
        return None
    return min(candidates, key=lambda o: hexmap.distance(unit.pos, o.coord)).coord


def _target_for(unit: Unit, state: GameState):
    if unit.domain == "land":
        t = combat.best_target(unit, state)
        if t is not None:
            return t
        t = engagement.best_sea_air_target(unit, state)
        return t
    return engagement.best_sea_air_target(unit, state)


def _enemy_goal_sea(unit: Unit, state: GameState) -> Optional[int]:
    enemies = [u for u in state.living()
               if u.color != unit.color and u.domain in ("sea", "sub")]
    if not enemies:
        return None
    return min(enemies, key=lambda u: hexmap.distance(unit.pos, u.pos)).pos


def _beach_water_goal(state: GameState, unit: Unit) -> Optional[int]:
    """Nearest water hex within 3 rings of the declared landing beach."""
    from collections import deque
    start = getattr(state, "landing_target", None)
    if start is None:
        return None
    q = deque([(start, 0)])
    seen = {start}
    while q:
        cur, d = q.popleft()
        if d > 0 and state.map.cond(cur) == hexmap.RIVER:
            return cur
        if d >= 3:
            continue
        for nb in state.map.neighbors(cur):
            if nb >= 0 and nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None


def _cached_path(state: GameState, key: tuple, fn):
    """Plan at most once per (unit, goal); A* dominates the tick cost."""
    if key in state.path_cache:
        path = state.path_cache[key]
        return list(path) if path else None
    path = fn()
    state.path_cache[key] = list(path) if path else None
    # Keep the cache small; stale entries are rare in a short campaign.
    if len(state.path_cache) > 512:
        state.path_cache.clear()
    return path


def auto_orders(state: GameState) -> List[Order]:
    """A small deterministic staff officer for the demo."""
    orders: List[Order] = []
    for u in state.living():
        if u.suppressed:
            orders.append(Order(u.uid, "hold"))
            continue
        if not state.map.valid(u.pos):
            # Public fixtures can contain an out-of-map operator; leave it
            # parked instead of letting A*/LOS raise.
            orders.append(Order(u.uid, "hold"))
            continue
        target = _target_for(u, state)
        if target is not None:
            orders.append(Order(u.uid, "fire", target=target.uid))
            continue
        if u.unit_type == "fort":
            orders.append(Order(u.uid, "hold"))
            continue
        if u.unit_type in ("artillery", "aa") and u.domain == "land":
            orders.append(Order(u.uid, "hold"))
            continue
        if u.domain in ("sea", "sub"):
            goal = None
            if getattr(u, "raw", {}).get("beaching") and getattr(state, "landing_target", None):
                goal = _beach_water_goal(state, u)
            if goal is None:
                # Ships pursue water objectives or enemy ships.
                for o in state.objectives:
                    if o.flag != u.color and state.map.cond(o.coord) == hexmap.RIVER:
                        if goal is None or hexmap.distance(u.pos, o.coord) < hexmap.distance(u.pos, goal):
                            goal = o.coord
            if goal is None:
                goal = _enemy_goal_sea(u, state)
            if goal is None:
                orders.append(Order(u.uid, "hold"))
                continue
            path = _cached_path(
                state, (u.uid, u.pos, goal),
                lambda: naval.naval_path(state.map, u.pos, goal, u))
            orders.append(Order(u.uid, "move", path=path) if path
                          else Order(u.uid, "hold"))
            continue
        goal = _objective_goal(u, state) or _enemy_goal(u, state)
        if goal is None or u.pos == goal:
            orders.append(Order(u.uid, "hold"))
            continue
        if u.domain == "air":
            path = _cached_path(
                state, (u.uid, u.pos, goal),
                lambda: air.air_path(state.map, u.pos, goal))
        else:
            mode = _move_mode(u)
            path = _cached_path(
                state, (u.uid, u.pos, goal),
                lambda: movement.find_path(state.map, u.pos, goal, mode))
            if not path:
                goal = _enemy_goal(u, state)
                path = _cached_path(
                    state, (u.uid, u.pos, goal),
                    lambda: movement.find_path(state.map, u.pos, goal, mode)) if goal is not None else None
        if path:
            orders.append(Order(u.uid, "move", path=path))
        else:
            orders.append(Order(u.uid, "hold"))
    return orders


class WargameEngine(object):
    def __init__(self, state: GameState):
        self.state = state
        self.tick_events: List[dict] = []

    def _event(self, etype: str, **kw) -> dict:
        ev = {
            "id": "EV-%06d" % self.state.next_event_id(),
            "tick": self.state.tick,
            "type": etype,
            "source": "engine",
            "level": kw.pop("level", "info"),
            "actors": kw.pop("actors", []),
            "location": kw.pop("location", []),
            "description": kw.pop("description", ""),
            "raw": kw.pop("raw", {}),
        }
        ev.update(kw)
        self.state.events.append(ev)
        self.tick_events.append(ev)
        return ev

    def _can_enter(self, unit: Unit, pos: int) -> bool:
        state = self.state
        if unit.domain == "land":
            return len([x for x in state.unit_at(pos, unit.color)
                        if x.domain == "land"]) < 4
        if unit.domain == "air":
            return True
        # Ships and submarines: one surface vessel per hex keeps the display
        # readable; submarines are separate.
        same = [x for x in state.unit_at(pos, unit.color)
                if x.domain == unit.domain and x.uid != unit.uid]
        return len(same) < 2

    def step(self, orders: Iterable[Order]) -> List[dict]:
        self.tick_events = []
        state = self.state
        event_start = len(state.events)
        by_uid = {o.uid: o for o in orders}
        # Reset per-tick transient flags.
        for u in state.living():
            u.moving = False
            u.marching = False
            if u.suppressed_ticks > 0:
                u.suppressed_ticks -= 1
                if u.suppressed_ticks <= 0:
                    u.suppressed = False

        # Movement first, then fire, so a unit that just moved can still be
        # engaged by a stationary enemy this tick.
        for u in state.living():
            if not u.alive or u.suppressed:
                continue
            order = by_uid.get(u.uid)
            if order and order.action == "move" and order.path:
                if u.domain == "sea" or u.domain == "sub":
                    entered = naval.advance_naval(
                        u, order.path, state.tick_seconds, state.map)
                elif u.domain == "air":
                    entered = air.advance_air(
                        u, order.path, state.tick_seconds, state.map)
                else:
                    entered = movement.advance(
                        u, order.path, state.tick_seconds, state.map,
                        _move_mode(u),
                        can_enter=lambda p, _u=u: self._can_enter(_u, p))
                u.moving = bool(entered)
                u.concealed = False
                if entered:
                    u.marching = False
                    if u.domain in ("sea", "sub") and u.path_index > 0:
                        u.heading = state.map.bearing_deg(
                            u.path[u.path_index - 1], u.pos)

        for u in state.living():
            if not u.alive or u.suppressed:
                continue
            order = by_uid.get(u.uid)
            target = None
            if order and order.action == "fire" and order.target is not None:
                target = state.units.get(order.target)
            if target is None:
                target = _target_for(u, state)
            if target is None:
                continue
            if u.domain in ("sea", "sub"):
                ev = naval.resolve_naval_fire(u, target, state)
            elif u.domain == "air":
                ev = air.resolve_air_fire(u, target, state)
            elif engagement.can_engage(u, target):
                ev = engagement.resolve_sea_air(u, target, state)
            else:
                ev = combat.resolve_direct_fire(u, target, state)
            if ev is not None and not target.alive:
                self._event(
                    "destroyed", level="warn", actors=[target.uid],
                    location=list(hexmap.decode(target.pos)),
                    description="%s 被摧毁" % target.name)

        # Fuel burn + resupply (supply points are optional).
        for u in state.living():
            air.update_fuel(u, state.tick_seconds)
        resupply_marks = getattr(state, "resupply_marks", None)
        if resupply_marks is None:
            resupply_marks = {}
            state.resupply_marks = resupply_marks
        for ev in logistics.resupply(state, state.tick_seconds):
            # One event per unit per 20 ticks; otherwise a unit parked on a
            # supply point floods the timeline.
            uid = (ev.get("actors") or [None])[0]
            if uid is not None and state.tick - resupply_marks.get(uid, -999) < 20:
                continue
            if uid is not None:
                resupply_marks[uid] = state.tick
            self._event("resupply", level="info", actors=ev.get("actors", []),
                        location=ev.get("location", []),
                        description=ev.get("description", ""), raw=ev.get("raw", {}))

        changes = state.control_objectives()
        for c in changes:
            self._event(
                "objective", level="info", actors=[],
                location=list(hexmap.decode(c["objective"]["coord"])),
                description="目标点「%s」归属 %s → %s" % (
                    c["objective"]["name"] or c["objective"]["coord"],
                    c["from"], c["to"]),
                raw=c)

        state.tick += 1
        winner = state.evaluate()
        if winner:
            self._event(
                "outcome", level="warn",
                description="推演结束：%s（%s）" % (winner, state.reason),
                raw=state.score())
        # Combat modules append straight to state.events; merge everything
        # emitted this tick so the campaign's per-stage events are complete.
        seen = {id(e) for e in self.tick_events}
        merged = list(self.tick_events)
        for e in state.events[event_start:]:
            if id(e) not in seen:
                merged.append(e)
        merged.sort(key=lambda e: e.get("id", ""))
        return merged


def run_demo(state: GameState, max_ticks: int = 60,
             orders_fn: Callable[[GameState], List[Order]] = auto_orders,
             verbose: bool = False) -> dict:
    engine = WargameEngine(state)
    for _ in range(max_ticks):
        if state.winner:
            break
        evs = engine.step(orders_fn(state))
        if verbose and evs:
            for e in evs:
                if e["type"] in ("direct_fire", "sea_air_fire", "destroyed",
                                 "objective", "outcome", "beaching"):
                    print("[%4d] %s" % (e["tick"], e["description"]))
    return {
        "scenario_id": state.scenario_id,
        "tick": state.tick,
        "winner": state.winner,
        "reason": state.reason,
        "score": state.score(),
        "units": [u.to_dict() for u in state.units.values()],
        "objectives": [o.to_dict() for o in state.objectives],
        "events": state.events,
    }
