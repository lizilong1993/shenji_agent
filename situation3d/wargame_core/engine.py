"""Turn loop for the clean-room land wargame core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from . import combat, detection, hexmap, movement
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


def auto_orders(state: GameState) -> List[Order]:
    """A small deterministic staff officer for the demo.

    It fires when a target is visible and in range; otherwise it moves toward
    the nearest enemy or objective.  This is not a competitive AI; it exists to
    exercise the engine loop end-to-end.
    """
    orders: List[Order] = []
    for u in state.living():
        if u.suppressed:
            orders.append(Order(u.uid, "hold"))
            continue
        target = combat.best_target(u, state)
        if target is not None:
            orders.append(Order(u.uid, "fire", target=target.uid))
            continue
        if u.unit_type in ("artillery", "aa", "fort"):
            orders.append(Order(u.uid, "hold"))
            continue
        goal = _objective_goal(u, state) or _enemy_goal(u, state)
        if goal is None:
            orders.append(Order(u.uid, "hold"))
            continue
        if u.pos == goal:
            orders.append(Order(u.uid, "hold"))
            continue
        mode = _move_mode(u)
        path = movement.find_path(state.map, u.pos, goal, mode)
        if not path:
            goal = _enemy_goal(u, state)
            path = movement.find_path(state.map, u.pos, goal, mode) if goal is not None else []
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

    def step(self, orders: Iterable[Order]) -> List[dict]:
        self.tick_events = []
        state = self.state
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
                entered = movement.advance(u, order.path, state.tick_seconds,
                                           state.map, _move_mode(u))
                u.moving = bool(entered)
                u.concealed = False
                if entered:
                    u.marching = False

        for u in state.living():
            if not u.alive:
                continue
            if u.suppressed:
                continue
            order = by_uid.get(u.uid)
            target = None
            if order and order.action == "fire" and order.target is not None:
                target = state.units.get(order.target)
            if target is None:
                target = combat.best_target(u, state)
            if target is not None and combat.can_fire(u, target, state):
                combat.resolve_direct_fire(u, target, state)
                if not target.alive:
                    self._event(
                        "destroyed", level="warn", actors=[target.uid],
                        location=list(hexmap.decode(target.pos)),
                        description="%s 被摧毁" % target.name)

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
        return list(self.tick_events)


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
                if e["type"] in ("direct_fire", "destroyed", "objective", "outcome"):
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
