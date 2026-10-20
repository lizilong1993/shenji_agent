"""Campaign orchestration for the joint wargame core.

A campaign is a sequence of stages.  Each stage is run by the same
``WargameEngine``; the state is carried forward and never re-randomised.  The
orchestrator adds:

* a stage state machine with exit conditions (state-driven, time as fallback),
* script events with preconditions and fallbacks,
* per-stage snapshots (positions, strength, ammo, fuel, objectives),
* a final outcome with both the engine result and the optional script outcome.

Synthetic demonstration model; see ``wargame_core_spec.md``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from wargame_core import engine, hexmap
from wargame_core.state import GameState, Unit


@dataclass
class Stage:
    id: str
    name: str
    index: int
    start_tick: int
    end_tick: int
    mode: str = "mixed"          # scripted | mixed | adjudicated | engine
    summary: str = ""
    narrative: str = ""
    camera: Dict[str, float] = field(default_factory=dict)
    focus_objectives: List[str] = field(default_factory=list)
    # Optional state-driven exit: {"objective": "OBJ-BEACH", "owner": "red"}.
    exit_conditions: List[dict] = field(default_factory=list)


@dataclass
class ScriptEvent:
    id: str
    tick: int
    type: str
    description: str
    actors: List[int] = field(default_factory=list)
    location: Optional[List[float]] = None
    effects: Dict[str, dict] = field(default_factory=dict)
    # Simple preconditions: {"unit_alive": uid, "min_fuel": uid: 0.2, ...}
    requires: Dict[str, object] = field(default_factory=dict)
    fallback: Optional[str] = None
    # Optional action: spawn_beach | convert_land | set_jam | clear_jam.
    action: Optional[str] = None
    action_args: Dict[str, object] = field(default_factory=dict)
    fired: bool = False


@dataclass
class CampaignSpec:
    id: str
    name: str
    theatre: str
    seed: int
    stages: List[Stage]
    objectives: List[dict]
    script_events: List[ScriptEvent] = field(default_factory=list)
    script_outcome: Optional[dict] = None
    disclaimer: str = ""
    supply_points: List[dict] = field(default_factory=list)
    tick_seconds: float = 30.0


class CampaignRunner(object):
    def __init__(self, spec: CampaignSpec, state: GameState,
                 orders_fn: Callable[[GameState], List[engine.Order]] = engine.auto_orders):
        self.spec = spec
        self.state = state
        self.engine = engine.WargameEngine(state)
        self.orders_fn = orders_fn
        self.stage_states: List[dict] = []
        self.stage_events: Dict[str, List[dict]] = {}
        self.fired_script: List[dict] = []
        self.frames: List[dict] = []
        self.state.supply_points = list(spec.supply_points or [])

    # -- helpers -----------------------------------------------------------
    def _objective_by_id(self, oid: str):
        for o in self.state.objectives:
            if o.name == oid or str(o.coord) == oid:
                return o
        return None

    def _stage_done(self, stage: Stage) -> bool:
        if self.state.tick >= stage.end_tick:
            return True
        for cond in stage.exit_conditions:
            oid = cond.get("objective")
            owner = cond.get("owner")
            obj = self._objective_by_id(oid) if oid else None
            if obj is None:
                continue
            cur = ("red" if obj.flag == 0 else "blue" if obj.flag == 1 else "neutral")
            if owner and cur == owner:
                return True
        return False

    def _script_preconditions(self, ev: ScriptEvent) -> bool:
        req = ev.requires or {}
        for key, val in req.items():
            if key == "unit_alive":
                for uid in (val if isinstance(val, list) else [val]):
                    u = self.state.units.get(int(uid))
                    if u is None or not u.alive:
                        return False
            elif key == "min_fuel":
                for uid, fuel in (val or {}).items():
                    u = self.state.units.get(int(uid))
                    if u is None or not u.alive or u.fuel < float(fuel):
                        return False
            elif key == "objective_owner":
                for oid, owner in (val or {}).items():
                    obj = self._objective_by_id(oid)
                    if obj is None:
                        continue
                    cur = ("red" if obj.flag == 0 else "blue" if obj.flag == 1 else "neutral")
                    if cur != owner:
                        return False
        return True

    def _apply_script_effects(self, ev: ScriptEvent) -> None:
        from wargame_core import beach
        for uid, eff in (ev.effects or {}).items():
            u = self.state.units.get(int(uid))
            if u is None:
                continue
            if "strength_delta" in eff:
                u.strength = max(0.0, u.strength + float(eff["strength_delta"]))
            if "blood_delta" in eff:
                u.blood = max(0, u.blood + int(eff["blood_delta"]))
                if u.blood <= 0:
                    u.alive = False
            if "status" in eff:
                u.raw["script_status"] = eff["status"]
            if "fuel_delta" in eff:
                u.fuel = max(0.0, min(1.0, u.fuel + float(eff["fuel_delta"])))
            if "ammo_delta" in eff:
                for k, d in (eff["ammo_delta"] or {}).items():
                    u.ammo[k] = max(0, u.ammo.get(k, 0) + int(d))
            if eff.get("convert_land"):
                beach.convert_to_land(self.state, u, eff.get("name"))
        action = ev.action
        if action == "spawn_beach":
            ship = self.state.units.get(int(ev.action_args.get("ship")))
            if ship is not None and ship.alive:
                craft = beach.spawn_landing_craft(
                    self.state, ship,
                    str(ev.action_args.get("name") or "登陆艇"),
                    blood=int(ev.action_args.get("blood") or 2))
                self.state.units[craft.uid] = craft
        elif action == "set_jam":
            self.state.jam = float(ev.action_args.get("value") or 0.4)
        elif action == "clear_jam":
            self.state.jam = 0.0
        elif action == "convert_land":
            for uid in (ev.action_args.get("units") or []):
                u = self.state.units.get(int(uid))
                if u is not None and u.alive:
                    beach.convert_to_land(self.state, u, ev.action_args.get("name"))

    def _record_frame(self, frames: List[dict]) -> None:
        """One compact frame per tick for the front end."""
        targets = []
        for u in self.state.units.values():
            if not u.alive:
                continue
            try:
                lon, lat = self.state.map.pos_to_lonlat(u.pos)
            except KeyError:
                lon, lat = 0.0, 0.0
            targets.append({
                "id": u.uid,
                "lon": round(lon, 6),
                "lat": round(lat, 6),
                "alt": 0.0,
                "speed_kn": round(u.speed_kmh / 1.852, 1),
                "heading": round(u.heading, 1),
                "status": ("destroyed" if not u.alive else
                           "damaged" if u.blood < u.max_blood else "ready"),
                "strength": round(u.strength, 1),
                "blood": u.blood,
                "domain": u.domain,
            })
        frames.append({
            "tick": self.state.tick,
            "t": round(self.state.tick * self.state.tick_seconds, 1),
            "stage": self.state.stage_id,
            "targets": targets,
            "objectives": {str(o.coord): ("red" if o.flag == 0
                                          else "blue" if o.flag == 1 else "neutral")
                           for o in self.state.objectives},
        })

    def _push_script_event(self, ev: ScriptEvent, note: str = "") -> None:
        self.fired_script.append({
            "id": ev.id,
            "tick": self.state.tick,
            "type": ev.type,
            "source": "script",
            "level": "info",
            "actors": list(ev.actors),
            "location": list(ev.location or []),
            "description": note or ev.description,
            "raw": {"script_id": ev.id},
        })

    def _fire_due_script(self, stage: Stage) -> None:
        """Fire script events whose tick has arrived (once each)."""
        for ev in self.spec.script_events:
            if ev.fired or ev.tick > self.state.tick:
                continue
            if self._script_preconditions(ev):
                self._apply_script_effects(ev)
                self._push_script_event(ev)
            else:
                msg = ev.fallback or ("脚本事件 %s 的前置条件不满足，已跳过" % ev.id)
                self._push_script_event(ev, note=msg)
            ev.fired = True

    # -- main loop ---------------------------------------------------------
    def run(self, max_ticks_per_stage: Optional[int] = None) -> dict:
        spec = self.spec
        for stage in spec.stages:
            self.state.stage_id = stage.id
            stage_events: List[dict] = []
            self._fire_due_script(stage)

            stage_budget = max_ticks_per_stage or (stage.end_tick - stage.start_tick + 2)
            steps = 0
            while not self._stage_done(stage) and steps < stage_budget:
                if self.state.winner:
                    break
                stage_events.extend(self.engine.step(self.orders_fn(self.state)))
                self._record_frame(self.frames)
                self._fire_due_script(stage)
                steps += 1
            # Persist the stage snapshot and events.
            self.stage_events[stage.id] = stage_events
            self.stage_states.append({
                "stage": stage.id,
                "name": stage.name,
                "index": stage.index,
                "start_tick": stage.start_tick,
                "end_tick": self.state.tick,
                "mode": stage.mode,
                "summary": stage.summary,
                "narrative": stage.narrative,
                "camera": stage.camera,
                "focus_objectives": stage.focus_objectives,
                "state": self.state.snapshot(),
                "events": len(stage_events),
            })
            if self.state.winner:
                break

        outcome = self._outcome()
        return {
            "campaign": {
                "id": spec.id,
                "name": spec.name,
                "theatre": spec.theatre,
                "seed": spec.seed,
                "disclaimer": spec.disclaimer,
            },
            "stages": self.stage_states,
            "events": self._all_events(),
            "frames": self.frames,
            "outcome": outcome,
            "final": self.state.snapshot(),
        }

    def _all_events(self) -> List[dict]:
        out: List[dict] = []
        for st in self.stage_states:
            out.extend(self.stage_events.get(st["stage"], []))
        # Script events are interleaved by tick.
        out.extend(self.fired_script)
        out.sort(key=lambda e: (e.get("tick", 0), e.get("id", "")))
        return out

    def _outcome(self) -> dict:
        s = self.state
        if s.winner is None:
            s.evaluate()
        eng = {
            "winner": s.winner or "undecided",
            "score": s.score(),
            "objectives": {str(o.coord): ("red" if o.flag == 0
                                          else "blue" if o.flag == 1 else "neutral")
                           for o in s.objectives},
            "losses": {
                "red": sum(1 for u in s.units.values() if not u.alive and u.color == 0),
                "blue": sum(1 for u in s.units.values() if not u.alive and u.color == 1),
            },
            "reason": s.reason or "推演结束",
        }
        script = self.spec.script_outcome
        match = None
        if script is not None:
            match = (script.get("winner") == eng["winner"])
        return {
            "engine": eng,
            "script": script,
            "match": match,
            "disclaimer": ("引擎结果由 wargame_core 根据最终状态计算；"
                           "脚本结局仅为虚构演示，二者都不是现实预测。"),
        }


def save_result(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
