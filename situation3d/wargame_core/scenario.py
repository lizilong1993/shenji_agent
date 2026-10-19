"""Scenario/map loading for the clean-room land wargame core.

The loader accepts the public 庙算 ``scenario.json`` and ``basic.json`` shapes,
so the same fixtures used by the closed SDK can be used as compatibility
vectors.  It does not import or call ``train_env``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from . import hexmap, weapons
from .state import GameState, Objective, Unit


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_map(path: str) -> hexmap.HexMap:
    return hexmap.HexMap(load_json(path))


def _unit_type(name: str, op_type: int) -> str:
    if op_type == 1:
        return "infantry"
    if op_type == 3:
        if "无人机" in name or "巡飞弹" in name:
            return "uav"
        if "直升机" in name:
            return "helicopter"
        return "aircraft"
    if op_type == 4:
        return "fort"
    if "炮兵" in name:
        return "artillery"
    if "防空" in name:
        return "aa"
    if "侦察" in name:
        return "recon"
    return "vehicle"


def _domain(op_type: int) -> str:
    return "air" if op_type == 3 else "land"


def unit_from_operator(op: dict) -> Unit:
    name = str(op.get("name") or "")
    op_type = int(op.get("type") or 2)
    sub_type = int(op.get("sub_type") or 0)
    profile = weapons.weapons_for(name, op_type, sub_type)
    ammo = {w.key: int(w.ammo) for w in profile}
    blood = int(op.get("blood") or 1)
    max_blood = int(op.get("max_blood") or blood)
    move_state = int(op.get("move_state") or 0)
    return Unit(
        uid=int(op.get("obj_id")),
        name=name or ("算子-%s" % op.get("obj_id")),
        color=int(op.get("color") or 0),
        op_type=op_type,
        sub_type=sub_type,
        pos=int(op.get("cur_hex")),
        blood=blood,
        max_blood=max_blood,
        armor=int(op.get("armor") or 0),
        speed_kmh=float(op.get("basic_speed") or 30.0),
        value=int(op.get("value") or (blood * 3)),
        unit_type=_unit_type(name, op_type),
        domain=_domain(op_type),
        weapons=profile,
        ammo=ammo,
        observe_distance={str(k): int(v) for k, v in (op.get("observe_distance") or {}).items()},
        suppressed=bool(op.get("keep")),
        move_state=move_state,
        concealed=(move_state == 4),
        marching=(move_state == 1),
        moving=(int(op.get("stop") or 1) == 0),
        raw=op,
    )


def build_state(scenario: dict, hmap: hexmap.HexMap,
                seed: int = 20261019,
                tick_seconds: float = 30.0) -> GameState:
    units: Dict[int, Unit] = {}
    for op in scenario.get("operators") or []:
        u = unit_from_operator(op)
        units[u.uid] = u
    objectives: List[Objective] = []
    for c in scenario.get("cities") or []:
        objectives.append(Objective(
            coord=int(c.get("coord")),
            value=int(c.get("value") or 0),
            flag=int(c.get("flag") if c.get("flag") is not None else -1),
            name=str(c.get("name") or ""),
        ))
    t = scenario.get("time") or {}
    max_time = int(t.get("max_time") or 1800)
    state = GameState(hmap, units, objectives, seed=seed,
                      tick_seconds=tick_seconds, max_time=max_time,
                      scenario_id=scenario.get("scenario_id"))
    _mark_stacking(state)
    return state


def _mark_stacking(state: GameState) -> None:
    by_pos: Dict[int, int] = {}
    for u in state.units.values():
        if u.domain == "land":
            by_pos[u.pos] = by_pos.get(u.pos, 0) + 1
    for u in state.units.values():
        u.stacked = by_pos.get(u.pos, 0) > 1


def load_scenario(scenario_path: str, map_path: str,
                  seed: int = 20261019,
                  tick_seconds: float = 30.0) -> GameState:
    scenario = load_json(scenario_path)
    hmap = load_map(map_path)
    return build_state(scenario, hmap, seed=seed, tick_seconds=tick_seconds)


def find_map_for_scenario(scenario: dict, data_dir: str) -> Optional[str]:
    """Best-effort lookup: ``scenario_id`` -> ``maps/map_<id>/basic.json``.

    Public scenario ids are not always the same as map ids; the caller can pass
    an explicit map path instead.  This helper exists for the common case.
    """
    sid = scenario.get("scenario_id")
    if sid is None:
        return None
    candidates = [
        os.path.join(data_dir, "maps", "map_%s" % sid, "basic.json"),
        os.path.join(data_dir, "maps", "map_%s" % str(sid)[:2], "basic.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None
