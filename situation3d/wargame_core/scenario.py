"""Scenario/map loading for the clean-room land wargame core.

The loader accepts the public 庙算 ``scenario.json`` and ``basic.json`` shapes,
so the same fixtures used by the closed SDK can be used as compatibility
vectors.  It does not import or call ``train_env``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from . import engagement, hexmap, sensors, weapons
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


def _domain_from_name(name: str, op_type: int) -> str:
    """Infer the theatre domain from the platform name.

    The public scenario format has no ``domain`` field, so the demo campaign
    scenarios declare it explicitly; this fallback keeps the public fixtures
    working.
    """
    if op_type == 3:
        return "air"
    if "潜艇" in name:
        return "sub"
    if any(k in name for k in ("驱逐舰", "护卫舰", "巡逻舰", "补给舰",
                               "登陆舰", "两栖", "航母", "舰")):
        return "sea"
    return "land"


def _domain(op_type: int) -> str:
    return "air" if op_type == 3 else "land"


def _rich_unit(op: dict, hmap: Optional[hexmap.HexMap] = None) -> Unit:
    """Build a unit and attach sea/air sensors and weapons when relevant.

    The public ``scenario.json`` schema has no ``domain`` field; campaign
    scenarios can add ``domain`` and ``altitude_band`` per operator, and this
    loader honours them.  Without those fields the platform name is used.
    """
    u = unit_from_operator(op)
    domain = str(op.get("domain") or "").strip()
    if domain in ("sea", "sub", "air", "land"):
        u.domain = domain
    else:
        u.domain = _domain_from_name(u.name, u.op_type)
    u.altitude_band = str(op.get("altitude_band") or
                          ("水面" if u.domain in ("sea", "sub")
                           else "中空" if u.domain == "air" else "ground"))
    u.depth = str(op.get("depth") or "surface")
    u.size_class = str(op.get("size_class") or
                       ("large" if u.domain == "sea" and "驱逐" in u.name
                        else "small" if u.domain == "air" else "medium"))
    u.heading = float(op.get("heading") or 0.0)
    u.readiness = float(op.get("readiness") or 0.85)
    u.fuel = float(op.get("fuel") or 1.0)
    u.emitting = bool(op.get("emitting", True))
    u.sensors = sensors.sensors_for(u.name, u.domain)
    # Public ``observe_distance`` (by target sub_type) is authoritative for
    # land observation; merge it into the visual sensor range so the public
    # fixtures behave as documented.
    if u.domain == "land" and u.observe_distance:
        seen = []
        for key, val in u.observe_distance.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v > 0:
                seen.append(v)
        if seen:
            hex_range = max(seen)
            km = hex_range * hmap.scale_m / 1000.0 if hmap else hex_range
            u.sensors = [sensors.Sensor("visual", km, 1.0)] + u.sensors
    if u.domain in ("sea", "sub", "air"):
        prof = engagement.sea_air_weapons_for(u.name, u.domain)
        u.sea_air_weapons = prof
        for w in prof:
            u.ammo.setdefault(w.key, int(w.ammo))
    if op.get("beaching"):
        u.raw["beaching"] = True
    return u


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
        u = _rich_unit(op, hmap)
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
    state.supply_points = list(scenario.get("supply_points") or [])
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


def load_scenario_dict(scenario: dict, map_path: str,
                       seed: int = 20261019,
                       tick_seconds: float = 30.0) -> GameState:
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
