"""Scenario/map loader for the geo engine.

Maps without ``grid`` metadata (official 庙算 ``basic.json``) are anchored to
a synthetic WGS84 origin (``geo_anchor`` in the scenario, or the demo default)
because the public files carry no georeference.  Campaign maps with a ``grid``
block use their real metadata.
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

from compat.legacy import LegacyHexGrid
from env.grid import MapGridEnvironment
from geo.rule_grid import RuleGrid
from rules.engagement import weapons_for as sea_air_weapons_for
from rules.weapons import WeaponTables, load_wpdata
from rules.context import WeaponView
from sim.entity import Entity, Pose
from sim.world import World
from sim import sensors as sensor_model
from sim.rolecards_bridge import (
    card_for_operator,
    domain_from_card,
    profile_from_card,
    sensors_from_card,
    unit_kind_from_card,
    unit_type_from_card,
    weapons_from_card,
)
from mobility.profile import profile_for

DEFAULT_ANCHOR = (119.9, 23.1)
DETECTION_MODELS = ("table", "spotting", "shadow")
DIRECT_FIRE_MODELS = ("crt", "physics", "shadow")


def _engine_section(value) -> dict:
    return value if isinstance(value, dict) else {}


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

INDIRECT_IDS = {72, 88, 89}

# Platform-name fallback when a scenario omits carry_weapon_ids (campaign
# scenarios rely on the legacy name mapping).
NAME_WEAPONS = {
    "重型坦克": [36, 43],
    "中型坦克": [37, 43],
    "重型战车": [54, 56, 43, 69],
    "中型战车": [56, 43, 73],
    "轻型战车": [54, 56, 43],
    "步兵": [29, 35, 71],
    "步兵小队": [29, 35, 71],
    "两栖步兵": [29, 35, 71],
    "两栖战车": [54, 56, 43],
    "炮兵": [72],
    "无人战车": [56, 43, 74],
    "侦察型战车": [56, 43],
    "防空导弹排": [71],
    "防空高炮": [1],
    "火箭布雷车": [43],
    "扫雷车": [43],
    "武装直升机": [54, 73],
    "巡飞弹": [76],
    "岸舰导弹营": [],
    "运输机": [],
    "无人机": [],
}
_TABLES = None


def weapon_tables() -> WeaponTables:
    global _TABLES
    if _TABLES is None:
        _TABLES = WeaponTables(load_wpdata())
    return _TABLES


def _unit_type(op: dict) -> str:
    name = str(op.get("name") or "")
    op_type = int(op.get("type") or 2)
    if op_type == 1:
        return "infantry"
    if op_type == 3:
        if "无人机" in name or "巡飞" in name:
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
    if "工程" in name or "扫雷" in name or "布雷" in name:
        return "engineering"
    if any(k in name for k in ("驱逐舰", "护卫舰", "巡逻舰", "补给舰", "登陆舰", "两栖舰", "航母")):
        return "ship"
    if "潜艇" in name:
        return "sub"
    if "两栖" in name:
        return "amphibious"
    return "vehicle"


def _domain(name: str, op_type: int, unit_type: str) -> str:
    if unit_type in ("ship",):
        return "surface"
    if unit_type in ("sub",):
        return "subsurface"
    if unit_type in ("amphibious",):
        return "amphibious"
    if op_type == 3:
        return "air"
    return "land"


def _unit_kind(op: dict, unit_type: str) -> str:
    name = str(op.get("name") or "")
    if unit_type == "infantry":
        return "infantry"
    if unit_type == "ship":
        return "ship"
    if unit_type == "sub":
        return "sub"
    if unit_type == "helicopter":
        return "helicopter"
    if unit_type == "uav":
        return "uav"
    if unit_type == "aircraft":
        return "helicopter"  # observation table has no fixed-wing row
    if "防空" in name:
        return "aa"
    if "侦察" in name:
        return "recon"
    return "vehicle"


def _grid_for(basic: dict, scenario: dict) -> LegacyHexGrid:
    if basic.get("grid"):
        return LegacyHexGrid.from_map_doc(basic)
    anchor = scenario.get("geo_anchor") or {}
    lon = float(anchor.get("lon", DEFAULT_ANCHOR[0]))
    lat = float(anchor.get("lat", DEFAULT_ANCHOR[1]))
    rows = int(basic.get("max_row") or len(basic.get("map_data") or []))
    cols = int(basic.get("max_column") or (len(basic["map_data"][0]) if basic.get("map_data") else 0))
    cell_m = float(basic.get("scale") or 200.0)
    return LegacyHexGrid(lon, lat, cols, rows, cell_m)


def _rule_grid_for(grid: LegacyHexGrid) -> RuleGrid:
    if abs(grid.cell_m - RuleGrid.RULE_CELL_M) < 1e-6:
        return RuleGrid(grid.center_lon, grid.center_lat, grid.cols, grid.rows)
    # Cover the whole legacy map at 200 m; rule distances are then absolute.
    a = grid.cell_center_lonlat(0, 0)
    b = grid.cell_center_lonlat(grid.rows - 1, grid.cols - 1)
    lon_min, lon_max = min(a[0], b[0]), max(a[0], b[0])
    lat_min, lat_max = min(a[1], b[1]), max(a[1], b[1])
    return RuleGrid.covering_bbox(lon_min, lat_min, lon_max, lat_max)


def _weapon_views(op: dict) -> Tuple[list, dict]:
    tables = weapon_tables()
    views = []
    ammo = {}
    ids = op.get("carry_weapon_ids")
    if not ids:
        ids = NAME_WEAPONS.get(str(op.get("name") or ""), [])
    for wid in ids:
        try:
            wid = int(wid)
            wtab = tables.tables_for_weapon(wid)
        except (TypeError, ValueError):
            continue
        if not wtab:
            continue
        kind = "indirect" if wid in INDIRECT_IDS else "direct"
        key = str(wid)
        views.append(WeaponView(key=key, weapon_id=wid, name=key, kind=kind,
                                shoot_on_move=wid in (36, 37, 54),
                                cooldown_s=75.0, ammo=10, tables=wtab))
        ammo[key] = 10
    return views, ammo


def build_world(scenario: dict, basic: dict, seed: int = 20261021,
                dt_s: float = 1.0, envpack: Optional[str] = None) -> World:
    if envpack:
        from env.pack_env import EnvPackEnvironment
        env = EnvPackEnvironment(envpack)
        grid = env.grid
    else:
        grid = _grid_for(basic, scenario)
        env = MapGridEnvironment(basic, grid)
    engine = _engine_section(scenario.get("engine"))
    detection_model = str(engine.get("detection_model") or "table").lower()
    if detection_model not in DETECTION_MODELS:
        detection_model = "table"
    direct_fire_model = str(engine.get("direct_fire_model") or "crt").lower()
    if direct_fire_model not in DIRECT_FIRE_MODELS:
        direct_fire_model = "crt"
    commander_interval_s = _as_float(engine.get("commander_interval_s"), 2.0)
    use_role_cards = _as_bool(engine.get("use_role_cards"), False)
    world = World(env, _rule_grid_for(grid), dt_s=dt_s, seed=seed, plan_grid=grid,
                  detection_model=detection_model,
                  direct_fire_model=direct_fire_model,
                  enable_radio_share=_as_bool(engine.get("enable_radio_share"), False),
                  enable_tactics=_as_bool(engine.get("enable_tactics"), False),
                  commander_interval_s=commander_interval_s)
    if _as_bool(engine.get("enable_commander"), False):
        from sim.commander import RuleCommander
        world.set_commander(RuleCommander(), commander_interval_s)
    for op in scenario.get("operators") or []:
        unit_type = _unit_type(op)
        domain = _domain(str(op.get("name") or ""), int(op.get("type") or 2), unit_type)
        if op.get("domain"):
            domain = str(op["domain"])
            if domain == "sea":
                domain = "surface"
            elif domain == "sub":
                domain = "subsurface"
        lon, lat = grid.to_lonlat(int(op["cur_hex"]))
        card_id, card = card_for_operator(op, domain, infer=use_role_cards)
        if card:
            domain = domain_from_card(card) or domain
            unit_type = unit_type_from_card(card_id)
            profile = profile_from_card(card, key=card_id)
        else:
            profile = profile_for(unit_type, domain)
        armor = op.get("armor")
        armor = int(armor) if armor is not None else int(profile.armor_class)
        if card:
            weapons, ammo = weapons_from_card(card)
            sea_weapons = []
            sensors = sensors_from_card(card)
            default_blood = int((card.get("protection") or {}).get("max_blood") or 1)
        else:
            weapons, ammo = _weapon_views(op)
            sea_weapons = sea_air_weapons_for(str(op.get("name") or ""), domain)
            for w in sea_weapons:
                ammo.setdefault(w.key, int(w.ammo))
            sensors = sensor_model.sensors_for(str(op.get("name") or ""), domain)
            default_blood = 1
        entity = Entity(
            uid=int(op["obj_id"]),
            name=str(op.get("name") or op["obj_id"]),
            affiliation=int(op.get("color") or 0),
            domain=domain,
            profile_id=card_id or unit_type,
            profile=profile,
            pose=Pose(lon, lat, env.elevation(lon, lat),
                      heading_deg=_as_float(op.get("heading_deg"), 0.0)),
            unit_type=unit_type,
            unit_kind=(unit_kind_from_card(card_id) if card
                       else _unit_kind(op, unit_type)),
            blood=int(op.get("blood") or default_blood),
            max_blood=int(op.get("max_blood") or op.get("blood") or default_blood),
            armor=armor,
            armor_profile={str(k): float(v) for k, v in (op.get("armor_profile") or {}).items()},
            size_class=str(op.get("size_class") or profile.size_class),
            concealment=(float(op["concealment"]) if op.get("concealment") is not None else None),
            comms_range_m=(600.0 if op.get("comms_range_m") is None
                           else _as_float(op.get("comms_range_m"), 600.0)),
            depth=str(op.get("depth") or "surface"),
            emitting=bool(op.get("emitting", True)),
            observe_distance={str(k): int(v) for k, v in (op.get("observe_distance") or {}).items()},
            ammo=ammo,
            weapons=weapons,
            sea_air_weapons=sea_weapons,
            sensors=sensors,
            value=int(op.get("value") or 0),
            readiness=float(op.get("readiness") or 0.85),
            concealed=(int(op.get("move_state") or 0) == 4),
            marching=(int(op.get("move_state") or 0) == 1),
            role_card=card,
        )
        world.add_entity(entity)
    # Units that start on water ride a friendly ship until it reaches land.
    ships = [e for e in world.entities.values() if e.domain == "surface"]
    for entity in world.entities.values():
        if entity.domain != "land":
            continue
        depth = env.water_depth(entity.pose.lon, entity.pose.lat)
        if depth is None or depth <= entity.profile.wading_depth_m:
            continue
        carrier = None
        for ship in ships:
            if ship.affiliation != entity.affiliation:
                continue
            cell_a = grid.cell_of(ship.pose.lon, ship.pose.lat)
            cell_b = grid.cell_of(entity.pose.lon, entity.pose.lat)
            if cell_a == cell_b or grid.distance(cell_a, cell_b) <= 1:
                carrier = ship
                break
        if carrier is not None:
            entity.on_board = True
            entity.carrier = carrier.uid
            entity.pose = Pose(carrier.pose.lon, carrier.pose.lat, carrier.pose.alt_m)
            carrier.passengers.append(entity.uid)
    for city in scenario.get("cities") or []:
        coord = city.get("coord")
        if coord is None:
            continue
        try:
            lon, lat = grid.to_lonlat(int(coord))
        except (TypeError, ValueError):
            continue
        world.add_objective(world.rule_grid.cell_of(lon, lat),
                            value=int(city.get("value") or 0),
                            name=str(city.get("name") or ""))
    for sp in scenario.get("supply_points") or []:
        if isinstance(sp, dict) and "lon" in sp and "lat" in sp:
            world.supply_points.append((float(sp["lon"]), float(sp["lat"])))
    return world


def load_world(scenario_path: str, map_path: str, seed: int = 20261021,
               dt_s: float = 1.0, envpack: Optional[str] = None) -> World:
    with open(scenario_path, encoding="utf-8") as fh:
        scenario = json.load(fh)
    with open(map_path, encoding="utf-8") as fh:
        basic = json.load(fh)
    return build_world(scenario, basic, seed=seed, dt_s=dt_s, envpack=envpack)
