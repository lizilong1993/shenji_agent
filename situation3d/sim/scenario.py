"""Scenario/map loader for the geo engine.

Maps without ``grid`` metadata (official 庙算 ``basic.json``) are anchored to
a synthetic WGS84 origin (``geo_anchor`` in the scenario, or the demo default)
because the public files carry no georeference.  Campaign maps with a ``grid``
block use their real metadata.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
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
DETECTION_MODELS = ("table", "spotting", "shadow", "physical")
DIRECT_FIRE_MODELS = ("crt", "physics", "shadow")
INDIRECT_FIRE_MODELS = ("table", "physics")
MISSILE_MODELS = ("instant", "flight")


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


def _op_lonlat(op, grid):
    """Real WGS84 when the scenario carries it; hex otherwise."""
    if op.get("lon") is not None and op.get("lat") is not None:
        return float(op["lon"]), float(op["lat"])
    return grid.to_lonlat(int(op["cur_hex"]))

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


def _region_ring(item: dict):
    ring = item.get("ring")
    if ring:
        return [[float(p[0]), float(p[1])] for p in ring]
    center = item.get("center") or [item.get("lon"), item.get("lat")]
    if center[0] is None or center[1] is None:
        return []
    import math as _math
    radius = _as_float(item.get("radius_m"), 300.0)
    lat0 = float(center[1])
    kx = 111320.0 * _math.cos(_math.radians(lat0))
    ky = 110540.0
    pts = []
    for i in range(24):
        ang = 2.0 * _math.pi * i / 24.0
        pts.append([float(center[0]) + radius * _math.cos(ang) / kx,
                    lat0 + radius * _math.sin(ang) / ky])
    pts.append(pts[0])
    return pts


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
    direct_fire_model = str(engine.get("direct_fire_model") or "physics").lower()
    if direct_fire_model not in DIRECT_FIRE_MODELS:
        direct_fire_model = "physics"
    indirect_fire_model = str(engine.get("indirect_fire_model") or "physics").lower()
    if indirect_fire_model not in INDIRECT_FIRE_MODELS:
        indirect_fire_model = "physics"
    missile_model = str(engine.get("missile_model") or "flight").lower()
    if missile_model not in MISSILE_MODELS:
        missile_model = "flight"
    commander_interval_s = _as_float(engine.get("commander_interval_s"), 2.0)
    scan_interval_s = _as_float(engine.get("scan_interval_s"), 30.0)
    if not math.isfinite(scan_interval_s) or scan_interval_s < 0.0:
        scan_interval_s = 30.0
    scan_interval_s = min(86400.0, scan_interval_s)
    use_role_cards = _as_bool(engine.get("use_role_cards"), False)
    nav = None
    if envpack:
        from mobility.nav_field import NavField
        nav_res = _as_float(engine.get("nav_res_m"), 30.0)
        if _as_bool(engine.get("nav_enabled"), True):
            nav = NavField(env, res_m=nav_res)
            nav.strict_roads = _as_bool(engine.get("strict_roads"), False)
    world = World(env, _rule_grid_for(grid), dt_s=dt_s, seed=seed, plan_grid=grid,
                  nav=nav,
                  detection_model=detection_model,
                  direct_fire_model=direct_fire_model,
                  indirect_fire_model=indirect_fire_model,
                  missile_model=missile_model,
                  enable_radio_share=_as_bool(engine.get("enable_radio_share"), False),
                  enable_tactics=_as_bool(engine.get("enable_tactics"), False),
                  commander_interval_s=commander_interval_s,
                  scan_interval_s=scan_interval_s)
    world.max_ineffective_shots = int(_as_float(engine.get("max_ineffective_shots"), 0))
    world.ineffective_block_s = _as_float(engine.get("ineffective_block_s"), 1800.0)
    weather_base = scenario.get("weather_base")
    weather_timeline = scenario.get("weather")
    start_hour = scenario.get("start_hour_utc")
    utc_offset = scenario.get("utc_offset_h")
    if (isinstance(weather_base, dict) or isinstance(weather_timeline, list)
            or start_hour is not None or utc_offset is not None):
        world.weather_enabled = True
    world.start_hour_utc = _as_float(start_hour, 12.0)
    world.utc_offset_h = _as_float(utc_offset, 0.0)
    base = weather_base if isinstance(weather_base, dict) else {}
    for key in ("visibility_m", "sea_state", "wind_ms"):
        if base.get(key) is not None:
            try:
                world.weather[key] = float(base[key])
            except (TypeError, ValueError):
                pass
    for item in (weather_timeline if isinstance(weather_timeline, list) else []):
        if not isinstance(item, dict):
            continue
        entry = {"at_s": _as_float(item.get("at_s"), 0.0)}
        for key in ("visibility_m", "sea_state", "wind_ms"):
            if item.get(key) is not None:
                try:
                    entry[key] = float(item[key])
                except (TypeError, ValueError):
                    continue
        world.weather_timeline.append(entry)
    world.weather_timeline.sort(key=lambda e: e["at_s"])
    if _as_bool(engine.get("enable_commander"), False):
        from sim.commander import RuleCommander
        world.set_commander(RuleCommander(), commander_interval_s)
    for item in scenario.get("minefields") or []:
        ring = _region_ring(item)
        if ring:
            world.minefields.append({
                "name": str(item.get("name") or "minefield"),
                "kind": str(item.get("kind") or "临时雷场"),
                "owner": int(item.get("owner", -1)),
                "ring": ring})
    for item in scenario.get("obstacles") or []:
        ring = _region_ring(item)
        if ring:
            world.obstacles.append({
                "name": str(item.get("name") or "obstacle"),
                "kind": str(item.get("kind") or "roadblock"),
                "ring": ring})
    for item in scenario.get("obscurants") or []:
        if "lon" not in item or "lat" not in item:
            continue
        radius = _as_float(item.get("radius_m"), 300.0)
        base = _as_float(item.get("base_m"), 0.0)
        top = _as_float(item.get("top_m"), 200.0)
        if radius <= 0.0:
            continue
        world.obscurants.append({
            "name": str(item.get("name") or "smoke"),
            "lon": float(item["lon"]),
            "lat": float(item["lat"]),
            "radius_m": radius,
            "base_m": min(base, top),
            "top_m": max(base, top),
            "from_s": _as_float(item.get("from_s"), 0.0),
            "to_s": _as_float(item.get("to_s"), 1e18),
            "attenuation": _as_float(item.get("attenuation"), 0.02),
        })
    for op in scenario.get("operators") or []:
        unit_type = _unit_type(op)
        domain = _domain(str(op.get("name") or ""), int(op.get("type") or 2), unit_type)
        if op.get("domain"):
            domain = str(op["domain"])
            if domain == "sea":
                domain = "surface"
            elif domain == "sub":
                domain = "subsurface"
        lon, lat = _op_lonlat(op, grid)
        card_id, card = card_for_operator(op, domain, infer=use_role_cards)
        if card:
            domain = domain_from_card(card) or domain
            unit_type = unit_type_from_card(card_id)
            profile = profile_from_card(card, key=card_id)
        else:
            profile = profile_for(unit_type, domain)
        # Scenario-declared GLB footprint wins over the card defaults, so the
        # collision body and the rendered model agree.
        if op.get("length_m") is not None:
            try:
                profile = replace(profile,
                                  length_m=float(op["length_m"]),
                                  width_m=float(op.get("width_m") or profile.width_m))
            except (TypeError, ValueError):
                pass
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
            pose=Pose(lon, lat,
                      _as_float(op.get("alt_m"), env.elevation(lon, lat)),
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
    # Explicit carrier assignment (new-engine scenarios).  This is the
    # authoritative path for "loaded on ship X"; the depth heuristic below only
    # fills in legacy fixtures that do not declare it.
    explicit_carriers = {}
    for op in scenario.get("operators") or []:
        if op.get("carrier") is None:
            continue
        explicit_carriers[int(op["obj_id"])] = int(op["carrier"])
    for uid, cuid in sorted(explicit_carriers.items()):
        passenger = world.entities.get(uid)
        carrier = world.entities.get(cuid)
        if passenger is None or carrier is None:
            raise ValueError("carrier reference %s -> %s not resolvable" % (uid, cuid))
        if carrier.domain != "surface":
            raise ValueError("carrier %s is not a surface ship" % cuid)
        if passenger.affiliation != carrier.affiliation:
            raise ValueError("carrier %s and passenger %s have different sides" % (cuid, uid))
        passenger.on_board = True
        passenger.carrier = cuid
        passenger.pose = Pose(carrier.pose.lon, carrier.pose.lat, carrier.pose.alt_m,
                              heading_deg=carrier.pose.heading_deg)
        carrier.passengers.append(uid)

    # Ground and amphibious units that start in deep water ride a friendly
    # ship until it can put them ashore (landing craft chain).
    ships = [e for e in world.entities.values() if e.domain == "surface"]
    for entity in world.entities.values():
        if entity.uid in explicit_carriers:
            continue
        if entity.on_board:
            continue
        if entity.domain not in ("land", "amphibious"):
            continue
        depth = env.water_depth(entity.pose.lon, entity.pose.lat)
        max_depth = (entity.profile.swim_depth_m
                     if entity.domain == "amphibious"
                     else entity.profile.wading_depth_m)
        if depth is None or depth <= max_depth:
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
    _spread_stacked_spawns(world)
    world._refresh_cell_cache()
    world.prepare_first_scan()
    for city in scenario.get("cities") or []:
        if city.get("lon") is not None and city.get("lat") is not None:
            lon, lat = float(city["lon"]), float(city["lat"])
        else:
            coord = city.get("coord")
            if coord is None:
                continue
            try:
                lon, lat = grid.to_lonlat(int(coord))
            except (TypeError, ValueError):
                continue
        owner = city.get("owner", -1)
        try:
            owner = int(owner)
        except (TypeError, ValueError):
            owner = -1
        if owner not in (0, 1):
            owner = -1
        world.add_objective(world.rule_grid.cell_of(lon, lat),
                            value=int(city.get("value") or 0),
                            name=str(city.get("name") or ""), owner=owner)
    for obj in scenario.get("objectives") or []:
        if obj.get("lon") is None or obj.get("lat") is None:
            continue
        try:
            owner = int(obj.get("owner", -1))
        except (TypeError, ValueError):
            owner = -1
        if owner not in (0, 1):
            owner = -1
        world.add_objective(world.rule_grid.cell_of(float(obj["lon"]), float(obj["lat"])),
                            value=int(obj.get("value") or 0),
                            name=str(obj.get("name") or ""), owner=owner)
    for sp in scenario.get("supply_points") or []:
        if isinstance(sp, dict) and "lon" in sp and "lat" in sp:
            world.supply_points.append((float(sp["lon"]), float(sp["lat"])))
    return world


def _spread_stacked_spawns(world: World) -> None:
    """Deterministically de-overlap units whose OBBs intersect at spawn.

    Only actually-overlapping units are moved (a deterministic ring search),
    offsets stay inside the same 200 m rule cell, and the point must be
    passable per the nav field (else the base map).  Scenario headings are
    preserved.
    """
    import math
    from sim.collision import obbs_overlap
    from sim.motion import entity_profile

    groups = {}
    for e in world.sorted_entities():
        if not e.alive or e.on_board or e.domain not in ("land", "amphibious"):
            continue
        key = world.rule_grid.cell_of(e.pose.lon, e.pose.lat)
        groups.setdefault(key, []).append(e)
    kx = max(1.0, 111320.0 * math.cos(math.radians(world.rule_grid.center_lat)))
    ky = 110540.0
    for key in sorted(groups):
        members = groups[key]
        if len(members) < 2:
            continue
        placed = [members[0]]
        for e in members[1:]:
            prof = entity_profile(e)

            def _overlaps(lon, lat):
                return any(obbs_overlap(
                    prof, lon, lat, e.pose.heading_deg,
                    entity_profile(other), other.pose.lon, other.pose.lat,
                    other.pose.heading_deg) for other in placed)

            if not _overlaps(e.pose.lon, e.pose.lat):
                placed.append(e)
                continue
            seat = None
            for radius in (12.0, 24.0, 36.0):
                for k in range(8):
                    angle = math.radians(45.0 * k)
                    lon = e.pose.lon + radius * math.cos(angle) / kx
                    lat = e.pose.lat + radius * math.sin(angle) / ky
                    if world.rule_grid.cell_of(lon, lat) != key:
                        continue
                    if world.nav is not None and world.nav.covers(lon, lat):
                        if not world.nav.motion_point_ok(prof, lon, lat):
                            continue
                    else:
                        from mobility.passability import passable
                        if not passable(prof, lon, lat, world.env,
                                        world.env.elevation(lon, lat)):
                            continue
                    if _overlaps(lon, lat):
                        continue
                    seat = (lon, lat)
                    break
                if seat is not None:
                    break
            if seat is not None:
                e.pose = Pose(seat[0], seat[1],
                              world.env.elevation(seat[0], seat[1]),
                              heading_deg=e.pose.heading_deg)
            placed.append(e)


def load_world(scenario_path: str, map_path: str, seed: int = 20261021,
               dt_s: float = 1.0, envpack: Optional[str] = None) -> World:
    with open(scenario_path, encoding="utf-8") as fh:
        scenario = json.load(fh)
    with open(map_path, encoding="utf-8") as fh:
        basic = json.load(fh)
    return build_world(scenario, basic, seed=seed, dt_s=dt_s, envpack=envpack)
