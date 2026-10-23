"""TW2 initial scenario for the geo engine (dual-mode test runs).

Only the *start* is described here: order of battle, initial positions, role
cards, objectives, missions and ROE.  No waypoints, per-tick positions,
release ticks or scripted events.  The engine decides how units move, fight
and complete tasks; ``scripts/run_situation_case.py`` exports the result.

All units and parameters are synthetic demonstration data.
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from engine import roads  # noqa: E402
from geo.geodesy import haversine_m  # noqa: E402

CASE_ID = "taiwan_dense_transfer_v1"
_BUILT = None
_OBJ_CACHE = None
THEATRE = "taiwan_strait"
TICK_SECONDS = 30
TICK_COUNT = 1440
TERMINAL_TICK = 1440

AREAS = {
    "D": {"name": "D区岸海衔接", "lon": 120.162500, "lat": 22.977500,
          "assembly": (120.1600, 22.9720)},
    "E": {"name": "E区岸上任务", "lon": 120.182500, "lat": 22.967500,
          "assembly": (120.1880, 22.9620)},
    "F": {"name": "F区保障", "lon": 120.202500, "lat": 22.962500,
          "assembly": (120.2030, 22.9570)},
}

BERTHS = [(120.135, 22.960), (120.135, 22.968),
          (120.135, 22.976), (120.135, 22.984)]

PHASES = [
    {"id": "S0", "name": "建立态势", "first_tick": 0},
    {"id": "S1", "name": "多域活动增强", "first_tick": 120},
    {"id": "S2", "name": "转运关系显现", "first_tick": 360},
    {"id": "S3", "name": "岸海衔接", "first_tick": 600},
    {"id": "S4", "name": "岸上任务与保障", "first_tick": 840},
    {"id": "S5", "name": "稳态评估", "first_tick": 1000},
]

STAGES = [
    {"id": "S0", "name": "建立态势", "start_tick": 0, "end_tick": 120},
    {"id": "S1", "name": "多域活动增强", "start_tick": 120, "end_tick": 360},
    {"id": "S2", "name": "转运关系显现", "start_tick": 360, "end_tick": 600},
    {"id": "S3", "name": "岸海衔接", "start_tick": 600, "end_tick": 840},
    {"id": "S4", "name": "岸上任务与保障", "start_tick": 840, "end_tick": 1000},
    {"id": "S5", "name": "稳态评估", "start_tick": 1000, "end_tick": TICK_COUNT + 1},
]

CLASSES = {
    "amphibious": {
        "model": "ZTD-05 两栖突击车", "icon": "light_vehicle",
        "platform": "两栖突击车", "subclass": "land.vehicle.amphibious",
        "role": "地面机动", "sensor_km": 6.0, "engage_km": 2.5,
        "speed_kn": 22.0, "length_m": 7.2, "width_m": 3.3, "height_m": 3.0,
    },
    "truck": {
        "model": "东风 EQ240 运输车", "icon": "truck",
        "platform": "运输车", "subclass": "land.vehicle.truck",
        "role": "地面运输", "sensor_km": 6.0, "engage_km": 1.5,
        "speed_kn": 19.0, "length_m": 7.4, "width_m": 2.5, "height_m": 3.1,
    },
    "engineer": {
        "model": "EQ240 工程车", "icon": "engineer_vehicle",
        "platform": "工程车", "subclass": "land.vehicle.engineer",
        "role": "地面保障", "sensor_km": 5.0, "engage_km": 1.2,
        "speed_kn": 18.0, "length_m": 7.0, "width_m": 2.5, "height_m": 3.0,
    },
    "carrier": {
        "model": "071 型船坞登陆舰", "icon": "replenishment",
        "platform": "船坞登陆舰", "subclass": "sea.surface.landing_ship",
        "role": "转运载体", "sensor_km": 80.0, "engage_km": 20.0,
        "speed_kn": 7.0, "length_m": 210.0, "width_m": 28.0, "height_m": 20.0,
    },
    "frigate_red": {
        "model": "054A 型护卫舰", "icon": "frigate",
        "platform": "护卫舰", "subclass": "sea.surface.frigate",
        "role": "近海巡逻", "sensor_km": 80.0, "engage_km": 20.0,
        "speed_kn": 10.0, "length_m": 134.0, "width_m": 16.0, "height_m": 12.0,
    },
    "corvette_red": {
        "model": "056A 型轻型护卫舰", "icon": "corvette",
        "platform": "轻型护卫舰", "subclass": "sea.surface.corvette",
        "role": "近海巡逻", "sensor_km": 60.0, "engage_km": 15.0,
        "speed_kn": 12.0, "length_m": 89.0, "width_m": 11.0, "height_m": 9.0,
    },
    "frigate_blue": {
        "model": "成功级护卫舰", "icon": "frigate",
        "platform": "护卫舰", "subclass": "sea.surface.frigate",
        "role": "近海巡逻", "sensor_km": 80.0, "engage_km": 20.0,
        "speed_kn": 10.0, "length_m": 125.0, "width_m": 13.0, "height_m": 11.0,
    },
    "corvette_blue": {
        "model": "沱江级巡逻舰", "icon": "corvette",
        "platform": "巡逻舰", "subclass": "sea.surface.corvette",
        "role": "近海巡逻", "sensor_km": 60.0, "engage_km": 15.0,
        "speed_kn": 14.0, "length_m": 65.0, "width_m": 9.0, "height_m": 7.0,
    },
    "fighter_red": {
        "model": "歼-16", "icon": "fighter",
        "platform": "战斗机", "subclass": "air.aircraft.fighter",
        "role": "空中掩护", "sensor_km": 150.0, "engage_km": 60.0,
        "speed_kn": 330.0, "length_m": 22.0, "width_m": 14.0, "height_m": 6.0,
    },
    "fighter_blue": {
        "model": "F-16C", "icon": "fighter",
        "platform": "战斗机", "subclass": "air.aircraft.fighter",
        "role": "空中掩护", "sensor_km": 150.0, "engage_km": 60.0,
        "speed_kn": 330.0, "length_m": 15.0, "width_m": 10.0, "height_m": 5.0,
    },
    "aew_red": {
        "model": "空警-500", "icon": "aew",
        "platform": "预警机", "subclass": "air.aircraft.aew",
        "role": "空中警戒", "sensor_km": 240.0, "engage_km": 0.0,
        "speed_kn": 300.0, "length_m": 34.0, "width_m": 38.0, "height_m": 11.0,
    },
    "aew_blue": {
        "model": "E-2D", "icon": "aew",
        "platform": "预警机", "subclass": "air.aircraft.aew",
        "role": "空中警戒", "sensor_km": 240.0, "engage_km": 0.0,
        "speed_kn": 300.0, "length_m": 18.0, "width_m": 25.0, "height_m": 6.0,
    },
}


def _road_nodes(net, lon, lat, count, radius_m=900.0):
    """Deterministic, connected road nodes near an assembly point."""
    start = net.nearest_node(lat, lon, max_km=3.0)
    if start is None:
        return [(lon, lat)] * count
    seen = {start}
    frontier = [start]
    for _ in range(4):
        nxt = []
        for u in frontier:
            for v, _dist, _kmh in net.adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                nxt.append(v)
        frontier = nxt
    ranked = []
    for i in seen:
        nlon, nlat = net.nodes[i]
        d = haversine_m((lon, lat), (nlon, nlat))
        if d <= radius_m:
            ranked.append((d, i, nlon, nlat))
    ranked.sort()
    if not ranked:
        nlon, nlat = net.nodes[start]
        return [(nlon, nlat)] * count
    out = []
    for k in range(count):
        d, _i, nlon, nlat = ranked[k % len(ranked)]
        out.append((nlon, nlat))
    return out


def objective_points():
    """Objective points snapped to the road network (convoy-strict TW2)."""
    global _OBJ_CACHE
    if _OBJ_CACHE is None:
        net = roads.load(THEATRE)
        _OBJ_CACHE = {k: _road_nodes(net, a["lon"], a["lat"], 1)[0]
                      for k, a in AREAS.items()}
    return _OBJ_CACHE


def _op(uid, case_id, cls_key, color, domain, lon, lat, alt=0.0,
        heading=90.0, blood=2, carrier=None, group="", task=""):
    cls = CLASSES[cls_key]
    op = {
        "obj_id": uid,
        "name": cls["model"],
        "color": color,
        "type": 3 if domain == "air" else 2,
        "sub_type": 0,
        "domain": domain,
        "lon": round(float(lon), 6),
        "lat": round(float(lat), 6),
        "alt_m": float(alt),
        "heading_deg": float(heading),
        "subclass_id": cls["subclass"],
        "length_m": float(cls["length_m"]),
        "width_m": float(cls["width_m"]),
        "height_m": float(cls["height_m"]),
        "blood": int(blood),
        "max_blood": int(blood),
        "value": 40 if domain == "air" else (30 if domain == "surface" else 10),
        "emitting": True,
    }
    if carrier is not None:
        op["carrier"] = int(carrier)
    return op


def build():
    global _BUILT
    if _BUILT is not None:
        return _BUILT
    net = roads.load(THEATRE)
    operators = []
    meta = {}

    def add(uid, case_id, cls_key, color, domain, lon, lat, **kw):
        operators.append(_op(uid, case_id, cls_key, color, domain, lon, lat, **kw))
        cls = CLASSES[cls_key]
        meta[case_id] = {
            "uid": uid, "name": cls["model"], "platform_class": cls["platform"],
            "domain": domain, "affiliation": color, "role": cls["role"],
            "model": cls["model"], "icon": cls["icon"],
            "sensor_km": cls["sensor_km"], "engage_km": cls["engage_km"],
            "influence_km": (120.0 if domain == "air" else
                             (40.0 if domain == "surface" else 0.5)),
            "speed_kn": cls["speed_kn"],
            "length_m": cls["length_m"], "width_m": cls["width_m"],
            "height_m": cls["height_m"],
            "subclass_id": cls["subclass"], "group": kw.get("group", ""),
            "task": kw.get("task", ""),
        }

    # Transfer ships and escorts (red).
    for i, (lon, lat) in enumerate([(120.095, 22.962), (120.097, 22.966),
                                    (120.099, 22.970), (120.101, 22.974)]):
        uid = 1001 + i
        add(uid, "TRS%03d" % (i + 1), "carrier", 0, "surface", lon, lat,
            heading=90.0, blood=6, group="TRS", task="转运平台待机")
    add(1005, "TRS005", "frigate_red", 0, "surface", 120.085, 22.948,
        heading=45.0, blood=4, group="TRS", task="海上护航")
    add(1006, "TRS006", "corvette_red", 0, "surface", 120.090, 22.935,
        heading=45.0, blood=3, group="TRS", task="海上护航")

    # Blue surface group.
    blue_ships = [("TBS001", "frigate_blue", 120.075, 22.935),
                  ("TBS002", "frigate_blue", 120.078, 22.950),
                  ("TBS003", "corvette_blue", 120.082, 22.965),
                  ("TBS004", "corvette_blue", 120.086, 22.980),
                  ("TBS005", "frigate_blue", 120.080, 22.995),
                  ("TBS006", "corvette_blue", 120.092, 22.988)]
    for i, (cid, key, lon, lat) in enumerate(blue_ships):
        add(1101 + i, cid, key, 1, "surface", lon, lat,
            heading=270.0, blood=4 if "frigate" in key else 3,
            group="TBS", task="海上巡逻")

    # Red ground groups.  A/B are loaded on TRS001/TRS002 and unload when the
    # ships reach the berth; the rest start ashore.
    vehicle_specs = []
    for k in range(8):
        vehicle_specs.append(("TRV%03d" % (k + 1), "amphibious", 1001, "TRV-A", "地面机动"))
    for k in range(8):
        vehicle_specs.append(("TRV%03d" % (k + 9), "amphibious", 1002, "TRV-B", "地面机动"))
    for k in range(6):
        vehicle_specs.append(("TRV%03d" % (k + 17), "truck" if k >= 4 else "amphibious",
                              None, "TRV-C", "地面运输"))
    for k in range(6):
        vehicle_specs.append(("TRV%03d" % (k + 23), "truck" if k >= 4 else "amphibious",
                              None, "TRV-D", "地面机动"))
    for k in range(4):
        vehicle_specs.append(("TRV%03d" % (k + 29), "truck", None, "TRV-E", "地面运输"))
    for k in range(4):
        vehicle_specs.append(("TRV%03d" % (k + 33), "engineer", None, "TRV-F", "地面保障"))

    group_assembly = {
        "TRV-C": AREAS["D"]["assembly"], "TRV-D": AREAS["D"]["assembly"],
        "TRV-E": AREAS["E"]["assembly"], "TRV-F": AREAS["F"]["assembly"],
    }
    group_nodes = {}
    group_count = {}
    for cid, key, carrier, group, task in vehicle_specs:
        if carrier is not None:
            continue
        if group not in group_nodes:
            lon, lat = group_assembly[group]
            group_nodes[group] = _road_nodes(net, lon, lat, 12)
            group_count[group] = 0
        idx = group_count[group]
        group_count[group] += 1
        lon, lat = group_nodes[group][idx % len(group_nodes[group])]
        uid = int(cid[3:]) + 2000
        add(uid, cid, key, 0, "land", lon, lat, heading=90.0, blood=2,
            group=group, task=task)
    for i, (cid, key, carrier, group, task) in enumerate(vehicle_specs):
        if carrier is None:
            continue
        uid = int(cid[3:]) + 2000
        lon, lat = (120.095, 22.962) if carrier == 1001 else (120.097, 22.966)
        add(uid, cid, key, 0, "land", lon, lat, heading=90.0, blood=2,
            carrier=carrier, group=group, task=task)

    blue_ground = [
        ("TBV%03d" % (k + 1), "amphibious" if k < 4 else "truck", (120.178, 22.976), "TBV-A") for k in range(8)
    ] + [
        ("TBV%03d" % (k + 9), "amphibious" if k < 4 else "truck", (120.193, 22.968), "TBV-B") for k in range(6)
    ] + [
        ("TBV%03d" % (k + 15), "amphibious" if k < 4 else "truck", (120.207, 22.962), "TBV-C") for k in range(6)
    ] + [
        ("TBV%03d" % (k + 21), "engineer", (120.214, 22.952), "TBV-D") for k in range(4)
    ]
    bgroup_nodes = {}
    bgroup_count = {}
    for cid, key, assembly, group in blue_ground:
        if group not in bgroup_nodes:
            bgroup_nodes[group] = _road_nodes(net, assembly[0], assembly[1], 12)
            bgroup_count[group] = 0
        idx = bgroup_count[group]
        bgroup_count[group] += 1
        lon, lat = bgroup_nodes[group][idx % len(bgroup_nodes[group])]
        add(int(cid[3:]) + 3000, cid, key, 1, "land", lon, lat,
            heading=270.0, blood=2, group=group, task="地面防御")

    # Aircraft (inside the envpack AOI so the demo camera keeps them in view).
    air = [("TRA001", "fighter_red", 0, 120.120, 22.990, 5200),
           ("TRA002", "fighter_red", 0, 120.150, 23.000, 5200),
           ("TRA003", "aew_red", 0, 120.180, 23.005, 6500),
           ("TRA004", "aew_red", 0, 120.210, 22.990, 6500),
           ("TBA001", "fighter_blue", 1, 120.230, 22.970, 4800),
           ("TBA002", "fighter_blue", 1, 120.225, 22.945, 4800),
           ("TBA003", "aew_blue", 1, 120.200, 22.930, 6200),
           ("TBA004", "aew_blue", 1, 120.170, 22.925, 6200)]
    for i, (cid, key, color, lon, lat, alt) in enumerate(air):
        add(4001 + i, cid, key, color, "air", lon, lat, alt=alt,
            heading=0.0, blood=2, group="TRA" if color == 0 else "TBA",
            task="空中活动")

    scenario = {
        "case_id": CASE_ID,
        "tick_seconds": TICK_SECONDS,
        "duration_s": TICK_SECONDS * TICK_COUNT,
        "geo_anchor": {"lon": 120.17, "lat": 22.97, "alt": 0.0},
        "start_time_utc": "2026-09-15T00:00:00Z",
        "engine": {
            "detection_model": "physical",
            "direct_fire_model": "physics",
            "indirect_fire_model": "physics",
            "missile_model": "flight",
            "scan_interval_s": 180,
            "max_ineffective_shots": 8,
            "ineffective_block_s": 1800.0,
            "nav_res_m": 30,
            "strict_roads": False,
            "use_role_cards": True,
            "enable_radio_share": True,
            "enable_commander": False,
            "enable_tactics": False,
        },
        "operators": operators,
        "objectives": [
            {"name": AREAS[k]["name"], "lon": objective_points()[k][0],
             "lat": objective_points()[k][1], "owner": -1, "value": 10}
            for k in sorted(AREAS)
        ],
        "missions": missions(),
        "roe": {"halt_on_contact": True, "engage_m": 2500.0,
                "resume_after_s": 60.0},
    }
    _BUILT = (scenario, meta, scenario["missions"])
    return _BUILT


def patrol_point(key, dlon, dlat):
    net = roads.load(THEATRE)
    base = objective_points()[key]
    return _road_nodes(net, base[0] + dlon, base[1] + dlat, 1)[0]


def missions():
    red_d_combat = ["TRV%03d" % i for i in range(1, 9)] + ["TRV%03d" % i for i in range(23, 27)]
    red_d_support = ["TRV%03d" % i for i in range(27, 29)]
    red_e_combat = ["TRV%03d" % i for i in range(9, 17)] + ["TRV%03d" % i for i in range(17, 21)]
    red_e_support = ["TRV%03d" % i for i in range(21, 23)]
    red_f_support = ["TRV%03d" % i for i in range(29, 37)]
    blue_d_combat = ["TBV%03d" % i for i in range(1, 5)]
    blue_d_support = ["TBV%03d" % i for i in range(5, 9)]
    blue_e_combat = ["TBV%03d" % i for i in range(9, 13)]
    blue_e_support = ["TBV%03d" % i for i in range(13, 15)]
    blue_f_combat = ["TBV%03d" % i for i in range(15, 19)]
    blue_f_support = ["TBV%03d" % i for i in range(19, 25)]
    by_id = {}
    for i in range(1, 37):
        by_id["TRV%03d" % i] = 2000 + i
    for i in range(1, 25):
        by_id["TBV%03d" % i] = 3000 + i

    def units(ids):
        return [by_id[x] for x in ids]

    out = []
    for i, (lon, lat) in enumerate(BERTHS):
        out.append({"id": "R-CARRIER-%d" % (i + 1), "side": 0,
                    "units": [1001 + i], "task": "move_to",
                    "objective": "岸海衔接泊位 %d" % (i + 1),
                    "lon": lon, "lat": lat, "start_s": 0 if i == 0 else 3600 * i,
                    "pace": "normal", "arrive_m": 300.0})

    def add_mission(mid, side, ids, task, obj_key, lon, lat,
                    after=None, pace="normal"):
        item = {"id": mid, "side": side, "units": units(ids), "task": task,
                "objective": AREAS[obj_key]["name"], "lon": lon, "lat": lat,
                "pace": pace}
        if after:
            item["after"] = after
        out.append(item)

    d_lon, d_lat = objective_points()["D"]
    e_lon, e_lat = objective_points()["E"]
    f_lon, f_lat = objective_points()["F"]
    add_mission("R-SEIZE-D", 0, red_d_combat, "seize", "D", d_lon, d_lat,
                after=["R-CARRIER-1"], pace="dash")
    add_mission("R-SUPPORT-D", 0, red_d_support, "hold", "D",
                *patrol_point("D", -0.0020, -0.0040), after=["R-CARRIER-1"])
    add_mission("R-SEIZE-E", 0, red_e_combat, "seize", "E", e_lon, e_lat,
                after=["R-CARRIER-2"])
    add_mission("R-SUPPORT-E", 0, red_e_support, "hold", "E",
                *patrol_point("E", -0.0020, -0.0040), after=["R-CARRIER-2"])
    add_mission("R-SUPPORT-F", 0, red_f_support, "hold", "F",
                *patrol_point("F", -0.0020, -0.0040), pace="cautious")

    add_mission("B-HOLD-D", 1, blue_d_combat, "hold", "D",
                *patrol_point("D", 0.0030, 0.0025))
    add_mission("B-SUPPORT-D", 1, blue_d_support, "hold", "D",
                *patrol_point("D", 0.0020, 0.0055))
    add_mission("B-HOLD-E", 1, blue_e_combat, "hold", "E",
                *patrol_point("E", 0.0030, 0.0025))
    add_mission("B-SUPPORT-E", 1, blue_e_support, "hold", "E",
                *patrol_point("E", 0.0020, 0.0055))
    add_mission("B-HOLD-F", 1, blue_f_combat, "hold", "F",
                *patrol_point("F", 0.0030, 0.0025))
    add_mission("B-SUPPORT-F", 1, blue_f_support, "hold", "F",
                *patrol_point("F", 0.0020, 0.0055))

    out.append({"id": "R-SEA", "side": 0, "units": [1005, 1006],
                "task": "hold", "objective": "海上警戒区",
                "lon": 120.085, "lat": 22.945, "pace": "normal"})
    out.append({"id": "R-AIR", "side": 0, "units": [4001, 4002, 4003, 4004],
                "task": "hold", "objective": "北部空域",
                "lon": 120.150, "lat": 23.000, "pace": "normal"})
    out.append({"id": "B-SEA", "side": 1,
                "units": [1101, 1102, 1103, 1104, 1105, 1106],
                "task": "hold", "objective": "南部海区",
                "lon": 120.080, "lat": 22.940, "pace": "normal"})
    out.append({"id": "B-AIR", "side": 1,
                "units": [4005, 4006, 4007, 4008],
                "task": "hold", "objective": "东南空域",
                "lon": 120.215, "lat": 22.940, "pace": "normal"})
    return out


def scenario_dict():
    scenario, _meta, _missions = build()
    return scenario


def target_meta():
    _scenario, meta, _missions = build()
    return meta


def case_meta():
    return {
        "case_id": CASE_ID,
        "case_name": "TW2",
        "case_description": "多域合成目标在安平真实路网与envpack环境上的转运、衔接、保障与交战演示（引擎推演）。",
        "region": "中国 台湾海峡",
        "theatre": THEATRE,
        "tick_seconds": TICK_SECONDS,
        "tick_count": TICK_COUNT,
        "terminal_tick": TERMINAL_TICK,
        "start_time_utc": "2026-09-15T00:00:00Z",
        "camera": {"lon": 120.17, "lat": 22.972, "height": 12000.0,
                   "heading": 0.0, "pitch": -60.0},
        "phases": PHASES,
        "stages": STAGES,
        "hotspots": [
            {"name": AREAS[k]["name"],
             "bbox": [AREAS[k]["lon"] - 0.012, AREAS[k]["lat"] - 0.010,
                      AREAS[k]["lon"] + 0.012, AREAS[k]["lat"] + 0.010],
             "center": [AREAS[k]["lat"], AREAS[k]["lon"]],
             "span_deg": [0.024, 0.020],
             "zoom": 13}
            for k in sorted(AREAS)
        ],
        "provenance_note": "合成目标与虚构参数；真实地理仅作背景。",
    }


def objectives():
    return [{"name": AREAS[k]["name"], "lon": objective_points()[k][0],
             "lat": objective_points()[k][1], "owner": -1, "value": 10}
            for k in sorted(AREAS)]
