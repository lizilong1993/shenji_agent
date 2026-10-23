"""中印西段（虚构演示）——campaign plugin for run_geo_demo.py.

All units, objectives and events are synthetic demonstration data for the
continuous geo engine; they are not real deployments or predictions.

Theatre: a 400 x 400 km high-altitude sector around (79.1E, 33.75N) with a
radar-derived 4 km grid (see ``scripts/build_campaign_map.py`` profile
``india_china_west``).  Stages: patrol/shaping -> pass seizure -> counter-
attack -> consolidation.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
REPO = os.path.dirname(SITUATION3D)
sys.path.insert(0, SITUATION3D)

import campaign as C  # noqa: E402

MAP_ID = 9002
MAP_PATH = os.path.join(REPO, "situation3d", "web", "data",
                        "campaign_maps", "map_9002", "basic.json")

# Hex anchors (row * 100 + col) on the 100 x 100 / 4 km theatre.
P = {
    "red_base": 7070,        # 红方后方集结地（东南）
    "red_airfield": 7575,
    "red_artillery": 6058,
    "pass_west": 4858,       # 山口西侧（接近通路）
    "pass_east": 5052,       # 山口东侧
    "ridge_north": 3248,     # 北侧高地
    "ridge_south": 6448,     # 南侧高地
    "blue_base": 3030,       # 蓝方后方基地（西北）
    "blue_depot": 3540,
    "blue_airfield": 2535,
    "valley": 5042,          # 谷地通路
    "lake": 5560,            # 湖盆（display only）
}


def _op(uid, name, color, op_type, sub_type, pos, blood, domain="land",
        armor=2, value=10, size_class="medium", altitude_band="ground"):
    return {
        "obj_id": uid, "name": name, "color": color, "type": op_type,
        "sub_type": sub_type, "cur_hex": pos, "blood": blood,
        "max_blood": blood, "armor": armor, "basic_speed": 25,
        "value": value, "domain": domain, "altitude_band": altitude_band,
        "size_class": size_class, "observe_distance": {"1": 10, "2": 25},
        "move_state": 0, "stop": 1, "readiness": 0.85, "fuel": 1.0,
        "emitting": True,
    }


def scenario_dict() -> dict:
    """Initial order of battle (fictional)."""
    red = [
        _op(3001, "重型坦克", 0, 2, 0, P["red_base"], 4, armor=4, value=15),
        _op(3002, "重型坦克", 0, 2, 0, P["red_base"], 4, armor=4, value=15),
        _op(3003, "中型战车", 0, 2, 1, P["red_base"], 3, armor=2, value=12),
        _op(3004, "步兵小队", 0, 1, 2, P["red_base"], 4, armor=0, value=8,
            size_class="small"),
        _op(3005, "步兵小队", 0, 1, 2, P["red_base"], 4, armor=0, value=8,
            size_class="small"),
        _op(3006, "炮兵", 0, 2, 3, P["red_artillery"], 3, armor=1, value=14,
            size_class="large"),
        _op(3007, "防空导弹排", 0, 1, 16, P["pass_east"], 2, armor=0, value=15),
        _op(3008, "无人战车", 0, 2, 0, P["pass_east"], 2, armor=1, value=10,
            size_class="small"),
        _op(3010, "无人机", 0, 3, 5, P["red_airfield"], 2, domain="air",
            armor=1, value=8, size_class="small", altitude_band="低空"),
        _op(3011, "战斗机", 0, 3, 0, P["red_airfield"], 3, domain="air",
            armor=1, value=30, size_class="small", altitude_band="中空"),
    ]
    blue = [
        _op(3101, "重型坦克", 1, 2, 0, P["blue_base"], 4, armor=4, value=15),
        _op(3102, "中型战车", 1, 2, 1, P["blue_base"], 3, armor=2, value=12),
        _op(3103, "中型战车", 1, 2, 1, P["blue_base"], 3, armor=2, value=12),
        _op(3104, "步兵小队", 1, 1, 2, P["valley"], 4, armor=0, value=8,
            size_class="small"),
        _op(3105, "步兵小队", 1, 1, 2, P["ridge_north"], 4, armor=0, value=8,
            size_class="small"),
        _op(3106, "炮兵", 1, 2, 3, P["blue_depot"], 3, armor=1, value=14,
            size_class="large"),
        _op(3107, "防空导弹排", 1, 1, 16, P["blue_depot"], 2, armor=0, value=15),
        _op(3108, "侦察型战车", 1, 2, 0, P["pass_west"], 2, armor=1, value=10,
            size_class="small"),
        _op(3110, "无人机", 1, 3, 5, P["blue_airfield"], 2, domain="air",
            armor=1, value=8, size_class="small", altitude_band="低空"),
        _op(3111, "战斗机", 1, 3, 0, P["blue_airfield"], 3, domain="air",
            armor=1, value=30, size_class="small", altitude_band="中空"),
    ]
    return {
        "scenario_id": MAP_ID,
        "time": {"cur_step": 0, "tick": 1.0, "max_time": 1440},
        "cities": [],
        "operators": red + blue,
        "supply_points": [
            {"color": 0, "pos": P["red_base"], "kind": "depot"},
            {"color": 0, "pos": P["red_airfield"], "kind": "airbase"},
            {"color": 1, "pos": P["blue_base"], "kind": "depot"},
            {"color": 1, "pos": P["blue_airfield"], "kind": "airbase"},
        ],
    }


def build_spec() -> C.CampaignSpec:
    stages = [
        C.Stage("patrol", "侦察对峙", 0, 0, 180, "engine",
                "双方前出侦察，争夺山口与高地观察位。",
                "接触线形成。", camera={"lon": 79.1, "lat": 33.75,
                                        "height": 380000.0, "heading": 0.0,
                                        "pitch": -70.0}),
        C.Stage("seize", "夺取山口", 1, 180, 540, "adjudicated",
                "红方沿谷地向山口推进，蓝方依托高地迟滞。",
                "山口争夺激烈。", camera={"lon": 79.1, "lat": 33.75,
                                           "height": 220000.0, "heading": 0.0,
                                           "pitch": -60.0}),
        C.Stage("counter", "高原反击", 2, 540, 1020, "adjudicated",
                "蓝方装甲预备队沿北侧高地实施反击。",
                "双方在谷地展开装甲对抗。", camera={"lon": 79.1, "lat": 33.75,
                                                     "height": 260000.0,
                                                     "heading": 0.0,
                                                     "pitch": -62.0}),
        C.Stage("outcome", "巩固/结局", 3, 1020, 1440, "engine",
                "控制山口与补给节点的一方取得优势。",
                "战役结束。", camera={"lon": 79.1, "lat": 33.75,
                                      "height": 400000.0, "heading": 0.0,
                                      "pitch": -70.0}),
    ]
    objectives = [
        {"coord": P["pass_west"], "value": 30, "flag": -1, "name": "OBJ-PASS"},
        {"coord": P["ridge_north"], "value": 20, "flag": -1, "name": "OBJ-RIDGE"},
        {"coord": P["valley"], "value": 20, "flag": -1, "name": "OBJ-VALLEY"},
        {"coord": P["blue_depot"], "value": 15, "flag": 1, "name": "OBJ-DEPOT"},
        {"coord": P["red_base"], "value": 15, "flag": 0, "name": "OBJ-REDBASE"},
    ]
    events = [
        C.ScriptEvent("EV-I0-01", 60, "recon", "蓝方无人机前出侦察谷地。",
                      actors=[3110], location=[78.4, 33.9]),
        C.ScriptEvent("EV-I1-01", 300, "advance", "红方坦克营向山口推进。",
                      actors=[3001], location=[78.9, 33.7]),
        C.ScriptEvent("EV-I2-01", 600, "counter", "蓝方装甲预备队发起反击。",
                      actors=[3101], location=[78.6, 34.0]),
        C.ScriptEvent("EV-I3-01", 1200, "ceasefire", "接触线稳定，战役收尾。",
                      location=[79.1, 33.75]),
    ]
    return C.CampaignSpec(
        id="india_china_west", name="中印西段（虚构演示）",
        theatre="ak_sai_chin", seed=20261022, stages=stages,
        objectives=objectives, script_events=events,
        script_outcome={"winner": "contested",
                        "description": "演示脚本：山口争夺未分胜负，按控制度判定。"},
        disclaimer=("本案例为虚构演示：单位、编成、事件与结局均为合成数据，"
                    "不代表现实部署或战场预测。"),
        supply_points=scenario_dict()["supply_points"],
        tick_seconds=30.0,
        landing_target=None)


def map_path() -> str:
    return MAP_PATH
