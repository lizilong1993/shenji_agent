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
        id="india_china_west", name="ZY1",
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


# ---------------------------------------------------------------------------
# run_campaign.py 需要的插件接口
# ---------------------------------------------------------------------------
# 这些函数与 taiwan_landing 的同名函数形状一致，只是本案例全场都在高原内陆，
# 没有登陆/两栖/海空部分，所以分类与影响半径的判定简单得多。

UNIT_CLASS_ZH = {
    "ship": "水面舰艇", "submarine": "潜艇", "aircraft": "作战飞机",
    "helicopter": "直升机", "uav": "无人机", "infantry": "步兵",
    "vehicle": "地面车辆", "artillery": "炮兵", "aa": "防空兵力",
    "recon": "侦察分队", "fort": "工事", "transport": "运输分队",
}


def runner_class():
    return C.CampaignRunner


def map_id() -> int:
    return MAP_ID


def region() -> str:
    return "中国 中印西段高原（虚构演示）"


def start_time_utc() -> str:
    return "2026-11-05T00:00:00Z"


def camera() -> dict:
    # 400 km 宽的高原战场：机位取山口与谷地之间，200 km 高、-62° 俯视，
    # 红蓝双方的后方基地都在视野内，单个营级目标仍可辨认。
    return {"lon": 79.10, "lat": 33.75, "height": 200000.0,
            "heading": 0.0, "pitch": -62.0}


def _icon_for(u) -> str:
    name = u.name
    domain = getattr(u, "domain", "")
    if domain == "air":
        if "无人" in name:
            return "uav"
        if "直升机" in name:
            return "helicopter"
        if "预警" in name:
            return "aew"
        return "fighter"
    if "炮兵" in name or "榴弹" in name or "火箭" in name:
        return "artillery"
    if "防空" in name or "导弹" in name:
        return "sam_battalion"
    if "侦察" in name:
        return "recon"
    if "运输" in name:
        return "transport"
    if "工兵" in name or "工程" in name:
        return "engineering"
    if "步兵" in name or "分队" in name:
        return "infantry"
    return "armor"


def _class_zh(u) -> str:
    return UNIT_CLASS_ZH.get(u.unit_type, u.unit_type)


def _engage_km(u) -> float:
    """武器可达距离；上限 60 km，避免前端为每个对偶画一个覆盖全图的圆。"""
    rng = 0.0
    for w in getattr(u, "sea_air_weapons", []) or []:
        rng = max(rng, float(getattr(w, "max_range_km", 0.0)))
    for w in getattr(u, "weapons", []) or []:
        rng = max(rng, float(getattr(w, "range_km", 0.0)))
    return round(min(rng, 60.0), 1)


def _influence_km(u) -> float:
    """传感器可达距离；地面 10 km 上限，空中返回 0（不占地）。"""
    if getattr(u, "domain", "") == "air":
        return 0.0
    rng = 0.0
    for s in getattr(u, "sensors", []) or []:
        rng = max(rng, float(getattr(s, "range_km", 0.0)))
    return round(min(rng, 10.0), 1)


def target_defs(state) -> list:
    out = []
    for u in state.units.values():
        out.append({
            "id": str(u.uid), "name": u.name,
            "affiliation": "red" if u.color == 0 else "blue",
            "domain": {"land": "land", "sea": "sea", "sub": "sea",
                       "air": "air"}.get(u.domain, u.domain),
            "unit_type": u.unit_type,
            "platform_class": _class_zh(u),
            "icon": _icon_for(u),
            "blood": u.blood, "max_blood": u.max_blood,
            "speed_kn": round(u.speed_kmh / 1.852, 1),
            "engage_km": _engage_km(u),
            "influence_km": _influence_km(u),
        })
    return out
