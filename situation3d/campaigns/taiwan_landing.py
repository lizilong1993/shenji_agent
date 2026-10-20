"""台海联合登陆战役演示（虚构）——campaign plugin for run_campaign.py.

All units, strengths, events and outcomes are synthetic demonstration data.
They are not real deployments, plans, or predictions.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
REPO = os.path.dirname(SITUATION3D)
sys.path.insert(0, SITUATION3D)

import campaign as C  # noqa: E402


MAP_ID = 9001
MAP_PATH = os.path.join(REPO, "situation3d", "web", "data",
                        "campaign_maps", "map_9001", "basic.json")

# Hex anchors (row, col) from the 100x100 theatre map.
P = {
    "beach": 7369,          # 安平岸（滩头）
    "tainan": 7372,         # 台南
    "kaohsiung": 8975,      # 高雄
    "penghu": 4950,         # 澎湖
    "strait_mid": 5231,     # 海峡中段
    "offshore_a": 6038,     # 安平外海
    "offshore_b": 6927,     # 外海 B
    "offshore_c": 7717,     # 外海 C
    "xiamen_off": 1310,     # 厦门外海
    "pingtan_off": 544,     # 平潭方向
    "east_coast": 4799,     # 东岸
    "kinmen": 1208,         # 金门
}


def _op(uid, name, color, op_type, sub_type, pos, blood, speed,
        domain=None, altitude_band=None, armor=2, value=10,
        size_class=None, depth=None, readiness=0.85, fuel=1.0,
        emitting=True):
    o = {
        "obj_id": uid, "name": name, "color": color, "type": op_type,
        "sub_type": sub_type, "cur_hex": pos, "blood": blood,
        "max_blood": blood, "armor": armor, "basic_speed": speed,
        "value": value, "observe_distance": {"1": 10, "2": 25, "3": 1},
        "move_state": 0, "stop": 1, "readiness": readiness, "fuel": fuel,
        "emitting": emitting,
    }
    if domain:
        o["domain"] = domain
    if altitude_band:
        o["altitude_band"] = altitude_band
    if size_class:
        o["size_class"] = size_class
    if depth:
        o["depth"] = depth
    return o


def scenario_dict() -> dict:
    """Initial order of battle (fictional)."""
    red = [
        # amphibious group
        _op(1001, "两栖船坞登陆舰", 0, 2, 0, P["offshore_b"], 4, 30,
            "sea", "水面", 3, 30, "large"),
        _op(1002, "登陆舰", 0, 2, 0, P["offshore_c"], 4, 32, "sea", "水面", 2, 20, "large"),
        _op(1003, "登陆舰", 0, 2, 0, P["offshore_a"], 4, 32, "sea", "水面", 2, 20, "large"),
        # surface escort
        _op(1010, "驱逐舰", 0, 2, 0, P["strait_mid"], 5, 50, "sea", "水面", 4, 35, "large"),
        _op(1011, "护卫舰", 0, 2, 0, P["offshore_a"], 4, 45, "sea", "水面", 3, 20, "medium"),
        _op(1012, "护卫舰", 0, 2, 0, P["offshore_c"], 4, 45, "sea", "水面", 3, 20, "medium"),
        _op(1013, "补给舰", 0, 2, 0, P["strait_mid"], 3, 28, "sea", "水面", 1, 8, "medium"),
        # air
        _op(1020, "战斗机", 0, 3, 0, P["pingtan_off"], 3, 1600, "air", "中空", 1, 30, "small"),
        _op(1021, "战斗机", 0, 3, 0, P["pingtan_off"], 3, 1600, "air", "中空", 1, 30, "small"),
        _op(1022, "预警机", 0, 3, 0, P["xiamen_off"], 3, 550, "air", "高空", 1, 25, "medium"),
        _op(1023, "电子战飞机", 0, 3, 0, P["xiamen_off"], 2, 800, "air", "高空", 1, 18, "medium"),
        _op(1024, "武装直升机", 0, 3, 0, P["offshore_a"], 2, 250, "air", "低空", 1, 12, "small"),
        # ground (ashore from the start in this demo; keep clear of the
        # defender's positions so the opening stage is assembly/dispersal)
        _op(1030, "重型坦克", 0, 2, 0, P["beach"], 3, 30, "land", "ground", 3, 15),
        _op(1031, "重型战车", 0, 2, 1, P["beach"] + 10, 3, 28, "land", "ground", 2, 12),
        _op(1032, "步兵小队", 0, 1, 2, P["beach"] + 20, 4, 6, "land", "ground", 0, 8),
        # coastal defence / air defence (onshore, far west)
        _op(1040, "岸舰导弹营", 0, 2, 0, P["kinmen"], 3, 20, "land", "ground", 1, 18),
        _op(1041, "防空导弹排", 0, 1, 16, P["kinmen"], 2, 10, "land", "ground", 0, 15),
        # logistics
        _op(1050, "补给舰", 0, 2, 0, P["offshore_b"], 3, 26, "sea", "水面", 1, 8, "medium"),
    ]
    blue = [
        # surface group
        _op(2001, "驱逐舰", 1, 2, 0, P["offshore_b"], 5, 50, "sea", "水面", 4, 35, "large"),
        _op(2002, "护卫舰", 1, 2, 0, P["offshore_c"], 4, 45, "sea", "水面", 3, 20, "medium"),
        _op(2003, "巡逻舰", 1, 2, 0, P["kaohsiung"], 3, 55, "sea", "水面", 1, 12, "small"),
        _op(2004, "补给舰", 1, 2, 0, P["east_coast"], 3, 28, "sea", "水面", 1, 8, "medium"),
        # air
        _op(2010, "战斗机", 1, 3, 0, P["tainan"], 3, 1500, "air", "中空", 1, 30, "small"),
        _op(2011, "战斗机", 1, 3, 0, P["tainan"], 3, 1500, "air", "中空", 1, 30, "small"),
        _op(2012, "预警机", 1, 3, 0, P["east_coast"], 3, 500, "air", "高空", 1, 25, "medium"),
        _op(2013, "无人机", 1, 3, 5, P["tainan"], 2, 300, "air", "低空", 1, 8, "small"),
        # coastal defence / air defence
        _op(2020, "岸舰导弹营", 1, 2, 0, P["kaohsiung"], 3, 20, "land", "ground", 1, 18),
        _op(2021, "防空导弹排", 1, 1, 16, P["tainan"], 2, 10, "land", "ground", 0, 15),
        # ground (defenders start near the objective; reinforced at S4)
        _op(2030, "重型坦克", 1, 2, 0, P["tainan"], 4, 30, "land", "ground", 4, 15),
        _op(2031, "重型战车", 1, 2, 1, P["tainan"] + 5, 3, 28, "land", "ground", 2, 12),
        _op(2032, "步兵小队", 1, 1, 2, P["kaohsiung"], 4, 6, "land", "ground", 0, 8),
        _op(2033, "炮兵", 1, 2, 3, P["tainan"] + 8, 3, 15, "land", "ground", 1, 14),
        # logistics
        _op(2040, "补给舰", 1, 2, 0, P["east_coast"], 3, 26, "sea", "水面", 1, 8, "medium"),
    ]
    return {
        "scenario_id": MAP_ID,
        "time": {"cur_step": 0, "tick": 1.0, "max_time": 960},
        "cities": [],
        "operators": red + blue,
        "supply_points": [
            {"color": 0, "pos": P["kinmen"], "kind": "port"},
            {"color": 0, "pos": P["xiamen_off"], "kind": "airbase"},
            {"color": 1, "pos": P["kaohsiung"], "kind": "port"},
            {"color": 1, "pos": P["tainan"], "kind": "airbase"},
            {"color": 1, "pos": P["east_coast"], "kind": "airbase"},
        ],
    }


def build_spec() -> C.CampaignSpec:
    stages = [
        C.Stage("assembly", "集结装载", 0, 0, 120, "mixed",
                "红方两栖编队在外海集结，蓝方ISR发现集结迹象。",
                "红方登陆编队完成装载，蓝方进入警戒。"),
        C.Stage("transit", "航渡与海上拦截", 1, 120, 300, "adjudicated",
                "红方编队向登陆海域航渡，蓝方水面编队实施拦截。",
                "双方海上交火，红方损失部分护航兵力。"),
        C.Stage("transfer", "换乘编波", 2, 300, 420, "mixed",
                "红方在换乘海域释放登陆艇，蓝方岸火与无人机侦察。",
                "换乘完成，部分登陆艇被拦阻。"),
        C.Stage("assault", "抢滩突击", 3, 420, 600, "adjudicated",
                "两栖车辆上陆，直升机近距支援，滩头交火。",
                "红方在安平滩头建立立足点。"),
        C.Stage("consolidate", "巩固滩头/反突击", 4, 600, 780, "mixed",
                "蓝方装甲营反击，红方第二梯队上陆。",
                "红方击退反突击，登陆场稳定。"),
        C.Stage("inland", "向内陆夺控", 5, 780, 900, "adjudicated",
                "红方沿路向内陆推进，争夺台南与机场。",
                "红方控制台南外围。"),
        C.Stage("outcome", "结局", 6, 900, 960, "engine",
                "达成条件或停火，输出结局。",
                "战役结束。"),
    ]
    objectives = [
        {"coord": P["beach"], "value": 30, "flag": 1, "name": "OBJ-BEACH"},
        {"coord": P["tainan"], "value": 25, "flag": 1, "name": "OBJ-TAINAN"},
        {"coord": P["kaohsiung"], "value": 20, "flag": 1, "name": "OBJ-KAOHSIUNG"},
        {"coord": P["penghu"], "value": 15, "flag": 1, "name": "OBJ-PENGHU"},
        {"coord": P["strait_mid"], "value": 10, "flag": -1, "name": "OBJ-STRAIT"},
    ]
    events = [
        C.ScriptEvent("EV-S0-01", 30, "isr", "蓝方预警机升空，发现红方集结编队。",
                      actors=[2012], location=[120.0, 23.0]),
        C.ScriptEvent("EV-S1-01", 150, "jam", "红方电子战飞机对蓝方雷达实施干扰。",
                      actors=[1023], action="set_jam", action_args={"value": 0.35}),
        C.ScriptEvent("EV-S1-02", 240, "jam_end", "干扰结束。", action="clear_jam"),        C.ScriptEvent("EV-S2-01", 320, "beaching", "红方两栖舰释放登陆艇。",
                      actors=[1001], action="spawn_beach",
                      action_args={"ship": 1001, "name": "登陆艇甲", "blood": 2}),
        C.ScriptEvent("EV-S2-02", 360, "beaching", "红方登陆舰释放第二波登陆艇。",
                      actors=[1002], action="spawn_beach",
                      action_args={"ship": 1002, "name": "登陆艇乙", "blood": 2}),
        C.ScriptEvent("EV-S4-01", 620, "reinforce",
                      "蓝方装甲预备队从台南方向反击。",
                      actors=[2030], requires={"unit_alive": 2030},
                      fallback="蓝方预备队已被摧毁，无法投入反击。"),
    ]
    return C.CampaignSpec(
        id="taiwan_landing", name="台海联合登陆战役演示（虚构）",
        theatre="taiwan_strait", seed=20261019, stages=stages,
        objectives=objectives, script_events=events,
        script_outcome={"winner": "red",
                        "description": "演示脚本设定：红方达成登陆战役目标。"},
        disclaimer=("本案例为虚构演示：单位、编成、事件与结局均为合成数据，"
                    "不代表现实部署或战场预测。"),
        supply_points=scenario_dict()["supply_points"],
        tick_seconds=30.0)


def map_path() -> str:
    return MAP_PATH


def map_id() -> int:
    return MAP_ID


def region() -> str:
    return "中国 台湾海峡南段（虚构演示）"


def start_time_utc() -> str:
    return "2026-11-03T00:00:00Z"


def camera() -> dict:
    return {"lon": 119.9, "lat": 23.3, "height": 420000.0,
            "heading": 0.0, "pitch": -70.0}


def runner_class():
    return C.CampaignRunner


def target_defs(state) -> list:
    out = []
    for u in state.units.values():
        out.append({
            "id": u.uid, "name": u.name, "affiliation": "red" if u.color == 0 else "blue",
            "domain": u.domain, "unit_type": u.unit_type,
            "platform_class": u.unit_type, "icon": "armor" if u.domain == "land"
            else ("destroyer" if u.domain == "sea" else "fighter"),
            "blood": u.blood, "max_blood": u.max_blood,
            "speed_kn": round(u.speed_kmh / 1.852, 1),
        })
    return out
