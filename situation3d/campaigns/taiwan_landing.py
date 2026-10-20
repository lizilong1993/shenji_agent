"""台海联合登陆战役演示（虚构）——campaign plugin for run_campaign.py.

All units, strengths, events and outcomes are synthetic demonstration data.
They are not real deployments, plans, or predictions.

Timeline (30 s ticks, 12 h = 1440 ticks):
  S0 集结装载        0–180
  S1 航渡/海上拦截   180–420
  S2 换乘编波        420–600
  S3 抢滩突击        600–780
  S4 巩固/反突击     780–1020
  S5 向内陆夺控      1020–1320
  S6 结局            1320–1440
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

# Hex anchors (row, col) verified against the generated theatre map.
P = {
    "beach": 7371,          # 台南西侧滩头（陆，紧邻水道）
    "beach_north": 6871,    # 北侧备用滩头
    "tainan": 7372,         # 台南（居民地）
    "tainan_east": 7576,    # 台南以东
    "kaohsiung": 8975,      # 高雄（居民地）
    "kaohsiung_port": 8974, # 高雄港区
    "penghu": 4950,         # 澎湖
    "strait_mid": 5231,     # 海峡中段
    "offshore_a": 6038,     # 安平外海
    "offshore_b": 6927,     # 外海 B
    "offshore_c": 7717,     # 外海 C
    "xiamen_off": 1310,     # 厦门外海
    "pingtan_off": 544,     # 平潭方向
    "east_coast": 4799,     # 东岸
    "kinmen": 1208,         # 金门
    "landing_lane": 6870,   # 换乘海域（水道）
}


def _op(uid, name, color, op_type, sub_type, pos, blood, speed,
        domain=None, altitude_band=None, armor=2, value=10,
        size_class=None, depth=None, readiness=0.85, fuel=1.0,
        emitting=True, beaching=False):
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
    if beaching:
        o["beaching"] = True
    return o


def scenario_dict() -> dict:
    """Initial order of battle (fictional)."""
    red = [
        # amphibious group (carries the landing force)
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
        # landing force: starts embarked as amphibious vehicles afloat
        _op(1030, "两栖战车", 0, 2, 4, P["offshore_b"], 3, 12, "sea", "水面", 2, 15, "small", beaching=True),
        _op(1031, "两栖战车", 0, 2, 4, P["offshore_b"], 3, 12, "sea", "水面", 2, 15, "small", beaching=True),
        _op(1032, "两栖步兵", 0, 1, 2, P["offshore_c"], 4, 10, "sea", "水面", 0, 8, "small", beaching=True),
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
        _op(2003, "巡逻舰", 1, 2, 0, P["kaohsiung_port"], 3, 55, "sea", "水面", 1, 12, "small"),
        _op(2004, "补给舰", 1, 2, 0, P["east_coast"], 3, 28, "sea", "水面", 1, 8, "medium"),
        # air
        _op(2010, "战斗机", 1, 3, 0, P["tainan"], 3, 1500, "air", "中空", 1, 30, "small"),
        _op(2011, "战斗机", 1, 3, 0, P["tainan"], 3, 1500, "air", "中空", 1, 30, "small"),
        _op(2012, "预警机", 1, 3, 0, P["east_coast"], 3, 500, "air", "高空", 1, 25, "medium"),
        _op(2013, "无人机", 1, 3, 5, P["tainan"], 2, 300, "air", "低空", 1, 8, "small"),
        # coastal defence / air defence
        _op(2020, "岸舰导弹营", 1, 2, 0, P["kaohsiung"], 3, 20, "land", "ground", 1, 18),
        _op(2021, "防空导弹排", 1, 1, 16, P["tainan"], 2, 10, "land", "ground", 0, 15),
        # ground: defenders inland of the beach
        _op(2030, "重型坦克", 1, 2, 0, P["tainan_east"], 4, 30, "land", "ground", 4, 15),
        _op(2031, "重型战车", 1, 2, 1, P["tainan_east"] + 3, 3, 28, "land", "ground", 2, 12),
        _op(2032, "步兵小队", 1, 1, 2, P["kaohsiung"], 4, 6, "land", "ground", 0, 8),
        _op(2033, "炮兵", 1, 2, 3, P["tainan_east"] + 6, 3, 15, "land", "ground", 1, 14),
        # logistics
        _op(2040, "补给舰", 1, 2, 0, P["east_coast"], 3, 26, "sea", "水面", 1, 8, "medium"),
    ]
    return {
        "scenario_id": MAP_ID,
        "time": {"cur_step": 0, "tick": 1.0, "max_time": 1440},
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
        C.Stage("assembly", "集结装载", 0, 0, 180, "mixed",
                "红方两栖编队在外海集结，蓝方ISR发现集结迹象。",
                "红方登陆编队完成装载，蓝方进入警戒。",
                camera={"lon": 119.55, "lat": 23.35, "height": 520000.0,
                        "heading": 0.0, "pitch": -70.0}),
        C.Stage("transit", "航渡与海上拦截", 1, 180, 420, "adjudicated",
                "红方编队向登陆海域航渡，蓝方水面编队实施拦截。",
                "双方海上交火，红方损失部分护航兵力。",
                camera={"lon": 119.35, "lat": 23.1, "height": 420000.0,
                        "heading": 0.0, "pitch": -68.0}),
        C.Stage("transfer", "换乘编波", 2, 420, 600, "mixed",
                "红方在换乘海域释放登陆兵力，蓝方岸火与无人机侦察。",
                "换乘完成，部分登陆兵力被拦阻。",
                camera={"lon": 120.0, "lat": 23.0, "height": 220000.0,
                        "heading": 0.0, "pitch": -60.0}),
        C.Stage("assault", "抢滩突击", 3, 600, 780, "adjudicated",
                "两栖车辆上陆，直升机近距支援，滩头交火。",
                "红方在安平滩头建立立足点。",
                camera={"lon": 120.12, "lat": 23.0, "height": 120000.0,
                        "heading": 0.0, "pitch": -55.0}),
        C.Stage("consolidate", "巩固滩头/反突击", 4, 780, 1020, "mixed",
                "蓝方装甲营反击，红方第二梯队上陆。",
                "红方击退反突击，登陆场稳定。",
                camera={"lon": 120.2, "lat": 22.99, "height": 160000.0,
                        "heading": 0.0, "pitch": -58.0}),
        C.Stage("inland", "向内陆夺控", 5, 1020, 1320, "adjudicated",
                "红方沿路向内陆推进，争夺台南与机场。",
                "红方控制台南外围。",
                camera={"lon": 120.25, "lat": 22.9, "height": 260000.0,
                        "heading": 0.0, "pitch": -62.0}),
        C.Stage("outcome", "结局", 6, 1320, 1440, "engine",
                "达成条件或停火，输出结局。",
                "战役结束。",
                camera={"lon": 120.0, "lat": 23.1, "height": 480000.0,
                        "heading": 0.0, "pitch": -70.0}),
    ]
    objectives = [
        {"coord": P["beach"], "value": 30, "flag": 1, "name": "OBJ-BEACH"},
        {"coord": P["tainan"], "value": 25, "flag": 1, "name": "OBJ-TAINAN"},
        {"coord": P["kaohsiung"], "value": 20, "flag": 1, "name": "OBJ-KAOHSIUNG"},
        {"coord": P["penghu"], "value": 15, "flag": 1, "name": "OBJ-PENGHU"},
        {"coord": P["strait_mid"], "value": 10, "flag": -1, "name": "OBJ-STRAIT"},
    ]
    events = [
        # S0
        C.ScriptEvent("EV-S0-01", 45, "isr", "蓝方预警机升空，发现红方集结编队。",
                      actors=[2012], location=[119.9, 23.4]),
        C.ScriptEvent("EV-S0-02", 120, "alert", "蓝方进入一级警戒，岸舰导弹营展开。",
                      actors=[2020], location=[120.3, 22.6]),
        # S1
        C.ScriptEvent("EV-S1-01", 210, "jam", "红方电子战飞机对蓝方雷达实施干扰。",
                      actors=[1023], action="set_jam", action_args={"value": 0.35}),
        C.ScriptEvent("EV-S1-02", 330, "jam_end", "干扰结束。", action="clear_jam"),
        C.ScriptEvent("EV-S1-03", 360, "salvo", "蓝方岸舰导弹营对红方编队实施齐射。",
                      actors=[2020], location=[119.8, 23.0],
                      effects={"1011": {"strength_delta": -10}}),
        # S2
        C.ScriptEvent("EV-S2-01", 450, "beaching", "红方两栖舰释放第一波登陆艇。",
                      actors=[1001], action="spawn_beach",
                      action_args={"ship": 1001, "name": "登陆艇甲", "blood": 2}),
        C.ScriptEvent("EV-S2-02", 500, "beaching", "红方登陆舰释放第二波登陆艇。",
                      actors=[1002], action="spawn_beach",
                      action_args={"ship": 1002, "name": "登陆艇乙", "blood": 2}),
        C.ScriptEvent("EV-S2-03", 540, "recon", "蓝方无人机发现换乘海域，岸火准备。",
                      actors=[2013], location=[119.95, 22.95]),
        # S3
        C.ScriptEvent("EV-S3-01", 610, "assault", "第一梯队抢滩：两栖车辆上陆。",
                      actors=[1030, 1031, 1032], action="convert_land",
                      action_args={"units": [1030, 1031, 1032]}),
        C.ScriptEvent("EV-S3-02", 660, "cas", "红方武装直升机进入滩头支援。",
                      actors=[1024], location=[120.08, 23.0]),
        C.ScriptEvent("EV-S3-03", 720, "assault", "第一批登陆艇抵滩，换乘部队上陆。",
                      action="convert_land",
                      action_args={"units": "beaching"}),
        # S4
        C.ScriptEvent("EV-S4-01", 800, "reinforce",
                      "蓝方装甲预备队从台南方向反击。",
                      actors=[2030], requires={"unit_alive": 2030},
                      fallback="蓝方预备队已被摧毁，无法投入反击。"),
        C.ScriptEvent("EV-S4-02", 860, "second_wave", "红方第二梯队在滩头展开。",
                      actors=[1003], action="spawn_beach",
                      action_args={"ship": 1003, "name": "登陆艇丙", "blood": 2}),
        C.ScriptEvent("EV-S4-03", 930, "engineer", "红方工兵开辟滩头通路。",
                      location=[120.12, 22.99]),
        # S5
        C.ScriptEvent("EV-S5-01", 1060, "objective", "红方开始争夺台南外围。",
                      location=[120.21, 22.99]),
        C.ScriptEvent("EV-S5-02", 1150, "delay", "蓝方迟滞防御，破坏道路与桥梁。",
                      actors=[2033], location=[120.25, 22.97]),
        C.ScriptEvent("EV-S5-03", 1250, "airfield", "红方接近机场外围，蓝方加强防空。",
                      actors=[2021], location=[120.21, 22.95]),
        # S6
        C.ScriptEvent("EV-S6-01", 1330, "ceasefire", "双方接触线稳定，战役进入收尾。",
                      location=[120.2, 23.0]),
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
        tick_seconds=30.0,
        landing_target=P["beach"])


def map_path() -> str:
    return MAP_PATH


def map_id() -> int:
    return MAP_ID


def region() -> str:
    return "中国 台湾海峡南段（虚构演示）"


def start_time_utc() -> str:
    return "2026-11-03T00:00:00Z"


def camera() -> dict:
    # 初始机位：海峡南段、厦门—澎湖—台南都能看到，又能分辨营级目标。
    return {"lon": 119.98, "lat": 23.10, "height": 230000.0,
            "heading": 0.0, "pitch": -62.0}


def runner_class():
    return C.CampaignRunner


ICONS = {
    "两栖船坞登陆舰": "replenishment", "登陆舰": "replenishment",
    "驱逐舰": "destroyer", "护卫舰": "frigate", "巡逻舰": "corvette",
    "补给舰": "replenishment", "战斗机": "fighter", "预警机": "aew",
    "电子战飞机": "aew", "无人机": "aew", "武装直升机": "fighter",
    "两栖战车": "armor", "两栖步兵": "armor", "重型坦克": "armor",
    "重型战车": "armor", "步兵小队": "armor", "炮兵": "artillery",
    "岸舰导弹营": "coastal_missile", "防空导弹排": "sam_battalion",
    "登陆艇甲": "armor", "登陆艇乙": "armor", "登陆艇丙": "armor",
}


def _engage_km(u) -> float:
    """Weapon reach in km, used by the front end to draw contact hotspots."""
    rng = 0.0
    for w in getattr(u, "sea_air_weapons", []) or []:
        rng = max(rng, float(getattr(w, "max_range_km", 0.0)))
    for w in getattr(u, "weapons", []) or []:
        rng = max(rng, float(getattr(w, "range_km", 0.0)))
    return round(rng, 1)


def _influence_km(u) -> float:
    """Sensor reach in km, used for the influence/occupation overlay.

    Air platforms do not hold ground, so their influence circle is kept small
    (the front end falls back to 160 km for air when this field is 0).  Ships
    are capped so their circles stay readable.
    """
    if getattr(u, "domain", "") == "air":
        return 30.0
    rng = 0.0
    for s in getattr(u, "sensors", []) or []:
        rng = max(rng, float(getattr(s, "range_km", 0.0)))
    if getattr(u, "domain", "") in ("sea", "sub"):
        rng = min(rng, 120.0)
    return round(rng, 1)


UNIT_CLASS_ZH = {
    "ship": "水面舰艇", "submarine": "潜艇", "aircraft": "作战飞机",
    "helicopter": "直升机", "uav": "无人机", "infantry": "步兵",
    "vehicle": "地面车辆", "artillery": "炮兵", "aa": "防空兵力",
    "recon": "侦察分队", "fort": "工事",
}


def target_defs(state) -> list:
    out = []
    for u in state.units.values():
        out.append({
            "id": str(u.uid), "name": u.name,
            "affiliation": "red" if u.color == 0 else "blue",
            "domain": {"land": "land", "sea": "sea", "sub": "sea",
                       "air": "air"}.get(u.domain, u.domain),
            "unit_type": u.unit_type,
            "platform_class": UNIT_CLASS_ZH.get(u.unit_type, u.unit_type),
            "icon": ICONS.get(u.name, "armor" if u.domain == "land" else "destroyer"),
            "blood": u.blood, "max_blood": u.max_blood,
            "speed_kn": round(u.speed_kmh / 1.852, 1),
            # 前端按这两个字段算“交火区/对峙区”热点与控制区覆盖。
            "engage_km": _engage_km(u),
            "influence_km": _influence_km(u),
        })
    return out


def plots() -> list:
    """Time-windowed situation plots (fictional)."""
    return [
        {"id": "PLOT-S1-1", "type": "attack_arrow", "layer": "plot.offense",
         "points": [[119.35, 23.25], [119.85, 23.05], [120.05, 22.97]],
         "label": "登陆编队航渡方向", "hostility": "friendly",
         "from_tick": 180, "to_tick": 600, "stage_id": "transit"},
        {"id": "PLOT-S2-1", "type": "zone", "layer": "plot.offense",
         "points": [[119.95, 23.05], [120.08, 23.03], [120.10, 22.94],
                    [119.96, 22.93]],
         "label": "换乘海域", "hostility": "friendly",
         "from_tick": 420, "to_tick": 780, "stage_id": "transfer"},
        {"id": "PLOT-S3-1", "type": "attack_arrow", "layer": "plot.offense",
         "points": [[120.05, 22.99], [120.12, 22.99], [120.18, 22.99]],
         "label": "主攻方向（安平滩头）", "hostility": "friendly",
         "from_tick": 600, "to_tick": 1020, "stage_id": "assault"},
        {"id": "PLOT-S3-2", "type": "zone", "layer": "plot.defense",
         "points": [[120.12, 23.03], [120.22, 23.02], [120.23, 22.90],
                    [120.12, 22.91]],
         "label": "蓝方滩头防御区", "hostility": "enemy",
         "from_tick": 420, "to_tick": 900, "stage_id": "transfer"},
        {"id": "PLOT-S4-1", "type": "attack_arrow", "layer": "plot.defense",
         "points": [[120.24, 22.96], [120.18, 22.99], [120.13, 23.0]],
         "label": "蓝方反突击方向", "hostility": "enemy",
         "from_tick": 780, "to_tick": 1020, "stage_id": "consolidate"},
        {"id": "PLOT-S5-1", "type": "attack_arrow", "layer": "plot.offense",
         "points": [[120.16, 22.99], [120.21, 22.97], [120.28, 22.72]],
         "label": "向内陆夺控", "hostility": "friendly",
         "from_tick": 1020, "to_tick": 1320, "stage_id": "inland"},
        {"id": "PLOT-S5-2", "type": "fan", "layer": "plot.fire",
         "points": [[120.30, 22.63], [120.22, 22.70], [120.26, 22.58]],
         "label": "蓝方岸防火力扇面", "hostility": "enemy",
         "from_tick": 1020, "to_tick": 1320, "stage_id": "inland"},
    ]
