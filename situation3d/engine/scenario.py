"""Taiwan Strait demonstration scenario for the situation3d engine.

This module defines a *simulated* situation over *real* geography. Waypoints are
real locations in and around the Taiwan Strait; speeds and altitudes are chosen
to match the real performance envelope of the represented platform classes.

It is deliberately not presented as observed data: per the agreed scope, imagery
comes from real datasets while the situation itself is engine-generated.

Targets are organised along two axes:

* ``domain``      -- ``sea`` / ``land`` / ``air``
* ``affiliation`` -- ``red`` / ``blue``

so that every combination the platform is expected to display is represented at
least once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import geo, symbology

# --------------------------------------------------------------------------
# Scenario constants
# --------------------------------------------------------------------------

SCENARIO_ID = "taiwan_strait_demo_20260915"
SCENARIO_NAME = "台海方向联合TS演示"
REGION = "中国 台湾海峡"

# 12 simulated hours at 30 s per tick.
TICK_SECONDS = 30.0
TICK_COUNT = 1440
# TS的仿真起始时刻（UTC）。提到常量：数据集里要写它，TS简报也要用它把
# "第 N 帧"换算成钟点，两处必须是同一个值，否则简报里会出现对不上的时间。
START_TIME_UTC = "2026-09-15T00:00:00Z"

# 意图集合（CES）预测的步长。模型链路逐帧跑代价过高（实测 5 目标案例 1645 s），
# 而 8 帧≈4 分钟一次已远超界面的刷新能力；步长是均匀的，且 no-reentry 投影仍在
# tick 顺序上执行，因此不改变模型的收缩行为。其余帧沿用上一个预测帧（见
# intent_predict.fill_carried），并在数据里标注为沿用而非新判定。
INTENT_PREDICT_STRIDE = 8

# Hard cap on how many frames the model is scored on. The stride alone does not
# bound the cost, because per-frame feature extraction walks road geometry and
# per-target neighbour scans: measured, the 16-target landing case did not finish
# `build_dataset` within 800 s at stride 8 (181 scored ticks x 16 targets), which
# left `web/data/taiwan_landing.json` ungenerated. The cap is applied on top of the
# stride so a case with more targets automatically coarsens instead of silently
# blowing up the build time. 40 scored frames is ~20 min of playback apart.
INTENT_MAX_PREDICT_TICKS = 40

# How long a candidate must stay below the threshold before it is permanently
# barred, expressed in **ticks** (30 s each) rather than in scored frames, so the
# meaning survives a change of stride.
#
# Calibration note: at stride 8 an earlier value of 8 ticks produced grace=2
# scored frames, and because banning that fast removed almost every candidate the
# India case ended with **4303 of 7205 annotated target-frames carrying an empty
# set**. On long, near-constant tracks (a patrol orbit, a supply run) the score
# genuinely wobbles around the threshold, so a short window turns noise into
# permanent exclusions. 48 ticks = 24 minutes of playback, which is the timescale
# over which a real intent change would be visible anyway.
REENTRY_GRACE_TICKS = 48

# Real geographic reference points (WGS84), used as waypoint anchors.
ANCHORS = {
    "fuzhou": (26.07, 119.30),      # 福州
    "pingtan": (25.50, 119.79),     # 平潭岛
    "xiamen": (24.48, 118.08),      # 厦门
    "kinmen": (24.44, 118.32),      # 金门
    "penghu": (23.57, 119.58),      # 澎湖列岛
    "kaohsiung": (22.62, 120.28),   # 高雄
    "taichung": (24.15, 120.68),    # 台中
    "keelung": (25.13, 121.74),     # 基隆
    "eluanbi": (21.90, 120.85),     # 鹅銮鼻
    "taipei": (25.03, 121.56),      # 台北
}

# Hotspot regions shown in the right-hand panel.
#
# Each entry carries the real geographic extent as a WGS84 bounding box
# (west, south, east, north) rather than a single point, because the panel
# displays a "range" column and a point cannot represent a strait or an
# archipelago. `center` and `span_deg` are derived from the box, so the three
# can never disagree -- which is what previously let the displayed coordinates
# drift away from the real region.
_HOTSPOT_BOXES = [
    ("中国 台湾海峡", (117.50, 23.00, 121.00, 26.00), "主场景区域", "strait"),
    ("中国 平潭", (119.60, 25.35, 120.00, 25.70), "海峡北口", "island"),
    ("中国 澎湖列岛", (119.30, 23.15, 119.75, 23.75), "海峡中段", "island"),
    ("中国 东海", (122.00, 26.00, 128.00, 31.00), "北部进出通道", "sea"),
    ("中国 巴士海峡", (120.00, 20.40, 122.50, 22.00), "南部进出通道", "strait"),
    ("中国 三亚", (108.90, 18.10, 109.75, 18.50), "南部支点", "base"),
    ("中国台湾省 台北", (121.35, 24.85, 121.75, 25.25), "北部要地", "base"),
    ("中国台湾省 高雄", (120.10, 22.45, 120.50, 22.85), "南部要地", "base"),
    ("日本 冲绳岛", (127.60, 26.05, 128.40, 26.90), "东北方向", "island"),
    ("美国 关岛", (144.60, 13.20, 145.00, 13.70), "西太平洋支点", "base"),
    # 东京羽田机场：机场用地约 35.52-35.58N / 139.73-139.84E，框取 0.20x0.18 度，
    # 含跑道与进近空域。它不在战区 bbox 内，靠热点分级覆盖单独下载影像与高程。
    ("日本 羽田机场", (139.70, 35.46, 139.90, 35.64), "东北方向空中枢纽", "airport"),
]


def _hotspot(name: str, bbox: Sequence[float], note: str, kind: str) -> dict:
    w, s, e, n = bbox
    return {
        "name": name,
        "bbox": [round(w, 3), round(s, 3), round(e, 3), round(n, 3)],
        "center": [round((s + n) / 2.0, 4), round((w + e) / 2.0, 4)],
        "span_deg": [round(e - w, 3), round(n - s, 3)],
        "note": note,
        "kind": kind,
    }


HOTSPOTS = [_hotspot(name, bbox, note, kind) for name, bbox, note, kind in _HOTSPOT_BOXES]


# Urban cores used for the WSF3D 3D building-block rendering.
#
# The hotspot boxes above span whole straits and archipelagos, which is the
# right extent for a situation display but far too large to extrude: at 90 m
# WSF3D resolution a 3 degree box is tens of millions of cells. These entries
# are deliberately small so individual city blocks resolve, and they are the
# only areas for which high-zoom WSF3D tiles are pre-downloaded (see
# tools/fetch_assets.py and tools/fetch_urban_wsf3d.sh).
# Each box is about 0.12 degrees (~13 km) on a side, which at the 128-cell
# aggregation grid used by the front end works out to roughly 100 m blocks --
# close to the 90 m native resolution of the product.
URBAN_AOI = [
    {"name": "中国台湾省 台北", "bbox": [121.505, 24.973, 121.625, 25.093], "zoom": 13},
    {"name": "中国台湾省 高雄", "bbox": [120.240, 22.570, 120.360, 22.690], "zoom": 13},
    {"name": "中国 厦门", "bbox": [118.030, 24.420, 118.150, 24.540], "zoom": 13},
    {"name": "中国 福州", "bbox": [119.240, 26.020, 119.360, 26.140], "zoom": 13},
]


# --------------------------------------------------------------------------
# Target definitions
# --------------------------------------------------------------------------


@dataclass
class Target:
    """One tracked platform in the demonstration situation."""

    target_id: str
    name: str
    platform_class: str          # 平台类别
    domain: str                  # sea / land / air
    affiliation: str             # red / blue
    waypoints: List[geo.Coord]
    speed_kn: float
    altitude_m: float = 0.0
    role: str = ""
    model: str = ""              # 真实装备型号
    model_note: str = ""         # 型号补充说明（北约代号等）
    icon: str = "unknown"        # 军标图形键，见 engine/symbology.py
    sensor_name: str = ""
    sensor_range_km: float = 0.0
    # Combat radius / endurance expressed as the radius the platform can
    # influence; used by the inference layer for coverage reasoning.
    influence_km: float = 0.0
    # Weapon engagement radius, kilometres. Drives the proximity term of the
    # threat score; an unarmed platform (the AEW aircraft) therefore projects
    # no threat of its own however large its radar picture is.
    engage_km: float = 15.0
    # True for platforms that fly or sail a closed patrol circuit: they cycle
    # the route instead of stopping at the last vertex.
    loop: bool = False
    # True when this land platform's movement must follow the road network.
    #
    # Not every `domain == "land"` target is road-bound. Area-defence batteries
    # (岸舰导弹营 / 远程防空营 / 要地防空连) occupy a position and fight from it;
    # they are defined by where they are emplaced, not by a road march, and forcing
    # them onto the network both misrepresents them and makes the dataset's
    # `road_constrained` flag depend on whether a road happens to pass near a
    # missile site. Only mobile land units - armour, mechanised infantry, artillery
    # moving between positions, logistics - are road-bound.
    road_bound: bool = True
    attributes: Dict[str, str] = field(default_factory=dict)

    @property
    def speed_mps(self) -> float:
        return geo.knots_to_mps(self.speed_kn)

    @property
    def speed_kmh(self) -> float:
        return self.speed_kn * geo.NM_IN_M / 1000.0

    @property
    def speed_unit(self) -> str:
        # Ground units are conventionally reported in km/h, everything else in kn.
        return "km/h" if self.domain == "land" else "kn"

    def to_dataset_entry(self) -> dict:
        """Serialise the definition for the web front end."""
        return {
            "id": self.target_id,
            "name": self.name,
            "platform_class": self.platform_class,
            "domain": self.domain,
            "affiliation": self.affiliation,
            "role": self.role,
            "model": self.model,
            "model_note": self.model_note,
            "icon": self.icon,
            "icon_label": symbology.describe(self.icon)["label"],
            "sensor_name": self.sensor_name,
            "sensor_range_km": self.sensor_range_km,
            "influence_km": self.influence_km,
            "engage_km": self.engage_km,
            "speed_kn": self.speed_kn,
            "speed_kmh": round(self.speed_kmh, 1),
            "speed_unit": self.speed_unit,
            "altitude_m": self.altitude_m,
            # Whether this platform's movement must follow the road network. The
            # front end and `road_selftest` both read it, so a positional battery is
            # not held to a constraint that does not apply to it.
            "road_bound": bool(self.road_bound),
            "attributes": self.attributes,
            "waypoints": [[round(a, 6), round(b, 6)] for a, b in self.waypoints],
        }


def _build_targets() -> List[Target]:
    """Define the twelve platforms of the demonstration situation.

    Seven red-force and five blue-force platforms, covering sea, land and air.
    """

    # -- red surface group: enters the strait from the north-east and heads south
    destroyer_route = [
        (26.20, 120.60),   # 东海入口
        (25.60, 120.20),   # 平潭以东
        (24.90, 119.90),   # 海峡北口
        (24.20, 119.70),   # 海峡中段
        (23.60, 119.30),   # 澎湖以西
        (23.00, 118.80),   # 海峡南段
        (22.40, 118.20),   # 南海入口
    ]
    frigate_route = [(lat + 0.10, lon - 0.14) for lat, lon in destroyer_route]
    supply_route = [(lat - 0.12, lon - 0.30) for lat, lon in destroyer_route]

    # -- blue surface group: enters from the south and works north up the strait
    blue_frigate_route = [
        (21.85, 120.55),   # 鹅銮鼻西南
        (22.45, 120.15),   # 高雄西南
        (23.10, 119.95),   # 澎湖以南
        (23.70, 119.95),   # 澎湖以西
        (24.30, 120.05),   # 海峡中段
        (24.90, 120.35),   # 海峡北段
        (25.35, 120.75),   # 海峡北口
    ]
    blue_corvette_route = [
        (22.15, 120.35),
        (22.80, 120.18),
        (23.40, 120.10),
        (24.00, 120.15),
        (24.60, 120.38),
        (25.05, 120.65),
    ]

    # -- red land: coastal defence battalion working along the Fujian seaboard
    coastal_route = [
        (26.10, 119.62),
        (25.62, 119.72),
        (25.05, 119.48),
        (24.62, 118.88),
        (24.32, 118.28),
        (24.06, 117.82),
    ]
    # -- red land: area air-defence battalion, small repositioning loop
    sam_route = [
        (26.02, 119.28),
        (25.92, 119.42),
        (25.86, 119.55),
        (25.94, 119.60),
        (26.02, 119.28),
    ]
    # -- blue land: point air-defence battery near the northern urban area
    blue_sam_route = [
        (25.06, 121.42),
        (24.98, 121.52),
        (24.92, 121.62),
        (24.99, 121.68),
        (25.06, 121.42),
    ]

    # -- red air: strike package out of the mainland coast
    red_fighter_route = [
        (25.60, 119.10),
        (25.30, 119.60),
        (24.90, 120.00),
        (24.50, 120.30),
        (24.10, 120.10),
        (23.80, 119.60),
        (23.90, 119.00),
    ]
    # -- red air: airborne early warning orbit west of the strait
    red_aew_route = [
        (25.90, 118.70),
        (25.50, 119.10),
        (25.10, 119.40),
        (24.80, 119.10),
        (24.90, 118.60),
        (25.30, 118.40),
        (25.90, 118.70),   # 闭环
    ]
    # -- blue air: combat air patrol over the island
    blue_fighter_route = [
        (24.90, 121.30),
        (24.70, 120.90),
        (24.40, 120.60),
        (24.10, 120.40),
        (23.80, 120.60),
        (23.90, 121.00),
        (24.30, 121.30),
    ]
    # -- blue air: airborne early warning orbit east of the island
    blue_aew_route = [
        (25.40, 122.10),
        (25.10, 121.80),
        (24.80, 121.60),
        (24.50, 121.80),
        (24.60, 122.20),
        (25.00, 122.40),
        (25.40, 122.10),   # 闭环
    ]

    def _t(tid, name, klass, dom, aff, wps, kn, **kw):
        return Target(
            target_id=tid, name=name, platform_class=klass, domain=dom,
            affiliation=aff, waypoints=wps, speed_kn=kn,
            altitude_m=kw.pop("altitude_m", 0.0),
            role=kw.pop("role", ""), model=kw.pop("model", ""),
            model_note=kw.pop("model_note", ""), icon=kw.pop("icon", "unknown"),
            sensor_name=kw.pop("sensor_name", ""),
            sensor_range_km=kw.pop("sensor_range_km", 0.0),
            influence_km=kw.pop("influence_km", 0.0),
            engage_km=kw.pop("engage_km", 15.0),
            loop=kw.pop("loop", False),
            road_bound=kw.pop("road_bound", True),
            attributes=kw.pop("attributes", {}) or {},
        )

    shift = lambda route, dlat, dlon: [(la + dlat, lo + dlon) for la, lo in route]

    # -- 红方：加一个水面编队、一个岸防火力群、一个陆上机动群 -----------------
    red_escort_route = shift(destroyer_route, 0.16, -0.05)
    red_patrol_route = shift(destroyer_route, 0.30, 0.22)
    red_lst_route = shift(destroyer_route, -0.30, -0.55)
    # ⚠ 陆上机动单位的路线**绝不能从 `coastal_route` 派生**：那条线是给 R04
    # （岸舰导弹营）用的，而 R04 是 `road_bound=False` 的**阵地** —— 它的点位根本
    # 不必在路边，实测其中一个点还在海里。拿它平移出来的装甲/炮兵直接路由失败
    # （"more than 25 km from any road"），而且会让整个数据集的 `road_constrained`
    # 变成 False，`road_selftest` 随即**跳过**整组检查 —— 一个坐标错误就这样把
    # 一整项验证悄悄关掉了。
    # 改成从**已经在路网上的**两栖分队环线（R09/B06，实测 routed）做 2~3 km 的
    # 极小平移：偏移远小于路由器 25 km 的吸附半径，必然落在同一段路上。
    red_armor_route = [(23.02, 120.17), (23.10, 120.22), (23.04, 120.28),
                       (22.97, 120.22), (23.02, 120.17)]
    # R16 用 R09 的顶点做 ≈500 m 的平移。先前用 2 km 平移时点位虽然吸附到了路上，
    # 但**吸附后落在不连通的路网分量里**，报的是 "no road route from A to B" ——
    # 与拉达克那段注释是同一件事：路网是碎的，偏移一大就跨了分量。
    # 500 m 远小于分量尺度，必然落到同一段路上。
    red_artillery_route = [(22.995, 120.155), (23.075, 120.205), (23.015, 120.265),
                           (22.945, 120.205), (22.995, 120.155)]

    # -- 蓝方：加一个水面编队、一个陆上机动群 -------------------------------
    blue_destroyer_route = shift(blue_frigate_route, 0.12, 0.26)
    blue_supply_route = shift(blue_frigate_route, -0.10, 0.36)
    # 同上：从 B06（实测 routed）的环线做极小平移，而不是从阵地路线派生。
    blue_armor_route = [(23.225, 120.215), (23.125, 120.295), (23.005, 120.145),
                        (23.105, 120.175), (23.225, 120.215)]
    blue_artillery_route = [(23.22, 120.25), (23.12, 120.33), (23.00, 120.18),
                            (23.10, 120.21), (23.22, 120.25)]

    return [
        # ================= red surface =================
        Target(
            target_id="R01",
            name="红方驱逐舰编队指挥舰",
            platform_class="驱逐舰",
            domain="sea",
            affiliation="red",
            waypoints=destroyer_route,
            speed_kn=18.0,
            role="编队指挥",
            model="052D 型驱逐舰",
            model_note="北约代号 Luyang III",
            icon="destroyer",
            sensor_name="346A 型相控阵雷达",
            sensor_range_km=400.0,
            influence_km=180.0,
            engage_km=120.0,
            attributes={
                "舰种": "导弹驱逐舰",
                "编制": "水面舰艇编队",
                "任务": "海峡穿越",
                "满载排水量": "约 7500 t",
                "最大航速": "30 kn",
            },
        ),
        Target(
            target_id="R02",
            name="红方护卫舰",
            platform_class="护卫舰",
            domain="sea",
            affiliation="red",
            waypoints=frigate_route,
            speed_kn=18.0,
            role="伴随护航",
            model="054A 型护卫舰",
            model_note="北约代号 Jiangkai II",
            icon="frigate",
            sensor_name="382 型三坐标对空雷达",
            sensor_range_km=250.0,
            influence_km=120.0,
            engage_km=60.0,
            attributes={
                "舰种": "导弹护卫舰",
                "编制": "水面舰艇编队",
                "任务": "伴随护航",
                "满载排水量": "约 4000 t",
                "最大航速": "27 kn",
            },
        ),
        Target(
            target_id="R03",
            name="红方综合补给舰",
            platform_class="补给舰",
            domain="sea",
            affiliation="red",
            waypoints=supply_route,
            speed_kn=16.0,
            role="后勤保障",
            model="903 型综合补给舰",
            model_note="北约代号 Fuchi",
            icon="replenishment",
            sensor_name="导航雷达",
            sensor_range_km=50.0,
            influence_km=30.0,
            engage_km=10.0,
            attributes={
                "舰种": "综合补给舰",
                "编制": "水面舰艇编队",
                "任务": "编队补给",
                "满载排水量": "约 23000 t",
                "最大航速": "19 kn",
            },
        ),
        # ================= red land =================
        Target(
            target_id="R04",
            name="红方岸舰导弹营",
            platform_class="岸舰导弹",
            domain="land",
            affiliation="red",
            # Positional area-defence battery: it fights from where it is
            # emplaced, so it is not routed on the road network.
            road_bound=False,
            waypoints=coastal_route,
            speed_kn=geo.kmh_to_knots(45.0),
            role="岸海打击",
            model="鹰击-62 岸舰导弹",
            model_note="机动岸防发射营",
            icon="coastal_missile",
            sensor_name="超视距岸基雷达",
            sensor_range_km=200.0,
            influence_km=280.0,
            engage_km=280.0,
            attributes={
                "兵种": "岸防导弹兵",
                "编制": "机动发射营",
                "任务": "岸海封控",
                "有效射程": "约 280 km",
                "机动方式": "公路机动",
            },
        ),
        Target(
            target_id="R05",
            name="红方远程防空营",
            platform_class="地面防空",
            domain="land",
            affiliation="red",
            # Positional area-defence battery: it fights from where it is
            # emplaced, so it is not routed on the road network.
            road_bound=False,
            waypoints=sam_route,
            speed_kn=geo.kmh_to_knots(30.0),
            role="要地防空",
            model="红旗-9B 远程防空导弹",
            model_note="旅属火力单元",
            icon="sam_battalion",
            sensor_name="HT-233 相控阵制导雷达",
            sensor_range_km=300.0,
            influence_km=250.0,
            engage_km=250.0,
            attributes={
                "兵种": "地面防空兵",
                "编制": "防空导弹营",
                "任务": "区域防空",
                "拦截斜距": "约 250 km",
                "拦截高度": "0.5 – 30 km",
            },
        ),
        # ================= red air =================
        Target(
            target_id="R06",
            name="红方多用途战斗机",
            platform_class="战斗机",
            domain="air",
            affiliation="red",
            waypoints=red_fighter_route,
            speed_kn=480.0,
            altitude_m=8500.0,
            role="空中掩护",
            model="歼-16 多用途战斗机",
            model_note="双座重型多用途平台",
            icon="fighter",
            sensor_name="有源相控阵雷达",
            sensor_range_km=200.0,
            influence_km=150.0,
            engage_km=100.0,
            attributes={
                "机型": "多用途战斗机",
                "任务": "空中掩护",
                "高度层": "中高空",
                "实用升限": "17000 m",
                "作战半径": "约 1500 km",
            },
        ),
        Target(
            target_id="R07",
            name="红方预警机",
            platform_class="预警机",
            domain="air",
            affiliation="red",
            waypoints=red_aew_route,
            speed_kn=320.0,
            altitude_m=9000.0,
            role="空中指挥",
            model="空警-500 预警机",
            model_note="背负式三面有源相控阵",
            icon="aew",
            sensor_name="机载预警雷达",
            sensor_range_km=470.0,
            influence_km=400.0,
            engage_km=15.0,
            attributes={
                "机型": "预警指挥机",
                "任务": "空中预警指挥",
                "高度层": "高空",
                "实用升限": "10000 m",
                "续航时间": "约 11 h",
            },
        ),
        # ================= blue surface =================
        Target(
            target_id="B01",
            name="蓝方护卫舰",
            platform_class="护卫舰",
            domain="sea",
            affiliation="blue",
            waypoints=blue_frigate_route,
            speed_kn=20.0,
            role="海上拦截",
            model="成功级护卫舰",
            model_note="佩里级衍生型",
            icon="frigate",
            sensor_name="SPS-49 对空搜索雷达",
            sensor_range_km=250.0,
            influence_km=120.0,
            engage_km=60.0,
            attributes={
                "舰种": "导弹护卫舰",
                "编制": "水面舰艇编队",
                "任务": "海上拦截",
                "满载排水量": "约 4100 t",
                "最大航速": "29 kn",
            },
        ),
        Target(
            target_id="B02",
            name="蓝方巡逻舰",
            platform_class="巡逻舰",
            domain="sea",
            affiliation="blue",
            waypoints=blue_corvette_route,
            speed_kn=24.0,
            role="近海巡逻",
            model="沱江级巡逻舰",
            model_note="高速隐身双体构型",
            icon="corvette",
            sensor_name="相控阵搜索雷达",
            sensor_range_km=120.0,
            influence_km=60.0,
            engage_km=40.0,
            attributes={
                "舰种": "导弹巡逻舰",
                "编制": "近海巡逻队",
                "任务": "近海警戒",
                "满载排水量": "约 600 t",
                "最大航速": "43 kn",
            },
        ),
        # ================= blue land =================
        Target(
            target_id="B03",
            name="蓝方要地防空连",
            platform_class="地面防空",
            domain="land",
            affiliation="blue",
            # Positional area-defence battery: it fights from where it is
            # emplaced, so it is not routed on the road network.
            road_bound=False,
            waypoints=blue_sam_route,
            speed_kn=geo.kmh_to_knots(25.0),
            role="要地防空",
            model="爱国者-3 防空导弹",
            model_note="PAC-3 MSE 构型",
            icon="sam_battalion",
            sensor_name="AN/MPQ-65 相控阵雷达",
            sensor_range_km=170.0,
            influence_km=140.0,
            engage_km=120.0,
            attributes={
                "兵种": "地面防空兵",
                "编制": "防空导弹连",
                "任务": "要地防空",
                "拦截斜距": "约 120 km",
                "拦截高度": "0.06 – 20 km",
            },
        ),
        # ================= blue air =================
        Target(
            target_id="B04",
            name="蓝方战斗机",
            platform_class="战斗机",
            domain="air",
            affiliation="blue",
            waypoints=blue_fighter_route,
            speed_kn=450.0,
            altitude_m=8000.0,
            role="战斗空中巡逻",
            model="F-16V 战斗机",
            model_note="Block 70/72 构型",
            icon="fighter",
            sensor_name="APG-83 有源相控阵雷达",
            sensor_range_km=150.0,
            influence_km=110.0,
            engage_km=90.0,
            loop=True,
            attributes={
                "机型": "多用途战斗机",
                "任务": "战斗空中巡逻",
                "高度层": "中空",
                "实用升限": "15000 m",
                "作战半径": "约 550 km",
            },
        ),
        Target(
            target_id="B05",
            name="蓝方预警机",
            platform_class="预警机",
            domain="air",
            affiliation="blue",
            waypoints=blue_aew_route,
            speed_kn=300.0,
            altitude_m=10000.0,
            role="空中警戒",
            model="E-2K 预警机",
            model_note="鹰眼系列预警指挥机",
            icon="aew",
            sensor_name="AN/APS-145 预警雷达",
            sensor_range_km=450.0,
            influence_km=380.0,
            engage_km=15.0,
            loop=True,
            attributes={
                "机型": "舰载预警指挥机",
                "任务": "空中预警指挥",
                "高度层": "高空",
                "实用升限": "10500 m",
                "续航时间": "约 6 h",
            },
        ),

        # ------------------------------------------------------------------
        # 编成扩充（本轮）：原 12 个平台 → 22 个。
        #
        # 新增项统一由下面的 `_t()` 构造，避免十几份逐字重复的字段列表；**字段语义与
        # 上面的条目完全一致**，属性只保留最关键的两三项（型号信息在 `model` 里）。
        #
        # 路线一律由既有航线做**小量平移**派生：同一战区里各编队本来就沿相近走廊
        # 机动，没必要手工再编一套坐标；平移量刻意互不相同，避免新目标与旧目标重合
        # 而让"编成"看起来只有几个平台。两条约束照旧适用：
        #   ① 陆上机动单位必须落在路网上（`road_constrained` 与 road_selftest 逐帧检查）；
        #   ② 阵地类（岸舰导弹、防空）用 `road_bound=False` —— 它们由驻防位置定义，
        #      不是由行军定义，强行寻路既不符合实际，也会让数据集的路网标志失真。
        # ------------------------------------------------------------------
        Target(
            target_id="R11", name="红方第二驱逐舰", platform_class="驱逐舰",
            domain="sea", affiliation="red", waypoints=red_escort_route,
            speed_kn=26.0, role="编队护航", model="052D 型驱逐舰",
            icon="destroyer", sensor_name="346A 相控阵雷达",
            sensor_range_km=280.0, influence_km=200.0, engage_km=120.0,
            attributes={"任务": "编队区域防空", "编成": "第二水面编队"},
        ),
        Target(
            target_id="R12", name="红方第二护卫舰", platform_class="护卫舰",
            domain="sea", affiliation="red", waypoints=shift(frigate_route, 0.20, 0.05),
            speed_kn=27.0, role="编队护航", model="054A 型护卫舰",
            icon="frigate", sensor_name="382 型三坐标雷达",
            sensor_range_km=180.0, influence_km=130.0, engage_km=70.0,
            attributes={"任务": "反潜护航", "编成": "第二水面编队"},
        ),
        Target(
            target_id="R13", name="红方登陆舰编队", platform_class="登陆舰",
            domain="sea", affiliation="red", waypoints=red_lst_route,
            speed_kn=18.0, role="兵力投送", model="071 型综合登陆舰",
            icon="replenishment", influence_km=90.0, engage_km=20.0,
            attributes={"任务": "输送登陆兵与装备", "装载": "两栖合成营"},
        ),
        Target(
            target_id="R14", name="红方巡逻艇分队", platform_class="巡逻舰",
            domain="sea", affiliation="red", waypoints=red_patrol_route,
            speed_kn=30.0, role="近岸巡逻", model="056A 型轻型护卫舰",
            icon="corvette", sensor_range_km=90.0, influence_km=60.0, engage_km=25.0,
            loop=True, attributes={"任务": "近岸警戒巡逻"},
        ),
        Target(
            target_id="R15", name="红方装甲突击群", platform_class="装甲分队",
            domain="land", affiliation="red", waypoints=red_armor_route,
            speed_kn=22.0, role="地面突击", model="99A 主战坦克",
            icon="armor", influence_km=35.0, engage_km=6.0,
            attributes={"编成": "合成营装甲连", "任务": "沿路网向内陆推进"},
        ),
        Target(
            target_id="R16", name="红方炮兵群", platform_class="炮兵分队",
            domain="land", affiliation="red", waypoints=red_artillery_route,
            speed_kn=20.0, role="地面火力支援", model="PLZ-05 自行加榴炮",
            icon="artillery", sensor_range_km=50.0, influence_km=60.0, engage_km=45.0,
            attributes={"编成": "炮兵营", "任务": "随伴火力支援"},
        ),
        Target(
            target_id="R17", name="红方岸舰导弹营（第二阵地）", platform_class="岸舰导弹",
            domain="land", affiliation="red",
            waypoints=[(24.42, 118.62), (24.36, 118.70), (24.42, 118.62)],
            speed_kn=6.0, role="岸防", model="鹰击-62 岸舰导弹",
            icon="coastal_missile", sensor_name="岸基超视距雷达",
            sensor_range_km=200.0, influence_km=180.0, engage_km=150.0,
            loop=True, road_bound=False,
            attributes={"任务": "封控海峡南口", "部署": "机动岸防阵地"},
        ),
        Target(
            target_id="R18", name="红方远程防空营（第二阵地）", platform_class="防空导弹",
            domain="land", affiliation="red",
            waypoints=[(25.44, 119.06), (25.52, 119.16), (25.44, 119.06)],
            speed_kn=5.0, role="区域防空", model="红旗-9B 远程防空导弹",
            icon="sam_battalion", sensor_name="HT-233 制导雷达",
            sensor_range_km=300.0, influence_km=220.0, engage_km=200.0,
            loop=True, road_bound=False,
            attributes={"任务": "掩护登陆编队上陆", "射程": "约 200 km"},
        ),
        Target(
            target_id="R19", name="红方第二战斗机编队", platform_class="战斗机",
            domain="air", affiliation="red",
            waypoints=shift(red_fighter_route, 0.22, 0.30),
            speed_kn=460.0, altitude_m=7600.0, role="制空掩护",
            model="歼-16 多用途战斗机", icon="fighter",
            sensor_name="有源相控阵雷达", sensor_range_km=170.0,
            influence_km=130.0, engage_km=110.0, loop=True,
            attributes={"任务": "护航与制空", "高度层": "中空"},
        ),
        Target(
            target_id="R20", name="红方第二预警机", platform_class="预警机",
            domain="air", affiliation="red", waypoints=shift(red_aew_route, 0.18, -0.12),
            speed_kn=300.0, altitude_m=9200.0, role="空中警戒",
            model="空警-500 预警机", icon="aew",
            sensor_name="数字阵列预警雷达", sensor_range_km=400.0,
            influence_km=340.0, engage_km=15.0, loop=True,
            attributes={"任务": "空中预警指挥", "高度层": "高空"},
        ),
        Target(
            target_id="B07", name="蓝方驱逐舰", platform_class="驱逐舰",
            domain="sea", affiliation="blue", waypoints=blue_destroyer_route,
            speed_kn=27.0, role="舰队防空", model="基隆级驱逐舰",
            icon="destroyer", sensor_name="AN/SPS-48E 三坐标雷达",
            sensor_range_km=260.0, influence_km=190.0, engage_km=110.0,
            attributes={"任务": "海上拦截", "编成": "北部舰队"},
        ),
        Target(
            target_id="B08", name="蓝方补给舰", platform_class="补给舰",
            domain="sea", affiliation="blue", waypoints=blue_supply_route,
            speed_kn=17.0, role="海上补给", model="磐石级油弹补给舰",
            icon="replenishment", influence_km=80.0, engage_km=10.0,
            attributes={"任务": "舰队油弹补给"},
        ),
        Target(
            target_id="B09", name="蓝方装甲营", platform_class="装甲分队",
            domain="land", affiliation="blue", waypoints=blue_armor_route,
            speed_kn=24.0, role="地面防御", model="M1A2T 主战坦克",
            icon="armor", influence_km=35.0, engage_km=6.0,
            attributes={"编成": "装甲营", "任务": "机动防御"},
        ),
        Target(
            target_id="B10", name="蓝方炮兵营", platform_class="炮兵分队",
            domain="land", affiliation="blue", waypoints=blue_artillery_route,
            speed_kn=20.0, role="地面火力支援", model="M109A6 自行榴弹炮",
            icon="artillery", sensor_range_km=40.0, influence_km=55.0, engage_km=30.0,
            attributes={"编成": "炮兵营", "任务": "反登陆火力压制"},
        ),
        Target(
            target_id="B11", name="蓝方第二战斗机编队", platform_class="战斗机",
            domain="air", affiliation="blue",
            waypoints=shift(blue_fighter_route, -0.20, -0.28),
            speed_kn=440.0, altitude_m=8200.0, role="战斗空中巡逻",
            model="IDF 经国号战斗机", icon="fighter",
            sensor_name="GD-53 雷达", sensor_range_km=120.0,
            influence_km=95.0, engage_km=75.0, loop=True,
            attributes={"任务": "战斗空中巡逻", "高度层": "中高空"},
        ),
    ]


ALL_TARGETS = _build_targets()


# --------------------------------------------------------------------------
# Trajectory generation
# --------------------------------------------------------------------------


def route_land_targets(targets: List[Target], theatre: Optional[str] = None
                       ) -> Tuple[Dict[str, List], Dict[str, dict]]:
    """Snap land targets' waypoints onto the road network and route between them.

    Returns `(routes, meta)` where `routes` maps target id -> denormalised
    (lat, lon) polyline and `meta` records, per target, whether it was routed and
    which road classes it uses.

    Sea and air targets are returned untouched: a ship is not on a road, and
    pretending otherwise would put `on_road` evidence into the intent layer for a
    platform where it means nothing.

    A land target whose waypoints cannot be snapped, or whose legs are
    unreachable, is **left on its straight-line route and reported** in `meta`
    rather than being silently dropped or silently routed the wrong way. The
    dataset carries that flag so the road self-test can tell "routed and correct"
    apart from "could not route".
    """
    from . import roads as roads_mod

    routes: Dict[str, List] = {}
    meta: Dict[str, dict] = {}

    network = None
    if theatre and any(t.domain == "land" for t in targets):
        try:
            network = roads_mod.load(theatre)
        except roads_mod.RoadError as e:
            network = None
            network_error = str(e)
        else:
            network_error = None
    else:
        network_error = "no theatre given"

    for t in targets:
        if t.domain != "land" or network is None or not getattr(t, "road_bound", True):
            routes[t.target_id] = list(t.waypoints)
            meta[t.target_id] = {
                "routed": False,
                "road_bound": bool(getattr(t, "road_bound", True)),
                "reason": ("not a land target" if t.domain != "land"
                           else "positional battery, not road-bound"
                           if not getattr(t, "road_bound", True)
                           else network_error or "no road network"),
            }
            continue
        try:
            path = network.route_latlon(t.waypoints)
        except roads_mod.RoadError as e:
            routes[t.target_id] = list(t.waypoints)
            meta[t.target_id] = {"routed": False, "road_bound": True,
                                 "reason": str(e)}
            continue
        routes[t.target_id] = path
        meta[t.target_id] = {
            "routed": True,
            "length_km": round(network.path_length_km(path), 2),
            "speed_kmh": round(network.path_speed_kmh(path), 1),
            "osm_base_utc": network.meta.get("osm_base_utc"),
        }

    return routes, meta


def build_frames(targets: Optional[List[Target]] = None,
                 theatre: Optional[str] = None,
                 routes: Optional[Dict[str, List]] = None) -> List[dict]:
    """Generate per-tick positions for every target.

    Each frame carries the simulation time plus one entry per target with its
    position, altitude, speed, heading and along-track distance. This is the
    canonical per-tick state that the front end replays and that a later
    PostgreSQL trajectory table would store.

    Land targets are normally routed along the road network first (see
    `route_land_targets`); pass the result as `routes` so the roads are not
    re-solved for every caller. Without it, land targets fall back to their
    straight-line waypoints.
    """
    targets = targets if targets is not None else ALL_TARGETS
    if routes is None:
        routes, _ = route_land_targets(targets, theatre)

    dense_routes = {}
    lengths = {}
    for t in targets:
        dense = geo.densify(routes.get(t.target_id) or t.waypoints,
                            max_segment_m=2000.0)
        dense_routes[t.target_id] = dense
        lengths[t.target_id] = geo.path_length_m(dense)

    frames = []
    for tick in range(TICK_COUNT + 1):
        t_seconds = tick * TICK_SECONDS
        state = []
        for t in targets:
            travelled = t.speed_mps * t_seconds
            total = lengths[t.target_id]
            if t.loop and total > 0.0:
                # Patrol circuits repeat; the along-track distance is the
                # position on the current lap, the odometer keeps counting.
                along = travelled % total
                on_lap = round(travelled / total, 3)
            else:
                along = min(travelled, total)
                on_lap = 0.0
            (lat, lon), bearing = geo.sample_polyline(dense_routes[t.target_id], along)
            state.append(
                {
                    "id": t.target_id,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "alt": round(t.altitude_m, 1),
                    "speed_kn": t.speed_kn,
                    "heading": round(bearing, 1),
                    "travelled_m": round(along, 1),
                    "laps": on_lap,
                    "reached_end": (not t.loop) and along >= total,
                }
            )
        frames.append({
            "tick": tick,
            "t_seconds": round(t_seconds, 1),
            # Operation phase, attached to every frame so the front end can label
            # the situation without re-deriving the timeline.
            "phase": phase_at(tick),
            # Per-unit phase ids. A unit may lag the operation (still crossing while
            # the first wave is ashore) or have finished (a transport holds at 航渡
            # rather than being carried into 向内陆夺控). Precomputed here so the cap
            # rule lives in one place.
            "unit_phase": {
                t.target_id: unit_phase_name(tick, t.target_id) for t in targets
            },
            "targets": state,
        })

    return frames


def targets_for(case) -> List[Target]:
    """Subset of the order of battle selected by a demo case.

    The pool depends on the case's theatre: the Taiwan Strait order of battle is
    maritime and the India-China one is a high-altitude land/air force, so mixing
    them would put a coastal missile brigade on a Ladakh road.
    """
    wanted = set(case.target_ids)
    pool = (INDIA_CHINA_TARGETS if getattr(case, "theatre", None) == "india_china"
            else (ALL_TARGETS + AMPHIBIOUS_TARGETS))
    picked = [t for t in pool if t.target_id in wanted]
    if len(picked) != len(wanted):
        missing = sorted(wanted - {t.target_id for t in picked})
        raise KeyError("case references unknown target ids: %s" % ", ".join(missing))
    return picked


# --------------------------------------------------------------------------
# Theatres
# --------------------------------------------------------------------------
#
# The Taiwan Strait theatre is the module-level default above (REGION, ANCHORS,
# _HOTSPOT_BOXES) so every existing caller keeps working. The India-China theatre
# lives here and is selected per case via `DemoCase.theatre`.

INDIA_CHINA_REGION = "中印边境西段（拉达克）"

# Amphibious lift for the Taiwan landing case: the ship-to-shore phase that a
# maritime order of battle alone cannot express.
#
# The landing is placed at 台南 安平 (≈23.00N, 120.15E) because the fetched road
# network reaches only to **120.55E**. The natural landing beaches further north
# (桃园/淡水, 121.1-121.4E) are 40+ km from any road in the graph, so land units
# there cannot be routable at all; 安平 snaps at 0.14 km and chains through it
# route in 33-56 km. Moving the beach to match the data is the honest option -
# the alternative was to bake a much larger road graph, and even then the
# cross-strait graph is not contiguous in OSM.
AMPHIBIOUS_TARGETS: List[Target] = [
    Target(
        target_id="R08",
        name="红方两栖船坞登陆舰",
        platform_class="两栖舰",
        domain="sea",
        affiliation="red",
        # 集结锚地 → 海峡航渡 → 换乘海域（安平外海）
        waypoints=[
            (25.60, 119.30), (24.90, 119.40), (24.10, 119.70),
            (23.40, 119.95), (23.05, 120.05),
        ],
        speed_kn=16.0,
        role="两栖输送",
        model="071 型船坞登陆舰",
        model_note="北约代号 Yuzhao",
        icon="replenishment",
        sensor_name="对海搜索雷达",
        sensor_range_km=120.0,
        influence_km=60.0,
        engage_km=8.0,
        attributes={
            "舰种": "船坞登陆舰",
            "任务阶段": "集结装载→航渡→换乘",
            "装载": "两栖战车 / 登陆兵",
            "满载排水量": "约 25000 t",
        },
    ),
    Target(
        target_id="R09",
        name="红方两栖突击车分队",
        platform_class="两栖战车",
        domain="land",
        affiliation="red",
        # 泛水 → 抢滩 → 沿路向内陆夺控。路网路由把陆上段约束在道路上，
        # 泛水段是水面到岸的直线过渡（两栖车本来就能自行泛水，不是陆上机动）。
        waypoints=[(23.00, 120.15), (23.08, 120.20), (23.02, 120.26),
                   (22.95, 120.20), (23.00, 120.15)],
        speed_kn=8.0,
        altitude_m=0.0,
        role="抢滩突击",
        model="05 式两栖突击车",
        model_note="泛水自行登陆",
        icon="armor",
        loop=True,
        sensor_name="车通/光电观瞄",
        sensor_range_km=6.0,
        influence_km=10.0,
        engage_km=3.0,
        attributes={
            "兵种": "两栖装甲",
            "阶段": "泛水→抢滩→向内陆夺控",
            "约束": "陆上段沿路网，泛水段自行航渡",
        },
    ),
    Target(
        target_id="R10",
        name="红方武装直升机分队",
        platform_class="武装直升机",
        domain="air",
        affiliation="red",
        # 近距支援：随抢滩梯队前出，沿滩头线往返
        waypoints=[(23.20, 120.05), (23.05, 120.20), (22.92, 120.28),
                   (23.02, 120.10), (23.20, 120.05)],
        speed_kn=140.0,
        altitude_m=600.0,
        role="近距空中支援",
        model="直-10 武装直升机",
        model_note="滩头火力支援",
        icon="fighter",
        loop=True,
        sensor_name="光电/毫米波",
        sensor_range_km=20.0,
        influence_km=15.0,
        engage_km=6.0,
        attributes={"兵种": "陆航", "任务": "滩头火力支援", "升限": "约 6000 m"},
    ),
    Target(
        target_id="B06",
        name="蓝方机动装甲分队",
        platform_class="装甲分队",
        domain="land",
        affiliation="blue",
        # 滩头反突击：从内陆向登陆场机动
        waypoints=[(23.22, 120.22), (23.12, 120.30), (23.00, 120.15),
                   (23.10, 120.18), (23.22, 120.22)],
        speed_kn=10.0,
        altitude_m=20.0,
        role="滩头反突击",
        model="CM-11 主战坦克",
        model_note="沿岸机动预备队",
        icon="armor",
        loop=True,
        sensor_name="车通/光电",
        sensor_range_km=8.0,
        influence_km=12.0,
        engage_km=4.0,
        attributes={"兵种": "装甲", "任务": "反登陆机动预备队"},
    ),
]

# Theatre bounding boxes (west, south, east, north). Single source for the tools
# that must bake or route per theatre: `tools/fetch_base_imagery.py` (offline
# imagery/terrain) and `tools/fetch_roads.py`. Kept here rather than duplicated in
# each tool because the two theatres are disjoint, and a bake that picks the wrong
# box silently re-bakes the other theatre and reports success.
THEATRE_BBOXES = {
    "taiwan_strait": (115.5, 20.0, 122.5, 26.5),
    "india_china": (77.0, 31.5, 81.5, 36.5),
}

# 战区的**人类可读名称**，与 THEATRE_BBOXES 一一对应、同样按 theatre 键索引。
#
# 为什么需要这张表：`region` 原先直接取模块级 `REGION`（台海），于是中印案例的数据集里
# `region` 写的是「中国 台湾海峡」——**实测确认**：`web/data/india_china_west.json`
# 与台海案例的 region 完全相同。根因是"模块级默认值泄漏进按案例生成的数据"，
# 不是某处写错了一个字符串——同一个文件里 `theatre` 是**按案例**求值的，而 region 是
# **按模块**取的，两者不同源，所以无论加多少案例都不会自己好。
#
# 与 THEATRE_BBOXES 放在一起，是为了让"一个战区一套参数"只有一个出处：
# 将来加战区时 bbox 与 region 必须一起加，漏一个会立刻互相矛盾。
THEATRE_REGIONS = {
    "taiwan_strait": REGION,
    "india_china": INDIA_CHINA_REGION,
}

# Landing operation phases, as (id, name, first_tick). The phases are a property
# of the *operation*, not of a single platform: every unit is somewhere in the same
# six-stage narrative at any moment, which is what lets the UI say "the operation
# is at the ship-to-shore stage" and what lets the intent sets be read in context
# (a `HOLD` during 巩固滩头 means something different from a `HOLD` during 航渡).
#
# Boundaries are declared rather than inferred from unit positions, because
# inferring them would mean guessing a stage transition from kinematics - the same
# mistake as guessing intent from a single frame. They are placed to match the
# route design: R08's final leg reaches the 安平 换乘 area at tick ~600, and the
# land units begin their inland run at ~1080.
LANDING_PHASES = [
    ("assembly", "集结装载", 0),
    ("transit", "航渡", 240),
    ("transfer", "换乘编波", 600),
    ("assault", "抢滩突击", 840),
    ("consolidate", "巩固滩头", 1080),
    ("inland", "向内陆夺控", 1320),
]

# Which units take part in each phase. Declared once here rather than as a field on
# every target, because it is the *operation's* structure, and because a unit that
# is still in an earlier phase than the operation is the interesting case: it shows
# as a phase lag rather than being flattened into the operation label.
PHASE_TARGETS = {
    # loading at the assembly anchorage
    "assembly": ("R03", "R08", "R09", "R10"),
    # crossing the strait under escort and air cover
    "transit": ("R01", "R02", "R03", "R06", "R07", "R08", "R09", "B01", "B02", "B04", "B05"),
    # ship-to-shore: the amphibious lift and its close support. R03 (the supply
    # ship) is deliberately absent - it stays with the fleet and is capped at 航渡.
    "transfer": ("R02", "R08", "R09", "R10", "B01", "B02"),
    # the beach assault itself
    "assault": ("R09", "R10", "R06", "B02", "B04", "B06"),
    # holding the beachhead
    "consolidate": ("R09", "R10", "R05", "B04", "B06"),
    # pushing inland
    "inland": ("R09", "R06", "B06"),
}


def unit_phase_ids(target_id: str) -> Optional[str]:
    """The last phase a unit takes part in, or None if it takes part in none.

    Exposed so the frame builder can precompute each unit's phase instead of the
    front end re-deriving it: the cap depends on `PHASE_TARGETS`, and duplicating
    that rule in JavaScript would let the two drift apart silently.
    """
    last = None
    for pid, _name, _start in LANDING_PHASES:
        if target_id in PHASE_TARGETS.get(pid, ()):
            last = pid
    return last


def unit_phase_name(tick: int, target_id: str) -> dict:
    """A unit's phase at `tick`: the operation's stage, capped at the last phase the
    unit takes part in."""
    op = phase_at(tick)
    cap = unit_phase_ids(target_id)
    if cap is None:
        return {"id": None, "name": "不参与登陆", "index": -1}
    names = {p[0]: (i, p[1]) for i, p in enumerate(LANDING_PHASES)}
    ci, cname = names[cap]
    if op["index"] <= ci:
        return {"id": op["id"], "name": op["name"], "index": op["index"]}
    return {"id": cap, "name": cname, "index": ci}


def phase_at(tick: int, target_ids: Optional[Sequence[str]] = None) -> dict:
    """Which landing phase a tick falls in.

    Returns `{"id", "name", "index", "since_tick", "ticks_in"}`. `ticks_in` lets the
    UI show how long the operation has been in this stage, which is the readable
    counterpart to the intent set's own contraction.

    When `target_ids` is given, the phase is reported from that unit's point of
    view: a phase the unit does not participate in is not "its" phase, even while
    the operation as a whole is in it. That matters for the reason per-unit phases
    exist at all - **a unit still in a phase the operation has left is behind
    schedule**, and with a single operation-level label that fact is invisible.
    A convoy still crossing the strait while the first wave is ashore is exactly the
    kind of thing a situation display should surface, not hide.
    """
    idx = 0
    for i, (_pid, _name, start) in enumerate(LANDING_PHASES):
        if tick >= start:
            idx = i
        else:
            break
    if target_ids:
        want = set(target_ids)
        # Phases this unit takes part in. Participation is not necessarily
        # contiguous, but for a landing operation it is: 航渡 precedes 换乘 which
        # precedes 抢滩, and the phases a unit is *not* listed in are the ones after
        # it has finished. So the unit's phase advances while the operation is in a
        # phase it takes part in, and is then capped at the last such phase - a
        # transport that has delivered its load holds at 航渡 instead of being
        # carried along into 向内陆夺控, and does not read as permanently stuck.
        active = [i for i, (pid_j, _n, _s) in enumerate(LANDING_PHASES)
                  if any(tid in want for tid in PHASE_TARGETS.get(pid_j, ()))]
        if active:
            idx = min(idx, active[-1])
    pid, name, start = LANDING_PHASES[idx]
    return {
        "id": pid,
        "name": name,
        "index": idx,
        "since_tick": start,
        "ticks_in": int(tick - start),
        "count": len(LANDING_PHASES),
    }

INDIA_CHINA_HOTSPOT_BOXES = [
    ("加勒万河谷", (78.50, 34.60, 78.80, 34.85), "2020 年冲突点", "valley"),
    ("班公湖", (78.45, 33.45, 79.10, 33.95), "摩擦点/脱离接触区", "lake"),
    ("德普桑平地", (77.65, 35.15, 78.05, 35.50), "巡逻权争议平原", "plain"),
    ("典角", (79.30, 32.60, 79.65, 32.90), "2024 脱离接触点", "valley"),
    ("楚舒勒", (78.60, 33.35, 78.95, 33.60), "前沿枢纽", "plain"),
    ("诺玛", (78.45, 33.05, 78.85, 33.35), "前进机场/集结", "base"),
    ("列城", (77.45, 34.00, 77.75, 34.35), "战区后勤枢纽", "base"),
]

INDIA_CHINA_HOTSPOTS = [_hotspot(n, b, note, kind)
                        for n, b, note, kind in INDIA_CHINA_HOTSPOT_BOXES]

# Land order of battle for the India-China theatre. Kept separate from ALL_TARGETS
# because the Taiwanese order of battle is a maritime one: reusing those ids here
# would put a coastal missile brigade on a Ladakh road.
#
# Every waypoint chain below was checked against the fetched network, and each one
# stays inside a single road component. That matters more here than in Taiwan:
# Ladakh's roads are *not* contiguous in OSM (52 components; Leh is in component
# 13, Chushul/Galwan in 16, Depsang/DBO in 43), so a chain that crosses a
# component boundary cannot be routed and would be reported as unrouted.
INDIA_CHINA_TARGETS: List[Target] = [
    Target(
        target_id="H01",
        name="红方高原合成营装甲分队",
        platform_class="装甲分队",
        domain="land",
        affiliation="red",
        # 典角—楚舒勒 沿 Indus 谷，实测 280.6 km
        waypoints=[
            (32.72, 79.47), (33.00, 79.10), (33.30, 78.90), (33.55, 78.75),
            (32.95, 79.15), (32.72, 79.47),
        ],
        speed_kn=8.0,
        altitude_m=4300.0,
        role="地面机动",
        model="15 式轻型坦克",
        model_note="高原型，功率降额",
        icon="armor",
        loop=True,
        sensor_name="车通/光电观瞄",
        sensor_range_km=8.0,
        influence_km=12.0,
        engage_km=4.0,
        attributes={
            "兵种": "高原装甲",
            "海拔": "4200–4600 m",
            "动力": "高原功率降额 ~30%",
            "机动": "需沿既有道路，越野受限",
        },
    ),
    Target(
        target_id="H02",
        name="蓝方列城—诺玛后勤运输队",
        platform_class="运输队",
        domain="land",
        affiliation="blue",
        # 列城—诺玛—班公湖 后勤干线，实测 283.1 km
        waypoints=[
            (34.16, 77.58), (33.95, 77.85), (33.60, 78.20), (33.20, 78.65),
            (33.55, 78.70),
        ],
        speed_kn=6.0,
        altitude_m=3800.0,
        role="后勤保障",
        model="高原运输车队",
        model_note="沿公路干线前送",
        icon="armor",
        sensor_name="导航",
        sensor_range_km=2.0,
        influence_km=4.0,
        engage_km=0.5,
        attributes={
            "兵种": "后勤运输",
            "路线": "列城—诺玛干线",
            "约束": "严格沿路网，脱离道路即失去机动能力",
        },
    ),
    Target(
        target_id="H03",
        name="蓝方德普桑前哨分队",
        platform_class="装甲分队",
        domain="land",
        affiliation="blue",
        # 德普桑—DBO 走廊，实测 22.3 km（独立连通分量）
        waypoints=[(35.30, 77.90), (35.40, 77.82), (35.42, 77.85), (35.30, 77.90)],
        speed_kn=5.0,
        altitude_m=4900.0,
        role="前沿警戒",
        model="高原哨所分队",
        model_note="DBO 走廊",
        icon="armor",
        loop=True,
        sensor_name="地面监视雷达",
        sensor_range_km=20.0,
        influence_km=15.0,
        engage_km=3.0,
        attributes={"兵种": "前沿警戒", "海拔": "约 4900 m"},
    ),
    Target(
        target_id="H04",
        name="蓝方列城航空兵分队",
        platform_class="战斗机",
        domain="air",
        affiliation="blue",
        waypoints=[(34.15, 77.60), (33.60, 78.10), (33.10, 78.70), (33.70, 79.00),
                   (34.15, 77.60)],
        speed_kn=420.0,
        altitude_m=9000.0,
        role="战斗空中巡逻",
        model="高空战斗机",
        model_note="高原机场起降",
        icon="fighter",
        loop=True,
        sensor_name="机载雷达",
        sensor_range_km=160.0,
        influence_km=80.0,
        engage_km=60.0,
        attributes={"兵种": "航空兵", "基地": "列城", "海拔": "起降场 3500 m"},
    ),
    Target(
        target_id="H05",
        name="红方无人机侦察分队",
        platform_class="无人机",
        domain="air",
        affiliation="red",
        waypoints=[(33.30, 79.60), (33.70, 79.10), (34.10, 78.70), (33.70, 79.10),
                   (33.30, 79.60)],
        speed_kn=110.0,
        altitude_m=7500.0,
        role="侦察监视",
        model="中高空长航时无人机",
        model_note="高原侦察",
        icon="aew",
        loop=True,
        sensor_name="光电/合成孔径雷达",
        sensor_range_km=120.0,
        influence_km=60.0,
        engage_km=0.0,
        attributes={"兵种": "无人侦察", "升限": "受高原空气密度限制"},
    ),

    # ----------------------------------------------------------------------
    # 编成扩充（本轮）：5 → 10 个平台。
    #
    # 这里**刻意只加"空中平台"与"阵地类"**，不加新的陆上机动单位 ——
    # 上面第 1258-1260 行的注释已经说明：拉达克的 OSM 路网是碎成 52 个连通分量的
    # （Leh 在 13、Chushul/Galwan 在 16、Depsang/DBO 在 43），跨分量的链路根本
    # 寻不了路，会被如实报成 unrouted。新增陆上机动单位如果不能确定落在既有分量
    # 里，就会把一个"数据问题"伪装成"路网问题"。
    # 阵地类用 `road_bound=False`：由驻防位置定义，不参与寻路。
    # ----------------------------------------------------------------------
    Target(
        target_id="H06",
        name="蓝方战斗机巡逻编队",
        platform_class="战斗机",
        domain="air",
        affiliation="blue",
        waypoints=[(34.05, 77.60), (34.30, 78.05), (34.05, 78.55), (33.75, 78.10),
                   (34.05, 77.60)],
        speed_kn=430.0,
        altitude_m=9500.0,
        role="战斗空中巡逻",
        model="阵风战斗机",
        model_note="高原机场起降",
        icon="fighter",
        loop=True,
        sensor_name="RBE2 有源相控阵雷达",
        sensor_range_km=160.0,
        influence_km=120.0,
        engage_km=95.0,
        attributes={"任务": "争夺制空权", "高度层": "高空"},
    ),
    Target(
        target_id="H07",
        name="红方第二无人机侦察分队",
        platform_class="无人机",
        domain="air",
        affiliation="red",
        waypoints=[(33.90, 78.35), (34.25, 78.85), (34.55, 79.35), (34.25, 78.85),
                   (33.90, 78.35)],
        speed_kn=120.0,
        altitude_m=8000.0,
        role="侦察监视",
        model="中高空长航时无人机",
        model_note="DBO 走廊方向",
        icon="aew",
        loop=True,
        sensor_name="光电/合成孔径雷达",
        sensor_range_km=130.0,
        influence_km=70.0,
        engage_km=0.0,
        attributes={"兵种": "无人侦察", "任务": "监视 DBO 走廊"},
    ),
    Target(
        target_id="H08",
        name="蓝方炮兵阵地",
        platform_class="炮兵分队",
        domain="land",
        affiliation="blue",
        waypoints=[(34.14, 77.52), (34.20, 77.60), (34.14, 77.52)],
        speed_kn=4.0,
        role="地面火力支援",
        model="M777 超轻型榴弹炮",
        icon="artillery",
        loop=True,
        road_bound=False,
        sensor_range_km=40.0,
        influence_km=45.0,
        engage_km=30.0,
        attributes={"部署": "高原驻锄阵地", "任务": "火力封锁前沿"},
    ),
    Target(
        target_id="H09",
        name="红方远程防空阵地",
        platform_class="防空导弹",
        domain="land",
        affiliation="red",
        waypoints=[(33.52, 78.92), (33.58, 79.00), (33.52, 78.92)],
        speed_kn=3.0,
        role="区域防空",
        model="红旗-16 中程防空导弹",
        icon="sam_battalion",
        loop=True,
        road_bound=False,
        sensor_name="搜索/制导雷达",
        sensor_range_km=180.0,
        influence_km=140.0,
        engage_km=70.0,
        attributes={"部署": "要地防空", "任务": "掩护集结地域"},
    ),
    Target(
        target_id="H10",
        name="红方远程火箭炮阵地",
        platform_class="炮兵分队",
        domain="land",
        affiliation="red",
        waypoints=[(33.62, 79.28), (33.68, 79.36), (33.62, 79.28)],
        speed_kn=4.0,
        role="远程火力打击",
        model="PHL-03 远程火箭炮",
        icon="artillery",
        loop=True,
        road_bound=False,
        sensor_range_km=60.0,
        influence_km=90.0,
        engage_km=70.0,
        attributes={"部署": "纵深火力阵地", "射程": "约 70 km"},
    ),
]


def build_dataset(case=None) -> dict:
    """Assemble the full demonstration dataset written to the web data folder.

    ``case`` is a :class:`engine.cases.DemoCase`; when omitted the default case
    (the complete order of battle) is used. The inference layer runs over the
    generated frames and is embedded in the same payload so the front end never
    has to recompute or guess a situation assessment.
    """
    from . import cases as cases_mod
    from . import inference as inference_mod

    if case is None:
        case = cases_mod.get_case(cases_mod.DEFAULT_CASE_ID)

    targets = targets_for(case)
    # 陆军先沿路网路由，再把路由结果交给 build_frames。放在这里而不是
    # build_frames 内部，是因为要同时拿到 meta（哪些目标真的路由成功）写进数据集，
    # 让 road_selftest 能区分"已约束且正确"与"没能路由"。
    theatre = getattr(case, "theatre", None)
    # Road-bound land platforms only. A positional battery is `domain == "land"`
    # but is not expected to be on a road, so including it would make the flag
    # false for a reason that says nothing about the routing.
    land_ids = {t.target_id for t in targets
                if t.domain == "land" and getattr(t, "road_bound", True)}
    land_routes, land_meta = route_land_targets(targets, theatre)
    frames = build_frames(targets, theatre=theatre, routes=land_routes)
    inferred = inference_mod.infer(frames, targets, case.hotspot_names)

    # 战术意图判定（22 种本体口径）。在推理之后做，因为规则要用 L1/L2/L3 之外的
    # 运动学与交互信息；结果**并入**推理帧的 targets，而不是替代它们——
    # inference 原有的 intent/intent_label 字段保持不动，旧客户端与简报不受影响。
    from . import intent as intent_mod
    intent_result = intent_mod.derive(frames, targets)
    intent_mod.merge_into_frames(inferred["frames"], intent_result)

    # 意图集合（CES/ShenJi）。有模型就走模型，模型缺失时**不注入**并回退规则结果，
    # 数据集的 `intent_source` 会如实写成 "rule"。绝不因为缺模型就伪造概率。
    intent_set_stats = None
    model = None
    try:
        from . import intent_model as intent_model_mod
        from . import intent_predict as intent_predict_mod
        from . import intent_features as intent_features_mod
        from . import roads as roads_mod

        model = intent_model_mod.load(intent_model_mod.model_path())
        if model is not None:
            road = None
            try:
                if theatre:
                    road = roads_mod.load(theatre)
            except roads_mod.RoadError:
                road = None
            hotspots = [h for h in (INDIA_CHINA_HOTSPOTS if theatre == "india_china"
                                    else HOTSPOTS)
                        if h["name"] in set(case.hotspot_names)]
            order = [t.target_id for t in targets]
            tracks = intent_features_mod.build_tracks(frames, targets)
            # Predict on a strided subset. Feature extraction walks road geometry
            # and per-target neighbour scans for every (target, frame); scoring all
            # 1441 frames took **1645 s** for a single 5-target case, which is not
            # a cost the dataset build can carry. INTENT_PREDICT_STRIDE ticks still
            # resolve to ~48 samples/min of playback, well above what the UI can
            # show, and the model's own temporal contraction is unaffected because
            # the stride is uniform and the projection still runs in tick order.
            # Bound the scored-frame count on top of the stride: with many targets
            # the per-frame cost dominates, and an unbounded count silently turns
            # into an unusable build time rather than a visible error.
            pred_ticks = list(range(0, len(frames), INTENT_PREDICT_STRIDE))
            if len(pred_ticks) > INTENT_MAX_PREDICT_TICKS:
                step = len(pred_ticks) / float(INTENT_MAX_PREDICT_TICKS)
                pred_ticks = [pred_ticks[int(i * step)]
                              for i in range(INTENT_MAX_PREDICT_TICKS)]
            # `reentry_grace` is counted in *scored* frames, so with a stride it
            # spans stride x grace ticks of real time. At stride 8 the default
            # grace of 2 bars a candidate after only 16 ticks, and measured on the
            # generated dataset that emptied the intent set entirely for every
            # land target (size 0, BACKGROUND for the whole case). Widening the
            # window keeps the intent of the parameter - a brief dip is noise, a
            # sustained drop is a decision - at the coarser sampling rate.
            grace = max(3, int(round(REENTRY_GRACE_TICKS / float(INTENT_PREDICT_STRIDE))))
            pred = intent_predict_mod.predict_frames(
                frames, tracks, order, model, inference_frames=inferred["frames"],
                hotspots=hotspots, routes=land_routes, road=road, ticks=pred_ticks,
                reentry_grace=grace)
            intent_predict_mod.fill_carried(inferred["frames"], pred, order)
            intent_set_stats = pred["stats"]
            intent_set_stats["predicted_ticks"] = len(pred_ticks)
            intent_set_stats["predict_stride"] = (
                pred_ticks[1] - pred_ticks[0] if len(pred_ticks) > 1
                else INTENT_PREDICT_STRIDE)
            intent_set_stats["max_predict_ticks"] = INTENT_MAX_PREDICT_TICKS
            intent_set_stats["reentry_grace"] = grace
    except Exception as e:  # noqa: BLE001
        # 模型链路是可选的；任何失败都退化为规则结果，但必须留下痕迹，
        # 否则"界面上没有集合"将无法区分"没模型"与"模型坏了"。
        intent_set_stats = {"error": "%s: %s" % (type(e).__name__, e)}
        model = None


    icons = [t.icon for t in targets]
    unknown = symbology.validate(icons)
    if unknown:
        raise ValueError("targets reference icons missing from the catalogue: %s" % unknown)

    # 把本体的压缩形态并入 model 块：前端据此把 code 翻成中文名、按父类分组，
    # 不必自己抄一份 22 条码表。
    inferred["model"]["intent_ontology"] = intent_mod.ontology_summary()

    # TS简报：把逐帧的推理结果压成一份可读的叙述。
    # 放在这里而不是前端算，是因为它依赖 case / targets 这些只有引擎侧才有的对象，
    # 而且要保证"前端显示的等级"与"简报里的等级"出自同一份推理输出。
    from . import brief as brief_mod
    situation_brief = brief_mod.build_brief(
        inferred["frames"],
        targets=targets,
        hotspot_names=case.hotspot_names,
        model=inferred["model"],
        start_utc=START_TIME_UTC,
        tick_seconds=TICK_SECONDS,
        case_id=case.case_id,
        case_name=case.name,
    )
    situation_brief["text"] = brief_mod.render_text(situation_brief)

    return {
        "schema": "situation3d.trajectory.v2",
        "case_id": case.case_id,
        "case_name": case.name,
        "case_description": case.description,
        "scenario_id": SCENARIO_ID,
        "scenario_name": SCENARIO_NAME,
        # 按**战区**取名，不是按模块默认值。见 THEATRE_REGIONS 的说明：
        # 以前这里写 REGION，于是中印案例的数据集里 region 是「中国 台湾海峡」。
        # `theatre` 为 None（旧调用方没给战区）时回落到模块级 REGION，行为与以前一致。
        "region": THEATRE_REGIONS.get(theatre, REGION),
        "theatre": theatre,
        # True only when every *land* target was actually routed on the network.
        # `road_selftest` checks the tracks when this is set and skips them
        # otherwise, instead of reporting a pass for data that was never
        # constrained. Non-land targets are excluded: a ship is not on a road, and
        # counting it would make the flag false for a reason that means nothing.
        "road_constrained": bool(land_ids) and all(
            land_meta[tid].get("routed") for tid in land_ids),
        "land_route_meta": land_meta,
        # Landing-phase timeline. Only the Taiwan landing operation is staged; the
        # India-China case is a standing confrontation with no landing sequence, so
        # it declares none rather than borrowing a narrative that does not fit.
        "phases": ([{"id": p, "name": n, "first_tick": t}
                    for (p, n, t) in LANDING_PHASES]
                   if theatre == "taiwan_strait" else []),
        "provenance": "simulated",
        "provenance_note": (
            "基于真实地理航路构造的仿真TS；航路点取自台海真实地理位置，"
            "速度与高度取自真实装备量级。陆军航路沿 OSM 路网路由。"
            "非真实 AIS/ADS-B 观测记录。"
        ),
        "tick_seconds": TICK_SECONDS,
        "tick_count": TICK_COUNT,
        "start_time_utc": START_TIME_UTC,
        "camera": case.camera,
        "symbology": symbology.catalog(),
        "hotspots": [
            h for h in (INDIA_CHINA_HOTSPOTS if theatre == "india_china" else HOTSPOTS)
            if h["name"] in set(case.hotspot_names)
        ],
        "urban_aoi": URBAN_AOI,
        "targets": [t.to_dataset_entry() for t in targets],
        "inference_model": inferred["model"],
        "inference_summary": inferred["summary"],
        "intent_summary": intent_result["summary"],
        # 意图集合（CES）链路的状态：模型版本、嵌套违例、no-reentry 次数，
        # 或缺失/失败原因。前端据此决定显示"模型"还是"规则（模型未加载）"。
        "intent_set": intent_set_stats,
        "intent_model_version": (model.meta.get("version") if model else None),
        "intent_source": ("model" if model else "rule"),
        "frames": frames,
        "inference_frames": inferred["frames"],
        "brief": situation_brief,
    }
