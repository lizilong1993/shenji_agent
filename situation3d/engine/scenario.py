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
from typing import Dict, List, Optional, Sequence

from . import geo, symbology

# --------------------------------------------------------------------------
# Scenario constants
# --------------------------------------------------------------------------

SCENARIO_ID = "taiwan_strait_demo_20260915"
SCENARIO_NAME = "台海方向联合态势演示"
REGION = "中国 台湾海峡"

# 12 simulated hours at 30 s per tick.
TICK_SECONDS = 30.0
TICK_COUNT = 1440
# 态势的仿真起始时刻（UTC）。提到常量：数据集里要写它，态势简报也要用它把
# "第 N 帧"换算成钟点，两处必须是同一个值，否则简报里会出现对不上的时间。
START_TIME_UTC = "2026-09-15T00:00:00Z"

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
    ]


ALL_TARGETS = _build_targets()


# --------------------------------------------------------------------------
# Trajectory generation
# --------------------------------------------------------------------------


def build_frames(targets: Optional[List[Target]] = None) -> List[dict]:
    """Generate per-tick positions for every target.

    Each frame carries the simulation time plus one entry per target with its
    position, altitude, speed, heading and along-track distance. This is the
    canonical per-tick state that the front end replays and that a later
    PostgreSQL trajectory table would store.
    """
    targets = targets if targets is not None else ALL_TARGETS

    routes = {}
    lengths = {}
    for t in targets:
        dense = geo.densify(t.waypoints, max_segment_m=2000.0)
        routes[t.target_id] = dense
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
            (lat, lon), bearing = geo.sample_polyline(routes[t.target_id], along)
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
        frames.append({"tick": tick, "t_seconds": round(t_seconds, 1), "targets": state})

    return frames


def targets_for(case) -> List[Target]:
    """Subset of the full order of battle selected by a demo case."""
    wanted = set(case.target_ids)
    picked = [t for t in ALL_TARGETS if t.target_id in wanted]
    if len(picked) != len(wanted):
        missing = sorted(wanted - {t.target_id for t in picked})
        raise KeyError("case references unknown target ids: %s" % ", ".join(missing))
    return picked


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
    frames = build_frames(targets)
    inferred = inference_mod.infer(frames, targets, case.hotspot_names)

    # 战术意图判定（22 种本体口径）。在推理之后做，因为规则要用 L1/L2/L3 之外的
    # 运动学与交互信息；结果**并入**推理帧的 targets，而不是替代它们——
    # inference 原有的 intent/intent_label 字段保持不动，旧客户端与简报不受影响。
    from . import intent as intent_mod
    intent_result = intent_mod.derive(frames, targets)
    intent_mod.merge_into_frames(inferred["frames"], intent_result)

    icons = [t.icon for t in targets]
    unknown = symbology.validate(icons)
    if unknown:
        raise ValueError("targets reference icons missing from the catalogue: %s" % unknown)

    # 把本体的压缩形态并入 model 块：前端据此把 code 翻成中文名、按父类分组，
    # 不必自己抄一份 22 条码表。
    inferred["model"]["intent_ontology"] = intent_mod.ontology_summary()

    # 态势简报：把逐帧的推理结果压成一份可读的叙述。
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
        "region": REGION,
        "provenance": "simulated",
        "provenance_note": (
            "基于真实地理航路构造的仿真态势；航路点取自台海真实地理位置，"
            "速度与高度取自真实装备量级。非真实 AIS/ADS-B 观测记录。"
        ),
        "tick_seconds": TICK_SECONDS,
        "tick_count": TICK_COUNT,
        "start_time_utc": START_TIME_UTC,
        "camera": case.camera,
        "symbology": symbology.catalog(),
        "hotspots": [h for h in HOTSPOTS if h["name"] in set(case.hotspot_names)],
        "urban_aoi": URBAN_AOI,
        "targets": [t.to_dataset_entry() for t in targets],
        "inference_model": inferred["model"],
        "inference_summary": inferred["summary"],
        "intent_summary": intent_result["summary"],
        "frames": frames,
        "inference_frames": inferred["frames"],
        "brief": situation_brief,
    }
