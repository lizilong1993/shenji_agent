"""Taiwan Strait demonstration scenario for the situation3d engine.

This module defines a *simulated* situation over *real* geography. Waypoints are
real locations in and around the Taiwan Strait; speeds and altitudes are chosen
to match the real performance envelope of the represented platform classes.

It is deliberately not presented as observed data: per the agreed scope, imagery
comes from real datasets while the situation itself is engine-generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import geo

# --------------------------------------------------------------------------
# Scenario constants
# --------------------------------------------------------------------------

SCENARIO_ID = "taiwan_strait_demo_20260915"
SCENARIO_NAME = "台海方向海上编队穿越与空中巡逻"
REGION = "台湾海峡"

# 12 simulated hours at 30 s per tick.
TICK_SECONDS = 30.0
TICK_COUNT = 1440

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
}

# Hotspot regions shown in the right-hand panel, mirroring the reference layout.
HOTSPOTS = [
    {"name": "中国 台湾海峡", "lat": 24.40, "lon": 119.60, "note": "主场景区域"},
    {"name": "中国 平潭", "lat": 25.50, "lon": 119.79, "note": "海峡北口"},
    {"name": "中国 澎湖列岛", "lat": 23.57, "lon": 119.58, "note": "海峡中段"},
    {"name": "中国 东海", "lat": 28.20, "lon": 123.50, "note": "北部进出通道"},
    {"name": "中国 巴士海峡", "lat": 21.20, "lon": 121.00, "note": "南部进出通道"},
    {"name": "中国 三亚", "lat": 18.24, "lon": 109.50, "note": "南部支点"},
    {"name": "冲绳岛", "lat": 26.40, "lon": 127.90, "note": "东北方向"},
    {"name": "美国 关岛", "lat": 13.44, "lon": 144.79, "note": "西太平洋支点"},
]


# --------------------------------------------------------------------------
# Target definitions
# --------------------------------------------------------------------------


@dataclass
class Target:
    """One tracked platform in the demonstration situation."""

    target_id: str
    name: str
    platform_class: str          # 平台类别（真实型号口径）
    domain: str                  # sea / air
    affiliation: str             # red / blue
    waypoints: List[geo.Coord]
    speed_kn: float
    altitude_m: float = 0.0
    role: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)

    @property
    def speed_mps(self) -> float:
        return geo.knots_to_mps(self.speed_kn)


def _build_targets() -> List[Target]:
    """Define the five platforms of the demonstration situation."""

    # 红方水面编队：自东海经台湾海峡南下。
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

    # 蓝方空中力量：海峡上空巡逻与高空警戒。
    fighter_route = [
        (24.90, 118.60),   # 沿海机场起飞
        (24.70, 119.30),
        (24.40, 119.90),
        (24.10, 120.20),
        (23.70, 120.00),
        (23.40, 119.40),
        (23.30, 118.70),   # 返场
    ]

    awacs_route = [
        (25.40, 121.20),
        (25.10, 120.60),
        (24.80, 120.00),
        (24.60, 119.40),
        (24.70, 118.90),
        (25.00, 119.10),
        (25.30, 119.80),
        (25.40, 121.20),   # 闭环
    ]

    return [
        Target(
            target_id="T01",
            name="红方驱逐舰编队指挥舰",
            platform_class="驱逐舰",
            domain="sea",
            affiliation="red",
            waypoints=destroyer_route,
            speed_kn=18.0,
            role="编队指挥",
            attributes={"舰种": "驱逐舰", "编制": "水面舰艇编队", "任务": "海峡穿越"},
        ),
        Target(
            target_id="T02",
            name="红方护卫舰",
            platform_class="护卫舰",
            domain="sea",
            affiliation="red",
            waypoints=frigate_route,
            speed_kn=18.0,
            role="伴随护航",
            attributes={"舰种": "护卫舰", "编制": "水面舰艇编队", "任务": "伴随护航"},
        ),
        Target(
            target_id="T03",
            name="红方综合补给舰",
            platform_class="补给舰",
            domain="sea",
            affiliation="red",
            waypoints=supply_route,
            speed_kn=16.0,
            role="后勤保障",
            attributes={"舰种": "综合补给舰", "编制": "水面舰艇编队", "任务": "编队补给"},
        ),
        Target(
            target_id="B01",
            name="蓝方战斗机",
            platform_class="战斗机",
            domain="air",
            affiliation="blue",
            waypoints=fighter_route,
            speed_kn=450.0,
            altitude_m=8000.0,
            role="空中巡逻",
            attributes={"机型": "战斗机", "任务": "战斗空中巡逻", "高度层": "中空"},
        ),
        Target(
            target_id="B02",
            name="蓝方预警机",
            platform_class="预警机",
            domain="air",
            affiliation="blue",
            waypoints=awacs_route,
            speed_kn=300.0,
            altitude_m=10000.0,
            role="空中警戒",
            attributes={"机型": "预警机", "任务": "空中预警指挥", "高度层": "高空"},
        ),
    ]


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
    targets = targets if targets is not None else _build_targets()

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
            capped = min(travelled, total)
            (lat, lon), bearing = geo.sample_polyline(routes[t.target_id], capped)
            state.append(
                {
                    "id": t.target_id,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "alt": round(t.altitude_m, 1),
                    "speed_kn": t.speed_kn,
                    "heading": round(bearing, 1),
                    "travelled_m": round(capped, 1),
                    "reached_end": capped >= total,
                }
            )
        frames.append({"tick": tick, "t_seconds": round(t_seconds, 1), "targets": state})

    return frames


def build_dataset() -> dict:
    """Assemble the full demonstration dataset written to the web data folder."""
    targets = _build_targets()
    frames = build_frames(targets)

    return {
        "schema": "situation3d.trajectory.v1",
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
        "start_time_utc": "2026-09-15T00:00:00Z",
        "hotspots": HOTSPOTS,
        "targets": [
            {
                "id": t.target_id,
                "name": t.name,
                "platform_class": t.platform_class,
                "domain": t.domain,
                "affiliation": t.affiliation,
                "role": t.role,
                "speed_kn": t.speed_kn,
                "altitude_m": t.altitude_m,
                "attributes": t.attributes,
                "waypoints": [[round(a, 6), round(b, 6)] for a, b in t.waypoints],
            }
            for t in targets
        ],
        "frames": frames,
    }
