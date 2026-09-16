"""Demo case registry for situation3d.

The platform exposes a menu that loads a demonstration case. A case selects a
subset of the order of battle, the hotspot regions that are relevant to it, and
the camera framing to open with. Everything else -- the geography, the tick
rate, the inference rule set -- is shared, so cases stay comparable.

Adding a case is a one-line change here; ``scripts/gen_demo_trajectory.py``
picks the registry up automatically and writes one dataset per case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# The order of battle is defined in scenario.py; these ids must stay in sync.
# scenario.build_dataset() raises if a case references an unknown target.


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    name: str
    description: str
    target_ids: List[str]
    hotspot_names: List[str]
    camera: Dict[str, float] = field(default_factory=dict)


DEFAULT_CAMERA = {
    "lon": 119.60,
    "lat": 24.40,
    "height": 900000.0,
    "heading": 0.0,
    "pitch": -75.0,
}

CASES: List[DemoCase] = [
    DemoCase(
        case_id="joint_full",
        name="联合态势演示（海陆空全要素）",
        description=(
            "红方水面编队自东海经台湾海峡南下，岸舰导弹营与远程防空营沿海一线展开；"
            "蓝方以护卫舰、巡逻舰实施海上拦截，以战斗机与预警机建立空中巡逻与预警。"
            "海、陆、空三个域与红蓝双方全部覆盖，用于展示完整的三级态势推理链路。"
        ),
        target_ids=["R01", "R02", "R03", "R04", "R05", "R06", "R07",
                    "B01", "B02", "B03", "B04", "B05"],
        hotspot_names=[
            "中国 台湾海峡", "中国 平潭", "中国 澎湖列岛", "中国 东海",
            "中国 巴士海峡", "中国 三亚", "中国台湾省 台北", "中国台湾省 高雄",
            "日本 冲绳岛", "美国 关岛",
        ],
        camera=dict(DEFAULT_CAMERA),
    ),
    DemoCase(
        case_id="sea_convoy",
        name="海上编队穿越与护航",
        description=(
            "聚焦水面兵力：红方驱逐舰、护卫舰、补给舰组成编队穿越海峡，"
            "蓝方护卫舰与巡逻舰实施海上拦截。用于验证编组级态势推理与编队形态判定。"
        ),
        target_ids=["R01", "R02", "R03", "R07", "B01", "B02", "B05"],
        hotspot_names=[
            "中国 台湾海峡", "中国 平潭", "中国 澎湖列岛", "中国 东海", "中国 巴士海峡",
        ],
        camera={"lon": 119.80, "lat": 24.30, "height": 700000.0,
                "heading": 0.0, "pitch": -72.0},
    ),
    DemoCase(
        case_id="air_control",
        name="空中巡逻与预警指挥",
        description=(
            "聚焦空中兵力：红蓝双方战斗机在中高空建立战斗空中巡逻，"
            "双方预警机在后方实施空中预警与指挥。用于验证空中域的目标级威胁推理与军标渲染。"
        ),
        target_ids=["R01", "R06", "R07", "B01", "B04", "B05"],
        hotspot_names=[
            "中国 台湾海峡", "中国 东海", "中国台湾省 台北", "中国台湾省 高雄",
        ],
        camera={"lon": 120.40, "lat": 24.60, "height": 800000.0,
                "heading": 0.0, "pitch": -70.0},
    ),
    DemoCase(
        case_id="coastal_defense",
        name="沿海要地防空反导",
        description=(
            "聚焦陆基与空中对抗：红方岸舰导弹营、远程防空营沿海一线机动展开，"
            "蓝方要地防空连在北部要地组织防空，双方战斗机与预警机构成空中掩护与预警。"
            "用于验证陆域目标与防空反导场景下的区域级态势推理。"
        ),
        target_ids=["R04", "R05", "R06", "R07", "B03", "B04", "B05"],
        hotspot_names=[
            "中国 台湾海峡", "中国 平潭", "中国 澎湖列岛",
            "中国台湾省 台北", "中国台湾省 高雄",
        ],
        camera={"lon": 120.10, "lat": 24.70, "height": 650000.0,
                "heading": 0.0, "pitch": -68.0},
    ),
]

DEFAULT_CASE_ID = "joint_full"

CASES_BY_ID: Dict[str, DemoCase] = {c.case_id: c for c in CASES}


def get_case(case_id: str) -> DemoCase:
    try:
        return CASES_BY_ID[case_id]
    except KeyError:
        raise KeyError("unknown demo case: %s (known: %s)"
                       % (case_id, ", ".join(sorted(CASES_BY_ID)))) from None


def index() -> dict:
    """Compact menu payload written alongside the datasets."""
    return {
        "default": DEFAULT_CASE_ID,
        "cases": [
            {
                "id": c.case_id,
                "name": c.name,
                "description": c.description,
                "target_count": len(c.target_ids),
                "hotspot_count": len(c.hotspot_names),
                "file": "data/%s.json" % c.case_id,
            }
            for c in CASES
        ],
    }
