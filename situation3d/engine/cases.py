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
    # Which theatre this case belongs to. Selects the road network used to
    # constrain land movement, and (once the front end is theatre-aware) the
    # image/terrain coverage. Defaults to the Taiwan Strait because every case
    # in the registry so far is there.
    theatre: str = "taiwan_strait"


DEFAULT_CAMERA = {
    "lon": 119.60,
    "lat": 24.40,
    "height": 900000.0,
    "heading": 0.0,
    "pitch": -75.0,
}

CASES: List[DemoCase] = [
    DemoCase(
        case_id="india_china_west",
        name="中印边境西段（拉达克）",
        description=(
            "高原边境对峙TS：红方高原合成营沿 Indus 谷机动，无人机实施侦察监视；"
            "蓝方以列城为后勤枢纽沿干线前送，德普桑前哨分队守备 DBO 走廊，"
            "航空兵在高原机场起降巡逻。双方处于脱离接触后的缓冲区状态，"
            "意图在证据不足时应保持为集合而非单一结论——本案例刻意**不收敛**，"
            "用于验证系统在证据有限时如实输出歧义的能力。"
        ),
        target_ids=["H01", "H02", "H03", "H04", "H05"],
        hotspot_names=[
            "加勒万河谷", "班公湖", "德普桑平地", "典角", "楚舒勒", "诺玛", "列城",
        ],
        camera={"lon": 78.60, "lat": 33.90, "height": 420000.0,
                "heading": 0.0, "pitch": -70.0},
        theatre="india_china",
    ),
    DemoCase(
        case_id="taiwan_landing",
        name="登台作战（海陆空全要素）",
        description=(
            "红方两栖编队自大陆沿海集结装载、经台湾海峡航渡，在护航与制空掩护下"
            "于西岸滩头换乘编波、抢滩突击，随后沿路网向内陆推进夺控要点；"
            "蓝方以护卫舰与巡逻舰实施海上拦截，以战斗机与预警机建立空中巡逻与预警，"
            "岸防兵力依托要点组织防御。海、陆、空三域与红蓝双方全部覆盖，"
            "用于展示完整的三级TS推理链路与意图集合的收缩过程。"
        ),
        target_ids=["R01", "R02", "R03", "R04", "R05", "R06", "R07",
                    "R08", "R09", "R10",
                    "B01", "B02", "B03", "B04", "B05", "B06"],
        hotspot_names=[
            "中国 台湾海峡", "中国 平潭", "中国 澎湖列岛", "中国 东海",
            "中国 巴士海峡", "中国 三亚", "中国台湾省 台北", "中国台湾省 高雄",
            "日本 冲绳岛", "美国 关岛", "日本 羽田机场",
        ],
        camera=dict(DEFAULT_CAMERA),
    ),
]

# The Taiwan landing case is the default: its offline tile coverage is baked,
# whereas the India-China theatre still has none (see the note above).
DEFAULT_CASE_ID = "taiwan_landing"

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
