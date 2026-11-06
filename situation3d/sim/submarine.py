"""潜艇的浮出 / 下潜状态。

此前 ``Entity.depth`` 字段一直存在但从未被读写：所有潜艇在感知、交战里
都按"下潜"处理（对雷达/光学/红外不可见，只能被声呐发现），既不能浮出
换气，也不能在浮出时被常规手段发现。

这里补一个最小状态机，只做三件真实存在的事：

* ``submerged``（默认）：只能被声呐发现，可用鱼雷；
* ``surfaced``：可被雷达/光学发现，但**不能**用鱼雷（鱼雷管在深度下）；
* ``periscope``：中间态——可被雷达发现但探测距离打三折（只露桅杆）。

不做自动决策（什么时候该浮出取决于战术，不是规则能定的），只提供动作与
查询；谁调它由战术层决定。
"""

from __future__ import annotations

from typing import Optional

STATES = ("submerged", "periscope", "surfaced")
DEFAULT_STATE = "submerged"

# 浮出状态下目标的有效探测截面倍数（periscope 只露桅杆，最难发现）。
DETECT_FACTOR = {"submerged": 0.0, "periscope": 0.3, "surfaced": 1.0}


def state_of(entity) -> str:
    value = str(getattr(entity, "depth", DEFAULT_STATE) or DEFAULT_STATE)
    return value if value in STATES else DEFAULT_STATE


def can_surface(entity) -> bool:
    """只有潜艇能改变浮出状态；其它兵种的 depth 保持默认即可。"""
    return getattr(entity, "domain", "") == "subsurface"


def set_state(entity, state: str) -> str:
    state = str(state or "").strip().lower()
    if state not in STATES:
        raise ValueError("submarine state must be one of %s" % ", ".join(STATES))
    if not can_surface(entity):
        raise ValueError("only subsurface units have a dive state")
    entity.depth = state
    return state


def is_detectable_by(entity, sensor_kind: str) -> bool:
    """给定传感器类型，当前状态是否可能被发现。

    与 ``perception.py`` 里的既有规则保持一致：潜艇在 submerged 下只对
    声呐可见；浮出后常规传感器也能看到。
    """
    state = state_of(entity)
    if getattr(entity, "domain", "") != "subsurface":
        return True
    if sensor_kind == "sonar":
        return True
    return state != "submerged"


def detection_factor(entity) -> float:
    """当前状态对探测距离的折减系数。0 表示完全不可见。"""
    if getattr(entity, "domain", "") != "subsurface":
        return 1.0
    return DETECT_FACTOR.get(state_of(entity), 0.0)


def can_use_torpedo(entity) -> bool:
    """鱼雷只能在深度下发射：浮出时鱼雷管在水面以上，无法装填发射。"""
    if getattr(entity, "domain", "") != "subsurface":
        return True
    return state_of(entity) == "submerged"


def label(state: str) -> str:
    return {"submerged": "潜航", "periscope": "潜望镜", "surfaced": "浮出"}.get(
        state, state or "未知")
