"""交战规则（ROE）：把"什么时候可以开火"变成可配置、可审计的判据。

此前决定"打不打"的逻辑散在 ``World._engage`` 与 ``_best_target`` 里，且不可配置：
任何探测到的敌方目标都会被开火，没有责任区、没有确认门槛、没有还击语义。

这里的做法是把政策面**显式化**：给定一份 ROE 手册，判定某次交战是否被允许，
并给出理由。默认值等于现有行为（自由开火、无额外门槛），任何收紧都要由
案例显式打开 —— 这样加这个模块不会悄悄改变已有推演的数值。

三条与公开作战条令一致的判据：

* ``weapons_free``   见到敌人即可打（默认）
* ``weapons_tight``  目标必须在**己方责任区**附近才打
* ``hold_fire``      只有被打了才还击

"确认"用 ``confirm_s`` 表达：目标必须连续被探测到这么久才允许开火，
避免一次瞬时接触就交战。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

POLICIES = ("weapons_free", "weapons_tight", "hold_fire")
DEFAULT_POLICY = "weapons_free"


def _num(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _flag(value: Any, default: bool = True) -> bool:
    """JSON 里的字符串 "false" 不能按 bool("false") 解释。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Rules:
    policy: str = DEFAULT_POLICY
    # 只在 weapons_tight 下生效：目标必须离己方目标点这么近（米）。
    # 0 表示不限制，退化为 weapons_free。
    area_radius_m: float = 0.0
    # 只在 hold_fire 下生效：被打之后多久内允许还击（秒）。
    return_fire_s: float = 300.0
    # 目标必须被连续探测到的最短时间（秒）；0 表示一次接触即可。
    confirm_s: float = 0.0
    # 是否尊重禁止打击清单（NSL）。默认尊重。
    respect_no_strike: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "Rules":
        if not payload:
            return cls()
        policy = str(payload.get("policy") or DEFAULT_POLICY).strip().lower()
        if policy not in POLICIES:
            # 写错政策名比写错数值更容易造成误判，不能静默降级。
            raise ValueError("roe policy must be one of %s" % ", ".join(POLICIES))
        return cls(
            policy=policy,
            area_radius_m=max(0.0, _num(payload.get("area_radius_m"), 0.0)),
            return_fire_s=max(0.0, _num(payload.get("return_fire_s"), 300.0)),
            confirm_s=max(0.0, _num(payload.get("confirm_s"), 0.0)),
            respect_no_strike=_flag(payload.get("respect_no_strike"), True),
        )

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "area_radius_m": self.area_radius_m,
            "return_fire_s": self.return_fire_s,
            "confirm_s": self.confirm_s,
            "respect_no_strike": self.respect_no_strike,
        }


def evaluate(rules: Rules, *, distance_m: float, now_s: float,
             first_seen_s: Optional[float], damaged_at_s: float,
             in_area: bool = True, no_strike: bool = False,
             manual_fire: bool = False) -> Tuple[bool, str]:
    """判定一次交战是否被允许，返回 ``(allowed, reason)``。

    ``manual_fire``（人工下达的开火命令）优先于自动判据 —— 指挥员的显式命令
    不该被自动规则挡下。NSL 是唯一例外：法律/政策层面的禁止，人工命令也
    不能越过。
    """
    if no_strike and rules.respect_no_strike:
        return False, "目标在禁止打击清单上，任何情况下不得交火"

    if manual_fire:
        return True, "人工开火命令"

    if rules.confirm_s > 0:
        if first_seen_s is None:
            return False, "目标尚未被持续探测（缺少首次发现时间）"
        if now_s - first_seen_s < rules.confirm_s:
            return False, ("目标仅被探测 %.0f 秒，未达确认门槛 %.0f 秒"
                           % (now_s - first_seen_s, rules.confirm_s))

    if rules.policy == "hold_fire":
        if damaged_at_s >= 0 and now_s - damaged_at_s <= rules.return_fire_s:
            return True, "遭到攻击，在还击窗口内"
        return False, "hold_fire：未遭攻击，不得主动开火"

    if rules.policy == "weapons_tight":
        if rules.area_radius_m > 0 and not in_area:
            return False, ("weapons_tight：目标在责任区（%.0f km）之外"
                           % (rules.area_radius_m / 1000.0))
        return True, "weapons_tight：目标在责任区内"

    return True, "weapons_free"


def describe(rules: Rules) -> str:
    """一句话说明当前规则，写进运行记录用。"""
    if rules.policy == "weapons_free":
        base = "自由开火"
    elif rules.policy == "weapons_tight":
        base = ("责任区内开火%s" % (
            "（半径 %.0f km）" % (rules.area_radius_m / 1000.0)
            if rules.area_radius_m > 0 else ""))
    else:
        base = "遭攻击后方可还击"
    extras = []
    if rules.confirm_s > 0:
        extras.append("确认门槛 %.0f 秒" % rules.confirm_s)
    if rules.respect_no_strike:
        extras.append("遵守禁止打击清单")
    return base + ("；" + "、".join(extras) if extras else "")
