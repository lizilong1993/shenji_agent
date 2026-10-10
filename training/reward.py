"""奖励函数：任务对齐 + 时间折扣 + 不当行为惩罚（I06 步骤 4）。

标准依据
--------
`31007 5.3.3 奖励函数设计指导`：任务目标对齐、及时性与长期性平衡、
惩罚不当行为、数值尺度与平滑性、可解释性与可调试性。

为什么单独一个模块而不是就地改 `selfplay._terminal_objective_score()`
--------------------------------------------------------------------
原函数有**两个必须保留**的性质：
  1. `_run_single()` 只在**终局**调用它一次（`selfplay.py:673`），返回一个标量；
  2. 它同时充当 `objective_mode=heuristic_policy_score` 的**回退实现**
     （【风险与回滚】明确要求「保留原启发式打分作为 fallback」）。

若就地把"每步奖励"塞进去，会同时破坏这两点。故本模块提供**每步**的塑形奖励，
`_terminal_objective_score()` 一行不改 —— 回退路径与既有标签产物都不受影响。

一个实测出来的动机（修复前基线）
--------------------------------
`experiments/shenji_agent/objective_labels/latest_after_auth/summary.json` 实测：
**2344 条标签里 2266 条 reward 为 0**（`zero_reward_count=2266`）。
`after_smoke_20260429_032914` 更极端：**13440 条全部为 0**。
原因是奖励只在终局给一次、且大量对局终局分数相同 ——
**稀疏到无法训练**。本模块的塑形项把「朝目标推进」变成**每步可见**的信号。

可解释性与可调试性（标准明确要求）
----------------------------------
`shape_step_reward()` 返回 `(total, components)`，`components` 里逐项列出
每个分量的原始值与权重后的贡献。**不允许只返回一个数** ——
否则"奖励变差了"无法定位到是折扣、惩罚还是对齐项的问题。
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

# ── 默认权重（集中一处，便于消融与审计）──────────────────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    # ① 任务对齐：朝任务目标推进（占位、夺控、歼灭）
    "align_city_value": 2.0,      # 每点「受控城市价值」增量的价值
    "align_score": 1.0,           # 每点官方分数的价值
    "align_kill": 5.0,            # 每次歼敌
    "align_damage": 0.25,         # 每点对敌伤害
    # ② 不当行为惩罚（标准要求「惩罚不当行为」）
    # ⚠ `pen_friendly_loss` **必须严格大于** `align_kill`：
    #    若两者相等，函数对「损失一个己方单位」与「歼敌一个」**完全无差别**，
    #    策略会学成"拿命换命"。这是我自己写测试时发现的（原为各 5.0），
    #    属标准说的"惩罚不当行为"，不是调参偏好。
    "pen_friendly_loss": 8.0,     # 每次己方损失（> align_kill，避免拿命换命）
    "pen_damage_taken": 0.25,
    "pen_idle": 0.5,              # 该动未动（有一手棋可走却原地不动）
    # ③ 时间折扣
    "gamma": 0.99,                # 每步折扣因子
    # ④ 尺度与平滑
    "clip": 50.0,                 # 单步奖励绝对值上限（数值尺度）
}

# 判定「原地不动」的动作类型集合（与 `ai/action_safety.py` 的字典同源）
_IDLE_ACTION_TYPES = frozenset({5, 6, 7, 10, 11, 12, 13, 14, 17, 22, 23, 24, 333})


def time_discount(gamma: float, steps_remaining: int) -> float:
    """时间折扣因子 `gamma ** steps_remaining`。

    「及时性与长期性平衡」（标准要求）：越晚的收益折得越多，
    但 `gamma` 接近 1 时不至于把长期目标抹掉。
    `steps_remaining<=0` 时返回 1.0（终局步不打折，避免把最终胜负也算小）。
    """
    if steps_remaining is None or steps_remaining <= 0:
        return 1.0
    return float(gamma) ** int(steps_remaining)


def _f(value: Any, default: float = 0.0) -> float:
    """容错取浮点：**畸形容器不得让奖励函数抛异常**（奖励算不出来会让训练直接崩）。"""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def shape_step_reward(progress: Optional[Mapping[str, Any]],
                      *,
                      steps_remaining: int = 0,
                      weights: Optional[Mapping[str, float]] = None,
                      mission_steps: Optional[Mapping[str, Any]] = None,
                      action_outcomes: Optional[Any] = None,
                      ) -> Tuple[float, Dict[str, Any]]:
    """对**单步**计算塑形奖励。返回 `(total, components)`。

    Parameters
    ----------
    progress:
        `selfplay._build_step_objective_records()` 产出的 progress 记录里
        **本方那一侧**的字典（含 `score_delta` / `city_value_delta` /
        `damage_dealt` / `damage_taken` / `kill_dealt` / `kill_taken`）。
        传 `None` 或非 dict 时按"无进展"处理，**不抛异常**。
    steps_remaining:
        距终局剩余步数，用于时间折扣。
    weights:
        覆盖默认权重（消融实验就靠这个参数，不需要改代码）。
    mission_steps:
        **任务对齐**的显式目标。若给了 `{"kind": "capture", "target_hex": H}`，
        则「推进到目标格」额外加分；给 `None` 时不启用该附加项。
    action_outcomes:
        本步动作结果列表；用于「不当行为」中的**无效动作**判定
        （`error` 非空 = 该动作被引擎拒绝）。

    Returns
    -------
    (total, components)
        `total` 已做尺度裁剪；`components` 逐项给出 `raw` / `weight` / `value`。
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = _f(v, w[k])

    comp: Dict[str, Any] = {}
    gains = 0.0
    losses = 0.0

    def add(name: str, raw: float, weight: float) -> float:
        """累加一个分量。**正负分开记**，不混进一个 total。

        ⚠ 为什么必须分开：我第一版用 `losses = total - gains if total > 0 else total`，
        结果当收益占优时负数被**整个丢掉** —— 实测某步 `gains=45, losses=0`，
        而惩罚项合计 −7.5 凭空消失。**惩罚形同虚设**，正是标准要防的
        "奖励塑形引入偏差"。故正负各记一路。
        """
        nonlocal gains, losses
        val = raw * weight
        comp[name] = {"raw": raw, "weight": weight, "value": val}
        if val >= 0:
            gains += val
        else:
            losses += val
        return val

    p = progress if isinstance(progress, Mapping) else {}

    # ① 任务对齐
    add("align_city_value", _f(p.get("city_value_delta")), w["align_city_value"])
    add("align_score", _f(p.get("score_delta")), w["align_score"])
    add("align_kill", _f(p.get("kill_dealt")), w["align_kill"])
    add("align_damage", _f(p.get("damage_dealt")), w["align_damage"])

    # 任务目标的显式对齐（可选）：抵达目标格给一次性正奖励
    mission_bonus = 0.0
    if isinstance(mission_steps, Mapping):
        kind = str(mission_steps.get("kind") or "")
        reached = bool(mission_steps.get("reached"))
        if kind and reached:
            mission_bonus = _f(mission_steps.get("bonus"), 1.0)
    if mission_steps is not None:
        comp["align_mission"] = {"raw": mission_bonus, "weight": 1.0, "value": mission_bonus}
        if mission_bonus >= 0:
            gains += mission_bonus
        else:
            losses += mission_bonus

    # ② 不当行为惩罚（以**负值**计入）
    add("pen_friendly_loss", -abs(_f(p.get("kill_taken"))), w["pen_friendly_loss"])
    add("pen_damage_taken", -abs(_f(p.get("damage_taken"))), w["pen_damage_taken"])

    # 无效动作：被引擎拒绝的动作是「不当行为」的一种（浪费回合）
    invalid = 0
    if isinstance(action_outcomes, (list, tuple)):
        for o in action_outcomes:
            if isinstance(o, Mapping) and o.get("error"):
                invalid += 1
    add("pen_invalid_action", -float(invalid), w.get("pen_idle", 0.0))

    # ③ 时间折扣：只对**收益**打折，**惩罚全额计入** ——
    #    否则早期的不当行为会被 gamma^T 抹平，等于纵容（标准要求"惩罚不当行为"）。
    disc = time_discount(w["gamma"], steps_remaining)
    shaped = gains * disc + losses
    comp["gains_losses"] = {"gains": gains, "losses": losses,
                            "net_shaped": shaped, "note": "惩罚不打折，收益按 gamma^t 折"}
    comp["time_discount"] = {"gamma": w["gamma"], "steps_remaining": int(steps_remaining or 0),
                             "factor": disc}

    # ④ 尺度与平滑：裁剪到 [-clip, clip]
    clip = abs(_f(w.get("clip"), 50.0)) or 50.0
    final = max(-clip, min(clip, shaped))
    comp["scale_clip"] = {"before": shaped, "clip": clip, "after": final, "clipped": shaped != final}
    return float(final), comp


def shaped_trajectory_reward(progress_records: Any, *,
                             weights: Optional[Mapping[str, float]] = None,
                             gamma: Optional[float] = None) -> Tuple[float, list]:
    """对一整条轨迹逐步塑形并求和。返回 `(total, per_step)`。

    **这一步才是把"稀疏"变"稠密"的地方** ——
    原实现只在终局给一个标量，故 2344 条标签里 2266 条为 0。
    """
    if not isinstance(progress_records, (list, tuple)):
        return 0.0, []
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: _f(v, w.get(k, 0.0)) for k, v in weights.items() if k in w})
    if gamma is not None:
        w["gamma"] = _f(gamma, w["gamma"])
    n = len(progress_records)
    per_step = []
    total = 0.0
    for i, rec in enumerate(progress_records):
        remaining = n - 1 - i
        r, comp = shape_step_reward(rec, steps_remaining=remaining, weights=w)
        per_step.append({"index": i, "steps_remaining": remaining, "reward": r,
                         "components": comp})
        total += r
    return float(total), per_step


def debug_defaults() -> Dict[str, Any]:
    """取证出口：默认权重与各项含义（供 `--json` 与审计）。"""
    return {
        "weights": dict(DEFAULT_WEIGHTS),
        "idle_action_types": sorted(_IDLE_ACTION_TYPES),
        "terms": ["align_city_value", "align_score", "align_kill", "align_damage",
                  "align_mission", "pen_friendly_loss", "pen_damage_taken",
                  "pen_invalid_action", "time_discount", "scale_clip"],
    }
