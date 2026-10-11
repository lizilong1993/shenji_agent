#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""专家评分聚合与**评分者间一致性**（`P3` 步骤 4 / `I17`；31007 6.2.3）。

## 标准依据

> 31007 6.2.3 战术合理性（**专家 0-10 分** + **规则冲突检测**）、策略鲁棒性。

标准给了**量表范围（0-10）**，**未给**：维度划分、维度权重、评分者人数、
一致性阈值。**故本模块把自定口径全部显式写出**，并在输出里回带。

## 本模块做什么

| 能力 | 说明 |
| --- | --- |
| **评分表结构** | 把 0-10 总分拆成**维度分**（见 `DIMENSIONS`），每维 0-10 |
| **加权总分** | 维度分 → 总分（权重显式声明；`None` 维度**不计入分母**） |
| **评分者间一致性** | **ICC(2,1)**（双向随机、绝对一致）—— 适用于"同一批样本被多名评分者打分" |
| **规则冲突检测** | 动作序列里**互相矛盾**的动作对（如"占领 A"与"撤离 A"同时出现） |

## ★ 为什么用 ICC 而不是 Cohen's κ

`κ` 适用于**分类**评分；本量表是 **0-10 有序/连续**。
**对有序数据用 κ 会先做分箱，丢掉信息**，且阈值解释也随之改变。

**ICC(2,1) 直接对连续评分算一致性**，是这类量表的标准做法。

## ⚠ 边界（必须写明）

1. **权重与维度是本模块自定**，标准未规定 —— 报告须注明。
2. **ICC 的阈值解释**（如 0.75 以上为良好）**是惯例而非标准规定** —— 故本模块
   **只回带数值与所用公式**，**不给"合格/不合格"判定**，把判定留给报告作者。
3. **缺分不补**：某评分者漏打某维时，该维**退出该样本的聚合**，并**计数报出** ——
   用均值填补会**人为抬高一致性**。
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 0-10 量表下的维度（本模块自定；标准未规定）
DIMENSIONS: Dict[str, str] = {
    "mission_fit": "任务契合（动作是否服务于当前任务目标）",
    "timing": "时机（是否在恰当的态势窗口内行动）",
    "resource_use": "资源运用（弹药/油料/兵力消耗是否合理）",
    "risk_control": "风险控制（是否避免不必要的暴露与损失）",
    "legality": "规则合规（是否违反交战规则/约束）",
}
# 权重（本模块自定；`legality` 给较高权重，因其为**否决性**维度）
DIMENSION_WEIGHTS: Dict[str, float] = {
    "mission_fit": 0.30,
    "timing": 0.20,
    "resource_use": 0.15,
    "risk_control": 0.15,
    "legality": 0.20,
}
SCALE_MIN, SCALE_MAX = 0, 10


def validate_score(v: Any) -> Tuple[Optional[float], Optional[str]]:
    """把输入规范化成 `[0,10]` 内的浮点；非法时返回 `(None, 理由)`。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None, "非数值"
    if not math.isfinite(f):
        return None, "非有限值"
    if f < SCALE_MIN or f > SCALE_MAX:
        return None, "超出 0-10 量程"
    return f, None


def weighted_total(scores: Dict[str, Any],
                   *, weights: Optional[Dict[str, float]] = None
                   ) -> Dict[str, Any]:
    """把维度分聚成加权总分。

    **缺失维度退出分母**（按可得分量归一化）—— 否则漏打一维会把总分压低，
    等于把"没评"当成"评了低分"。
    """
    w = dict(DIMENSION_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in DIMENSIONS})

    usable, invalid = {}, []
    for k in DIMENSIONS:
        if k not in scores:
            continue
        v, why = validate_score(scores[k])
        if v is None:
            invalid.append({"dimension": k, "value": scores[k], "reason": why})
        else:
            usable[k] = v

    out: Dict[str, Any] = {
        "total": None,
        "dimensions_used": sorted(usable),
        "dimensions_missing": sorted(set(DIMENSIONS) - set(scores)),
        "dimensions_invalid": invalid,
        "weights_declared": dict(w),
        "weights_used": {},
        "note": "维度、权重与量程为本模块自定（31007 6.2.3 只规定 0-10 分），报告须注明",
    }
    if not usable:
        out["reason"] = "没有任何合法维度分"
        return out
    denom = sum(w[k] for k in usable)
    used = {k: w[k] / denom for k in usable}
    out["weights_used"] = used
    out["total"] = sum(used[k] * v for k, v in usable.items())
    if out["dimensions_missing"] or invalid:
        out["reason"] = "部分维度缺失或非法，已按可得分量归一化"
    return out


def icc_2_1(ratings: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """**ICC(2,1)**：双向随机效应、**绝对一致性**、单次测量。

    `ratings` 形状：**每个目标一行、每个评分者一列**（行内不得有缺 —— 缺分请先剔除该行）。

    返回 `icc` / `n_targets` / `n_raters` / 各均方（`MSR`/`MSC`/`MSE`）与所用公式。
    **样本不足时返回 `icc=None` + 理由，不编造数值。**
    """
    rows = [list(r) for r in ratings]
    n = len(rows)
    if n == 0:
        return {"icc": None, "reason": "无目标", "n_targets": 0, "n_raters": 0}
    k = len(rows[0])
    if k < 2:
        return {"icc": None, "reason": "评分者少于 2 人，无法计算一致性",
                "n_targets": n, "n_raters": k}
    if any(len(r) != k for r in rows):
        return {"icc": None, "reason": "行长度不一致（存在缺分）—— 请先剔除含缺分的目标",
                "n_targets": n, "n_raters": k}
    if n < 2:
        return {"icc": None, "reason": "目标少于 2 个，无法估计目标间方差",
                "n_targets": n, "n_raters": k}

    flat = [x for r in rows for x in r]
    grand = sum(flat) / len(flat)
    row_means = [sum(r) / k for r in rows]
    col_means = [sum(rows[i][j] for i in range(n)) / n for j in range(k)]

    ss_total = sum((x - grand) ** 2 for x in flat)
    ss_row = k * sum((m - grand) ** 2 for m in row_means)
    ss_col = n * sum((m - grand) ** 2 for m in col_means)
    ss_err = ss_total - ss_row - ss_col

    df_row, df_col, df_err = n - 1, k - 1, (n - 1) * (k - 1)
    if df_err <= 0:
        return {"icc": None, "reason": "自由度不足", "n_targets": n, "n_raters": k}

    msr = ss_row / df_row
    msc = ss_col / df_col
    mse = ss_err / df_err

    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    out: Dict[str, Any] = {
        "icc": None, "n_targets": n, "n_raters": k,
        "MSR": msr, "MSC": msc, "MSE": mse,
        "formula": "ICC(2,1) = (MSR - MSE) / (MSR + (k-1)MSE + k(MSC - MSE)/n)",
        "note": "阈值解释（如 0.75 良好）属惯例而非标准规定 —— 本模块只给数值",
    }
    if abs(denom) < 1e-15:
        out["reason"] = "分母为零（无方差）"
        return out
    out["icc"] = (msr - mse) / denom
    return out


def rule_conflicts(actions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """**规则冲突检测**：找出同一主体上**互相矛盾**的动作对。

    当前规则集（可扩展；**规则本身是本模块自定**）：

    | 规则 | 冲突对 |
    | --- | --- |
    | `occupy_vs_withdraw` | 同一 `actor` 在同一 `target` 上同时"占领"与"撤离" |
    | `attack_vs_hold_fire` | 同一 `actor` 对同一 `target` 同时"打击"与"停火" |
    | `engage_vs_disengage` | 同一 `actor` 同时"交战"与"脱离" |

    **返回每个冲突的双方索引与规则名**，便于人工复核（**不自动判定谁错**）。
    """
    RULES = (
        ("occupy_vs_withdraw", "occupy", "withdraw"),
        ("attack_vs_hold_fire", "attack", "hold_fire"),
        ("engage_vs_disengage", "engage", "disengage"),
    )
    acts = list(actions)
    out: List[Dict[str, Any]] = []
    for name, a_kind, b_kind in RULES:
        idx_a = [i for i, x in enumerate(acts) if str(x.get("kind")) == a_kind]
        idx_b = [i for i, x in enumerate(acts) if str(x.get("kind")) == b_kind]
        for i, j in itertools.product(idx_a, idx_b):
            ai, bj = acts[i], acts[j]
            if ai.get("actor") != bj.get("actor"):
                continue
            if ai.get("target") != bj.get("target"):
                continue
            out.append({
                "rule": name,
                "index_a": i, "index_b": j,
                "actor": ai.get("actor"), "target": ai.get("target"),
                "kinds": (a_kind, b_kind),
            })
    return out


def aggregate_expert_scores(ratings_by_rater: Dict[str, Dict[str, Dict[str, Any]]]
                            ) -> Dict[str, Any]:
    """多名评分者对多个目标的评分 → **每目标总分 + 总体 ICC**。

    输入形状：`{评分者: {目标: {维度: 分}}}`

    返回 `per_target`（每目标的总分均值/极差）、`icc`（在**总分**上算）、
    `excluded`（因缺分/非法而无法进入 ICC 的目标及其理由）。
    """
    raters = list(ratings_by_rater)
    targets: List[str] = []
    for r in raters:
        for t in ratings_by_rater[r]:
            if t not in targets:
                targets.append(t)

    per_target: Dict[str, Any] = {}
    matrix_rows, matrix_targets, excluded = [], [], []
    for t in targets:
        totals = {}
        for r in raters:
            sc = ratings_by_rater[r].get(t)
            if sc is None:
                continue
            wt = weighted_total(sc)
            if wt["total"] is None:
                continue
            totals[r] = wt["total"]
        if not totals:
            per_target[t] = {"total_mean": None, "reason": "无有效评分"}
            excluded.append({"target": t, "reason": "无有效评分"})
            continue
        vals = list(totals.values())
        per_target[t] = {
            "total_mean": sum(vals) / len(vals),
            "total_min": min(vals),
            "total_max": max(vals),
            "spread": max(vals) - min(vals),
            "raters": sorted(totals),
        }
        if len(totals) == len(raters) and len(raters) >= 2:
            matrix_rows.append([totals[r] for r in raters])
            matrix_targets.append(t)
        else:
            excluded.append({"target": t,
                             "reason": "评分者不全（%d/%d）" % (len(totals), len(raters))})

    out: Dict[str, Any] = {
        "per_target": per_target,
        "raters": raters,
        "n_targets": len(targets),
        "icc": None,
        "excluded_from_icc": excluded,
        "note": "缺分目标不进入 ICC（用均值填补会人为抬高一致性）",
    }
    if len(matrix_rows) >= 2 and len(raters) >= 2:
        out["icc"] = icc_2_1(matrix_rows)
        out["icc_targets"] = matrix_targets
    else:
        out["icc_reason"] = "可用目标或评分者不足，无法计算 ICC"
    return out
