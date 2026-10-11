#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略鲁棒性 · 跨域泛化 · 抗扰容错阈值（`I17` 步骤 4；31007 6.2.4）。

## 标准依据

> 31007 6.2.4 系统效能：**认知对齐度**、**跨域泛化**、**抗扰容错阈值**、决策可解释性、决策安全约束。
> 31007 6.2.3 决策质量：战术合理性、**策略鲁棒性**。

**标准点名了这三项，但未给测量口径。** 故本模块的做法**全部自定且显式声明**。

## 三项的口径（**本项目自定**）

| 指标 | 问的问题 | 口径 |
| --- | --- | --- |
| **抗扰容错阈值** | 扰动加到多大，性能才跌破可接受线？ | **二分搜索**最小的扰动强度 `d*`，使 `perf(d*) < 可接受线` |
| **跨域泛化** | 换到未见过的域，性能掉多少？ | 未见域性能均值与**训练域性能均值**之差（归一化到 `[-1,1]` 相似度意义） |
| **策略鲁棒性** | 同一策略在参数扰动下的表现稳不稳？ | 扰动下性能的**变异系数**的补（`1 - CV`，截断到 `[0,1]`）与**最差情形下界** |

## ⚠ 边界（必须写明）

1. **可接受线、扰动尺度、域划分都是输入参数**，本模块**不预设**任何"合格值"；
2. **找不到越界点时必须如实返回 `None`** —— 不能把"没探到"说成"阈值很高"；
3. 三项都是**对已评测结果的聚合**，本模块**不跑推演**。
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_ACCEPT = 0.5      # "可接受线"（本模块默认；调用方应显式给出）


# ══════════════════════════════════════════════════════
# ① 抗扰容错阈值
# ══════════════════════════════════════════════════════

def disturbance_threshold(perf_at: Callable[[float], Optional[float]],
                          *, lo: float = 0.0, hi: float = 1.0,
                          accept: float = DEFAULT_ACCEPT,
                          tol: float = 1e-3, max_iter: int = 40) -> Dict[str, Any]:
    """二分搜索**最小的扰动强度** `d*`，使性能首次跌破 `accept`。

    假设 `perf_at` 在 `[lo, hi]` 上**单调不增**（扰动越大性能越差）——
    **若该假设不成立，二分不适用，本函数会检测并如实报出。**

    Returns
    -------
    dict
        `threshold`（`d*` 或 `None`）· `perf_lo` / `perf_hi` · `monotonic_ok` ·
        `evaluations`（调用次数，可核验 O(log)）· `reason`（取不到时的理由）。
    """
    out: Dict[str, Any] = {
        "threshold": None, "accept": float(accept),
        "perf_lo": None, "perf_hi": None,
        "monotonic_ok": None, "evaluations": 0,
        "method": "bisection on monotonically non-increasing perf(d)",
    }
    if not (hi > lo):
        out["reason"] = "区间非法（hi <= lo）"
        return out

    evals = 0

    def probe(d: float) -> Optional[float]:
        nonlocal evals
        evals += 1
        v = perf_at(d)
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    p_lo, p_hi = probe(lo), probe(hi)
    out["perf_lo"], out["perf_hi"] = p_lo, p_hi
    out["evaluations"] = evals

    if p_lo is None or p_hi is None:
        out["reason"] = "端点处取不到有效性能值"
        return out
    if p_lo < accept:
        out["threshold"] = float(lo)
        out["reason"] = "起点即已跌破可接受线"
        out["monotonic_ok"] = True
        return out
    if p_hi >= accept:
        # **关键**：不能把"没探到"说成"阈值很高"
        out["reason"] = ("在上界 hi=%.6g 仍未跌破 accept=%.6g —— "
                         "**阈值高于搜索上界，本轮未测得**" % (hi, accept))
        out["threshold"] = None
        return out

    # 单调性抽查（用中点做一次独立检查）
    mid = (lo + hi) / 2.0
    p_mid = probe(mid)
    out["evaluations"] = evals
    if p_mid is not None:
        mono = (p_lo + tol >= p_mid >= p_hi - tol)
        out["monotonic_ok"] = bool(mono)
        if not mono:
            out["reason"] = ("中点性能 %.6g 不在端点之间（[%.6g, %.6g]）—— "
                             "**单调性假设不成立，二分结果不可信**" % (p_mid, p_hi, p_lo))
            return out
    else:
        out["monotonic_ok"] = None

    a, b = lo, hi
    while (b - a) > tol and evals < max_iter:
        m = (a + b) / 2.0
        p = probe(m)
        if p is None:
            out["reason"] = "搜索过程中取不到有效性能值（d=%.6g）" % m
            out["evaluations"] = evals
            return out
        if p < accept:
            b = m
        else:
            a = m
    out["evaluations"] = evals
    out["threshold"] = (a + b) / 2.0
    return out


# ══════════════════════════════════════════════════════
# ② 跨域泛化
# ══════════════════════════════════════════════════════

def cross_domain_generalisation(perf_by_domain: Dict[str, Sequence[float]],
                                *, seen_domains: Sequence[str],
                                unseen_domains: Sequence[str]) -> Dict[str, Any]:
    """**跨域泛化**：未见域性能相对**已见域**性能的保持程度。

    口径：`generalisation = mean(unseen) / mean(seen)`，截断到 `[0, 1]`。
    **> 1 的情况会如实报出**（未见域反而更好），但不再放大到 1 以上作为"泛化强"的证据。

    **缺数据时返回 `None` + 理由**，不编造。
    """
    seen = [d for d in seen_domains if d in perf_by_domain]
    unseen = [d for d in unseen_domains if d in perf_by_domain]
    out: Dict[str, Any] = {
        "generalisation": None,
        "seen_domains_used": seen, "unseen_domains_used": unseen,
        "missing_seen": [d for d in seen_domains if d not in perf_by_domain],
        "missing_unseen": [d for d in unseen_domains if d not in perf_by_domain],
        "mean_seen": None, "mean_unseen": None,
        "method": "generalisation = mean(unseen) / mean(seen)",
    }

    def _mean(ds: Sequence[str]) -> Optional[float]:
        vals: List[float] = []
        for d in ds:
            for v in perf_by_domain.get(d) or []:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f):
                    vals.append(f)
        return sum(vals) / len(vals) if vals else None

    ms, mu = _mean(seen), _mean(unseen)
    out["mean_seen"], out["mean_unseen"] = ms, mu
    if ms is None or mu is None:
        out["reason"] = "已见域或未见域无有效性能值"
        return out
    if ms == 0:
        out["reason"] = "已见域均值为 0，无法归一化"
        return out
    raw = mu / ms
    out["raw_ratio"] = raw
    out["generalisation"] = min(1.0, max(0.0, raw))
    if raw > 1.0:
        out["note"] = ("未见域表现**优于**已见域（raw_ratio=%.4f）—— "
                       "已截断为 1.0；这**不构成「泛化能力强」的证据**，"
                       "更可能是域间难度不同" % raw)
    return out


# ══════════════════════════════════════════════════════
# ③ 策略鲁棒性
# ══════════════════════════════════════════════════════

def policy_robustness(perf_under_perturbation: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    """**策略鲁棒性**：同一策略在参数扰动下的稳定性。

    口径（两个都要给，**不合并成一个数**）：

    · `stability = 1 - CV`，截断 `[0,1]`；`CV = stdev/|mean|`（变异系数）
    · `worst_case` = 所有扰动下性能的**最小值**（下界）

    **只给 stability 会掩盖"均值高但最差情形很糟"**，故两者并列。
    """
    all_vals: List[float] = []
    per_pert: Dict[str, Dict[str, Any]] = {}
    for k, vs in (perf_under_perturbation or {}).items():
        vals = []
        for v in vs or []:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                vals.append(f)
        per_pert[k] = {"n": len(vals),
                       "mean": (sum(vals) / len(vals)) if vals else None}
        all_vals.extend(vals)

    out: Dict[str, Any] = {
        "stability": None, "worst_case": None, "mean": None, "stdev": None,
        "cv": None, "n": len(all_vals), "per_perturbation": per_pert,
        "method": "stability = 1 - CV (C.V. = stdev/|mean|), clipped to [0,1]",
        "note": "stability 与 worst_case **并列报告** —— 只看 stability 会掩盖最差情形",
    }
    if len(all_vals) < 2:
        out["reason"] = "有效样本少于 2，无法估计离散度"
        if all_vals:
            out["mean"] = all_vals[0]
            out["worst_case"] = all_vals[0]
        return out

    n = len(all_vals)
    mean = sum(all_vals) / n
    var = sum((v - mean) ** 2 for v in all_vals) / (n - 1)
    sd = math.sqrt(var)
    out.update(mean=mean, stdev=sd, worst_case=min(all_vals))
    if mean == 0:
        out["reason"] = "均值为 0，变异系数无定义"
        return out
    cv = sd / abs(mean)
    out["cv"] = cv
    out["stability"] = min(1.0, max(0.0, 1.0 - cv))
    return out


def summarize_three(perf_at: Optional[Callable[[float], Optional[float]]] = None,
                    *,
                    perf_by_domain: Optional[Dict[str, Sequence[float]]] = None,
                    seen_domains: Sequence[str] = (),
                    unseen_domains: Sequence[str] = (),
                    perf_under_perturbation: Optional[Dict[str, Sequence[float]]] = None,
                    accept: float = DEFAULT_ACCEPT,
                    tol: float = 1e-3) -> Dict[str, Any]:
    """把三项合成一份摘要，供评估报告 L3 层使用（对应 `31007 6.2.4`）。"""
    out: Dict[str, Any] = {"accept_line": accept}
    out["disturbance_threshold"] = (
        disturbance_threshold(perf_at, accept=accept, tol=tol) if perf_at is not None
        else {"threshold": None, "reason": "未提供性能函数"})
    out["cross_domain_generalisation"] = (
        cross_domain_generalisation(perf_by_domain or {}, seen_domains=seen_domains,
                                    unseen_domains=unseen_domains)
        if perf_by_domain else {"generalisation": None, "reason": "未提供分域性能"})
    out["policy_robustness"] = (
        policy_robustness(perf_under_perturbation or {})
        if perf_under_perturbation else {"stability": None, "reason": "未提供扰动性能"})
    return out
