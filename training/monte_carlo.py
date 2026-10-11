#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""蒙特卡洛聚合：把同一用例的 **N 次重复**聚成**均值 + 置信区间**（`I16` 判据 2）。

## 判据原文

> 2) 蒙特卡洛 N≥30 且**输出置信区间**。

## 为什么先写这个（而不等算力）

判据 2 有两半：**跑推演**（要算力）与**把 N 次结果聚成区间**（纯计算，不要算力）。
**本模块做后半** —— 于是算力一获批即可出结果，**不必再等开发**。

## 口径（可审计）

| 量 | 定义 |
| --- | --- |
| 样本 | N 次重复的标量结果（如某指标的真实值） |
| 均值 | 算术平均 |
| **置信区间** | **t 区间**（小样本、方差未知）：`mean ± t_{(1-α/2, N-1)} · s/√N` |
| 自由度不足 | `N < 2` → **无区间**（如实返回 `None` + 理由），**不假装能算** |

**为什么用 t 而非正态**：`N≥30` 是判据下限，而 30 属小样本边界；
**t 区间在 N 小时更保守**，且 N→∞ 时收敛到正态 —— **故它不会比正态更松**。

## ⚠ 边界

本模块**只做统计聚合**：它**不知道**样本是否独立同分布。
**若 N 次重复共用了同一 seed，样本不独立，区间会偏窄** ——
故 `aggregate()` 要求传入 `seeds` 并**检查唯一性**，重复 seed 会被报出。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# t 分布双侧分位数（95% 置信，df = N-1）
# 只列到 df=30，之后用正态 1.96 近似（df≥30 时两者差 <2%）
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_Z95 = 1.959964


def t_critical(df: int, alpha: float = 0.05) -> float:
    """双侧 t 分位数。`df >= 30` 用正态近似（`alpha != 0.05` 时同样用正态）。"""
    if alpha != 0.05:
        return _Z95          # 只标定 95%；其它水平如实退回正态近似
    if df <= 0:
        raise ValueError("df 必须 >= 1")
    return _T95.get(df, _Z95)


def aggregate(samples: Sequence[float], *, seeds: Optional[Sequence[Any]] = None,
              min_n: int = 30, alpha: float = 0.05) -> Dict[str, Any]:
    """把 N 次重复聚成均值与置信区间。

    Parameters
    ----------
    samples:
        N 次重复的标量结果。
    seeds:
        每次重复用的 seed。**用于检查独立性** —— 有重复 seed 时区间不可信，会写入
        `seed_duplicates` 并以 `ci_reliable=False` 标出。
    min_n:
        判据要求的样本量下限（默认 30）。低于它**不报错**，但 `meets_min_n=False`。
    alpha:
        显著性水平（默认 0.05 → 95% 区间）。

    Returns
    -------
    dict
        固定字段：`n` / `mean` / `stdev` / `ci_low` / `ci_high` / `ci_method` /
        `meets_min_n` / `ci_reliable` / 诊断字段。**取不到区间时 `ci_low/ci_high` 为 `None`
        并给出 `ci_unavailable_reason`，不编造数值。**
    """
    vals: List[float] = []
    for x in samples:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        vals.append(f)

    out: Dict[str, Any] = {
        "n": len(vals),
        "n_input": len(samples),
        "mean": None, "stdev": None, "ci_low": None, "ci_high": None,
        "ci_method": None, "ci_unavailable_reason": None,
        "meets_min_n": len(vals) >= min_n,
        "min_n_required": min_n,
        "alpha": alpha,
        "ci_reliable": True,
        "seed_duplicates": [],
        "dropped_non_finite": len(samples) - len(vals),
    }

    if len(vals) < 2:
        out["ci_unavailable_reason"] = "n < 2，无法估计方差"
        out["ci_reliable"] = False
        if vals:
            out["mean"] = vals[0]
        return out

    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)      # 样本方差（无偏）
    sd = math.sqrt(var)
    out["mean"] = mean
    out["stdev"] = sd

    if seeds is not None:
        seen, dups = set(), []
        for s in seeds:
            if s in seen:
                dups.append(s)
            seen.add(s)
        out["seed_duplicates"] = sorted({str(d) for d in dups})
        if dups:
            # 重复 seed → 样本不独立 → 区间偏窄，**不谎称可靠**
            out["ci_reliable"] = False

    if sd == 0.0:
        out["ci_low"] = out["ci_high"] = mean
        out["ci_method"] = "t(degenerate, sd=0)"
        return out

    df = n - 1
    t = t_critical(df, alpha)
    half = t * sd / math.sqrt(n)
    out["ci_low"] = mean - half
    out["ci_high"] = mean + half
    out["ci_half_width"] = half
    out["ci_method"] = "t(df=%d, alpha=%.2f)" % (df, alpha) if df in _T95 else \
        "normal-approx(df=%d, alpha=%.2f)" % (df, alpha)
    return out


def aggregate_by_case(results: Sequence[Dict[str, Any]], *, metric: str,
                      case_key: str = "case_id", seed_key: str = "seed",
                      min_n: int = 30) -> Dict[str, Dict[str, Any]]:
    """按用例分组聚合：`{case_id: aggregate(...)}`。

    输入形状：`[{"case_id": ..., "seed": ..., metric: 值}, ...]`
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(str(r.get(case_key)), []).append(r)
    out = {}
    for cid, rows in groups.items():
        out[cid] = aggregate([r.get(metric) for r in rows],
                             seeds=[r.get(seed_key) for r in rows], min_n=min_n)
    return out


def overall_summary(per_case: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """把所有用例的区间汇总成一句可写进报告的结论（**不夸大**）。"""
    n_cases = len(per_case)
    enough = [c for c, v in per_case.items() if v.get("meets_min_n")]
    reliable = [c for c, v in per_case.items() if v.get("ci_reliable")]
    return {
        "cases": n_cases,
        "cases_meeting_min_n": len(enough),
        "cases_with_reliable_ci": len(reliable),
        "all_meet_min_n": n_cases > 0 and len(enough) == n_cases,
        "all_ci_reliable": n_cases > 0 and len(reliable) == n_cases,
    }
