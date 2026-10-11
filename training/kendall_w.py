#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kendall's W（协调系数）—— `I17` 步骤 1 **点名**的一致性系数。

## 为何补它

`I17` 步骤 1 原文：「…样本量与一致性系数（**如 Kendall W**）要求」。

我在第 107 轮做协议时用了 **ICC(2,1)**，理由写的是"κ 适用于分类而本量表有序/连续"。
**但 `I17` 点名的是 Kendall's W，我未核对过它与 ICC 的差别就选了 ICC。**

> **这不是"ICC 错"** —— 两者衡量的是**不同问题**。**本轮补上 W，并把差别实测出来**，
> **由数据决定报告该用哪个**，而不是由我的偏好决定。

## 两者衡量什么（**关键区别**）

| 系数 | 问的问题 | 数据要求 |
| --- | --- | --- |
| **Kendall's W** | **评分者对"被评对象"的排序是否一致**？ | 评分者把 n 个对象**排序**（或分数可转秩） |
| **ICC(2,1)** | **同一对象上，评分者的绝对分数是否可互换**？ | 评分者对每个对象给**绝对分** |

**结论（实测见测试）**：

· W 对**单调变换不敏感**（都加 2 分、都乘 2，W 不变）；
· ICC(2,1) 是**绝对一致性**，系统性偏移会**降低**它。

**故两者回答不同问题，报告应并报、并说明所选口径。**
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

TIES = "average"


def _rank_with_ties(vals: Sequence[float]) -> List[float]:
    """秩（并列取平均秩）。"""
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0        # 秩从 1 开始
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def kendall_w(ratings: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Kendall's W：`ratings` 为**每个对象一行、每个评分者一列**。

    公式（含并列修正）：

        W = 12·S / ( m²·(n³ − n) − m·ΣT )
        S = Σ (R_j − R̄)²          （R_j = 对象 j 的秩和）
        T = Σ (t³ − t)            （每个评分者内，同分组的并列修正）

    其中 `m` = 评分者数、`n` = 对象数。

    **样本不足时返回 `w=None` + 理由**，不编造。
    """
    rows = [list(r) for r in ratings]
    n = len(rows)
    if n == 0:
        return {"w": None, "reason": "无对象", "n_items": 0, "n_raters": 0}
    m = len(rows[0])
    if m < 2:
        return {"w": None, "reason": "评分者少于 2 人", "n_items": n, "n_raters": m}
    if any(len(r) != m for r in rows):
        return {"w": None, "reason": "行长度不一致（存在缺分）", "n_items": n, "n_raters": m}
    if n < 2:
        return {"w": None, "reason": "对象少于 2 个", "n_items": n, "n_raters": m}

    # 逐评分者（列）做秩
    col_ranks: List[List[float]] = [[] for _ in range(m)]
    tie_sum = 0.0
    for j in range(m):
        col = [rows[i][j] for i in range(n)]
        rk = _rank_with_ties(col)
        col_ranks[j] = rk
        # 并列修正 T = Σ(t³ - t)
        counts: Dict[float, int] = {}
        for v in col:
            counts[v] = counts.get(v, 0) + 1
        for t in counts.values():
            if t > 1:
                tie_sum += (t ** 3 - t)

    rank_sums = [sum(col_ranks[j][i] for j in range(m)) for i in range(n)]
    mean_rs = sum(rank_sums) / n
    S = sum((rs - mean_rs) ** 2 for rs in rank_sums)

    denom = (m ** 2) * (n ** 3 - n) - m * tie_sum
    out: Dict[str, Any] = {
        "w": None, "n_items": n, "n_raters": m, "S": S,
        "tie_correction": tie_sum,
        "formula": "W = 12S / (m^2(n^3-n) - m*ΣT)   [含并列修正]",
        "note": ("Kendall's W 衡量**排序一致性**，对单调变换不敏感；"
                 "ICC(2,1) 衡量**绝对一致性**，系统性偏移会降低它"),
    }
    if denom <= 0:
        out["reason"] = "分母为零或负（评分者全同分）"
        return out
    out["w"] = 12.0 * S / denom
    return out


def compare_w_and_icc(ratings: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """**并报**两种系数，供报告说明所选口径。

    **不判定"哪个对"** —— 它们回答不同问题（见模块 docstring）。
    """
    from training.expert_rating import icc_2_1

    w = kendall_w(ratings)
    icc = icc_2_1(ratings)
    return {
        "kendall_w": w,
        "icc_2_1": icc,
        "why_report_both": ("W 对**单调变换不敏感**（都加 2 分不影响 W）；"
                            "ICC(2,1) 是**绝对一致性**（系统性偏移会降低它）。"
                            "**两者回答不同问题，报告须写明所采用的口径。**"),
    }
