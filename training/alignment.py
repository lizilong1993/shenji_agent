#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""认知对齐度（`P3` 步骤 5 / `I17`）：**显性行动吻合度 + 隐性意图推理匹配度**。

## 标准依据（31007 6.2.4）

> 认知对齐度 = **显性行动吻合度** + **隐性意图推理匹配度**。

原文给了**两个分量**，但**未给权重与归一化方式** —— 故本模块的做法**必须显式声明口径**
（见 `DEFAULT_WEIGHTS` 与各函数的 docstring），**不假装这是原文规定**。

## 两个分量

| 分量 | 含义 | 本模块的口径 |
| --- | --- | --- |
| **显性行动吻合度** | 系统给出的**动作**与人类操作者实际动作的一致程度 | 逐步比对动作序列，取**最长公共子序列 / 较长序列长度**（顺序敏感、允许缺项） |
| **隐性意图推理匹配度** | 系统推断的**意图**与人类真实意图的一致程度 | 按意图标签比对；**用置信度加权**（低置信的正确比高置信的错误贡献小） |

## ⚠ 边界（必须写明）

1. **权重是本模块自定**（`0.5/0.5`），标准未规定 —— 报告中必须注明。
2. **动作比对用 LCS 而非严格相等**：实际操作序列常有插入/删除，严格相等会把"大体一致"判成 0。
3. **无样本时不返回 0**，而返回 `None` + 理由 —— **"没数据"不等于"对齐度为 0"**。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_WEIGHTS = {"explicit": 0.5, "implicit": 0.5}


def _lcs_len(a: Sequence[Any], b: Sequence[Any]) -> int:
    """最长公共子序列长度（标准 DP）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def action_agreement(system_actions: Sequence[Any],
                     human_actions: Sequence[Any]) -> Optional[float]:
    """**显性行动吻合度** ∈ [0,1]；无任何一方有动作时返回 `None`。

    口径：`LCS(系统, 人工) / max(len(系统), len(人工))`。

    · 用 `max` 而非 `len(人工)` 作分母：**多报**动作也应受罚（否则"全报一遍"能刷满分）。
    · 顺序敏感：`[A,B]` 与 `[B,A]` 得 `1/2`，不是 `1`。
    """
    s, h = list(system_actions), list(human_actions)
    if not s and not h:
        return None
    denom = max(len(s), len(h))
    if denom == 0:
        return None
    return _lcs_len(s, h) / float(denom)


def intent_match(system_intents: Sequence[Any],
                 human_intents: Sequence[Any],
                 *,
                 confidences: Optional[Sequence[float]] = None) -> Optional[float]:
    """**隐性意图推理匹配度** ∈ [0,1]；无样本时返回 `None`。

    口径：逐位置比对意图标签，**用置信度加权**：

        Σ (置信度_i · 1[系统意图_i == 人类意图_i]) / Σ 置信度_i

    · `confidences` 省略时全部取 1.0（退化为**未加权命中率**）。
    · 长度不等时按**较短者**对齐并**只统计较短长度** —— 多出的部分不计（如实处理，
      不擅自补齐；补齐会人为拉低或抬高）。
    """
    s, h = list(system_intents), list(human_intents)
    n = min(len(s), len(h))
    if n == 0:
        return None
    if confidences is None:
        w = [1.0] * n
    else:
        w = []
        for c in list(confidences)[:n]:
            try:
                f = float(c)
            except (TypeError, ValueError):
                f = 0.0
            if not math.isfinite(f) or f < 0:
                f = 0.0
            w.append(f)
        if sum(w) == 0:
            return None
    total = sum(w)
    if total == 0:
        return None
    hit = sum(wi for wi, a, b in zip(w, s[:n], h[:n]) if a == b)
    return hit / total


def cognitive_alignment(system_actions: Sequence[Any],
                        human_actions: Sequence[Any],
                        system_intents: Sequence[Any],
                        human_intents: Sequence[Any],
                        *,
                        confidences: Optional[Sequence[float]] = None,
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """**认知对齐度** = 加权和；**任一分量缺失时按可得分量归一化权重**。

    Returns
    -------
    dict
        `value`（∈[0,1] 或 `None`）· 两个分量 · 实际使用的权重 · 说明。
        **取不到时 `value=None` 并给 `reason`，不返回 0 冒充"对齐度为零"。**
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in DEFAULT_WEIGHTS})

    explicit = action_agreement(system_actions, human_actions)
    implicit = intent_match(system_intents, human_intents, confidences=confidences)

    parts = [(k, v) for k, v in (("explicit", explicit), ("implicit", implicit)) if v is not None]
    out: Dict[str, Any] = {
        "explicit_action_agreement": explicit,
        "implicit_intent_match": implicit,
        "weights_declared": dict(w),
        "weights_used": {},
        "value": None,
        "reason": None,
        "note": "权重为本模块自定（标准 31007 6.2.4 未规定权重），报告须注明",
    }
    if not parts:
        out["reason"] = "两个分量都取不到（无动作或无意图样本）"
        return out

    # 只有一部分可得 → **按可得分量归一化**，否则会因缺项而被压低
    norm = sum(w[k] for k, _ in parts)
    used = {k: w[k] / norm for k, _ in parts}
    out["weights_used"] = used
    out["value"] = sum(used[k] * v for k, v in parts)
    if len(parts) == 1:
        out["reason"] = "仅一个分量可得（%s），已按其归一化" % parts[0][0]
    return out


def aggregate_alignment(per_episode: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把多局的认知对齐度聚成**均值 + 95% t 区间**（复用 `training.monte_carlo`）。

    **只统计 `value is not None` 的局**；并如实报出被排除的局数。
    """
    from training.monte_carlo import aggregate  # 局部导入，避免循环

    vals, dropped = [], 0
    for e in per_episode:
        v = e.get("value")
        if v is None:
            dropped += 1
            continue
        vals.append(v)
    r = aggregate(vals, min_n=1)
    r["episodes_total"] = len(per_episode)
    r["episodes_excluded_missing_value"] = dropped
    return r
