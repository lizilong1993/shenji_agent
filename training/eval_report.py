#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估报告模板（`P3` 步骤 6）：把三层指标**组装成一份缺项可见的报告**。

## 标准依据

> `P3` 步骤 6：把上述指标**固化为评估报告模板**并接入 `training/evaluate.py` 输出。
> `P3`【预期结果】：评估报告含三层框架与全部规定指标，**缺项有明确说明**；
> **评估结论不再依赖单一胜率**。

## 三层框架（`P3` 步骤 1）

| 层 | 名称 | 数据来源 |
| --- | --- | --- |
| **L1 仿真环境验证** | 用例与边界覆盖 | `Shenji/configs/eval_cases_intensity_v1.json` |
| **L2 实际推演校验** | 蒙特卡洛 + 置信区间 | `training/monte_carlo.py` |
| **L3 横向对比** | 专家评分 + 认知对齐度 | `training/expert_rating.py` / `training/alignment.py` |

## ★ 本模块的核心原则：**缺项必须显式可见**

`P3` 原文要求「**缺项有明确说明**」。故模板**不允许**：

1. **缺失指标静默消失** —— 每个指标要么有值，要么进 `missing` 并带理由；
2. **用 0 代替"没测"** —— `None` 与 `0.0` 在本报告里**语义不同**，渲染时必须区分；
3. **结论只看胜率** —— `conclusion` 由**可用指标集合**推出；**可用指标不足时明确说"不足以支撑结论"**。

## ⚠ 边界

本模块**只组装与渲染**，不重算：所有数值由调用方传入（或由既有模块算出）。
**它是模板，不是评测器。**
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# 三层 × 各层必需指标（缺一即计入 missing）
REQUIRED: Dict[str, tuple] = {
    "L1_simulation": ("cases_per_level", "boundary_coverage"),
    "L2_replay": ("win_rate", "monte_carlo_ci"),
    "L3_comparison": ("expert_total_mean", "cognitive_alignment"),
}


def _is_measured(v: Any) -> bool:
    """`None` = 未测；`0.0` = 测了且为零 —— **二者必须区分**。"""
    return v is not None


def build_report(*,
                 framework: Optional[Dict[str, Any]] = None,
                 layers: Dict[str, Dict[str, Any]],
                 episode_counts: Optional[Dict[str, int]] = None,
                 provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """组装报告骨架。

    Parameters
    ----------
    layers:
        `{"L1_simulation": {...}, "L2_replay": {...}, "L3_comparison": {...}}`，
        每层是「指标名 → 值」，**值为 `None` 表示未测**。
    episode_counts:
        各层实际用了多少样本，便于判断区间是否可信。
    provenance:
        数据来源与口径（如权重是否自定、用例文件哈希）。

    Returns
    -------
    dict
        `layers`（原样回带）· `missing`（缺失指标 + 理由）· `measured_count` ·
        `conclusion`（**由可用指标推出**，不足时明说）· `disclaimer`。
    """
    layers = {k: dict(v) for k, v in (layers or {}).items()}
    missing: List[Dict[str, str]] = []

    for layer, keys in REQUIRED.items():
        got = layers.get(layer) or {}
        for k in keys:
            if not _is_measured(got.get(k)):
                missing.append({
                    "layer": layer,
                    "metric": k,
                    "reason": "未提供" if k not in got else "值为 None（未测）",
                })

    total_required = sum(len(v) for v in REQUIRED.values())
    measured = total_required - len(missing)

    out: Dict[str, Any] = {
        "schema": "eval_report/1",
        "framework": framework or {
            "L1_simulation": "仿真环境验证",
            "L2_replay": "实际推演校验",
            "L3_comparison": "横向对比",
        },
        "layers": layers,
        "required_metrics_total": total_required,
        "measured_count": measured,
        "missing": missing,
        "episode_counts": dict(episode_counts or {}),
        "provenance": dict(provenance or {}),
        "conclusion": None,
        "conclusion_basis": [],
        "disclaimer": ("值为 None 表示**未测**，与 0.0（测得为零）语义不同；"
                       "缺失指标不得按 0 计入任何均值"),
    }
    out["conclusion"], out["conclusion_basis"] = _conclude(layers, missing, measured, total_required)
    return out


def _conclude(layers: Dict[str, Dict[str, Any]], missing: List[Dict[str, str]],
              measured: int, total: int) -> tuple:
    """由**可用指标**推出结论；**不足时明确说不足**。"""
    basis: List[str] = []
    if measured == 0:
        return "INSUFFICIENT_EVIDENCE（三项判据均未测，不足以支撑任何结论）", basis

    l1 = layers.get("L1_simulation") or {}
    l2 = layers.get("L2_replay") or {}
    l3 = layers.get("L3_comparison") or {}

    if _is_measured(l1.get("boundary_coverage")):
        basis.append("L1 边界覆盖已测")
    if _is_measured(l2.get("monte_carlo_ci")):
        basis.append("L2 蒙特卡洛区间已测")
    if _is_measured(l3.get("cognitive_alignment")):
        basis.append("L3 认知对齐度已测")
    if _is_measured(l2.get("win_rate")):
        basis.append("L2 胜率已测")

    # **不把胜率当唯一结论** —— 只有在多个维度可用时才给"支撑"结论
    if measured >= 4 and not missing:
        verdict = "SUPPORTED（三类指标齐备）"
    elif measured >= 2:
        verdict = "PARTIALLY_SUPPORTED（仅 %d/%d 项指标可测）" % (measured, total)
    else:
        verdict = "INSUFFICIENT_EVIDENCE（仅 %d/%d 项可测）" % (measured, total)

    if any("胜率" in b for b in basis) and len(basis) == 1:
        # 只有胜率 → 明确说明**不构成**标准要求的结论
        verdict += "；**且仅有胜率，按 31007 要求不足以单独支撑评估结论**"
    return verdict, basis


def render_markdown(report: Dict[str, Any]) -> str:
    """渲染成 Markdown（**缺项逐条列出，不省略**）。"""
    L: List[str] = []
    L.append("# 评估报告（三层框架）")
    L.append("")
    L.append("**结论**：%s" % report.get("conclusion"))
    L.append("")
    if report.get("conclusion_basis"):
        L.append("**结论依据**：" + "；".join(report["conclusion_basis"]))
        L.append("")
    L.append("**指标齐备度**：%d / %d" % (report.get("measured_count", 0),
                                          report.get("required_metrics_total", 0)))
    L.append("")
    L.append("> %s" % report.get("disclaimer", ""))
    L.append("")

    for layer, label in (report.get("framework") or {}).items():
        L.append("## %s　%s" % (layer, label))
        L.append("")
        got = report["layers"].get(layer) or {}
        if not got:
            L.append("（该层未提供任何指标）")
            L.append("")
            continue
        L.append("| 指标 | 值 | 状态 |")
        L.append("| --- | --- | --- |")
        for k, v in got.items():
            status = "**未测**" if v is None else "已测"
            shown = "—（未测）" if v is None else _fmt(v)
            L.append("| `%s` | %s | %s |" % (k, shown, status))
        L.append("")

    L.append("## 缺项清单（%d 项）" % len(report.get("missing") or []))
    L.append("")
    if not report.get("missing"):
        L.append("无缺项。")
    else:
        L.append("| 层 | 指标 | 原因 |")
        L.append("| --- | --- | --- |")
        for m in report["missing"]:
            L.append("| `%s` | `%s` | %s |" % (m["layer"], m["metric"], m["reason"]))
    L.append("")

    if report.get("episode_counts"):
        L.append("## 样本量")
        L.append("")
        for k, v in report["episode_counts"].items():
            L.append("- %s：%s" % (k, v))
        L.append("")
    if report.get("provenance"):
        L.append("## 口径与来源")
        L.append("")
        for k, v in report["provenance"].items():
            L.append("- **%s**：%s" % (k, v))
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "%.4f" % v
    if isinstance(v, (dict, list)):
        return "`" + json.dumps(v, ensure_ascii=False)[:120] + "`"
    return str(v)
