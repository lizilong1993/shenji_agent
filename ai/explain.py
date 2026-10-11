#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""决策可解释性与决策安全约束（`I17` 步骤 4 补遗；31007 6.2.4）。

## 标准依据

> 31007 6.2.4 系统效能：认知对齐度、跨域泛化、抗扰容错阈值、
> **决策可解释性**、**决策安全约束**。

**标准点名了这两项，未给口径。** 本模块的口径**全部自定且显式声明**。

## 设计原则：**不另写一份规格**

本仓已有权威来源：

| 来源 | 内容 |
| --- | --- |
| `ai/features.py::FEATURE_NAMES` | 108 维特征的**名字**（P2 侧） |
| `ai/adapters/llm_adapter.py::_action_spec_required` | 每类动作的**必填字段**（源自 `ACTION_SPECS`） |

**故本模块不重复列举特征名或字段规则，而是引用它们** ——
本会话已两次记录「在既有规格之外又手写一份」的教训（§51 动作字段、§102.4 调和逻辑）。

## 两项的口径（本项目自定）

### 决策可解释性

`explain_decision()` 把决策拆成**可读的贡献项**：

| 字段 | 含义 |
| --- | --- |
| `chosen` | 被选中的动作 |
| `alternatives` | 被考虑但未选的动作（**保留其分数**，便于核"为什么不是它"） |
| `margin` | 首选与次选的**分数差** —— **差额小意味着决策接近随机，必须能被看见** |
| `contributors` | 分数的**具名来源**（特征名或规则名 → 贡献值） |
| `basis` | `rule` / `learned` / `heuristic` —— **决策依据的类型** |

**关键**：`contributors` 只接受**具名**来源。给不出名字的贡献**计入 `unattributed` 并报出**，
而不是静默并进总数 —— 那会把"算不出为什么"伪装成"已解释"。

### 决策安全约束

`check_safety()` 检查**硬约束**（违反即为不安全，不是扣分）：

| 约束 | 判定 |
| --- | --- |
| `required_fields` | 动作必填字段是否齐（**复用仓内 `ACTION_SPECS` 口径**） |
| `actor_controllable` | 动作主体是否在可控列表内 |
| `no_self_target` | 不得以自己为目标 |
| `weapon_available` | 打击类动作须有可用武器 |
| `explicit_deny` | 调用方给出的**显式禁止集** |

**硬约束与软因素分开报**：`violations`（违反）与 `warnings`（可疑但未违反）。
**把两者混在一起会让"违规"看起来只是"警告"。**
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

BASES = ("rule", "learned", "heuristic", "unknown")


# ══════════════════════════════════════════════════════
# ① 决策可解释性
# ══════════════════════════════════════════════════════

def explain_decision(chosen: Optional[Dict[str, Any]],
                     alternatives: Sequence[Dict[str, Any]] = (),
                     *,
                     score_key: str = "score",
                     contributors: Optional[Sequence[Dict[str, Any]]] = None,
                     basis: str = "unknown",
                     required_contributions: Optional[float] = None) -> Dict[str, Any]:
    """把一次决策拆成可读的贡献项。

    Parameters
    ----------
    chosen:
        被选中的动作（含 `score`）。`None` 表示**没有选出任何动作**。
    alternatives:
        被考虑的其他动作。
    contributors:
        具名贡献项：`[{"name": ..., "value": ...}, ...]`。**必须具名** ——
        无名项进 `unattributed`。
    required_contributions:
        调用方声称的总贡献（用于**核对**具名项之和是否吻合）。

    Returns
    -------
    dict
        `chosen` · `alternatives` · `margin` · `contributors` · `unattributed` ·
        `basis` · `explained`（是否达到可解释要求）· `reason`。
    """
    out: Dict[str, Any] = {
        "chosen": chosen, "alternatives": list(alternatives),
        "margin": None, "contributors": [], "unattributed": [],
        "contributions_named_sum": None,
        "basis": basis if basis in BASES else "unknown",
        "explained": False, "reason": None,
        "note": ("贡献项必须**具名**；给不出名字的贡献进 unattributed 并报出，"
                 "**不静默并进总数**"),
    }
    if chosen is None:
        out["reason"] = "未选出任何动作 —— 无可解释对象"
        return out

    # 贡献项过滤：只留具名的
    named: List[Dict[str, Any]] = []
    unnamed = 0
    for c in contributors or []:
        name = c.get("name")
        if not name or not str(name).strip():
            unnamed += 1
            continue
        try:
            v = float(c.get("value"))
        except (TypeError, ValueError):
            out["unattributed"].append({"reason": "贡献值非数值", "item": c})
            continue
        if not math.isfinite(v):
            out["unattributed"].append({"reason": "贡献值非有限", "item": c})
            continue
        named.append({"name": str(name), "value": v})
    if unnamed:
        out["unattributed"].append({"reason": "%d 个贡献项缺 name" % unnamed})
    out["contributors"] = named
    out["contributions_named_sum"] = sum(c["value"] for c in named) if named else None

    # 与声称的总贡献核对
    if required_contributions is not None and named:
        try:
            want = float(required_contributions)
        except (TypeError, ValueError):
            want = None
        if want is not None and math.isfinite(want):
            gap = want - out["contributions_named_sum"]
            out["contributions_gap"] = gap
            if abs(gap) > 1e-6:
                out["unattributed"].append({
                    "reason": "具名贡献之和与声称总值差 %.6g" % gap,
                    "claimed": want, "named_sum": out["contributions_named_sum"],
                })

    # 决策余量
    try:
        cs = float(chosen.get(score_key))
    except (TypeError, ValueError):
        cs = None
    best_alt = None
    for a in alternatives:
        try:
            v = float(a.get(score_key))
        except (TypeError, ValueError):
            continue
        if best_alt is None or v > best_alt:
            best_alt = v
    if cs is not None and best_alt is not None:
        out["margin"] = cs - best_alt
        if out["margin"] <= 0:
            out["warnings"] = ["首选分数并不高于次选（margin<=0）—— 选择依据存疑"]

    # 可解释性的要求：有具名贡献 **且** 无未归属项
    out["explained"] = bool(named) and not out["unattributed"]
    if not named:
        out["reason"] = "没有任何具名贡献项 —— 该决策**不可解释**"
    elif out["unattributed"]:
        out["reason"] = "存在未归属贡献（%d 项）—— 尚未完全解释" % len(out["unattributed"])
    return out


# ══════════════════════════════════════════════════════
# ② 决策安全约束
# ══════════════════════════════════════════════════════

def _required_fields_for(action_type: Any) -> Optional[List[str]]:
    """复用仓内 `ACTION_SPECS` 口径（**不另写一份字段规则**）。"""
    try:
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from ai.adapters.llm_adapter import _action_spec_required  # noqa: PLC0415
        return _action_spec_required(int(action_type))
    except Exception:
        return None      # 取不到就**不假装能校**，由调用方决定


def check_safety(action: Optional[Dict[str, Any]], *,
                 controllable_ops: Optional[Sequence[Any]] = None,
                 deny: Optional[Sequence[str]] = None,
                 known_kinds: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """检查**硬约束**；违反即为不安全（**不是扣分**）。

    Returns
    -------
    dict
        `safe`（`True`/`False`/`None`）· `violations`（硬违反）· `warnings`（可疑）·
        `checked`（实际执行的检查项）· `unchecked`（无法执行的检查及理由）。
        **`safe=None` 表示"有检查无法执行"，而不是"安全"** —— 二者必须区分。
    """
    out: Dict[str, Any] = {
        "safe": None, "violations": [], "warnings": [],
        "checked": [], "unchecked": [],
        "note": ("硬约束违反进 violations；可疑但不构成违反进 warnings。"
                 "**两者混同会让违规看起来只是警告**；"
                 "**safe=None 表示有检查无法执行，不等于安全**"),
    }
    if action is None:
        out["safe"] = False
        out["violations"].append({"rule": "no_action", "detail": "动作为空"})
        return out
    if not isinstance(action, dict):
        out["safe"] = False
        out["violations"].append({"rule": "not_a_dict", "detail": "动作不是字典"})
        return out

    # ① 幻影字段：有字段但无对应动作规格 → 该规格查不到，记 unchecked
    atype = action.get("type")
    if atype is not None:
        req = _required_fields_for(atype)
        if req is None:
            out["unchecked"].append({
                "rule": "required_fields",
                "reason": "取不到动作类型 %r 的字段规格（ACTION_SPECS 未覆盖）" % atype})
        else:
            missing = [f for f in req if action.get(f) in (None, "", [], {})]
            out["checked"].append("required_fields")
            if missing:
                out["violations"].append({
                    "rule": "required_fields",
                    "detail": "缺必填字段: %s" % missing, "missing": missing})
    else:
        out["unchecked"].append({"rule": "required_fields", "reason": "动作无 type 字段"})

    # ② 主体可控
    if controllable_ops is not None:
        allowed = set()
        for x in controllable_ops:
            try:
                allowed.add(int(x))
            except (TypeError, ValueError):
                allowed.add(x)
        obj = action.get("obj_id")
        out["checked"].append("actor_controllable")
        if obj not in allowed:
            out["violations"].append({
                "rule": "actor_controllable",
                "detail": "主体 %r 不在可控列表 %s 内" % (obj, sorted(map(str, allowed)))})
    else:
        out["unchecked"].append({"rule": "actor_controllable", "reason": "未提供可控主体列表"})

    # ③ 不得以自己为目标
    tgt = action.get("target_obj_id")
    obj = action.get("obj_id")
    out["checked"].append("no_self_target")
    if tgt is not None and obj is not None and tgt == obj:
        out["violations"].append({"rule": "no_self_target",
                                  "detail": "目标 %r 与主体相同" % tgt})

    # ④ 打击类动作须有武器
    kind = str(action.get("kind") or "").lower()
    if kind in ("attack", "strike", "fire", "shoot"):
        w = action.get("weapon_id")
        out["checked"].append("weapon_available")
        if w in (None, "", [], {}):
            out["violations"].append({"rule": "weapon_available",
                                      "detail": "打击类动作未给出 weapon_id"})
    # ⑤ 显式禁止集
    if deny:
        out["checked"].append("explicit_deny")
        hit = [d for d in deny if d and d in str(action.get("kind", ""))]
        for h in hit:
            out["violations"].append({"rule": "explicit_deny",
                                      "detail": "命中显式禁止项 %r" % h})
    # 未知动作类型 → 可疑（不是硬违反：可能只是本模块不认识）
    if known_kinds is not None and kind and kind not in set(known_kinds):
        out["warnings"].append({"rule": "unknown_kind",
                                "detail": "动作类型 %r 不在已知集合内" % kind})

    if out["unchecked"]:
        # **有检查无法执行 → 不能声称安全**
        out["safe"] = False if out["violations"] else None
        out["reason"] = ("有 %d 项检查无法执行 —— **不足以判定安全**"
                         % len(out["unchecked"]))
    else:
        out["safe"] = not out["violations"]
    return out


def safety_summary(checks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总多步决策的安全检查，**把"未判定"单独计数**。"""
    n = len(checks)
    unsafe = sum(1 for c in checks if c.get("safe") is False and c.get("violations"))
    undecided = sum(1 for c in checks if c.get("safe") is None)
    safe = sum(1 for c in checks if c.get("safe") is True)
    return {
        "n_actions": n, "safe": safe, "unsafe": unsafe, "undecided": undecided,
        "all_safe": n > 0 and safe == n,
        "note": ("**undecided（有检查无法执行）与 unsafe 不同，也不等于 safe** —— "
                 "报告须分别列出"),
        "violations": [v for c in checks for v in (c.get("violations") or [])],
    }
