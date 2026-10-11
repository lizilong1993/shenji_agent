"""`I17` / `P3` 步骤 4 的**文档—实现一致性**测试。

## 为什么需要

`docs/expert_rating_protocol.md` 是**评分协议**，而 `training/expert_rating.py` 是**实现**。
**两者若漂移，协议就会变成一份描述不存在行为的文档** ——
而"文档声称"与"代码事实"不一致正是本会话反复发现的问题
（§104 产物与生成器脱节、§92 落点指向不存在的路径）。

**故把协议里的关键数值与名字**全部与实现比对**。**
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.expert_rating import (  # noqa: E402
    DIMENSIONS, DIMENSION_WEIGHTS, SCALE_MAX, SCALE_MIN, rule_conflicts,
)

DOC = REPO / "docs" / "expert_rating_protocol.md"


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("expert_rating_protocol.md not present")
    return DOC.read_text(encoding="utf-8")


# ══════════════ 文档存在且覆盖五要素

def test_document_exists() -> None:
    assert DOC.is_file(), "缺 docs/expert_rating_protocol.md"


def test_document_declares_the_scale() -> None:
    t = _doc()
    assert "0—10" in t or "0-10" in t, "未声明量程"
    assert str(SCALE_MIN) in t and str(SCALE_MAX) in t


def test_document_covers_all_dimensions() -> None:
    t = _doc()
    missing = [k for k in DIMENSIONS if k not in t]
    assert not missing, "协议未列出维度: %s" % missing


def test_document_weights_match_the_implementation() -> None:
    """**权重必须与实现一致** —— 否则协议说的和算的不是一回事。"""
    t = _doc()
    bad = []
    for k, w in DIMENSION_WEIGHTS.items():
        # 在维度所在表格行里找该权重
        row = next((l for l in t.split("\n") if "`%s`" % k in l), "")
        if ("%.2f" % w) not in row:
            bad.append((k, w, row[:80]))
    assert not bad, "协议里的权重与实现不符: %s" % bad


def test_document_states_weights_are_self_chosen() -> None:
    """**必须声明权重是自定的** —— 否则等于把自定口径冒充标准规定。"""
    t = _doc()
    assert "自定" in t or "本项目自定" in t
    assert "31007" in t and "只规定" in t.replace(" ", "")


# ══════════════ 流程与样本量

def test_document_specifies_the_rating_process() -> None:
    t = _doc()
    for kw in ("盲评", "独立", "一致性", "复核"):
        assert kw in t, "协议缺流程要素: %s" % kw


def test_document_specifies_sample_size_floor() -> None:
    t = _doc()
    assert "≥ 2" in t or "≥2" in t, "未给出评分者/样本下限"
    assert "评分者" in t and "样本" in t


# ══════════════ ICC 口径

def test_document_names_icc_2_1() -> None:
    assert "ICC(2,1)" in _doc()


def test_document_explains_why_not_kappa() -> None:
    t = _doc()
    assert "κ" in t or "kappa" in t.lower(), "未说明为何不用 κ"
    assert "分箱" in t, "未说明 κ 需分箱丢信息"


def test_document_states_no_pass_fail_verdict() -> None:
    """**必须写明不给合格判定**（阈值属惯例）—— 否则读者会以为有标准阈值。"""
    t = _doc()
    assert "不给" in t and ("合格" in t or "阈值" in t)
    assert "惯例" in t


def test_document_forbids_mean_imputation() -> None:
    t = _doc()
    assert "不得用均值填补" in t or "不得用均值" in t


# ══════════════ 规则冲突

def test_document_lists_the_conflict_rules_present_in_code() -> None:
    """协议列出的规则名必须**真的在实现里**。"""
    acts = [{"kind": "occupy", "actor": 1, "target": "A"},
            {"kind": "withdraw", "actor": 1, "target": "A"},
            {"kind": "attack", "actor": 2, "target": 3},
            {"kind": "hold_fire", "actor": 2, "target": 3},
            {"kind": "engage", "actor": 4, "target": 5},
            {"kind": "disengage", "actor": 4, "target": 5}]
    impl_rules = {h["rule"] for h in rule_conflicts(acts)}
    t = _doc()
    for r in impl_rules:
        assert r in t, "实现有规则 %s，但协议未写" % r
    assert len(impl_rules) == 3, impl_rules


def test_document_states_the_false_positive_boundaries() -> None:
    """两条**防误报**边界必须写进协议（不同主体 / 不同目标都不是冲突）。"""
    t = _doc()
    assert "不同主体" in t and "不是冲突" in t
    assert "不同目标" in t


# ══════════════ 如实列出未完成部分

def test_document_lists_what_is_not_done() -> None:
    """**协议必须自己写明未完成部分** —— 否则读者以为评分已做过。"""
    t = _doc()
    assert "未完成" in t
    assert "专家资源" in t or "评分者" in t
    assert "需要的是**评分者**" in t or "评分者" in t


def test_document_does_not_claim_results_exist() -> None:
    """**不得声称已有评分结果** —— 那是虚报。"""
    t = _doc()
    assert "已就绪" in t or "未完成" in t
    # 不应出现"已完成评分"这类断言
    for bad in ("已完成评分", "评分已完成", "已有评分结果"):
        assert bad not in t, "文档声称了不存在的评分结果: %s" % bad
