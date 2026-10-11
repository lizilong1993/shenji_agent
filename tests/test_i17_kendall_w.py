"""Kendall's W（`I17` 步骤 1 点名的一致性系数）的测试。

## 为什么补它

`I17` 步骤 1 原文写「一致性系数（**如 Kendall W**）」，
而我在第 107 轮用了 ICC(2,1) —— **未核对差别就选了 ICC**。
本轮补 W，并**实测两者差别**，由数据说明该报哪个。

## 重点

1. **W 的端点**：完全一致 → 1；完全无关 → 约 0；
2. **W 对单调变换不敏感**（+2 / ×2 不变）—— 这正是它与 ICC 的关键差别；
3. **ICC 对系统性偏移敏感** —— 两者并报时差别可见；
4. **样本不足 → `None` + 理由**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.kendall_w import compare_w_and_icc, kendall_w  # noqa: E402


# ══════════════ 端点

def test_perfect_agreement_gives_w_one() -> None:
    """所有评分者给出**相同排序** → W = 1。"""
    rows = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    r = kendall_w(rows)
    assert r["w"] == pytest.approx(1.0, abs=1e-9), r


def test_reversed_agreement_gives_low_w() -> None:
    """评分者排序**完全相反** → W 应接近 0（不是 1）。"""
    rows = [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]]
    r = kendall_w(rows)
    assert r["w"] is not None and r["w"] < 0.2, r


def test_w_is_between_zero_and_one() -> None:
    rows = [[5.0, 7.0, 6.0], [3.0, 2.0, 4.0], [9.0, 8.0, 7.0], [1.0, 1.0, 2.0]]
    w = kendall_w(rows)["w"]
    assert 0.0 <= w <= 1.0, w


# ══════════════ ★ 与 ICC 的关键差别

def test_w_is_invariant_under_a_constant_shift() -> None:
    """**W 对"都加 2 分"不敏感** —— 排序没变。"""
    base = [[5.0, 7.0], [6.0, 8.0], [4.0, 6.0], [7.0, 9.0]]
    shifted = [[x + 2.0 for x in row] for row in base]
    assert kendall_w(base)["w"] == pytest.approx(kendall_w(shifted)["w"], abs=1e-12)


def test_w_is_invariant_under_scaling() -> None:
    base = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    scaled = [[x * 2.0 for x in row] for row in base]
    assert kendall_w(base)["w"] == pytest.approx(kendall_w(scaled)["w"], abs=1e-12)


def test_icc_drops_while_w_stays_under_systematic_offset() -> None:
    """**本轮的核心实测**：同一批数据，评分者系统性偏高时 ——

    · **W 不变**（排序一致）；
    · **ICC(2,1) 下降**（绝对一致性被破坏）。

    **这正是"两者回答不同问题"的证据**，也是报告必须写明口径的理由。
    """
    base = [[5.0, 7.0], [6.0, 8.0], [4.0, 6.0], [7.0, 9.0], [3.0, 5.0], [8.0, 10.0]]
    shifted = [[r[0], r[1] + 3.0] for r in base]      # 第二位评分者一律高 3 分
    cmp_base = compare_w_and_icc(base)
    cmp_shift = compare_w_and_icc(shifted)
    assert cmp_base["kendall_w"]["w"] == pytest.approx(cmp_shift["kendall_w"]["w"], abs=1e-12), \
        "W 不应受整体平移影响"
    assert cmp_shift["icc_2_1"]["icc"] < cmp_base["icc_2_1"]["icc"], \
        "ICC(2,1) 是绝对一致性，系统性偏移**应当**降低它"


# ══════════════ 并列修正

def test_ties_are_corrected() -> None:
    """含并列时**必须做并列修正** —— 否则 W 会被高估。"""
    tied = [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [4.0, 4.0]]
    r = kendall_w(tied)
    assert r["tie_correction"] > 0, "未检测到并列"
    assert r["w"] == pytest.approx(1.0, abs=1e-9), r


def test_all_tied_yields_no_w_with_reason() -> None:
    """全部同分 → 分母为 0 → **`None` + 理由**，不是编造一个值。"""
    r = kendall_w([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    assert r["w"] is None
    assert r["reason"]


# ══════════════ 边界

def test_requires_at_least_two_raters() -> None:
    r = kendall_w([[1.0], [2.0], [3.0]])
    assert r["w"] is None and "评分者" in r["reason"]


def test_requires_at_least_two_items() -> None:
    r = kendall_w([[1.0, 2.0]])
    assert r["w"] is None and "对象" in r["reason"]


def test_rejects_ragged_rows() -> None:
    r = kendall_w([[1.0, 2.0], [3.0]])
    assert r["w"] is None and "缺分" in r["reason"]


def test_formula_is_reported() -> None:
    r = kendall_w([[1.0, 2.0], [3.0, 4.0]])
    assert "Kendall" in r["note"] or "W =" in r["formula"]
    assert "12" in r["formula"]


# ══════════════ 并报

def test_compare_reports_both_and_explains() -> None:
    r = compare_w_and_icc([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert "kendall_w" in r and "icc_2_1" in r
    assert "单调变换" in r["why_report_both"]
    assert "绝对一致性" in r["why_report_both"]
