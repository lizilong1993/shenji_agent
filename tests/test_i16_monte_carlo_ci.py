"""`I16` 判据 2 的**聚合侧**测试：置信区间必须算得对、且**算不出时如实说**。

## 判据原文

> 2) 蒙特卡洛 N≥30 且**输出置信区间**。

## 重点：校准性 + 诚实性

**不只测"能给个数"，而是测三件事：**

1. **区间覆盖**：对已知分布的样本，区间应覆盖真值（用固定样本验证端点公式）；
2. **t vs 正态**：小样本时 t 区间必须**比正态更宽**（否则就是算错）；
3. **算不出时如实说**：`n<2` / 全相同值时**不编造区间**，给 `None` + 理由。

**第 3 条最重要** —— 它的反例正是本会话反复出现的"拿形式当内容"。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.monte_carlo import (  # noqa: E402
    aggregate, aggregate_by_case, overall_summary, t_critical,
)

Z95 = 1.959964


# ══════════════ ① 端点公式

def test_ci_endpoints_match_the_t_formula() -> None:
    """区间端点必须等于 `mean ± t·s/√n` —— 手算一组并与实现比对。"""
    xs = [10.0, 12.0, 11.0, 13.0, 9.0, 12.0, 11.0, 10.0]
    r = aggregate(xs, min_n=1)
    n = len(xs)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    t = t_critical(n - 1)
    half = t * sd / math.sqrt(n)
    assert r["mean"] == pytest.approx(mean, rel=1e-12)
    assert r["stdev"] == pytest.approx(sd, rel=1e-12)
    assert r["ci_low"] == pytest.approx(mean - half, rel=1e-12)
    assert r["ci_high"] == pytest.approx(mean + half, rel=1e-12)


def test_ci_is_symmetric_about_the_mean() -> None:
    r = aggregate([1.0, 2.0, 3.0, 4.0, 5.0], min_n=1)
    assert r["ci_low"] + r["ci_high"] == pytest.approx(2 * r["mean"], rel=1e-12)


def test_constant_samples_give_a_degenerate_interval() -> None:
    """所有样本相同 → 区间收缩为一点，且**方法名标明退化**，不假装是真区间。"""
    r = aggregate([7.0] * 40, min_n=30)
    assert r["stdev"] == 0.0
    assert r["ci_low"] == r["ci_high"] == 7.0
    assert "degenerate" in r["ci_method"]


# ══════════════ ② t vs 正态（小样本必须更宽）

def test_t_interval_is_wider_than_normal_for_small_n() -> None:
    """**校准性核心**：N 小时 t 区间必须比正态**更宽** —— 否则就是用了错的公式。"""
    xs = [10.0, 12.0, 11.0, 13.0, 9.0]          # n=5
    r = aggregate(xs, min_n=1)
    n = len(xs)
    half_t = r["ci_half_width"]
    half_z = Z95 * r["stdev"] / math.sqrt(n)
    assert half_t > half_z, "小样本用了正态临界值（区间偏窄 → 假显著）"


def test_t_converges_to_normal_at_large_n() -> None:
    """N 大时 t 收敛到正态（差 <5%）—— 证明用的是同一族，不是另一套公式。"""
    xs = [float(i % 7) for i in range(200)]
    r = aggregate(xs, min_n=1)
    n = len(xs)
    half_t = r["ci_half_width"]
    half_z = Z95 * r["stdev"] / math.sqrt(n)
    assert abs(half_t - half_z) / half_z < 0.05


def test_ci_shrinks_as_n_grows() -> None:
    """区间宽度应随 N 增大而收窄（√N 律）—— 否则 N 的意义不成立。"""
    base = [1.0, 3.0, 2.0, 4.0, 2.0, 3.0]
    small = aggregate(base * 2, min_n=1)["ci_half_width"]
    big = aggregate(base * 20, min_n=1)["ci_half_width"]
    assert big < small


# ══════════════ ③ 算不出时**如实说**（最重要）

def test_single_sample_yields_no_interval_and_says_why() -> None:
    """`n=1` **不能**给区间 —— 必须 `None` + 理由，**不编造**。"""
    r = aggregate([5.0], min_n=30)
    assert r["ci_low"] is None and r["ci_high"] is None
    assert r["ci_unavailable_reason"], "未说明为何没有区间"
    assert r["ci_reliable"] is False
    assert r["mean"] == 5.0, "均值仍应给出（它是可算的）"


def test_empty_input_yields_nothing_but_does_not_crash() -> None:
    r = aggregate([], min_n=30)
    assert r["n"] == 0 and r["mean"] is None
    assert r["ci_unavailable_reason"]
    assert r["meets_min_n"] is False


def test_non_finite_values_are_dropped_and_counted() -> None:
    """`nan`/`inf` 必须被丢弃并**计数** —— 静默丢弃会让 N 与区间都对不上。"""
    r = aggregate([1.0, float("nan"), 2.0, float("inf"), 3.0], min_n=1)
    assert r["n"] == 3
    assert r["dropped_non_finite"] == 2
    assert r["n_input"] == 5


# ══════════════ ④ N≥30 的判据口径

def test_meets_min_n_is_false_below_threshold() -> None:
    assert aggregate([1.0] * 29, min_n=30)["meets_min_n"] is False
    assert aggregate([1.0] * 30, min_n=30)["meets_min_n"] is True


def test_min_n_is_reported_back() -> None:
    """必须回报要求的 N —— 否则报告里无法说明"是否达标"。"""
    r = aggregate([1.0, 2.0], min_n=30)
    assert r["min_n_required"] == 30
    assert r["meets_min_n"] is False


# ══════════════ ⑤ 独立性（重复 seed → 区间不可信）

def test_duplicate_seeds_make_the_ci_unreliable() -> None:
    """**独立性检查**：同一 seed 跑 N 次**不是** N 个独立样本。

    若不检查，报告会写"95% 置信区间"而样本实际只有 1 个独立观测。
    """
    xs = [1.0, 2.0, 3.0, 4.0]
    r = aggregate(xs, seeds=[7, 7, 7, 7], min_n=1)
    assert r["seed_duplicates"], "未报出重复 seed"
    assert r["ci_reliable"] is False, "重复 seed 时不得声称区间可靠"


def test_distinct_seeds_keep_the_ci_reliable() -> None:
    r = aggregate([1.0, 2.0, 3.0], seeds=[1, 2, 3], min_n=1)
    assert r["seed_duplicates"] == []
    assert r["ci_reliable"] is True


# ══════════════ ⑥ 分组与汇总

def test_aggregate_by_case_groups_and_uses_seeds() -> None:
    rows = []
    for cid, base in (("A", 10.0), ("B", 20.0)):
        for i, s in enumerate(range(5)):
            rows.append({"case_id": cid, "seed": 100 + i, "score": base + i * 0.1})
    out = aggregate_by_case(rows, metric="score", min_n=5)
    assert set(out) == {"A", "B"}
    assert out["A"]["n"] == 5
    assert out["B"]["mean"] > out["A"]["mean"]


def test_overall_summary_flags_incomplete_runs() -> None:
    """汇总必须**如实报出**未达标/不可靠的用例数 —— 不把 3/30 说成达标。"""
    rows = []
    for i in range(3):                       # 只跑 3 次，低于 N=30
        rows.append({"case_id": "A", "seed": i, "score": 1.0 + i})
    per = aggregate_by_case(rows, metric="score", min_n=30)
    s = overall_summary(per)
    assert s["cases"] == 1
    assert s["cases_meeting_min_n"] == 0
    assert s["all_meet_min_n"] is False


def test_overall_summary_true_when_all_met() -> None:
    rows = [{"case_id": "A", "seed": i, "score": float(i)} for i in range(30)]
    s = overall_summary(aggregate_by_case(rows, metric="score", min_n=30))
    assert s["all_meet_min_n"] is True
    assert s["all_ci_reliable"] is True
