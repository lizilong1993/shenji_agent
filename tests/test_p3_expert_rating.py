"""`P3` 步骤 4 的测试：专家评分聚合 + ICC 一致性 + 规则冲突检测。

## 重点

1. **加权总分的归一化**：缺维**退出分母**，不把"没评"当"评了低分"；
2. **ICC 的边界**：评分者 <2 / 目标 <2 / 行内有缺 → **`None` + 理由**，不编造；
3. **ICC 的方向性**：评分完全一致 → 接近 1；评分完全无关 → 接近 0 或负；
4. **规则冲突**：只在**同主体同目标**上判，不同主体不得误报。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.expert_rating import (  # noqa: E402
    DIMENSION_WEIGHTS, DIMENSIONS, aggregate_expert_scores, icc_2_1,
    rule_conflicts, validate_score, weighted_total,
)


def _full(v: float) -> dict:
    return {k: v for k in DIMENSIONS}


# ══════════════ 量程校验

def test_validate_score_accepts_within_range() -> None:
    assert validate_score(0)[0] == 0.0
    assert validate_score(10)[0] == 10.0
    assert validate_score(7.5)[0] == 7.5


def test_validate_score_rejects_out_of_range_and_non_numeric() -> None:
    """越界/非数值必须**报出理由**，不是静默接受。"""
    for bad in (10.1, -0.5, "abc", None, float("nan"), float("inf")):
        v, why = validate_score(bad)
        assert v is None and why, bad


# ══════════════ 加权总分

def test_all_dimensions_equal_gives_that_score() -> None:
    r = weighted_total(_full(8.0))
    assert r["total"] == pytest.approx(8.0)
    assert r["dimensions_missing"] == []


def test_missing_dimension_leaves_the_denominator() -> None:
    """**缺维退出分母** —— 否则漏打一维会把总分压低（把"没评"当"低分"）。"""
    scores = {k: 8.0 for k in DIMENSIONS if k != "timing"}
    r = weighted_total(scores)
    assert r["total"] == pytest.approx(8.0), "缺维被当成 0 分压低了总分"
    assert r["dimensions_missing"] == ["timing"]


def test_invalid_dimension_is_reported_and_excluded() -> None:
    scores = _full(8.0)
    scores["risk_control"] = 99          # 越界
    r = weighted_total(scores)
    assert r["total"] == pytest.approx(8.0)
    assert r["dimensions_invalid"][0]["dimension"] == "risk_control"
    assert "risk_control" not in r["dimensions_used"]


def test_no_valid_dimension_returns_none() -> None:
    r = weighted_total({"mission_fit": -1})
    assert r["total"] is None
    assert r["reason"]


def test_weights_are_declared() -> None:
    r = weighted_total(_full(5.0))
    assert r["weights_declared"] == DIMENSION_WEIGHTS
    assert "自定" in r["note"]


def test_legality_dimension_carries_weight() -> None:
    """`legality` 是**否决性**维度，应带权重（本模块自定 0.20）。"""
    assert DIMENSION_WEIGHTS["legality"] > 0


# ══════════════ ICC(2,1)

def test_perfect_agreement_gives_icc_near_one() -> None:
    rows = [[5.0, 5.0], [7.0, 7.0], [3.0, 3.0], [9.0, 9.0]]
    r = icc_2_1(rows)
    assert r["icc"] == pytest.approx(1.0, abs=1e-9), r


def test_uncorrelated_ratings_give_a_low_or_negative_icc() -> None:
    """评分者之间**无关甚至相反**时 ICC 应低（可能为负）—— 不得恒为高值。"""
    rows = [[9.0, 1.0], [8.0, 2.0], [2.0, 8.0], [1.0, 9.0]]
    r = icc_2_1(rows)
    assert r["icc"] is not None and r["icc"] < 0.2, r


def test_consistent_offset_lowers_icc_under_absolute_agreement() -> None:
    """**绝对一致性口径**：评分者系统性偏高（+2）**也应降低** ICC。

    若用"一致性"(consistency) 口径则不会降 —— 故本条把口径钉住。
    """
    rows = [[5.0, 7.0], [6.0, 8.0], [4.0, 6.0], [7.0, 9.0]]
    r = icc_2_1(rows)
    assert r["icc"] is not None and r["icc"] < 0.9, "系统性偏移未被计入（口径不是绝对一致性）"


def test_icc_requires_at_least_two_raters() -> None:
    r = icc_2_1([[1.0], [2.0], [3.0]])
    assert r["icc"] is None and "评分者" in r["reason"]


def test_icc_requires_at_least_two_targets() -> None:
    r = icc_2_1([[1.0, 2.0]])
    assert r["icc"] is None and "目标" in r["reason"]


def test_icc_rejects_ragged_rows() -> None:
    """行内有缺分 → **拒绝计算并说明**（不静默按短行处理）。"""
    r = icc_2_1([[1.0, 2.0], [3.0]])
    assert r["icc"] is None and "缺分" in r["reason"]


def test_icc_reports_the_formula_and_no_verdict() -> None:
    """必须回带**所用公式**，且**不给合格判定**（阈值解释属惯例）。"""
    r = icc_2_1([[1.0, 2.0], [3.0, 4.0]])
    assert "ICC(2,1)" in r["formula"]
    assert "惯例" in r["note"]
    for k in ("MSR", "MSC", "MSE"):
        assert k in r


# ══════════════ 规则冲突

def test_conflicting_pair_on_same_actor_target_is_detected() -> None:
    acts = [{"kind": "occupy", "actor": 1, "target": "A"},
            {"kind": "withdraw", "actor": 1, "target": "A"}]
    hits = rule_conflicts(acts)
    assert len(hits) == 1
    assert hits[0]["rule"] == "occupy_vs_withdraw"
    assert hits[0]["index_a"] == 0 and hits[0]["index_b"] == 1


def test_different_actors_are_not_a_conflict() -> None:
    """**不同主体**做相反动作**不是**冲突（是正常对抗）。"""
    acts = [{"kind": "occupy", "actor": 1, "target": "A"},
            {"kind": "withdraw", "actor": 2, "target": "A"}]
    assert rule_conflicts(acts) == []


def test_different_targets_are_not_a_conflict() -> None:
    acts = [{"kind": "occupy", "actor": 1, "target": "A"},
            {"kind": "withdraw", "actor": 1, "target": "B"}]
    assert rule_conflicts(acts) == []


def test_no_conflict_when_only_one_side_present() -> None:
    assert rule_conflicts([{"kind": "occupy", "actor": 1, "target": "A"}]) == []


def test_attack_and_hold_fire_conflict() -> None:
    acts = [{"kind": "attack", "actor": 9, "target": 3},
            {"kind": "hold_fire", "actor": 9, "target": 3}]
    assert rule_conflicts(acts)[0]["rule"] == "attack_vs_hold_fire"


def test_conflicts_report_both_sides_for_human_review() -> None:
    """冲突必须**给出双方索引**（便于人工复核），**不自动判定谁错**。"""
    acts = [{"kind": "engage", "actor": 1, "target": 1},
            {"kind": "disengage", "actor": 1, "target": 1}]
    h = rule_conflicts(acts)[0]
    assert set(h) >= {"rule", "index_a", "index_b", "actor", "target"}


# ══════════════ 多人多目标聚合

def test_aggregate_combines_raters_and_computes_icc() -> None:
    data = {
        "expert_1": {"t1": _full(8.0), "t2": _full(4.0), "t3": _full(6.0)},
        "expert_2": {"t1": _full(8.5), "t2": _full(3.5), "t3": _full(6.5)},
    }
    r = aggregate_expert_scores(data)
    assert r["n_targets"] == 3
    assert r["per_target"]["t1"]["total_mean"] == pytest.approx(8.25)
    assert r["icc"]["icc"] is not None
    assert r["icc"]["icc"] > 0.9, r["icc"]


def test_targets_with_partial_raters_are_excluded_from_icc() -> None:
    """**缺分目标退出 ICC 并登记理由** —— 用均值填补会人为抬高一致性。

    ⚠ 我第一版断言 `icc is not None`，**那是错的**：本例只剩 **1 个**双评目标，
    而 ICC 需要 ≥2 个才有目标间方差 → 返回 `None` **才是正确行为**。
    故改为断言"目标被正确排除"+"剩余不足时如实给理由"。
    """
    data = {
        "expert_1": {"t1": _full(8.0), "t2": _full(4.0)},
        "expert_2": {"t1": _full(8.0)},                 # t2 只有一人评
    }
    r = aggregate_expert_scores(data)
    assert r["per_target"]["t2"]["total_mean"] == pytest.approx(4.0)
    # t2 被排除（只有 1 名评分者）
    assert [e["target"] for e in r["excluded_from_icc"]] == ["t2"]
    # 排除后只剩 t1 一个目标 → ICC 无法计算，且**必须给出理由**
    assert r["icc"] is None
    assert r["icc_reason"]
    assert "填补" in r["note"]


def test_icc_computed_when_enough_fully_rated_targets() -> None:
    """**两个以上目标都有双评时，ICC 正常算出** —— 与上一条形成对照。"""
    data = {
        "expert_1": {"t1": _full(8.0), "t2": _full(4.0), "t3": _full(6.0)},
        "expert_2": {"t1": _full(8.0), "t2": _full(4.0), "t3": _full(6.0)},
    }
    r = aggregate_expert_scores(data)
    assert r["excluded_from_icc"] == []
    assert r["icc"] is not None and r["icc"]["icc"] == pytest.approx(1.0, abs=1e-9)


def test_insufficient_data_returns_icc_reason() -> None:
    data = {"e1": {"t1": _full(5.0)}}
    r = aggregate_expert_scores(data)
    assert r["icc"] is None
    assert r["icc_reason"]
