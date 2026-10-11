"""认知对齐度（`P3` 步骤 5 / 31007 6.2.4）的测试。

## 重点

1. **两个分量各自的口径**（LCS / 置信度加权）；
2. **"多报"要受罚**（否则刷满分）；
3. **无样本时返回 `None` 而非 0** —— "没数据" ≠ "对齐度为零"；
4. **缺一个分量时按可得分量归一化**，不因缺项被压低。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.alignment import (  # noqa: E402
    action_agreement, aggregate_alignment, cognitive_alignment, intent_match,
)


# ══════════════ 显性行动吻合度

def test_identical_sequences_score_one() -> None:
    assert action_agreement([1, 2, 3], [1, 2, 3]) == 1.0


def test_disjoint_sequences_score_zero() -> None:
    assert action_agreement([1, 2], [3, 4]) == 0.0


def test_order_matters() -> None:
    """`[A,B]` vs `[B,A]` 只得 1/2 —— 顺序敏感，不是集合比对。"""
    assert action_agreement(["A", "B"], ["B", "A"]) == pytest.approx(0.5)


def test_insertions_are_tolerated_via_lcs() -> None:
    """系统多插一个动作 → LCS 仍能识别大体一致（严格相等会得 0）。"""
    assert action_agreement([1, 9, 2, 3], [1, 2, 3]) == pytest.approx(0.75)


def test_over_reporting_is_penalised() -> None:
    """**多报必须受罚**：分母取 `max(len)` 而非 `len(人工)`。

    否则"把所有动作都报一遍"就能刷到 1.0。
    """
    human = [1, 2, 3]
    flooded = list(range(1, 31))
    assert action_agreement(flooded, human) < 0.2, "多报未被惩罚"


def test_empty_both_sides_returns_none() -> None:
    """两侧都空 → **`None`**（没数据），不是 0。"""
    assert action_agreement([], []) is None


def test_empty_one_side_scores_zero() -> None:
    """一侧有动作、另一侧没有 → 真的一致度为 0（这**是**有数据可判的情形）。"""
    assert action_agreement([1, 2], []) == 0.0
    assert action_agreement([], [1, 2]) == 0.0


# ══════════════ 隐性意图推理匹配度

def test_intent_all_match_scores_one() -> None:
    assert intent_match(["A", "B"], ["A", "B"]) == 1.0


def test_intent_none_match_scores_zero() -> None:
    assert intent_match(["A", "B"], ["C", "D"]) == 0.0


def test_confidence_weighting_changes_the_score() -> None:
    """**置信度加权**：低置信的正确 vs 高置信的错误，得分应不同。"""
    # 第 1 位正确(低置信)、第 2 位错误(高置信)
    low_conf_right = intent_match(["A", "B"], ["A", "X"], confidences=[0.2, 0.8])
    # 反之：第 1 位错误(低置信)、第 2 位正确(高置信)
    high_conf_right = intent_match(["A", "B"], ["X", "B"], confidences=[0.2, 0.8])
    assert low_conf_right == pytest.approx(0.2)
    assert high_conf_right == pytest.approx(0.8)
    assert high_conf_right > low_conf_right


def test_missing_confidences_degenerates_to_hit_rate() -> None:
    assert intent_match(["A", "B", "C"], ["A", "X", "C"]) == pytest.approx(2.0 / 3.0)


def test_zero_total_confidence_returns_none() -> None:
    """全零置信度 → **`None`**（无法加权），不是 0。"""
    assert intent_match(["A"], ["A"], confidences=[0.0]) is None


def test_length_mismatch_uses_the_shorter() -> None:
    """长度不等时按较短者统计，**不擅自补齐**。"""
    assert intent_match(["A", "B"], ["A", "B", "C"]) == 1.0
    assert intent_match(["A", "B", "C"], ["A", "B"]) == 1.0


def test_negative_or_nonfinite_confidence_is_zeroed() -> None:
    """负值/非有限置信度按 0 处理 —— 不让异常值放大或缩小结果。"""
    assert intent_match(["A"], ["A"], confidences=[-1.0]) is None
    assert intent_match(["A", "B"], ["A", "B"], confidences=[float("nan"), 1.0]) == 1.0


# ══════════════ 合成

def test_alignment_combines_both_components() -> None:
    r = cognitive_alignment([1, 2], [1, 2], ["A"], ["A"])
    assert r["value"] == pytest.approx(1.0)
    assert r["explicit_action_agreement"] == 1.0
    assert r["implicit_intent_match"] == 1.0


def test_missing_component_is_renormalised_not_penalised() -> None:
    """**只有动作、没有意图**时，不得因为缺项而把总分压低。

    若不做归一化，`0.5·1.0 + 0.5·(缺)` 会算成 0.5 —— 那是把"没测"当成"没对齐"。
    """
    r = cognitive_alignment([1, 2], [1, 2], [], [])
    assert r["implicit_intent_match"] is None
    assert r["value"] == pytest.approx(1.0), "缺分量时未按可得分量归一化"
    assert "仅一个分量" in (r["reason"] or "")


def test_no_data_returns_none_not_zero() -> None:
    """**两侧都无数据 → `None`**，绝不返回 0 冒充"对齐度为零"。"""
    r = cognitive_alignment([], [], [], [])
    assert r["value"] is None
    assert r["reason"], "未说明为何取不到"


def test_weights_are_declared_in_the_output() -> None:
    """**权重必须在输出里声明** —— 标准未规定权重，报告要注明。"""
    r = cognitive_alignment([1], [1], ["A"], ["A"])
    assert r["weights_declared"] == {"explicit": 0.5, "implicit": 0.5}
    assert "自定" in r["note"]


def test_custom_weights_are_honoured() -> None:
    r = cognitive_alignment([1, 2], [1], ["A"], ["X"],
                            weights={"explicit": 1.0, "implicit": 0.0})
    # explicit = LCS([1,2],[1]) / max(2,1) = 1/2；implicit = 0
    assert r["value"] == pytest.approx(0.5)


# ══════════════ 聚合

def test_aggregate_reports_excluded_episodes() -> None:
    """聚合只能统计有值的局，且**如实报出被排除的局数**。"""
    per = [{"value": 0.8}, {"value": 0.9}, {"value": None}]
    r = aggregate_alignment(per)
    assert r["n"] == 2
    assert r["episodes_total"] == 3
    assert r["episodes_excluded_missing_value"] == 1
    assert r["mean"] == pytest.approx(0.85)


def test_aggregate_gives_a_ci_when_enough_samples() -> None:
    per = [{"value": v} for v in (0.5, 0.6, 0.7, 0.8)]
    r = aggregate_alignment(per)
    assert r["ci_low"] is not None and r["ci_high"] is not None
    assert r["ci_low"] < r["mean"] < r["ci_high"]
