"""`I17` 步骤 4 补遗（31007 6.2.4）的测试：决策可解释性 + 决策安全约束。

## 重点（三条诚实性）

1. **不可解释必须能被看见**：没有具名贡献 → `explained=False`，不假装已解释；
2. **未归属贡献要报出**，不能静默并进总数 —— 否则"算不出为什么"会伪装成"已解释"；
3. **`safe=None`（有检查无法执行）≠ `safe=True`** —— 混同会把"没查"当"查过了"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ai.explain import (  # noqa: E402
    explain_decision, check_safety, safety_summary,
)


# ══════════════════ ① 决策可解释性

def test_named_contributors_make_a_decision_explained() -> None:
    r = explain_decision(
        {"kind": "move", "score": 0.8},
        [{"kind": "hold", "score": 0.5}],
        contributors=[{"name": "situation_score", "value": 0.5},
                      {"name": "objective_score", "value": 0.3}],
        basis="learned",
    )
    assert r["explained"] is True
    assert r["basis"] == "learned"
    assert len(r["contributors"]) == 2


def test_margin_is_reported_and_flags_a_coin_flip() -> None:
    """**余量必须报出**：差额小意味着决策接近随机。"""
    r = explain_decision({"kind": "move", "score": 0.501},
                         [{"kind": "hold", "score": 0.500}],
                         contributors=[{"name": "a", "value": 0.501}])
    assert r["margin"] == pytest.approx(0.001)


def test_non_positive_margin_raises_a_warning() -> None:
    """首选分数不高于次选 → 选择依据存疑，必须给 warning。"""
    r = explain_decision({"kind": "move", "score": 0.4},
                         [{"kind": "hold", "score": 0.9}],
                         contributors=[{"name": "a", "value": 0.4}])
    assert r["margin"] < 0
    assert r["warnings"], "首选更差却无告警"


def test_unnamed_contributors_are_reported_not_absorbed() -> None:
    """**核心**：无名贡献进 `unattributed` 并**使解释不成立**。"""
    r = explain_decision(
        {"kind": "move", "score": 1.0},
        contributors=[{"name": "named", "value": 0.4}, {"value": 0.6}],
    )
    assert r["explained"] is False
    assert r["unattributed"], "无名贡献被静默吸收"
    assert "未归属" in r["reason"]


def test_no_contributors_means_not_explained() -> None:
    r = explain_decision({"kind": "move", "score": 1.0})
    assert r["explained"] is False
    assert "不可解释" in r["reason"]


def test_claimed_total_mismatch_is_reported() -> None:
    """具名之和与声称总值不符 → 差额必须报出（否则等于默认它说得对）。"""
    r = explain_decision(
        {"kind": "move", "score": 1.0},
        contributors=[{"name": "a", "value": 0.4}],
        required_contributions=1.0,
    )
    assert r["contributions_gap"] == pytest.approx(0.6)
    assert any("差" in str(u.get("reason")) for u in r["unattributed"])
    assert r["explained"] is False


def test_matching_total_keeps_explanation_valid() -> None:
    r = explain_decision(
        {"kind": "move", "score": 1.0},
        contributors=[{"name": "a", "value": 0.4}, {"name": "b", "value": 0.6}],
        required_contributions=1.0,
    )
    assert r["explained"] is True
    assert r["contributions_gap"] == pytest.approx(0.0)


def test_non_finite_contribution_is_not_absorbed() -> None:
    r = explain_decision(
        {"kind": "move", "score": 1.0},
        contributors=[{"name": "a", "value": float("nan")}],
    )
    assert r["contributors"] == []
    assert r["unattributed"]
    assert r["explained"] is False


def test_no_action_selected_is_handled() -> None:
    r = explain_decision(None)
    assert r["explained"] is False
    assert "未选出" in r["reason"]


def test_unknown_basis_is_normalised() -> None:
    r = explain_decision({"score": 1.0}, contributors=[{"name": "a", "value": 1.0}],
                         basis="bogus")
    assert r["basis"] == "unknown"


# ══════════════════ ② 决策安全约束

def test_self_target_is_a_hard_violation() -> None:
    r = check_safety({"obj_id": 5, "target_obj_id": 5, "type": 2})
    assert r["safe"] is False
    assert any(v["rule"] == "no_self_target" for v in r["violations"])


def test_non_controllable_actor_is_a_violation() -> None:
    r = check_safety({"obj_id": 9, "type": 2}, controllable_ops=[1, 2, 3])
    assert r["safe"] is False
    assert any(v["rule"] == "actor_controllable" for v in r["violations"])


def test_attack_without_weapon_is_a_violation() -> None:
    r = check_safety({"kind": "attack", "obj_id": 1, "target_obj_id": 2})
    assert r["safe"] is False
    assert any(v["rule"] == "weapon_available" for v in r["violations"])


def test_explicit_deny_hits_are_violations() -> None:
    r = check_safety({"kind": "withdraw", "obj_id": 1}, deny=["withdraw"])
    assert r["safe"] is False
    assert any(v["rule"] == "explicit_deny" for v in r["violations"])


def test_unexecuted_checks_yield_safe_none_not_true() -> None:
    """**核心**：有检查无法执行时 `safe=None` —— **不等于安全**。"""
    r = check_safety({"kind": "move"})     # 无 type、无可控列表
    assert r["safe"] is None, "无法判定时不得声称安全"
    assert r["unchecked"]
    assert "不足以判定安全" in r["reason"]


def test_full_checks_yield_a_definite_answer() -> None:
    """所有检查都能执行且无违反 → `safe=True`（这才是真的安全）。"""
    # ⚠ 夹具必须**满足 ACTION_SPECS**：type=1（移动）实测要求 ['obj_id','move_path']。
    # 我首版只给了 obj_id → 实现正确地报了"缺必填字段"，是**我的夹具不合法**，不是实现错。
    r = check_safety({"kind": "move", "obj_id": 1, "target_obj_id": 2, "type": 1,
                      "move_path": [101, 102]},
                     controllable_ops=[1, 2, 3], known_kinds=["move", "attack"])
    assert r["unchecked"] == [], r["unchecked"]
    assert r["safe"] is True, r["violations"]


def test_violations_and_warnings_are_separate() -> None:
    """**硬违反与可疑必须分开** —— 混同会让违规看起来只是警告。"""
    r = check_safety({"kind": "teleport", "obj_id": 1, "type": 1,
                      "move_path": [101]},
                     controllable_ops=[1], known_kinds=["move"])
    assert r["safe"] is True, r["violations"]
    assert r["warnings"], "未知类型应有告警"
    assert r["violations"] == []
    assert "混同" in r["note"]


def test_missing_required_fields_uses_action_specs() -> None:
    """必填字段检查**复用仓内 `ACTION_SPECS` 口径**（不另写一份规则）。"""
    r = check_safety({"kind": "attack", "type": 2, "obj_id": 1}, controllable_ops=[1])
    # 若 ACTION_SPECS 可查，则 attack 缺 weapon_id 应被报出
    if any("required_fields" == c for c in r["checked"]):
        assert any(v["rule"] == "required_fields" for v in r["violations"]), r
    else:
        # 取不到规格 → 必须记 unchecked，而不是默认通过
        assert any(u["rule"] == "required_fields" for u in r["unchecked"])


def test_none_action_is_unsafe() -> None:
    r = check_safety(None)
    assert r["safe"] is False


def test_non_dict_action_is_unsafe() -> None:
    r = check_safety("move")
    assert r["safe"] is False


# ══════════════════ 汇总

def test_safety_summary_separates_undecided_from_safe() -> None:
    """**汇总必须把"未判定"单独计数** —— 不能并进 safe。"""
    checks = [
        check_safety({"kind": "move", "obj_id": 1, "target_obj_id": 2, "type": 1,
                      "move_path": [101]},
                     controllable_ops=[1], known_kinds=["move"]),
        check_safety({"obj_id": 5, "target_obj_id": 5, "type": 2}, controllable_ops=[5]),
        check_safety({"kind": "move"}),                  # 未判定
    ]
    s = safety_summary(checks)
    assert s["n_actions"] == 3
    assert s["safe"] == 1
    assert s["unsafe"] == 1
    assert s["undecided"] == 1
    assert s["all_safe"] is False
    assert "undecided" in s["note"]


def test_safety_summary_all_safe_requires_every_action_safe() -> None:
    ok = check_safety({"kind": "move", "obj_id": 1, "target_obj_id": 2, "type": 1,
                       "move_path": [101]},
                      controllable_ops=[1], known_kinds=["move"])
    s = safety_summary([ok, ok])
    assert s["all_safe"] is True


def test_safety_summary_of_nothing_is_not_all_safe() -> None:
    """**空集合不得判为"全安全"** —— 那是把"没有数据"当"没有违规"。"""
    s = safety_summary([])
    assert s["all_safe"] is False
    assert s["n_actions"] == 0
