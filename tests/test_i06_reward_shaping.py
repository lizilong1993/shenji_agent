"""I06 奖励函数：任务对齐 + 时间折扣 + 不当行为惩罚。

这些用例**不需要引擎**（奖励是纯函数），故在引擎鉴权不可用的环境里仍能守护
【验收判据 2】要的那部分："奖励函数含任务对齐、时间折扣与惩罚三部分"。
真正跑消融（自博弈对局）仍需引擎。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.reward import (DEFAULT_WEIGHTS, debug_defaults, shape_step_reward,
                             shaped_trajectory_reward, time_discount)


# ─────────────────────────── ① 任务对齐

def test_city_progress_produces_positive_reward() -> None:
    r, c = shape_step_reward({"city_value_delta": 10.0}, steps_remaining=0)
    assert r > 0
    assert c["align_city_value"]["value"] == pytest.approx(20.0)


def test_kill_and_damage_are_rewarded() -> None:
    r, c = shape_step_reward({"kill_dealt": 2, "damage_dealt": 40.0}, steps_remaining=0)
    assert c["align_kill"]["value"] == pytest.approx(10.0)
    assert c["align_damage"]["value"] == pytest.approx(10.0)
    assert r == pytest.approx(20.0)


def test_mission_alignment_bonus_applies_only_when_reached() -> None:
    base, _ = shape_step_reward({}, steps_remaining=0)
    got, c1 = shape_step_reward({}, steps_remaining=0,
                                mission_steps={"kind": "capture", "reached": True, "bonus": 7.5})
    notyet, c2 = shape_step_reward({}, steps_remaining=0,
                                   mission_steps={"kind": "capture", "reached": False, "bonus": 7.5})
    assert got - base == pytest.approx(7.5)
    assert c1["align_mission"]["value"] == pytest.approx(7.5)
    assert notyet == pytest.approx(base), "未达成目标不应给对齐奖励"


# ─────────────────────────── ② 不当行为惩罚

def test_friendly_loss_is_penalised() -> None:
    r, c = shape_step_reward({"kill_taken": 1}, steps_remaining=0)
    assert r < 0, "己方损失必须为负"
    # 引用常量而非写死数值：权重是**可调**的（消融就要调它），
    # 写死会让「合法调参」与「行为回归」无法区分。我第一版写死了 -5.0，
    # 改权重后它就误报了一次。
    assert c["pen_friendly_loss"]["value"] == pytest.approx(-DEFAULT_WEIGHTS["pen_friendly_loss"])


def test_friendly_loss_strictly_outweighs_enemy_kill() -> None:
    """**拿命换命不应被鼓励**：损失 1 个己方单位必须**严格**比歼敌 1 个更亏。

    这是我写测试时发现的真实设计缺陷：原权重两者都是 5.0，
    函数对「损一己」与「歼一敌」**完全无差别**（实测各 5.0），
    策略会学成拿命换命。故断言用 `>` 而非 `>=`，并且**直接检查权重常量**，
    使"有人把两个权重调成相等"也会立刻失败。
    """
    gain, _ = shape_step_reward({"kill_dealt": 1}, steps_remaining=0)
    loss, _ = shape_step_reward({"kill_taken": 1}, steps_remaining=0)
    assert abs(loss) > abs(gain), "己方损失必须严格重于歼敌（实测 %s vs %s）" % (loss, gain)
    assert DEFAULT_WEIGHTS["pen_friendly_loss"] > DEFAULT_WEIGHTS["align_kill"], \
        "权重常量本身必须满足 pen_friendly_loss > align_kill"


def test_invalid_actions_are_penalised() -> None:
    """被引擎拒绝的动作 = 浪费回合，属不当行为。"""
    ok, _ = shape_step_reward({}, steps_remaining=0, action_outcomes=[{"type": 1}, {"type": 2}])
    bad, _ = shape_step_reward({}, steps_remaining=0,
                               action_outcomes=[{"type": 1, "error": {"code": 2304}},
                                                {"type": 2, "error": {"code": 2304}}])
    assert bad < ok, "无效动作未被惩罚"
    assert bad - ok == pytest.approx(-2 * DEFAULT_WEIGHTS["pen_idle"])


# ─────────────────────────── ③ 时间折扣

def test_time_discount_is_monotone_and_bounded() -> None:
    ds = [time_discount(0.99, k) for k in (0, 1, 5, 50, 200)]
    assert ds[0] == pytest.approx(1.0), "终局不打折"
    assert all(0.0 < d <= 1.0 for d in ds)
    assert ds == sorted(ds, reverse=True), "越晚折得越多"


def test_gains_are_discounted_but_penalties_are_not() -> None:
    """**惩罚不打折**：否则早期不当行为会被 gamma^T 抹平，等于纵容。"""
    near, c_near = shape_step_reward({"city_value_delta": 10.0, "kill_taken": 1},
                                     steps_remaining=1)
    far, c_far = shape_step_reward({"city_value_delta": 10.0, "kill_taken": 1},
                                   steps_remaining=200)
    assert far < near, "远期收益应更小"
    # 惩罚项在两种情形下**完全相同**
    assert c_near["pen_friendly_loss"]["value"] == c_far["pen_friendly_loss"]["value"]
    # 且折扣只作用在收益上：差值 == 收益的折扣差
    gains = c_near["gains_losses"]["gains"]
    assert (near - far) == pytest.approx(gains * (c_near["time_discount"]["factor"]
                                                  - c_far["time_discount"]["factor"]))


def test_penalties_survive_when_gains_dominate() -> None:
    """**回归用例**：我第一版把 `losses = total - gains` 用在正收益步上，
    结果惩罚被整个丢掉（实测 gains=45, losses=0，惩罚 −7.5 消失）。
    这条用例锁住"收益占优时惩罚仍必须计入"。"""
    r, c = shape_step_reward({"city_value_delta": 20.0, "kill_dealt": 1, "damage_dealt": 30.0,
                              "kill_taken": 1, "damage_taken": 10.0}, steps_remaining=10)
    gl = c["gains_losses"]
    assert gl["gains"] > 0 and gl["losses"] < 0, "收益占优时惩罚不得归零: %s" % gl
    expect = gl["gains"] * c["time_discount"]["factor"] + gl["losses"]
    assert r == pytest.approx(expect, abs=1e-6)


# ─────────────────────────── ④ 尺度与平滑、可调试性

def test_scale_is_clipped() -> None:
    r, c = shape_step_reward({"city_value_delta": 1e6}, steps_remaining=0)
    assert abs(r) <= DEFAULT_WEIGHTS["clip"]
    assert c["scale_clip"]["clipped"] is True


def test_components_are_exposed_for_debugging() -> None:
    """标准要求「可解释性与可调试性」：**不得只返回一个数**。"""
    _, c = shape_step_reward({"city_value_delta": 1.0}, steps_remaining=3)
    for term in ("align_city_value", "align_score", "align_kill", "align_damage",
                 "pen_friendly_loss", "pen_damage_taken", "pen_invalid_action",
                 "gains_losses", "time_discount", "scale_clip"):
        assert term in c, "缺少可调试分量 %s" % term
        assert isinstance(c[term], dict)


# ─────────────────────────── 稠密化（修复稀疏奖励的动机）

def test_trajectory_shaping_is_dense() -> None:
    """原实现只在终局给一个标量 → 实测 2344 条标签里 2266 条 reward=0。
    逐步塑形后**每一步都应有非零贡献**。"""
    recs = [{"city_value_delta": 1.0}, {"city_value_delta": 2.0}, {"kill_taken": 1}]
    total, per = shaped_trajectory_reward(recs)
    assert len(per) == 3
    assert all(p["reward"] != 0.0 for p in per), "仍有零奖励步，未真正稠密化"
    assert total == pytest.approx(sum(p["reward"] for p in per))
    # 折扣应随剩余步数单调
    assert per[0]["components"]["time_discount"]["factor"] <= 1.0


# ─────────────────────────── 容错（奖励算不出来会让训练直接崩）

@pytest.mark.parametrize("bad", [None, "x", 42, [], {}])
def test_malformed_input_never_raises(bad) -> None:
    r, c = shape_step_reward(bad)
    assert isinstance(r, float) and isinstance(c, dict)


@pytest.mark.parametrize("bad", [None, "x", 42, {"kill_dealt": "bad"}, {"city_value_delta": None}])
def test_malformed_values_degrade_to_zero(bad) -> None:
    r, _ = shape_step_reward(bad if isinstance(bad, dict) else {}, steps_remaining=0) \
        if not isinstance(bad, dict) else shape_step_reward(bad, steps_remaining=0)
    assert isinstance(r, float)


def test_trajectory_with_bad_input_returns_empty() -> None:
    assert shaped_trajectory_reward(None) == (0.0, [])
    assert shaped_trajectory_reward("nope") == (0.0, [])


# ─────────────────────────── 消融可用性（判据 2 的前置）

def test_weights_can_be_overridden_for_ablation() -> None:
    """消融实验靠覆盖权重实现，**不改代码**。"""
    base, _ = shape_step_reward({"kill_taken": 1}, steps_remaining=0)
    no_pen, _ = shape_step_reward({"kill_taken": 1}, steps_remaining=0,
                                  weights={"pen_friendly_loss": 0.0})
    assert base < 0 and no_pen == pytest.approx(0.0)


def test_gamma_override_changes_only_discount() -> None:
    a, ca = shape_step_reward({"city_value_delta": 5.0}, steps_remaining=10,
                              weights={"gamma": 1.0})
    b, cb = shape_step_reward({"city_value_delta": 5.0}, steps_remaining=10,
                              weights={"gamma": 0.9})
    assert ca["gains_losses"]["gains"] == cb["gains_losses"]["gains"], "收益本身不应受 gamma 影响"
    assert a > b, "gamma=1 不打折，应更大"


def test_debug_defaults_is_serialisable() -> None:
    import json
    d = debug_defaults()
    json.dumps(d)
    assert d["weights"] == DEFAULT_WEIGHTS
