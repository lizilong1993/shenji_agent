"""I06 步骤 4 的第二轮：按 **31007 5.3.3 原文** 逐条核对奖励函数。

为什么会有这一轮
----------------
第五十轮我按**索引摘要**实现了奖励函数并标了「✓」。第五十五轮找到原文
（`.scratch/std/s31007.txt`，5.3.3 见 `:507-528`）后逐条比对，发现**四项没做**：
  - b) 最终胜负的长期奖励项 —— 完全没有；
  - b) 阶段性成果的小额奖励 —— 完全没有；
  - c) 友军误伤 —— 完全没有；
  - c) 消极拖延 —— `pen_idle` 存在但**未接线**（没有任何调用点传 `idle_steps`）。

还发现一处**权重错用**：`pen_invalid_action` 用的是 `pen_idle` 的权重，
于是"引擎拒收"与"消极拖延"两个概念共用同一个数，**无法独立调参** ——
而标准 e) 恰恰要求"根据各分量贡献调整奖励参数"。

本文件把原文五条（a–e）落成可执行断言。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.reward import DEFAULT_WEIGHTS, shape_step_reward  # noqa: E402

STD = REPO.parent.parent / ".scratch" / "std" / "s31007.txt"


# ───────────────── a) 任务目标对齐（原文 :510-512）

def test_a_positive_outcomes_are_rewarded() -> None:
    """「歼灭对方兵力、达成占领目标等正面成果应赋予正向奖励」。"""
    r_kill, _ = shape_step_reward({"kill_dealt": 1}, steps_remaining=0)
    r_city, _ = shape_step_reward({"city_value_delta": 1.0}, steps_remaining=0)
    assert r_kill > 0 and r_city > 0


def test_a_own_losses_are_negative() -> None:
    """「己方损失……应给予负向奖励」。"""
    r, _ = shape_step_reward({"kill_taken": 1}, steps_remaining=0)
    assert r < 0


# ───────────────── b) 及时性与长期性平衡（原文 :513-516）

def test_b_terminal_outcome_component_exists_and_dominates() -> None:
    """「应以最终结果的长期奖励为主」。

    判据：终局项存在，且在典型分差下**量级大于**过程项之和。
    """
    assert "terminal_outcome" in DEFAULT_WEIGHTS
    r_term, c_term = shape_step_reward({}, steps_remaining=0, terminal_margin=10.0)
    r_proc, _ = shape_step_reward({"city_value_delta": 1.0, "kill_dealt": 1},
                                  steps_remaining=0, milestones=[1, 2, 3])
    assert abs(c_term["terminal_outcome"]["value"]) > abs(r_proc), \
        "长期项应为主：终局项 %s vs 过程项 %s" % (r_term, r_proc)


def test_b_terminal_outcome_is_not_silently_clipped() -> None:
    """**终局项不得被 `clip` 削掉** —— 否则"长期为主"名存实亡。

    这是我第一版的实际错误：权重 10.0 × 分差 10 = 100，而 clip=50 → 被裁成 50。
    故断言：典型分差（±10）下 `clipped` 必须为 False。
    """
    for margin in (10.0, -10.0):
        _, c = shape_step_reward({}, steps_remaining=0, terminal_margin=margin)
        assert c["scale_clip"]["clipped"] is False, \
            "分差 %s 时终局项被裁剪：%s" % (margin, c["scale_clip"])
    # 权重与 clip 必须配套
    assert DEFAULT_WEIGHTS["terminal_outcome"] * 10 <= DEFAULT_WEIGHTS["clip"], \
        "terminal_outcome × 典型分差 超出 clip，长期项会被削"


def test_b_terminal_outcome_is_not_discounted() -> None:
    """长期项不打折：它是"最终结果"，按 `gamma^t` 折会把长期信号变小。"""
    near, _ = shape_step_reward({}, steps_remaining=0, terminal_margin=5.0)
    far, _ = shape_step_reward({}, steps_remaining=200, terminal_margin=5.0)
    assert near == pytest.approx(far), "终局项被折扣了：%s vs %s" % (near, far)


def test_b_milestones_give_a_small_positive_reward() -> None:
    """「对阶段性成果……给予**小额**奖励」。"""
    assert "milestone_bonus" in DEFAULT_WEIGHTS
    r, c = shape_step_reward({}, steps_remaining=0, milestones=[1, 2, 3])
    assert c["milestone_bonus"]["value"] > 0
    # "小额"：必须显著小于终局项
    _, c_term = shape_step_reward({}, steps_remaining=0, terminal_margin=10.0)
    assert abs(c["milestone_bonus"]["value"]) < abs(c_term["terminal_outcome"]["value"]) / 5, \
        "阶段性成果奖励不够'小额'"


def test_b_no_milestones_means_no_bonus() -> None:
    """不给阶段性成果时不得凭空加分。"""
    r, c = shape_step_reward({}, steps_remaining=0)
    assert "milestone_bonus" not in c or c["milestone_bonus"]["value"] == 0


# ───────────────── c) 惩罚不当行为（原文 :517-519）

def test_c_friendly_fire_is_penalised() -> None:
    """原文举例「友军误伤给予负奖励」—— 首版**完全没有**这一项。"""
    assert "pen_friendly_fire" in DEFAULT_WEIGHTS
    r, c = shape_step_reward({}, steps_remaining=0, friendly_fire=1)
    assert c["pen_friendly_fire"]["value"] < 0
    assert r < 0


def test_c_idle_is_actually_wired() -> None:
    """原文举例「消极拖延给予负奖励」。

    首版 `pen_idle` **存在但未接线** —— 没有任何路径传入 `idle_steps`，
    故该项恒为 0，等于没做。这条断言把它接上。
    """
    r, c = shape_step_reward({}, steps_remaining=0, idle_steps=3)
    assert c["pen_idle"]["value"] < 0, "消极拖延未接线（恒为 0）"
    assert r < 0


def test_c_rule_violation_is_penalised_and_distinct_from_invalid() -> None:
    """「超出规则的动作」与"引擎拒收的无效动作"是**两件事**，权重必须独立。"""
    assert "pen_rule_violation" in DEFAULT_WEIGHTS
    _, c_v = shape_step_reward({}, steps_remaining=0, rule_violations=1)
    _, c_i = shape_step_reward({}, steps_remaining=0,
                               action_outcomes=[{"error": {"code": 1}}])
    assert c_v["pen_rule_violation"]["value"] < 0
    assert c_i["pen_invalid_action"]["value"] < 0
    assert c_v["pen_rule_violation"]["weight"] != c_i["pen_invalid_action"]["weight"], \
        "两类违规共用权重 —— 无法按分量独立调参（违反 e)）"


def test_c_each_penalty_has_its_own_weight() -> None:
    """标准 e) 要求按分量调参 → 每个惩罚项必须有**独立**权重键。

    这是回归用例：首版 `pen_invalid_action` 借用 `pen_idle` 的权重，
    两个概念共用一个数。
    """
    for k in ("pen_idle", "pen_friendly_fire", "pen_rule_violation",
              "pen_friendly_loss", "pen_damage_taken"):
        assert k in DEFAULT_WEIGHTS, "缺独立权重 %s" % k


# ───────────────── d) 数值尺度与平滑性（原文 :520-523）

def test_d_rewards_are_clipped_to_a_bounded_range() -> None:
    """「可裁剪累计回报或设定奖励上下限」。"""
    r_hi, c_hi = shape_step_reward({"city_value_delta": 1e6}, steps_remaining=0)
    r_lo, _ = shape_step_reward({"kill_taken": 10_000}, steps_remaining=0)
    clip = DEFAULT_WEIGHTS["clip"]
    assert -clip <= r_hi <= clip and -clip <= r_lo <= clip
    assert c_hi["scale_clip"]["clipped"] is True


def test_d_different_situations_are_clearly_distinguishable() -> None:
    """「确保不同情境下奖励差异明显，便于模型区分成功/失败和优劣策略」。"""
    win, _ = shape_step_reward({}, steps_remaining=0, terminal_margin=10.0)
    lose, _ = shape_step_reward({}, steps_remaining=0, terminal_margin=-10.0)
    assert win > 0 > lose
    assert abs(win - lose) >= 20, "胜/负两种结局差异不够明显: %s vs %s" % (win, lose)


# ───────────────── e) 可解释性与可调试性（原文 :524-527）

def test_e_total_is_decomposed_into_named_components() -> None:
    """「可将总奖励拆分为多项……并在日志中记录各项」。"""
    _, c = shape_step_reward({"city_value_delta": 1.0}, steps_remaining=0,
                             milestones=[1], friendly_fire=1, terminal_margin=3.0)
    for term in ("align_city_value", "milestone_bonus", "pen_friendly_fire",
                 "terminal_outcome", "gains_losses", "time_discount", "scale_clip"):
        assert term in c, "缺少可调试分量 %s" % term
        assert isinstance(c[term], dict)


def test_e_components_sum_explains_the_total() -> None:
    """分量必须能**解释**总分（否则"可调试"是空话）。"""
    r, c = shape_step_reward({"city_value_delta": 2.0}, steps_remaining=0,
                             friendly_fire=1, terminal_margin=1.0)
    gl = c["gains_losses"]
    # 无折扣（steps_remaining=0）时：总分 == 收益 + 损失，再经裁剪
    expect = gl["gains"] + gl["losses"]
    clipped = max(-c["scale_clip"]["clip"], min(c["scale_clip"]["clip"], expect))
    assert r == pytest.approx(clipped, abs=1e-6)


# ───────────────── 与原文对得上（防自造）

def test_standard_5_3_3_text_contains_the_five_principles() -> None:
    """**防止自造**：原文 5.3.3 确实列出 a)–e) 五条，且含关键词。"""
    if not STD.is_file():
        pytest.skip("31007 source text not present in this checkout")
    t = STD.read_text(encoding="utf-8", errors="replace")
    assert "5.3.3 奖励函数设计指导" in t
    for key in ("任务目标对齐", "及时性与长期性平衡", "惩罚不当行为",
                "数值尺度与平滑性", "可解释性与可调试性"):
        assert key in t, "原文缺原则 %s" % key
    # 本文件断言的几个"举例"必须真的在原文里
    for example in ("友军误伤", "消极拖延", "超出规则的动作", "阶段性成果", "最终结果"):
        assert example in t, "原文缺举例词 %s（本文件据此断言，故必须核对）" % example
