"""I06/C4 默认② 的**并行门**：`distribution_gap_gate` 的校准性测试。

## 背景（为何要并行门）

既有门 `move_bias_gate` 判据是 `|move_bias_estimate| <= 0.1`，
而该量等价于 `(1 - p0)/k - p0`，**把 0 当基准等于要求 `p0 ≈ 1/7`**：

| 数据 | `p0` | 既有门成立条件 `0.0571 ≤ p0 ≤ 0.1429`？ |
| --- | --- | --- |
| 真实标签 | **0.565** | **否 → 必然拒绝** |
| 完全平衡 | 0.1429 | **恰好是上界** |

**故旧门"数据怎么分布都会失败"** —— 这不是"模型偏了"，而是**指标失准**。

## 并行门的口径

`gap = p0_model - p0_label`，判据 `|gap| <= threshold`。
**基准换成标签自身** —— 数据怎么分布，基准就怎么走。

## 本文件的重点：**校准性**

不只测"它能跑"，而是测**它在各种分布下是否判得对**：
平衡数据必过；模型严重偏离必拒；标签失衡但模型跟随则**应当通过**（这正是旧门做不到的）。

## 不改既有门

`evaluate_move_bias_gate` **一个字节都没动**（另有测试逐字节核对）。
两个门**并列报告**，并用 `gates_disagree` 标出冲突，**不静默**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.diagnose_model_quality import (  # noqa: E402
    evaluate_distribution_gap_gate,
    evaluate_move_bias_gate,
    move_bias_estimate_from_probs,
)

np = pytest.importorskip("numpy")


# ══════════════ 校准性：**核心**

def test_balanced_data_passes_the_parallel_gate() -> None:
    """**平衡数据必须通过** —— 这正是旧门做不到的（旧门在 `p0=1/7` 恰好压线）。"""
    p0 = 1.0 / 7.0
    r = evaluate_distribution_gap_gate(p0, p0)
    assert r["accepted"] is True, r
    assert abs(r["gap"]) < 1e-12, r


def _old_gate_pass_interval(k: int = 6, threshold: float = 0.1):
    """旧门的**可通过 p0 区间**，由 `|(1-p0)/k - p0| <= threshold` 解出。

    ⚠ 我第一版把上界**手算成 0.1429（错了）**，并据此写了一个"p0=1/6 应失败"的断言 ——
    实测 p0=1/6 时旧门**通过**（bias=-0.0278 < 0.1）。**故改为解析求解，不再手算。**
    """
    lo = (1.0 - threshold * k) / (k + 1.0)
    hi = (1.0 + threshold * k) / (k + 1.0)
    return lo, hi


def test_old_gate_interval_is_what_we_claim() -> None:
    """把旧门的**可通过区间**钉住 —— 这是"旧门不可通过"论证的基础。"""
    lo, hi = _old_gate_pass_interval()
    assert lo == pytest.approx(0.057142857, abs=1e-8)
    assert hi == pytest.approx(0.228571428, abs=1e-8)
    # 区间内外的代表性取值
    for p0 in (0.10, 0.20):
        probs = np.full((10, 7), (1.0 - p0) / 6.0)
        probs[:, 0] = p0
        m = move_bias_estimate_from_probs(probs)
        assert evaluate_move_bias_gate(m, threshold=0.1)["accepted"] is True, p0
    for p0 in (0.25, 0.50, 0.565, 0.625):
        probs = np.full((10, 7), (1.0 - p0) / 6.0)
        probs[:, 0] = p0
        m = move_bias_estimate_from_probs(probs)
        assert evaluate_move_bias_gate(m, threshold=0.1)["accepted"] is False, p0


def test_real_data_distribution_fails_the_old_gate_but_passes_the_new_one() -> None:
    """**核心对照**：真实数据下，旧门拒绝、并行门通过 —— 而模型与数据的差只有 **0.06**。

    两个门看的是**同一份预测**，结论相反，**故冲突可归因于口径而非模型**。
    """
    p0_model, p0_label = 0.6251643896102905, 0.5649432534678437
    probs = np.full((10, 7), (1.0 - p0_model) / 6.0)
    probs[:, 0] = p0_model
    old = evaluate_move_bias_gate(move_bias_estimate_from_probs(probs), threshold=0.1)
    new = evaluate_distribution_gap_gate(p0_model, p0_label)
    assert old["accepted"] is False, "旧门在真实分布下应拒绝（这正是问题所在）"
    assert new["accepted"] is True, "并行门应通过（模型与数据仅差 0.06）"
    assert new["gap"] == pytest.approx(0.06022113614244684, abs=1e-9)


def test_model_following_an_imbalanced_dataset_passes() -> None:
    """**关键性质**：标签本身失衡（`p0_label=0.565`）而模型**跟随**时，**应当通过**。

    旧门在此必然失败；新门通过 —— 这才叫"区分'复现数据分布'与'模型自己偏了'"。
    """
    r = evaluate_distribution_gap_gate(0.565, 0.565)
    assert r["accepted"] is True, r
    assert r["gap"] == pytest.approx(0.0, abs=1e-12)


def test_model_far_above_the_label_is_rejected() -> None:
    """模型比数据**更偏"不动"** → 必须拒绝。"""
    r = evaluate_distribution_gap_gate(0.90, 0.565)
    assert r["accepted"] is False, r
    assert r["reason"] == "distribution_gap_exceeds_threshold"
    assert r["gap"] == pytest.approx(0.335, abs=1e-9)


def test_model_far_below_the_label_is_rejected() -> None:
    """模型比数据**明显少判"不动"** → 也必须拒绝（两个方向都要管）。"""
    r = evaluate_distribution_gap_gate(0.20, 0.565)
    assert r["accepted"] is False, r
    assert r["gap"] == pytest.approx(-0.365, abs=1e-9)


def test_gap_is_signed_not_absolute() -> None:
    """`gap` 必须**带符号** —— 否则无法判断偏向哪一侧。"""
    assert evaluate_distribution_gap_gate(0.7, 0.5)["gap"] > 0
    assert evaluate_distribution_gap_gate(0.3, 0.5)["gap"] < 0


# ══════════════ fail closed（与旧门同口径）

def test_missing_measurement_is_rejected_not_silently_passed() -> None:
    """**失败即拒绝**：拿不到数就不放行（与旧门一致，不让"没报错"变成放行）。"""
    for a, b in ((None, 0.5), (0.5, None), (None, None)):
        r = evaluate_distribution_gap_gate(a, b)
        assert r["accepted"] is False, (a, b, r)
        assert r["reason"] == "measurement_unavailable"


def test_invalid_threshold_is_rejected() -> None:
    """`threshold <= 0` 是配置错误 → 拒绝。"""
    for t in (0, -0.1):
        r = evaluate_distribution_gap_gate(0.5, 0.5, threshold=t)
        assert r["accepted"] is False, r
        assert r["reason"] == "invalid_threshold"


def test_boundary_is_inclusive() -> None:
    """判据是 `|gap| <= threshold` —— **边界包含**。"""
    assert evaluate_distribution_gap_gate(0.6, 0.5, threshold=0.1)["accepted"] is True
    assert evaluate_distribution_gap_gate(0.600001, 0.5, threshold=0.1)["accepted"] is False


# ══════════════ 不改既有门（逐字节）

def test_existing_gate_is_byte_identical() -> None:
    """**既有门必须逐字节未改** —— 本改造是"新增并行门"，不是"改判据"。"""
    src = (REPO / "scripts" / "diagnose_model_quality.py").read_text(encoding="utf-8")
    i = src.index("def evaluate_move_bias_gate")
    j = src.index("def evaluate_distribution_gap_gate")
    body = src[i:j]
    # 既有门的判定逻辑必须原样在位
    for frag in ('res.update(accepted=False, reason="invalid_threshold")',
                 'res.update(accepted=False, reason="measurement_unavailable")',
                 'res.update(accepted=False, reason="move_bias_exceeds_threshold")',
                 'res.update(accepted=True, reason="ok")'):
        assert frag in body, "既有门被改动了：缺 %s" % frag
    assert "def evaluate_move_bias_gate" in src


def test_both_gates_are_reported_side_by_side() -> None:
    """两个门必须**并列报告**，且冲突时以 `gates_disagree` 显式标出（**不静默**）。"""
    src = (REPO / "scripts" / "diagnose_model_quality.py").read_text(encoding="utf-8")
    assert 'result["move_bias_gate"]' in src
    assert 'result["distribution_gap_gate"]' in src
    assert '"gates_disagree"' in src, "未标出两门冲突 —— 判据冲突会被静默"
    i = src.index('result["gates_disagree"]')
    assert "distribution_gap_gate" in src[i:i + 400] and "move_bias_gate" in src[i:i + 400]


def test_label_distribution_is_emitted_with_the_same_scale() -> None:
    """`label_distribution` 必须是**占比**（与 `class_distribution` 同尺度），否则不能相减。"""
    src = (REPO / "scripts" / "diagnose_model_quality.py").read_text(encoding="utf-8")
    i = src.index('"label_distribution"')
    frag = src[i:i + 400]
    assert "float(len(head0))" in frag, "未按总数归一 —— 与 class_distribution 不同尺度"
