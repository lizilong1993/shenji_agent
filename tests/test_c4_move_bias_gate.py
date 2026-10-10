"""C4 步骤 4：偏差训练门 —— 判据 2 的负向用例就在这里。

门的核心不变量是 **fail closed（拿不到数就不放行）**，
因为 `move_bias_estimate_from_probs` 在退化输入上返回 0.0，
而 0.0 恰好也是"完全平衡"的取值 —— 若混同，**一次失败的测量会被读成一次完美通过**。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_diag():
    spec = importlib.util.spec_from_file_location(
        "diag_gate", REPO / "scripts" / "diagnose_model_quality.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 负向用例（判据 2）

def test_gate_rejects_when_bias_exceeds_threshold() -> None:
    """**判据 2 的负向用例**：偏差超标必须拒绝放行。"""
    diag = _load_diag()
    r = diag.evaluate_move_bias_gate(-0.317, threshold=0.1, metric_source="intent_real_data")
    assert r["accepted"] is False
    assert r["reason"] == "move_bias_exceeds_threshold"
    assert r["measurement"] == pytest.approx(-0.317)


def test_gate_rejects_positive_bias_over_threshold_too() -> None:
    """两个方向都要挡：判定用 |·|，不是只看负值。"""
    diag = _load_diag()
    r = diag.evaluate_move_bias_gate(+0.25, threshold=0.1)
    assert r["accepted"] is False and r["reason"] == "move_bias_exceeds_threshold"


def test_gate_accepts_within_threshold() -> None:
    diag = _load_diag()
    r = diag.evaluate_move_bias_gate(-0.0629, threshold=0.1, metric_source="intent_real_data")
    assert r["accepted"] is True and r["reason"] == "ok"


def test_gate_boundary_is_inclusive() -> None:
    """恰好等于阈值算通过（判据写的是 <=）。"""
    diag = _load_diag()
    assert diag.evaluate_move_bias_gate(0.1, threshold=0.1)["accepted"] is True
    assert diag.evaluate_move_bias_gate(-0.1, threshold=0.1)["accepted"] is True


# ---------------------------------------------------------------- fail closed

def test_gate_fails_closed_when_measurement_unavailable() -> None:
    """拿不到测量值 → **拒绝**，而不是当作 0（=完美平衡）放过。"""
    diag = _load_diag()
    r = diag.evaluate_move_bias_gate(None, threshold=0.1)
    assert r["accepted"] is False
    assert r["reason"] == "measurement_unavailable"
    assert r["measurement"] is None


def test_gate_rejects_invalid_threshold() -> None:
    diag = _load_diag()
    for bad in (0, -0.1):
        r = diag.evaluate_move_bias_gate(0.0, threshold=bad)
        assert r["accepted"] is False and r["reason"] == "invalid_threshold"


def test_measurement_returns_none_on_degenerate_input() -> None:
    """这正是与 `move_bias_estimate_from_probs`（退化时返回 0.0）的关键区别。"""
    diag = _load_diag()
    assert diag.move_bias_measurement(np.zeros((0, 8))) is None      # 无样本
    assert diag.move_bias_measurement(np.zeros((4, 1))) is None      # 只有一个类
    assert diag.move_bias_measurement(None) is None
    # 而"看起来平衡"的输入应给出 0.0，**不是** None
    assert diag.move_bias_measurement(np.full((4, 8), 1 / 8)) == pytest.approx(0.0)


def test_degenerate_and_balanced_are_distinguishable() -> None:
    """把危险点写成显式断言：0.0（真平衡）与 None（测量失败）必须可区分。"""
    diag = _load_diag()
    balanced = diag.move_bias_measurement(np.full((4, 8), 1 / 8))
    degenerate = diag.move_bias_measurement(np.zeros((0, 8)))
    assert balanced == pytest.approx(0.0)
    assert degenerate is None
    assert balanced != degenerate          # 类型与语义都不同
    # 前者放行，后者拒绝 —— 这就是 fail closed 的意义
    assert diag.evaluate_move_bias_gate(balanced)["accepted"] is True
    assert diag.evaluate_move_bias_gate(degenerate)["accepted"] is False


# ---------------------------------------------------------------- 输出契约

def test_gate_result_shape_is_stable() -> None:
    """字段固定，便于训练侧/CI 直接消费。"""
    diag = _load_diag()
    r = diag.evaluate_move_bias_gate(0.05, threshold=0.1, metric_source="intent_real_data")
    assert set(r) == {"metric_source", "threshold", "measurement", "accepted", "reason"}
    assert r["metric_source"] == "intent_real_data"
    assert r["threshold"] == pytest.approx(0.1)
