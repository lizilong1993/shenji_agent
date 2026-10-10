"""C4：移动偏差指标的**量纲**与批次无关性。

这个指标曾被写成 `sum(probs[:,1:])/(n-1) - mean(probs[:,0])`：
前半段对**所有样本**求和、后半段对样本**取均值**，量纲不一致，
于是指标**随批次样本数线性增长**。实测（同一真实训练集）：

    样本数   33      93      193     393
    原公式  +3.61  +10.67  +22.46  +45.93      ≈ 0.117 × 样本数
    本函数  -0.078 -0.066 -0.062 -0.061       稳定

证据文件里的 `-0.317` 因此**只能**来自"恰好 4 个样本"，换任何真实样本集都会得到完全不同的数。
本测试把「批次无关」钉死，防止它再退化回去。
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
        "diag_metric", REPO / "scripts" / "diagnose_model_quality.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _probs(n_samples: int, n_classes: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n_samples, n_classes))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_metric_is_batch_size_invariant() -> None:
    """核心性质：换样本数，指标不应系统性漂移。"""
    diag = _load_diag()
    rng = np.random.default_rng(7)
    big = _probs(4000, seed=7)
    values = [diag.move_bias_estimate_from_probs(big[:n]) for n in (50, 200, 800, 4000)]
    spread = max(values) - min(values)
    assert spread < 0.05, "指标随样本数漂移过大（说明又变成求和量纲了）: %s" % values
    # 且必须落在概率差的合法区间内
    for v in values:
        assert -1.0 <= v <= 1.0


def test_metric_matches_hand_computed_value() -> None:
    diag = _load_diag()
    p = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]], dtype="float64")
    # mean(p[:,1:]) = (0.5+0+0+0.5)/4 = 0.25 ; mean(p[:,0]) = 0.5
    assert abs(diag.move_bias_estimate_from_probs(p) - (0.25 - 0.5)) < 1e-12


def test_metric_sign_conventions() -> None:
    diag = _load_diag()
    stay_heavy = np.array([[0.9, 0.1, 0.0], [0.9, 0.0, 0.1]])
    move_heavy = np.array([[0.1, 0.6, 0.3], [0.1, 0.3, 0.6]])
    assert diag.move_bias_estimate_from_probs(stay_heavy) < 0, "偏不动应为负"
    assert diag.move_bias_estimate_from_probs(move_heavy) > 0, "偏移动应为正"


def test_metric_degrades_safely() -> None:
    """退化输入不得抛异常（诊断器最怕自己把整轮跑挂）。"""
    diag = _load_diag()
    assert diag.move_bias_estimate_from_probs(np.zeros((0, 8))) == 0.0
    assert diag.move_bias_estimate_from_probs(np.zeros((4, 1))) == 0.0
    assert diag.move_bias_estimate_from_probs(np.zeros((4, 8))) == 0.0
    assert diag.move_bias_estimate_from_probs(None) == 0.0


def test_old_formula_would_have_failed_this() -> None:
    """把原公式写出来，证明它确实与批次相关 —— 这就是被修掉的缺陷。"""
    p = _probs(300, seed=3)
    n = p.shape[1]
    old = float(np.sum(p[:, 1:])) / float(n - 1) - float(np.mean(p[:, 0]))
    new = _load_diag().move_bias_estimate_from_probs(p)
    assert abs(old) > 1.0, "原公式在 300 样本上已远超合法区间 [-1,1]：%r" % old
    assert -1.0 <= new <= 1.0
