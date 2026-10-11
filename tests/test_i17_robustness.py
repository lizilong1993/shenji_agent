"""`I17` 步骤 4（31007 6.2.4）的测试：抗扰阈值 · 跨域泛化 · 策略鲁棒性。

## 重点（三条诚实性）

1. **抗扰阈值**：上界内探不到 → **`None` + 理由**，**不把"没探到"说成"阈值很高"**；
2. **单调性**是二分法的前提 → **假设不成立时必须报出**，而不是给一个不可信的数；
3. **鲁棒性**必须**并列**给 `stability` 与 `worst_case` —— 只看均值会掩盖最差情形。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.robustness import (  # noqa: E402
    cross_domain_generalisation, disturbance_threshold, policy_robustness, summarize_three,
)


# ══════════════════ ① 抗扰容错阈值

def test_threshold_found_for_monotone_perf() -> None:
    """线性衰减：`perf(d) = 1 - d`，accept=0.5 → 阈值应为 0.5。"""
    r = disturbance_threshold(lambda d: 1.0 - d, lo=0.0, hi=1.0, accept=0.5, tol=1e-6)
    assert r["threshold"] == pytest.approx(0.5, abs=1e-4), r
    assert r["monotonic_ok"] is True


def test_threshold_uses_logarithmic_evaluations() -> None:
    """**必须二分**：调用次数应为 O(log(1/tol))，不是线性扫描。"""
    calls = []

    def perf(d):
        calls.append(d)
        return 1.0 - d

    r = disturbance_threshold(perf, lo=0.0, hi=1.0, accept=0.5, tol=1e-6)
    assert r["evaluations"] == len(calls)
    assert r["evaluations"] <= 30, "调用次数 %d 过多，疑似线性扫描" % r["evaluations"]


def test_never_crossing_returns_none_with_reason() -> None:
    """**最关键**：上界内始终高于可接受线 → 必须 `None` + 明确理由。"""
    r = disturbance_threshold(lambda d: 0.9 - 0.1 * d, lo=0.0, hi=1.0, accept=0.5)
    assert r["threshold"] is None
    assert "未测得" in r["reason"] or "未跌破" in r["reason"]
    assert "上界" in r["reason"]


def test_already_below_at_start_returns_lo() -> None:
    """起点即跌破 → 阈值为 `lo`，且**说明原因**。"""
    r = disturbance_threshold(lambda d: 0.1, lo=0.0, hi=1.0, accept=0.5)
    assert r["threshold"] == 0.0
    assert "起点" in r["reason"]


def test_non_monotone_is_detected_and_not_trusted() -> None:
    """**单调性假设不成立时必须报出** —— 否则给出的是不可信的数。"""
    def bumpy(d):                     # 中段反而变好
        return 1.0 - d + (0.6 if 0.4 < d < 0.6 else 0.0)

    r = disturbance_threshold(bumpy, lo=0.0, hi=1.0, accept=0.5, tol=1e-6)
    assert r["monotonic_ok"] is False
    assert r["threshold"] is None, "单调性不成立却给了阈值"
    assert "单调性" in r["reason"]


def test_invalid_interval_is_rejected() -> None:
    r = disturbance_threshold(lambda d: 1 - d, lo=1.0, hi=1.0)
    assert r["threshold"] is None and r["reason"]


def test_perf_returning_none_is_handled() -> None:
    r = disturbance_threshold(lambda d: None, lo=0.0, hi=1.0)
    assert r["threshold"] is None
    assert "取不到" in r["reason"]


# ══════════════════ ② 跨域泛化

def test_generalisation_ratio_is_computed() -> None:
    perf = {"map_123": [0.8, 0.8], "map_1531": [0.4, 0.4]}
    r = cross_domain_generalisation(perf, seen_domains=["map_123"], unseen_domains=["map_1531"])
    assert r["mean_seen"] == pytest.approx(0.8)
    assert r["mean_unseen"] == pytest.approx(0.4)
    assert r["generalisation"] == pytest.approx(0.5)


def test_unseen_better_is_clipped_and_flagged() -> None:
    """未见域反而更好 → 截断为 1.0，**但实际上报并说明不是"泛化强"**。"""
    perf = {"a": [0.5], "b": [1.0]}
    r = cross_domain_generalisation(perf, seen_domains=["a"], unseen_domains=["b"])
    assert r["generalisation"] == 1.0
    assert r["raw_ratio"] == pytest.approx(2.0)
    assert "note" in r and "难度" in r["note"]


def test_missing_domains_are_reported() -> None:
    r = cross_domain_generalisation({"a": [0.5]}, seen_domains=["a"], unseen_domains=["zzz"])
    assert r["generalisation"] is None
    assert r["missing_unseen"] == ["zzz"]


def test_zero_seen_mean_returns_reason() -> None:
    r = cross_domain_generalisation({"a": [0.0], "b": [0.5]},
                                    seen_domains=["a"], unseen_domains=["b"])
    assert r["generalisation"] is None and "0" in r["reason"]


# ══════════════════ ③ 策略鲁棒性

def test_stability_and_worst_case_are_both_reported() -> None:
    """**并列报告**：只看 `stability` 会掩盖"均值高但最差很糟"。"""
    r = policy_robustness({"p1": [0.9, 0.9], "p2": [0.1, 0.1]})
    assert r["stability"] is not None
    assert r["worst_case"] == pytest.approx(0.1)
    assert "并列" in r["note"]


def test_low_variance_gives_high_stability() -> None:
    r = policy_robustness({"p1": [0.50, 0.51], "p2": [0.49, 0.50]})
    assert r["stability"] > 0.95
    assert r["cv"] < 0.05


def test_high_variance_lowers_stability() -> None:
    r = policy_robustness({"p1": [0.9, 0.9], "p2": [0.1, 0.1]})
    assert r["stability"] < 0.5, r


def test_per_perturbation_means_are_returned() -> None:
    r = policy_robustness({"p1": [1.0, 2.0], "p2": [3.0]})
    assert r["per_perturbation"]["p1"]["mean"] == pytest.approx(1.5)
    assert r["per_perturbation"]["p2"]["mean"] == pytest.approx(3.0)


def test_insufficient_samples_returns_reason() -> None:
    r = policy_robustness({"p1": [0.5]})
    assert r["stability"] is None
    assert "少于 2" in r["reason"]
    assert r["worst_case"] == 0.5, "样本虽少，最差情形仍应给出"


def test_zero_mean_returns_reason_not_a_number() -> None:
    r = policy_robustness({"p1": [0.0, 0.0], "p2": [0.0]})
    assert r["stability"] is None and "变异系数" in r["reason"]


# ══════════════════ 汇总

def test_summarize_three_reports_all_components() -> None:
    r = summarize_three(
        lambda d: 1.0 - d,
        perf_by_domain={"seen": [0.8], "unseen": [0.4]},
        seen_domains=["seen"], unseen_domains=["unseen"],
        perf_under_perturbation={"p1": [0.8, 0.9]},
        accept=0.5, tol=1e-6,
    )
    # ⚠ 手算：perf(d)=1-d 在 accept=0.5 处 → 阈值应为 0.5。
    # 首版未传 tol，用了默认 1e-3 → 得 0.50049，超出 abs=1e-4 的容差。
    # 这是**参数未透传**，不是算法错；现已显式传 tol 并把容差收紧。
    assert r["disturbance_threshold"]["threshold"] == pytest.approx(0.5, abs=1e-5)
    assert r["cross_domain_generalisation"]["generalisation"] == pytest.approx(0.5)
    assert r["policy_robustness"]["stability"] is not None


def test_summarize_three_handles_missing_inputs_honestly() -> None:
    """**不提供输入时如实说"未提供"**，不返回 0 冒充"结果为零"。"""
    r = summarize_three()
    assert r["disturbance_threshold"]["threshold"] is None
    assert "未提供" in r["disturbance_threshold"]["reason"]
    assert r["cross_domain_generalisation"]["generalisation"] is None
    assert r["policy_robustness"]["stability"] is None
