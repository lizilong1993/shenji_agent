"""`P3` 步骤 6 的测试：评估报告模板必须**让缺项可见**、**不让胜率独占结论**。

## 三条核心性质

1. **`None`（未测）与 `0.0`（测得为零）必须区分** —— 混同会把"没测"写成"结果为零"；
2. **缺项逐条进 `missing` 且带理由** —— `P3` 原文要求「缺项有明确说明」；
3. **结论不依赖单一胜率** —— 只有胜率时必须明确说"不足以支撑"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.eval_report import (  # noqa: E402
    REQUIRED, build_report, render_markdown,
)


def _full_layers():
    return {
        "L1_simulation": {"cases_per_level": {"L1": 110, "L2": 110, "L3": 110},
                          "boundary_coverage": {"covered": 6, "total": 6}},
        "L2_replay": {"win_rate": 0.55, "monte_carlo_ci": {"mean": 0.55, "ci_low": 0.51, "ci_high": 0.59}},
        "L3_comparison": {"expert_total_mean": 7.2, "cognitive_alignment": 0.78},
    }


# ══════════════ 完全齐备

def test_complete_report_has_no_missing() -> None:
    r = build_report(layers=_full_layers())
    assert r["missing"] == []
    assert r["measured_count"] == r["required_metrics_total"]
    assert "SUPPORTED" in r["conclusion"]


def test_required_metrics_match_the_three_layers() -> None:
    r = build_report(layers=_full_layers())
    assert set(r["layers"]) == set(REQUIRED)
    assert r["required_metrics_total"] == sum(len(v) for v in REQUIRED.values())


# ══════════════ ★ None 与 0.0 必须区分

def test_none_means_unmeasured_and_is_listed() -> None:
    layers = _full_layers()
    layers["L2_replay"]["monte_carlo_ci"] = None
    r = build_report(layers=layers)
    m = [(x["layer"], x["metric"]) for x in r["missing"]]
    assert ("L2_replay", "monte_carlo_ci") in m
    assert any("未测" in x["reason"] for x in r["missing"] if x["metric"] == "monte_carlo_ci")


def test_zero_is_measured_and_not_missing() -> None:
    """**关键区分**：`0.0` 是"测得为零"，**不是**缺项。"""
    layers = _full_layers()
    layers["L2_replay"]["win_rate"] = 0.0
    r = build_report(layers=layers)
    assert not any(x["metric"] == "win_rate" for x in r["missing"]), \
        "0.0 被误判为未测 —— 那会把'一场没赢'写成'没测过'"


def test_absent_key_is_missing_with_reason() -> None:
    layers = _full_layers()
    del layers["L3_comparison"]["cognitive_alignment"]
    r = build_report(layers=layers)
    hit = [x for x in r["missing"] if x["metric"] == "cognitive_alignment"]
    assert hit and hit[0]["reason"] == "未提供"


def test_missing_metrics_are_never_averaged_in_as_zero() -> None:
    """模板**不得**把缺失指标折算成 0 参与任何计算 —— 报告里只登记、不填值。"""
    layers = {"L1_simulation": {"cases_per_level": {"L1": 110}, "boundary_coverage": None}}
    r = build_report(layers=layers)
    assert r["layers"]["L1_simulation"]["boundary_coverage"] is None
    # 且明确写在免责声明里
    assert "0.0" in r["disclaimer"] and "None" in r["disclaimer"]


# ══════════════ 结论不依赖单一胜率

def test_win_rate_alone_does_not_support_the_conclusion() -> None:
    """**31007 的关键要求**：结论不得依赖单一胜率。"""
    layers = {"L2_replay": {"win_rate": 0.9}}
    r = build_report(layers=layers)
    assert "INSUFFICIENT" in r["conclusion"]
    assert "胜率" in r["conclusion"] and "不足以单独支撑" in r["conclusion"]


def test_partial_evidence_is_labelled_partial() -> None:
    layers = {"L1_simulation": {"cases_per_level": {"L1": 110}, "boundary_coverage": None},
              "L2_replay": {"win_rate": 0.5, "monte_carlo_ci": {"mean": 0.5}}}
    r = build_report(layers=layers)
    assert "PARTIALLY_SUPPORTED" in r["conclusion"]


def test_no_measurement_yields_insufficient_evidence() -> None:
    r = build_report(layers={})
    assert "INSUFFICIENT_EVIDENCE" in r["conclusion"]
    assert r["measured_count"] == 0
    assert len(r["missing"]) == r["required_metrics_total"]


# ══════════════ Markdown 渲染

def test_markdown_lists_every_missing_item() -> None:
    layers = {"L1_simulation": {"cases_per_level": None, "boundary_coverage": None}}
    md = render_markdown(build_report(layers=layers))
    assert "缺项清单" in md
    assert "`cases_per_level`" in md and "`boundary_coverage`" in md
    assert "未测" in md


def test_markdown_distinguishes_none_from_zero() -> None:
    layers = _full_layers()
    layers["L2_replay"]["win_rate"] = 0.0
    layers["L1_simulation"]["boundary_coverage"] = None
    md = render_markdown(build_report(layers=layers))
    assert "0.0000" in md, "0.0 未按数值渲染"
    assert "—（未测）" in md, "None 未按未测渲染"


def test_markdown_shows_provenance_and_episode_counts() -> None:
    r = build_report(layers=_full_layers(),
                     episode_counts={"L2_replay": 30},
                     provenance={"权重": "本模块自定"})
    md = render_markdown(r)
    assert "样本量" in md and "口径与来源" in md
    assert "本模块自定" in md


def test_markdown_has_the_three_layer_headings() -> None:
    md = render_markdown(build_report(layers=_full_layers()))
    for layer in REQUIRED:
        assert layer in md


def test_report_is_json_serialisable() -> None:
    """报告要能落盘（`P3` 验证通道提到归档到 `reports/`）。"""
    import json
    r = build_report(layers=_full_layers())
    json.dumps(r, ensure_ascii=False)      # 不抛即通过
