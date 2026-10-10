"""P1 步骤 3/4/5：评估有效性守门（三层框架 + 平局率 / 区分度）。

标准依据（**原文** `.scratch/std/s31007.txt`）：
  - 6.1.1 评估方法（`:663-677`）：**三层递进框架**
    「仿真环境验证-实际推演校验-横向对比分析」；
  - 6.2.3 决策质量指标评估（`:734-740`）：
    「胜率指标应通过**足够次数的重复推演**获得，以确保具有**统计显著性和稳定性**」。

本文件把 `P1` 要的「平局率与区分度作为评估有效性**前置指标**」做成可执行断言，
并**复现报告里的事实**：20 轮 100% 平局 vs 自家基线 → 按标准**不可采信**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.eval_validity import (ALL_LAYERS, LAYER_COMPARATIVE, LAYER_LIVE,  # noqa: E402
                                    LAYER_SIMULATION, SOURCE_EXTERNAL, SOURCE_SELF,
                                    SOURCE_UNKNOWN, evaluate_validity,
                                    inventory_opponents, label_claim, summarise_outcomes)

STD = REPO.parent.parent / ".scratch" / "std" / "s31007.txt"


# ─────────────────── 统计口径

def test_summary_counts_and_rates() -> None:
    s = summarise_outcomes([1] * 6 + [-1] * 2 + [0] * 2)
    assert s["total"] == 10 and s["wins"] == 6 and s["losses"] == 2 and s["draws"] == 2
    assert s["draw_rate"] == pytest.approx(0.2)
    assert s["win_rate"] == pytest.approx(0.6)
    # 区分度 = 非平局中的胜率
    assert s["discrimination"] == pytest.approx(6 / 8)


def test_all_draws_give_zero_discrimination_not_half() -> None:
    """**关键口径**：全平局时区分度必须是 **0**，不能落到 0.5。

    若用"平局算半分"的算法，100% 平局会得到 0.5，**看起来还行** ——
    而它恰恰是"完全分不出强弱"的极端情形。
    """
    s = summarise_outcomes([0] * 20)
    assert s["draw_rate"] == pytest.approx(1.0)
    assert s["discrimination"] == 0.0, "全平局必须给出区分度 0"


def test_outcome_forms_are_equivalent() -> None:
    """同一批结果的不同表示法必须得到相同统计。"""
    a = summarise_outcomes([1, -1, 0])
    b = summarise_outcomes(["win", "loss", "draw"])
    c = summarise_outcomes([{"winner": 1}, {"winner": -1}, {"winner": None}])
    for x in (b, c):
        assert x["wins"] == a["wins"] and x["losses"] == a["losses"] and x["draws"] == a["draws"]


def test_unknown_entries_are_ignored_not_counted_as_draws() -> None:
    """无法识别的条目**不计入**（也不当作平局）—— 否则统计会被垃圾数据稀释。"""
    s = summarise_outcomes([1, "???", None, object(), 1])
    assert s["total"] == 2 and s["wins"] == 2


# ─────────────────── 步骤 4：守门（复现报告里的事实）

def test_reported_evidence_is_not_admissible() -> None:
    """**复现并判定报告里的情形**：20 轮 100% 平局、vs 自家基线、仅仿真层。

    按 6.2.3 的"统计显著性和稳定性"，该评估**不足以支撑任何对外成绩表述**。
    """
    r = evaluate_validity([0] * 20, opponent_source=SOURCE_SELF,
                          layers_present=[LAYER_SIMULATION])
    assert r["admissible"] is False
    failed = {g["gate"] for g in r["gates"] if not g["ok"]}
    assert "draw_rate" in failed, "100% 平局却通过了平局率门"
    assert "discrimination" in failed, "区分度为 0 却通过了区分度门"
    assert LAYER_LIVE in r["layers_missing"]
    assert LAYER_COMPARATIVE in r["layers_missing"]
    # 外部对比层必须标为待平台方配合，**不得**标为已达成
    assert r["comparative_layer_status"] == "pending_platform_support"


def test_self_play_result_is_flagged_as_not_external_capability() -> None:
    """`P1` 步骤 5：自博弈成绩**不得**表述为对外/实战能力。"""
    r = evaluate_validity([1] * 15 + [-1] * 5, opponent_source=SOURCE_SELF,
                          layers_present=list(ALL_LAYERS))
    assert any("自家基线" in w and "不得" in w for w in r["warnings"]), r["warnings"]


def test_missing_source_label_makes_it_inadmissible() -> None:
    """来源未标注 → **不可采信**（否则自博弈成绩可能被当对外成绩引用）。"""
    r = evaluate_validity([1] * 15 + [-1] * 5, opponent_source=SOURCE_UNKNOWN,
                          layers_present=list(ALL_LAYERS))
    assert r["admissible"] is False
    assert any(g["gate"] == "opponent_source_labelled" and not g["ok"] for g in r["gates"])


def test_external_opponent_with_all_layers_passes() -> None:
    """外部对手 + 三层齐 + 有区分度 + 样本够 → **可采信**。"""
    r = evaluate_validity([1] * 12 + [-1] * 8, opponent_source=SOURCE_EXTERNAL,
                          layers_present=list(ALL_LAYERS))
    assert r["admissible"] is True, [g for g in r["gates"] if not g["ok"]]
    assert r["comparative_layer_status"] == "present"
    assert r["warnings"] == []


def test_sample_size_gate_enforces_enough_repetitions() -> None:
    """6.2.3「足够次数的重复推演」→ 样本量是独立的门。"""
    r = evaluate_validity([1, -1, 1], opponent_source=SOURCE_EXTERNAL,
                          layers_present=list(ALL_LAYERS), min_games=20)
    assert r["admissible"] is False
    assert any(g["gate"] == "sample_size" and not g["ok"] for g in r["gates"])


def test_thresholds_are_declared_as_engineering_values_not_standard() -> None:
    """**不得**把本仓阈值说成"标准规定"。标准未给具体数值。"""
    r = evaluate_validity([1] * 12 + [-1] * 8, opponent_source=SOURCE_EXTERNAL,
                          layers_present=list(ALL_LAYERS))
    note = r["thresholds_note"]
    assert "标准未规定具体数值" in note, "阈值口径说明缺失或含糊: %r" % note
    assert "本仓" in note


def test_discrimination_gate_is_independent_of_draw_gate() -> None:
    """两门**独立**：可以平局率达标而区分度不达标（如 10 胜 0 负 10 平）。"""
    r = evaluate_validity([1] * 10 + [0] * 10, opponent_source=SOURCE_EXTERNAL,
                          layers_present=list(ALL_LAYERS),
                          max_draw_rate=0.6, min_discrimination=0.2)
    gates = {g["gate"]: g["ok"] for g in r["gates"]}
    assert gates["draw_rate"] is True       # 0.5 <= 0.6
    assert gates["discrimination"] is True  # 10/10 = 1.0


# ─────────────────── 步骤 5：来源标签强制

def test_label_claim_always_carries_a_source_tag() -> None:
    for src in (SOURCE_SELF, SOURCE_EXTERNAL, SOURCE_UNKNOWN):
        got = label_claim("胜率 88.5%", opponent_source=src)
        assert "［口径" in got, "未加来源标签: %r" % got


def test_self_play_label_says_internal_only() -> None:
    got = label_claim("胜率 88.5%", opponent_source=SOURCE_SELF)
    assert "内部对比" in got and "非对外能力" in got


def test_unlabelled_claim_is_marked_not_referenceable() -> None:
    got = label_claim("胜率 88.5%", opponent_source=SOURCE_UNKNOWN)
    assert "不得对外引用" in got


# ─────────────────── 步骤 1：对手来源盘点

def test_opponent_inventory_separates_self_from_external() -> None:
    inv = inventory_opponents([
        {"name": "weights_incumbent_rule", "source": "self_baseline"},
        {"name": "baseline_rule_v1", "source": "self_baseline"},
        {"name": "official_demo_ai", "source": "external"},
        {"name": "mystery"},
    ])
    assert inv["counts"]["self_baseline"] == 2
    assert inv["counts"]["external"] == 1
    assert inv["counts"]["unknown"] == 1
    assert inv["external_available"] is True


def test_opponent_inventory_reports_external_unavailable() -> None:
    """**不虚报达成**：外部对手为空时必须显式报告不可用。"""
    inv = inventory_opponents([{"name": "a", "source": "self_baseline"}])
    assert inv["external_available"] is False
    assert "pending_platform_support" in inv["note"]


# ─────────────────── 配置与标准原文核对（防自造）

def test_config_declares_layers_and_unavailable_external_opponent() -> None:
    """`P1` 步骤 2 的配置文件必须存在，且**显式声明外部对手不可用**。"""
    import yaml

    cfg_path = REPO / "configs" / "eval_official_demo.yaml"
    if not cfg_path.is_file():
        pytest.skip("eval_official_demo.yaml not present")
    d = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ev = d["evaluation"]
    assert set(ev["layers"]) == set(ALL_LAYERS), "三层框架未逐层声明"
    ext = [o for o in ev["opponents"] if o.get("source") == "external"]
    assert ext, "未声明任何外部对手槽位"
    assert all(o.get("available") is False for o in ext), \
        "外部对手被标为可用，但平台侧可获取性尚未确认 —— 不得虚报"


def test_standard_text_contains_the_three_layer_framework() -> None:
    """**防自造**：原文确有 6.1.1 三层框架与 6.2.3 的统计显著性要求。"""
    if not STD.is_file():
        pytest.skip("31007 source text not present in this checkout")
    t = STD.read_text(encoding="utf-8", errors="replace")
    assert "6.1.1 评估方法" in t
    assert "仿真环境验证-实际推演校验-横向对比分析的三层递进框架" in t.replace(" ", "")
    assert "6.2.3 决策质量指标评估" in t
    assert "统计显著性和稳定性" in t
    # 横向对比层的原文依据必须是"与已有的同类算法和模型进行对比"
    assert "与已有的同类算法和模型进行对比" in t
