"""D1 / Q7：对抗强度定义与算力口径（`docs/eval_intensity_spec.md`）。

`docs/decision_register.md` 的 `Q7` 明确要求产出本文件并「给出算力预算与实测单例耗时，
据此确认可执行规模」，且「**不得虚报**」。本文件把该要求落成可执行断言。

关键判断（本文件要守住的）：
  31007 6.1.2 a) 按**对抗强度**组织（每档>100 用例），
  而 `collection_plan_v1.json` 按**意图**组织 —— **两个不同维度**。
  文档必须点明这一点，否则 `Q7` 会被误认为"已解决"。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOC = REPO / "docs" / "eval_intensity_spec.md"
PLAN = REPO / "Shenji" / "configs" / "collection_plan_v1.json"
STD = REPO.parent.parent / ".scratch" / "std" / "s31007.txt"

THREE_LEVELS = ("L1", "L2", "L3")


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("eval_intensity_spec.md not present")
    return DOC.read_text(encoding="utf-8")


# ─────────────────── 标准原文核对（防自造）

def test_standard_requires_per_intensity_case_counts() -> None:
    """**防自造**：原文 6.1.2 a) 确有「每种对抗强度下生成超过 100 种标准化测试用例」。"""
    if not STD.is_file():
        pytest.skip("31007 source text not present in this checkout")
    t = STD.read_text(encoding="utf-8", errors="replace")
    assert "6.1.2 评估流程" in t
    flat = t.replace(" ", "").replace("\n", "")
    assert "每种对抗强度下生成超过100种标准化测试用例" in flat, \
        "原文里找不到 6.1.2 a) 的关键要求"
    # 三档的语义雏形也必须在原文里
    for kw in ("常规对抗", "战场迷雾", "通信丢包", "多域复合打击", "电子干扰"):
        assert kw in t, "原文缺 %r —— 文档的三档定义据此，故必须核对" % kw


def test_doc_defines_three_levels_grounded_in_the_source() -> None:
    """三档定义必须存在，且**每档都引用原文措辞**（不是自创）。"""
    d = _doc()
    for lv in THREE_LEVELS:
        assert lv in d, "缺档位 %s" % lv
    for kw in ("常规对抗", "战场迷雾", "通信丢包", "多域复合打击", "电子干扰", "蒙特卡洛"):
        assert kw in d, "三档定义未引用原文措辞 %r" % kw


def test_doc_states_gt_100_and_monte_carlo_n() -> None:
    """「每档 >100 用例」与「蒙特卡洛 N≥30」必须写明，且**不得混为一谈**。"""
    d = _doc()
    assert "100" in d and "30" in d
    # 必须解释 100（用例数）与 N（重复次数）不是同一件事
    assert "不同乘" in d or "不是同一件事" in d or "不同" in d, \
        "未区分「用例数」与「重复次数」"


# ─────────────────── 关键判断：维度不同

def test_doc_flags_the_dimension_mismatch() -> None:
    """**本文件最重要的断言**：计划按意图组织，6.1.2 按强度组织。"""
    d = _doc()
    assert "不存在" in d or "完全不同" in d or "不是同一个维度" in d
    assert "意图" in d and "对抗强度" in d
    # 必须点明这是 Q7 卡住的地方
    assert "Q7" in d


def test_plan_really_has_no_intensity_dimension() -> None:
    """**实测复核**：计划里确实没有对抗强度字段。"""
    if not PLAN.is_file():
        pytest.skip("collection_plan_v1.json not present")
    t = PLAN.read_text(encoding="utf-8")
    for kw in ("adversarial_intensity", "对抗强度", "intensity"):
        assert kw not in t, "计划里出现了 %r —— 文档的「不存在」判断需重写" % kw
    d = json.loads(t)
    assert "intent_targets" in d
    assert len(d["intent_targets"]) == 22, "意图数应为 22"


# ─────────────────── 实测数字必须是真的

def test_doc_documents_the_collection_plan_targets() -> None:
    """文档里的 v1 目标合计必须与计划文件**算出来的一致**。"""
    if not PLAN.is_file():
        pytest.skip("collection_plan_v1.json not present")
    d = json.loads(PLAN.read_text(encoding="utf-8"))
    total = sum(v["v1_min_segments"] for v in d["intent_targets"].values())
    assert total == 11600, "v1 目标合计应为 11600，实际 %d" % total
    doc = _doc()
    assert "11,600" in doc or "11600" in doc, "文档未记录 v1 目标合计"


def test_doc_records_pilot_attainment_as_partial() -> None:
    """**不得虚报**：文档必须写出 14/22 而非"已达成"。"""
    d = _doc()
    assert "14 / 22" in d or "14/22" in d, "未记录 pilot 达成率"
    assert re.search(r"52\.4", d), "未记录 v1 完成率"


def test_doc_measures_actual_data_if_present() -> None:
    """若本机有采集产物，**复核**文档里的场次数与 segments 数。"""
    root = REPO / "Shenji" / "raw" / "canonical_v4_candidate"
    if not root.is_dir():
        pytest.skip("no collected raw data on this host")
    eps = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not eps:
        pytest.skip("no episodes")
    total = 0
    for ep in eps:
        f = ep / "intent_segments.jsonl"
        if f.is_file():
            with f.open(encoding="utf-8") as fh:
                total += sum(1 for l in fh if l.strip())
    doc = _doc()
    assert str(len(eps)) in doc, "文档里的场次数 %s 与实测 %d 不符" % (None, len(eps))
    assert "{:,}".format(total) in doc or str(total) in doc, \
        "文档里的 segments 合计与实测 %d 不符" % total


def test_doc_records_the_status_json_gap() -> None:
    """`status.json` 缺失使 `coverage` 误报 0 —— 影响 D1 步骤 4，必须写明。"""
    d = _doc()
    assert "status.json" in d
    assert "missing_status" in d
    assert "0" in d


def test_doc_labels_unmeasured_items_as_unmeasured() -> None:
    """**Q7 要求"不得虚报"**：未实测的耗时必须标为未实测，不得编数字。"""
    d = _doc()
    assert "未实测" in d or "未做" in d, "未标注未实测项"
    # 不得出现看起来像实测的耗时数字（如 "秒/场" 带具体值）
    assert "单场墙钟耗时未在本文实测" in d or "耗时未实测" in d


def test_doc_states_the_two_remaining_gaps() -> None:
    """`Q7` 的两个真缺口必须写明，否则会被当作已关闭。"""
    d = _doc()
    assert "换算" in d, "未写明「用例↔场次↔segment 换算」缺口"
    assert "耗时" in d, "未写明单例耗时缺口"
    assert "开放" in d, "未声明 Q7 仍开放"


def test_docs_reference_the_spec_as_produced() -> None:
    """`README.md` / `decision_register.md` 不应再把该文件列为「待产出」。"""
    for name in ("README.md", "decision_register.md"):
        p = REPO / "docs" / name
        if not p.is_file():
            continue
        t = p.read_text(encoding="utf-8")
        for line in t.split("\n"):
            if "eval_intensity_spec" in line:
                assert "待产出" not in line, "%s 仍把该文件列为待产出" % name
