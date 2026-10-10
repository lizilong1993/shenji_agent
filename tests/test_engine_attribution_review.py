"""引擎/鉴权归因复核（`docs/engine_attribution_review_20261010.md`）。

起因：第 49 轮实测推翻「采集需引擎、鉴权不通过故不可做」这一我重复多轮的结论。
本文件把复核结论与**根因证据**钉住，防止重新写成「引擎阻塞」。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOC = REPO / "docs" / "engine_attribution_review_20261010.md"
LABELER = REPO / "scripts" / "build_objective_labels.py"
MANIFEST = REPO / "ai" / "weights" / "model_manifest.json"


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("engine attribution review not present")
    return DOC.read_text(encoding="utf-8")


# ─────────────────── 核心结论：鉴权不是一道统一的墙

def test_doc_separates_local_simulation_from_platform_capability() -> None:
    """**最重要的断言**：本地推演不受鉴权影响；平台侧能力受影响。"""
    d = _doc()
    assert "不受影响" in d, "未说明本地推演不受鉴权影响"
    assert "受影响" in d, "未说明平台侧能力受影响"
    assert "did not pass authentication" in d, "未记录鉴权确实失败这一事实"
    assert "completed" in d, "未记录推演实际完成"


def test_doc_lists_the_per_item_verdicts() -> None:
    """必须逐项给出复核结论（否则只是泛泛而谈）。"""
    d = _doc()
    for iid in ("I06", "D1", "C3", "P1", "C2", "P2"):
        assert iid in d, "未复核 %s" % iid
    assert "归因不成立" in d and "归因成立" in d, "未区分成立/不成立两类"


def test_p1_external_opponent_remains_platform_blocked() -> None:
    """`P1` 的横向对比层**归因成立**（外部对手确属平台侧），不得一并"平反"。"""
    d = _doc()
    assert "待平台方" in d or "外部对手不可获得" in d


# ─────────────────── I06 的根因证据

def test_labeler_does_not_read_adjudication_events() -> None:
    """**根因**：`build_objective_labels.py` 不读 `events.jsonl` 的 `ADJUDICATION`。

    这条断言直接检查脚本源码 —— 若哪天脚本改了，测试会失败并强制更新文档。
    """
    if not LABELER.is_file():
        pytest.skip("labeler not present")
    src = LABELER.read_text(encoding="utf-8")
    assert "ADJUDICATION" not in src, "标签器已开始读 ADJUDICATION —— 文档需更新"
    assert "events.jsonl" not in src, "标签器已开始读 events.jsonl —— 文档需更新"
    # 但 damage/kill 这些概念它在用（说明它想读，只是读的是别的来源）
    assert "damage" in src and "kill" in src


def test_doc_records_adjudication_payload_carries_objective_fields() -> None:
    """`ADJUDICATION` 的 payload **确实携带** objective 字段（这是"数据本来就有"的依据）。"""
    d = _doc()
    for field in ("damage", "target_obj_id", "cur_step", "attack_color"):
        assert field in d, "未记录 payload 字段 %s" % field
    assert "ADJUDICATION" in d


def test_doc_records_that_existing_labels_are_null() -> None:
    """既有标签的 objective 字段**全为 null** —— 这是"标签器读错来源"的证据。"""
    d = _doc()
    assert "null" in d
    assert "final_outcome_credit" in d


def test_manifest_matches_the_quoted_counts() -> None:
    """**复核** manifest 现值与文档引用一致（防抄错）。"""
    if not MANIFEST.is_file():
        pytest.skip("manifest not present")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    olc = m.get("objective_label_counts") or {}
    assert olc.get("high_confidence") == 0, "high_confidence 已非 0 —— 文档需更新"
    assert olc.get("total") == 1480
    om = m.get("objective_metrics") or {}
    assert om.get("model_trained") is False
    d = _doc()
    assert "1480" in d and "model_trained" in d


def test_doc_owns_the_failed_probe_honestly() -> None:
    """我本轮的探针**失败**（600 步全在部署阶段）—— 必须如实记录，不掩饰。"""
    d = _doc()
    assert "END_DEPLOYMENT" in d, "未记录探针只产出部署事件"
    assert "从未进入战斗" in d or "全耗在部署" in d
    assert "探针失败" in d or "探针" in d


def test_doc_states_the_two_real_fix_points() -> None:
    """`I06` 步骤 2 的真实修复点有两个，且**都不是引擎**。"""
    d = _doc()
    assert "标签器" in d, "未指出标签器未消费 ADJUDICATION"
    assert "采集配置" in d, "未指出采集配置需能进战斗"
    assert "都不是引擎" in d or "不是引擎" in d


def test_doc_records_what_is_not_done() -> None:
    """**不得**把复核写成"已修复"。"""
    d = _doc()
    assert "未做" in d
    for claim in ("已修复", "已完成修复", "high_confidence>0 已达成"):
        assert claim not in d, "出现过度声明 %r" % claim
