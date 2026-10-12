"""I10 / D3 步骤 1：31002 表2 的 19 项元数据映射。

判据 1：「表2 19 项字段映射 **100% 覆盖**或明确 N/A」。

本条先更正了一处**转述**：任务清单写「该 19 项为报告的转述，报告自述『表2 不能逐项核对』」。
**原文就在 `.scratch/std/s31002.txt`（表2 见 `:198-219`），19 项一项不缺。**
本文件把「原文 19 项」「映射表 19 项」「每项有必填态」「统计自洽」都做成断言。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOC = REPO / "docs" / "situation_metadata_mapping.md"
STD = REPO.parent.parent / ".scratch" / "std" / "s31002.txt"

# 标准表2 的 19 项（**逐字**，取自 s31002.txt:201-219）
TABLE2_FIELDS = [
    "想定唯一标识符", "想定标题", "想定类型", "适用层级", "适用军兵种/任务领域",
    "作战地域/环境类型", "预估推演时长", "主要威胁/对手", "主要己方", "想定目标/训练目的",
    "想定复杂度等级", "开发/修订日期", "版本号", "作者/来源单位", "密级",
    "关键词/标签", "关联想定/素材", "适用模拟系统/平台", "评估指标",
]


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("situation_metadata_mapping.md not present")
    return DOC.read_text(encoding="utf-8")


def _rows(text: str, start: str, end: str):
    seg = text[text.find(start):text.find(end)]
    assert seg, "找不到区间 %s → %s" % (start, end)
    return [l for l in seg.split("\n") if re.match(r"^\|\s*\d+\s*\|", l)]


def _cell(row: str, idx: int) -> str:
    return [c.strip() for c in row.strip().strip("|").split("|")][idx]


# ─────────────────── 原文核对：19 项真的在标准里

def test_standard_table2_really_has_19_items() -> None:
    """**先确认原文在手**（本会话第四次同类教训，故做成断言）。"""
    if not STD.is_file():
        pytest.skip("31002 source text not present in this checkout")
    t = STD.read_text(encoding="utf-8", errors="replace")
    assert "表2" in t and "核心元数据字段" in t
    assert "至少应包括表 2 中所列条目" in t.replace("  ", " ").replace("\n", "")
    missing = [f for f in TABLE2_FIELDS if f.split("（")[0][:6] not in t]
    assert missing == [], "原文里找不到这些表2 字段: %s" % missing


def test_doc_quotes_the_original_table_with_19_rows() -> None:
    rows = _rows(_doc(), "## 1. 表2 原文", "> **注意第 19 项**")
    assert len(rows) == 19, "原文抄录应为 19 行，实际 %d" % len(rows)
    nums = [int(_cell(r, 0)) for r in rows]
    assert nums == list(range(1, 20)), "编号应 1..19 连续，实际 %s" % nums
    # 第 19 项必须带「（可选）」—— 原文自带，不许丢
    assert "可选" in _cell(rows[18], 1), "第 19 项原文标「（可选）」，抄录里丢了"


# ─────────────────── 判据 1：映射表覆盖 19 项

def test_mapping_table_covers_all_19_items() -> None:
    rows = _rows(_doc(), "## 3. 19 项映射表", "### 3.1 统计")
    assert len(rows) == 19, "映射表应覆盖 19 项，实际 %d" % len(rows)
    nums = [int(_cell(r, 0)) for r in rows]
    assert nums == list(range(1, 20)), "映射表编号应 1..19 连续，实际 %s" % nums


def test_every_item_states_required_or_optional() -> None:
    """步骤 1 要求「逐项标注**必填/选填**」。"""
    for r in _rows(_doc(), "## 3. 19 项映射表", "### 3.1 统计"):
        cell = _cell(r, 2)
        assert cell in ("必填", "可选", "**可选**"), \
            "第 %s 行的必填态不明确: %r" % (_cell(r, 0), cell)


def test_only_item_19_is_optional_per_the_standard() -> None:
    """原文**只在第 19 项**标了「（可选）」，其余按 5.2.2 默认必填。"""
    rows = _rows(_doc(), "## 3. 19 项映射表", "### 3.1 统计")
    optional = [int(_cell(r, 0)) for r in rows if "可选" in _cell(r, 2)]
    assert optional == [19], "只有第 19 项应标可选，实际 %s" % optional


def test_every_item_has_a_non_empty_gap_column() -> None:
    """**判据 1「无空白」**：每项的「缺失情况」列必须有内容，不许留空或只写破折号。"""
    for r in _rows(_doc(), "## 3. 19 项映射表", "### 3.1 统计"):
        last = _cell(r, -1)
        assert last and last not in ("—", "-", "N/A", "无", ""), \
            "第 %s 行缺失情况列空白" % _cell(r, 0)
        assert len(last) >= 4, "第 %s 行缺失情况过短: %r" % (_cell(r, 0), last)


def test_no_pending_placeholder_in_the_mapping() -> None:
    d = _doc()
    seg = d[d.find("## 3. 19 项映射表"):d.find("## 4. 与")]
    for bad in ("待定", "待补", "TBD", "TODO", "？？？"):
        assert bad not in seg, "映射表出现占位词 %r" % bad


def test_statistics_are_self_consistent() -> None:
    """§3.1 的分类计数之和必须等于 19。"""
    d = _doc()
    seg = d[d.find("### 3.1 统计"):d.find("## 4. 与")]
    nums = [int(x) for x in re.findall(r"\*\*(\d+)\*\*", seg)]
    assert 19 in nums
    # 三个分类数相加应为 19
    three = [n for n in nums[:4] if n != 19]
    assert sum(three) == 19, "分类计数 %s 之和不为 19" % three


# ─────────────────── 现状实测必须是真的

def test_p3_scenario_json_really_lacks_metadata() -> None:
    """**防编现状**：文档称 P3 想定 JSON 只有 4 个键、19 项里 0 项。

    这条断言直接读真实想定文件核对；若哪天 P3 加了元数据，测试会失败并强制改文档。
    """
    import json

    d = REPO / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
    files = sorted(d.glob("*.json"))[:3]
    if not files:
        pytest.skip("P3 scenarios not present")
    for f in files:
        j = json.loads(f.read_text(encoding="utf-8"))
        assert isinstance(j, dict)
        keys = set(j)
        assert "scenario_id" in keys, "%s 缺 scenario_id" % f.name
        # 文档断言：除 scenario_id 外没有任何元数据段
        for meta_key in ("metadata", "meta", "title", "name", "version", "classification"):
            assert meta_key not in keys, \
                "%s 出现元数据键 %r —— 文档的「19 项里 0 项」需重写" % (f.name, meta_key)


def test_doc_records_p4_as_measured() -> None:
    """P4 已远程部署并实跑 → 文档必须写**实测**并给出部署落点，不得再冒「待核」。"""
    d = _doc()
    assert "待核" not in d, "P4 已实测，文档仍标「待核」"
    assert "实测" in d, "文档未标「实测」"
    assert "/remote-home/lizilong/demo20260104" in d, "未给出远程部署落点，无法复核"


def test_doc_admits_metadata_is_not_readable_back() -> None:
    """`SituationOut` 保持旧结构 → 「可写不可回读」必须**自己写出来**，不得隐瞒。"""
    d = _doc()
    assert "回读" in d or "不可回读" in d, "未登记「元数据写入后不可回读」这一缺口"


def test_doc_does_not_claim_i10_is_complete() -> None:
    """**不得**把本条写成已完成 —— 步骤 5（P3 元数据块）与元数据回读仍未做。"""
    d = _doc()
    assert "步骤 5" in d, "未提步骤 5"
    assert "未加元数据块" in d, "未如实说明步骤 5 只完成一半"
    for claim in ("已完成全部", "全部完成", "19 项已实现"):
        assert claim not in d, "出现过度声明 %r" % claim
