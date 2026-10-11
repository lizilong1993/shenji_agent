"""`REPRODUCE.md` 的核对章节必须与 `check_all.py` 的**实际检查项一一对应**。

## 为什么需要（§137 → §128 的同类）

本会话已记录：**文档落后于实现**是反复出现的问题。
`REPRODUCE.md` 此前写「四项核对」，而 `check_all.py` 实际已有七项。

**故本测试从代码里抽出检查项名，要求文档逐个提到** ——
**文档不能再落后于实现，除非有人显式删掉这个测试。**

## ⚠ 本测试能做什么、不能做什么

| 能 | 不能 |
| --- | --- |
| 检出「代码加了检查项、文档没写」 | 判断文档写的**退出码语义**是否与代码一致（那只由人工与本测试之外的用例保证） |
| 检出「文档写了检查项、代码里没有」 | 判断耗时数字是否仍然准确 |
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "REPRODUCE.md"
ALL = REPO / "scripts" / "check_all.py"


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("REPRODUCE.md not present")
    return DOC.read_text(encoding="utf-8")


def _code_check_names() -> list:
    """从 check_all.py 的 checks 列表里抽出检查项名（**不手抄**）。"""
    src = ALL.read_text(encoding="utf-8")
    return re.findall(r'\(\s*"([^"]{2,20})"\s*,\s*\[str\(SCRIPTS', src)


def test_code_exposes_the_check_names() -> None:
    assert ALL.is_file(), "缺 scripts/check_all.py"
    names = _code_check_names()
    assert len(names) >= 5, "从 check_all.py 只抽到 %d 个检查项" % len(names)


def test_every_code_check_is_named_in_the_doc() -> None:
    """**核心（防文档落后）**：代码里的每个检查项名都要在文档出现。"""
    names = _code_check_names()
    doc = _doc()
    missing = [n for n in names if n not in doc]
    assert not missing, "文档未提到这些检查项: %s" % missing


def test_doc_does_not_claim_checks_that_do_not_exist() -> None:
    """**反向**：文档表格里列的脚本都必须在代码里**真的被调用**。

    ⚠ `check_all.py` **本身是入口**，不是它调用的检查项 —— 故排除。
    （首版没排除 → 报"文档提到的脚本未被调用"，是本测试自己的假阳性。）
    """
    doc = _doc()
    code = ALL.read_text(encoding="utf-8")
    scripts = sorted(set(re.findall(r"`(check_[a-z_]+\.py|audit_tasklist_landing\.py|"
                                    r"grade_landing_refs\.py|which_side\.py)`", doc)))
    scripts = [s for s in scripts if s != "check_all.py"]
    assert scripts, "文档未列出任何核对脚本"
    missing = [s for s in scripts if s not in code]
    assert not missing, "文档提到的脚本未被 check_all.py 调用: %s" % missing


def test_doc_mentions_the_environment_preflight() -> None:
    """**关键**：必须写明入口**先报环境**（§135/§136 的落实）。"""
    doc = _doc()
    assert "which_side" in doc
    assert "先报环境" in doc or "先打印" in doc


def test_doc_states_why_environment_comes_first() -> None:
    """必须写明**为何**先报环境（否则会被当成可删的装饰）。"""
    doc = _doc()
    assert "2,000" in doc or "2000" in doc
    assert "95,000" in doc or "95000" in doc
    assert "找不到" in doc


def test_doc_documents_the_grading_levels() -> None:
    """分级核对的五级应全部写明（含只有 MISS 需人工）。"""
    doc = _doc()
    for lv in ("OK", "NOTE", "PLAN", "EXT", "MISS"):
        assert lv in doc, "未写明分级 %s" % lv
    assert "仅 `MISS` 需人工" in doc or "只有 `MISS`" in doc or "MISS` 需人工" in doc


def test_doc_records_the_grading_counts() -> None:
    """分级结果的实际数字应记录（便于日后对比是否变化）。"""
    doc = _doc()
    assert "158" in doc, "未记录引用总数"
    assert "101" in doc and "39" in doc and "13" in doc, "未记录分级计数"
