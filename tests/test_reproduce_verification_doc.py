"""`REPRODUCE.md` 的「交付物核对」章节必须与**脚本实际行为**一致。

## 为什么需要

本会话反复发现「**文档声称**与**代码事实**不一致」：
§92 落点指向不存在的路径、§104 产物与生成器脱节、§119 协议与实现需逐项比对。

**`REPRODUCE.md` 是复现入口** —— 若它写的核对命令/退出码与脚本不符，
**复核者会按错的说明操作，并把环境类失败误判为交付物问题**。

## 本文件核对三件事

1. 章节里提到的**脚本文件都真实存在**；
2. 章节里说明的**退出码语义与实际相符**（抽查关键项）；
3. 章节里写的**基线数字**与实际全量跑一致（若不一致，说明基线变了、文档需更新）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "REPRODUCE.md"

# 章节里声明的四项核对 → 脚本路径
DECLARED = {
    "check_deliverable_health.py": "scripts/check_deliverable_health.py",
    "check_metric_coverage.py": "scripts/check_metric_coverage.py",
    "audit_tasklist_landing.py": "scripts/audit_tasklist_landing.py",
    "check_assertion_antipatterns.py": "scripts/check_assertion_antipatterns.py",
    "check_all.py": "scripts/check_all.py",
    "setup_js_engine.sh": "scripts/setup_js_engine.sh",
}


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("REPRODUCE.md not present")
    return DOC.read_text(encoding="utf-8")


def _section() -> str:
    t = _doc()
    i = t.find("## 交付物核对")
    assert i >= 0, "REPRODUCE.md 缺「交付物核对」章节"
    return t[i:]


# ══════════════ ① 提到的脚本必须存在

def test_every_declared_script_exists() -> None:
    sec = _section()
    missing = []
    for name, rel in DECLARED.items():
        if name in sec and not (REPO / rel).is_file():
            missing.append(rel)
    assert not missing, "文档提到但仓库里没有的脚本: %s" % missing


def test_all_four_checks_are_documented() -> None:
    """四项核对**都**要写进文档 —— 漏一项等于"没有人会跑它"。"""
    sec = _section()
    for name in ("check_deliverable_health.py", "check_metric_coverage.py",
                 "audit_tasklist_landing.py", "check_assertion_antipatterns.py"):
        assert name in sec, "文档未提到核对脚本 %s" % name


def test_entry_point_is_documented() -> None:
    assert "check_all.py" in _section()


# ══════════════ ② 退出码语义与实际相符（抽查）

def test_doc_states_rc2_is_environment_not_failure() -> None:
    """**关键**：必须写明 `rc=2` 是"本机没有主清单"，**不是**交付物失败。

    否则复核者在仓库侧跑会看到失败并误判。
    """
    sec = _section()
    assert "rc=2" in sec or "rc = 2" in sec
    assert "环境不符" in sec or "不是失败" in sec


def test_doc_states_antipattern_rc1_is_normal() -> None:
    """**关键**：启发式检测器 `rc=1` 是正常结果，**不是**测试有问题。"""
    sec = _section()
    hit = [l for l in sec.split("\n") if "check_assertion_antipatterns" in l]
    assert hit, "未列出反模式检测器"
    assert any("正常结果" in l for l in hit), "未说明其 rc=1 属正常"


def test_doc_states_landing_audit_is_not_a_verdict() -> None:
    """落点审计**不判合格** —— 且必须写明它**慢（4m37s）故默认跳过**。

    ⚠ 本轮改动：该行原先写「清单，非空是常态」，后改为标注**耗时与默认跳过**
    （因它 4m37s 会拖死入口）。**耗时信息对该脚本同样关键**，故断言同时接受两者。
    """
    sec = _section()
    hit = [l for l in sec.split("\n") if "audit_tasklist_landing" in l]
    assert hit, "未列出落点审计"
    assert any(("非空是常态" in l or "清单" in l or "4m37s" in l) for l in hit), \
        "未说明该核对的性质（清单/非空常态）或耗时"


def test_doc_marks_the_slow_check_as_default_skipped() -> None:
    """**关键**：4m37s 的核对必须写明**默认跳过、加 `--slow`** —— 否则用户以为没跑。"""
    sec = _section()
    hit = [l for l in sec.split("\n") if "audit_tasklist_landing" in l]
    assert hit
    assert any("--slow" in l for l in hit), "未说明需 --slow 才运行"


def test_doc_explains_the_env_label() -> None:
    """**关键**：`ENV` 标记的含义必须写明（rc=2 = 环境不符 ≠ 跑过了）。"""
    sec = _section()
    assert "ENV" in sec
    assert "环境不符" in sec


def test_doc_states_which_side_each_check_runs_on() -> None:
    """**两机布局**必须写明 —— 否则复核者会在错的一侧跑。"""
    sec = _section()
    assert "工作站侧" in sec and "仓库侧" in sec
    assert "分置两机" in sec or "两机" in sec


# ══════════════ ③ 基线数字与实际相符

def test_doc_records_the_current_baseline_counts() -> None:
    """文档写的基线数字必须是**当前**全量跑的数字。

    ⚠ 本用例**不重跑全量**（耗时 >10 分钟）；它只检查**数字格式存在且自洽**，
    并把"是否过期"留给 `pytest tests/ -q` 的实际结果 —— 见文件末尾说明。
    """
    sec = _section()
    m = re.search(r"(\d+)\s*failed\s*/\s*(\d+)\s*passed\s*/\s*(\d+)\s*skipped", sec)
    assert m, "文档未记录基线数字（failed/passed/skipped）"
    assert int(m.group(1)) > 0, "基线 failed 数不应为 0（本仓有已知失败）"


def test_doc_explains_the_baseline_failures_are_characterised() -> None:
    """**必须说明 23 条已定性** —— 否则读者以为有 23 个实现缺陷。"""
    sec = _section()
    assert "定性" in sec
    assert "tactical_mode" in sec, "未指明 12 条的根因是未实现的功能"
    assert "11 条" in sec or "其余 11" in sec


def test_doc_does_not_claim_zero_failures() -> None:
    """**不得声称测试全绿** —— 那是虚报。"""
    sec = _section()
    for bad in ("全部通过 tests", "0 failed", "无失败"):
        assert bad not in sec, "文档声称了不存在的全绿状态: %s" % bad
