"""`check_done_landing.py` 的测试：用**合成清单**验证它真的会报、且不误报。

## 两种模式（本文件都测）

| 模式 | 触发 | 裸文件名 | 真实清单耗时 |
| --- | --- | --- | --- |
| **默认（建索引）** | 无环境变量 | **能解析** | **约 18 s** |
| **快速（不建索引）** | 'DONE_LANDING_NO_INDEX=1' | **判为缺失**（无法解析） | **约 0.23 s** |

> **实测**：10 万+ 文件的全树遍历在本机（网络文件系统）要 18 s（`rglob` 版 **6m41s**），
> **故本测试用快速模式**（夹具里全用**相对路径**，无需 basename 回退）。
> 真实清单的**两种模式结果不同**（6 条 vs 19 条），**差异全部来自裸文件名** —— 这已记进 docstring。

## 为什么需要测试

上一轮该脚本只在**真实清单**上跑过一次（报 6 条缺失）。
**没有测试证明它在那 6 条之外不会误报、也不会漏报。**

> 与 §110 的教训同类：一个**已提交但未测**的检测器，可能静默地什么也不报。

## 夹具（三种情形）

| 用例 | 期望 |
| --- | --- |
| **已完成项指向存在的文件** | **不报**（防误报） |
| **已完成项指向不存在的文件** | **报出**（防漏报） |
| **阻塞项的"新增 X"不存在** | **不报**（计划产出，正常） |
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_done_landing.py"
DOC = REPO / "docs" / "_tasklist_for_check.md"

HEADER = """# 合成清单

## 3. 任务明细

### T1 存在的产物
- 【落点】`scripts/check_done_landing.py`
- 【状态】已完成

### T2 不存在的产物
- 【落点】`docs/definitely_missing_xyz.md`
- 【状态】已完成

### T3 阻塞项的计划产出
- 【落点】新增 `docs/never_created_abc.md`
- 【状态】阻塞-P0-2

## 4. 依赖关系

（略）
"""


def _write_doc(text: str) -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _run(no_index: bool = True):
    env = dict(os.environ)
    if no_index:
        env["DONE_LANDING_NO_INDEX"] = "1"
    else:
        env.pop("DONE_LANDING_NO_INDEX", None)
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO, env=env,
                          capture_output=True, encoding="utf-8", errors="replace")


@pytest.fixture(autouse=True)
def _isolate_and_cleanup():
    """**保证清理**：上一轮曾因清理不彻底把合成清单留在仓库里，
    导致下一次真实运行读到的是夹具内容。故此处 finally 必删。"""
    backup = DOC.read_text(encoding="utf-8") if DOC.is_file() else None
    try:
        yield
    finally:
        if DOC.is_file():
            DOC.unlink()
        if backup is not None:                      # 还原本来的真实副本
            with open(DOC, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(backup)
        else:
            # ⚠ 无 backup 说明**测试开始时就没有副本** —— 那时**必须确保不留残留**。
            # （test_missing_doc_is_reported_loudly 自己删掉了副本，
            #   而它断言脚本应报错退出；若此处不删，夹具内容会留在仓库里，
            #   导致下一次真实运行读到夹具。实测就这样残留过一次。）
            if DOC.is_file():
                DOC.unlink()


def test_script_present() -> None:
    assert SCRIPT.is_file(), "缺 scripts/check_done_landing.py"


# ══════════════ 快速模式（本测试使用）

def test_reports_a_missing_reference_from_a_completed_item() -> None:
    """**核心（防漏报）**：已完成项指向不存在的文件 → 必须报出。"""
    _write_doc(HEADER)
    out = (_run().stdout or "") + (_run().stderr or "")
    assert "definitely_missing_xyz.md" in out, "未报出缺失引用"


def test_does_not_report_an_existing_reference() -> None:
    """**核心（防误报）**：已完成项指向存在的文件 → 不得计入缺失。"""
    _write_doc(HEADER)
    r = _run()
    miss_lines = [l for l in (r.stdout or "").split("\n") if "缺失" in l]
    assert not any("check_done_landing.py" in l for l in miss_lines), \
        "把存在的文件报成了缺失"


def test_does_not_report_a_planned_output_of_a_blocked_item() -> None:
    """**关键**：阻塞项的「新增 X」是**计划产出**，不得计入缺失。"""
    _write_doc(HEADER)
    r = _run()
    miss_lines = [l for l in (r.stdout or "").split("\n") if "缺失" in l]
    assert not any("never_created_abc.md" in l for l in miss_lines), \
        "把阻塞项的计划产出报成了缺失"


def test_only_one_reference_is_missing() -> None:
    """夹具里**恰好一条**应缺失 —— 数量级也要对，防「多报/少报」。"""
    _write_doc(HEADER)
    r = _run()
    out = (r.stdout or "") + (r.stderr or "")
    assert "1 条引用缺失" in out, out[-400:]


def test_counts_are_reported() -> None:
    _write_doc(HEADER)
    out = (_run().stdout or "")
    assert "已完成项" in out and "缺失" in out


def test_exit_code_is_zero_even_when_references_are_missing() -> None:
    """**缺失不返回非零** —— 产出是**清单**，需人工判定性质（不是合格判定）。"""
    _write_doc(HEADER)
    assert _run().returncode == 0, "缺失时返回了非零，会被误当成「检测失败」"


def test_states_that_it_does_not_judge() -> None:
    """必须写明**不自动下结论** —— 否则会被读成「N 处失败」。"""
    _write_doc(HEADER)
    out = (_run().stdout or "") + (_run().stderr or "")
    assert "人工判定" in out or "不下结论" in out


def test_missing_doc_is_reported_loudly() -> None:
    """**不静默通过**：清单副本不存在时应报错退出，而不是「0 缺失」。"""
    if DOC.is_file():
        DOC.unlink()
    assert _run().returncode != 0, "清单缺失却返回 0（会被误读为「全部存在」）"


# ══════════════ 索引模式（慢，故只测行为差异，不测真实清单）

def test_index_mode_resolves_a_bare_filename_that_fast_mode_cannot() -> None:
    """**两种模式的差异必须被钉住**：裸文件名在索引模式下能解析、快速模式下不能。

    这条防的是「以为两种模式等价」—— 实测真实清单上是 **6 条 vs 19 条**，
    **差异全部来自裸文件名**。
    """
    _write_doc(HEADER.replace("`docs/definitely_missing_xyz.md`", "`check_done_landing.py`"))
    fast = _run(no_index=True).stdout or ""
    idx = _run(no_index=False).stdout or ""
    # 裸文件名：快速模式判缺失，索引模式不判
    assert "缺失" in fast
    assert "0 条引用缺失" in idx or "check_done_landing.py" not in idx.split("结论")[-1]
