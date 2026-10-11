"""`check_all.py` 的测试：入口必须**先报环境**、**正确标记**、**不把没跑的当通过**。

## 为什么需要

`check_all.py` 是本项目**唯一的核对入口**，而它**此前没有任何测试**。
本会话已记录：**已提交但未测的检测器可能静默失效**（§110 的反模式检测器一直在跳过文件）。

## 重点（全部用**合成的最小仓库**，不依赖真实仓库）

| # | 断言 |
| --- | --- |
| 1 | 入口能跑通，并**打印环境判明** |
| 2 | 缺脚本时报**缺失**（不静默跳过） |
| 3 | `rc=2` 标为 **`ENV`**，**不得**标 `OK` |
| 4 | 慢项在无 `--slow` 时**跳过**并说明 |
| 5 | 结果表里出现**每一条检查项** |
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "scripts" / "check_all.py"


def _run(cwd: Path, *args):
    return subprocess.run([sys.executable, str(cwd / "scripts" / "check_all.py"), *args],
                          cwd=cwd, capture_output=True, encoding="utf-8", errors="replace",
                          timeout=300)


def _make_sandbox(root: Path, scripts: dict) -> None:
    """造一个最小仓库：scripts/ + 若干假核对脚本。"""
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ENTRY, root / "scripts" / "check_all.py")
    for name, body in scripts.items():
        p = root / "scripts" / name
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)


def test_entry_exists() -> None:
    assert ENTRY.is_file(), "缺 scripts/check_all.py"


def test_environment_preflight_is_run_when_present(tmp_path: Path) -> None:
    """**核心 1**：`which_side.py` 存在时，入口必须打印环境判明（§135/§136 的落实）。

    ⚠ 反向也成立：`which_side.py` **不存在时入口不报错也不打印** ——
    故本测试显式提供该脚本，避免把"没提供"误判成"没实现"。
    """
    _make_sandbox(tmp_path, {"which_side.py": "print('  这一侧 = 测试侧')\n"})
    out = ((_run(tmp_path).stdout or "") + (_run(tmp_path).stderr or ""))
    assert "统一核对" in out, out[-300:]
    assert "这一侧" in out, "提供了 which_side.py 却未打印环境判明"


def test_missing_preflight_does_not_break_the_entry(tmp_path: Path) -> None:
    """**容错**：`which_side.py` 缺失时入口仍应跑完并报「缺失」，不得崩溃。"""
    _make_sandbox(tmp_path, {})
    r = _run(tmp_path)
    out = (r.stdout or "") + (r.stderr or "")
    assert "统一核对" in out
    assert "Traceback" not in out, "缺 which_side.py 导致入口崩溃"


def test_missing_script_is_reported_not_skipped(tmp_path: Path) -> None:
    """**核心 2**：脚本缺失必须报「缺失」，不能静默略过。"""
    _make_sandbox(tmp_path, {})
    out = ((_run(tmp_path).stdout or ""))
    assert "缺失" in out, "缺脚本时未报「缺失」"


def test_rc2_is_labelled_env_not_ok(tmp_path: Path) -> None:
    """**核心 3**：rc=2 = 本机缺输入 → 标 `ENV`，**不得**标 `OK`（§132 的回归防线）。"""
    _make_sandbox(tmp_path, {
        "check_deliverable_health.py":
            "import sys\nprint('  缺主清单')\nsys.exit(2)\n",
        "which_side.py": "print('  这一侧 = 测试')\n",
    })
    out = (_run(tmp_path).stdout or "")
    lines = [l for l in out.split("\n") if "交付物健康" in l]
    assert lines, "结果表缺该行"
    assert "ENV" in lines[0], "rc=2 未标 ENV：%r" % lines[0]
    assert " OK " not in lines[0], "rc=2 被标成了 OK"


def test_rc0_is_labelled_ok(tmp_path: Path) -> None:
    _make_sandbox(tmp_path, {
        "check_metric_coverage.py": "import sys\nprint('  全过')\nsys.exit(0)\n",
        "which_side.py": "print('  这一侧 = 测试')\n",
    })
    out = (_run(tmp_path).stdout or "")
    lines = [l for l in out.split("\n") if "指标覆盖" in l]
    assert lines and "OK" in lines[0], lines


def test_rc1_outside_expected_is_flagged(tmp_path: Path) -> None:
    """退出码不在预期集合 → 标 `注意`，不标 `OK`。"""
    _make_sandbox(tmp_path, {
        "check_metric_coverage.py": "import sys\nprint('  有缺口')\nsys.exit(1)\n",
        "which_side.py": "print('  这一侧 = 测试')\n",
    })
    out = (_run(tmp_path).stdout or "")
    lines = [l for l in out.split("\n") if "指标覆盖" in l]
    assert lines and "注意" in lines[0], lines


def test_all_check_rows_appear_in_the_result_table(tmp_path: Path) -> None:
    """**核心 5**：每个检查项都要在结果表里出现（不能只跑不报）。"""
    _make_sandbox(tmp_path, {"which_side.py": "print('  这一侧 = 测试')\n"})
    out = (_run(tmp_path).stdout or "")
    for name in ("交付物健康", "指标覆盖", "待批准就绪度", "快照断言",
                 "落点分级", "断言反模式", "判据证据"):
        assert name in out, "结果表缺检查项: %s" % name


def test_timed_out_check_is_labelled(tmp_path: Path) -> None:
    """**防挂死**：超时应标 `TIMEOUT`，而不是让入口挂住。"""
    _make_sandbox(tmp_path, {
        "which_side.py": "print('  这一侧 = 测试')\n",
        "check_deliverable_health.py": "import time\ntime.sleep(400)\n",
    })
    p = tmp_path / "scripts" / "check_all.py"
    src = p.read_text(encoding="utf-8").replace("timeout: int = 120", "timeout: int = 3")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    out = (_run(tmp_path).stdout or "")
    assert "TIMEOUT" in out, "超时未标 TIMEOUT"


def test_reports_dont_judge_checks_are_named(tmp_path: Path) -> None:
    """非严格项必须在结果表里写明「不判合格」的理由。"""
    _make_sandbox(tmp_path, {"which_side.py": "print('  这一侧 = 测试')\n"})
    out = (_run(tmp_path).stdout or "")
    assert "不判" in out or "非严格" in out, "未说明哪些项不参与判合格"
