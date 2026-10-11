"""反模式检测器的**自证**测试：证明它**真的会报**，而不只是"跑起来不报错"。

## 为什么必须有这一层

检测器若"从不报任何东西"，在用法上与"没有检测器"等价 ——
**那正是它自己要捕获的 AP5（恒真断言）**。

故本文件的核心不是"它能运行"，而是 **"喂给它已知的反模式，它必须报出来"**。

## 同时测**反向**：正例不得误报

只测"会报"会导致另一个极端：见什么都报。
故也测**干净的代码不得被报** —— 两侧都测，检测器才有意义。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_assertion_antipatterns.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(), reason="check_assertion_antipatterns.py not present")


def _run(target: Path):
    return subprocess.run([sys.executable, str(SCRIPT), str(target)],
                          cwd=REPO, capture_output=True, text=True)


def _write(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "probe"
    d.mkdir(exist_ok=True)
    (d / "test_probe.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return d


# ══════════════ ① 真的会报（否则等价于没有检测器）

def test_detects_trivially_true_assertion(tmp_path: Path) -> None:
    """**AP5**：`assert x or True` 必须被报出。"""
    d = _write(tmp_path, '''
        def test_vacuous():
            v = {"a": 1}
            assert v.get("a") is None or True
    ''')
    r = _run(d)
    assert r.returncode == 1, "未报出恒真断言（rc=%d）:\n%s" % (r.returncode, r.stdout)
    assert "恒真" in r.stdout, r.stdout


def test_detects_self_comparison(tmp_path: Path) -> None:
    """**AP5**：`assert f(5) == f(5)` 必须被报出。"""
    d = _write(tmp_path, '''
        def test_self_compare():
            r = {"t": 1}
            assert r["t"] == r["t"]
    ''')
    r = _run(d)
    assert r.returncode == 1, r.stdout
    assert "自比较" in r.stdout, r.stdout


def test_detects_permission_bit_dependency(tmp_path: Path) -> None:
    """**AP8**：断言依赖文件权限位必须被报出（这正是 §98 的真实命中）。"""
    d = _write(tmp_path, '''
        import os
        import stat

        def test_mode():
            assert os.stat("x").st_mode & stat.S_IXUSR
    ''')
    r = _run(d)
    assert r.returncode == 1, r.stdout
    assert "权限位" in r.stdout, r.stdout


def test_detects_negative_literal_in_own_docstring(tmp_path: Path) -> None:
    """**AP6**：否定式断言的字面量出现在本文件 docstring 里 → 应报出。"""
    d = _write(tmp_path, '''
        def test_guard():
            """修前错误值是 legacy_bad_value_here，故断言它不在下面文本中。"""
            note = "现在用新值 legacy_bad_value_here 已被替换"
            assert "legacy_bad_value_here" not in note
    ''')
    r = _run(d)
    assert r.returncode == 1, r.stdout
    assert "docstring" in r.stdout, r.stdout


# ══════════════ ② 正例不得误报（否则"见谁都报"同样无用）

def test_clean_test_file_produces_no_hits(tmp_path: Path) -> None:
    """**反向**：一段干净的测试不得被报 —— 否则检测器只是在制造噪声。"""
    d = _write(tmp_path, '''
        def test_real_assertion():
            got = 2 + 2
            assert got == 4
            assert got > 3
            assert isinstance(got, int)
    ''')
    r = _run(d)
    assert r.returncode == 0, "干净代码被误报:\n%s" % r.stdout
    assert "未发现可疑断言" in r.stdout, r.stdout


def test_short_literal_in_negative_assertion_is_not_flagged(tmp_path: Path) -> None:
    """短字面量（如 `off`）与文档重合是**必然**的，不应报 —— 我第一版由此产生 515 条噪声。"""
    d = _write(tmp_path, '''
        def test_short():
            """模式有 on 与 off 两种。"""
            assert "off" not in "on"
    ''')
    r = _run(d)
    assert r.returncode == 0, "短字面量被误报:\n%s" % r.stdout


def test_explicit_nonexistence_assertion_is_not_flagged(tmp_path: Path) -> None:
    """**否定式**路径断言（"故意验证不存在"）不得报 —— 我第一版由此误报 2 条。"""
    d = _write(tmp_path, '''
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[1]

        def test_placeholder_absent():
            """docs/xxx.md 是占位符，不是真文件。"""
            assert (ROOT / "docs" / "xxx.md").is_file() is False
    ''')
    r = _run(d)
    assert r.returncode == 0, "故意的否例被误报:\n%s" % r.stdout


# ══════════════ ③ fail loud（版本与解析失败不得静默）

def test_version_below_39_is_refused_not_silently_skipped() -> None:
    """**AP2 的自查**：低版本必须**拒绝运行**（退出 2），而不是少扫几个文件。

    做法：用当前解释器跑，但把 `sys.version_info` 伪装成 3.8 不可行，
    故改为**读源码断言该守卫存在**（可核验的替代）。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    # ⚠ 我第一版用 `src.index("sys.version_info < (3, 9)")` 定位守卫，
    # 而该串**在文件头注释里也出现过** → 定位到注释、断言位置错误（测试误报）。
    # 改为**精确定位 `if` 语句行**。
    lines = src.split("\n")
    guard = [i for i, l in enumerate(lines)
             if l.strip().startswith("if sys.version_info < (3, 9)")]
    assert guard, "缺版本守卫（未找到 `if sys.version_info < (3, 9):` 语句）"
    assert "return 2" in src, "版本不足应返回 2"
    # ⚠ 我第二版用 `src.index("_iter_test_files(roots)")` 找"扫描处"，
    # 而该串**首先是函数定义（L113）** → 断言位置错乱。改为**定位调用行**。
    calls = [i for i, l in enumerate(lines)
             if "_iter_test_files(roots)" in l and not l.strip().startswith("def ")]
    assert calls, "未找到 _iter_test_files(roots) 的调用点"
    assert guard[0] < calls[0], \
        "版本守卫（L%d）必须在扫描调用（L%d）之前" % (guard[0] + 1, calls[0] + 1)


def test_parse_failure_is_fatal_not_skipped() -> None:
    """**AP2 的自查**：解析失败必须让整体失败，而不是"打印一行继续"。"""
    src = SCRIPT.read_text(encoding="utf-8")
    lines = src.split("\n")
    # ⚠ 我第一版取"解析失败"周围 200 字符，而 `return 3` 在更远处 → 误报"未 fail loud"。
    # 改为**结构化**：定位 except 块，检查其体内是否 return（而非 continue/pass）。
    i = next(k for k, l in enumerate(lines)
             if "except (SyntaxError" in l or "except SyntaxError" in l)
    body = []
    for l in lines[i + 1:]:
        if l.strip() and not l.startswith("            ") and not l.startswith("        #"):
            if not l.startswith("            "):
                break
        body.append(l)
    joined = "\n".join(body)
    assert "return 3" in joined, \
        "解析失败的 except 块未 fail loud（应 return 非 0），实际:\n%s" % joined[:300]
    assert "continue" not in joined, "解析失败仍被 continue 跳过（即漏检）"
    assert "漏检" in joined, "未说明'解析不了即漏检'的理由"


def test_detector_states_its_own_limits() -> None:
    """**能力边界必须写在文件里**：不得暗示"通过=有效"。"""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "启发式" in src, "未声明是启发式"
    assert "不能用" in src and "断言都有效" in src, "未声明'通过≠有效'"
    assert "误报" in src, "未声明会有误报"


# ══════════════ ④ 在本项目上跑得起来（回归）

def test_runs_on_this_repository_without_crashing() -> None:
    """在本仓 `tests/` 上实跑：**不得崩溃**（退出码只能是 0 或 1）。"""
    r = _run(REPO / "tests")
    assert r.returncode in (0, 1), \
        "检测器在本仓上崩溃（rc=%d）:\n%s\n%s" % (r.returncode, r.stdout[-800:], r.stderr[-400:])


def test_hit_count_is_small_enough_to_be_actionable() -> None:
    """**噪声即无效**：命中数必须小到可人工复核（我第一版 576 条，等于没报）。"""
    r = _run(REPO / "tests")
    if r.returncode == 0:
        pytest.skip("本次无命中")
    import re
    m = re.search(r"合计\s+(\d+)\s+处可疑", r.stdout)
    assert m, "输出缺少合计行:\n%s" % r.stdout[-500:]
    n = int(m.group(1))
    assert n <= 40, "命中 %d 条 —— 数量过大时使用者会忽略全部结果" % n
