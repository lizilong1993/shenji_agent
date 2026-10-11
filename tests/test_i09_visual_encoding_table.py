"""I09 步骤 6：视觉编码对照表**由源码生成且可自校验**。

## 为什么要"生成 + 自校验"而不是手写

原文：「编写视觉编码对照表，逐项列出**编码维度、取值、对应文件与行号**，并留截图位置。」

**「文件与行号」是这份表最容易腐烂的部分** —— 手写一次，下次改代码就全错；
而**错的引用比没有引用更糟**（会让人按错行号去找，进而怀疑整份表）。

故：`scripts/gen_visual_encoding_table.py` **从源码抽取**变量名、取值与行号，
本文件再**独立核对**表与源码一致（生成器与校验器分开，避免"自己验自己"）。

## 本文件的能力边界

本环境**无浏览器、无 JS 运行时**，故：
- **能**断言：表与 `theme.css`/`theme.js`/`index.html` **逐项一致**（变量存在、取值相同、行号正确）；
- **不能**断言：界面上"逐项可复现"与"截图评审通过"（判据 1）—— 那需要前端环境。

**截图位置在本表中如实留为「待补」，不假装已评审。**
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
S3 = REPO / "situation3d"
SRC = S3 / "web" / "src"
TABLE = REPO / "docs" / "visual_encoding_table.md"
GEN = REPO / "scripts" / "gen_visual_encoding_table.py"
CSS = SRC / "theme.css"
JS = SRC / "theme.js"
HTML = S3 / "web" / "index.html"

DIMENSIONS = ("阵营", "类型", "状态", "强度", "时效")


def _css_vars() -> dict:
    """独立解析 `:root` 里的变量（**不复用生成器的函数**，以便交叉验证）。"""
    if not CSS.is_file():
        pytest.skip("theme.css not present")
    out = {}
    lines = CSS.read_text(encoding="utf-8").split("\n")
    in_root = False
    for i, l in enumerate(lines, 1):
        if re.match(r"\s*:root\s*\{", l):
            in_root = True
            continue
        if in_root and l.strip() == "}":
            in_root = False
            continue
        if in_root:
            m = re.match(r"\s*(--[a-z0-9-]+)\s*:\s*([^;]+);", l)
            if m:
                out[m.group(1)] = (m.group(2).strip(), i)
    return out


def _table() -> str:
    if not TABLE.is_file():
        pytest.skip("visual_encoding_table.md not present")
    return TABLE.read_text(encoding="utf-8")


def _table_rows():
    """抽出对照表第 2 节的数据行：(维度, 变量, 取值, 文件, 行号)。"""
    rows = []
    for m in re.finditer(
            r"^\|\s*(\S+)\s*\|\s*`(--[a-z0-9-]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|$",
            _table(), re.M):
        rows.append((m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))))
    return rows


# ────────────── 表存在且结构完整

def test_table_exists_and_names_five_dimensions() -> None:
    """【预期结果】要求**五类维度**各有唯一配置来源。"""
    t = _table()
    for d in DIMENSIONS:
        assert d in t, "对照表缺维度「%s」" % d


def test_table_has_rows_with_file_and_line() -> None:
    """步骤 6 明确要求列出「对应文件与行号」—— 行号不可省。"""
    rows = _table_rows()
    assert len(rows) >= 15, "对照表数据行过少（%d）" % len(rows)
    for dim, var, val, f, ln in rows:
        assert f.endswith(".css") or f.endswith(".js"), "文件列不像路径: %s" % f
        assert ln > 0, "行号必须为正: %s" % var


# ────────────── 表与源码逐项一致（核心）

def test_every_tabled_variable_exists_in_css() -> None:
    """表里的每个变量都必须在 `theme.css` **真实存在**。"""
    cv = _css_vars()
    missing = [v for _, v, _, _, _ in _table_rows() if v not in cv]
    assert not missing, "对照表列了 theme.css 中不存在的变量: %s" % missing


def test_every_tabled_value_matches_css() -> None:
    """表里的**取值**必须与 `theme.css` 一致（防止表写的是旧值）。"""
    cv = _css_vars()
    bad = {}
    for _dim, var, val, _f, _ln in _table_rows():
        if var in cv and cv[var][0] != val:
            bad[var] = (val, cv[var][0])
    assert not bad, "取值不一致（表 vs css）: %s" % bad


def test_every_tabled_line_number_points_at_that_variable() -> None:
    """**最关键的一条**：表里的行号必须**确实指向该变量**。

    这是"行号会腐烂"的直接守卫 —— 若代码增删一行而表未重新生成，本断言会失败。
    """
    cv = _css_vars()
    lines = CSS.read_text(encoding="utf-8").split("\n")
    bad = {}
    for _dim, var, _val, _f, ln in _table_rows():
        actual = lines[ln - 1] if 0 < ln <= len(lines) else ""
        if not re.match(r"\s*" + re.escape(var) + r"\s*:", actual):
            bad[var] = (ln, actual.strip()[:60])
    assert not bad, "行号未指向该变量（表已过期？）: %s" % bad


def test_table_is_reproducible_by_the_generator() -> None:
    """**生成器可重跑**：重新生成后内容应与当前完全一致。

    若不一致，说明有人手改了表（或源码变了但表没重生）—— 两者都应被发现。
    """
    if not GEN.is_file():
        pytest.skip("generator not present")
    before = _table()
    r = subprocess.run([sys.executable, str(GEN)], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    after = _table()
    assert before == after, "重跑生成器后表内容变化 —— 表与源码不同步"


# ────────────── 五个维度确有唯一来源

def test_each_dimension_maps_to_a_single_config_source() -> None:
    """【预期结果】「五类维度**各有唯一配置来源**」—— 不得同一语义两个来源。"""
    t = _table()
    # 每种语义只能出现在一个文件里（本表统一指向 theme.css）
    files = {}
    for _dim, _var, _val, f, _ln in _table_rows():
        files.setdefault(f, 0)
        files[f] += 1
    assert len(files) == 1, "对照表引用了多个配置文件，需确认是否同一语义有两个来源: %s" % files
    assert list(files)[0].endswith("theme.css")


def test_spec_mandated_values_are_present_and_exact() -> None:
    """判据 2/3 的**确定值**必须在表中且精确：1.5s / 0.8s / 8px / 200ms。"""
    rows = {v: val for _d, v, val, _f, _l in _table_rows()}
    for var, want in (("--pulse-period-default", "1.5s"),
                      ("--pulse-period-fast", "0.8s"),
                      ("--hover-offset", "8px"),
                      ("--hover-transition", "200ms")):
        assert rows.get(var) == want, "%s 期望 %s，表中为 %s" % (var, want, rows.get(var))


def test_pulse_threshold_is_fifty() -> None:
    """判据 2 的前提：阈值 50km/h（条文是「>50」，故 50 本身用默认周期）。"""
    rows = {v: val for _d, v, val, _f, _l in _table_rows()}
    assert rows.get("--pulse-speed-threshold-kmh") == "50"


# ────────────── 加载顺序与截图位置（如实留空）

def test_table_documents_script_load_order() -> None:
    """加载顺序错了只在运行时报错，故必须在表中可见。"""
    t = _table()
    assert "加载顺序" in t
    assert "theme.js" in t and "app.js" in t
    if HTML.is_file():
        h = HTML.read_text(encoding="utf-8")
        assert h.find("src/theme.js") < h.find("src/app.js"), "theme.js 必须在 app.js 之前"


def test_screenshot_section_is_honestly_left_pending() -> None:
    """**如实留空**：本环境无浏览器，故截图位置标「待补」，**不假装已评审**。"""
    t = _table()
    assert "截图" in t, "缺截图位置小节"
    assert "待补" in t, "截图位置未标明待补（会让人以为已评审）"
    assert "无浏览器" in t or "无 JS 运行时" in t, "未说明为何留空"


def test_generator_documents_why_it_generates() -> None:
    """生成器必须写明**为何生成而非手写**（否则后人会改为手写而重新引入腐烂）。"""
    if not GEN.is_file():
        pytest.skip("generator not present")
    s = GEN.read_text(encoding="utf-8")
    assert "错的引用比没有引用更糟" in s, "未写明生成的理由"
    assert "行号" in s
