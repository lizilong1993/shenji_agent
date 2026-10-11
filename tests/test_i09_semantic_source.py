"""I09【预期结果】：语义色**唯一来源** —— 三个文件已改经 `Theme` 取值。

## 核清的事实（修正上一轮的误判）

| 项 | 上一轮记录 | 本轮核实 |
| --- | --- | --- |
| `theme.js` 的 `DEFAULTS` | 「硬编码重复」 | **是文档化的"CSS 优先、内置兜底"回退**（`Theme.debug()` 报 **0 不一致**）—— **不是第二份定义** |
| `app.js` | 「仍硬编码」 | **已迁移**（引用 `Theme` 6 处，既有测试已断言） |
| `index.html` 加载序 | 未核 | `theme.js` 在 **253 行**，早于三个消费文件 → `window.Theme` **可用** |
| **真正未迁移的** | — | **`symbology.js` / `overview.js` / `plotting.js`** |

## 改法：**降级为兜底**，而不是删掉

三处改为 `_semantic(key, fallback)`：从 `Theme` 取值，**仅当 `Theme` 不可用**时退回字面量。
这与 `theme.js` 本来的口径一致（"CSS 变量优先，内置默认兜底"），
且**不会因 `Theme` 缺失而白屏**。

## ⚠ 本文件的能力边界

本环境**无浏览器、无 JS 运行时**，故断言是**源码结构级**：
能证明"取值改为经 `Theme`、字面量仅作兜底参数"；
**不能**证明运行时 `Theme` 真的被调用。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
S3 = REPO / "situation3d"
SRC = S3 / "web" / "src"
HTML = S3 / "web" / "index.html"

MIGRATED = ("symbology.js", "overview.js", "plotting.js")
SEMANTIC_HEX = re.compile(r"#(?:E74C3C|2E86C1|95A5A6|F1C40F|d0342c|1f5fbf|7d8a99)", re.I)


def _src(name: str) -> str:
    p = SRC / name
    if not p.is_file():
        pytest.skip("%s not present" % name)
    return p.read_text(encoding="utf-8")


# ────────────── 三个文件已改经 Theme

def test_migrated_files_define_the_semantic_helper() -> None:
    """三个文件都必须有 `_semantic()` 助手（经 `Theme` 取值 + 兜底）。"""
    for f in MIGRATED:
        s = _src(f)
        assert "function _semantic(" in s, "%s 缺 _semantic 助手" % f
        assert "window.Theme" in s, "%s 未引用 window.Theme" % f


def test_migrated_files_call_theme_for_faction_colours() -> None:
    """**关键**：阵营色必须**经 `_semantic` 取**，而不是直接写字面量。"""
    for f in MIGRATED:
        s = _src(f)
        # 允许**字面键**（'faction-red'）与**动态键**（'faction-' + k）两种形态。
        # ⚠ 我第一版只匹配字面键，于是把 symbology.js 的动态写法误判为"未迁移" ——
        # 与"查错地方就报缺失"同类。
        calls = re.findall(r"_semantic\(\s*'faction-[a-z]*'\s*(?:\+|,)", s)
        assert calls, "%s 没有经 _semantic 取任何阵营色" % f


def test_semantic_hex_literals_are_only_fallbacks() -> None:
    """标准色字面量**只能作为 `_semantic` 的第二个参数（兜底）**出现。

    这是"唯一来源"的实质断言：若某处仍把字面量**直接当值用**，
    那它就是"第二份定义"，改了 `theme.css` 也不会生效。
    """
    for f in MIGRATED:
        s = _src(f)
        offenders = []
        for ln in s.split("\n"):
            st = ln.strip()
            if st.startswith("//") or st.startswith("*"):
                continue
            if not SEMANTIC_HEX.search(st):
                continue
            # 合法形态：作为 _semantic 的第二参数
            if re.search(r"_semantic\(\s*'faction-[a-z]+'\s*,\s*'#[0-9a-fA-F]{6}'\s*\)", st):
                continue
            # `ink` 等**非语义**显示色可保留（它不是阵营主色）
            if "ink:" in st:
                continue
            offenders.append(st[:100])
        assert not offenders, "%s 仍直接使用标准色字面量: %s" % (f, offenders)


# ────────────── 加载顺序（改动的前提条件）

def test_theme_loads_before_the_consumers() -> None:
    """**改动的前提**：`theme.js` 必须先于三个消费文件加载，否则 `Theme` 取不到。

    这条是本改造最容易引入的运行时回归 —— 顺序错了只会在浏览器里报错。
    """
    if not HTML.is_file():
        pytest.skip("index.html not present")
    h = HTML.read_text(encoding="utf-8")
    i_theme = h.find("src/theme.js")
    assert i_theme > 0, "未加载 theme.js"
    for f in MIGRATED:
        i = h.find("src/" + f)
        if i < 0:
            continue
        assert i_theme < i, "theme.js 必须早于 %s（否则 Theme 未就绪）" % f


# ────────────── 唯一来源：theme.css 是应然来源

def test_theme_css_is_the_declared_single_source() -> None:
    """`theme.css` 的 `:root` 必须定义四个阵营色 —— 它才是唯一来源。"""
    css = SRC / "theme.css"
    if not css.is_file():
        pytest.skip("theme.css not present")
    s = css.read_text(encoding="utf-8")
    m = re.search(r":root\s*\{(.*?)\n\}", s, re.S)
    assert m, "theme.css 缺 :root"
    for k in ("--faction-red", "--faction-blue", "--faction-neutral", "--faction-unknown"):
        assert k in m.group(1), "唯一来源缺 %s" % k


def test_theme_js_defaults_agree_with_css() -> None:
    """`theme.js` 的兜底值必须与 `theme.css` 一致（既有测试亦要求）。

    二者若不一致，"兜底"就会在某次 `Theme` 不可用时**悄悄换成另一个颜色**。
    """
    css, js = SRC / "theme.css", SRC / "theme.js"
    if not (css.is_file() and js.is_file()):
        pytest.skip("theme files not present")
    m = re.search(r":root\s*\{(.*?)\n\}", css.read_text(encoding="utf-8"), re.S)
    cv = {k.lstrip("-"): v.strip()
          for k, v in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", m.group(1))}
    jd = dict(re.findall(r"['\"]([a-z][a-z0-9-]*)['\"]\s*:\s*['\"]([^'\"]+)['\"]",
                         js.read_text(encoding="utf-8")))
    mism = {k: (cv[k], jd[k]) for k in jd if k in cv and cv[k] != jd[k]}
    assert not mism, "theme.js 兜底值与 theme.css 不一致: %s" % mism


# ────────────── 友/敌映射（易被误读为"改反了"）

def test_nato_palette_maps_friendly_to_red_per_the_standard() -> None:
    """**按标准校正映射**：31004 §8.4.1「红方=我方，蓝方=敌方」。

    `plotting.js` 的调色板叫 `NATO_COLOR`，而 NATO 约定是 friendly=蓝、enemy=红，
    与我国标准**相反**。故此处 `friendly` 取 `faction-red`、`enemy` 取 `faction-blue`
    —— 这是**约定映射**，不是把两个色搞反。本断言把该映射钉住，避免后人"顺手改回来"。
    """
    s = _src("plotting.js")
    assert re.search(r"friendly:\s*_semantic\('faction-red'", s), \
        "friendly 未按标准取 faction-red"
    assert re.search(r"enemy:\s*_semantic\('faction-blue'", s), \
        "enemy 未按标准取 faction-blue"


def test_migration_did_not_remove_the_fallback() -> None:
    """**不得删掉兜底**：`Theme` 不可用时仍须有值，否则渲染会拿到 `undefined`。"""
    for f in MIGRATED:
        s = _src(f)
        # _semantic 必须有 fallback 形参并在异常时返回它
        assert re.search(r"function _semantic\(key,\s*fallback\)", s), \
            "%s 的 _semantic 缺 fallback 形参" % f
        assert "return fallback;" in s, "%s 的 _semantic 未返回 fallback" % f
