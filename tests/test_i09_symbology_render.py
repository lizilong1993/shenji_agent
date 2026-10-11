"""I09 判据 1 的**执行级**证据：在 quickjs 里渲染 `symbology.js` 的 SVG，核对视觉编码。

## 为什么这比"对照表 + 源码匹配"强

`I09` 判据 1 要求「视觉编码对照表**逐项可在界面复现**且截图评审通过」。
"截图评审"必须浏览器，**但"对照表里的值是否真的进入了渲染输出"可以执行核验** ——
只要能把 `symbolSvg()` 跑起来并检查其 SVG 的 `fill`。

本文件即做这件事：**执行** `theme.js` + `symbology.js`，对每个阵营渲染符号，
再从 SVG 里**读回** `fill`，与 `theme.css` 的值逐一比对。

## ⚠ 能力边界

这**不等于**截图评审：它证明「编码值进入了 SVG」，
**不证明**「界面上看起来对」（配色是否可辨、方位是否合直觉、像素是否对齐）。
后者仍需浏览器。

## 本轮由此发现的**真实缺陷**

`FALLBACK_AFFILIATIONS` **只定义了 `red`/`blue`**，而 `symbology.js` 用
`Object.keys(AFFILIATIONS)` 循环去取 `Theme` 的值 —— 于是
**`green`/`neutral`/`unknown` 取不到值，静默回落到 `#6b7688`（状态边框色），阵营维度失效**。

**该缺陷由"执行后读回 SVG 的 fill"发现** —— 源码匹配看不出它
（键都在、函数都在、字面量都对，唯独**运行时结果错**）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / ".tools" / "jsengine"))

import js_harness  # noqa: E402  （先导入：它负责把仓库内引擎目录加进 sys.path）

pytest.importorskip("quickjs", reason="需要 quickjs（见 scripts/setup_js_engine.sh）")

import quickjs  # noqa: E402

SRC = REPO / "situation3d" / "web" / "src"
CSS = SRC / "theme.css"


def _css_root() -> dict:
    m = re.search(r":root\s*\{(.*?)\n\}", CSS.read_text(encoding="utf-8"), re.S)
    assert m, "theme.css 缺 :root"
    return {k.lstrip("-"): v.strip()
            for k, v in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", m.group(1))}


@pytest.fixture(scope="module")
def ctx():
    """加载 theme.js + symbology.js，并灌入 theme.css 的变量。"""
    c = quickjs.Context()
    c.eval(js_harness.SHIM)
    for f in ("theme.js", "symbology.js"):
        p = SRC / f
        if not p.is_file():
            pytest.skip("%s not present" % f)
        c.eval(p.read_text(encoding="utf-8"))
    for k, v in _css_root().items():
        c.eval("window.__setVar(%s, %s);" % (json.dumps("--" + k), json.dumps(v)))
    return c


def _render(ctx, affiliation: str, icon: str = "destroyer", selected: bool = False) -> str:
    return ctx.eval("window.Symbology.symbolSvg(%s, %s, {selected: %s})"
                    % (json.dumps(icon), json.dumps(affiliation),
                       "true" if selected else "false"))


def _fills(svg: str) -> set:
    return {f.lower() for f in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)}


# ══════════════ 前提

def test_symbology_loads_and_renders(ctx) -> None:
    """前提：真实 `symbolSvg()` 能执行并产出 SVG 文本。"""
    svg = _render(ctx, "red")
    assert isinstance(svg, str) and svg.startswith("<svg"), svg[:80]
    assert "</svg>" in svg


# ══════════════ 阵营：对照表的值必须真的进入 SVG

@pytest.mark.parametrize("affiliation,key", [
    ("red", "faction-red"),
    ("blue", "faction-blue"),
    ("green", "faction-green"),
    ("neutral", "faction-neutral"),
])
def test_faction_colour_reaches_the_rendered_svg(ctx, affiliation, key) -> None:
    """**判据 1 的核心（阵营维度）**：`theme.css` 的阵营色必须出现在 SVG 的 `fill` 里。

    ⚠ 本用例会在**补全 `FALLBACK_AFFILIATIONS` 之前**失败 —— 那正是被发现的真实缺陷。
    """
    want = _css_root()[key].lower()
    got = _fills(_render(ctx, affiliation))
    assert want in got, \
        "阵营 %s 期望 %s 进入 SVG，实际 fill=%s（回落到别的颜色 = 阵营维度失效）" % (
            affiliation, want, sorted(got))


def test_all_declared_factions_have_a_fallback_entry() -> None:
    """**根因守卫**：`FALLBACK_AFFILIATIONS` 必须覆盖 `theme.css` 声明的**全部**阵营。

    否则 `Object.keys()` 循环覆盖不到，该阵营会静默回落到别色。
    （原实现只列 `red`/`blue`。）
    """
    src = (SRC / "symbology.js").read_text(encoding="utf-8")
    m = re.search(r"const FALLBACK_AFFILIATIONS = \{(.*?)\n  \};", src, re.S)
    assert m, "未找到 FALLBACK_AFFILIATIONS"
    body = m.group(1)
    declared = {k for k in re.findall(r'(--faction-[a-z]+)\s*:', CSS.read_text(encoding="utf-8"))}
    declared = {k.replace("--faction-", "") for k in declared}
    # contested 不是"阵营"（它是争夺状态），unknown 是阵营
    expected = declared - {"contested"}
    missing = [k for k in expected if not re.search(r"\b%s\s*:" % k, body)]
    assert not missing, "FALLBACK_AFFILIATIONS 缺阵营 %s（会静默回落）" % missing


def test_fallback_colours_match_theme_css() -> None:
    """兜底色必须与 `theme.css` 一致 —— 否则 `Theme` 不可用时会换成一个非标准色。"""
    src = (SRC / "symbology.js").read_text(encoding="utf-8")
    m = re.search(r"const FALLBACK_AFFILIATIONS = \{(.*?)\n  \};", src, re.S)
    root = _css_root()
    for k, v in re.findall(r"(\w+):\s*\{\s*color:\s*'(#[0-9a-fA-F]{6})'", m.group(1)):
        want = root.get("faction-" + k)
        if want:
            assert v.lower() == want.lower(), \
                "%s 兜底色 %s 与 theme.css %s 不一致" % (k, v, want)


# ══════════════ 对抗关系（帧形）

def test_hostile_uses_a_diamond_and_friendly_a_rectangle(ctx) -> None:
    """对照表「阵营」行的帧形语义：**友方=矩形、敌方=尖角菱形**（对抗关系靠帧形编码）。"""
    red = _render(ctx, "red")
    blue = _render(ctx, "blue")
    assert "<polygon" in red, "敌方（red）未使用多边形（菱形）帧形"
    assert "<rect" in blue, "友方（blue）未使用矩形帧形"


# ══════════════ 类型（图标形状）

def test_unknown_icon_falls_back_but_still_renders(ctx) -> None:
    """未知类型不得渲染成空 —— 必须走 `UNKNOWN_GLYPH` 兜底。"""
    svg = _render(ctx, "red", icon="no-such-icon")
    assert "<svg" in svg and len(svg) > 100
    assert _fills(svg), "未知类型没有任何 fill，疑似渲染成空"


# ══════════════ 状态（选中态）

def test_selected_adds_an_extra_ring(ctx) -> None:
    """对照表「状态」行：选中态要**额外**出现高亮环，且不得与状态边框混淆。"""
    plain = _render(ctx, "red", selected=False)
    sel = _render(ctx, "red", selected=True)
    assert len(re.findall(r"<", sel)) > len(re.findall(r"<", plain)), \
        "选中态未产生额外的图形元素"
    assert "#ffd24a" in sel.lower(), "选中态高亮环颜色缺失"


# ══════════════ 载体

def test_symbol_uri_is_usable_as_an_image_source(ctx) -> None:
    """`symbolUri()` 必须产出可用的图像源（data URI 或资源路径）。"""
    uri = ctx.eval("window.Symbology.symbolUri('destroyer','red',{})")
    assert isinstance(uri, str) and uri, "symbolUri 返回空"
    assert uri.startswith("data:image/svg+xml") or uri.startswith("./") or uri.startswith("/"), \
        "symbolUri 既不是 data URI 也不是路径: %s" % uri[:60]


def test_capability_boundary_is_documented() -> None:
    """**能力边界必须写在文件里** —— 否则后人会把"执行级证据"当成"截图评审"。"""
    doc = Path(__file__).read_text(encoding="utf-8")
    assert "不等于" in doc and "截图" in doc, "未声明能力边界"
    assert "仍需浏览器" in doc or "必须浏览器" in doc, "未说明哪部分需浏览器"
