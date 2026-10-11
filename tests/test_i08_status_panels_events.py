"""I08 步骤 2 收尾：`statusPanels.js` 的自建提示改为投递 `EventFeed`。

## 核清的事实（避免把"没找到"写成"不存在"）

| 文件 | 是否自建提示 | 处理 |
| --- | --- | --- |
| `dataHealth.js` | 已全部经 `EventFeed.push()` | **无需改** |
| `renderGovernor.js` | **无任何面向用户的提示**（只有内部 `reasons.push` 数组） | **无需改**（无可改造对象） |
| `statusPanels.js` | **有**：空态字符串 `未加载TS` / `未选择目标` 直接写进 DOM | **本轮改造** |

**故步骤 2 的实质缺口只有一处** —— 我先前以为三处，实际只有一处。

## ⚠ 本文件的验证能力边界（必须写明）

本环境**没有任何 JS 运行时**（`node`/`deno`/`bun`/`quickjs`/`dukpy` 均不可用，`npm` 也没有）。
故以下断言是**源码结构级**的，**不能**证明运行时行为正确 ——
这与 `tests/test_i08_event_model.py` 既有 16 条的约定一致。

**能证明的**：改造存在、两处空态都投递事件、去重与降级分支都在、导出接口存在、
括号配平（词法级）。

**不能证明的**：在浏览器里点击后事件**真的**进入队列、去重**真的**生效。
那需要 JS 运行时或浏览器，本环境不具备。
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "situation3d" / "web" / "src"
SP = WEB / "statusPanels.js"
RG = WEB / "renderGovernor.js"
DH = WEB / "dataHealth.js"
EF = WEB / "eventFeed.js"

EMPTY_STATES = ("未加载TS", "未选择目标")


def _src(p: Path) -> str:
    if not p.is_file():
        pytest.skip("%s not present" % p.name)
    return p.read_text(encoding="utf-8")


# ────────────── statusPanels 已改造

def test_status_panels_pushes_empty_states_into_the_event_feed() -> None:
    """两处空态都必须**同时**投递事件（DOM 占位可保留，但状态须入事件流）。"""
    s = _src(SP)
    for text in EMPTY_STATES:
        assert text in s, "空态文案丢失: %s" % text
        # 该文案所在行附近必须有 _emitEmptyState 调用
        i = s.index(text)
        ctx = s[max(0, i - 200):i + 300]
        assert "_emitEmptyState(" in ctx, \
            "空态 %s 未投递事件（仍是自建提示）" % text


def test_status_panels_declares_a_dedup_guard() -> None:
    """**去重必须有**：空态每帧重渲染，不去重会把事件队列刷满、淹没真告警。"""
    s = _src(SP)
    assert "_lastEmptyKey" in s, "无去重状态变量"
    # 去重判断必须在投递之前
    i_guard = s.index("_lastEmptyKey === content")
    i_push = s.index("EventFeed.push")
    assert i_guard < i_push, "去重判断不在投递之前"


def test_status_panels_degrades_without_crashing_when_feed_is_absent() -> None:
    """事件模型缺失时**不得让面板整体渲染失败**（此处是界面占位，与 dataHealth 的告警不同）。"""
    s = _src(SP)
    i = s.index("function _emitEmptyState")
    body = s[i:i + 1200]
    assert "console.warn" in body, "缺降级告警"
    assert "catch" in body, "缺 try/catch 兜底"


def test_status_panels_exposes_the_hook_for_testing() -> None:
    """暴露 `_emitEmptyState` / `_resetEmptyState`，使后续接入 JS 运行时时可测。"""
    s = _src(SP)
    assert "_emitEmptyState: _emitEmptyState" in s
    assert "_resetEmptyState" in s


# ────────────── 另两个文件：核实后**无需改**

def test_render_governor_has_no_user_facing_prompts() -> None:
    """**核实结论**：`renderGovernor.js` 无面向用户的提示 → 步骤 2 对它**无改造对象**。

    它的 `.push()` 是内部数组（`reasons`），不是提示结构。
    本断言把这个"已核实为无"钉住，避免后人重复排查。
    """
    s = _src(RG)
    for bad in ("innerHTML", "textContent", "EventFeed.push", "toast"):
        assert bad not in s, "renderGovernor 出现了提示结构 %s —— 需重新评估步骤 2" % bad
    assert "reasons.push" in s, "reasons.push 不见了 —— 文件结构已变，需重核"


def test_data_health_already_routes_through_the_event_feed() -> None:
    """`dataHealth.js` 已全部经 `EventFeed.push()` → 无需改。"""
    s = _src(DH)
    assert "window.EventFeed.push({" in s
    # 且**不静默**：事件模型缺失时抛错（告警不得悄悄消失）
    assert "throw new Error" in s


# ────────────── 词法级检查（无 JS 运行时时最接近"可执行性"的检查）

def test_edited_js_files_are_lexically_balanced() -> None:
    """**词法级**：括号配平 + 无未终止字符串。

    ⚠ 这**不是**语法校验，更不是行为验证 —— 本环境无 JS 运行时。
    它只排除最常见的低级错误（漏一个 `}`、少一个引号）。
    """
    for p in (SP, RG, DH):
        s = _src(p)
        for a, b in (("(", ")"), ("{", "}"), ("[", "]")):
            assert s.count(a) == s.count(b), "%s 里 %s%s 不配平" % (p.name, a, b)
        # tokenize 能查出未终止字符串（JS 正则字面量会误报，故只对**我改过的文件**用）
        try:
            list(tokenize.generate_tokens(io.StringIO(s).readline))
        except Exception as exc:  # noqa: BLE001
            pytest.fail("%s 词法检查失败: %s" % (p.name, exc))


def test_event_feed_export_surface_is_intact() -> None:
    """步骤 5 的导出接口仍在（我改的是消费方，不应影响 eventFeed 本身）。"""
    s = _src(EF)
    for name in ("push", "filter", "exportCSV", "exportJSON"):
        assert re.search(r"\b%s\s*:" % name, s), "eventFeed 缺接口 %s" % name


def test_index_html_loads_event_feed_before_status_panels() -> None:
    """**加载顺序**：`statusPanels.js` 依赖 `EventFeed`，必须在它之后加载。

    顺序错了会在运行时报 "EventFeed 未加载" —— 那是本改造最容易引入的回归。
    """
    html = REPO / "situation3d" / "web" / "index.html"
    if not html.is_file():
        pytest.skip("index.html not present")
    s = html.read_text(encoding="utf-8")
    i_feed = s.find("eventFeed.js")
    i_sp = s.find("statusPanels.js")
    assert i_feed > 0 and i_sp > 0, "两个脚本标签缺失"
    assert i_feed < i_sp, "eventFeed.js 必须在 statusPanels.js 之前加载"


def test_no_page_keeps_its_own_ad_hoc_prompt_structure() -> None:
    """**步骤 2 的实质要求**：禁止各页自建提示结构。

    检查三个落点文件里是否还有"直接写提示文案而不投递事件"的写法。
    """
    for p in (SP, RG, DH):
        s = _src(p)
        # 允许 DOM 空态，但空态必须同时投递事件
        for m in re.finditer(r"innerHTML\s*=\s*'[^']*<div class=\"(?:os|ts)-empty\">[^<]+</div>'", s):
            tail = s[m.end():m.end() + 300]
            assert "_emitEmptyState(" in tail, \
                "%s 有空态未投递事件: %s" % (p.name, m.group(0)[:60])
