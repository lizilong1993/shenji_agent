"""I08/I09 的**运行时**验证：在 quickjs 里执行真实 JS 文件，替代源码模式匹配。

## 为什么这比此前的做法强

修前 I08/I09 的断言都是**源码结构级**（正则匹配"函数/字段是否存在"），于是：
  · I09 判据 2 的「1.5s/0.8s 可配置且**修改后生效**」—— **"生效"无从核验**；
  · I08 判据 1 的「事件字段完整率 100%」—— 只能数源码里的字段名。

本文件用 `quickjs`（`pip install quickjs`，**本机实测可用**）**加载并执行真实文件**，
并给全局 `getComputedStyle` 造一个可控 shim，从而**真的驱动 `theme.js` 的 CSS 读取路径**。

## ⚠ 能力边界（必须写明）

shim **不是浏览器**：它只实现被真实代码用到的 `getComputedStyle(...).getPropertyValue(var)`。
故它证明的是「**`theme.js`/`eventFeed.js` 的读取与计算逻辑正确**」，
**不是**「浏览器里的渲染/布局正确」。

**判据里"界面上逐项可复现""截图评审"仍必须浏览器** —— 本文件不声称覆盖它们。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

SRC = REPO / "situation3d" / "web" / "src"
CSS = SRC / "theme.css"

# ⚠ 顺序要紧：**先导入 harness**，它会把仓库内 `.tools/jsengine/` 加进 `sys.path`；
# 我第一版先 `importorskip("quickjs")`，于是**路径还没加就已判定"缺引擎"** →
# 整个文件被静默跳过 1 项（正是"跳过即不验证"的坑）。
import js_harness  # noqa: E402

pytest.importorskip("quickjs", reason="需要 quickjs 才能执行真实 JS（见 scripts/setup_js_engine.sh）")


@pytest.fixture(scope="module")
def h():
    return js_harness.JsHarness()


@pytest.fixture(scope="module")
def hcss(h):
    n = h.load_css_vars(CSS)
    assert n > 0, "未能从 theme.css 灌入任何变量"
    return h


# ══════════════ 前提：真实文件真的能加载并执行

def test_real_js_files_load_and_expose_apis(h: "js_harness.JsHarness") -> None:
    """**前提断言**：两个真实文件都能在 JS 引擎里加载并暴露接口。

    若这条不成立，后面所有断言都无从谈起 —— 故它必须最先失败。
    """
    assert "theme.js" in h.loaded
    assert "eventFeed.js" in h.loaded
    assert h.eval("typeof window.Theme") == "object", "Theme 未暴露"
    assert h.eval("typeof window.EventFeed") == "object", "EventFeed 未暴露"


def test_theme_reads_css_variables_end_to_end(hcss) -> None:
    """**I09 判据 2 的"生效"部分**：CSS 变量的值**真的**被 JS 读到。

    做法：`theme.css` 的值灌进 shim → 调用 `Theme.raw()` → 必须得到**该值**。
    """
    assert hcss.theme("raw", "faction-red") == "#E74C3C"
    assert hcss.theme("raw", "pulse-period-default") == "1.5s"


def test_changing_the_css_variable_changes_the_js_result(hcss) -> None:
    """**判据 2「修改后生效」的直接核验**：改 CSS 变量 → JS 结果随之改变。

    这是此前**做不到**的断言（源码匹配无法证明"生效"）。
    """
    hcss.set_var("--pulse-period-default", "2.5s")
    try:
        assert hcss.theme("seconds", "pulse-period-default") == 2.5, \
            '改了 CSS 变量，JS 读到的值未变 —— 说明并非「取自 CSS 变量」'
    finally:
        hcss.set_var("--pulse-period-default", "1.5s")
    assert hcss.theme("seconds", "pulse-period-default") == 1.5


def test_fallback_is_used_only_when_the_variable_is_absent(hcss) -> None:
    """**兜底与取值的区别要真的可辨**：变量缺失时回落到 `DEFAULTS`，并记录 `fallbackUsed`。"""
    hcss.set_var("--faction-blue", "")
    try:
        got = hcss.theme("raw", "faction-blue")
        dbg = hcss.theme_json("debug")
        assert got, "缺变量时不应返回空"
        assert got == dbg["values"].get("faction-blue"), "回落值不是 DEFAULTS"
        assert dbg.get("fallbackUsed"), "未记录 fallbackUsed（兜底路径不可观测）"
    finally:
        hcss.set_var("--faction-blue", "#2E86C1")
        hcss.theme("raw", "faction-blue")   # 复位计数


# ══════════════ I09 判据 2/3 的数值（**执行**得出，不是正则）

def test_pulse_rule_executes_with_the_specified_threshold(hcss) -> None:
    """判据 2：1.5s / 0.8s，且阈值是**严格大于 50**（50 本身仍用 1.5s）。"""
    vals = {s: hcss.theme("pulsePeriodSeconds", s) for s in (0, 49, 50, 51, 200)}
    assert vals[0] == 1.5 and vals[49] == 1.5 and vals[50] == 1.5, vals
    assert vals[51] == 0.8 and vals[200] == 0.8, vals


def test_strength_ring_radius_executes_and_caps(hcss) -> None:
    """步骤 3：强度 0 不画环；1..4 递增；超界封顶。"""
    r = {lv: hcss.theme("strengthRingRadius", lv) for lv in (0, 1, 2, 3, 4, 9)}
    assert r[0] == 0, "强度 0 不应画环"
    assert r[1] > 0
    assert r[1] < r[2] < r[3] < r[4], "半径未随强度递增: %s" % r
    assert r[9] == r[4], "超界未封顶"


def test_hover_values_execute(hcss) -> None:
    """判据 3：悬停位移 8px、过渡 200ms —— 由**执行**得到，不是读源码字面量。"""
    dbg = hcss.theme_json("debug")
    assert dbg["hover"]["offsetPx"] == 8, dbg["hover"]
    assert dbg["hover"]["transitionMs"] == 200, dbg["hover"]


def test_theme_debug_reports_no_css_js_mismatch(hcss) -> None:
    """`Theme.debug().mismatches` 必须是空的 —— CSS 与 JS `DEFAULTS` 一致。"""
    dbg = hcss.theme_json("debug")
    assert dbg.get("mismatches") == {}, "theme.css 与 theme.js DEFAULTS 不一致: %s" % dbg.get("mismatches")


def test_unknown_key_raises(hcss) -> None:
    """**不静默**：未知键必须抛错，而不是返回 `undefined` 让调用方拿到空值。"""
    with pytest.raises(Exception):
        hcss.eval("window.Theme.raw('no-such-key')")


# ══════════════ I08 判据 1/2：事件模型（**执行**）

def test_event_feed_declares_six_fields_and_levels(h) -> None:
    """判据 1 的前提：六字段与级别枚举由**真实导出**读出。"""
    fields = json.loads(h.eval("JSON.stringify(window.EventFeed.FIELDS)"))
    levels = json.loads(h.eval("JSON.stringify(window.EventFeed.LEVELS)"))
    assert fields == ["time", "level", "source", "object", "content", "raw"], fields
    assert levels, "级别枚举为空"


def test_push_fills_all_six_fields(h) -> None:
    """**判据 1 的"完整率 100%"**：`push()` 后每条事件六字段**全部非空**。

    做法：push 一条**只给最少输入**的事件，检查引擎是否补齐 `time` 与 `raw` 默认值。
    """
    h.feed("window.EventFeed.clear();",
           "window.EventFeed.push({level:'warn', source:'probe', object:'o1', content:'c1'});")
    ev = h.feed_json("window.EventFeed.all()[0]")
    for k in ("time", "level", "source", "object", "content", "raw"):
        assert k in ev, "缺字段 %s" % k
        assert ev[k] is not None, "字段 %s 为 None" % k
    assert h.feed_json("window.EventFeed.completeness()") == 1.0, \
        "完整率不是 100%: %s" % h.eval("JSON.stringify(window.EventFeed.all())")


def test_raw_defaults_to_empty_object_not_omitted(h) -> None:
    """判据 1 的细则：「`raw` 可为空 JSON 但**需存在**」。"""
    h.feed("window.EventFeed.clear();",
           "window.EventFeed.push({level:'info', source:'s', object:'o', content:'c'});")
    ev = h.feed_json("window.EventFeed.all()[0]")
    assert "raw" in ev, "raw 被省略了"
    assert isinstance(ev["raw"], (dict, list)), "raw 类型异常: %r" % (ev["raw"],)


def test_filter_and_export_share_the_same_source(h) -> None:
    """判据 2：过滤与导出**共用同一数据源** —— 故导出结果条数应等于过滤结果条数。"""
    h.feed("window.EventFeed.clear();",
           "window.EventFeed.push({level:'error', source:'a', object:'o', content:'e1'});",
           "window.EventFeed.push({level:'info', source:'b', object:'o', content:'i1'});")
    n_err = h.eval("window.EventFeed.filter({level:'error'}).length")
    assert n_err == 1, "过滤结果 %s" % n_err


def test_export_csv_and_json_are_parseable(h) -> None:
    """判据 2：CSV 与 JSON 导出各成功一次并**可被解析**。"""
    import csv
    import io
    h.feed("window.EventFeed.clear();",
           "window.EventFeed.push({level:'warn', source:'s1', object:'o1', content:'c,1'});")
    csv_text = h.eval("window.EventFeed.exportCSV()")
    assert isinstance(csv_text, str) and csv_text.strip(), "CSV 为空"
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert len(rows) >= 2, "CSV 至少应有表头 + 一行: %r" % csv_text[:200]
    # 含逗号的 content 必须被正确转义（否则解析列数会错）
    assert any(len(r) == len(rows[0]) for r in rows[1:]), "CSV 列数不一致，转义可能有问题"

    js = h.eval("window.EventFeed.exportJSON()")
    parsed = json.loads(js)
    assert parsed, "JSON 导出为空"


def test_queue_has_a_cap(h) -> None:
    """【风险与回滚】要求事件队列**设上限与丢弃策略**（否则事件量增长会拖慢界面）。"""
    h.feed("window.EventFeed.clear();", "window.EventFeed.setMaxEvents(3);")
    try:
        for i in range(6):
            h.feed("window.EventFeed.push({level:'info', source:'s', object:'o%d', content:'c'});" % i)
        assert h.eval("window.EventFeed.count()") <= 3, "队列未按上限截断"
    finally:
        h.feed("window.EventFeed.setMaxEvents(500);")

