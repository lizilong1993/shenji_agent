"""I08 判据 3 的**执行级**证据：断网 → 恢复 → **标记陈旧区间** 的完整链路。

## 判据 3 原文

> 3) 注入一次进程异常后 **5s 内页面出现对应事件**，**断网 10 分钟缓存可回放**。

拆成两半：

| 半句 | 可核验性 |
| --- | --- |
| 「**断网 10 分钟…并在恢复后区分陈旧数据**」 | **可执行核验** —— `noteConnectivity(isOnline, nowMs)` **接受注入时间戳**，故 10 分钟断网**无需真的等** |
| 「5s 内**页面**出现」 | **需浏览器**（涉及渲染时序） |

本文件覆盖**前半**，并**明确标注**后半仍需浏览器。

## 为什么这是实质提升

此前 `I08` 的断言是**源码级**（"函数是否存在"），**无法**核验：
  · 断网**只报一次**（去重是否真生效）；
  · 恢复后**陈旧区间是否真被算出**（`stale_seconds` 是否等于 600）。

现在两者都由**执行**得到：注入 `T0` 与 `T0 + 600000`，读出 `stale_seconds == 600`。

## ⚠ 能力边界

**不是浏览器**：证明「逻辑与载荷正确」，**不证明**「5s 内渲染出来」。
"""
from __future__ import annotations

import json
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
T0 = 1_700_000_000_000          # 固定基准时刻，保证确定性
TEN_MIN = 10 * 60 * 1000


@pytest.fixture()
def ctx():
    """加载 theme.js + eventFeed.js + dataHealth.js 的干净上下文。"""
    c = quickjs.Context()
    c.eval(js_harness.SHIM)
    for f in ("theme.js", "eventFeed.js", "dataHealth.js"):
        p = SRC / f
        if not p.is_file():
            pytest.skip("%s not present" % f)
        c.eval(p.read_text(encoding="utf-8"))
    c.eval("window.EventFeed.clear();")
    return c


def _note(ctx, online: bool, t: int):
    return ctx.eval("window.DataHealth.noteConnectivity(%s, %d)"
                    % ("true" if online else "false", t))


def _events(ctx):
    return json.loads(ctx.eval("JSON.stringify(window.EventFeed.all())"))


# ══════════════ 前提

def test_data_health_loads_and_exposes_the_hooks(ctx) -> None:
    """前提：`noteConnectivity` / `staleInfo` 真的可调用。"""
    assert ctx.eval("typeof window.DataHealth") == "object"
    for fn in ("noteConnectivity", "staleInfo", "emitReport", "report"):
        assert ctx.eval("typeof window.DataHealth.%s" % fn) == "function", "缺 %s" % fn


# ══════════════ 断网：报一次且去重

def test_disconnect_emits_exactly_one_warning(ctx) -> None:
    """断网必须**只报一次** —— 反复断网不得刷屏（否则真告警会被淹没）。"""
    assert _note(ctx, False, T0) is None, "断网不应返回事件（只记录）"
    _note(ctx, False, T0 + 1000)
    _note(ctx, False, T0 + 2000)
    evs = _events(ctx)
    assert len(evs) == 1, "断网报了 %d 次，去重未生效" % len(evs)
    assert evs[0]["level"] == "warn", evs[0]
    assert evs[0]["source"] == "dataHealth"
    assert "断开" in evs[0]["content"]


def test_stale_info_reports_offline_state(ctx) -> None:
    """断网期间面板要能显示"数据陈旧" —— `staleInfo().offline` 必须为真。"""
    _note(ctx, False, T0)
    info = json.loads(ctx.eval("JSON.stringify(window.DataHealth.staleInfo())"))
    assert info["offline"] is True, info
    assert info["offlineSince"], "未记录断网起点"
    # 起点应是注入的那个时刻
    assert info["offlineSince"].startswith("2023-11-14"), info["offlineSince"]


# ══════════════ 恢复：算出陈旧区间（**判据 3 的核心**）

def test_recovery_computes_the_stale_interval(ctx) -> None:
    """**「断网 10 分钟…恢复后区分陈旧数据」的执行级核验。**

    注入 `T0` 与 `T0 + 600000`（相隔正好 10 分钟），
    恢复事件的 `raw.stale_seconds` 必须等于 **600** ——
    这是"界面能区分陈旧数据"所依赖的那个数。
    """
    _note(ctx, False, T0)
    _note(ctx, True, T0 + TEN_MIN)
    evs = _events(ctx)
    rec = [e for e in evs if "恢复" in e["content"]]
    assert len(rec) == 1, "恢复事件应有且只有一条: %s" % [e["content"] for e in evs]
    raw = rec[0]["raw"]
    assert raw["stale_seconds"] == 600, \
        "陈旧区间算错：期望 600 秒，实际 %s" % raw.get("stale_seconds")
    assert raw["offline_since"] and raw["recovered_at"], "未给出区间的两端"
    assert rec[0]["level"] == "info", rec[0]


def test_recovery_clears_the_offline_state(ctx) -> None:
    """恢复后必须**清掉**断网状态，否则面板会一直显示"陈旧"。"""
    _note(ctx, False, T0)
    _note(ctx, True, T0 + TEN_MIN)
    info = json.loads(ctx.eval("JSON.stringify(window.DataHealth.staleInfo())"))
    assert info["offline"] is False, info
    assert info["offlineSince"] is None, info
    assert info["staleSeconds"] == 0, info


def test_recovery_without_a_prior_disconnect_is_silent(ctx) -> None:
    """**无断网史上的"恢复"不得报事件** —— 否则会误报网络异常。"""
    _note(ctx, True, T0)
    assert _events(ctx) == [], "未断网却报了恢复事件"


def test_a_second_disconnect_after_recovery_is_reported_again(ctx) -> None:
    """恢复后**再次**断网必须重新报 —— 去重不能变成"永久静默"。"""
    _note(ctx, False, T0)
    _note(ctx, True, T0 + TEN_MIN)
    _note(ctx, False, T0 + TEN_MIN + 1000)
    evs = _events(ctx)
    off = [e for e in evs if "断开" in e["content"]]
    assert len(off) == 2, "第二轮断网被永久静默了: %s" % [e["content"] for e in evs]


def test_short_disconnect_interval_is_proportional(ctx) -> None:
    """陈旧区间必须**按实际时长**算，不能是常数。"""
    for minutes, want in ((1, 60), (30, 1800)):
        c = quickjs.Context()
        c.eval(js_harness.SHIM)
        for f in ("eventFeed.js", "dataHealth.js"):
            c.eval((SRC / f).read_text(encoding="utf-8"))
        c.eval("window.EventFeed.clear();")
        c.eval("window.DataHealth.noteConnectivity(false, %d);" % T0)
        c.eval("window.DataHealth.noteConnectivity(true, %d);" % (T0 + minutes * 60 * 1000))
        evs = json.loads(c.eval("JSON.stringify(window.EventFeed.all())"))
        rec = [e for e in evs if "恢复" in e["content"]][0]
        assert rec["raw"]["stale_seconds"] == want, \
            "%d 分钟断网期望 %d 秒，实际 %s" % (minutes, want, rec["raw"]["stale_seconds"])


# ══════════════ 事件本身的质量（与判据 1 衔接）

def test_connectivity_events_satisfy_the_six_field_contract(ctx) -> None:
    """断网/恢复事件也必须满足 I08 判据 1 的六字段契约。"""
    _note(ctx, False, T0)
    _note(ctx, True, T0 + TEN_MIN)
    for e in _events(ctx):
        for k in ("time", "level", "source", "object", "content", "raw"):
            assert k in e and e[k] is not None, "事件缺字段 %s: %s" % (k, e)
        assert e["object"] == "network", e


def test_capability_boundary_is_stated() -> None:
    """**能力边界必须写在文件里**：不得把执行级证据当成"5s 内页面出现"。"""
    doc = Path(__file__).read_text(encoding="utf-8")
    assert "需浏览器" in doc, "未声明哪一半需浏览器"
    assert "5s" in doc, "未点明判据 3 后半句（5s 内页面出现）需浏览器"
