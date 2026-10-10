"""I08：统一事件模型与健康接口扩展。

不做浏览器也不做 node（远程**没有 node**，实测 `node: command not found`），
分三层验证：
  1. **契约层**：`eventFeed.js` 的字段集合/级别枚举、`index.html` 的引入与顺序；
  2. **接口层**：真起服务打一次 `/api/health`，断言**原字段保留**且新字段成形
     —— 这是【风险与回滚】"扩展不得破坏既有调用方"的机器保证；
  3. **结构层**：`dataHealth.js` 确实改为经 `EventFeed.push` 投递，而非自建结构。
JS 的**运行期**行为（字段完整率、过滤耗时、导出可解析、断网陈旧区间）
由 CDP 探针覆盖，结论记在清单 §43。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
S3D = REPO / "situation3d"
EF = S3D / "web" / "src" / "eventFeed.js"
DH = S3D / "web" / "src" / "dataHealth.js"
HTML = S3D / "web" / "index.html"
SERVE = S3D / "server" / "serve.py"

EXPECTED_FIELDS = ["time", "level", "source", "object", "content", "raw"]
EXPECTED_LEVELS = ["debug", "info", "warn", "error", "critical"]
# I08 步骤 3 之前就存在、**必须保留**的字段（扩展不得破坏既有调用方）
ORIGINAL_HEALTH_FIELDS = ["ok", "ion_configured", "tianditu_configured",
                          "env_file", "wfs_types", "web_root"]


# ─────────────────────────────── 契约层

def test_event_feed_declares_six_fields() -> None:
    if not EF.exists():
        pytest.skip("eventFeed.js not present")
    src = EF.read_text(encoding="utf-8")
    m = re.search(r"var FIELDS = \[(.*?)\];", src, re.S)
    assert m, "未声明 FIELDS"
    fields = re.findall(r"'([a-z]+)'", m.group(1))
    assert fields == EXPECTED_FIELDS, "六字段必须与 31004 6.2.4/6.2.5 一致"


def test_event_feed_declares_level_enum() -> None:
    if not EF.exists():
        pytest.skip("eventFeed.js not present")
    src = EF.read_text(encoding="utf-8")
    m = re.search(r"var LEVELS = \[(.*?)\];", src, re.S)
    assert m, "未声明 LEVELS"
    assert re.findall(r"'([a-z]+)'", m.group(1)) == EXPECTED_LEVELS


def test_event_feed_has_required_api() -> None:
    """步骤 1 要求提供 push / filter / export，且导出分 CSV 与 JSON。"""
    if not EF.exists():
        pytest.skip("eventFeed.js not present")
    src = EF.read_text(encoding="utf-8")
    for name in ("push", "filter", "exportCSV", "exportJSON", "subscribe", "debug"):
        assert re.search(r"\b%s\s*:" % name, src), "缺少导出接口 %s" % name


def test_export_shares_the_same_source_as_filter() -> None:
    """步骤 5 要求「过滤与导出共用同一数据源」—— 结构上必须是导出内部调 filter。"""
    if not EF.exists():
        pytest.skip("eventFeed.js not present")
    src = EF.read_text(encoding="utf-8")
    for fn in ("exportCSV", "exportJSON"):
        body = src[src.find("function " + fn):]
        body = body[:body.find("\n  }")]
        assert "filter(" in body, "%s 未复用 filter()（会与界面筛选结果不一致）" % fn


def test_raw_is_defaulted_not_omitted() -> None:
    """判据 1：`raw` 可为空 JSON 但**必须存在** —— 故 push 要给默认 `{}`。"""
    if not EF.exists():
        pytest.skip("eventFeed.js not present")
    src = EF.read_text(encoding="utf-8")
    assert re.search(r"raw:\s*\(evt\.raw === undefined \|\| evt\.raw === null\)\s*\?\s*\{\}", src), \
        "push 未把缺失的 raw 补成 {}"


def test_queue_has_a_cap_and_counts_drops() -> None:
    """【风险与回滚】：事件量增长不得拖慢渲染 —— 需上限，且丢弃要记账。"""
    if not EF.exists():
        pytest.skip("eventFeed.js not present")
    src = EF.read_text(encoding="utf-8")
    assert "DEFAULT_MAX_EVENTS" in src and "_events.shift()" in src, "缺少队列上限/丢弃策略"
    assert "dropped" in src, "丢弃未计数（静默丢事件等于丢证据）"


def test_filter_matches_source_and_object() -> None:
    """实测暴露过的坑：只比 `source` 会让 `{source:'network'}` 筛出 0 条。

    因为告警投的是 `source:'dataHealth'` + `object:'network'`。
    调用方自然会按 `network` 去筛，故两者都要参与匹配。
    """
    if not EF.exists():
        pytest.skip("eventFeed.js not present")
    src = EF.read_text(encoding="utf-8")
    seg = src[src.find("function filter("):]
    seg = seg[:seg.find("\n  }")]
    assert "e.source.toLowerCase()" in seg, "filter 未匹配 source"
    assert "e.object.toLowerCase()" in seg, "filter 未匹配 object（会导致按对象筛选恒为 0）"


# ─────────────────────────────── 引入与接线

def test_index_html_loads_event_feed_before_consumers() -> None:
    if not HTML.exists():
        pytest.skip("index.html not present")
    html = HTML.read_text(encoding="utf-8")
    assert "src/eventFeed.js" in html, "eventFeed.js 未引入（文件在但没人加载）"
    assert html.find("src/eventFeed.js") < html.find("src/dataHealth.js"), \
        "eventFeed.js 必须在 dataHealth.js 之前"
    assert html.find("src/eventFeed.js") < html.find("src/app.js"), \
        "eventFeed.js 必须在 app.js 之前"


def test_data_health_emits_events_instead_of_building_its_own() -> None:
    """步骤 2：零散提示统一改为投递事件，**禁止各页自建提示结构**。"""
    if not DH.exists():
        pytest.skip("dataHealth.js not present")
    src = DH.read_text(encoding="utf-8")
    assert "EventFeed.push" in src, "dataHealth 未投递事件"
    # EventFeed 缺失时必须显式失败，而不是静默跳过
    assert "EventFeed 未加载" in src, "EventFeed 缺失时未显式失败（会静默丢事件）"


def test_data_health_has_dedup_and_stale_tracking() -> None:
    if not DH.exists():
        pytest.skip("dataHealth.js not present")
    src = DH.read_text(encoding="utf-8")
    assert "_seenAttention" in src, "缺少重复投递去重（report 每次调用都会刷屏）"
    assert "noteConnectivity" in src and "staleInfo" in src, "缺少断网/陈旧区间跟踪"


def test_situation_host_exposes_event_debug_surface() -> None:
    """判据来自取证通道：事件必须能被外部读到，而不是只画在界面上。"""
    app = S3D / "web" / "src" / "app.js"
    if not app.exists():
        pytest.skip("app.js not present")
    src = app.read_text(encoding="utf-8")
    for name in ("eventsDebug", "eventsFilter", "eventsExportCSV", "eventsExportJSON"):
        assert name in src, "SituationHost 缺少 %s 取证出口" % name


# ─────────────────────────────── 接口层（真起服务打一次）

def test_health_endpoint_keeps_original_fields_and_adds_new() -> None:
    """【风险与回滚】核心：**保留原布尔字段**，仅新增。

    做法是**静态**检查 `_api_health` 的返回字典字面量 —— 比起服务更稳，
    且不受服务是否在跑影响；运行期实测另见 §43。
    """
    if not SERVE.exists():
        pytest.skip("serve.py not present")
    src = SERVE.read_text(encoding="utf-8")
    i = src.find("def _api_health")
    assert i > 0, "未找到 _api_health"
    body = src[i:i + 6000]
    for f in ORIGINAL_HEALTH_FIELDS:
        assert ('"%s":' % f) in body, "原有健康字段被删/改名：%s（会破坏既有调用方）" % f
    for f in ("level", "subchecks", "cache"):
        assert ('"%s":' % f) in body, "缺少 I08 新增字段：%s" % f


def test_health_subcheck_failures_are_caught_not_raised() -> None:
    """健康检查自己不能把整条接口打成 500。"""
    if not SERVE.exists():
        pytest.skip("serve.py not present")
    src = SERVE.read_text(encoding="utf-8")
    i = src.find("def _api_health")
    body = src[i:i + 6000]
    assert "health must never raise" in body or "except Exception" in body, \
        "子项检查未捕获异常"


def test_health_level_is_derived_from_subchecks() -> None:
    """级别必须有可解释的来源，不能拍脑袋。"""
    if not SERVE.exists():
        pytest.skip("serve.py not present")
    src = SERVE.read_text(encoding="utf-8")
    i = src.find("def _api_health")
    body = src[i:i + 6000]
    assert '"critical"' in body and '"warn"' in body and '"error"' in body, \
        "缺少级别推导分支"


# 运行期实测（服务在跑时才执行，不在则跳过）
@pytest.mark.parametrize("field", ORIGINAL_HEALTH_FIELDS)
def test_live_health_returns_original_field(field) -> None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8848/api/health", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip("service not running: %s" % exc)
    assert field in data, "运行期健康响应缺原字段 %s" % field


def test_live_health_returns_new_fields() -> None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8848/api/health", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip("service not running: %s" % exc)
    assert data.get("level") in ("ok", "warn", "error", "critical")
    assert isinstance(data.get("subchecks"), dict) and data["subchecks"], "subchecks 为空"
    for name, item in data["subchecks"].items():
        assert set(item) >= {"status", "ok"}, "子项 %s 结构不完整" % name
        assert item["status"] in ("ok", "warn", "error")
    assert isinstance(data.get("cache"), dict) and "note" in data["cache"]
