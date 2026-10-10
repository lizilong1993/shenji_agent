"""I15 步骤 1 收尾：把「六类记录」**显式化**，并更正一处错误记录。

## 任务原文（`I15`【具体步骤】1）

> 定义 `replays\\schema.json`：**动作、裁决、时间线、智能体内部决策摘要、
> 随机种子、人类操作六类记录**。

## 本轮核对结果

| 六类 | 修前 | 修后 |
| --- | --- | --- |
| 动作 | `actions` 已声明 | ✓ |
| 裁决 | `judge_info` 已声明为裁决类，**但其 note 写「v1 中实测为空数组」——实测为错** | ✓ **note 已更正**（22 帧 / 24 条） |
| **时间线** | **只隐含在 `time`/`meta`，无显式声明** | ✓ 已声明 `frame_fields.timeline` |
| 智能体内部决策摘要 | `agent` + `dual_core.agent_data` | ✓ |
| 随机种子 | `meta.seed` + `dual_core.seed` | ✓ |
| 人类操作 | `human` + `dual_core.human_ops` | ✓ |

**差集只有「时间线」**；且存在**一处错误记录**（judge_info 被写成空数组）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.replay import Replay  # noqa: E402

SCHEMA = REPO / "replays" / "schema.json"
REAL = REPO / "replays" / "replay_1770632023.2594852.zip"

SIX = ("actions", "adjudication", "timeline", "agent_decisions", "seed", "human_ops")


def _schema() -> dict:
    if not SCHEMA.is_file():
        pytest.skip("schema not present")
    return json.loads(SCHEMA.read_text(encoding="utf-8"))


# ─────────────────── 六类显式声明

def test_schema_declares_exactly_six_record_types() -> None:
    """判据：schema 必须**显式**列出六类，使"六类"可被机器核对。"""
    d = _schema()
    rt = d.get("record_types")
    assert isinstance(rt, dict), "schema 缺 record_types 段（六类未显式化）"
    assert rt.get("count") == 6
    assert set(rt["types"]) == set(SIX), "六类不符: %s" % sorted(rt["types"])


def test_each_record_type_names_its_frame_path() -> None:
    """每类必须指向具体 `frame_path`，否则"声明"无法核对到数据。"""
    rt = _schema()["record_types"]["types"]
    for k, v in rt.items():
        assert v.get("frame_path"), "%s 未给 frame_path" % k
        assert v.get("zh"), "%s 未给中文名" % k


def test_timeline_field_is_declared() -> None:
    """**差集项**：时间线此前只隐含在 `time`，现须显式声明。"""
    ff = _schema()["frame_fields"]
    assert "timeline" in ff, "frame_fields 仍缺 timeline 声明"
    tl = ff["timeline"]
    assert tl.get("derived_from") == "time", "应说明它由 time 派生（不新造数据）"


def test_agent_seed_human_declared_absent_in_v1() -> None:
    """三条 v1.1 主线必须标 `present_in_v1: false`（如实记录缺口）。"""
    rt = _schema()["record_types"]["types"]
    for k in ("agent_decisions", "seed", "human_ops"):
        assert rt[k].get("present_in_v1") is False, "%s 的 present_in_v1 不实" % k
        assert rt[k].get("required_in") == ">=1.1.0"


# ─────────────────── 一处**错误记录**的更正

def test_judge_info_note_no_longer_claims_it_is_empty() -> None:
    """**更正**：原 note 写「v1 中实测为空数组」—— 实测为**错**。

    真实复盘里 22 帧有 judge_info、共 24 条。若允许该错误留着，
    后续会误判「裁决这一类缺失」并去"补录"本已存在的数据。
    """
    ji = _schema()["frame_fields"]["judge_info"]
    note = ji.get("note") or ""
    assert "实测更正" in note or "是错的" in note, "未更正错误 note: %s" % note[:80]
    assert "22" in note and "24" in note, "未记录更正后的实测数字"


def test_real_replay_actually_has_adjudication_data() -> None:
    """**用数据证明**更正是对的：真实复盘的 judge_info **非空**。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        recs = r.adjudication_records()
    assert len(recs) > 0, "judge_info 真的为空 —— 那 note 的更正需回退"


# ─────────────────── 可执行的六类出口

def test_timeline_returns_one_entry_per_frame() -> None:
    """`timeline()` 应逐帧给出时间线条目（不新造数据，来自 `time`）。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        tl = r.timeline()
        assert len(tl) == r.frame_count
        first = tl[0]
        assert "frame_index" in first and "cur_step" in first
        # 与逐帧 time 段一致（不新造）
        for entry in tl[:20]:
            f = r.frame(entry["frame_index"])
            assert entry["cur_step"] == (f.get("time") or {}).get("cur_step")


def test_record_type_coverage_reports_empty_types_honestly() -> None:
    """**不粉饰**：v1 复盘里三条 v1.1 主线应为空，且被如实列出。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        cov = r.record_type_coverage()
    assert cov["frame_count"] > 0
    for k in ("actions", "adjudication", "timeline"):
        assert cov[k] > 0, "%s 在 v1 里应有数据" % k
    for k in ("agent_decisions", "seed", "human_ops"):
        assert cov[k] == 0, "%s 在 v1 里不该有数据" % k
    assert set(cov["empty_types"]) == {"agent_decisions", "seed", "human_ops"}


def test_upgraded_replay_covers_five_of_six_types() -> None:
    """**升级后**应覆盖六类中的五类（`agent`/`seed`/`human` 补齐）。

    `human_ops` 只在有命令的帧上出现（条件必需），故按帧计可能为 0；
    此处只断言 `agent_decisions` 与 `seed` 已补齐。
    """
    if not REAL.is_file():
        pytest.skip("real replay not present")
    import importlib.util
    up = REPO / "scripts" / "upgrade_replay_v11.py"
    if not up.is_file():
        pytest.skip("upgrade tool not present")
    spec = importlib.util.spec_from_file_location("urv11", up)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "v11.zip"
        m.upgrade_replay(REAL, dst, seed=42, scenario="1231")
        with Replay(dst) as r:
            cov = r.record_type_coverage()
        assert cov["agent_decisions"] > 0, "升级后仍无 agent 决策段"
        assert cov["seed"] > 0, "升级后仍无 seed"
