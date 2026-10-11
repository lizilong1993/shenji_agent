"""I15 写入端：真实**历史复盘帧** → 写入端 → 校验 + 精确回放。

## ⚠ 本文件**不是**端到端验证（文件名保留了历史，但内容不是）

**独立评审（两位）各自独立指出本文件的原有陈述不实，现已更正：**

| 原陈述 | 事实 |
| --- | --- |
| 「用**真实引擎推演**产出的真实帧」 | 帧取自**历史复盘 zip** `replay_1770632023.2594852.zip`，**未运行引擎** |
| 「60 步 → 61 帧 / 800 步 → 801 帧」 | 无任何证据；`61`/`801` 恰为 `namelist[:60]`/`[:800]` 长度加一 |
| 「两次真实推演」 | **`run_offline_games.py` 至今从未端到端跑过** |

**本文件实际验证的是**：`land_wargame_sdk/offline_replay.write_offline_replay()`
**在给定一批帧时**能否产出通过 schema 校验的 v1.1.0 复盘，以及该产物是否支持按 tick 读取。
**它不验证**：`run_offline_games.py` 的接入是否可用（那需要**真跑一次对局**）。

## 保留下来的真实发现

历史复盘里 `actions` 与 `judge_info` 覆盖率很低（167 / 22 帧，共 1802 帧），
与第 50 轮 `I06` 探针「600 步仍未走出部署阶段」**同源**。
**该事实由历史复盘本身支持，不依赖任何未做的推演。**
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SDK = REPO / "land_wargame_sdk"
for _p in (str(REPO), str(SDK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REAL = REPO / "replays" / "replay_1770632023.2594852.zip"


def _states(n: int = 20):
    """真实 `state[GREEN]` 帧（用既有真实复盘作代理，结构相同）。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with zipfile.ZipFile(REAL) as z:
        names = z.namelist()[:n]
        return [json.loads(z.read(x).decode("utf-8")) for x in names]


# ─────────────────── 端到端：真实帧 → 写入端 → 校验

def test_historic_frames_round_trip_through_the_offline_writer(tmp_path: Path) -> None:
    """**历史**帧（非本次推演）经引擎侧写入端产出合法 v1.1.0 复盘。"""
    from offline_replay import write_offline_replay  # noqa: PLC0415
    from training.replay import validate  # noqa: PLC0415

    states = _states(20)
    out = tmp_path / "e2e.zip"
    rep = write_offline_replay(states, zip_name=str(out), run_id="e2e",
                               seed=20261010, scenario="1231")
    assert rep["frame_version"] == "1.1.0"
    assert rep["frames"] == len(states)
    v = validate(path=out)
    assert v["ok"] is True, v.get("missing_keys")
    assert v["declared_version"] == "1.1.0"
    assert v["missing_keys"] == []


def test_written_replay_supports_precise_indexed_playback(tmp_path: Path) -> None:
    """写出的复盘必须**可按 tick 精确回放**（判据 1 与写入端闭环）。"""
    from offline_replay import write_offline_replay  # noqa: PLC0415
    from training.replay import Replay  # noqa: PLC0415

    out = tmp_path / "e2e.zip"
    write_offline_replay(_states(20), zip_name=str(out), run_id="e2e", seed=1)
    with Replay(out) as r:
        res = r.verify_precise_replay()
        assert res["ok"] is True, res
        assert res["mismatches"] == []


def test_written_replay_carries_timeline_and_seed(tmp_path: Path) -> None:
    """写入端必须**逐帧**给出时间线、决策摘要与种子（本轮实测确认的三类）。"""
    from offline_replay import write_offline_replay  # noqa: PLC0415
    from training.replay import Replay  # noqa: PLC0415

    states = _states(20)
    out = tmp_path / "e2e.zip"
    write_offline_replay(states, zip_name=str(out), run_id="e2e", seed=42)
    with Replay(out) as r:
        cov = r.record_type_coverage()
    assert cov["frame_count"] == len(states)
    assert cov["timeline"] == len(states), "时间线未逐帧写入"
    assert cov["agent_decisions"] == len(states), "决策摘要未逐帧写入"
    assert cov["seed"] == len(states), "种子未逐帧写入"


# ─────────────────── 如实记录：空类是数据事实

def test_actions_and_adjudication_are_absent_in_this_scenario() -> None:
    """**如实断言**：`1231` 想定 + 规则基线下，真实复盘的
    `actions`/`judge_info` 覆盖率极低。

    本断言的作用是**把"空类"钉成已核实的事实**，
    使后人不会把它误读成写入端漏写：
    `replay_1770632023.2594852.zip` 是**真实引擎产物**，
    它自己也只有少数帧带这两类。
    """
    if not REAL.is_file():
        pytest.skip("real replay not present")
    from training.replay import Replay  # noqa: PLC0415
    with Replay(REAL) as r:
        cov = r.record_type_coverage()
    n = cov["frame_count"]
    assert n > 1000
    # 真实产物里这两类**有**数据，但远非逐帧
    assert 0 < cov["actions"] < n, "actions 覆盖率异常"
    assert 0 < cov["adjudication"] < n, "adjudication 覆盖率异常"
    # 而 v1.1 的三条主线在 **v1 产物**里必然为 0（那是缺口，不是漏写）
    assert cov["agent_decisions"] == 0
    assert cov["seed"] == 0
    assert cov["human_ops"] == 0


def test_historic_replay_shows_deployment_dominates_the_early_phase() -> None:
    """记录一个**历史产物事实**（**不是**我的推演结果）：

    这与第 50 轮 `I06` 的探针发现同源（600 步只产出 `END_DEPLOYMENT`/`DEPLOY_EMBARK`）。
    **本断言只核对既有历史产物**，不声称我做过任何推演。
    """
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with zipfile.ZipFile(REAL) as z:
        names = z.namelist()[:800]
        n_act = 0
        for name in names:
            f = json.loads(z.read(name).decode("utf-8"))
            if f.get("actions") or f.get("judge_info"):
                n_act += 1
    # ⚠ 这里原本还有一条 `assert n_act <= len(names)` ——
    # **它恒真**（某类帧数不可能超过总帧数），是**占位式假断言**，
    # 我自己自查时发现并删除。留下下面这条**真的会失败**的断言。
    assert n_act < len(names) * 0.5, \
        "前 800 帧里 %d/%d 帧有动作或裁决 —— 与「部署阶段耗时长」的结论不符" % (
            n_act, len(names))


# ─────────────────── 写入端不夸大也不漏写

def test_writer_does_not_invent_the_empty_types(tmp_path: Path) -> None:
    """**不编造**：源帧没有 `actions`/`judge_info` 时，写入端不得凭空补。

    ⚠ 必须用**结构完整**的帧，只摘掉这两类 ——
    我第一版用了 `{"time": {...}}` 这种最小帧，结果**因缺 v1 必需键而校验失败并降级**；
    那不是写入端"编造"，而是**输入本身不合法**（降级路径工作正常，也证明它有效）。
    """
    from offline_replay import write_offline_replay  # noqa: PLC0415
    from training.replay import Replay  # noqa: PLC0415

    stripped = []
    for f in _states(5):
        g = dict(f)
        g["actions"] = []
        g["judge_info"] = []
        stripped.append(g)
    out = tmp_path / "stripped.zip"
    rep = write_offline_replay(stripped, zip_name=str(out), run_id="stripped",
                               seed=7, frame_version="1.1.0")
    assert rep["frame_version"] == "1.1.0", "结构完整的帧不应降级：%s" % rep
    with Replay(out) as r:
        cov = r.record_type_coverage()
        assert cov["timeline"] == 5
        assert cov["seed"] == 5
        assert cov["actions"] == 0
        assert cov["adjudication"] == 0
        for _, f in r.iter_frames():
            assert "decisions" not in (f.get("agent") or {}), "伪造了 decisions"


def test_writer_degrades_when_input_frames_are_not_valid_v1() -> None:
    """**降级路径的显式验证**：输入帧缺 v1 必需键时，
    v1.1.0 校验会失败 → 写入端**退回 v1.0.0 并打印原因**，而不是抛异常或产半成品。

    这条把上一版测试无意中发现的行为**固定下来** —— 降级不是缺陷，是设计。
    """
    from offline_replay import write_offline_replay  # noqa: PLC0415

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "degraded.zip"
        rep = write_offline_replay([{"time": {"cur_step": 0}}],
                                   zip_name=str(out), run_id="deg", seed=7,
                                   frame_version="1.1.0")
        assert rep["frame_version"] == "1.0.0", "不合法的帧应触发降级"
        assert out.is_file(), "降级后仍应产出可读产物"
