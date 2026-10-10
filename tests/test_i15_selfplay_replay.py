"""I15 写入端：`selfplay.py` 产出 v1.1.0 双核心复盘。

## 做了什么

`training/selfplay.py` 此前**完全不写 replay**（`replay` 零命中）。
本轮新增 `training/selfplay_replay.py`（记录器，纯函数式）并接进 `_run_single()`：
签名加 `replay_dir` / `replay_seed`（**默认 None → 行为不变**），
每步记录、结束时写盘，并把 `replay` 取证放进返回的 stats。

## 数据来源（全部取自主循环已有变量，不新造）

| 六类记录 | 来源 |
| --- | --- |
| 动作 | 主循环每步的 `actions` |
| 裁决 | `after_state` 的 `judge_info` |
| 时间线 | `after_state` 的 `time` |
| 智能体决策摘要 | 调用方给的 `agent_summary`（该步 phase/turn/action_count） |
| 随机种子 | 调用方传入（**不传不写**，不编造） |
| 人类操作 | **自博弈里没有人类** → `action_count: 0`，`actor_kind` 全为 `agent` |

> **人类操作在自博弈里本就不存在。** 本实现**如实标"无人类"**（`action_count: 0`），
> 而不是伪造人类命令，也不是省略该段（省略会让"有没有人类"无法区分）。
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.replay import Replay, validate  # noqa: E402
from training.selfplay_replay import (  # noqa: E402
    MatchReplayRecorder,
    build_selfplay_frame,
)

REAL = REPO / "replays" / "replay_1770632023.2594852.zip"
SELFPLAY = REPO / "training" / "selfplay.py"


def _real_frames(n: int = 6) -> list:
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with zipfile.ZipFile(REAL) as z:
        return [json.loads(z.read(name).decode("utf-8")) for name in z.namelist()[:n]]


# ─────────────────── 接线：默认行为不变

def test_run_single_accepts_replay_params_defaulting_to_none() -> None:
    """**关键**：`replay_dir` 默认 `None` → 不传时行为与改动前一致。"""
    if not SELFPLAY.is_file():
        pytest.skip("selfplay.py not present")
    import inspect
    from training.selfplay import _run_single
    sig = inspect.signature(_run_single)
    assert "replay_dir" in sig.parameters
    assert sig.parameters["replay_dir"].default is None
    assert sig.parameters["replay_seed"].default is None


def test_selfplay_recorder_is_imported_and_used() -> None:
    """接线必须真的存在（否则只是加了个没人调用的模块）。"""
    if not SELFPLAY.is_file():
        pytest.skip("selfplay.py not present")
    src = SELFPLAY.read_text(encoding="utf-8")
    assert "MatchReplayRecorder" in src
    assert "_recorder.record(" in src
    assert "_recorder.write(" in src
    assert "selfplay_has_no_human_operator" not in src  # 该串在记录器里，不在主循环


# ─────────────────── 记录器：产出合法 v1.1.0

def test_recorder_produces_valid_v11_replay(tmp_path: Path) -> None:
    """端到端：记录 → 写盘 → **schema 校验通过**、版本 1.1.0。"""
    rec = MatchReplayRecorder(match_id="sp_case", seed=20260713, scenario="1231")
    for i, obs in enumerate(_real_frames(6)):
        assert rec.record(observation=obs, actions=obs.get("actions") or [],
                          agent_summary={"turn": i})
    out = tmp_path / "sp.zip"
    report = rec.write(out)
    assert report["ok"] is True, report.get("missing_keys")
    assert report["declared_version"] == "1.1.0"
    assert out.is_file()
    # 回读核对帧数
    with Replay(out) as r:
        assert r.frame_count == 6


def test_recorder_refuses_to_write_an_empty_replay(tmp_path: Path) -> None:
    """**不产空文件**：没有帧时必须抛错，而不是写出一个空复盘。"""
    rec = MatchReplayRecorder(match_id="empty")
    with pytest.raises(ValueError):
        rec.write(tmp_path / "empty.zip")
    assert not (tmp_path / "empty.zip").exists()


def test_record_returns_false_when_observation_is_unusable() -> None:
    """观测取不到时**跳过并返回 False**（不伪造空帧）。"""
    rec = MatchReplayRecorder(match_id="bad")
    assert rec.record(observation=None, actions=[]) is False
    assert rec.record(observation=42, actions=[]) is False
    assert len(rec) == 0


# ─────────────────── 六类记录：自博弈场景下的真实覆盖

def test_selfplay_marks_all_actions_as_agent_and_no_human() -> None:
    """**如实标注**：自博弈里的人类操作数为 **0**，命令全标 `agent`。"""
    frames = _real_frames(3)
    f = build_selfplay_frame(index=0, observation=frames[1],
                             actions=frames[1].get("actions") or [],
                             seed=1, scenario="1231")
    h = f.get("human") or {}
    assert h.get("action_count") == 0, "自博弈不应出现人类操作"
    assert h.get("source") == "selfplay_has_no_human_operator"
    for a in f.get("actions") or []:
        assert a["actor_kind"] == "agent", "命令应标为 agent"
        assert a.get("message") is not None or a.get("error") is not None


def test_frame_carries_timeline_and_seed() -> None:
    """时间线（来自 `time`）与种子必须写入。"""
    frames = _real_frames(2)
    f = build_selfplay_frame(index=0, observation=frames[0], actions=[], seed=99)
    assert (f.get("meta") or {}).get("seed") == 99
    assert isinstance((f.get("agent") or {}).get("timeline"), dict)


def test_seed_is_omitted_when_not_given() -> None:
    """**不编造**：不给 seed 时不得凭空写一个。"""
    frames = _real_frames(1)
    f = build_selfplay_frame(index=0, observation=frames[0], actions=[], seed=None)
    assert "seed" not in (f.get("meta") or {})


def test_recorder_summary_reports_coverage_honestly() -> None:
    """`summary()` **只报事实**：自博弈场景下 `human_action_count` 必为 0。"""
    rec = MatchReplayRecorder(match_id="cov", seed=7)
    for i, obs in enumerate(_real_frames(6)):
        rec.record(observation=obs, actions=obs.get("actions") or [],
                   agent_summary={"turn": i})
    s = rec.summary()
    assert s["frame_count"] == 6
    assert s["timeline"] == 6
    assert s["seed"] == 6
    assert s["agent_decisions"] == 6
    assert s["human_action_count"] == 0
    assert s["agent_action_count"] >= 0


def test_agent_summary_is_optional_and_not_fabricated() -> None:
    """不给 `agent_summary` 时不得伪造决策摘要。"""
    frames = _real_frames(1)
    f = build_selfplay_frame(index=0, observation=frames[0], actions=[], seed=1)
    agent = f.get("agent") or {}
    assert "decisions" not in agent, "未给摘要却写了 decisions"
    assert agent, "agent 段仍应有 timeline/scores 等已有字段"
