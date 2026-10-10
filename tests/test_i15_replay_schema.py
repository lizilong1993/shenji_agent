"""I15：复盘 schema 校验与按索引/按 tick 回放。

为什么用**合成 zip** 而不是仓库里那个 3.6MB 真实复盘：
CI 不应依赖体积较大的历史产物，真实文件也未必随仓库分发。
故：格式契约用合成文件钉住；真实文件的实测结论写在 schema.json 与类型注释里。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from training.replay import (ENTRY_RE, Replay, ReplayFormatError,
                             frame_entries, load_schema, validate)

V1_REQUIRED = ["operators", "passengers", "time", "jm_points", "cities", "scores",
               "landmarks", "terrain_id", "judge_info", "role_and_grouping_info",
               "communication", "scenario_id", "valid_actions", "actions"]


def _v1_frame(step: int = 0) -> dict:
    """最小 v1 帧（键齐全、内容从简）。"""
    f = {k: None for k in V1_REQUIRED}
    f["time"] = {"cur_step": step, "tick": 1.0, "max_time": 1800, "max_step": 1800, "stage": 1}
    for k in ("operators", "passengers", "jm_points", "cities", "judge_info", "communication", "actions"):
        f[k] = []
    for k in ("scores", "landmarks", "role_and_grouping_info", "valid_actions"):
        f[k] = {}
    f["terrain_id"] = 123
    f["scenario_id"] = 1
    return f


def _write_zip(path: Path, frames, run_id: str = "run1") -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for i, f in enumerate(frames):
            z.writestr("%s/%d" % (run_id, i), json.dumps(f, ensure_ascii=False, separators=(",", ":")))
    return path


def test_schema_file_is_valid_and_self_describing() -> None:
    s = load_schema()
    assert s["schema_version"]
    # 双核心要求必须在 schema 里显式记录（31006 4.4.2）
    assert "human_ops" in s["dual_core"] and "agent_data" in s["dual_core"]
    assert s["dual_core"]["agent_data"]["present_in_v1"] is False
    assert s["dual_core"]["seed"]["present_in_v1"] is False
    # v1 必需键必须与实测的 14 个顶层键一致
    assert sorted(s["frame_required_v1"]) == sorted(V1_REQUIRED)


def test_entry_naming_contract() -> None:
    assert ENTRY_RE.match("1770632023.2594852/0")
    assert ENTRY_RE.match("run/1801")
    assert not ENTRY_RE.match("run")          # 无索引
    assert not ENTRY_RE.match("run/abc")      # 索引非数字
    assert not ENTRY_RE.match("a/b/1")        # 多余层级


def test_historical_v1_replay_passes_and_is_not_judged_as_v1_1(tmp_path: Path) -> None:
    """历史 v1 文件不得因缺 meta/agent/human 而失败——那是缺口，不是文件之错。"""
    p = _write_zip(tmp_path / "v1.zip", [_v1_frame(0), _v1_frame(1), _v1_frame(2)])
    rep = validate(path=p)
    assert rep["ok"] is True, rep
    assert rep["declared_version"] == "1.0.0"
    assert rep["frame_count"] == 3
    assert rep["missing_keys"] == []
    assert any("v1 replay" in w for w in rep["warnings"])


def test_frame_declaring_v1_1_must_carry_agent_human_meta(tmp_path: Path) -> None:
    f = _v1_frame(0)
    f["meta"] = {"schema_version": "1.1.0"}          # 声明 v1.1，但缺 agent/human
    p = _write_zip(tmp_path / "bad11.zip", [f])
    rep = validate(path=p)
    assert rep["ok"] is False
    assert "agent" in rep["missing_keys"] and "human" in rep["missing_keys"]


def test_fully_populated_v1_1_passes(tmp_path: Path) -> None:
    f = _v1_frame(0)
    f["meta"] = {"schema_version": "1.1.0", "seed": 42, "run_id": "run1"}
    f["agent"] = {"decisions": [], "deterministic": True}
    f["human"] = {"operations": []}
    p = _write_zip(tmp_path / "ok11.zip", [f])
    rep = validate(path=p)
    assert rep["ok"] is True, rep
    assert rep["declared_version"] == "1.1.0"


def test_non_contiguous_indices_are_reported(tmp_path: Path) -> None:
    p = tmp_path / "gap.zip"
    with zipfile.ZipFile(p, "w") as z:
        for i in (0, 2):                              # 缺 1
            z.writestr("run1/%d" % i, json.dumps(_v1_frame(i)))
    rep = validate(path=p)
    assert rep["contiguous"] is False
    assert rep["bad_entries"]
    assert rep["ok"] is False


def test_size_limit_warns_but_does_not_fail(tmp_path: Path) -> None:
    """31006 5.1.2 注为「建议」，故超限只警告并提示分片。"""
    p = _write_zip(tmp_path / "big.zip", [_v1_frame(0)])
    rep = validate(path=p, size_limit_bytes=1)        # 人为设成 1 字节
    assert rep["ok"] is True, rep
    assert any("sharding" in w for w in rep["warnings"])


def test_playback_by_index_and_by_tick(tmp_path: Path) -> None:
    p = _write_zip(tmp_path / "play.zip", [_v1_frame(s) for s in (0, 5, 10)])
    with Replay(p) as r:
        assert r.frame_count == 3
        assert r.index_range() == (0, 2)
        assert r.frame(0)["time"]["cur_step"] == 0
        assert r.frame(2)["time"]["cur_step"] == 10
        # 按 tick 定位（索引与 tick 不必相同）
        assert r.frame_at_tick(5)["time"]["cur_step"] == 5
        # 精确重现：同一 tick 两次读取完全一致
        assert r.frame_at_tick(5) == r.frame_at_tick(5)
        # 越界报明确错误
        with pytest.raises(ReplayFormatError):
            r.frame(99)
        with pytest.raises(ReplayFormatError):
            r.frame_at_tick(999)


def test_missing_file_is_reported_not_raised(tmp_path: Path) -> None:
    rep = validate(path=tmp_path / "nope.zip")
    assert rep["ok"] is False and rep["error"] == "file not found"
