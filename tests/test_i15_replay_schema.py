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


def test_frame_declaring_v1_1_must_carry_agent_and_seed(tmp_path: Path) -> None:
    """声明 v1.1 就必须有 `agent` 段与 `meta.seed`。

    ⚠ **不再要求 `human`** —— 该帧 `actions` 为空，**没有命令可标注**。
    判定依据是 `replays/schema.json` 的 `dual_core`：
    `agent_data.required_in` 与 `seed.required_in` 均为 `">=1.1.0"`，
    而 `human_ops` **没有** `required_in`（schema 明写 `present_in_v1: true`，
    其 `frame_path` 就是 v1 已有的 `actions`）。

    修前实现把 `frame_optional_v1_1` 整份当成必需，于是"没有人类命令的帧"被判失败 ——
    实测：1802 帧真实复盘升级时报 `missing=['human']`，那不是文件的错。
    """
    f = _v1_frame(0)
    f["meta"] = {"schema_version": "1.1.0"}          # 声明 v1.1，但缺 agent / seed
    p = _write_zip(tmp_path / "bad11.zip", [f])
    rep = validate(path=p)
    assert rep["ok"] is False
    assert "agent" in rep["missing_keys"]
    assert "meta.seed" in rep["missing_keys"]


def test_human_is_required_only_when_the_frame_has_actions(tmp_path: Path) -> None:
    """`human` 是 `actions` 的**标注** —— 有命令才要求，无命令不要求。

    两半都要测：只测"没有就不要求"会放过"有命令却没标注"的缺陷。
    """
    # (a) 有 actions 却没 human → 必须失败
    f = _v1_frame(0)
    f["meta"] = {"schema_version": "1.1.0", "seed": 42}
    f["agent"] = {"decisions": []}
    f["actions"] = [{"cur_step": 0, "message": {"actor": 1}, "error": None}]
    p1 = _write_zip(tmp_path / "no_human.zip", [f])
    rep1 = validate(path=p1)
    assert rep1["ok"] is False, "有 actions 却缺 human，应判失败"
    assert "human" in rep1["missing_keys"]

    # (b) 无 actions 且无 human → 不应因缺 human 而失败
    g = _v1_frame(0)
    g["meta"] = {"schema_version": "1.1.0", "seed": 42}
    g["agent"] = {"decisions": []}
    g["actions"] = []
    p2 = _write_zip(tmp_path / "no_actions.zip", [g])
    rep2 = validate(path=p2)
    assert "human" not in rep2["missing_keys"], \
        "无 actions 的帧不应要求 human（修前正是这里判错）"
    assert rep2["ok"] is True, rep2


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
