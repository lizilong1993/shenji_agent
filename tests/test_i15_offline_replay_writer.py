"""I15 步骤 2 收尾：`run_offline_games.py` 产出 v1.1.0 双核心复盘。

## 背景

`land_wargame_sdk/run_offline_games.py` 有**两处**写盘（单智能体 / 多智能体模式），
原样都写 `state[GREEN]` 快照 → `json.dumps` → `writestr`，即 **v1.0.0 单核心**。

本轮把两处交给 `land_wargame_sdk/offline_replay.py`：
默认升级 **v1.1.0** 并过 schema 校验；`SHENJI_REPLAY_VERSION=1.0.0`
可切回旧格式（`I15`【风险与回滚】要求的"新格式写入可用开关切换"）。

**关键：旧格式路径与原文逐字节等价** —— 否则回滚就不是回滚。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SDK = REPO / "land_wargame_sdk"
HELPER = SDK / "offline_replay.py"
SCRIPT = SDK / "run_offline_games.py"
REAL = REPO / "replays" / "replay_1770632023.2594852.zip"

for _p in (str(REPO), str(SDK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _helper():
    if not HELPER.is_file():
        pytest.skip("offline_replay.py not present")
    spec = importlib.util.spec_from_file_location("offline_replay", HELPER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _states(n: int = 8):
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with zipfile.ZipFile(REAL) as z:
        names = z.namelist()[:n]
        return [json.loads(z.read(x).decode("utf-8")) for x in names]


# ─────────────────── 接线：两处写盘都被替换

def test_both_write_blocks_now_call_the_helper() -> None:
    """**两处**（单/多智能体）都必须改到 —— 只改一处会留下不一致。"""
    if not SCRIPT.is_file():
        pytest.skip("run_offline_games.py not present")
    src = SCRIPT.read_text(encoding="utf-8")
    assert src.count("write_offline_replay(") == 2, \
        "应有 2 处调用，实际 %d" % src.count("write_offline_replay(")
    # 原文的裸写法必须已消失
    assert "json.dumps(ob, ensure_ascii=False" not in src, "仍残留旧写盘写法"


def test_script_still_compiles_and_documents_the_switch() -> None:
    """脚本必须仍可编译，且注释写明回滚开关。"""
    if not SCRIPT.is_file():
        pytest.skip("run_offline_games.py not present")
    import py_compile
    py_compile.compile(str(SCRIPT), doraise=True)
    src = SCRIPT.read_text(encoding="utf-8")
    assert "SHENJI_REPLAY_VERSION" in src, "未写明回滚开关"


# ─────────────────── 默认路径：v1.1.0

def test_default_writes_v11_and_passes_schema(tmp_path: Path) -> None:
    m = _helper()
    os.environ.pop(m._ENV_KEY, None)
    assert m.resolve_frame_version() == "1.1.0"
    out = tmp_path / "v11.zip"
    rep = m.write_offline_replay(_states(8), zip_name=str(out), run_id="t1", seed=1)
    assert rep["frame_version"] == "1.1.0"
    assert rep["validate_ok"] is True, rep
    from training.replay import validate
    v = validate(path=out)
    assert v["ok"] is True and v["declared_version"] == "1.1.0"


# ─────────────────── 回滚开关：v1.0.0 且**逐字节等价**

def test_env_switch_rolls_back_to_v1(tmp_path: Path) -> None:
    m = _helper()
    os.environ[m._ENV_KEY] = "1.0.0"
    try:
        assert m.resolve_frame_version() == "1.0.0"
        out = tmp_path / "v10.zip"
        rep = m.write_offline_replay(_states(8), zip_name=str(out), run_id="t2")
        assert rep["frame_version"] == "1.0.0"
    finally:
        os.environ.pop(m._ENV_KEY, None)
    from training.replay import validate
    v = validate(path=out)
    assert v["ok"] is True and v["declared_version"] == "1.0.0"


def test_v1_path_is_byte_identical_to_the_original_writing(tmp_path: Path) -> None:
    """**回滚必须是真的回滚**：v1 路径与原文写法逐字节一致。

    这条是本文件最要紧的断言 —— 若"回滚"产出与原文不同，
    那只是"另一种格式"，不是回滚，历史读取方可能出问题。
    """
    m = _helper()
    states = _states(8)
    out = tmp_path / "v10.zip"
    os.environ[m._ENV_KEY] = "1.0.0"
    try:
        m.write_offline_replay(states, zip_name=str(out), run_id="ref")
    finally:
        os.environ.pop(m._ENV_KEY, None)

    # 原文写法（逐字复刻 run_offline_games.py 改动前的代码）
    ref = tmp_path / "ref.zip"
    with zipfile.ZipFile(ref, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for i, ob in enumerate(states):
            z.writestr("ref/%d" % i,
                       json.dumps(ob, ensure_ascii=False, separators=(",", ":")))

    def entries(p):
        with zipfile.ZipFile(p) as z:
            return [(n, z.read(n)) for n in sorted(z.namelist(),
                                                   key=lambda s: int(s.split("/")[-1]))]
    a, b = entries(out), entries(ref)
    assert [x[0] for x in a] == [x[0] for x in b], "条目命名不一致"
    assert all(x[1] == y[1] for x, y in zip(a, b)), "条目字节不一致 —— 回滚不等价"


# ─────────────────── 降级：助手不可用时不炸

def test_falls_back_to_v1_when_upgrade_helper_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    """**引擎侧脚本不能因升级助手缺失而失败** —— 退回 v1 原行为。"""
    m = _helper()
    monkeypatch.setattr(m, "_load_upgrade_helpers", lambda: None)
    os.environ.pop(m._ENV_KEY, None)
    states = [{"time": {"cur_step": i}} for i in range(4)]
    out = tmp_path / "fallback.zip"
    rep = m.write_offline_replay(states, zip_name=str(out), run_id="fb")
    assert rep["frame_version"] == "1.0.0", "助手缺失时未降级"
    assert out.is_file() and rep["frames"] == 4


def test_creates_parent_directory(tmp_path: Path) -> None:
    """目录不存在时应自行创建（原文用 os.makedirs，行为须保留）。"""
    m = _helper()
    nested = tmp_path / "a" / "b" / "rep.zip"
    m.write_offline_replay([{"time": {"cur_step": 0}}], zip_name=str(nested),
                           run_id="n", frame_version="1.0.0")
    assert nested.is_file()


def test_explicit_frame_version_overrides_env() -> None:
    """显式参数优先于环境变量（便于测试与逐调用控制）。"""
    m = _helper()
    os.environ[m._ENV_KEY] = "1.0.0"
    try:
        assert m.resolve_frame_version("1.1.0") == "1.1.0"
    finally:
        os.environ.pop(m._ENV_KEY, None)


def test_helper_does_not_invent_human_or_seed() -> None:
    """**不编造**：助手不传 seed/人类参数时不得凭空写。"""
    m = _helper()
    frames = m._upgrade_frames(_states(3), seed=None, scenario=None, human_actor=0)
    assert frames, "升级失败"
    for f in frames:
        assert "seed" not in (f.get("meta") or {}), "未给 seed 却写了"
