"""I15 步骤 2/3：v1 → v1.1.0 双核心**写侧**（`scripts/upgrade_replay_v11.py`）。

## 本轮查清的事实

| 项 | 实测 |
| --- | --- |
| `land_wargame_sdk/run_offline_games.py` | **仍写旧 `state[GREEN]` 整帧快照**（第 83/125/229/331 行） |
| `training/selfplay.py` | **不写 replay**（`replay` 零命中） |
| `replays/schema.json` | 有 `dual_core` 声明，但**缺** `adjudication`/`timeline` 键 |

## 实现方式

`run_offline_games.py` 是引擎侧入口；故写侧做成**纯函数式**模块：
读 v1 复盘 → 逐帧升级 → 写 → **用真实 schema 校验** → **回读核对**。
这样可在**不依赖引擎**的前提下验证写侧，且将来该脚本只需把"构造帧"换成调用本模块。

## 过程中修掉的两个**自己的缺陷**（都由真实数据暴露）

1. `validate()` 把 `frame_optional_v1_1`（名字就是 optional）**整份当成必需**，
   于是"没有人类命令的帧"被判失败 → 1802 帧真实复盘升级时报 `missing=['human']`。
   **判定依据应是 schema 的 `dual_core[*].required_in`**。
2. `annotate_human_actions()` 只给**每条动作**打 `actor_kind`，
   **没写帧级 `human` 段** → schema 要求"有 actions 的帧"有 `human`，仍失败。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

UPGRADE = REPO / "scripts" / "upgrade_replay_v11.py"
REPLAY = REPO / "training" / "replay.py"
SCHEMA = REPO / "replays" / "schema.json"
REAL = REPO / "replays" / "replay_1770632023.2594852.zip"


def _mod():
    if not UPGRADE.is_file():
        pytest.skip("upgrade tool not present")
    spec = importlib.util.spec_from_file_location("urv11", UPGRADE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ─────────────────── 写侧核心：真实复盘可升级并校验通过

def test_upgrade_real_replay_produces_valid_v11(tmp_path: Path) -> None:
    """**端到端**：真实 1802 帧复盘升级后必须通过 schema 校验且版本为 1.1.0。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    m = _mod()
    dst = tmp_path / "v11.zip"
    res = m.upgrade_replay(REAL, dst, seed=20260713, scenario="1231")
    assert res["version"] == "1.1.0", "版本未写为 1.1.0"
    assert res["report"]["ok"] is True, res["report"].get("missing_keys")
    assert res["frames"] == res["readback_frames"], "回读帧数不一致"
    assert dst.is_file()


def test_agent_section_present_on_every_frame(tmp_path: Path) -> None:
    """`agent` 段是 v1.1 的**聚合必需**项，且本实现应逐帧都能给出。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    m = _mod()
    res = m.upgrade_replay(REAL, tmp_path / "v11.zip", seed=1, scenario="1231")
    assert res["upgraded_with_agent"] == res["frames"], \
        "只有 %d/%d 帧带 agent 段" % (res["upgraded_with_agent"], res["frames"])


def test_seed_is_written_and_not_fabricated(tmp_path: Path) -> None:
    """`meta.seed` 必须**按传入值**写入；不传时**不得编造**。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    m = _mod()
    a = tmp_path / "seeded.zip"
    m.upgrade_replay(REAL, a, seed=777, scenario="1231")
    with zipfile.ZipFile(a) as z:
        f = json.loads(z.read(z.namelist()[0]).decode("utf-8"))
    assert (f.get("meta") or {}).get("seed") == 777

    # 不传 seed → validate 应因缺 meta.seed 失败（**不编造**）
    b = tmp_path / "unseeded.zip"
    with pytest.raises(Exception):
        m.upgrade_replay(REAL, b, seed=None, scenario="1231")


def test_size_limit_is_reported(tmp_path: Path) -> None:
    """判据 3：文件大小有校验。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    m = _mod()
    dst = tmp_path / "v11.zip"
    res = m.upgrade_replay(REAL, dst, seed=1, scenario="1231")
    # 真实 1802 帧升级后应远小于 100MB（判据 3）
    assert dst.stat().st_size < 100 * 1024 * 1024
    assert isinstance(res["report"], dict)


# ─────────────────── 修掉的缺陷 1：必需字段与 schema 对齐

def test_validate_does_not_require_optional_fields() -> None:
    """`frame_optional_v1_1` 的名字是 optional —— **不得**整份当成必需。"""
    if not REPLAY.is_file():
        pytest.skip("replay.py not present")
    src = REPLAY.read_text(encoding="utf-8")
    i = src.find("def _frame_required(")
    assert i > 0
    body = src[i:i + 1600]
    # 修后不应再把 frame_optional_v1_1 整份并入必需
    assert 'req += list(schema.get("frame_optional_v1_1"' not in body, \
        "又把 optional 整份当成必需了"
    # 应改为"有 actions 才要求 human"
    assert "human:only_if_actions" in body


def test_dual_core_required_follows_schema_declarations() -> None:
    """聚合必需项应**读自 schema**，而不是硬编码。"""
    if not REPLAY.is_file():
        pytest.skip("replay.py not present")
    src = REPLAY.read_text(encoding="utf-8")
    assert "def _dual_core_required(" in src
    i = src.find("def _dual_core_required(")
    body = src[i:i + 1200]
    assert "required_in" in body, "未读 schema 的 required_in"
    assert "dual_core" in body


def test_schema_human_ops_has_no_required_in() -> None:
    """**防自造**：schema 里 `human_ops` 确实**没有** `required_in`
    （其 `frame_path` 指向 v1 就有的 `actions`），
    而 `agent_data` / `seed` 有。这是本次判定的权威依据。"""
    if not SCHEMA.is_file():
        pytest.skip("schema not present")
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    dc = d.get("dual_core") or {}
    assert "required_in" not in (dc.get("human_ops") or {}), \
        "schema 的 human_ops 出现了 required_in —— 判定依据需重估"
    assert (dc.get("agent_data") or {}).get("required_in") == ">=1.1.0"
    assert (dc.get("seed") or {}).get("required_in") == ">=1.1.0"


# ─────────────────── 修掉的缺陷 2：帧级 human 段

def test_annotate_produces_frame_level_human_section() -> None:
    """`annotate_human_actions` 必须**同时**产出帧级 `human` 段。"""
    if not REPLAY.is_file():
        pytest.skip("replay.py not present")
    src = REPLAY.read_text(encoding="utf-8")
    i = src.find("def annotate_human_actions(")
    assert i > 0
    j = src.find("\ndef ", i + 10)
    body = src[i:j if j > 0 else i + 4000]
    assert 'out["human"]' in body, "仍未写帧级 human 段"


def test_human_section_contains_the_split_actions() -> None:
    """`human` 段内容应是**按 actor_kind 分出的命令**（真实数据，非编造）。"""
    from training.replay import annotate_human_actions
    frame = {
        "actions": [
            {"cur_step": 0, "message": {"actor": 0}, "error": None},
            {"cur_step": 1, "message": {"actor": 1}, "error": None},
        ]
    }
    out = annotate_human_actions(frame, human_actor=0)
    assert "human" in out
    h = out["human"]
    assert h["action_count"] == 1 and h["agent_action_count"] == 1
    assert h["actions"][0]["actor_kind"] == "human"
    # 只加标注，不改原字段
    assert out["actions"][1]["actor_kind"] == "agent"
    assert out["actions"][0]["message"] == {"actor": 0}


def test_annotate_leaves_frameless_actions_alone() -> None:
    """没有 actions 时不得伪造 `human` 段。"""
    from training.replay import annotate_human_actions
    out = annotate_human_actions({"actions": []})
    assert "human" not in out or out.get("human", {}).get("action_count") == 0


# ─────────────────── 历史兼容（不得破坏 v1 读取）

def test_v1_replays_still_validate() -> None:
    """升级改造**不得**破坏历史 v1 文件的读取。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    from training.replay import validate
    rep = validate(path=REAL)
    assert rep["ok"] is True, rep
    assert rep["declared_version"] == "1.0.0", "v1 文件被误判为别的版本"
    # v1 缺新字段应记为 warning，而非失败
    assert any("v1" in w for w in rep.get("warnings") or [])


def test_upgrade_tool_documents_why_it_is_not_the_engine_script() -> None:
    """模块文档必须写明为何做成纯函数式模块（引擎侧脚本改动的替代路径）。"""
    if not UPGRADE.is_file():
        pytest.skip("upgrade tool not present")
    src = UPGRADE.read_text(encoding="utf-8")
    assert "run_offline_games.py" in src
    assert "state[GREEN]" in src, "未说明旧格式是什么"
