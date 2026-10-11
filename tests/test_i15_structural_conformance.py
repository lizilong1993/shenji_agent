"""I15 判据 2 的**结构性**校验：`validate()` 下钻到 `frame_fields.*.fields`。

## 为什么需要这个文件

独立评审实测：原 `validate()` **只查顶层键存在**（`missing_keys`），
**从不深入 `frame_fields.*.fields`** —— 于是：

| 段 | schema 声明 | 写入端实际产出 | 旧校验器的判定 |
| --- | --- | --- | --- |
| `agent` | `decisions`/`model_refs`/`deterministic` | `{timeline, scores, valid_action_kinds}` | **ok=True**（看不见差集） |
| `human` | `operations` | **整段不存在** 或 `{action_count, actions, ...}` | **ok=True** |

**判据 2 原文是「复盘文件通过 `schema.json` 校验」** ——
只看顶层键的校验器撑不起这句话，故判据 2 只能算"仅对浅校验成立"。

## 本轮的修复（三件事，顺序不可颠倒）

1. **schema 补声明**：把写入端**实际产出**的键也声明进去，并标 `emitted_by`；
2. **写入端补产出**：`human.operations`（无人类操作时 `[]`，**如实为空非缺字段**）
   与 `agent.decisions`（由该帧自己的 `time` 派生）；
3. **`validate()` 下钻**：类型不符 → 判失败；无产出者的字段 → 不判定；未声明的键 → warning。

> **顺序重要**：先补声明与产出再启用下钻，否则会先把现有产物**全部判失败**。
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "land_wargame_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.replay import validate  # noqa: E402

SCHEMA = REPO / "replays" / "schema.json"
REAL = REPO / "replays" / "replay_1770632023.2594852.zip"


def _good() -> dict:
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with zipfile.ZipFile(REAL) as z:
        f = json.loads(z.read(z.namelist()[0]).decode("utf-8"))
    f = dict(f)
    f["meta"] = {"schema_version": "1.1.0", "seed": 1}
    f["agent"] = {"timeline": {}, "scores": {}, "valid_action_kinds": [],
                  "decisions": [{"cur_step": 0, "seat": None, "action_kinds": []}]}
    f["human"] = {"operations": [], "action_count": 0, "source": "test"}
    f["actions"] = []
    return f


def _write(frames, path, run_id="t"):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, f in enumerate(frames):
            z.writestr("%s/%d" % (run_id, i),
                       json.dumps(f, ensure_ascii=False, separators=(",", ":")))


# ────────────── 校验器确实下钻

def test_validator_reports_structural_check_count() -> None:
    """下钻必须真的执行（报告里应有 `structural_checked`，且 > 0）。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    v = validate(path=REAL)
    assert v.get("structural_checked", 0) > 0, "未下钻 —— 判据 2 仍只是浅校验"


def test_type_mismatch_is_rejected(tmp_path: Path) -> None:
    """**类型不符必须判失败** —— 这是"下钻"的核心价值。"""
    f = _good()
    f["human"]["operations"] = "not-an-array"      # schema 声明 array
    p = tmp_path / "bad_type.zip"
    _write([f], p)
    v = validate(path=p)
    assert v["ok"] is False, "类型不符竟被判通过"
    assert any("operations" in s and "type mismatch" in s
               for s in (v.get("structural_issues") or [])), v.get("structural_issues")


def test_agent_decisions_type_mismatch_is_rejected(tmp_path: Path) -> None:
    """`agent.decisions` 声明为 array，给 object 必须失败。"""
    f = _good()
    f["agent"]["decisions"] = {"cur_step": 0}
    p = tmp_path / "bad_dec.zip"
    _write([f], p)
    v = validate(path=p)
    assert v["ok"] is False


def test_undeclared_keys_produce_warnings_not_failures(tmp_path: Path) -> None:
    """段内**未声明的键** → warning（便于发现新差集），**不判失败**。

    若这里判失败，任何写入端的小扩展都会让全部历史产物失效 —— 过严。
    """
    f = _good()
    f["agent"]["some_new_key"] = 1
    p = tmp_path / "extra.zip"
    _write([f], p)
    v = validate(path=p)
    assert v["ok"] is True, "未声明的键不应判失败"
    assert "some_new_key" in (v.get("undeclared_keys") or {}).get("agent", [])
    assert any("not declared" in w for w in v.get("warnings") or [])


def test_fields_without_emitters_are_not_judged(tmp_path: Path) -> None:
    """`emitted_by` 为空的字段（如 `agent.model_refs`）**不参与判定**。

    **如实声明"目前无产出者"，而不是假装已实现再让它判失败。**
    """
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    mr = d["frame_fields"]["agent"]["fields"]["model_refs"]
    assert mr.get("emitted_by") == [], "model_refs 不该声称有产出者"
    f = _good()
    # 不给 model_refs —— 不应因此失败
    p = tmp_path / "no_mr.zip"
    _write([f], p)
    v = validate(path=p)
    assert v["ok"] is True


# ────────────── schema 声明与写入端产出一致

def test_schema_declares_every_key_the_writers_emit() -> None:
    """**消除差集的核心断言**：写入端产出的键必须在 schema 里有声明。"""
    if not SCHEMA.is_file():
        pytest.skip("schema not present")
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    ff = d["frame_fields"]
    # offline_replay / selfplay_replay 实际产出的键（本轮实测）
    emitted = {
        "agent": {"timeline", "scores", "valid_action_kinds", "decisions"},
        "human": {"operations", "actor_kind", "source", "action_count",
                  "agent_action_count", "human_actor", "actions"},
        "meta": {"schema_version", "frame_version", "seed", "scenario"},
    }
    for sec, keys in emitted.items():
        declared = set((ff[sec].get("fields") or {}))
        missing = keys - declared
        assert not missing, "%s 段产出了未声明的键: %s" % (sec, sorted(missing))


def test_schema_declares_normative_dual_core_fields() -> None:
    """双核心的**规范字段**必须在 schema 里声明。"""
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    ff = d["frame_fields"]
    assert "operations" in (ff["human"].get("fields") or {}), "缺 human.operations"
    assert "decisions" in (ff["agent"].get("fields") or {}), "缺 agent.decisions"


def test_writers_emit_human_operations() -> None:
    """两个写入端都必须产出 `human.operations`（自博弈下为 `[]`，**如实为空**）。"""
    import inspect
    for rel in ("land_wargame_sdk/offline_replay.py",
                "training/selfplay_replay.py"):
        p = REPO / rel
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8")
        assert '"operations"' in src, "%s 未产出 human.operations" % rel


def test_real_replay_still_passes_with_structural_check() -> None:
    """**历史 v1 产物不得因新增下钻而失败**（向后兼容）。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    v = validate(path=REAL)
    assert v["ok"] is True, v.get("missing_keys") or v.get("structural_issues")
    assert v.get("structural_issues") == []


def test_schema_declares_the_structural_validation_contract() -> None:
    """schema 必须**自述**下钻规则，使"判定口径"可被机器读到。"""
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    sv = d.get("structural_validation") or {}
    assert sv.get("enabled") is True
    assert "rule" in sv and "requirement" in sv
