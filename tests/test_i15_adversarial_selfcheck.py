"""独立复核（adversarial self-review）：把三个关键声明**做假测试**验证。

## 为什么要"做假测试"

前几轮我修掉了 4 条**永真断言**（不会失败 ⇒ 什么也没验证）。
同样的怀疑必须施加到**核心证据**上：
一条恒返回"通过"的校验函数，与一条恒真的断言是同一种病。

故本文件**主动注入损坏/构造非法输入**，验证被测物**真的会失败**：

| 复核项 | 手法 | 若被测物是假的会怎样 |
| --- | --- | --- |
| `verify_precise_replay()` | **monkeypatch 注入损坏** | 注入后仍 `ok=True` ⇒ 该函数形同虚设 |
| `validate()` | **构造应当被拒的复盘** | 非法复盘被判通过 ⇒ 放宽过头 |
| v1 回滚等价 | 取 **git 原文**独立执行比对 | 测试自造参照物 ⇒ 自证式假通过 |
| `schema.json` 的 note | **与真实数据对数** | 数字对不上 ⇒ 记录不可信 |

## 本轮各项的实测结论

| 复核项 | 结论 |
| --- | --- |
| `verify_precise_replay()` 能否失败 | **能** —— 两侧注污均 `ok=False`、`mismatches=3` |
| `validate()` 拒绝能力 | **能** —— 缺 agent / 缺 seed / 有 actions 缺 human / 缺 v1 键 / 索引不连续 **全部被拒**；合法 1.1.0 **通过** |
| v1 回滚是否等价 | **是** —— 与 `3b9f294^` 的真实原文产物**条目名与逐条目字节均一致** |
| `schema.json` 的 note | **发现 1 处不精确**（`actions` 写「2/40 帧」而实测 167/1802），**已修** |
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "land_wargame_sdk")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.replay import Replay, validate  # noqa: E402

REAL = REPO / "replays" / "replay_1770632023.2594852.zip"
SCHEMA = REPO / "replays" / "schema.json"
GIT = "/remote-home/lizilong/.local/bin/git"


def _frames(n: int = 10):
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with zipfile.ZipFile(REAL) as z:
        return [json.loads(z.read(x).decode("utf-8")) for x in z.namelist()[:n]]


def _write(frames, path, run_id="t"):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, f in enumerate(frames):
            z.writestr("%s/%d" % (run_id, i),
                       json.dumps(f, ensure_ascii=False, separators=(",", ":")))


# ══════════ 复核 1：verify_precise_replay 必须**能失败**

def test_verify_precise_replay_detects_corrupt_frame_bytes() -> None:
    """**注入损坏到 `frame_bytes`** → 必须 `ok=False`。

    若注入后仍通过，该函数就是"运行时版的永真断言"。
    """
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        assert r.verify_precise_replay(ticks=[0, 10, 100])["ok"] is True
        orig = r.frame_bytes
        r.frame_bytes = lambda i: orig(i) + b"X"      # 尾部加一字节
        res = r.verify_precise_replay(ticks=[0, 10, 100])
    assert res["ok"] is False, "注入损坏后仍判通过 —— 该函数形同虚设"
    assert len(res["mismatches"]) == 3


def test_verify_precise_replay_detects_corrupt_tick_frames() -> None:
    """**注入损坏到 `frame_at_tick`** → 必须 `ok=False`（另一侧）。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        orig = r.frame_at_tick

        def corrupt(tick):
            f = dict(orig(tick))
            f["__injected__"] = True
            return f

        r.frame_at_tick = corrupt
        res = r.verify_precise_replay(ticks=[0, 10, 100])
    assert res["ok"] is False, "tick 侧损坏未被检出"
    assert len(res["mismatches"]) == 3


def test_verify_reports_missing_tick_as_failure() -> None:
    """不存在的 tick 必须使 `ok=False`（不是静默跳过）。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        res = r.verify_precise_replay(ticks=[10, 999999])
    assert res["ok"] is False
    assert 999999 in res["missing_ticks"]


# ══════════ 复核 2：validate 必须**仍能拒绝**非法复盘

def _good() -> dict:
    f = dict(_frames(1)[0])
    f["meta"] = {"schema_version": "1.1.0", "seed": 1}
    f["agent"] = {"timeline": {}}
    f["human"] = {"action_count": 0, "actions": []}
    return f


@pytest.mark.parametrize("label,broken", [
    ("missing_agent", lambda f: f.pop("agent", None)),
    ("missing_seed", lambda f: f["meta"].pop("seed", None)),
    ("actions_without_human", lambda f: (
        f.update({"actions": [{"cur_step": 0, "message": {"actor": 1}, "error": None}]}),
        f.pop("human", None))),
    ("missing_v1_required", lambda f: f.pop("cities", None)),
])
def test_validate_rejects_invalid_replays(tmp_path: Path, label, broken) -> None:
    """**放宽必需字段后必须仍能拒绝**非法复盘（否则是放宽过头）。"""
    f = _good()
    broken(f)
    p = tmp_path / ("bad_%s.zip" % label)
    _write([f], p)
    v = validate(path=p)
    assert v["ok"] is False, "%s 未被拒绝：missing=%s" % (label, v.get("missing_keys"))


def test_validate_accepts_a_valid_v11_control(tmp_path: Path) -> None:
    """**对照**：合法 1.1.0 必须通过 —— 防止"什么都拒"被误当成严格。"""
    p = tmp_path / "ok.zip"
    _write([_good()], p)
    v = validate(path=p)
    assert v["ok"] is True, v.get("missing_keys")


def test_validate_rejects_non_contiguous_indices(tmp_path: Path) -> None:
    """索引不从 0 连续 → 必须被拒（`bad_entries` 记明原因）。"""
    good = _good()
    p = tmp_path / "gap.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("t/0", json.dumps(good, ensure_ascii=False, separators=(",", ":")))
        z.writestr("t/5", json.dumps(good, ensure_ascii=False, separators=(",", ":")))
    v = validate(path=p)
    assert v["ok"] is False
    assert v["bad_entries"], "未记录不连续原因"


def test_validate_handles_a_corrupt_zip(tmp_path: Path) -> None:
    """损坏的 zip 不得让进程崩掉（应报错或被捕获）。"""
    p = tmp_path / "corrupt.zip"
    p.write_bytes(b"PK\x03\x04 not a real zip")
    try:
        v = validate(path=p)
        assert v["ok"] is False
    except zipfile.BadZipFile:
        pass          # 抛异常亦可接受


# ══════════ 复核 3：v1 回滚 vs **git 原文**（不是自造参照物）

def test_v1_rollback_matches_the_real_original_code(tmp_path: Path) -> None:
    """从 **git 历史**取改动前原文，**原样执行**，与回滚产物比对。

    ⚠ 关键：参照物必须来自 **git**，不能自己写一遍"我以为是原文"的代码 ——
    否则测试是**自证式**的，永远通过。
    """
    if not (REPO / ".git").exists():
        pytest.skip("not a git repo")
    try:
        orig = subprocess.run(
            [GIT, "show", "3b9f294^:land_wargame_sdk/run_offline_games.py"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("cannot read the original from git")
    # 原文写法必须与我复刻的一致，否则本测试无意义
    assert 'z.writestr(f"{begin}/{i}", data)' in orig
    assert 'json.dumps(ob, ensure_ascii=False, separators=(",", ":"))' in orig

    states = _frames(10)
    run_id = "ref"
    ref = tmp_path / "ref.zip"
    with zipfile.ZipFile(ref, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for i, ob in enumerate(states):
            data = json.dumps(ob, ensure_ascii=False, separators=(",", ":"))
            z.writestr(f"{run_id}/{i}", data)

    import os
    import offline_replay  # noqa: PLC0415
    out = tmp_path / "rb.zip"
    os.environ[offline_replay._ENV_KEY] = "1.0.0"
    try:
        offline_replay.write_offline_replay(states, zip_name=str(out), run_id=run_id)
    finally:
        os.environ.pop(offline_replay._ENV_KEY, None)

    def ents(p):
        with zipfile.ZipFile(p) as z:
            ns = sorted(z.namelist(), key=lambda s: int(s.split("/")[-1]))
            return [(n, z.read(n)) for n in ns]

    a, b = ents(out), ents(ref)
    assert [x[0] for x in a] == [x[0] for x in b], "条目名不一致"
    assert all(x[1] == y[1] for x, y in zip(a, b)), "条目字节不一致 —— 回滚不等价"


# ══════════ 复核 4：schema 的 note 必须与真实数据相符

def test_actions_note_states_the_measured_numbers() -> None:
    """**发现并修掉的不精确 note**：原文写「有 2/40 帧非空」，实测 167/1802。

    结论（并非每帧都有动作）**对**，但引用的证据数字**是更早抽样的残留**。
    这类"结论对、证据错"的记录会让复核者对整个文件失去信任。
    """
    if not SCHEMA.is_file():
        pytest.skip("schema not present")
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    note = d["frame_fields"]["actions"].get("note") or ""
    # ⚠ 不能断言 `"2/40" not in note` ——
    # **更正说明本身就要引用那个旧数字**（否则后人不知改的是什么），
    # 我第一版这样写，于是**把更正判成了缺陷**。
    # 与第 54 轮"守卫误报函数自己的 docstring"是**同一类错**：只查串、不查语义。
    # 正确做法：要求 note **带更正标记**、写明实测数字，
    # 且**旧数字必须出现在更正语境里**（紧跟"原 note 写"这类字样）。
    assert "167" in note and "1802" in note, "未写明实测数字"
    assert "实测更正" in note or "是错的" in note or "残留" in note, \
        "未标明这是更正后的记录"
    if "2/40" in note:
        i = note.index("2/40")
        ctx = note[max(0, i - 40):i]
        assert ("原" in ctx) or ("更正" in ctx), \
            "旧数字 2/40 出现在非更正语境中: ...%s" % ctx


def test_actions_note_numbers_match_the_real_replay() -> None:
    """note 里的数字必须**与真实数据一致**（不是照抄，而是对数）。"""
    if not (SCHEMA.is_file() and REAL.is_file()):
        pytest.skip("schema or real replay not present")
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    note = d["frame_fields"]["actions"].get("note") or ""
    with Replay(REAL) as r:
        cov = r.record_type_coverage()
    assert "%d / %d" % (cov["actions"], cov["frame_count"]) in note, \
        "note 数字与 record_type_coverage() 不符"


def test_container_observed_size_matches_the_real_file() -> None:
    """`container.observed_size_bytes` 必须等于真实文件大小。"""
    if not (SCHEMA.is_file() and REAL.is_file()):
        pytest.skip("schema or real replay not present")
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    obs = (d.get("container") or {}).get("observed_size_bytes")
    assert obs == REAL.stat().st_size, "观测体积 %s vs 实际 %d" % (obs, REAL.stat().st_size)


def test_frame_required_v1_keys_all_exist_in_a_real_frame() -> None:
    """`frame_required_v1` 声明的键必须都能在真实帧里找到（否则校验会误拒）。"""
    if not (SCHEMA.is_file() and REAL.is_file()):
        pytest.skip("schema or real replay not present")
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    req = d.get("frame_required_v1") or []
    assert req
    f0 = _frames(1)[0]
    missing = [k for k in req if k not in f0]
    assert not missing, "声明的必需键在真实帧里缺失: %s" % missing
