"""I15：用**真实复盘文件**校验 schema 与按索引回放。

为什么单独一个文件
------------------
`tests/test_i15_replay_schema.py` 的用例全部用**合成帧**（`tmp_path` 临时 zip）。
合成帧能证明校验器逻辑自洽，但**证明不了它能读真东西** ——
真实复盘由闭源引擎写入，字段集与合成帧未必一致。

故本文件只做一件事：**拿仓库里那份真实复盘当证据**，
把「schema 可校验真实文件」「按索引/按 tick 精确重现」钉在真实产物上。

真实产物（已在 git 中）：`replays/replay_1770632023.2594852.zip`
  - 3,812,007 B；1,802 帧；帧序号 0..1801 连续
  - 声明 `schema_version=1.0.0`；校验器用当前 schema(1.1.0) 校验并**按 1.0.0 的必需字段**判定
  - `ok=True`，`missing_keys=[]`，`bad_entries=[]`
  - 唯一 warning：v1 缺 `meta/agent/human`（即**双核心缺口**，schema.dual_core 已登记）

若该文件不存在，本模块**跳过**而不是判失败 —— 缺证据不等于证据为假；
但一旦在，就必须真的通过。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.replay import Replay, ReplayFormatError, run_summary, validate  # noqa: E402

REAL = REPO / "replays" / "replay_1770632023.2594852.zip"

# 实测钉住的常量。**故意写死**：真实产物若被替换/损坏，这些断言会失败并暴露出来。
REAL_FRAMES = 1802
REAL_BYTES = 3812007
REAL_RUN_ID = "1770632023.2594852"
REAL_STEP_MIN = 0
REAL_STEP_MAX = 1800
REAL_SCENARIO_ID = 1231


pytestmark = pytest.mark.skipif(
    not REAL.exists(), reason="real replay artifact not present in this checkout")


def test_real_replay_is_tracked_and_intact() -> None:
    """真实产物必须真的在仓库里、且字节数与我实测的一致。"""
    assert REAL.is_file(), "真实复盘文件缺失"
    assert REAL.stat().st_size == REAL_BYTES, (
        "复盘文件字节数变了（实测 %d，当前 %d）—— 产物被替换或损坏，"
        "下面所有基于它的结论都要重测" % (REAL_BYTES, REAL.stat().st_size))


def test_real_replay_passes_schema_validation() -> None:
    """判据 2：复盘文件通过 schema.json 校验 —— **在真实文件上**。"""
    rep = validate(path=REAL)
    assert rep["ok"] is True, "真实复盘未通过校验: %s" % rep
    assert rep["missing_keys"] == [], "真实复盘缺必需字段: %s" % rep["missing_keys"]
    assert rep["bad_entries"] == [], "真实复盘有条目读不出: %s" % rep["bad_entries"]
    assert rep["contiguous"] is True, "帧序号不连续"
    assert rep["frame_count"] == REAL_FRAMES


def test_real_replay_declares_v1_so_dual_core_absence_is_by_design() -> None:
    """双核心缺口必须被**识别**，而不是被当成通过。

    真实产物声明 `1.0.0`，而 `meta/agent/human` 是 `1.1.0` 才必需。
    故校验器应：按 1.0.0 判定 → 通过；同时**把缺口报成 warning**。
    这是「通过」与「假装完整」的分界：**过了，但缺口写明了**。
    """
    rep = validate(path=REAL)
    assert rep["declared_version"] == "1.0.0", "该产物应声明 1.0.0"
    assert rep["schema_version"] == "1.1.0", "校验器应用当前 schema 校验"
    joined = " ".join(rep["warnings"])
    assert "dual-core" in joined or "agent/human" in joined, (
        "未把双核心缺口报成 warning —— 那等于把「缺两条主线之一」当成完整通过")


def test_real_replay_has_a_size_check() -> None:
    """判据 3：文件大小有校验与分片提示。"""
    rep = validate(path=REAL)
    assert rep["size_bytes"] == REAL_BYTES
    assert rep["limit_bytes"] == 104857600, "100MB 限制（31006 5.1.2）"
    assert rep["size_bytes"] < rep["limit_bytes"], "真实产物本身应远小于上限"
    # 人为把上限压到 1 字节，必须给出分片提示且**不判失败**（建议性条款）
    tiny = validate(path=REAL, size_limit_bytes=1)
    assert tiny["ok"] is True, "超限是建议性条款，不该硬失败"
    assert any("sharding" in w for w in tiny["warnings"]), "超限未提示分片"


def test_real_replay_index_range_and_o1_lookup() -> None:
    """判据 1（前半）：按索引可精确取出指定 tick。"""
    with Replay(REAL) as r:
        assert r.frame_count == REAL_FRAMES
        assert r.index_range() == (0, REAL_FRAMES - 1)
        assert r.run_id == REAL_RUN_ID
        first, last = r.frame(0), r.frame(REAL_FRAMES - 1)
        assert first["time"]["cur_step"] == REAL_STEP_MIN
        assert last["time"]["cur_step"] == REAL_STEP_MAX


def test_real_replay_reproduces_any_tick_exactly() -> None:
    """**判据 1 的核心**：指定任意 tick 可精确重放且结果与原局一致。

    「精确重现」的可操作定义：**同一 tick 反复读取得到逐字段相同的结果**。
    抽若干 tick 覆盖首、尾、中段。
    """
    with Replay(REAL) as r:
        for tick in (0, 1, 17, 500, 900, 1500, REAL_STEP_MAX):
            a = r.frame_at_tick(tick)
            b = r.frame_at_tick(tick)
            assert a == b, "tick=%d 两次读取不一致" % tick
            assert a["time"]["cur_step"] == tick


# 实测：1802 帧、1801 个不同 step —— **step 0 出现了两次**（帧 0 与帧 1）。
# 故「按索引」与「按 tick」**不是 1:1**；schema.playback 已写明：
# 「允许多帧同 step，取第一帧」。下面按**那个契约**断言，而不是按 1:1 的错觉。
DUP_STEP = 0


def test_real_replay_has_a_duplicated_step_as_documented() -> None:
    """把「同 step 多帧」这个**真实存在**的性质钉住。

    若将来产物变成 1:1，本测试会失败并提醒：schema.playback 的
    「取第一帧」约定是否还需要保留，需要重新表态。
    """
    with Replay(REAL) as r:
        steps = [f["time"]["cur_step"] for _, f in r.iter_frames()]
    assert len(steps) == REAL_FRAMES
    assert len(set(steps)) == REAL_FRAMES - 1, (
        "实测应为 1801 个不同 step（step 0 重复一次），当前 %d 个" % len(set(steps)))
    occurrences = [i for i, s in enumerate(steps) if s == DUP_STEP]
    assert occurrences == [0, 1], "step 0 应出现在帧 0 与帧 1，当前 %s" % occurrences


def test_frame_at_tick_returns_the_first_frame_of_that_step() -> None:
    """契约：同 step 多帧时 `frame_at_tick` 取**第一帧**（schema.playback 明文）。

    这是「按 tick 回放」的语义基础：同一个 tick 必须确定地映射到唯一一帧，
    否则「指定任意 tick 可精确重放」不成立。
    """
    with Replay(REAL) as r:
        rows = [(i, f["time"]["cur_step"]) for i, f in r.iter_frames()]
        by_step = {}
        for i, step in rows:
            by_step.setdefault(step, []).append(i)
        for step in (DUP_STEP, 1, 17, 500, 900, REAL_STEP_MAX):
            idxs = by_step[step]
            assert r.frame_at_tick(step) == r.frame(idxs[0]), (
                "step=%d 应取第一帧 idx=%d" % (step, idxs[0]))


def test_real_replay_iter_frames_covers_everything() -> None:
    """全量遍历不丢帧、不重复，且每帧都带必需字段。"""
    with Replay(REAL) as r:
        seen = 0
        prev = None
        for idx, frame in r.iter_frames():
            seen += 1
            if prev is not None:
                assert idx == prev + 1, "帧序号不连续: %d -> %d" % (prev, idx)
            prev = idx
            for key in ("operators", "time", "scores", "actions", "valid_actions"):
                assert key in frame, "帧 %d 缺必需字段 %s" % (idx, key)
        assert seen == REAL_FRAMES


def test_real_replay_bounds_are_enforced() -> None:
    """越界不能静默返回空值 —— 回放越界必须显式失败。"""
    with Replay(REAL) as r:
        with pytest.raises(ReplayFormatError):
            r.frame(REAL_FRAMES)
        with pytest.raises(ReplayFormatError):
            r.frame(-1)
        with pytest.raises(ReplayFormatError):
            r.frame_at_tick(REAL_STEP_MAX + 10_000)


def test_real_replay_summary_matches_measured_values() -> None:
    """摘要端点与实测一致（防止产物被换成另一局而测试仍"通过"）。"""
    s = run_summary(REAL)
    assert s["run_id"] == REAL_RUN_ID
    assert s["frames"] == REAL_FRAMES
    assert s["step_min"] == REAL_STEP_MIN
    assert s["step_max"] == REAL_STEP_MAX
    assert s["scenario_id"] == REAL_SCENARIO_ID


def test_real_replay_human_and_agent_streams_are_the_known_gap() -> None:
    """**如实断言缺口**：真实 v1 复盘只记了原始 `actions`，没有标注发出者。

    31006 4.4.2 要求双核心。这里**不假装满足**，而是把现状钉住：
    `actions` 存在（人类操作原始命令），但 `agent` / `human` 两条主线缺失。
    将来写入端改造后，本测试会失败 —— 那正是它该提醒的时候。
    """
    with Replay(REAL) as r:
        f = r.frame(0)
    assert "actions" in f, "v1 应至少记下原始命令"
    for absent in ("meta", "agent", "human"):
        assert absent not in f, (
            "帧里出现了 %r —— 若写入端已按 1.1.0 改造，请更新本测试与 schema.dual_core"
            % absent)
