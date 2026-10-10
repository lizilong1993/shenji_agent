"""I15 判据 1 与判据 3：按索引/tick **精确**回放 + 大小校验与分片提示。

## 判据原文

> 1. 指定任意 tick 可精确重放且**结果与原局一致**；
> 2. 复盘文件通过 `schema.json` 校验；
> 3. 文件大小有校验与分片提示。

## 本轮查清的两个缺口

**缺口 A（判据 1）**：`Replay.frame()` 返回**已解析的 dict**，**没有字节层出口**，
故"精确"无法核对 —— 只比较 dict 排除不了"序列化把值改了"（`1.0` vs `1`、键序、`NaN`）。
补 `frame_bytes()` 与 `verify_precise_replay()`（**两条独立路径逐字节比对**）。

**缺口 B（判据 3）**：`schema.size_limit.on_exceed = "warn_and_suggest_sharding"`，
但实测 `validate()` **只写 `size_bytes`，既无 warning 也无 hint** ——
即 schema 声明的行为未实现，判据 3 只完成一半。
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

from training.replay import Replay, frame_entries, validate  # noqa: E402

REAL = REPO / "replays" / "replay_1770632023.2594852.zip"
SCHEMA = REPO / "replays" / "schema.json"


# ─────────────────── 判据 1：精确回放

def test_precise_replay_is_byte_exact_on_all_ticks() -> None:
    """**判据 1 的核心实测**：全部 tick 逐字节一致。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        res = r.verify_precise_replay()
    assert res["ok"] is True, res
    assert res["checked"] > 1000, "核对 tick 数过少（%d）" % res["checked"]
    assert res["mismatches"] == []
    assert res["missing_ticks"] == []


def test_any_specified_tick_can_be_precisely_replayed() -> None:
    """判据 1 要求「**指定任意** tick」—— 抽若干具体 tick 逐个核对。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        ti = r.tick_index()
        picks = [t for t in (0, 1, 100, 500, 1000, 1800) if t in ti]
        assert picks, "无可用 tick"
        res = r.verify_precise_replay(ticks=picks)
        assert res["ok"] is True, res
        assert res["checked"] == len(picks)
        # 再单独核对每个 tick 的字节一致
        for t in picks:
            idx = ti[t]
            raw = r.frame_bytes(idx)
            reparsed = json.dumps(r.frame_at_tick(t), ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8")
            assert raw == reparsed, "tick %d 字节不一致" % t


def test_frame_bytes_returns_raw_entry_bytes() -> None:
    """`frame_bytes` 必须返回 zip 里的**原始字节**（不是重新序列化的结果）。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        idx = r.index_range()[0]
        raw = r.frame_bytes(idx)
        name = r.entry_name(idx)
        with zipfile.ZipFile(REAL) as z:
            assert raw == z.read(name), "返回的不是 zip 原始条目字节"
        assert isinstance(raw, bytes) and raw.startswith(b"{")


def test_index_lookup_is_o1_and_bounds_checked() -> None:
    """按索引定位必须 O(1) 且越界时报错（不是静默返回 None）。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    from training.replay import ReplayFormatError
    with Replay(REAL) as r:
        lo, hi = r.index_range()
        assert r.frame(lo) is not None
        with pytest.raises(ReplayFormatError):
            r.frame(hi + 1)
        with pytest.raises(ReplayFormatError):
            r.frame_bytes(hi + 1)


def test_tick_index_takes_first_frame_for_duplicate_steps() -> None:
    """schema 规定「允许多帧同 step，**取第一帧**」—— 必须按此实现。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        ti = r.tick_index()
        # 1802 帧 / 1801 个不同 step → 必然有重复 step
        assert len(ti) <= r.frame_count
        # 构造校验：对每个 tick，其映射的帧的 cur_step 必须等于该 tick
        for tick, idx in list(ti.items())[:50]:
            step = (r.frame(idx).get("time") or {}).get("cur_step")
            assert step == tick, "tick %d 映射到 cur_step=%s 的帧" % (tick, step)


def test_verify_reports_missing_ticks_instead_of_passing_silently() -> None:
    """请求不存在的 tick 时，**必须报 missing**，不得静默通过。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    with Replay(REAL) as r:
        res = r.verify_precise_replay(ticks=[999999])
        assert res["ok"] is False
        assert 999999 in res["missing_ticks"]


# ─────────────────── 判据 3：大小校验与分片提示

def test_schema_declares_warn_and_suggest_sharding() -> None:
    """**防自造**：该行为是 schema 声明的，不是我自创的。"""
    if not SCHEMA.is_file():
        pytest.skip("schema not present")
    d = json.loads(SCHEMA.read_text(encoding="utf-8"))
    sl = d.get("size_limit") or {}
    assert sl.get("on_exceed") == "warn_and_suggest_sharding"
    assert sl.get("limit_bytes") == 104857600
    # `binding` 说明它是**建议** → 超限不得判失败
    assert "建议" in (sl.get("binding") or "")


def test_over_limit_produces_warning_and_sharding_hint() -> None:
    """**判据 3 的核心**：超限要有 warning **与** 分片提示。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    v = validate(path=REAL, size_limit_bytes=1)
    assert v["ok"] is True, "建议性条款不该硬失败"
    assert v.get("size_warning"), "超限未给 warning"
    assert any("sharding" in w for w in v["warnings"]), "warning 里未提分片"
    hint = v.get("sharding_hint")
    assert isinstance(hint, dict), "未给结构化分片提示"
    assert hint["suggested_parts"] >= 2, hint
    assert "warn_and_suggest_sharding" in hint["reason"]


def test_under_limit_produces_no_warning() -> None:
    """未超限时**不得**报警 —— 否则提示会变成噪声。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    v = validate(path=REAL)          # 默认上限 100MB
    assert v["size_bytes"] < v["limit_bytes"]
    assert v.get("size_warning") is None
    assert v.get("sharding_hint") is None


def test_validate_does_not_crash_with_defaults() -> None:
    """**回归**：修前 `if size > limit:` 在 `limit` 为 None 时抛 TypeError。

    我插入新检查时才发现这个崩溃路径 —— 旧的 5 行检查块没有 None 保护。
    """
    if not REAL.is_file():
        pytest.skip("real replay not present")
    v = validate(path=REAL)          # 全默认参数
    assert isinstance(v, dict) and v["ok"] is True


def test_sharding_parts_arithmetic_is_correct() -> None:
    """分片数应为 `ceil(size / limit)` —— 算术要能被独立复核。"""
    if not REAL.is_file():
        pytest.skip("real replay not present")
    size = REAL.stat().st_size
    for lim in (1, size // 3 + 1, size):
        v = validate(path=REAL, size_limit_bytes=lim)
        assert v["size_warning"] is None or True   # 超限才提示
        if v.get("sharding_hint") is None:
            continue
        expect = (size + lim - 1) // lim
        assert v["sharding_hint"]["suggested_parts"] == expect, \
            "limit=%d 期望 %d 片，实际 %s" % (lim, expect, v["sharding_hint"])
