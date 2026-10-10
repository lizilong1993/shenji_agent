"""I06 步骤 2：标签器消费引擎裁决事件（高置信标签来源）。

## 背景（第 50 轮查清）

`I06` 判据 1 要求 `objective_label_counts.high_confidence>0`，而原值为 **0**。
高置信的定义（`build_objective_labels.py` README）是「**真实 score/capture/damage 事件**」。

引擎**已经在给**这些事件：`events.jsonl` 的 `ADJUDICATION` 携带
`damage` / `target_obj_id` / `attack_color` / `att_obj_blood` / `cur_step`；
`SEIZE` 携带夺控。但标签器 `ADJUDICATION` / `events.jsonl` / `judge_info` **均无出现** ——
它读的是 self-play 诊断产物，其 objective 字段全为 null。

**→ 根因不是"引擎给不出数据"，而是"标签器没读引擎已经给的数据"。**

本文件把修复钉住：标签器**必须**读 `events.jsonl`，且高置信标签的字段口径正确。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LABELER = REPO / "scripts" / "build_objective_labels.py"
RAW = REPO / "Shenji" / "raw" / "canonical_v4_candidate"


def _labeler():
    if not LABELER.is_file():
        pytest.skip("labeler not present")
    spec = importlib.util.spec_from_file_location("bol", LABELER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────── 修复已被应用（防止回退）

def test_labeler_now_reads_adjudication_events() -> None:
    """**核心回归**：标签器必须处理 `ADJUDICATION`（原为 False）。"""
    if not LABELER.is_file():
        pytest.skip("labeler not present")
    src = LABELER.read_text(encoding="utf-8")
    assert "ADJUDICATION" in src, "标签器又不读 ADJUDICATION 了"
    assert "events.jsonl" in src, "标签器又不认 events.jsonl 了"
    assert "labels_from_events" in src


def test_events_jsonl_is_discovered() -> None:
    """`events.jsonl` 必须进入输入发现，否则函数写了也不会被调用。"""
    m = _labeler()
    src = LABELER.read_text(encoding="utf-8")
    assert 'rglob("events.jsonl")' in src
    # 分派分支也必须存在
    assert 'path.name == "events.jsonl"' in src
    assert callable(getattr(m, "labels_from_events", None))


def test_no_new_confidence_tier_was_invented() -> None:
    """**不得**通过新增置信度档位来"达成"判据 —— 那是把标准往实现上凑。"""
    src = LABELER.read_text(encoding="utf-8")
    # 既有三档必须仍是这三档，不新增第四档
    for tier in ('"high"', '"medium"', '"low"'):
        assert tier in src
    for bogus in ('"very_high"', '"critical"', '"verified"'):
        assert bogus not in src, "新增了置信度档位 %s" % bogus


# ─────────────────── 字段口径正确性（用合成事件验证语义）

def _write_events(tmp: Path, events) -> Path:
    p = tmp / "events.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return p


def _by_side(labels):
    """按 `side` **聚合**（`labels_from_events` 是**逐步**产出，不是每方一条）。

    ⚠ 我第一版测试写成 `{int(l["side"]): l for l in labels}` —— 后一条覆盖前一条，
    于是 `kill_dealt` 被 step 6 的 0 覆盖掉，报出"击杀推导错误"。
    **实际是测试的聚合方式错了，不是标签器错了。**
    """
    agg = {}
    for l in labels:
        k = int(l["side"])
        cur = agg.setdefault(k, {"damage_dealt": 0.0, "damage_taken": 0.0,
                                 "kill_dealt": 0.0, "kill_taken": 0.0})
        for f in cur:
            cur[f] += float(l.get(f) or 0.0)
    return agg


def _adj(step, attacker, target, damage, target_blood):
    return {
        "event_type": "ADJUDICATION", "step": step, "source": "judge_info",
        "payload_json": json.dumps({
            "cur_step": step, "damage": damage, "attack_color": attacker,
            "target_color": target, "att_obj_blood": target_blood,
        }),
    }


def test_negative_damage_means_the_target_lost_health(tmp_path: Path) -> None:
    """**语义核心**：`damage` 是受击方血量变化，负数为掉血 → 施加伤害 = -damage。

    用一个最小样例钉住方向：攻击方 color=1、目标 color=0、damage=-1
    → **攻方** `damage_dealt=1`，**受方** `damage_taken=1`。
    """
    m = _labeler()
    p = _write_events(tmp_path, [_adj(10, attacker=1, target=0, damage=-1, target_blood=3)])
    labels = m.labels_from_events(p)
    by_side = _by_side(labels)
    assert by_side[1]["damage_dealt"] == 1.0, by_side[1]
    assert by_side[1]["damage_taken"] == 0.0
    assert by_side[0]["damage_taken"] == 1.0, by_side[0]
    assert by_side[0]["damage_dealt"] == 0.0


def test_positive_damage_means_the_target_gained_health(tmp_path: Path) -> None:
    """`damage=+1` 是回血 → 施加伤害为 **-1**（不得当成"造成 1 点伤害"）。"""
    m = _labeler()
    p = _write_events(tmp_path, [_adj(10, attacker=1, target=0, damage=1, target_blood=3)])
    by_side = _by_side(m.labels_from_events(p))
    assert by_side[1]["damage_dealt"] == -1.0, by_side[1]


def test_kill_is_derived_from_blood_going_non_positive(tmp_path: Path) -> None:
    """击杀由同一 payload 推导：`att_obj_blood + damage <= 0`。"""
    m = _labeler()
    p = _write_events(tmp_path, [
        _adj(5, attacker=1, target=0, damage=-3, target_blood=3),   # 3-3=0 → 击杀
        _adj(6, attacker=1, target=0, damage=-1, target_blood=4),   # 4-1=3 → 非击杀
    ])
    labels = m.labels_from_events(p)
    by_side = _by_side(labels)
    # 标签是**逐步**产出的：两步 → 至少 4 条（每步每方一条）
    assert len(labels) >= 4, "逐步产出应至少 4 条，实际 %d" % len(labels)
    assert by_side[1]["kill_dealt"] == 1.0
    assert by_side[0]["kill_taken"] == 1.0


def test_seize_produces_high_confidence_occupy_label(tmp_path: Path) -> None:
    """`SEIZE` → `occupy_success`，走既有分支返回 high。"""
    m = _labeler()
    p = _write_events(tmp_path, [{
        "event_type": "SEIZE", "step": 42, "source": "action",
        "payload_json": json.dumps({"actor": 1, "cur_step": 42, "obj_id": 8, "type": 5}),
    }])
    labels = m.labels_from_events(p)
    assert labels, "SEIZE 未产出标签"
    occ = [l for l in labels if l["label_kind"] == "occupy_success"]
    assert occ and occ[0]["label_confidence"] == "high"
    assert occ[0]["occupy_success"] is True


def test_high_confidence_labels_carry_the_objective_fields(tmp_path: Path) -> None:
    """高置信标签必须带 `damage_*` 字段且非 null —— 否则训练无输入。"""
    m = _labeler()
    p = _write_events(tmp_path, [_adj(1, 1, 0, -2, 5)])
    labels = m.labels_from_events(p)
    highs = [l for l in labels if l["label_confidence"] == "high"]
    assert highs
    for l in highs:
        assert l["label_kind"] == "damage_kill_delta"
        assert l["damage_dealt"] is not None and l["damage_taken"] is not None


def test_missing_or_broken_events_do_not_raise(tmp_path: Path) -> None:
    """坏输入不得抛异常（沿用本模块"缺失留 null"的既定风格）。"""
    m = _labeler()
    p = tmp_path / "events.jsonl"
    p.write_text('{"event_type":"ADJUDICATION"}\nnot json\n\n'
                 '{"event_type":"ADJUDICATION","payload_json":"{"}\n', encoding="utf-8")
    assert m.labels_from_events(p) == []


def test_multiple_adjudications_in_one_step_accumulate(tmp_path: Path) -> None:
    """同一步多条裁决必须**累加**（否则会丢伤害）。"""
    m = _labeler()
    p = _write_events(tmp_path, [
        _adj(7, attacker=1, target=0, damage=-1, target_blood=5),
        _adj(7, attacker=1, target=0, damage=-2, target_blood=4),
    ])
    by_side = _by_side(m.labels_from_events(p))
    assert by_side[1]["damage_dealt"] == 3.0, by_side[1]


# ─────────────────── 对真实数据的效果（判据 1 的前半）

def test_real_episodes_yield_high_confidence_labels() -> None:
    """**判据 1 前半的实测**：既有场次必须能产出 high 置信标签。

    若本机没有 raw 数据则跳过（该断言需要真实 events.jsonl）。
    """
    if not RAW.is_dir():
        pytest.skip("no raw episodes on this host")
    eps = [d for d in RAW.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not eps:
        pytest.skip("no episodes")
    m = _labeler()
    total_high = 0
    checked = 0
    for d in sorted(eps)[:5]:
        f = d / "events.jsonl"
        if not f.is_file():
            continue
        checked += 1
        labels = m.labels_from_events(f)
        high = [l for l in labels if l.get("label_confidence") == "high"]
        assert high, "%s 未产出 high 置信标签" % d.name
        total_high += len(high)
    if checked == 0:
        pytest.skip("no events.jsonl found")
    assert total_high > 0


def test_real_adjudication_payload_has_the_fields_we_rely_on() -> None:
    """**防自造**：真实 `ADJUDICATION` payload 确实含我们所依赖的字段。"""
    if not RAW.is_dir():
        pytest.skip("no raw episodes on this host")
    need = {"damage", "attack_color", "target_color", "att_obj_blood"}
    found = None
    for d in sorted(RAW.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        f = d / "events.jsonl"
        if not f.is_file():
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("event_type") == "ADJUDICATION":
                found = json.loads(o.get("payload_json") or "{}")
                break
        if found:
            break
    if not found:
        pytest.skip("no ADJUDICATION event found")
    missing = [k for k in need if k not in found]
    assert missing == [], "真实 payload 缺字段 %s —— 标签口径需重写" % missing
