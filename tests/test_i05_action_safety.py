"""I05 步骤 4：动作 schema 双校验单测。

判据 2 要求「非法动作拒绝率 100%」，故本文件的核心是**负向用例**：
缺字段被拒、越界被拒、畸形被拒 —— 且必须在 `strict` 模式下**逐个**被拒。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ai.action_safety import (ACTION_SPECS, check_consequences, check_preconditions,
                              supported_action_types, validate_action, validate_actions)


# ─────────────────────────────── 合法动作必须通过（三类判据之一）

@pytest.mark.parametrize("action", [
    {"actor": 0, "type": 1, "obj_id": 101, "move_path": [1, 2, 3]},          # 机动
    {"actor": 0, "type": 2, "obj_id": 101, "target_obj_id": 202, "weapon_id": 1},  # 打击
    {"actor": 0, "type": 5, "obj_id": 101},                                   # 夺控
    {"actor": 0, "type": 6, "obj_id": 101, "target_state": 3},                # 切换状态
    {"actor": 0, "type": 21, "obj_id": 101, "direction": 0},                  # 局部干扰（下界）
    {"actor": 0, "type": 21, "obj_id": 101, "direction": 5},                  # 局部干扰（上界）
    {"actor": 0, "type": 22},                                                 # 全局干扰（无附加字段）
    {"actor": 0, "type": 333},                                                # 结束部署
    {"actor": 0, "type": 404, "sub_type": 1, "color": 0, "hex": 1234},        # 导演增加算子
    {"actor": 0, "type": 205, "msg_body": {"hexs": [1111], "graphic_type": "attention"}},
])
def test_valid_actions_pass_strict(action) -> None:
    r = validate_action(action, mode="strict")
    assert r["ok"] is True, r["problems"]


# ─────────────────────────────── 缺字段被拒

@pytest.mark.parametrize("action,missing", [
    ({"actor": 0, "type": 1, "obj_id": 101}, "move_path"),
    ({"actor": 0, "type": 2, "obj_id": 101, "weapon_id": 1}, "target_obj_id"),
    ({"actor": 0, "type": 6, "obj_id": 101}, "target_state"),
    ({"actor": 0, "type": 21, "obj_id": 101}, "direction"),
    ({"actor": 0, "type": 207, "seat": 1, "hex": 2, "start_time": 0, "end_time": 9,
      "unit_ids": [1], "route": [2]}, None),   # 合法对照组（本行应通过）
])
def test_missing_required_field_is_rejected(action, missing) -> None:
    r = validate_action(action, mode="strict")
    if missing is None:
        assert r["ok"] is True, r["problems"]
        return
    assert r["ok"] is False
    assert any(p == "missing_required_field:%s" % missing for p in r["problems"]), r["problems"]


def test_missing_actor_is_rejected() -> None:
    r = validate_action({"type": 5, "obj_id": 1}, mode="strict")
    assert r["ok"] is False and "missing_actor" in r["problems"]


def test_missing_type_is_rejected() -> None:
    r = validate_action({"actor": 0, "obj_id": 1}, mode="strict")
    assert r["ok"] is False
    assert any(p.startswith("missing_or_invalid_type") for p in r["problems"])


def test_empty_required_list_is_rejected() -> None:
    """空 move_path 等于"没动"，属畸形而非合法。"""
    r = validate_action({"actor": 0, "type": 1, "obj_id": 1, "move_path": []}, mode="strict")
    assert r["ok"] is False
    assert any(p.startswith("empty_list_field") for p in r["problems"])


# ─────────────────────────────── 越界被拒

@pytest.mark.parametrize("action,bad", [
    ({"actor": 0, "type": 21, "obj_id": 1, "direction": 6}, "direction"),   # 上界+1
    ({"actor": 0, "type": 21, "obj_id": 1, "direction": -1}, "direction"),  # 下界-1
    ({"actor": 0, "type": 6, "obj_id": 1, "target_state": 6}, "target_state"),
    ({"actor": 0, "type": 404, "sub_type": 1, "color": 2, "hex": 1}, "color"),
    ({"actor": 0, "type": 204, "to_all": 2, "msg_body": "x"}, "to_all"),
])
def test_out_of_range_field_is_rejected(action, bad) -> None:
    r = validate_action(action, mode="strict")
    assert r["ok"] is False
    assert any(p.startswith("field_out_of_range:%s" % bad) for p in r["problems"]), r["problems"]


def test_non_numeric_where_number_expected() -> None:
    r = validate_action({"actor": 0, "type": 21, "obj_id": 1, "direction": "上"}, mode="strict")
    assert r["ok"] is False
    assert any(p.startswith("field_not_numeric:direction") for p in r["problems"])


# ─────────────────────────────── 畸形输入

@pytest.mark.parametrize("bad", [None, 42, "type5", [], [1, 2]])
def test_malformed_action_is_rejected(bad) -> None:
    r = validate_action(bad, mode="strict")
    assert r["ok"] is False


def test_actions_not_a_list_is_rejected() -> None:
    r = validate_actions({"type": 5}, mode="strict")
    assert r["ok"] is False and "actions_not_a_list" in r["problems"]


# ─────────────────────────────── 模式语义（风险与回滚要求）

def test_warn_mode_never_blocks_but_records() -> None:
    """默认 warn：**记录但不拦截** —— 过严的校验会拦掉合法战术动作。"""
    r = validate_action({"actor": 0, "type": 1, "obj_id": 1}, mode="warn")
    assert r["ok"] is True                    # 不拦
    assert r["problems"]                      # 但记下了问题
    assert any("move_path" in p for p in r["problems"])


def test_off_mode_does_nothing() -> None:
    r = validate_action({"actor": 0, "type": 1}, mode="off")
    assert r["ok"] is True and r["problems"] == []


def test_invalid_mode_fails_closed() -> None:
    """模式写错不能静默变成"不校验"。"""
    r = validate_action({"actor": 0, "type": 5, "obj_id": 1}, mode="lenient")
    assert r["ok"] is False
    assert any(p.startswith("invalid_mode") for p in r["problems"])


def test_default_mode_is_warn() -> None:
    r = validate_action({"actor": 0, "type": 1, "obj_id": 1})
    assert r["mode"] == "warn" and r["ok"] is True


def test_allowlist_bypasses() -> None:
    """I05 要求支持白名单旁路。"""
    bad = {"actor": 0, "type": 1, "obj_id": 1}      # 缺 move_path
    assert validate_action(bad, mode="strict")["ok"] is False
    r = validate_action(bad, mode="strict", allowlist=[1])
    assert r["ok"] is True and r.get("allowlisted") is True


# ─────────────────────────────── 未知类型：不判非法

def test_unknown_action_type_is_flagged_not_rejected() -> None:
    """平台会新增动作，**未知不等于非法**：warn 记录、strict 才拦。"""
    act = {"actor": 0, "type": 99999}
    w = validate_action(act, mode="warn")
    assert w["ok"] is True
    assert any(p.startswith("unknown_action_type") for p in w["problems"])
    s = validate_action(act, mode="strict")
    assert s["ok"] is False


def test_type_as_string_is_accepted() -> None:
    """线上报文可能以字符串到达，应兼容而非判失败。"""
    r = validate_action({"actor": 0, "type": "5", "obj_id": 1}, mode="strict")
    assert r["ok"] is True and r["action_type"] == 5


# ─────────────────────────────── 后果约束（第二类校验）

def test_target_not_present_is_flagged() -> None:
    obs = {"operators": [{"obj_id": 101}, {"obj_id": 102}]}
    act = {"actor": 0, "type": 2, "obj_id": 101, "target_obj_id": 999, "weapon_id": 1}
    probs = check_consequences(act, obs)
    assert any("target_obj_not_present:999" in p for p in probs)


def test_operator_not_controllable_is_flagged() -> None:
    act = {"actor": 0, "type": 5, "obj_id": 777}
    probs = check_consequences(act, None, controllable_ops={1, 2, 3})
    assert any("operator_not_controllable:777" in p for p in probs)


def test_consequences_without_observation_reports_nothing() -> None:
    """**查不了 ≠ 不合法**：无环境可比的离线场景不应报问题。"""
    act = {"actor": 0, "type": 2, "obj_id": 1, "target_obj_id": 999, "weapon_id": 1}
    assert check_consequences(act, None) == []


def test_director_action_by_non_director_is_flagged() -> None:
    act = {"actor": 0, "type": 401, "target_obj_id": 5}
    probs = check_consequences(act, {"role": 1, "operators": [{"obj_id": 5}]})
    assert "director_action_by_non_director" in probs


# ─────────────────────────────── 判据 2：拒绝率 100%

def test_strict_rejection_rate_is_100_percent_on_invalid_corpus() -> None:
    """构造一批**全部非法**的动作，strict 下必须**无一漏网**。"""
    invalid = [
        {"actor": 0, "type": 1, "obj_id": 1},                                  # 缺 move_path
        {"actor": 0, "type": 2, "obj_id": 1, "weapon_id": 1},                  # 缺 target
        {"actor": 0, "type": 1, "obj_id": 1, "move_path": []},                 # 空列表
        {"actor": 0, "type": 21, "obj_id": 1, "direction": 9},                 # 越界
        {"actor": 0, "type": 6, "obj_id": 1, "target_state": -1},              # 越界
        {"type": 5, "obj_id": 1},                                              # 缺 actor
        {"actor": 0, "obj_id": 1},                                             # 缺 type
        {"actor": 0, "type": "abc"},                                           # type 非法
        None, 42, "x",                                                         # 畸形
    ]
    r = validate_actions(invalid, mode="strict")
    assert r["checked"] == len(invalid)
    assert len(r["rejected"]) == len(invalid), \
        "拒绝率必须 100%%，漏网: %s" % [i for i, a in enumerate(invalid)
                                        if a not in [x["action"] for x in r["rejected"]]]
    assert r["ok"] is False


# ─────────────────────────────── 字典自身的一致性

def test_action_specs_are_self_consistent() -> None:
    types = supported_action_types()
    assert len(types) == len(ACTION_SPECS)
    for t, spec in ACTION_SPECS.items():
        assert isinstance(t, int)
        assert isinstance(spec.get("required"), list), "type=%d 缺 required" % t
        for f in spec.get("required", []):
            assert isinstance(f, str) and f, "type=%d 的必需字段名非法" % t


def test_dictionary_is_grounded_in_the_platform_doc() -> None:
    """字典必须**由平台文档支撑**，而不是拍脑袋写的。

    做法：直接从 `docs/wargame_ref/actions_reference.txt` 抽出所有 `"type": N`，
    断言**全部落在字典里**。这比断言一个魔数强 ——
    字典可以增补（文档 "Field Name/Data Type" 表格里的
    type 25/27/28/29/30/31/32/33 就是 JSON 示例未覆盖、但文档成表列出的），
    但**不允许出现文档里没有的类型**。
    """
    import re
    from pathlib import Path as _P

    doc = REPO / "docs" / "wargame_ref" / "actions_reference.txt"
    if not doc.exists():
        pytest.skip("platform action reference not present")
    txt = doc.read_text(encoding="utf-8", errors="replace").replace("&quot;", '"')
    doc_types = {int(m) for m in re.findall(r'"type"\s*:\s*(\d+)', txt)}

    assert doc_types, "未能从平台文档解析出任何动作类型"
    missing = sorted(doc_types - set(ACTION_SPECS))
    assert missing == [], "字典缺少文档中出现的动作类型: %s" % missing

    # 文档 JSON 示例覆盖 42 种；另有 8 种只在 "Field Name/Data Type" 表格里成表列出
    assert len(doc_types) == 42
    assert len(ACTION_SPECS) == 50

    # 代表性类型必须在
    for t in (1, 2, 5, 6, 21, 22, 100, 207, 333, 401, 404):
        assert t in ACTION_SPECS, "缺少 type=%d" % t


def test_preconditions_do_not_require_observation() -> None:
    """前置条件校验必须能**离线**跑（这是它与后果约束的分界）。"""
    assert check_preconditions({"actor": 0, "type": 1, "obj_id": 1, "move_path": [1]}) == []
