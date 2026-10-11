"""I04 判据 3 的**机械核验**：P3 侧 108 维的赋值覆盖，由 AST + 实跑双重确定。

## 为什么需要它

`docs/data_dictionary_mapping.md` 声称 P3 侧「**50 维从未被写入**：`38,39,60..107`」。
该声称**必须可自动核验**，否则文档会随源码漂移而无人发现。

## 两种手段互证

| 手段 | 得到什么 |
| --- | --- |
| **AST 解析** `operator_to_feature` | 哪些下标被**赋值语句**覆盖（静态） |
| **实跑完备输入** | 哪些下标在**任何输入下**都不可能非零（动态） |

**两者独立**，若一致则结论可靠。

## 本轮另发现的**条件维**（与死维必须区分）

`55`（城市数/16）与 `59`（最大城市价值/100）**仅在 `frame["cities"]` 非空时**写入。
**它们是数据依赖，不是设计缺口** —— 混为一谈会把"可实现"误报为"未实现"。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

np = pytest.importorskip("numpy")
from ai.features import FEATURE_DIM, operator_to_feature  # noqa: E402

DOC = REPO / "docs" / "data_dictionary_mapping.md"
FEATURES_PY = REPO / "ai" / "features.py"

# 文档声称的死维范围
DOC_DEAD = sorted([38, 39] + list(range(60, 108)))


def _numeric_fields() -> list:
    src = FEATURES_PY.read_text(encoding="utf-8")
    i = src.index("NUMERIC_FIELDS = [")
    j = src.index("]", i)
    return re.findall(r'"(\w+)"', src[i:j])


def _full_op() -> dict:
    op = {k: 1 for k in _numeric_fields()}
    op.update({"cur_hex": 123, "remain_bullet_nums": 5, "carry_weapon_ids": [1],
               "valid_passenger_types": [1], "max_passenger_nums": 2,
               "observe_distance": 3, "passenger_ids": [1], "launch_ids": [1],
               "see_enemy_bop_ids": [1]})
    return op


def _full_frame(cities=True) -> dict:
    return {
        "time": {"cur_step": 7, "tick": 1.0},
        "scores": {"red_total": 1, "blue_total": 1, "red_win": 1, "blue_win": 1},
        "operators": [{}] * 3,
        "cities": [{"value": 50}] if cities else [],
        "valid_actions": {"0": {}},
        "communication": [{}],
        "tank_enemy": [{}],
    }


# ══════════════ 静态：AST

def test_ast_constant_assignment_indices() -> None:
    """AST 解析出的常量下标应为 **40..59**，循环覆盖 **0..37**。"""
    src = FEATURES_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "operator_to_feature")
    const = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                        and t.value.id == "vector"
                        and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, int)):
                    const.add(t.slice.value)
    assert sorted(const) == list(range(40, 60)), sorted(const)
    assert len(_numeric_fields()) == 38, "NUMERIC_FIELDS 应为 38 个（下标 0..37）"


# ══════════════ 动态：完备输入实跑

def test_dead_dimensions_are_exactly_the_documented_50() -> None:
    """**核心**：任何输入都写不进的维，必须**恰好**是 `38,39,60..107`。"""
    v = operator_to_feature(_full_op(), frame=_full_frame())
    assert len(v) == FEATURE_DIM == 108
    dead = sorted(i for i in range(FEATURE_DIM) if v[i] == 0)
    assert dead == DOC_DEAD, "死维与文档不符：实测 %s" % dead
    assert len(dead) == 50, len(dead)


def test_used_dimensions_are_58() -> None:
    """完备输入下应有 **58 维可用**（0..37 + 40..59）—— 与 AST 推理吻合。"""
    v = operator_to_feature(_full_op(), frame=_full_frame())
    nonzero = sorted(i for i in range(FEATURE_DIM) if v[i] != 0)
    assert len(nonzero) == 58, len(nonzero)
    assert nonzero == sorted(list(range(38)) + list(range(40, 60))), nonzero


def test_dead_dimensions_stay_dead_under_minimal_input() -> None:
    """死维在**退化输入**下也必须为 0 —— 否则"死维"结论不成立。"""
    v = operator_to_feature(None, frame=None)
    for i in DOC_DEAD:
        assert v[i] == 0, "维 %d 在退化输入下非零" % i


def test_dead_dimensions_stay_dead_without_a_frame() -> None:
    """无 `frame` 时，死维仍为 0（排除"只是没给数据"的解释）。"""
    v = operator_to_feature(_full_op(), frame=None)
    for i in DOC_DEAD:
        assert v[i] == 0, "维 %d 在无 frame 时非零" % i


# ══════════════ 条件维（与死维**必须区分**）

def test_conditional_dimensions_are_city_dependent() -> None:
    """`55` 与 `59` 是**条件写入**：有 cities 才有值 —— 它们是**数据依赖，不是死维**。"""
    with_city = operator_to_feature(_full_op(), frame=_full_frame(cities=True))
    without = operator_to_feature(_full_op(), frame=_full_frame(cities=False))
    for i in (55, 59):
        assert with_city[i] != 0, "维 %d 在有 cities 时应非零" % i
        assert without[i] == 0, "维 %d 在无 cities 时应为 0（条件写入）" % i
    # 它们**不在**死维里 —— 这正是必须区分的点
    assert 55 not in DOC_DEAD and 59 not in DOC_DEAD


def test_conditional_dims_are_not_counted_as_dead() -> None:
    """把条件维算进"空洞"会把**数据依赖**误报成**设计缺口**。"""
    v = operator_to_feature(_full_op(), frame=_full_frame(cities=True))
    dead = {i for i in range(FEATURE_DIM) if v[i] == 0}
    assert 55 not in dead and 59 not in dead


# ══════════════ 文档一致性

def test_document_records_the_dead_range_and_the_conditional_dims() -> None:
    """文档必须同时记录**死维范围**与**条件维** —— 缺后者会误导读者。"""
    doc = DOC.read_text(encoding="utf-8")
    assert "50 维" in doc, "文档未记录空洞数量"
    assert "38,39,60..107" in doc or ("38, 39, 60..107" in doc), "文档未记录空洞范围"
    assert "条件空洞" in doc, "文档未把条件维与死维区分开（会误报为设计缺口）"
    for i in ("55", "59"):
        assert i in doc, "文档未列出条件维 %s" % i


def test_document_segments_the_void_into_decisions() -> None:
    """文档必须把 50 维**分段**并列出**待定决策** —— 否则"未写入"不可裁定。"""
    doc = DOC.read_text(encoding="utf-8")
    assert "2.3" in doc, "缺 §2.3 逐段登记"
    for seg in ("38..39", "60..63", "64..107"):
        assert seg in doc, "未登记空段 %s" % seg
    assert "待定的决策" in doc or "待定" in doc, "未列出待定决策项"
