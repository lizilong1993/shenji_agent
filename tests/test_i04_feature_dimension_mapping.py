"""I04 步骤 4 / Q10：两套 108 维的**语义对照**与维度契约。

本条（对照表）的结论建立在两组**实测事实**上，故用测试把它们钉住：
不会因为将来有人改了 `features.py` 或 `intent_features.py` 而**悄悄失效**。

⚠ 本文件**不需要引擎、不需要权重**：只读源码 + 构造向量。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import ai.features as F  # noqa: E402

DOC = REPO / "docs" / "data_dictionary_mapping.md"


# ─────────────────────── P3 侧：108 维里有多少是**真的被赋值的**

def _assignment_indices(func_src: str) -> set:
    """取一段函数源码里 `vector[N] =` 的显式下标。"""
    return {int(m) for m in re.findall(r"vector\[(\d+)\]\s*=", func_src)}


def _func_src(name: str) -> str:
    src = (REPO / "ai" / "features.py").read_text(encoding="utf-8")
    i = src.find("def %s(" % name)
    assert i > 0, "未找到函数 %s" % name
    j = src.find("\ndef ", i + 1)
    return src[i:j if j > 0 else len(src)]


def test_situation_vector_covers_exactly_58_dimensions() -> None:
    """P3 态势向量的**覆盖维数 = 58 / 108**（实测）。

    `operator_to_feature` 的 108 维由三块拼成：
      - `0..37`：`NUMERIC_FIELDS`（38 个）由循环填充；
      - 显式赋值：40..59（20 个）；
      - 其余 **50 维从未被写入**。

    这条测试的作用：若有人**补上**了那些维，本测试会失败并**强制更新对照表**；
    若有人**加长了** `NUMERIC_FIELDS` 或改动了下标，也会失败。
    """
    body = _func_src("operator_to_feature")
    explicit = _assignment_indices(body)
    covered = set(range(len(F.NUMERIC_FIELDS))) | explicit
    assert F.FEATURE_DIM == 108
    assert len(covered) == 58, "覆盖维数变了（%d），必须同步更新 docs/data_dictionary_mapping.md" % len(covered)
    assert explicit == set(range(40, 60)), "显式赋值下标变了: %s" % sorted(explicit)


def test_uncovered_dimensions_are_exactly_the_documented_holes() -> None:
    """未被覆盖的 **50 维**必须正好是文档里写的那 50 个。

    文档声称空洞是 `38, 39, 60..107`。这条把它钉死。
    """
    body = _func_src("operator_to_feature")
    covered = set(range(len(F.NUMERIC_FIELDS))) | _assignment_indices(body)
    missing = sorted(set(range(F.FEATURE_DIM)) - covered)
    expected = [38, 39] + list(range(60, 108))
    assert missing == expected, (
        "空洞位置与文档不符。文档写 38,39,60..107；实测 %s" % missing)
    assert len(missing) == 50


def test_situation_vector_has_50_always_zero_dimensions_on_real_shaped_input() -> None:
    """**行为验证**（不只是读源码）：按真实算子字段形状构造输入，那 50 维恒为 0。

    先证明"源码没写"→ 再证明"跑出来确实是 0"。两者都成立才敢写进文档。
    """
    op = {"obj_id": 101, "color": 0, "type": 3, "sub_type": 1, "blood": 80.0,
          "max_blood": 100.0, "cur_hex": 1234, "speed": 12.5, "value": 7,
          "remain_bullet_nums": [1, 2], "max_passenger_nums": 3}
    frame = {"time": {"cur_step": 10, "tick": 100},
             "scores": {"red_total": 5, "blue_total": 3, "red_win": 1, "blue_win": 0},
             "cities": [{"value": 50}], "operators": [op],
             "valid_actions": {"a": 1}, "communication": [], "tank_enemy": []}
    v = F.operator_to_feature(op, frame=frame)
    assert len(v) == 108
    holes = [38, 39] + list(range(60, 108))
    assert all(float(v[i]) == 0.0 for i in holes), \
        "有空洞维被写入了非零值 —— 文档需更新: %s" % [i for i in holes if v[i] != 0]


def test_graph_feature_path_is_a_different_vector() -> None:
    """`operator_to_graph_feature` 会写 `vector[70..72]`，**但那是另一条路径**。

    文档明确区分了这一点；若有人误以为 `operator_to_feature` 也有 70..72，会得出错结论。
    """
    body = _func_src("operator_to_graph_feature")
    assert {70, 71, 72} <= _assignment_indices(body)
    assert not ({70, 71, 72} & _assignment_indices(_func_src("operator_to_feature")))


def test_no_other_module_writes_the_situation_vector_holes() -> None:
    """全仓确认没有第二处写入 `vector[60..107]`（排除 `experiments/` 下的历史副本）。"""
    offenders = []
    for p in REPO.rglob("*.py"):
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith(("experiments/", ".git/", "venvs/")):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"vector\[(6[0-9]|7[0-9]|8[0-9]|9[0-9]|10[0-7])\]\s*=", txt):
            offenders.append(rel)
    assert offenders == ["ai/features.py"] or offenders == [], (
        "除 ai/features.py 外还有模块写这些下标: %s" % offenders)


# ─────────────────────── P2 侧：每一维都**有名字**

def test_p2_ifv2_is_fully_named_with_108_entries() -> None:
    """P2 的 108 维**无未标注项**（判据 3 的 P2 侧达成）。

    `FEATURE_NAMES` 是显式名字表，且与 `FEATURE_DIM` 一致。
    """
    s3d = REPO / "situation3d"
    if not (s3d / "engine" / "intent_features.py").is_file():
        pytest.skip("P2 engine not present in this checkout")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ifv2_mod", s3d / "engine" / "intent_features.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        pytest.skip("cannot import P2 intent_features standalone: %s" % exc)
    names = mod.feature_names()
    assert len(names) == 108, "P2 名字表长度应为 108，实际 %d" % len(names)
    assert all(isinstance(n, str) and n for n in names), "存在空名字"
    assert len(set(names)) == 108, "名字有重复"
    assert mod.feature_dim() == 108
    assert mod.FEATURE_SCHEMA_VERSION == "ifv2"


# ─────────────────────── 对照表本身

def test_mapping_doc_records_the_contrast_and_the_holes() -> None:
    """对照表必须**同时**写出「两套长度相同、语义不同」与「P3 有 50 维空洞」。

    这两条是 `Q10` 的核心结论；若文档被改得只剩"两者都是 108 维"，本测试失败。
    """
    if not DOC.is_file():
        pytest.skip("data_dictionary_mapping.md not present")
    d = DOC.read_text(encoding="utf-8")
    assert "50" in d and "108" in d, "未记录 50/108 的空洞"
    assert "ifv2" in d, "未提到 P2 的 ifv2"
    assert "FEATURE_NAMES" in d, "未说明 P2 有名字表"
    for token in ("38", "39", "60", "107"):
        assert token in d, "未记录空洞边界 %s" % token
    # 必须点明"不可当同义合并"
    assert "语义" in d, "未做语义区分"
