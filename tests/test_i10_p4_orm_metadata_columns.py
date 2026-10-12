# -*- coding: utf-8 -*-
"""P4 `Situation` ORM 必须覆盖 `situation_config` 里 16 个 NOT NULL 元数据列。

背景：`010-extend-situation-metadata.sql` 把 16 列设为 `NOT NULL` 且无 DB 默认值，
而 P4 的 ORM 只映射 6 列 —— 于是经 API 创建想定必然 500（远程实跑发现，见任务清单 §187.7）。
本测试对**已部署的 P4 源码**做静态断言，缺文件则跳过（本仓不含 P4 源码）。
"""
import ast
import re
from pathlib import Path

import pytest

P4_MODELS = Path("/remote-home/lizilong/demo20260104/backend/app/db/models.py")

REQUIRED_COLUMNS = [
    "scenario_type",
    "applicable_level",
    "service_domain",
    "environment_type",
    "estimated_duration_min",
    "opposing_force_desc",
    "friendly_force_desc",
    "complexity_level",
    "developed_date",
    "revised_date",
    "version_no",
    "author_org",
    "classification",
    "keywords",
    "related_scenarios",
    "applicable_platforms",
]


def _models_text() -> str:
    if not P4_MODELS.exists():
        pytest.skip("本机无已部署的 P4 源码：%s" % P4_MODELS)
    return P4_MODELS.read_text(encoding="utf-8")


def _situation_class_source(text: str) -> str:
    """截出 `class Situation(Base):` 的源码（到下一个顶层 class/def 为止）。"""
    start = text.index("class Situation(Base)")
    rest = text[start + 1 :]
    m = re.search(r"\n(?=class |def |@)", rest)
    return rest[: m.start()] if m else rest


def _mapped_columns(src: str) -> set[str]:
    """取 `Situation` 里每个 `mapped_column(...)` 实际对应的**列名**。

    列名有两种写法：`mapped_column("列名", 类型)`（属性名≠列名）与
    `mapped_column(类型)`（属性名即列名）。两种都要认。
    """
    cols = set()
    for line in src.splitlines():
        attr = re.match(r"\s*([A-Za-z_]\w*)\s*:\s*Mapped\[.*?\]\s*=\s*mapped_column\(", line)
        if not attr:
            continue
        named = re.search(r'mapped_column\(\s*"([A-Za-z_]\w*)"', line)
        cols.add(named.group(1) if named else attr.group(1))
    return cols


def test_all_not_null_metadata_columns_are_mapped() -> None:
    src = _situation_class_source(_models_text())
    mapped = _mapped_columns(src)
    missing = [c for c in REQUIRED_COLUMNS if c not in mapped]
    assert not missing, "ORM 未映射这些 NOT NULL 列：%s" % missing


def test_optional_item_19_is_nullable() -> None:
    """第 19 项原文自带「（可选）」，ORM 侧不得标成 NOT NULL。"""
    src = _situation_class_source(_models_text())
    m = re.search(r"evaluation_metrics:[^\n]*\n", src)
    assert m, "ORM 未映射 evaluation_metrics"
    assert "nullable=True" in m.group(0), "evaluation_metrics 应为 nullable=True：%s" % m.group(0).strip()


def test_required_columns_carry_no_python_side_default() -> None:
    """无 DB 默认值 → ORM 也不能给默认值，否则「缺字段」会被静默补成假值。"""
    src = _situation_class_source(_models_text())
    for col in REQUIRED_COLUMNS:
        m = re.search(r"^\s*%s:[^\n]*$" % re.escape(col), src, re.M)
        assert m, "ORM 未声明 %s" % col
        assert "default=" not in m.group(0), "%s 不应有 Python 侧默认值：%s" % (col, m.group(0).strip())


def test_module_parses_and_keeps_created_at_synonym() -> None:
    """上轮修的 `created_at = synonym("started_at")` 不得被改回去。"""
    text = _models_text()
    ast.parse(text)  # 语法层面
    assert 'created_at: Mapped[datetime] = synonym("started_at")' in text
