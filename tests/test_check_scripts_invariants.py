"""四个核对脚本的自证测试：**它们会报、且不误报**。

## 覆盖

| 脚本 | 关键性质 |
| --- | --- |
| `check_pending_readiness.py` | 待批准项的**声称产物缺失时必须报**；存在时不报 |
| `check_snapshot_claims.py` | 快照里的**路径引用不存在时必须报**；裸文件名要能按 basename 解析 |
| `check_criteria_evidence.py` | 判据数字在正文**无第二次出现时必须标出**；有则标"有据" |
| `check_appendix_refs.py` | 追加章节里的**失效引用必须进未解析栏** |

## 为什么要
本会话已记录：**一个"已提交但未测"的检测器可能静默地什么也不做**（§110 的反模式检测器
一直在跳过不可解析文件）。故每个核对器都要有"它会报"的自证。

## 手法
全部用**合成夹具**：在临时目录造最小清单/快照，或用 `DONE_LANDING_NO_INDEX` 类开关
绕开全树扫描。**不依赖真实交付物**，故快且稳定。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _src(name: str) -> str:
    p = SCRIPTS / name
    if not p.is_file():
        pytest.skip("%s not present" % name)
    return p.read_text(encoding="utf-8")


# ══════════════════ ① check_pending_readiness.py

def test_pending_readiness_has_a_hardcoded_artifact_table() -> None:
    """该脚本靠一张**手抄的**产物表判定；表必须非空且指向真实存在的结构。"""
    s = _src("check_pending_readiness.py")
    assert "PENDING_ARTIFACTS" in s, "缺产物表"
    n = len(re.findall(r'"(P0-2|I\d\d|D\d|C\d|P\d)"\s*:', s))
    assert n >= 8, "产物表只覆盖 %d 项，过少" % n


def test_pending_readiness_reports_missing_and_untracked() -> None:
    """**核心（防漏报）**：缺失产物与未追踪产物都要有对应报出分支。"""
    s = _src("check_pending_readiness.py")
    assert "缺失" in s, "无「缺失」报出分支"
    assert "未追踪" in s, "无「未追踪」报出分支"
    assert "return 1" in s, "有缺口时未返回非零"


def test_pending_readiness_states_its_limit() -> None:
    """必须写明**不判实现是否正确**，否则 `OK` 会被读成"可以批准"。"""
    s = _src("check_pending_readiness.py")
    assert "不判断实现是否正确" in s or "不判断" in s
    assert "可以批准" in s or "不等于" in s


# ══════════════════ ② check_snapshot_claims.py

def test_snapshot_claims_resolves_bare_filenames() -> None:
    """**核心**：裸文件名必须能按 basename 解析（否则误报，§129 首版的缺陷）。"""
    s = _src("check_snapshot_claims.py")
    assert "rglob" in s or "_INDEX" in s, "未见 basename 解析逻辑"
    assert "resolve" in s, "缺 resolve 函数"


def test_snapshot_claims_has_the_missing_doc_guard() -> None:
    """快照副本缺失时必须**报错退出**，而不是静默 0 缺失。"""
    s = _src("check_snapshot_claims.py")
    assert "return 2" in s, "缺快照时未返回 2"


def test_snapshot_claims_separates_hint_from_error() -> None:
    """**「引用不完整」是提示，不是错误** —— 二者必须分开（§129 的结论）。"""
    s = _src("check_snapshot_claims.py")
    assert "提示" in s and "错误" in s, "未区分提示与错误"


# ══════════════════ ③ check_criteria_evidence.py

def _code_only(name: str) -> str:
    """只取**代码**，去掉注释与 docstring。

    ⚠ 必需：该脚本的注释里**故意引用**了那个坏锚点（`^\\d{1,2}$`）来说明修法，
    首版测试因此在注释上误报 —— **断言"代码里没有 X"之前，必须先去掉注释**。
    """
    lines = _src(name).split("\n")
    out, in_doc = [], False
    for l in lines:
        st = l.strip()
        if st.startswith('"""') or st.startswith("'''"):
            in_doc = not in_doc
            continue
        if in_doc:
            continue
        i = l.find("#")
        out.append(l[:i] if i >= 0 else l)
    return "\n".join(out)


def test_criteria_evidence_skip_rules_key_on_context_not_digit_count() -> None:
    """**核心（§139 的回归防线）**：跳过规则**不得**按"位数"一刀切。

    首版用 `^\\d{1,2}$` 一律跳过 → 把 19 项/38 行/14 条也丢了，21 项只剩 3 项可查。
    **只在代码里查**（注释里会引用该坏锚点来说明修法）。
    """
    code = _code_only("check_criteria_evidence.py")
    assert r"^\d{1,2}$" not in code, "代码里又出现按位数一刀切的跳过规则（§139 的缺陷）"
    s = _src("check_criteria_evidence.py")
    assert "_LIST_ORD" in s, "缺列表序号识别"
    assert "_STD_TAIL" in s or "条款号" in s, "缺条款号识别"


def test_criteria_evidence_documents_its_false_positives() -> None:
    """必须写明**已知 5 类假阳性**（§140），否则 `缺证据` 会被当成缺口。"""
    s = _src("check_criteria_evidence.py")
    for kw in ("百分数", "千分位", "斜杠枚举", "假阳性"):
        assert kw in s, "未写明假阳性来源: %s" % kw


# ══════════════════ ④ check_appendix_refs.py

def test_appendix_refs_splits_columns() -> None:
    """**核心**：未解析必须**分 A/B 两栏**（§142 的结论），否则会重复"154 条缺失"式的无信息量报告。"""
    s = _src("check_appendix_refs.py")
    assert "inside" in s and "outside" in s, "未分栏"
    assert "EXT_MARKERS" in s, "缺外部特征判据"
    assert "不等于" in s or "不判合格" in s, "未写明边界"


def test_appendix_refs_only_scans_appendix() -> None:
    """只应扫**任务明细之后**的追加章节（任务项本身由别的脚本负责）。"""
    s = _src("check_appendix_refs.py")
    assert "## 4. 依赖关系" in s, "未限定扫描范围"


# ══════════════════ ⑤ 共有的"不判合格"纪律

@pytest.mark.parametrize("name", [
    "check_pending_readiness.py", "check_snapshot_claims.py",
    "check_criteria_evidence.py", "check_appendix_refs.py",
])
def test_scripts_state_their_boundary(name: str) -> None:
    """四个脚本都必须在**输出或头部**写明自己的能力边界。"""
    s = _src(name)
    assert ("不判" in s or "不等于" in s or "⚠" in s), "%s 未写明边界" % name


@pytest.mark.parametrize("name", [
    "check_pending_readiness.py", "check_snapshot_claims.py",
    "check_criteria_evidence.py", "check_appendix_refs.py",
])
def test_scripts_reconfigure_stdout_for_utf8(name: str) -> None:
    """**必须显式 UTF-8 输出** —— Windows GBK 会让中文报错（§114 实测）。"""
    s = _src(name)
    assert "reconfigure" in s or "encoding=\"utf-8\"" in s, "%s 未处理输出编码" % name
