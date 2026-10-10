"""任务清单路径引用核对（`docs/tasklist_path_audit_20261010.md`）。

起因：第 45—46 轮两次因清单路径写错而多花时间（`D1` 的 `collect_cli.py`）。
本文件把「清单引用的文件是否存在」变成**可重复执行的核对**，
并守住几条已确认的事实，避免它们被重新写错。

**注意**：本文件不断言"未产出文件一定不该被引用" ——
清单里有些引用是**计划要产出的**（`docs/merge_buckets.md` 等），
故只断言**已确认的路径错误**与**已确认的缺失**。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOC = REPO / "docs" / "tasklist_path_audit_20261010.md"
TASKLIST = REPO.parent.parent / "outputs" / "兵棋推演9项团体标准对标任务清单_20261010.md"


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("tasklist_path_audit not present")
    return DOC.read_text(encoding="utf-8")


# ─────────────────── 已确认的路径订正（防止被改回）

def test_collect_cli_real_path_is_recorded() -> None:
    """`D1`【落点】曾写 `Shenji\\collect_cli.py`（不存在）。

    真实路径是 `Shenji/src/shenji_dataset/collect_cli.py`。
    """
    d = _doc()
    assert "Shenji\\src\\shenji_dataset\\collect_cli.py" in d or \
           "Shenji/src/shenji_dataset/collect_cli.py" in d, "未记录 collect_cli 的真实路径"
    # 真实文件必须存在（若不存在则说明又变了）
    real = REPO / "Shenji" / "src" / "shenji_dataset" / "collect_cli.py"
    if not real.is_file():
        pytest.skip("repo layout differs on this host")
    assert real.is_file()


def test_placeholder_references_are_recorded_as_errors() -> None:
    """`docs/xxx.md` 与 `tools/test_x.py` 是**占位符**，不是真文件。"""
    d = _doc()
    for ph in ("xxx.md", "test_x.py"):
        assert ph in d, "未记录占位符引用 %s" % ph
    assert (REPO / "docs" / "xxx.md").is_file() is False
    assert (REPO / "tools" / "test_x.py").is_file() is False


def test_baseline_ledger_mapping_is_recorded() -> None:
    """`docs\\baseline_ledger.md` 不存在；实际为 `outputs\\基线台账_20261010.md`。"""
    d = _doc()
    assert "baseline_ledger.md" in d
    assert "基线台账_20261010.md" in d


def test_tally_filename_correction_is_recorded() -> None:
    """清单曾写 `标准量化指标对标台账`，实际文件名含「基线与」。"""
    d = _doc()
    assert "标准量化指标基线与对标台账_20261010.md" in d
    # 实际文件必须在 outputs 下
    real = REPO.parent.parent / "outputs" / "标准量化指标基线与对标台账_20261010.md"
    if (REPO.parent.parent / "outputs").is_dir():
        assert real.is_file(), "订正后的文件名仍不存在"


# ─────────────────── gen_eval_cases.py：与 Q7 直接相连的缺口

def test_gen_eval_cases_absence_is_recorded() -> None:
    """**最要紧的一条**：评测用例生成脚本全工作区不存在。

    这与 `Q7`/`I16`「每种对抗强度>100 用例」直接相连 ——
    **不只缺定义，也缺工具**。
    """
    d = _doc()
    assert "gen_eval_cases.py" in d
    # 复核：仓库里确实没有
    hits = list(REPO.rglob("gen_eval_cases*"))
    assert hits == [], "仓库里出现了 gen_eval_cases —— 文档需更新"


def test_audit_links_the_missing_generator_to_q7() -> None:
    """该缺口必须与 `Q7` 关联，否则会被当成孤立问题。"""
    d = _doc()
    assert "Q7" in d
    assert "100" in d, "未指出与「每档>100 用例」的关系"


# ─────────────────── 6 个 P4 文件

def test_six_p4_files_are_recorded_as_absent_everywhere() -> None:
    """6 个 `backend/...` 文件在**三个** P4 树里命中均为 0/6。"""
    d = _doc()
    for f in ("consistency_check.py", "sync.py", "ws_auth.py",
              "test_sync_consistency.py", "bench_ws.py", "bench_sync_latency.py"):
        assert f in d, "未记录 %s" % f
    assert "0 / 6" in d, "未记录命中数 0/6"


def test_p4_trees_really_lack_those_files() -> None:
    """**复核**：本地 P4 树里确实找不到（有则强制改文档）。"""
    code = REPO.parent
    trees = [code / "demo20260104", code / "demo20251219", code / "demo"]
    present = [t.name for t in trees if t.is_dir()]
    if not present:
        pytest.skip("no P4 trees on this host")
    for t in trees:
        if not t.is_dir():
            continue
        for name in ("consistency_check.py", "sync.py", "ws_auth.py"):
            assert not list(t.rglob(name)), "%s 里出现了 %s —— 文档需更新" % (t.name, name)


# ─────────────────── 发布信息：许可证是"待决定"而非"未做"

def test_data_card_license_is_recorded_as_a_decision_not_a_gap() -> None:
    """`D1` 判据 3 要「版本、哈希、许可证」。

    实测：版本与哈希**已实现**；**许可证是 `DATA_CARD.md` 明写"intentionally not
    selected"，属待项目负责人决定** —— 不该被算作实现缺口。
    """
    d = _doc()
    assert "许可证" in d
    assert "intentionally not selected" in d or "待你决定" in d or "待决定" in d

    card = REPO / "Shenji" / "DATA_CARD.md"
    if not card.is_file():
        pytest.skip("DATA_CARD.md not present")
    t = card.read_text(encoding="utf-8", errors="replace")
    assert "intentionally not selected" in t, "DATA_CARD 的许可证声明变了，文档需更新"
    assert "Licensing" in t


def test_release_script_provides_version_and_hash() -> None:
    """`release.py` 确实提供版本与哈希（否则判据 3 的另外两项也不成立）。"""
    rel = REPO / "Shenji" / "src" / "shenji_dataset" / "release.py"
    if not rel.is_file():
        pytest.skip("release.py not present")
    t = rel.read_text(encoding="utf-8", errors="replace")
    assert "sha256" in t
    for k in ("dataset_version", "schema_version", "ontology_version"):
        assert k in t, "release.py 缺 %s" % k


# ─────────────────── 审计本身的可复现性

def test_audit_documents_its_method() -> None:
    """核对方法必须写明，使审计可复现（否则只是断言）。"""
    d = _doc()
    assert "180" in d, "未记录 basename 总数"
    assert "256" in d or "141" in d, "未记录引用总数/命中数"
    assert "可复现" in d or "核对方法" in d


def test_tasklist_still_contains_the_references_audited() -> None:
    """若任务清单已改（如已订正路径），审计文档需同步。"""
    if not TASKLIST.is_file():
        pytest.skip("tasklist not present")
    t = TASKLIST.read_text(encoding="utf-8")
    assert "gen_eval_cases" in t, "清单已不提 gen_eval_cases —— 审计需更新"
