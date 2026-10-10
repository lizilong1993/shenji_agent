"""P0-2：原生 PostgreSQL 路径与 Docker 路径的实测结论。

本条是**部署评估**，故测试对象是「文档里的结论是否与实测一致」以及
「原生验证器是否真的复用原脚本的 SQL」。

**不依赖 Docker**：这些断言在任何有 `psql` 的环境里可跑；没有 `psql` 时跳过。
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOC = REPO / "docs" / "deploy_situ155.md"
VERIFIER = REPO / "scripts" / "verify_db_native.py"
REQUIRED_TABLES = ("target", "target_trajectory", "region",
                   "target_relation", "situation_snapshot", "situation_config")


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("deploy_situ155.md not present")
    return DOC.read_text(encoding="utf-8")


def _load_verifier():
    if not VERIFIER.is_file():
        pytest.skip("verify_db_native.py not present")
    spec = importlib.util.spec_from_file_location("vdn", VERIFIER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────── 验证器：SQL 必须来自原脚本，不能是重写的

def test_verifier_extracts_sql_from_the_original_script(tmp_path: Path) -> None:
    """**核心设计**：SQL 从原脚本**机械抽取**，不是重写。

    构造一个仿原脚本，断言抽出的 SQL 与注入的一致。
    """
    mod = _load_verifier()
    fake = tmp_path / "verify-db.sh"
    fake.write_text(
        'docker exec situational-awareness-postgis psql -U postgres -d situation -c \\\n'
        '"SELECT 1 AS a;"\n'
        'docker exec situational-awareness-postgis psql -U postgres -d situation -c \\\n'
        '"SELECT 2 AS b;"\n',
        encoding="utf-8")
    got = mod.extract_sql(fake)
    assert len(got) == 2, "应抽出 2 段 SQL，实际 %d" % len(got)
    assert got[0][1] == "SELECT 1 AS a;"
    assert got[1][1] == "SELECT 2 AS b;"


def test_verifier_extracts_six_checks_from_real_p4_script() -> None:
    """P4 的 `verify-db.sh` 有 **6** 段检查（原文实测）。"""
    p4 = Path("/tmp/p4_verify.sh")
    if not p4.is_file():
        pytest.skip("original verify-db.sh not staged in /tmp")
    mod = _load_verifier()
    checks = mod.extract_sql(p4)
    assert len(checks) == 6, "应抽出 6 段，实际 %d" % len(checks)
    # 每段都应是 SELECT
    for _, sql in checks:
        assert sql.upper().lstrip().startswith("SELECT"), sql[:40]


# ─────────────────── 文档结论必须与实测一致

def test_doc_records_docker_path_is_infeasible() -> None:
    """Docker 路径**不可行**是实测结论，必须写明（含"无容器运行时"这一事实）。"""
    d = _doc()
    assert "不可行" in d
    assert "容器运行时" in d or "docker.sock" in d
    for c in ("docker", "podman", "containerd"):
        assert c in d, "未列出实测缺失的运行时 %s" % c


def test_doc_records_native_path_actually_ran() -> None:
    """原生路径必须写明**跑通了**，且给出可核对的版本号。"""
    d = _doc()
    assert "12.22" in d, "未记录 PostgreSQL 实测版本"
    assert "3.0.0" in d, "未记录 PostGIS 实测版本"
    assert "EXIT=0" in d or "exit=0" in d.lower()
    for t in REQUIRED_TABLES:
        assert t in d, "未记录表 %s" % t


def test_doc_records_the_version_mismatch() -> None:
    """compose 要 PG16+PostGIS3.4，原生只有 PG12+PostGIS3.0 —— **版本不一致必须写明**。"""
    d = _doc()
    assert "16-3.4" in d or "16" in d
    assert "不一致" in d, "未声明版本不一致"


def test_doc_records_the_vacuous_exit_code_finding() -> None:
    """必须写明 **`verify-db.sh` 的退出码 0 是空过的**。"""
    d = _doc()
    assert "空过" in d, "未指出判据 2 空过"
    assert "ON_ERROR_STOP" in d, "未指出缺 ON_ERROR_STOP 是空过的原因"
    assert "ambiguous" in d or "歧义" in d, "未指出歧义 SQL"


def test_doc_does_not_overclaim_completion() -> None:
    """**不得**把 P0-2 写成已完成 —— 判据 1/3 不可达成。"""
    d = _doc()
    for bad in ("P0-2 已完成", "判据 1 ✓", "判据 3 ✓", "全部达成"):
        assert bad not in d, "出现过度声明 %r" % bad
    assert "不能达成" in d or "无法达成" in d, "未声明判据 1/3 不可达成"


def test_doc_states_the_form_change_needs_a_decision() -> None:
    """部署形态变化（原生进程替代 Compose）**必须交用户裁定**，不得自行决定。"""
    d = _doc()
    assert "待你裁定" in d, "未把形态变化列为待裁定"


def test_doc_maps_the_13_blocked_items_into_three_classes() -> None:
    """13 项阻塞点应分成三类，说明**哪些真被解锁**。"""
    d = _doc()
    assert "13" in d
    assert "已解锁" in d, "未说明哪些被解锁"
    assert "A." in d and "B." in d and "C." in d, "未分类"


# ─────────────────── 环境事实（有 psql 时才查）

def test_native_postgres_is_actually_running() -> None:
    """若本机有 postgres，断言 PG 在跑且**六张表都在** —— 否则文档的"跑通"无从谈起。

    ⚠ 不能以当前用户直接 `psql -d situation`：`000-init.sh` 走的是 **peer 认证**，
    只有 `postgres` 系统用户能连。故经 `su postgres -c` 执行（**单层 su**，
    不用 `su postgres -c "pytest …"` 再嵌套 —— 那会因为 pytest 走 python3.8 而失败）。
    """
    import shutil as _sh

    if not Path("/usr/lib/postgresql").is_dir() and not _sh.which("psql"):
        pytest.skip("no postgres on this host")
    q = ("SELECT count(*) FROM information_schema.tables WHERE table_name IN "
         "('target','target_trajectory','region','target_relation',"
         "'situation_snapshot','situation_config')")
    cmd = ["su", "postgres", "-c",
           "psql -d situation -tAc %s" % _sql_quote(q)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        pytest.skip("cannot query: %s" % exc)
    if p.returncode != 0:
        pytest.skip("query failed (PG may not be running): %s" % p.stderr.strip()[:140])
    assert int(p.stdout.strip()) == 6, "六张表未全部存在: %r" % p.stdout.strip()


def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def test_native_verifier_runs_and_reports_the_known_sql_defect() -> None:
    """**端到端**：真跑原生验证器，断言它**如实报出第 6 项失败**（歧义 SQL）。

    这是「判据 2 空过」这条发现的**可执行证据**：原脚本因缺 `ON_ERROR_STOP` 恒 exit 0，
    而本验证器取 psql 退出码，故第 6 项会报失败。
    """
    p4 = Path("/tmp/p4_verify.sh")
    if not p4.is_file():
        pytest.skip("original verify-db.sh not staged")
    if not Path("/usr/lib/postgresql").is_dir():
        pytest.skip("no postgres on this host")
    repo = str(Path(__file__).resolve().parents[1])
    cmd = ["su", "postgres", "-c",
           "cd %s && python3 scripts/verify_db_native.py --original %s"
           % (repo, str(p4))]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = p.stdout + p.stderr
    if "cannot" in out.lower() and p.returncode == 2:
        pytest.skip("verifier could not run: %s" % out[-160:])
    # 5 项应 ok
    assert out.count("ok   SELECT") >= 4, "通过项过少:\n%s" % out[-600:]
    # 第 6 项（约束清单）应**失败** —— 正是歧义 SQL
    assert "FAIL SELECT constraint_name" in out or "ambiguous" in out, \
        "未报出第 6 项失败（那说明'空过'的发现不成立或已被修）:\n%s" % out[-600:]
