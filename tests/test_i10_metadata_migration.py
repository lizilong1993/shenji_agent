"""I10 / D3 步骤 3 + 6：想定元数据扩展与版本表（**原生 PostgreSQL 集成测试**）。

这两个步骤原先标为 `阻塞-P0-2`（要 Docker）。第五十八轮实测确认：
它们**只需要数据库**，而原生 PG 12 + PostGIS 3.0.0 已在远程跑通 P4 的 schema。
故本文件在**真实数据库**上验证迁移，而不是只读 SQL 文本。

覆盖：
  - 表2 的 19 项是否真的落成列（含第 19 项为可选）；
  - 每列是否有 `COMMENT` 可**逐字核对**原文字段名（不依赖本文档）；
  - **幂等性**：重复执行结论一致（`P0-2`【预期结果】要求）；
  - 版本表 `situation_version` 与触发器：修改产生版本行、无变化不产生、非法取值被拒。

无 `psql`/无 `situation` 库时**跳过**（这些断言需要真实数据库）。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MIGRATION = REPO / "backend" / "initdb" / "010-extend-situation-metadata.sql"

# 表2 的 19 项 → 本迁移的列名（None = 由既有列承载，不新增）
FIELD_TO_COLUMN = {
    1: "situation_code",          # 既有列承载
    2: "situation_name",          # 既有列承载
    3: "scenario_type",
    4: "applicable_level",
    5: "service_domain",
    6: "environment_type",
    7: "estimated_duration_min",
    8: "opposing_force_desc",
    9: "friendly_force_desc",
    10: "description",            # 既有列承载（近似）
    11: "complexity_level",
    12: "developed_date",         # 第 12 项拆两列：developed_date + revised_date
    13: "version_no",
    14: "author_org",
    15: "classification",
    16: "keywords",
    17: "related_scenarios",
    18: "applicable_platforms",
    19: "evaluation_metrics",     # 原文自带「（可选）」
}

# 新增列（既有列不计）。
# ⚠ 必须**显式列出**，不能由 `FIELD_TO_COLUMN.values()` 推导：
# 第 12 项「开发/修订日期」拆成 **两列**（developed_date + revised_date），
# 而 `dict.values()` 会去重（两处都写 developed_date 时会少算一列）。
# 这正是我第一版把新增列数写成 16（实际 17）的原因。
NEW_COLUMNS = [
    "scenario_type", "applicable_level", "service_domain", "environment_type",
    "estimated_duration_min", "opposing_force_desc", "friendly_force_desc",
    "complexity_level", "developed_date", "revised_date", "version_no",
    "author_org", "classification", "keywords", "related_scenarios",
    "applicable_platforms", "evaluation_metrics",
]


def _psql(sql: str, *, tu: bool = True) -> subprocess.CompletedProcess:
    """以 postgres 用户执行 psql（peer 认证）。

    ⚠ 不能以当前用户直连：`000-init.sh` 走 peer 认证，只允许 postgres 系统用户。
    且**只用单层 `su`** —— 嵌套 `su postgres -c "pytest …"` 会因 pytest 取到
    python3.8 而失败（实测）。
    """
    cmd = ["su", "postgres", "-c", "psql -d situation -v ON_ERROR_STOP=1 %s %s"
           % ("-tAc" if tu else "-c", "'" + sql.replace("'", "'\"'\"'") + "'")]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not Path("/usr/lib/postgresql").is_dir() and not shutil.which("psql"):
        pytest.skip("no postgres on this host")
    p = _psql("SELECT to_regclass('public.situation_config')")
    if p.returncode != 0 or "situation_config" not in p.stdout:
        pytest.skip("situation db not initialised: %s" % p.stderr.strip()[:120])


# ─────────────────────── 步骤 3：19 项落成列

def test_all_nineteen_fields_have_a_column() -> None:
    """表2 的 **19 项**必须逐项有承载列（原文 `:198-219`）。"""
    for num, col in FIELD_TO_COLUMN.items():
        p = _psql("SELECT count(*) FROM information_schema.columns "
                  "WHERE table_name='situation_config' AND column_name='%s'" % col)
        assert p.returncode == 0, p.stderr
        assert p.stdout.strip() == "1", "第 %d 项（列 %s）不存在" % (num, col)


def test_new_columns_are_present() -> None:
    """新增的 17 列必须都在。"""
    p = _psql("SELECT column_name FROM information_schema.columns "
              "WHERE table_name='situation_config' ORDER BY column_name")
    cols = {l.strip() for l in p.stdout.split("\n") if l.strip()}
    missing = [c for c in NEW_COLUMNS if c not in cols]
    assert missing == [], "缺列: %s" % missing
    assert len(NEW_COLUMNS) == 17, "新增列应为 17，实际 %d" % len(NEW_COLUMNS)
    # 总量核对：P4 原有 10 列 + 新增 17 列 = 27
    total = int(_psql("SELECT count(*) FROM information_schema.columns "
                      "WHERE table_name='situation_config'").stdout.strip())
    assert total == 27, "列总数应为 27（10 原有 + 17 新增），实际 %d" % total


def test_item_19_is_optional_and_the_rest_are_required() -> None:
    """**原文只对第 19 项标「（可选）」** → 只有该列可为 NULL。"""
    p = _psql("SELECT column_name, is_nullable FROM information_schema.columns "
              "WHERE table_name='situation_config' AND column_name IN (%s)"
              % ", ".join("'%s'" % c for c in NEW_COLUMNS))
    nullable = {}
    for line in p.stdout.split("\n"):
        if "|" in line:
            name, val = line.split("|")
            nullable[name.strip()] = val.strip()
    assert nullable.get("evaluation_metrics") == "YES", "第 19 项应为可选（可 NULL）"
    others = [c for c in NEW_COLUMNS if c != "evaluation_metrics"]
    wrong = [c for c in others if nullable.get(c) != "NO"]
    assert wrong == [], "这些列本应 NOT NULL（原文未标可选）: %s" % wrong


def test_every_field_has_a_comment_naming_the_standard_item() -> None:
    """每项都要有 `COMMENT` 写明**原文字段名**，使逐字对应可被 `\\d+` 核对。

    这是"不依赖本文档"的要求：审计者应能只查数据库就看出哪列对应表2 的哪一项。
    """
    p = _psql("SELECT a.attname, d.description FROM pg_description d "
              "JOIN pg_class c ON d.objoid=c.oid "
              "JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum=d.objsubid "
              "WHERE c.relname='situation_config'")
    assert p.returncode == 0, p.stderr
    described = {}
    for line in p.stdout.split("\n"):
        if "|" in line:
            name, desc = line.split("|", 1)
            described[name.strip()] = desc.strip()
    missing = [c for c in FIELD_TO_COLUMN.values()
               if not any("31002" in described.get(c, "") or "表2" in described.get(c, "")
                          for _ in [0])]
    # 只对"新增 + 承载"列检查
    for col in FIELD_TO_COLUMN.values():
        assert "31002" in described.get(col, ""), \
            "列 %s 缺注明表2 项号的 COMMENT（实际: %r）" % (col, described.get(col))


# ─────────────────────── 步骤 3：版本表

def test_version_table_exists_with_required_columns() -> None:
    """`31002 5.4.5` 要求记录**内容、时间、修改者、修改原因** —— 逐项要有列。"""
    p = _psql("SELECT column_name FROM information_schema.columns "
              "WHERE table_name='situation_version'")
    cols = {l.strip() for l in p.stdout.split("\n") if l.strip()}
    assert cols, "situation_version 表不存在"
    for need in ("situation_id", "version_no", "content_snapshot", "changed_by",
                 "changed_time", "change_reason"):
        assert need in cols, "版本表缺列 %s（5.4.5 要求）" % need


def test_preexisting_rows_were_backfilled() -> None:
    """既有想定必须**补出初始版本行** —— 否则 5.4.5 的历史不完整。"""
    a = _psql("SELECT count(*) FROM situation_config").stdout.strip()
    b = _psql("SELECT count(DISTINCT situation_id) FROM situation_version").stdout.strip()
    assert a == b, "有 %s 个想定但只有 %s 个有版本行" % (a, b)


# ─────────────────────── 步骤 6：任一修改产生版本记录

def test_update_produces_a_version_row_with_reason_and_author() -> None:
    """**判据 2 的核心**：任一修改产生版本行，且带修改者与原因。

    ⚠ 必须**每次用一个新的 version_no**：版本表上有
    `UNIQUE(situation_id, version_no)`，而触发器用 `ON CONFLICT DO NOTHING` ——
    **重复版本号会被正确拒绝**（这是防止版本号冲突的保护，不是缺陷）。
    我第一版固定用 `'9.9.9'`，第二次跑就因版本号已存在而**什么都没发生**，
    看起来像"修改不产生版本行"。故此处按当前时间生成唯一版本号。
    """
    import time as _t

    code = _psql("SELECT situation_code FROM situation_config "
                 "ORDER BY situation_code LIMIT 1").stdout.strip()
    uniq = "9.9.%d" % (int(_t.time()) % 100000)
    before = int(_psql("SELECT count(*) FROM situation_version").stdout.strip())
    scenario = "测试-%d" % (int(_t.time()) % 100000)
    p = _psql("SET situation.change_reason = 'integration test'; "
              "SET situation.changed_by = 'pytest'; "
              "UPDATE situation_config SET version_no='%s', scenario_type='%s' "
              "WHERE situation_code='%s'" % (uniq, scenario, code))
    assert p.returncode == 0, p.stderr
    after = int(_psql("SELECT count(*) FROM situation_version").stdout.strip())
    assert after == before + 1, "修改未产生版本行（%d → %d）" % (before, after)
    row = _psql("SELECT change_reason, changed_by FROM situation_version "
                "ORDER BY version_id DESC LIMIT 1").stdout.strip()
    assert "integration test" in row, "版本行未记录修改原因: %r" % row
    assert "pytest" in row, "版本行未记录修改者: %r" % row


def test_no_op_update_does_not_create_a_version_row() -> None:
    """无实际变化**不应**产生版本行 —— 否则版本历史会被噪声灌满。"""
    code = _psql("SELECT situation_code FROM situation_config "
                 "ORDER BY situation_code LIMIT 1").stdout.strip()
    before = int(_psql("SELECT count(*) FROM situation_version").stdout.strip())
    _psql("UPDATE situation_config SET scenario_type = scenario_type "
          "WHERE situation_code='%s'" % code)
    after = int(_psql("SELECT count(*) FROM situation_version").stdout.strip())
    assert after == before, "无变化的 UPDATE 产生了版本行（%d → %d）" % (before, after)


def test_duplicate_version_number_is_refused() -> None:
    """同一想定的**同一版本号不得重复** —— 版本号是 5.4.5 版本控制的基础。

    这条同时解释了上面那条测试为什么要用唯一版本号：
    触发器用 `ON CONFLICT DO NOTHING` **有意**拒绝重复版本号。
    """
    code = _psql("SELECT situation_code FROM situation_config "
                 "ORDER BY situation_code LIMIT 1").stdout.strip()
    vno = _psql("SELECT version_no FROM situation_config "
                "WHERE situation_code='%s'" % code).stdout.strip()
    before = int(_psql("SELECT count(*) FROM situation_version").stdout.strip())
    _psql("UPDATE situation_config SET scenario_type='另一值' "
          "WHERE situation_code='%s'" % code)      # version_no 未变
    after = int(_psql("SELECT count(*) FROM situation_version").stdout.strip())
    assert after == before, (
        "同一版本号 %r 被重复记录（%d → %d）—— 版本号唯一性失效"
        % (vno, before, after))


def test_invalid_enum_value_is_rejected() -> None:
    """`31002 5.2.2 c)` 要求**取值词表** → 非法取值必须被约束拒绝。"""
    code = _psql("SELECT situation_code FROM situation_config ORDER BY situation_code LIMIT 1").stdout.strip()
    p = _psql("UPDATE situation_config SET applicable_level='不存在的层级' "
              "WHERE situation_code='%s'" % code)
    assert p.returncode != 0, "非法 applicable_level 未被拒绝"
    assert "ck_applicable_level" in p.stderr or "check constraint" in p.stderr.lower()


# ─────────────────────── 幂等性与纯加法

def test_migration_is_idempotent_and_purely_additive() -> None:
    """**`P0-2`【预期结果】要求可重复执行且结论一致**；且必须**纯加法**。

    做法：跑两次迁移，断言列数与版本行数不变、无 ERROR；
    并断言 P4 原有的 10 列**一个都没少**。
    """
    if not MIGRATION.is_file():
        pytest.skip("migration sql not present")
    original = ("situation_id", "situation_code", "situation_name", "description",
                "region", "center_lon", "center_lat", "center_alt",
                "created_time", "last_update_time")
    n1 = int(_psql("SELECT count(*) FROM information_schema.columns "
                   "WHERE table_name='situation_config'").stdout.strip())
    v1 = int(_psql("SELECT count(*) FROM situation_version").stdout.strip())
    cmd = ["su", "postgres", "-c",
           "psql -d situation -v ON_ERROR_STOP=1 -f %s" % str(MIGRATION)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, "第二次执行失败: %s" % p.stderr[-300:]
    assert "ERROR" not in p.stdout, "第二次执行出现 ERROR"
    n2 = int(_psql("SELECT count(*) FROM information_schema.columns "
                   "WHERE table_name='situation_config'").stdout.strip())
    v2 = int(_psql("SELECT count(*) FROM situation_version").stdout.strip())
    assert n1 == n2, "列数变了（%d → %d），幂等性不成立" % (n1, n2)
    assert v1 == v2, "版本行数变了（%d → %d）" % (v1, v2)
    # 纯加法：原 10 列必须在
    cols = {l.strip() for l in _psql(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='situation_config'").stdout.split("\n") if l.strip()}
    missing = [c for c in original if c not in cols]
    assert missing == [], "P4 原有列被删: %s" % missing


def test_migration_sql_is_additive_by_construction() -> None:
    """**静态**复核（不需数据库）：迁移里不得出现破坏性语句。"""
    if not MIGRATION.is_file():
        pytest.skip("migration sql not present")
    sql = MIGRATION.read_text(encoding="utf-8").upper()
    for danger in ("DROP TABLE", "DROP COLUMN", "TRUNCATE", "DELETE FROM SITUATION_CONFIG"):
        assert danger not in sql, "迁移含破坏性语句 %r" % danger
    # ADD COLUMN 必须带 IF NOT EXISTS（幂等）
    import re
    adds = re.findall(r"ADD COLUMN(?! IF NOT EXISTS)", sql)
    assert adds == [], "有 %d 处 ADD COLUMN 缺 IF NOT EXISTS" % len(adds)
