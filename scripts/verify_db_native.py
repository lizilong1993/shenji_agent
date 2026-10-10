#!/usr/bin/env python3
"""P4 数据库验证 —— **原生 PostgreSQL 版**（替代 `backend/verify-db.sh`）。

## 为什么需要它

`backend/verify-db.sh` 的 6 项检查全部写成
`docker exec situational-awareness-postgis psql -U postgres -d situation -c "<SQL>"`。
P0-2 要求用 **Docker Compose** 跑四服务，而远程 situ155 **本身就是一个容器**，
**没有任何容器运行时**（docker/podman/containerd/runc 实测全无，无 `docker.sock`）——
在容器里再起 dockerd 需要特权，不是可靠路径。

但实测发现：**这些检查里只有 `docker exec` 那层是 Docker 特有的，SQL 本身完全是标准 PostgreSQL。**

## 本脚本怎么保证与原脚本一致

**SQL 从原脚本里机械抽取**（正则匹配 `docker exec … psql … -c "<SQL>"`），
**不是重写的**。故原脚本改一处 SQL，这里重新抽取即同步；
本文件只替换"在哪里执行"这一层。

用法（远程）：
    su postgres -c "python3 scripts/verify_db_native.py --original /tmp/p4_verify.sh"

退出码：`0` 全部检查通过且数据非空；`1` 有检查失败或数据为空；`2` 用法/环境错误。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

DB = "situation"
PG_USER = "postgres"


def extract_sql(original: Path) -> List[Tuple[str, str]]:
    """从原 `verify-db.sh` 抽取 (标签, SQL)。标签用 SQL 的首个 SELECT 目标做提示。"""
    text = original.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(
        r'docker exec\s+\S+\s+psql\s+-U\s+postgres\s+-d\s+situation\s+-c\s+\\?\n?"(.*?)"',
        re.S)
    out = []
    for i, m in enumerate(pat.findall(text), 1):
        sql = m.strip()
        out.append(("check%02d" % i, sql))
    return out


def _already_pg_user() -> bool:
    """当前是否已经是 `postgres` 用户。

    ⚠ 我第一版无条件用 `su postgres -c …`，而脚本本身就是用
    `su postgres -c "python3 …"` 启动的 —— **嵌套 su 会要求密码**，
    于是 6 项检查全部以 `su: Authentication failure` 失败，
    **看起来像"数据库有问题"，实际是调用方式错了**。
    """
    try:
        import pwd
        return pwd.getpwuid(os.geteuid()).pw_name == PG_USER
    except Exception:  # noqa: BLE001
        return False


def run_sql(sql: str) -> Tuple[int, str, str]:
    """执行 SQL（peer 认证，与 `000-init.sh` 同一路径）。

    已是 postgres 时**直接调用 psql**；否则经 `su` 切换。两种情形都走 peer 认证，
    与 `000-init.sh` 里的 `psql --username postgres` 等价。
    """
    psql = ["psql", "-d", DB, "-v", "ON_ERROR_STOP=1", "-tAc", sql]
    if _already_pg_user():
        cmd = psql
    else:
        cmd = ["su", PG_USER, "-c", " ".join(_shell_quote(a) for a in psql)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Native-Postgres P4 DB verification.")
    ap.add_argument("--original", default="/tmp/p4_verify.sh",
                    help="path to the original verify-db.sh (SQL is extracted from it)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    orig = Path(args.original)
    if not orig.is_file():
        print("original script not found: %s" % orig, file=sys.stderr)
        return 2

    checks = extract_sql(orig)
    if not checks:
        print("no SQL extracted from %s" % orig, file=sys.stderr)
        return 2

    results = []
    failed = 0
    print("native Postgres verification of P4 schema")
    print("  original: %s" % orig)
    print("  extracted %d checks (SQL reused verbatim)" % len(checks))
    print()
    for label, sql in checks:
        rc, out, err = run_sql(sql)
        ok = rc == 0
        if not ok:
            failed += 1
        rows = [l for l in out.split("\n") if l.strip()]
        results.append({"check": label, "ok": ok, "rows": len(rows),
                        "error": err[:200] if err else None})
        head = " ".join(sql.split())[:58]
        print("  %-4s %-58s rows=%d" % ("ok" if ok else "FAIL", head, len(rows)))
        if not ok:
            print("       %s" % err[:160])

    # 附加断言：核心表必须非空（原脚本只打印，这里把它变成判据）
    rc, out, err = run_sql(
        "SELECT (SELECT count(*) FROM situation_config), (SELECT count(*) FROM target)")
    counts = out.split("|") if rc == 0 else []
    non_empty = False
    if len(counts) == 2:
        non_empty = int(counts[0]) > 0 and int(counts[1]) > 0
    print()
    print("  core tables non-empty: %s (situation_config=%s, target=%s)"
          % (non_empty, counts[0] if counts else "?", counts[1] if counts else "?"))

    ok_all = failed == 0 and non_empty
    print()
    print("RESULT: %s (failed=%d)" % ("OK" if ok_all else "FAIL", failed))
    if args.json:
        import json
        print(json.dumps({"ok": ok_all, "failed": failed, "non_empty": non_empty,
                          "checks": results}, ensure_ascii=False, indent=2))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
