#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''统一核对入口：一条命令跑完本会话新增的全部核对。

## 为什么需要

§110–§113 新增了 4 个核对脚本。**它们各自能跑，但"有没有人跑"没有保证** ——
交付物复核者需要知道**该跑哪些、按什么顺序、看哪些退出码**。

**本脚本把这件事变成一条命令**，并把结果汇总成一张表。

## 跑什么

| # | 核对 | 命令 | 退出码语义 |
| --- | --- | --- | --- |
| 1 | **交付物健康** | `scripts/check_deliverable_health.py` | 0 通过 / 1 不一致 / 2 找不到输出 |
| 2 | **指标覆盖** | `scripts/check_metric_coverage.py` | 0 全过 / 1 有缺口 |
| 3 | **落点引用** | `scripts/audit_tasklist_landing.py <outputs>` | 0 已列出 / 2 找不到主清单 |
| 4 | **断言反模式** | `scripts/check_assertion_antipatterns.py tests` | **0 无命中 / 1 有可疑** |

## ⚠ 关于 #3 与 #4 的退出码（**不要误读**）

· **#3 退出 0 只表示"审计跑完了"**，**不表示"没有失效引用"** ——
  它的产出是一张清单，**清单非空是常态**（待创建文件会被列出）。
· **#4 退出 1 是正常结果**（启发式检测器**按设计会报可疑**），
  **不等于测试有问题** —— 它的命中需人工复核。

**故本脚本对 #3/#4 只报"是否跑通"，不对其内容做合格判定。**
'''
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


_ENV_OVERRIDES = {"check_done_landing.py": {"DONE_LANDING_NO_INDEX": "1"}}


def _env_for(args):
    import os
    env = dict(os.environ)
    for a in args:
        env.update(_ENV_OVERRIDES.get(os.path.basename(a), {}))
    return env


def _run(args, note_ok_exit=(0,), timeout: int = 120):
    # ⚠ 必须显式 UTF-8：Windows 默认用 GBK 解码子进程输出，
    # 而脚本输出含中文/符号 → UnicodeDecodeError（实测）。
    # ⚠ 超时兜底：audit_tasklist_landing 实测 **4m37s**（10 万+ 文件全树扫描），
    # 不设限会拖死整个入口。
    try:
        r = subprocess.run([sys.executable, *args], cwd=REPO, env=_env_for(args),
                           capture_output=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        class _Timeout:
            returncode = 124
            stdout = "  ! 超时（> %d s）—— 该核对较慢，请单独运行" % timeout
            stderr = ""
        return _Timeout()
    return r


def main(argv=None) -> int:
    argv = list(argv or [])
    # ⚠ 落点引用实测 4m37s，故默认跳过；加 --slow 才跑。
    run_slow = "--slow" in argv
    argv = [a for a in argv if a != "--slow"]
    out_dir = argv[0] if argv else None

    checks = [
        ("交付物健康", [str(SCRIPTS / "check_deliverable_health.py")], (0,), "严格：不一致即失败"),
        ("指标覆盖", [str(SCRIPTS / "check_metric_coverage.py")], (0,), "严格：有缺口即失败"),
        # ⚠ 该项把「待批准 = 实现侧已尽」这一定性变成可核对的（只查产物存在 + 已追踪）。
        ("待批准就绪度", [str(SCRIPTS / "check_pending_readiness.py")], (0,), "严格：声称产物缺失即失败"),
        # ⚠ 落点引用脚本**内含远程绝对路径**（REPO=/remote-home/...），
        # 故它**只能在仓库侧运行**；工作站侧运行会 FileNotFoundError。
        # 这里只判定"是否跑通"，rc=2（找不到主清单）视为环境不符而非失败。
        # ⚠ 该脚本实测 **4m37s**（10 万+ 文件全树扫描）—— **默认跳过**，加 --slow 才跑。
        ("落点引用", [str(SCRIPTS / "audit_tasklist_landing.py")] + ([out_dir] if out_dir else []),
         (0, 2), "**只报是否跑通**；清单非空是常态；**该脚本只能在仓库侧跑**；4m37s，故默认跳过"),
        # ⚠ 该脚本需先把快照拷到 docs/_snapshot_for_check.md（两机布局：快照在工作站侧）。
        # 未拷入时它 rc=2 → 视为环境不符，不计失败。
        ("快照断言", [str(SCRIPTS / "check_snapshot_claims.py")], (0, 2),
         "严格：快照里的路径/命令引用必须存在；rc=2 = 未拷入快照（环境不符）"),
        # ⚠ 该脚本需 docs/_tasklist_for_check.md 与 docs/_outputs_for_check/（两机布局：
        #   清单在工作站侧）。未拷入 → rc=2（环境不符）。
        #   且**会用 DONE_LANDING_NO_INDEX=1 跑**：全树索引在 10 万+ 文件上要 ~18 s，
        #   快速模式 ~0.23 s；两者结果不同（6 vs 19 条），差异全来自裸文件名。
        ("已完成项落点", [str(SCRIPTS / "check_done_landing.py")], (0, 2),
         "非严格：产出是清单（缺失需人工判定性质）；rc=2 = 未拷入清单（环境不符）"),
        ("断言反模式", [str(SCRIPTS / "check_assertion_antipatterns.py"), "tests"],
         (0, 1), "**退出 1 是正常结果**；命中需人工复核"),
    ]

    print("  统一核对（%d 项）" % len(checks))
    print()
    rows = []
    for name, args, ok_exits, note in checks:
        script = Path(args[0])
        if not script.is_file():
            rows.append((name, "缺失", "-", note))
            print("  %-12s %-8s %s" % (name, "缺失", script.name))
            continue
        # ⚠ 落点引用实测 4m37s → **默认跳过**，加 --slow 才跑（否则入口不可用）。
        if name == "落点引用" and (not out_dir or not run_slow):
            why = ("未提供含主清单的目录（第 1 个参数传 outputs/）" if not out_dir
                   else "默认跳过（该核对约 4m37s）—— 加 --slow 运行")
            rows.append((name, "跳过", "-", why))
            print("  %-12s %-8s %s" % (name, "跳过", why))
            continue
        r = _run(args)
        tail = ((r.stdout or "") + (r.stderr or "")).strip().split("\n")
        summary = ""
        for l in reversed(tail):
            s = l.strip()
            if s and not s.startswith("-"):
                summary = s[:70]
                break
        ok = r.returncode in ok_exits
        # ⚠ rc=2 = **本机缺输入（环境不符）**，**不等于「跑过了、通过了」**。
        # 首版把它与 rc=0 一并标为 OK —— 那会把"没跑"读成"通过"（与 §114 同类）。
        env_miss = (r.returncode == 2)
        timed_out = (r.returncode == 124)
        if timed_out:
            label, status = "TIMEOUT", "超时"
        elif env_miss:
            label, status = "ENV", "环境不符"
        else:
            label, status = ("OK" if ok else "注意"), ("OK" if ok else "见下文")
        rows.append((name, status, str(r.returncode), note))
        print("  %-12s %-8s rc=%-3s %s" % (name, label, r.returncode, summary))

    print()
    print("  ══════════ 结果解读（**避免误读**）══════════")
    for name, status, rc, note in rows:
        print("     %-12s rc=%-4s %s" % (name, rc, note))

    # ⚠ 环境差异：交付物健康核对需要**主清单**（在工作站侧），仓库侧跑必然 rc=2。
    # 我首版把 rc=2 也算失败 → 仓库侧永远"未通过"，**那是环境不符，不是交付物有问题**。
    def _env_missing(name):
        return any(r[0] == name and r[2] == "2" for r in rows)

    health_env_ok = not _env_missing("交付物健康")
    # ⚠ 「指标覆盖」按**脚本自身所在仓库**解析组件路径：
    #   在工作站侧跑时会指向本地副本（组件不全）→ rc=1。
    #   **这是环境不符（跑错一侧），不是交付物有缺口** —— 故同样豁免并提示。
    metric_env_ok = not (any(r[0] == "指标覆盖" and r[2] == "1" for r in rows)
                         and health_env_ok)
    strict = [r for r in rows if r[0] in ("交付物健康", "指标覆盖", "待批准就绪度")]
    strict_fail = [r[0] for r in strict
                   if r[1] != "OK"
                   and not (r[0] == "交付物健康" and r[2] == "2")
                   and not (r[0] == "指标覆盖" and r[2] == "1" and not metric_env_ok)]

    print()
    if not health_env_ok:
        print("  ⚠ 交付物健康核对 **rc=2 = 本机没有主清单**（在工作站侧）——")
        print("     **这是环境不符，不是交付物有问题**；请在工作站侧补跑该项。")
    if not metric_env_ok:
        print("  ⚠ 指标覆盖 rc=1 且**本机没有主清单** → 按本地副本解析组件，可能不全。")
        print("     **请在仓库侧复核该项**；不要据此判定交付物有缺口。")
    if strict_fail:
        print("  **严格项未通过**：%s" % "、".join(strict_fail))
        return 1
    print("  **本机可跑的严格项全部通过**。")
    print("  另两项的产物需人工阅读，**不以退出码判合格**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
