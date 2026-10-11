#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''产出 `P2` 判据 1 所需的**实测报告**：三侧各自的命令 / 通过数 / 失败数 / 耗时。

## 判据原文
> 1. 三侧各有一份实测报告（含命令、通过数、失败数、耗时）
> 2. 报告中每条测试结论可追溯到一次具体执行
> 3. 无法复跑项有书面说明

## 本轮做什么
· **P3 侧**：跑 `pytest tests/ -q`，取实测三元组与耗时
· **P2 侧**：找 `situation3d/tools/*selftest*.py` 并逐个跑
· **P4 侧**：**如实标注"源码不在本检出，无法复跑"**（不伪造）

**输出**：一张含命令与实测数字的表（即判据 1 的对象）。
'''
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]


def run(cmd, timeout=1800):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return r.returncode, ((r.stdout or "") + (r.stderr or "")), time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, "（超时）", time.time() - t0


print("  ══════════ P2 判据 1：三侧实测报告 ══════════")
print()

# ── P3 侧
import re
rc, out, dt = run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"])
m = re.search(r"(\d+) failed,\s*(\d+) passed,\s*(\d+) skipped", out)
print("  【P3 侧】")
print("     命令   : python3 -m pytest tests/ -q")
if m:
    print("     结果   : **%s failed / %s passed / %s skipped**" % m.groups())
else:
    print("     结果   : 未解析出三元组（rc=%d）" % rc)
print("     耗时   : **%.1f s**" % dt)
print("     退出码 : %d" % rc)

# ── P2 侧
print()
print("  【P2 侧】")
tools = sorted((REPO / "situation3d" / "tools").glob("*selftest*.py"))
if not tools:
    print("     **未找到 situation3d/tools/*selftest*.py**")
else:
    for p in tools:
        rc2, out2, dt2 = run([sys.executable, str(p.relative_to(REPO))], timeout=600)
        tail = [l.strip() for l in out2.strip().split("\n") if l.strip()][-1:] or ["（无输出）"]
        print("     %-52s rc=%-3d %6.1fs  %s" % (p.name, rc2, dt2, tail[0][:56]))

# ── P4 侧
print()
print("  【P4 侧】")
p4 = REPO / "code" / "demo20260104"
print("     **无法复跑**：%s **不在本检出**" % p4.relative_to(REPO))
print("     （依 §92：真实阻断 = P4 应用层源码不在本检出）")
