# -*- coding: utf-8 -*-
"""快照/清单里的**基线数字**自动同步为实测值。

## 发现的缺陷

§127 末行写「全量套件 23 failed / **1115** passed / 11 skipped」（实测值），
而快照 v1.2 写「23 failed / **1105** passed」 —— **两者不一致**。

**根因**：快照的 MD5 与行数是**逐轮自动同步**的，**但基线数字是手写的** ——
于是它**在每次新增测试后变得陈旧**。

**这与本会话反复记录的问题同类**（§110 已提交版本仍在跳过文件、§113 提交计数未核对）：
**一个"看起来是事实"的数字，其实没人核过。**

## 修法

把基线数字也纳入**与 MD5/行数相同的自动同步**：
同步脚本接收**实测三元组**（failed/passed/skipped）并写入快照；
**不猜测、不沿用**旧值。
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if len(sys.argv) < 4:
    print("  用法: sync_baseline.py <failed> <passed> <skipped>")
    raise SystemExit(2)

failed, passed, skipped = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = Path(r"D:\WHUSAI\Projects\SituationAwareness\outputs")
SNAP = OUT / "当前可交付状态快照_20261011.v1.2.md"

pat = re.compile(r"\d+\s*failed\s*/\s*\d+\s*passed\s*/\s*\d+\s*skipped")
new = "%s failed / %s passed / %s skipped" % (failed, passed, skipped)

s = SNAP.read_text(encoding="utf-8")
hits = pat.findall(s)
print("  快照中原有基线数字: %s" % hits)
s2, n = pat.subn(new, s)
SNAP.write_text(s2, encoding="utf-8", newline="\n")
print("  ✓ 快照已同步为 %s（替换 %d 处）" % (new, n))
