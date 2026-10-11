#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对清单里自称的**提交哈希**是否真实存在于本仓库。

## 为什么

主清单与追加章节里大量出现「提交 `6fcfab9`」「随 `d2b9a7f` 入版本控制」这类陈述。
**若某个哈希不存在，那就是对"我做过并提交了"的虚假陈述** ——
而这类陈述**从未被核过**（§144 的"第N轮"、§146 的"规模数字"都是同类）。

## 口径

1. 抽出形如 `` `7hex` `` 的引用（**恰好 7—12 位十六进制**）。
2. 用 `git cat-file -e <hash>` 判存在性。
3. **按"是否在当前状态章节"分组** —— 与 §146 一致：
   追加章节里的哈希是**当时的快照**，**一旦存在就永远存在**，故陈旧不适用。

## ⚠ 边界

· 只判**哈希是否存在**；**不判它对应的提交内容是否与描述相符**（那需读 diff）。
· 形如 `31007`、`6.2.3` 这类**不是**哈希，已按长度与字符集排除。
'''
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "_tasklist_for_check.md"
GIT = "/remote-home/lizilong/.local/bin/git"
if not Path(GIT).exists():
    GIT = "git"

# ⚠ 只取 **7—10 位**：12 位多为 **Docker 容器 ID**（实测 `465a18f93ac6` 即如此），
# 而本仓短哈希惯例不超过 10 位。全数字的另行排除（年份/条款号）。
PAT = re.compile(r"`([0-9a-f]{7,10})`")

# 上下文含这些词 → 该哈希属于**别的机器/仓库**（P4 本地检出），单列不计缺陷
EXTERNAL_HINTS = ("本地", "demo", "P4 ")


def exists(h: str) -> bool:
    r = subprocess.run([GIT, "cat-file", "-e", h], cwd=REPO,
                       capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")

    cands = {}
    for m in PAT.finditer(src):
        h = m.group(1)
        # 排除纯数字（年份/条款号）
        if h.isdigit():
            continue
        # 排除常见非哈希数字串
        if h.startswith("3100") or h.startswith("2026"):
            continue
        cands[h] = cands.get(h, 0) + 1

    print("  候选哈希 %d 个（去重）" % len(cands))
    miss, external = [], []
    for h in sorted(cands):
        if exists(h):
            continue
        # 看上下文：属于外部仓库的哈希**不计缺陷**
        ctxs = []
        for m in re.finditer(re.escape(h), src):
            ctxs.append(src[max(0, m.start() - 120):m.end() + 60])
        if any(any(x in c for x in EXTERNAL_HINTS) for c in ctxs):
            external.append(h)
        else:
            miss.append(h)

    print()
    if external:
        print("  · **%d 个属外部仓库/机器**（上下文含「本地/demo/P4」），**不计缺陷**：%s"
              % (len(external), ", ".join(external)))
    if miss:
        print("  ✗ **不存在 %d 个**：" % len(miss))
        for h in miss:
            print("     %s（清单里出现 %d 次）" % (h, cands[h]))
    else:
        print("  ✓ **本仓哈希全部存在**（%d 个候选，%d 个属外部）"
              % (len(cands), len(external)))
    print()
    print("  ⚠ 只判哈希**是否存在**；**不判它对应的提交内容是否与描述相符**。")
    return 1 if miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
