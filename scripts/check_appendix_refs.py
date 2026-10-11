#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对**追加章节**（§1 往后的台账/说明节）里提到的路径是否都真实存在。

## 为什么

§129 只核对了**快照 v1.2** 里的引用；§134 只核对了**各任务项的【落点】**。
**追加章节（§99–§141 等）从未被核过** —— 而那些章节里大量出现
`scripts/xxx.py`、`training/xxx.py`、`tests/xxx.py` 这类"我说我做过了"的引用。

**若其中某个脚本名写错或从未存在，那就是新增的失效引用。**

## 口径

1. 只取**形如路径的反引号引用**（含 `/` 且以已知扩展名结尾）。
2. 逐个在仓库侧解析（先相对路径，再全树 basename）。
3. **按章节统计**，并只列**未解析**的。

## ⚠ 边界

**"未解析"不等于"写错"** —— 可能指向外部系统（P4 源码、暂存区、待创建文件）。
**故输出里按"是否含外部特征"分两栏，且不判合格。**
'''
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path("/remote-home/lizilong/shenji_agent")
DOC = REPO / "docs" / "_tasklist_for_check.md"
EXT = ("py", "md", "json", "js", "css", "sh", "txt", "yaml", "yml")
EXT_MARKERS = ("backend/", "code/demo", "frontend/", ".scratch/", "三维态势系统/")
_INDEX = None


def _index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    idx = {}
    skip = {".git", "__pycache__", "node_modules", ".venv"}
    for root, dirs, files in os.walk(str(REPO)):
        dirs[:] = [d for d in dirs if d not in skip]
        for fn in files:
            idx.setdefault(fn, Path(root) / fn)
    _INDEX = idx
    return idx


def resolve(ref: str):
    rr = ref.replace(chr(92), "/")
    p = REPO / rr
    if p.is_file():
        return p
    return _index().get(Path(rr).name)


def main() -> int:
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")
    # 只看「任务明细」之后的追加章节
    tail = src[src.find("## 4. 依赖关系"):]
    secs = re.split(r"(?m)^(?=## )", tail)

    pat = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./\\-]*[.](?:%s))`" % "|".join(EXT))
    n_total = n_ok = 0
    unresolved = []
    seen = set()
    for sec in secs:
        head = sec.split("\n")[0][:40]
        refs = {r.replace(chr(92), "/") for r in pat.findall(sec)}
        for r in refs:
            if r in seen:
                continue
            seen.add(r)
            n_total += 1
            if resolve(r):
                n_ok += 1
            else:
                unresolved.append((head, r))

    print("  ══════════ 追加章节的路径引用 ══════════")
    print("  去重引用 %d 条；**可解析 %d 条**；**未解析 %d 条**" % (n_total, n_ok, len(unresolved)))
    print()
    inside = [(h, r) for h, r in unresolved if not any(m in r for m in EXT_MARKERS)]
    outside = [(h, r) for h, r in unresolved if any(m in r for m in EXT_MARKERS)]
    print("  ── A. 未解析且**不含外部特征**（最可能是写错/不存在）%d 条 ──" % len(inside))
    for h, r in inside:
        print("     %-34s %s" % (h, r))
    print()
    print("  ── B. 未解析且**含外部特征**（P4 源码 / 暂存区 / 待创建）%d 条 ──" % len(outside))
    for h, r in outside[:25]:
        print("     %-34s %s" % (h, r))
    if len(outside) > 25:
        print("     …（其余 %d 条略）" % (len(outside) - 25))
    print()
    print("  ⚠ 「未解析」不等于「写错」；**A 栏最需人工看**。本核对不判合格。")
    c = Counter(h for h, _ in unresolved)
    print("  未解析条数最多的章节：%s" % c.most_common(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
