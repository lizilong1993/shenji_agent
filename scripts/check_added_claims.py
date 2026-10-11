#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对**"某提交新增了某脚本"**这一可精确判定的陈述。

## 为什么只做这一种

上一版（`check_file_claims.py`）按**小节**配对"哈希 ↔ 文件"，产出**大量配对错误**：
小节里提到的文件**不一定**是被该提交改过的（可能只是"被引用的对象"、"相关文档"、
"核查目标"）。实测三例：
· §129 讲"核对快照引用"，`theme.js` 是**被核对的对象**，本就不该在该提交里；
· §138 讲"文档升到七项"，提交实际只改 `REPRODUCE.md` + 一个新测试 —— **与该节声称完全一致**；
· §146 讲"核对哈希"，`6fcfab9` 是**被引用的历史提交**。

**故"按小节配对"是错误口径，已弃用。**

## 本脚本只做一种精确判定

若某句声称「提交 `H`：**新增** `scripts/X.py`」，则 `H` 的 `--diff-filter=A` 应包含 `X.py`。
**"新增"一词使归属明确**（不像"产出""相关"那样含糊）。
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

HASH = re.compile(r"`([0-9a-f]{7,10})`")
NEWFILE = re.compile(r"新增\s*`([A-Za-z0-9_][A-Za-z0-9_./\\-]*[.][A-Za-z0-9]{1,5})`")


def added_files(h: str):
    r = subprocess.run([GIT, "show", "--name-status", "--format=", "--diff-filter=A", h],
                       cwd=REPO, capture_output=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return None
    out = set()
    for line in (r.stdout or "").split("\n"):
        p = line.split("\t")
        if len(p) >= 2:
            out.add(p[-1].strip())
    return out


def main() -> int:
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")

    cache = {}
    n_pairs = n_ok = 0
    bad = []
    # 只看**同一句**里既有哈希又有「新增 `file`」
    for sent in re.split(r"[。\n]", src):
        hm = HASH.search(sent)
        fm = NEWFILE.search(sent)
        if not hm or not fm:
            continue
        h = hm.group(1)
        if h.isdigit() or h.startswith(("3100", "2026")):
            continue
        f = fm.group(1).replace(chr(92), "/")
        if h not in cache:
            cache[h] = added_files(h)
        adds = cache[h]
        if adds is None:
            continue
        n_pairs += 1
        if f in adds or Path(f).name in {Path(x).name for x in adds}:
            n_ok += 1
        else:
            bad.append((h, f, sorted(adds)[:3]))

    print("  ══════════ 「提交 H … 新增 X」型陈述 ══════════")
    print("  可判定 %d 组；**命中 %d**；**未命中 %d**" % (n_pairs, n_ok, len(bad)))
    print()
    for h, f, adds in bad:
        print("  ✗ %s 称新增 %s；该提交实际新增: %s" % (h, f, adds))
    print()
    print("  ⚠ **口径**：只核「新增」一词使归属明确的情形；")
    print("     「产出/相关/被核对」等含糊表述**不纳入**（那会产出配对错误，见本节起点说明）。")
    print("  ⚠ 只判**文件是否被新增**；不判其内容是否符合描述。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
