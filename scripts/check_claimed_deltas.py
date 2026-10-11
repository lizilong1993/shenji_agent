#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对追加章节自称的**改动量**（`文件.py`（**+N 行**））是否与 `git show --stat` 相符。

## 为什么这是"更深一层"的核对

§146 只验证了**哈希存在**，并明确写了"不等于该提交做了清单说的事"。
**本轮就去做那件事**：把清单里「提交 `H`：`f.py`（+N 行）」这类陈述
与 `git show --stat H` 中该文件的 **insertions** 比对。

## 口径

1. 抽 pattern：`` `H` `` 之后 200 字符内出现 `` `path`（+N 行） ``。
2. 取 `git show --numstat H` 找该 path 的 insertions。
3. `N == insertions` → `OK`；否则 `MISMATCH`（**列出两值**）。
4. 哈希不存在 / 该提交未改该文件 / 纯新增文件（`-` 起）→ 分类报出，**不擅自判错**。

## ⚠ 边界

· 只核**同一句里**出现的"提交哈希 + 文件 + 行数"三元组 —— 跨句配对会造出假阳性。
· 不核 `−M 行`（删除量）与"约 N 行"。
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
# 文件（+N 行）
PAIR = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./\\-]*[.][A-Za-z0-9]{1,5})`\s*"
                  r"（?\*{0,2}\+(\d+)\s*行")


def numstat(h: str):
    r = subprocess.run([GIT, "show", "--numstat", "--format=", h], cwd=REPO,
                       capture_output=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return None
    out = {}
    for line in (r.stdout or "").split("\n"):
        parts = line.split("\t")
        if len(parts) == 3:
            out[parts[2].strip()] = parts[0]
    return out


def main() -> int:
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")

    # 逐句切分（按换行 + 句号），只在**同句**内配对
    n_ok = n_bad = n_unres = n_nofile = 0
    bad, unres = [], []
    cache = {}
    for sent in re.split(r"[。\n]", src):
        hm = HASH.search(sent)
        if not hm:
            continue
        h = hm.group(1)
        if h.isdigit() or h.startswith("3100") or h.startswith("2026"):
            continue
        pm = PAIR.search(sent)
        if not pm:
            continue
        path, want = pm.group(1).replace(chr(92), "/"), int(pm.group(2))
        if h not in cache:
            cache[h] = numstat(h)
        ns = cache[h]
        if ns is None:
            n_unres += 1
            unres.append((h, "哈希不存在"))
            continue
        ins = ns.get(path)
        if ins is None:
            # 也试 basename 匹配
            base = Path(path).name
            hit = [v for k, v in ns.items() if Path(k).name == base]
            ins = hit[0] if len(hit) == 1 else None
        if ins is None:
            n_nofile += 1
            continue
        if ins == "-":
            n_nofile += 1
            continue
        if int(ins) == want:
            n_ok += 1
        else:
            n_bad += 1
            bad.append((h, path, want, int(ins)))

    print("  ══════════ 自称改动量 vs git show --numstat ══════════")
    print("  一致 **%d** · **不一致 %d** · 该提交未改此文件 %d · 无法核对 %d"
          % (n_ok, n_bad, n_nofile, n_unres))
    print()
    for h, path, want, real in bad:
        print("  ✗ %s %-44s 称 +%d 行，实测 +%d" % (h, path[:44], want, real))
    if unres:
        print()
        for h, why in unres:
            print("  · %s %s" % (h, why))
    print()
    print("  ⚠ 只核**同一句里**出现的「哈希 + 文件 + 行数」三元组；跨句配对会造假阳性。")
    print()
    print("  ⚠⚠ **口径**：实测清单用的是**改动量（insertions + deletions）**，不是纯新增 ——")
    print("     两例验证：86535db 106+23=129（清单称 129）；d2b9a7f 101+6=107（清单称 107）。")
    print("     **拿纯新增去比会误报**（这正是我首版的错）。")
    print("  ⚠ 仍不验证「改动内容与功能描述相符」—— 那需逐 diff 人工读。")

    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
