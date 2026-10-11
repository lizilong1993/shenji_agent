#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''只核**"当前状态"章节**里的规模数字 —— 历史章节的数字是**当时的快照**，陈旧属正常。

## 区分（关键）

| 章节类型 | 数字的性质 | 是否必须与当前一致 |
| --- | --- | --- |
| **追加章节（§NN 台账）** | **当时的快照** | **不必** —— 文件后来长大了，那是正常历史 |
| **状态/索引章节**（如 §2 指标表、§126 索引） | **对当前状态的陈述** | **必须一致** |

首版把所有数字一起核 → 27 处"不符"，**其中绝大多数是历史快照**，属噪声。

## 本脚本只核后一类。
'''
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path("/remote-home/lizilong/shenji_agent")
DOC = REPO / "docs" / "_tasklist_for_check.md"
OUTSIDE = REPO / "docs" / "_outputs_for_check"

# 只扫这些「当前状态」章节（按标题前缀匹配）
CUR_SECTIONS = (
    "## 2. ", "## 126. ", "## 1. ", "## 3. ", "## 5. ", "## 7. ",
)
_INDEX = None


def _index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    idx = {}
    skip = {".git", "__pycache__", "node_modules", ".venv"}
    for base in (REPO, OUTSIDE):
        if not base.is_dir():
            continue
        for root, dirs, files in os.walk(str(base)):
            dirs[:] = [d for d in dirs if d not in skip]
            for fn in files:
                idx.setdefault(fn, Path(root) / fn)
    _INDEX = idx
    return idx


def resolve(ref: str):
    rr = ref.replace(chr(92), "/")
    for base in (REPO, OUTSIDE):
        p = base / rr
        if p.is_file():
            return p
    return _index().get(Path(rr).name)


PAT = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./\\-]*[.][A-Za-z0-9]{1,5})`"
                 r"[^`\n]{0,40}?(?<![+])\*{0,2}(\d[\d,]*)\s*(字节|行|表格行)\*{0,2}")


def main() -> int:
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")
    secs = re.split(r"(?m)^(?=## )", src)
    n_ok = n_bad = n_unres = 0
    bad = []
    scanned = 0
    for sec in secs:
        head = sec.split("\n")[0]
        if not head.startswith(CUR_SECTIONS):
            continue
        scanned += 1
        for m in PAT.finditer(sec):
            ref, want, unit = m.group(1), int(m.group(2).replace(",", "")), m.group(3)
            p = resolve(ref)
            if p is None or (p.suffix != "" and unit != "字节" and p.suffix in (".docx", ".png")):
                n_unres += 1
                continue
            actual = (p.stat().st_size if unit == "字节"
                      else (sum(1 for l in p.read_text(encoding="utf-8", errors="ignore").split("\n")
                                if l.startswith("|")) if unit == "表格行"
                            else p.read_text(encoding="utf-8", errors="ignore").count("\n") + 1))
            if actual == want:
                n_ok += 1
            else:
                n_bad += 1
                bad.append((head[:30], ref, unit, want, actual))

    print("  ══════════ 「当前状态」章节里的规模数字 ══════════")
    print("  扫过 %d 个当前状态章节；一致 **%d** · **不一致 %d** · 无法解析 %d"
          % (scanned, n_ok, n_bad, n_unres))
    print()
    for head, ref, unit, want, actual in bad:
        print("  ✗ [%s] %-42s 称 %s %s，实测 %s" % (head, ref[:42], want, unit, actual))
    print()
    print("  ⚠ **追加章节的数字不在此列** —— 那是当时的快照，陈旧属正常（§146 已说明）。")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
