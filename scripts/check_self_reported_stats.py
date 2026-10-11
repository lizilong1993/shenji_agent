#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对主清单里**自称的规模数字**是否与真值一致。

## 查什么

主清单与追加章节里大量出现这类陈述：

· 「`docs/decision_register.md`（**7968 字节**，随 `6fcfab9` 入版本控制）」
· 「**159 行 / 58 表格行**」
· 「B 台账（**196 行**）」

**这些是可机械验证的事实陈述**，而**从未被核过**。

## 口径

1. 抽出形如 `路径`（**N 字节**）与（**N 行**）的配对。
2. 对**能解析到的文件**实测其字节数 / 行数。
3. 一致 → `OK`；不一致 → `MISMATCH`（**列出两值**）。
4. 解析不到的路径 → `UNRESOLVED`（不判错）。

## ⚠ 边界

· 只核**带单位的数字**（字节/行/表格行）；**不核**"约 N 项"这类模糊表述。
· **一处陈述可能与另一处指向不同文件** —— 本核对按"最近的路径"配对，**配对失败即跳过**而非猜测。
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


# 形如：`path`（**N 字节**） / `path`（N 行） / `path`（N 行 / M 表格行）
PAT = re.compile(
    r"`([A-Za-z0-9_][A-Za-z0-9_./\\-]*[.][A-Za-z0-9]{1,5})`"
    r"[^`\n]{0,40}?"
    r"(?<![+])\*{0,2}(\d[\d,]*)\s*(字节|行|表格行)\*{0,2}")


def main() -> int:
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")

    n_ok = n_bad = n_unres = 0
    bad = []
    for m in PAT.finditer(src):
        ref, num_s, unit = m.group(1), m.group(2), m.group(3)
        want = int(num_s.replace(",", ""))
        p = resolve(ref)
        if p is None:
            n_unres += 1
            continue
        if unit == "字节":
            actual = p.stat().st_size
        elif unit == "行":
            actual = p.read_text(encoding="utf-8", errors="ignore").count("\n") + 1
        else:  # 表格行
            actual = sum(1 for l in p.read_text(encoding="utf-8",
                                                 errors="ignore").split("\n")
                         if l.startswith("|"))
        # 容器与 .docx 的行数不适用，跳过
        if p.suffix in (".docx", ".png", ".json") and unit != "字节":
            n_unres += 1
            continue
        if actual == want:
            n_ok += 1
        else:
            n_bad += 1
            bad.append((ref, unit, want, actual))

    print("  ══════════ 自称规模数字 vs 实测 ══════════")
    print("  一致 **%d** · **不一致 %d** · 无法解析/不适用 %d" % (n_ok, n_bad, n_unres))
    print()
    for ref, unit, want, actual in bad:
        print("  ✗ %-52s 称 %s %s，实测 %s" % (ref[:52], want, unit, actual))
    print()
    print("  ⚠ 只核**带单位**（字节/行/表格行）的数字；不核「约 N 项」这类模糊表述。")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
