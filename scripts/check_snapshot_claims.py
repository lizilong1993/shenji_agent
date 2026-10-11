#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对交付物里的**可机械判定的断言**是否与事实相符。

## 动机

§128 发现快照的「基线数字」是手写的、无人核 → 陈旧。
**同一份快照里还有别的"看起来是事实"的断言**：文件路径、计数、命令引用。

## ★ 第一版的两个假阳性（已修）

第一版用 `re.findall(r"`([A-Za-z0-9_./]+\\.ext)`")` 抽路径，其中 `[A-Za-z0-9_./]+` 是**贪婪**的，
但因为它要求整段是"路径字符"，形如 `` `theme.js` `` 的引用**只捕到文件名**（`theme.js`），
于是我拿它去仓根找 → **报"不存在"**，而实际在 `situation3d/web/src/theme.js`。

**两处"不符"都是我自己的匹配问题，文件都存在。**

**修法**：对**裸文件名**（不含 `/`）在整棵树里按 basename 搜索，并报出真实路径。

## ⚠ 边界

**只判"存在性"**，不判"内容是否正确"。`OK` = 该文件存在。
'''
from __future__ import annotations

import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MIRROR = Path("/remote-home/lizilong/shenji_agent")
SNAP = MIRROR / "docs" / "_snapshot_for_check.md"   # 由调用方 scp 进来

_ok: list = []
_bad: list = []
_warn: list = []


def chk(cond, label, detail=""):
    (_ok if cond else _bad).append(label)
    print("  %s %-48s %s" % ("✓" if cond else "✗", label, detail[:76]))


def resolve(ref: str) -> "Path | None":
    """把引用解析成真实文件：先按相对路径，再按 basename 全树搜索。"""
    if "/" in ref:
        p = MIRROR / ref
        return p if p.is_file() else None
    hits = [p for p in MIRROR.rglob(ref) if p.is_file()
            and ".git" not in p.parts and "__pycache__" not in p.parts]
    return hits[0] if hits else None


def main() -> int:
    if not SNAP.is_file():
        print("  ! 未找到快照副本 %s" % SNAP)
        return 2
    snap = SNAP.read_text(encoding="utf-8")

    print("  ══════════ ① 快照提到的文件 ══════════")
    refs = sorted(set(re.findall(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*[.](?:py|md|json|js|css|sh|txt|yaml|yml))`",
                                 snap)))
    for r in refs:
        p = resolve(r)
        if p is None:
            chk(False, "存在: %s" % r)
        else:
            rel = p.relative_to(MIRROR).as_posix()
            note = "" if rel == r else "→ %s" % rel
            if rel != r:
                _warn.append("%s → %s（引用不完整）" % (r, rel))
            chk(True, "存在: %s" % r, note)
    print("  ── %d 个路径引用 ──" % len(refs))

    print()
    print("  ══════════ ② 命令与测试引用 ══════════")
    cmds = sorted(set(re.findall(r"python3 (scripts/[A-Za-z0-9_]+[.]py)", snap)))
    for c in cmds:
        chk((MIRROR / c).is_file(), "脚本: %s" % c)
    tests = sorted(set(re.findall(r"(tests/[A-Za-z0-9_]+[.]py)", snap)))
    for t in tests:
        chk((MIRROR / t).is_file(), "测试: %s" % t)
    print("  ── %d 条命令 + %d 个测试引用 ──" % (len(cmds), len(tests)))

    print()
    print("  ══════════ 结论 ══════════")
    print("  通过 %d；不符 %d；提示 %d" % (len(_ok), len(_bad), len(_warn)))
    for b in _bad:
        print("     ✗ %s" % b)
    for w in _warn:
        print("     · %s" % w)
    print()
    print("  ⚠ 本核对只判**存在性**；「引用不完整」是提示，不是错误。")
    return 1 if _bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
