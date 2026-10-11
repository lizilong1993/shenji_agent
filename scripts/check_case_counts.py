#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对**快照 v1.2** 与**主清单 §126 索引**里的「（N 例）」是否与**实测**一致。

真值 = `pytest <file> --collect-only -q` 的 collected 数。

**用法**：把快照与主清单分别放到 `docs/_snap.md`、`docs/_main.md` 后运行。
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
PAT = re.compile(r"`tests/([A-Za-z0-9_]+[.]py)`[^`\n]{0,40}?(\d+)\s*例")


def measured(name: str):
    rel = "tests/" + name
    if not (REPO / rel).is_file():
        return None
    r = subprocess.run([sys.executable, "-m", "pytest", rel, "-q", "--collect-only"],
                       cwd=REPO, capture_output=True, encoding="utf-8",
                       errors="replace", timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"(\d+)\s+tests?\s+collected", out)
    if m:
        return int(m.group(1))
    n = len([l for l in out.split("\n") if "::" in l])
    return n or None


def extract(p: Path, start=None, end=None):
    if not p.is_file():
        return {}
    s = p.read_text(encoding="utf-8")
    if start:
        i = s.find(start)
        if i < 0:
            return {}
        j = s.find(end, i) if end else len(s)
        s = s[i:j]
    return {m.group(1): int(m.group(2)) for m in PAT.finditer(s)}


def main() -> int:
    snap = extract(REPO / "docs" / "_snap.md")
    main_ = extract(REPO / "docs" / "_main.md", "## 126. ", "## 127. ")
    print("  快照 v1.2 引用 %d 个测试文件；§126 索引引用 %d 个"
          % (len(snap), len(main_)))
    print()
    names = sorted(set(snap) | set(main_))
    print("  %-44s %-6s %-6s %-6s" % ("测试文件", "快照", "§126", "实测"))
    print("  " + "-" * 66)
    cache, bad = {}, 0
    for n in names:
        if n not in cache:
            cache[n] = measured(n)
        real = cache[n]
        a, b = snap.get(n), main_.get(n)
        ok = (a == b == real)
        if not ok:
            bad += 1
        print("  %s %-42s %-6s %-6s %-6s" % ("✓" if ok else "✗", n,
                                             a if a is not None else "-",
                                             b if b is not None else "-",
                                             real if real is not None else "?"))
    print()
    print("  不一致 **%d** 项（共 %d）" % (bad, len(names)))
    print()
    print("  ⚠ 三处口径相同（都是「用例数」），故**不一致即至少一处陈旧**。")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
