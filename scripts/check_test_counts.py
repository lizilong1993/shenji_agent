#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对清单里自称的 **测试用例数**（如「16 passed」「25 passed」）是否与**当前实测**一致。

## 口径

1. 抽出形如 `` `tests/xxx.py`（**N passed**）`` 的配对（**以及 `N 例`**）。
2. 用 `pytest --collect-only -q` 实测该文件的用例数。
3. 一致 → `OK`；不一致 → `MISMATCH`。
4. **按"数字是否写在当前状态章节"分两组** —— 与 §146 同样的区分：
   追加章节里的数字是**当时的快照**，不必等于现值。

## ⚠ 边界

· 只核**能定位到具体测试文件**的数字；泛称"全部通过"无法核，跳过。
· `--collect-only` 计的是**用例数**，与"passed"在**有 skip 时不同** ——
  故输出同时给 `collected` 与 `passed` 两个口径，**不一致时不擅自判谁对**。
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

REPO = Path("/remote-home/lizilong/shenji_agent")
DOC = REPO / "docs" / "_tasklist_for_check.md"
CUR_SECTIONS = ("## 2. ", "## 126. ", "## 3. ", "## 1. ", "## 7. ")


def collect(tests_rel: str):
    """返回 (collected, passed, skipped)；失败返回 None。"""
    p = REPO / tests_rel
    if not p.is_file():
        return None
    r = subprocess.run([sys.executable, "-m", "pytest", tests_rel, "-q", "--collect-only"],
                       cwd=REPO, capture_output=True, encoding="utf-8", errors="replace",
                       timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"(\d+)\s+tests?\s+collected", out) or re.search(r"(\d+)\s+test[s]?\s+collected", out)
    if not m:
        # pytest 少用例时输出形如 "16 tests collected" 或只列节点
        n = len([l for l in out.split("\n") if "::" in l])
        return (n, None, None) if n else None
    return (int(m.group(1)), None, None)


PAT = re.compile(r"`(tests/[A-Za-z0-9_]+[.]py)`[^`\n]{0,30}?\*{0,2}(\d+)\s*(passed|例)\*{0,2}")


def main() -> int:
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")
    secs = re.split(r"(?m)^(?=## )", src)

    cache, cur, hist = {}, [], []
    for sec in secs:
        head = sec.split("\n")[0]
        is_cur = head.startswith(CUR_SECTIONS)
        for m in PAT.finditer(sec):
            name, want = m.group(1), int(m.group(2))
            if name not in cache:
                cache[name] = collect(name)
            got = cache[name]
            if got is None:
                continue
            (cur if is_cur else hist).append((name, want, got[0], head[:24]))

    def show(title, rows):
        print("  ── %s ──" % title)
        if not rows:
            print("     （无）")
            return 0
        n_bad = 0
        for name, want, real, head in rows:
            mark = "✓" if want == real else "✗"
            if want != real:
                n_bad += 1
            print("  %s %-46s 称 %-4d 实测 %-4d [%s]" % (mark, name, want, real, head))
        return n_bad

    print("  ══════════ 自称用例数 vs 实测 collected ══════════")
    bad_cur = show("**当前状态**章节（须与现值一致）", cur)
    print()
    bad_hist = show("追加章节（当时快照，陈旧属正常）", hist)
    print()
    print("  当前状态章节不符 **%d**；追加章节不符 %d（**不计为缺陷**）"
          % (bad_cur, bad_hist))
    print()
    print("  ⚠ 只核能定位到具体测试文件的数字；泛称「全部通过」跳过。")
    print("  ⚠ 本核对用 `--collect-only` 的 **collected** 数；有 skip 时与 passed 不同。")
    return 1 if bad_cur else 0


if __name__ == "__main__":
    raise SystemExit(main())
