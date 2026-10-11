#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对【落点】引用：**三项改进**（承接 §133 的教训）

## 相对 §130 的三项改进

| # | §130 的做法 | 本轮 |
| --- | --- | --- |
| 1 | 正则不含**中文括号** → 订正备注里的新路径抓不到 | 已放宽为「反引号间任意短串」 |
| 2 | 只在**仓库侧**找 → `outputs/*.md` 里的真实产物被判缺失 | **也查 `outputs/` 侧** |
| 3 | 只读【落点】字段 | **读整节正文**（§133 证明真实产物常写在【备注】里） |

## 判定分级（**关键：不再一律报"缺失"**）

| 级别 | 条件 | 含义 |
| --- | --- | --- |
| `OK` | 在仓库或 outputs 侧找到 | 引用成立 |
| `NOTE` | 该文件**在本节正文别处出现过**（通常是【备注】里的真实产物名） | **落点过时**，非缺失 |
| `PLAN` | 引用紧跟在「新增」「产出」等词后 | **计划产出**，未创建正常 |
| `EXT` | 路径含 `backend/`/`code/demo` 等**外部依赖**特征 | 指向 P4 源码（**不在本检出**） |
| `MISS` | 以上都不是 | **需人工判定** |
'''
from __future__ import annotations

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
OUTSIDE = REPO / "docs" / "_outputs_for_check"
EXT_MARKERS = ("backend/", "code/demo", "frontend/", "app/main.py", "core/security.py")

_INDEX = None


def _index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    import os
    idx: dict = {}
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


def classify(item: str, ref: str, block: str) -> str:
    p = resolve(ref)
    if p is not None:
        return "OK"
    name = Path(ref.replace(chr(92), "/")).name
    # NOTE：该**文件名**在本节正文别处出现（通常是【备注】里的真实产物）
    if block.count(name) > 1:
        return "NOTE"
    # PLAN：紧跟"新增/产出/计划"
    for kw in ("新增", "产出", "计划", "拟"):
        if re.search(re.escape(kw) + r"[^`\n]{0,12}`" + re.escape(ref) + "`", block):
            return "PLAN"
    if any(m in ref.replace(chr(92), "/") for m in EXT_MARKERS):
        return "EXT"
    return "MISS"


def main() -> int:
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")
    i, j = src.find("## 3. 任务明细"), src.find("## 4. 依赖关系")
    blocks = [b for b in re.split(r"(?m)^(?=### )", src[i:j]) if b.startswith("### ")]

    tally = Counter()
    rows = []
    for b in blocks:
        m = re.match(r"### (\S+)\s", b.split("\n")[0])
        st = re.search(r"(?m)^- 【状态】(.+)$", b)
        if not m or not st:
            continue
        item, status = m.group(1), st.group(1).strip()
        land = re.search(r"(?m)^- 【落点】(.+?)(?=\n- 【|\Z)", b, re.S)
        if not land:
            continue
        text = land.group(1)
        refs = re.findall(r"`([^`]{4,300}?[.](?:py|md|json|js|css|sh|txt|yaml|yml))`", text)
        refs = [r.replace(chr(92), "/") for r in refs if " " not in r]
        for r in sorted(set(refs)):
            c = classify(item, r, b)
            tally[c] += 1
            rows.append((item, status, r, c))

    print("  ══════════ 判定分级汇总 ══════════")
    for k in ("OK", "NOTE", "PLAN", "EXT", "MISS"):
        print("     %-6s %d" % (k, tally.get(k, 0)))

    print()
    print("  ══════════ MISS（**需人工判定**）══════════")
    if not any(r[3] == "MISS" for r in rows):
        print("     无")
    for item, status, ref, c in rows:
        if c == "MISS":
            print("     %-9s %-12s %s" % (item, status, ref))

    print()
    print("  ══════════ NOTE（落点过时：真实产物写在本节别处）══════════")
    for item, status, ref, c in rows:
        if c == "NOTE":
            print("     %-9s %-12s %s" % (item, status, ref))

    print()
    print("  ══════════ EXT（外部依赖：P4 源码，不在本检出）══════════")
    for item, status, ref, c in rows:
        if c == "EXT":
            print("     %-9s %-12s %s" % (item, status, ref))

    print()
    print("  ══════════ 结论 ══════════")
    print("  引用 %d 条：OK %d · NOTE %d · PLAN %d · EXT %d · **MISS %d**"
          % (len(rows), tally.get("OK", 0), tally.get("NOTE", 0),
             tally.get("PLAN", 0), tally.get("EXT", 0), tally.get("MISS", 0)))
    print("  ⚠ **MISS 才需人工判定**；NOTE 是「落点过时」（应订正落点，非缺失）。")
    print("  ⚠ 本核对只判**存在性**，不判内容是否正确。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
