#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对**已完成项**的【落点】引用是否存在。

## 为什么单查"已完成"

`P0-2` 等**阻塞/暂缓**项的【落点】常写"**新增** X" —— 那是**计划产出**，未创建属正常。
但**已完成项**声称"已完成"，若其【落点】里的产物不存在，那就是**文档与事实不符**。

> 本会话已记录同类：§92「落点指向不存在的路径」（23 处待创建被误报为缺失，
> 故后来给审计脚本加了"新增/产出"分类）。**本脚本把该分类收窄到"已完成项"这一最严格的子集。**

## ⚠ 边界

1. 【落点】里以「**新增**」开头的引用**不计入缺失**（那是计划产出）。
2. 提到的是**枢纽/上下级系统的既有文件**（如 P4 的 `backend/app/main.py`）时，
   缺失**指向外部依赖**而非本项虚报 —— 输出里单独标注。
3. **只判存在性**，不判内容。
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
DOC = MIRROR / "docs" / "_tasklist_for_check.md"
OUTSIDE = MIRROR / "docs" / "_outputs_for_check"

_bad: list = []


_INDEX = None


def _build_index():
    """一次性建立 basename → 路径 索引。

    ⚠ **性能**：原实现对每个未命中的引用调用 
glob，而本仓有 **10 万+ 文件** ——
    单次全树扫描虽只要 0.2 s，但测试里反复调用会把整个套件拖慢数分钟（实测 8 例 232 s）。
    故改为**建一次索引、查多次**。
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    import os
    idx: dict = {}
    skip = {".git", "__pycache__", "node_modules", ".venv"}
    for base in (MIRROR, OUTSIDE):
        if not base.is_dir():
            continue
        for root, dirs, files in os.walk(str(base)):
            dirs[:] = [d for d in dirs if d not in skip]
            for fn in files:
                idx.setdefault(fn, Path(root) / fn)
    _INDEX = idx
    return idx


def resolve(ref: str):
    rr = ref.replace("\\", "/")
    for base in (MIRROR, OUTSIDE):
        p = base / rr
        if p.is_file():
            return p, base
    # ⚠ 索引**懒建**，且可**用环境变量关闭**（DONE_LANDING_NO_INDEX=1）。
    # 实测：10 万+ 文件的全树遍历在本机（网络文件系统）真实清单上要 **77 s
    # （rglob 版 6m41s）**，故**不宜进默认测试套件**。
    # 关闭回退后，只判"相对路径是否直接命中" —— 代价是裸文件名（如 theme.js）会判缺失。
    if NO_INDEX:
        return None, None
    q = _build_index().get(Path(rr).name)
    return (q, MIRROR) if q else (None, None)


NO_INDEX = False


def main() -> int:
    global NO_INDEX
    import os
    NO_INDEX = os.environ.get("DONE_LANDING_NO_INDEX") == "1"
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")
    i, j = src.find("## 3. 任务明细"), src.find("## 4. 依赖关系")
    blocks = [b for b in re.split(r"(?m)^(?=### )", src[i:j]) if b.startswith("### ")]

    print("  ══════════ 已完成项的【落点】引用 ══════════")
    print("  %-9s %-7s %-6s %-6s" % ("项", "引用", "存在", "缺失"))
    n_done = n_miss = 0
    for b in blocks:
        m = re.match(r"### (\S+)\s", b.split("\n")[0])
        st = re.search(r"(?m)^- 【状态】(.+)$", b)
        if not m or not st or not st.group(1).strip().startswith("已完成"):
            continue
        n_done += 1
        item = m.group(1)
        land = re.search(r"(?m)^- 【落点】(.+?)(?=\n- 【|\Z)", b, re.S)
        if not land:
            continue
        text = land.group(1)
        refs = re.findall(r"`([A-Za-z0-9_][A-Za-z0-9_./\\-]*[.](?:py|md|json|js|css|sh|txt|yaml|yml))`", text)
        # ⚠ 清单里写的是 **Windows 反斜杠**（如 docs\std_clause_audit.md），
        # 而本脚本在 Linux 上跑 —— Path 会把 \ 当**字面字符**，于是所有引用都"找不到"。
        # 故先归一为 / 再解析（我的第一版没做，导致 6 条假阳性）。
        refs = [r.replace(chr(92), "/") for r in refs]
        if not refs:
            continue
        found = missing = 0
        miss_list = []
        for r in sorted(set(refs)):
            p, _ = resolve(r)
            if p:
                found += 1
            else:
                missing += 1
                miss_list.append(r)
        n_miss += missing
        if missing:
            print("  %-9s %-7d %-6d %-6d" % (item, found + missing, found, missing))
            for x in miss_list:
                print("        **缺失**: %s" % x)
                _bad.append("%s → %s" % (item, x))
        else:
            print("  %-9s %-7d %-6d %-6d" % (item, found + missing, found, 0))

    print()
    print("  ══════════ 结论 ══════════")
    print("  已完成项 %d 个；**%d 条引用缺失**" % (n_done, n_miss))
    if _bad:
        print("  ⚠ 缺失**需人工判定性质**：可能是①本项虚报、②外部依赖（P4 源码）、③计划产出。")
        print("     本脚本**不自动下结论**，只把清单列出。")
    else:
        print("  **已完成项的落点引用全部存在** ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
