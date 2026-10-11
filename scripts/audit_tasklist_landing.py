#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''**落点引用审计**：核对 51 项【落点】里的文件路径是否真实存在。

## 为什么需要（§92 的推广）

§92 我**逐个发现**了 13 项阻塞-P0-2 的共同落点是 `code/demo20260104/backend/...`（**不存在**）。
那是**个案**。本脚本把它变成**完整清单**。

## ★ 关键在于**分类**，否则会把"待办"误报成"缺陷"

落点里混着**两类完全不同**的东西：

| 类 | 例子 | 性质 |
| --- | --- | --- |
| **A. 标注「新增/产出」** | 「新增 `docs\\competition_platform_assessment.md`」 | **尚未创建，属正常** |
| **B. 引用现有物** | 「P4 `code\\demo20260104\\backend\\app\\main.py`」 | **该代码库不在检出 → 真问题** |

**不分类就会把 A 报成 B** —— 而本会话第一阶段就犯过"把不可完成项降级/把待办当缺陷"的错。

## 另需排除**工作站侧产出**

`outputs/`、`scratch/`、`进度汇报/` 下的文件**在仓库里本就不存在**（它们在另一台机器上），
**不构成"仓库内引用失效"**。

## 边界

落点文本含非路径词，抽取会产生**碎片**（如 `2测试说明.docx`）→ 需人工过一遍。
**本脚本只做保守判定**：仅当**文件名在仓库任意位置都找不到**时才报。
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
GIT = "/remote-home/lizilong/.local/bin/git"
MAIN_NAME = "兵棋推演9项团体标准对标任务清单_20261010.md"

# 工作站侧的产出目录 —— 在仓库里本就不存在，排除
EXCLUDED_PREFIXES = ("outputs/", "scratch/", ".scratch/", "进度汇报/", "dist/")

PATH_RE = re.compile(
    r"([\w\u4e00-\u9fff\\/\-]{3,140}?\.(?:py|json|md|sh|yaml|yml|sql|txt|jsonl|proto|js|css|html|docx|csv|zip))")


def find_main() -> "Path | None":
    cands = []
    if len(sys.argv) > 1:
        cands.append(Path(sys.argv[1]))
    cands += [Path("/tmp/out101"), Path.cwd(), REPO.parent / "outputs"]
    for c in cands:
        p = (c / MAIN_NAME) if c.is_dir() else c
        if p.is_file():
            return p
    return None


def main() -> int:
    main_md = find_main()
    if main_md is None:
        print("  ! 未找到主清单（传入所在目录作为参数）")
        return 2

    r = subprocess.run([GIT, "ls-files"], cwd=REPO, capture_output=True, text=True)
    tracked = [l.strip() for l in (r.stdout or "").split("\n") if l.strip()]
    by_name = {Path(t).name for t in tracked}
    on_disk = {p.name for p in REPO.rglob("*") if p.is_file()}

    src = main_md.read_text(encoding="utf-8")
    i, j = src.find("## 3. 任务明细"), src.find("## 4. 依赖关系")
    blocks = [b for b in re.split(r"(?m)^(?=### )", src[i:j]) if b.startswith("### ")]

    def norm(p: str) -> str:
        return p.replace("\\", "/").strip("`（）() \t").lstrip("./")

    a_paths, b_paths = set(), {}
    for b in blocks:
        m = re.match(r"### (\S+)\s", b.split("\n")[0])
        lm = re.search(r"(?m)^- 【落点】(.+?)(?=\n- 【|\Z)", b, re.S)
        if not m or not lm:
            continue
        tid, text = m.group(1), lm.group(1)
        for pm in PATH_RE.finditer(text):
            p = norm(pm.group(1))
            if len(p) > 120 or p.startswith(EXCLUDED_PREFIXES) or "/outputs/" in p:
                continue
            base = Path(p).name
            if (REPO / p).is_file() or base in by_name or base in on_disk:
                continue
            pre = text[max(0, pm.start() - 40):pm.start()]
            if "新增" in pre or "产出" in pre:
                a_paths.add((tid, p))
            else:
                b_paths.setdefault(tid, set()).add(p)

    print("  ══════════ A 类：标注「新增/产出」= 未创建，**属正常** ══════════")
    print("     合计 %d 个路径（%d 项）" % (len(a_paths), len({t for t, _ in a_paths})))
    for tid, p in sorted(a_paths)[:8]:
        print("     【%s】%s" % (tid, p))
    if len(a_paths) > 8:
        print("     ... 另有 %d 个" % (len(a_paths) - 8))

    print()
    print("  ══════════ B 类：引用现有物但**不存在** ══════════")
    tot = 0
    for tid in sorted(b_paths):
        ps = sorted(b_paths[tid])
        tot += len(ps)
        print("     【%s】%d 个：%s" % (tid, len(ps), "; ".join(ps[:3])))
    print("     合计 **%d** 个路径，涉及 **%d** 项" % (tot, len(b_paths)))

    print()
    print("  ══════════ 按根因归类 ══════════")
    roots = {"P4 代码库(code/demo*)": 0, "P4 backend/ 相对路径": 0, "其它": 0}
    others = []
    for tid, ps in b_paths.items():
        for p in ps:
            if p.startswith("code/demo"):
                roots["P4 代码库(code/demo*)"] += 1
            elif p.startswith("backend/"):
                roots["P4 backend/ 相对路径"] += 1
            else:
                roots["其它"] += 1
                others.append((tid, p))
    for k, v in roots.items():
        print("     %-28s %d" % (k, v))
    if others:
        print()
        print("     「其它」明细（含**抽取碎片**，需人工过目）:")
        for tid, p in others:
            print("       【%s】%s" % (tid, p))

    print()
    print("  ⚠ 只做保守判定；落点文本含非路径词，会产出碎片，**需人工过一遍**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
