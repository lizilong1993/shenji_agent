#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''判明"我在哪一侧"，并对**残缺环境**给出显式警告。

## 动机（§135）

本地镜像 `code/shenji_agent/` 只有 **2,104** 个文件，而远程仓库有 **102,557**。
**我此前多轮用镜像判仓库内容 → 结论系统性偏向"找不到"。**

**本脚本把那次的发现变成每次运行都会打印的警告** —— 不依赖我记住。

## 判据（全部可机械判定）

| 信号 | 含义 |
| --- | --- |
| 'outputs/' 与主清单是否存在 | **交付物侧**（工作站） |
| 文件总数（`git ls-files` 或 walk） | **完整仓库** 约 10 万；**残缺镜像** 约 2 千 |
| `.git` 是否为真仓库 | 镜像可能没有版本库 |

## ⚠ 边界

**本脚本只报"环境性质"，不判任何交付物对错。** 退出码恒为 0。
'''
from __future__ import annotations

import os
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
# 阈值取两处实测值的几何中点附近：2,104 与 102,557 → 10 万为界
FULL_REPO_MIN_FILES = 20000


def _count_files(root: Path) -> int:
    n = 0
    skip = {".git", "__pycache__", "node_modules", ".venv"}
    for r, dirs, files in os.walk(str(root)):
        dirs[:] = [d for d in dirs if d not in skip]
        n += len(files)
        if n > 200000:
            break
    return n


def _git_tracked(root: Path) -> int:
    import subprocess
    try:
        r = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                           encoding="utf-8", errors="replace", timeout=60)
        if r.returncode == 0:
            return len([l for l in (r.stdout or "").split("\n") if l.strip()])
    except Exception:
        pass
    return -1


def main() -> int:
    n_files = _count_files(REPO)
    n_tracked = _git_tracked(REPO)
    has_git = (REPO / ".git").exists()

    # 交付物侧信号
    out_dirs = [p for p in (REPO / "outputs", REPO.parent / "outputs") if p.is_dir()]
    has_tasklist = any(any(d.glob("*任务清单*.md")) for d in out_dirs)

    print("  ══════════ 环境判明 ══════════")
    print("     根目录        : %s" % REPO)
    print("     文件数（walk）: %d" % n_files)
    print("     git 追踪文件数: %s" % (n_tracked if n_tracked >= 0 else "（非 git 仓库）"))
    print("     有 .git       : %s" % has_git)
    print("     outputs/ + 主清单: %s" % has_tasklist)

    print()
    print("  ══════════ 判定 ══════════")
    full_repo = (n_files >= FULL_REPO_MIN_FILES) or (n_tracked >= FULL_REPO_MIN_FILES)

    if full_repo and not has_tasklist:
        side = "仓库侧（完整）"
        print("     这一侧 = **%s**" % side)
        print("     · 可跑：指标覆盖 / 待批准就绪度 / 快照断言 / 已完成项落点 / 落点分级 / 断言反模式")
        print("     · **不可跑**：交付物健康（**需就地在工作站侧**，本侧无主清单）")
    elif has_tasklist and not full_repo:
        side = "交付物侧（残缺镜像）"
        print("     这一侧 = **%s**" % side)
        print("     · 可跑：交付物健康")
        print("     · **不可用于判仓库内容**：本侧只有 %d 个文件，而完整仓库有 **102,557**" % n_files)
        print("     · **§135 的教训：用本侧判仓库内容，结论会系统性偏向「找不到」。**")
    elif has_tasklist and full_repo:
        side = "两侧皆备"
        print("     这一侧 = **%s**（罕见，可跑全部）" % side)
    else:
        side = "**两者皆非**（既非完整仓库，也无主清单）"
        print("     这一侧 = %s" % side)
        print("     · **任何「找不到」的结论都不可信** —— 先确认自己在正确的机器上。")

    print()
    print("  ⚠ 本脚本**只报环境性质**，不判任何交付物对错；退出码恒为 0。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
