#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''交付物健康核对：状态计数 · 交付物同步 · 引用一致 · 编码卫生 · 主线结构 · 组件在位。

## 为什么需要

主清单已超 10000 行、120+ 次提交。**规模本身成为风险**：
某份交付物未同步、快照引用了旧哈希、某个关键组件其实没提交 —— 都可能无人发现。

**本脚本做机械核对，不产生新内容。**

## 本轮它抓到的两个真问题

1. **`scripts/check_assertion_antipatterns.py` 有未提交改动** ——
   该文件的 fail-loud 修正**被验证过但从未提交**，于是**已提交的版本仍在静默跳过不可解析文件**。
2. **主清单 §8 缺号**（§7 之后直接 §9）—— **原始版本即如此**，非本轮所致。

## ⚠ 边界

只核对**形式一致性**（计数、哈希、存在性），**不判内容正确性**。
'''
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# 必须显式设 UTF-8：Windows 控制台默认 GBK，打印 ✓/✗ 会抛 UnicodeEncodeError。
# 这个坑只在仓库侧 Linux 上不会暴露 —— 又一处「只在一边能跑」。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def _resolve_out() -> "Path | None":
    """定位 `outputs/` 目录 —— **主清单在工作站侧，仓库侧没有**。

    ⚠ 我第一版把它硬编码为工作站路径，于是在**仓库侧运行直接报"主清单不存在"**（rc=2）。
    现按三种来源依次尝试：
      1. 命令行第一个参数；
      2. 环境变量 `TASKLIST_OUT`；
      3. 常见候选路径。
    """
    import os
    cands = []
    if len(sys.argv) > 1:
        cands.append(Path(sys.argv[1]))
    if os.environ.get("TASKLIST_OUT"):
        cands.append(Path(os.environ["TASKLIST_OUT"]))
    cands += [
        Path(r"D:\WHUSAI\Projects\SituationAwareness\outputs"),
        Path.cwd() / "outputs",
        Path.home() / "outputs",
    ]
    for c in cands:
        if (c / "兵棋推演9项团体标准对标任务清单_20261010.md").is_file():
            return c
    return None


OUT = _resolve_out()
if OUT is None:
    print("  ! 未找到主清单所在的 outputs/ —— 请传入路径或设置 TASKLIST_OUT")
    print("    （主清单在工作站侧，仓库侧不含它；⑥ 组件核对需在仓库侧运行）")
    raise SystemExit(2)
MAIN = OUT / "兵棋推演9项团体标准对标任务清单_20261010.md"
V11M = OUT / "兵棋推演9项团体标准对标任务清单_20261010.v1.1.md"
# ⚠ 快照改为**自动取版本号最大者** —— 硬编码 v1.1 会在出 v1.2 后误报（本轮实测）。
def _latest_snapshot(out: Path) -> Path:
    cands = sorted(out.glob("当前可交付状态快照_*.v*.md"))
    def key(p: Path):
        # ⚠ 版本号是**两段**（`.v1.2.md`），我首版写成 `[.]v([0-9]+)[.]md$` ——
        # **少了一段** → 永不匹配 → 所有 key 都为 0 → max 取第一个（结果是旧的 v1.1）。
        # 现按两段匹配，两段都取数以便排序。
        m = re.search(r"[.]v([0-9]+)[.]([0-9]+)[.]md$", p.name)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return max(cands, key=key) if cands else out / "当前可交付状态快照_missing.md"


SNAP = _latest_snapshot(OUT)
REPO = Path("/remote-home/lizilong/shenji_agent")
GIT = "/remote-home/lizilong/.local/bin/git"

# 已知的**原始**编号缺口（非本轮所致），显式登记而非放行
KNOWN_SECTION_GAPS = (8,)

COMPONENTS = (
    "scripts/check_assertion_antipatterns.py",
    "tests/test_assertion_antipattern_detector.py",
    "tests/test_i0809_runtime_js.py",
    "tests/test_i09_symbology_render.py",
    "tests/test_i08_data_health_offline.py",
    "tests/test_i04_feature_coverage_mechanical.py",
    "tests/test_i06_parallel_distribution_gate.py",
    "scripts/setup_js_engine.sh",
)

_fails = []


def chk(cond, label, detail=""):
    if not cond:
        _fails.append(label)
    print("  %s %-46s %s" % ("✓" if cond else "✗", label, detail))


def main() -> int:
    if not MAIN.is_file():
        print("  ! 主清单不存在: %s" % MAIN)
        return 2
    src = MAIN.read_text(encoding="utf-8")

    print("  ══════════ ① 状态计数 ══════════")
    cnt = Counter()
    for l in src.split("\n"):
        if l.startswith("- 【状态】"):
            cnt[l.replace("- 【状态】", "").strip()] += 1
    chk(sum(cnt.values()) == 51, "合计 51 项", "实际 %d" % sum(cnt.values()))
    chk(cnt.get("已完成") == 21 and cnt.get("待批准") == 11 and cnt.get("阻塞-P0-2") == 13,
        "21/11/13 一致", "已完成=%s 待批准=%s 阻塞=%s"
        % (cnt.get("已完成"), cnt.get("待批准"), cnt.get("阻塞-P0-2")))

    print()
    print("  ══════════ ② 交付物同步 ══════════")
    for f in (MAIN, V11M):
        chk(f.is_file(), f.name, "%d B" % f.stat().st_size if f.is_file() else "缺失")
    if MAIN.is_file() and V11M.is_file():
        chk(MAIN.read_bytes() == V11M.read_bytes(), "主清单与 v1.1 逐字节一致",
            "MD5 %s" % hashlib.md5(MAIN.read_bytes()).hexdigest()[:16].upper())

    print()
    print("  ══════════ ③ 快照引用是否指向当前主清单 ══════════")
    if SNAP.is_file():
        s = SNAP.read_text(encoding="utf-8")
        real = hashlib.md5(MAIN.read_bytes()).hexdigest()[:16].upper()
        m = re.search(r"MD5 ([0-9A-F]{16})", s)
        chk(m and m.group(1) == real, "快照内嵌 MD5 == 实际",
            "快照=%s 实际=%s" % (m.group(1) if m else "?", real))
        m2 = re.search(r"\*\*(\d+) 行\*\*", s)
        chk(m2 and int(m2.group(1)) == src.count("\n") + 1, "快照内嵌行数 == 实际",
            "快照=%s 实际=%d" % (m2.group(1) if m2 else "?", src.count("\n") + 1))
    else:
        chk(False, "快照存在", "缺失")

    print()
    print("  ══════════ ④ 编码卫生 ══════════")
    bad = sum(1 for ch in src if (ord(ch) < 0x20 and ch not in "\n\t") or 0x7f <= ord(ch) <= 0x9f)
    chk(bad == 0, "无控制字符", "实际 %d" % bad)
    chk(src.count("\r\n") == 0, "无 CRLF", "实际 %d" % src.count("\r\n"))
    chk(src.endswith("\n"), "以换行结尾")

    print()
    print("  ══════════ ⑤ 主线结构 ══════════")
    secs = {int(x) for x in re.findall(r"(?m)^## (\d+)\.", src)}
    gaps = sorted(i for i in range(1, max(secs) + 1) if i not in secs)
    unexpected = [g for g in gaps if g not in KNOWN_SECTION_GAPS]
    chk(not unexpected, "§1–§%d 无**非预期**缺号" % max(secs),
        "已知缺号 %s；非预期 %s" % (list(KNOWN_SECTION_GAPS), unexpected or "无"))

    print()
    print("  ══════════ ⑥ 关键组件在位且**已提交** ══════════")
    if not REPO.is_dir():
        print("  ! 仓库不可达（本脚本依赖远程路径）→ 跳过 ⑥")
    else:
        for rel in COMPONENTS:
            p = REPO / rel
            exists = p.is_file()
            r = subprocess.run([GIT, "ls-files", "--error-unmatch", rel],
                               cwd=REPO, capture_output=True, text=True)
            tracked = r.returncode == 0
            chk(exists and tracked, rel,
                ("存在" if exists else "缺失") + "/" + ("已追踪" if tracked else "**未追踪**"))

    print()
    print("  ══════════ 结论 ══════════")
    if _fails:
        print("  **%d 项不一致，需处理**：" % len(_fails))
        for f in _fails:
            print("     - %s" % f)
        return 1
    print("  **全部核对通过**（仅形式一致性；内容正确性不在本脚本范围）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
