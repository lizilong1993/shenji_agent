#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''`C6` 判据 5：**提交前扫描 `SECRET_KEY`、天地图 key、Cesium ion token，确认无密钥入库**。

## 为什么这一条不需要 P4 源码
它是**对本仓库的扫描**（看是否有密钥被提交），与 P4 源码是否存在无关。
**故它是 13 个阻塞项中第一个可**直接判定**的判据。**

## 扫描对象
1. **git 已追踪文件**（即「入库」的内容）—— 用 `git ls-files`，不用工作区扫描
2. 关键模式：`SECRET_KEY`、天地图 key（`tk=`/`tianditu`）、Cesium ion token
3. **排除**占位符/示例值（如 `changeme`、`your-`、`xxx`、`<...>`）

## ⚠ 边界
· 只判**模式是否出现在已追踪文件里**；**不判该值是否真的有效**。
· 命中的可能是**误报**（如文档里讨论"如何生成 SECRET_KEY"）—— 故输出**分级**。
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
GIT = "/remote-home/lizilong/.local/bin/git"
if not Path(GIT).exists():
    GIT = "git"

# (标签, 正则, 是否"看起来像真值")
PATTERNS = [
    ("SECRET_KEY", re.compile(r"SECRET_KEY\s*[:=]\s*['\"]?([A-Za-z0-9_\-+/=]{12,})")),
    ("天地图 key", re.compile(r"(?:tianditu|天地图)[^\n]{0,40}?tk\s*=\s*([A-Za-z0-9]{16,})")),
    ("天地图 tk 参数", re.compile(r"\btk=([A-Za-z0-9]{20,})")),
    ("Cesium ion", re.compile(r"(?:cesium|ion)[^\n]{0,30}?(?:token|accessToken)\s*[:=]\s*['\"]?([A-Za-z0-9_\-.]{20,})",
                              re.I)),
    ("Cesium token", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
]

# 占位符特征（命中不算真值）
PLACEHOLDER = ("changeme", "your-", "your_", "xxx", "placeholder", "<", ">", "example",
               "todo", "test-", "dummy", "fake", "sample", "直接替换", "自行生成")


def main() -> int:
    r = subprocess.run([GIT, "ls-files"], cwd=REPO, capture_output=True,
                       encoding="utf-8", errors="replace")
    tracked = [f for f in (r.stdout or "").split("\n") if f.strip()]
    print("  ══════════ 扫描对象：**已追踪文件 %d 个**（即「入库」的内容）══════════" % len(tracked))
    print()

    real, suspect = [], []
    for f in tracked:
        p = REPO / f
        if not p.is_file() or p.stat().st_size > 2_000_000:
            continue
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip",
                                ".pt", ".pth", ".bin", ".tif", ".tiff", ".gz"):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for label, pat in PATTERNS:
            for m in pat.finditer(txt):
                val = m.group(1) if m.groups() else m.group(0)
                low = val.lower()
                tail = txt[max(0, m.start() - 60):m.end() + 60].lower()
                if any(ph in low for ph in PLACEHOLDER) or any(ph in tail for ph in PLACEHOLDER):
                    suspect.append((f, label, val[:24]))
                else:
                    real.append((f, label, val[:24]))

    print("  ══════════ 结果 ══════════")
    print("  **疑真值命中 %d 处**；占位符/示例命中 %d 处" % (len(real), len(suspect)))
    print()
    if real:
        print("  ✗ **需处置**：")
        for f, label, v in real[:20]:
            print("     %-46s %-14s %s…" % (f[:46], label, v))
    else:
        print("  ✓ **未在已追踪文件中发现真值形态的密钥**")
    if suspect:
        print()
        print("  · 占位符/示例（**不算入库密钥**）：%d 处；示例：" % len(suspect))
        for f, label, v in suspect[:5]:
            print("     %-46s %-14s %s…" % (f[:46], label, v))
    print()
    print("  ⚠ 只判**模式是否出现在已追踪文件里**；不判该值是否有效。")
    print("  ⚠ 命中的可能是误报（如文档讨论「如何生成 SECRET_KEY」）—— 故分级输出。")
    return 1 if real else 0


if __name__ == "__main__":
    raise SystemExit(main())
