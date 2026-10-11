#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''核对**已完成项**：「判据里写的数字」有没有在**该项的证据**中出现过。

## 为什么

一项的【验收判据】常写成"每档 ≥100 用例""表2 为 19 项""台账收录 38 行"这类**带数字的要求**。
若**那个数字在整项正文里从未出现**（连"实测 110"这类证据都没有），
**那"已完成"就没有数字层面的支撑** —— 这正是本会话反复记录的问题类型。

## 口径（**保守，只标可疑**）

1. 从【验收判据】抽出**数字**（含 `≥100`、`19 项`、`N≥30` 等）。
2. 在**该项全文**里找同一数字是否出现**第二次**（判据里那次不算）。
3. 出现 → `OK`；**不出现 → `NO_EVIDENCE`**（可疑，需人工看）。
4. **跳过**纯序号/条款号（如 `31007`、`6.1.2`、`2026`、`1)`、`2)`）。

## ⚠ 边界（重要）

**`NO_EVIDENCE` 不等于"没做"** —— 证据可能写在**别处**（台账、其他节、仓库文件）。
**本脚本只标出"该项正文里没写数字证据"的地方，供人工去别处找。**
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

DOC = Path("/remote-home/lizilong/shenji_agent/docs/_tasklist_for_check.md")

# ⚠ 跳过的应是**序号与条款号**，不是"1–2 位数字"。
# 首版把 `^\d{1,2}$` 一律跳过 → 把 **38 行 / 19 项 / 17 项 / 14 条** 这些**真实数量要求**也丢了，
# 于是 21 项里只剩 3 项"可查"。现改为：① 列表序号（行首或 `1)` 形式）；
# ② 标准号/条款号（31002、6.1.2 等）；③ 年份。
SKIP_PATTERNS = (
    r"^3100[0-9]$", r"^6\.\d", r"^5\.\d", r"^8\.\d", r"^9\.\d",
    r"^20\d\d$", r"^Q\d+$",
)
_LIST_ORD = re.compile(r"(?:^|\n)\s*(\d{1,2})[).、]")
_STD_TAIL = re.compile(r"(?:3100\d|§|表B|表)\s*\.?\s*(\d{1,2})")


def _skip(n: str, ctext: str) -> bool:
    if any(re.match(p, n) for p in SKIP_PATTERNS):
        return True
    # 列表序号
    if n in _LIST_ORD.findall(ctext):
        return True
    # 条款号尾巴（如 "6.2.3" 里的 2 与 3）
    if n in _STD_TAIL.findall(ctext):
        return True
    return False


def main() -> int:
    if not DOC.is_file():
        print("  ! 未找到清单副本 %s" % DOC)
        return 2
    src = DOC.read_text(encoding="utf-8")
    i, j = src.find("## 3. 任务明细"), src.find("## 4. 依赖关系")
    blocks = [b for b in re.split(r"(?m)^(?=### )", src[i:j]) if b.startswith("### ")]

    n_items = n_ok = 0
    suspicious = []

    for b in blocks:
        m = re.match(r"### (\S+)\s", b.split("\n")[0])
        st = re.search(r"(?m)^- 【状态】(.+)$", b)
        if not m or not st or not st.group(1).strip().startswith("已完成"):
            continue
        item = m.group(1)
        crit = re.search(r"(?m)^- 【验收判据】(.+?)(?=\n- 【|\Z)", b, re.S)
        if not crit:
            continue
        n_items += 1
        ctext = crit.group(1)
        nums = [x for x in re.findall(r"\d+", ctext) if not _skip(x, ctext)]
        if not nums:
            continue
        # 判据之外的正文
        rest = b.replace(ctext, "", 1)
        miss = []
        for x in sorted(set(nums), key=lambda s: -len(s)):
            # 该数字在**正文别处**是否出现（作为独立数字）
            if not re.search(r"(?<!\d)" + re.escape(x) + r"(?!\d)", rest):
                miss.append(x)
        if miss:
            suspicious.append((item, miss, ctext.strip()[:90]))
        else:
            n_ok += 1

    print("  ══════════ 已完成项：判据数字是否有正文证据 ══════════")
    print("  检查 %d 项：**全部数字有证据 %d 项**；**有数字无证据 %d 项**"
          % (n_items, n_ok, len(suspicious)))
    print()
    for item, miss, ctext in suspicious:
        print("  【%s】缺证据的数字: %s" % (item, ", ".join(miss)))
        print("        判据: %s" % ctext)
    print()
    print("  ⚠ **`缺证据` 不等于「没做」** —— 证据可能写在台账/别处/仓库文件里。")
    print("     本表只标出「该项正文里没有该数字的第二次出现」，供人工去别处找。")
    print()
    print("  ⚠⚠ **本核对的已知假阳性来源（§140 实测：8 处标记全部是假阳性）**：")
    print("     · 百分数（`100%`）—— 数字正则未跨 `%` 匹配")
    print("     · 阈值符号（`P95≤1s`）—— 未识别 `P95`")
    print("     · 千分位（`10,000`）—— 被拆成 `10` 与 `000`")
    print("     · 斜杠枚举（`401/403`）—— 被拆成两个数字")
    print("     · 列表序号与阈值混判（`返回 200`）")
    print("     **故本核对的 `缺证据` 必须人工复核，不得直接当作缺口。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
