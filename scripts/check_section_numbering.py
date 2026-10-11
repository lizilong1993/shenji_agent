# -*- coding: utf-8 -*-
"""核对追加章节的「第N轮」序号与实际轮次是否一致（并给出订正清单）。

## 问题

§4 起的追加章节标题形如「## 123. 第一百二十五轮追加：…」。
**那些"第N轮"是我写的**，而**实际轮次由会话决定** —— 若两者对不上，
标题就在陈述一个**不成立的事实**（本会话反复记录的问题类型）。

**且本文件是累计交付物**：读者会用"第N轮"去对应会话轮次。
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(r"D:\WHUSAI\Projects\SituationAwareness\outputs\兵棋推演9项团体标准对标任务清单_20261010.md")
lines = P.read_text(encoding="utf-8").split("\n")

CN = "零一二三四五六七八九"
def cn2int(s: str):
    """把「一百二十五」这类中文数字转成整数（仅支持到 999）。"""
    if not s:
        return None
    # 去掉"第"和"轮"
    s = s.strip("第轮 ")
    total, cur = 0, 0
    for ch in s:
        if ch == "百":
            cur = (cur or 1) * 100
            total += cur
            cur = 0
        elif ch == "十":
            cur = (cur or 1) * 10
            total += cur
            cur = 0
        elif ch in CN:
            cur = CN.index(ch)
        else:
            return None
    return total + cur


rows = []
for n, line in enumerate(lines, 1):
    m = re.match(r"^## (\d+)\. (第([零一二三四五六七八九十百]+)轮)追加", line)
    if m:
        rows.append((n, int(m.group(1)), m.group(3), cn2int(m.group(3)), line[:56]))

print("  ══════════ 追加章节：节号 vs 声称的轮次 ══════════")
print("  %-6s %-6s %-14s %-6s %s" % ("行", "节号", "标题里的轮次", "解析", "标题"))
print("  " + "-" * 92)
bad = []
for n, sec, cn, num, title in rows:
    # 节号从 99 起对应第 100 轮附近（第 99 节 = 第九十九轮追加）
    expect = sec
    ok = (num == expect)
    if not ok:
        bad.append((n, sec, cn, num))
    print("  %-6d %-6d %-14s %-6s %s%s" % (n, sec, cn, num if num is not None else "?",
                                            title, "" if ok else "   ← **不符**"))

print()
print("  ══════════ 汇总 ══════════")
print("  追加章节 %d 个；**轮次与节号一致 %d 个**；**不一致 %d 个**"
      % (len(rows), len(rows) - len(bad), len(bad)))
if bad:
    print("  不一致的节号: %s" % [f"§{s}（称第{cn}轮）" for _, s, cn, _ in bad[:20]])
    diffs = {s - num for _, s, _, num in bad if num is not None}
    print("  偏差（节号 − 声称轮次）取值集合: %s" % sorted(diffs))
    print()
    print("  ⚠ 标题里的「第N轮」与节号不符 = **陈述一个不成立的事实**（§135 订正了 97 处）。")
    print("     修法：把标题轮次改为与节号一致（节号是结构标识，不宜动）。")
    raise SystemExit(1)
print()
print("  ⚠ 本核对只判「节号 ↔ 标题轮次」自洽，**不判该节内容是否正确**。")
raise SystemExit(0)
