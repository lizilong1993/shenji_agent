#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I09 步骤 6：生成**视觉编码对照表**（由源码抽取，避免手工维护的引用漂移）。

## 为什么生成而不是手写

原文：「编写视觉编码对照表，逐项列出**编码维度、取值、对应文件与行号**，并留截图位置。」

**「文件与行号」是这份表最容易腐烂的部分** —— 手写一次，下次改代码就全错，
而错的引用比没有引用更糟（会让人按错行号去找）。

故本脚本**从 `theme.css` / `theme.js` / `index.html` 实际抽取**变量名、取值与行号，
生成 `docs/visual_encoding_table.md`。表里的行号**必然与当前代码一致**。

## 五个维度（对应【预期结果】「阵营、类型、状态、强度、时效五类维度各有唯一配置来源」）

| 维度 | 唯一配置来源 |
| --- | --- |
| 阵营 | `theme.css` 的 `--faction-*` |
| 类型 | `symbology.js` 的形状函数 |
| 状态 | `--status-border-*` |
| 强度 | `--ring-*` / `--strength-*` + `theme.js` 的半径规则 |
| 时效 | `--pulse-*` + `theme.js` 的周期规则 |
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path("/remote-home/lizilong/shenji_agent")
S3 = REPO / "situation3d"
SRC = S3 / "web" / "src"
OUT = REPO / "docs" / "visual_encoding_table.md"

CSS = SRC / "theme.css"
JS = SRC / "theme.js"
HTML = S3 / "web" / "index.html"


def css_vars(path: Path):
    """从 :root 块抽 (变量名, 取值, 行号)。"""
    out = []
    if not path.is_file():
        return out
    lines = path.read_text(encoding="utf-8").split("\n")
    in_root = False
    for i, l in enumerate(lines, 1):
        if re.match(r"\s*:root\s*\{", l):
            in_root = True
            continue
        if in_root and l.strip() == "}":
            in_root = False
            continue
        if in_root:
            m = re.match(r"\s*(--[a-z0-9-]+)\s*:\s*([^;]+);", l)
            if m:
                out.append((m.group(1), m.group(2).strip(), i))
    return out


def js_consts(path: Path):
    """从 theme.js 抽 DEFAULTS/常量表里的 (键, 取值, 行号)。"""
    out = {}
    if not path.is_file():
        return out
    for i, l in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        m = re.match(r"\s*['\"]?([a-z][a-z0-9-]*)['\"]?\s*:\s*['\"]?([^,'\"]+)['\"]?,", l)
        if m:
            out.setdefault(m.group(1), (m.group(2).strip(), i))
    return out


def find_first(path: Path, pattern: str):
    if not path.is_file():
        return None
    for i, l in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if re.search(pattern, l):
            return i, l.strip()
    return None


cv = css_vars(CSS)
jv = js_consts(JS)
print("  theme.css 抽出 %d 个变量；theme.js 抽出 %d 个常量" % (len(cv), len(jv)))

DIMS = [
    ("阵营", "faction", "阵营主色（填充）",
     ["faction-red", "faction-blue", "faction-green", "faction-neutral", "faction-contested"]),
    ("状态", "status-border", "状态边框（样式 + 宽度）",
     ["status-border-normal", "status-border-alert", "status-border-selected", "status-border-width"]),
    ("强度", "ring|strength", "强度内环（半径分级）",
     ["ring-strength-base-radius", "ring-strength-step", "ring-max-rings",
      "strength-0", "strength-1", "strength-2", "strength-3", "strength-4"]),
    ("时效", "pulse", "时效外圈脉冲（周期 + 开关 + 阈值）",
     ["pulse-period-default", "pulse-period-fast", "pulse-speed-threshold-kmh",
      "pulse-enabled"]),
    ("交互", "hover", "悬停/选中反馈",
     ["hover-offset", "hover-transition"]),
]
KNOWN = {k for _, _, _, ks in DIMS for k in ks}


def row(name):
    hit = [(n, v, ln) for n, v, ln in cv if n == "--" + name]
    if hit:
        return hit[0]
    if name in jv:
        v, ln = jv[name]
        return ("(js) " + name, v, ln)
    return (name, "**（未找到）**", None)


L = []
A = L.append
A("# 视觉编码对照表（I09 步骤 6）")
A("")
A("> **本文件由 `scripts/gen_visual_encoding_table.py` 从源码自动生成**，")
A("> 故表里的**文件与行号必然与当前代码一致**。手写这份表会在下次改代码后立刻腐烂，")
A("> 而**错的引用比没有引用更糟**（会让人按错行号去找）。")
A("")
A("生成时间戳由调用方决定；本表只描述**结构**，不含运行输出。")
A("")
A("## 1. 五个编码维度与**唯一配置来源**")
A("")
A("| 维度 | 唯一配置来源 | 说明 |")
A("| --- | --- | --- |")
A("| 阵营 | `web/src/theme.css` 的 `--faction-*` | 阵营主色（填充） |")
A("| 类型 | `web/src/symbology.js` 的形状函数 | 按 GJB 6277 方向规则 |")
A("| 状态 | `--status-border-*` | 状态边框，与选中态区分 |")
A("| 强度 | `--ring-*` / `--strength-*` + `theme.js` 半径规则 | 强度内环 |")
A("| 时效 | `--pulse-*` + `theme.js` 周期规则 | 时效外圈脉冲 |")
A("")
A("## 2. 逐项对照（含**真实行号**）")
A("")
A("| 维度 | 变量 | 取值 | 文件 | 行号 |")
A("| --- | --- | --- | --- | --- |")
for dim, _pat, _desc, keys in DIMS:
    for k in keys:
        name, val, ln = row(k)
        if ln:
            A("| %s | `%s` | `%s` | `situation3d/web/src/theme.css` | %d |" % (dim, name, val, ln))
        else:
            A("| %s | `%s` | %s | — | — |" % (dim, name, val))
A("")
A("## 3. 规则实现位置（`theme.js` 的纯函数）")
A("")
A("| 规则 | 函数 | 行号 |")
A("| --- | --- | --- |")
for label, pat in (("脉冲周期选择（>50km/h → 0.8s，否则 1.5s）", r"function\s+\w*[Pp]ulse\w*"),
                   ("强度内环半径（强度 0 不画；1..4 递增；封顶）", r"function\s+\w*[Rr]ing\w*|function\s+\w*[Ss]trength\w*"),
                   ("读取 CSS 变量", r"function\s+\w*[Cc]ssVar\w*|getComputedStyle")):
    hit = find_first(JS, pat)
    if hit:
        ln, txt = hit
        m = re.search(r"function\s+(\w+)", txt)
        A("| %s | `%s` | %d |" % (label, m.group(1) if m else "（见该行）", ln))
    else:
        A("| %s | **（未找到）** | — |" % label)
A("")
A("## 4. 加载顺序（顺序错了只在运行时报错）")
A("")
if HTML.is_file():
    h = HTML.read_text(encoding="utf-8")
    order = []
    for name in ("style.css", "theme.css", "theme.js", "app.js", "eventFeed.js",
                 "statusPanels.js", "dataHealth.js", "renderGovernor.js"):
        i = h.find("src/" + name)
        if i > 0:
            order.append((h[:i].count("\n") + 1, name))
    order.sort()
    A("| 顺序 | 脚本 | `index.html` 行号 |")
    A("| --- | --- | --- |")
    for n, (ln, name) in enumerate(order, 1):
        A("| %d | `%s` | %d |" % (n, name, ln))
    A("")
    A("**约束**：`theme.js` 必须在 `app.js` 之前（`app.js` 经 `Theme` 取语义色）；")
    A("`eventFeed.js` 必须在 `statusPanels.js` / `dataHealth.js` 之前。")
else:
    A("（`index.html` 缺失，无法列出加载顺序）")
A("")
A("## 5. 截图位置（**待补**）")
A("")
A("> 判据 1) 要求「视觉编码对照表逐项可在界面复现**且截图评审通过**」。")
A(">")
A("> **本节如实留空**：本环境**没有浏览器**（也无 JS 运行时），")
A("> 无法渲染界面，故**不能**提供截图。请在有前端的机器上按本表逐项核对后补图。")
A("")
A("| 维度 | 截图文件 | 状态 |")
A("| --- | --- | --- |")
for dim, *_ in DIMS:
    A("| %s | `docs/shots/%s.png` | **待补（本环境无浏览器）** |" % (dim, dim))
A("")
A("## 6. 本表的验证")
A("")
A("`tests/test_i09_visual_encoding_table.py` 断言：")
A("")
A("- 本表的**每个变量名**都在 `theme.css` 中真实存在；")
A("- 表中标出的**行号**确实指向该变量（防止生成后代码又改而表未更新）；")
A('- 五个维度**各有唯一配置来源**，且不出现「同一语义两个来源」。')
A("")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(L) + "\n")
print("  ✓ 已生成 %s（%d 行）" % (OUT.relative_to(REPO), len(L)))
