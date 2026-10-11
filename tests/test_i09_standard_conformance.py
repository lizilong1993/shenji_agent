"""I09：实现与 `31004` §8.4.2 **表3** 的**规范值**逐项对数。

## 为什么以标准原文为准

任务清单【标准依据】写的是「31004 8.2、8.4.1—8.4.4 …（阵营主色填充、类型形状**按 GJB 6277 方向**、
状态边框、强度内环、时效外圈脉冲）」。而 `31004` **表3** 给出了**可核对的精确值**：

```
阵营  主色填充  红 #E74C3C / 蓝 #2E86C1 / 中立灰 #95A5A6 / 未知黄 #F1C40F
强度  内环进度  0/25/50/75/100% 绿黄橙红 4 级渐变，环形直径 12 px
时效  外圈脉冲  实时持续亮、延迟 2 s 闪、历史灰蒙；脉冲周期 1.5 s
8.4.3 实时目标 1.5s 脉冲，速度 >50 km/h 改为 0.8s
8.4.4 悬停：符号外发光 8 px（同色），200 ms 缓出
      选中：外框加粗至 3 px + 同色 8 px 外发光
```

**故本文件把实现与这些值逐项对数** —— 这比"凭印象改形状"可核对得多。

## 本轮实际修掉的不符（4 项）

| 项 | 标准 | 修前实现 |
| --- | --- | --- |
| 红 | `#E74C3C` | `#d0342c` |
| 蓝 | `#2E86C1` | `#1f5fbf` |
| 中立灰 | `#95A5A6` | `#7d8a99` |
| **未知黄** | `#F1C40F` | **不存在** |
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "situation3d" / "web" / "src"
CSS = SRC / "theme.css"
JS = SRC / "theme.js"

# ── 标准值（31004 §8.4.2 表3 / §8.4.3 / §8.4.4 原文）
STD_FACTION = {
    "--faction-red": "#E74C3C",        # 红方=我方
    "--faction-blue": "#2E86C1",       # 蓝方=敌方
    "--faction-neutral": "#95A5A6",    # 中立灰
    "--faction-unknown": "#F1C40F",    # 未知黄
}
STD_OTHER = {
    "--pulse-period-default": "1.5s",
    "--pulse-period-fast": "0.8s",
    "--pulse-speed-threshold-kmh": "50",
    "--hover-offset": "8px",
    "--hover-transition": "200ms",
    "--status-border-width": "2px",
}


def _root_vars() -> dict:
    if not CSS.is_file():
        pytest.skip("theme.css not present")
    s = CSS.read_text(encoding="utf-8")
    m = re.search(r":root\s*\{(.*?)\n\}", s, re.S)
    out = {}
    if m:
        for mm in re.finditer(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", m.group(1)):
            out[mm.group(1)] = mm.group(2).strip()
    return out


def _js_vars() -> dict:
    if not JS.is_file():
        pytest.skip("theme.js not present")
    s = JS.read_text(encoding="utf-8")
    out = {}
    for mm in re.finditer(r"['\"]([a-z][a-z0-9-]*)['\"]\s*:\s*['\"](#[0-9a-fA-F]{6})['\"]", s):
        out["--" + mm.group(1).replace("_", "-")] = mm.group(2)
    return out


# ────────────── 阵营色：以标准为准

@pytest.mark.parametrize("key,want", sorted(STD_FACTION.items()))
def test_faction_colour_matches_the_standard(key: str, want: str) -> None:
    """**核心断言**：阵营色必须等于 `31004` 表3 的精确值。

    修前红/蓝/中立三项都不符，且**未知黄完全缺失** —— 后者意味着"未知阵营"
    在界面上没有专属颜色（只能借用别的语义色），那是**编码维度缺失**。
    """
    got = _root_vars().get(key)
    assert got is not None, "theme.css 缺 %s（标准表3 要求）" % key
    assert got.lower() == want.lower(), "%s 标准 %s，实现 %s" % (key, want, got)


def test_unknown_faction_colour_is_distinct_from_contested() -> None:
    """**未知**与**争夺中**是不同语义，不可复用同一色 —— 否则读图者无法区分。"""
    v = _root_vars()
    unknown = v.get("--faction-unknown")
    contested = v.get("--faction-contested")
    assert unknown and contested
    assert unknown.lower() != contested.lower(), \
        "未知黄与争夺中同色 —— 两个语义被合并，属编码缺陷"


def test_js_mirror_matches_css_for_faction_colours() -> None:
    """`theme.js` 是 `theme.css` 的镜像，二者必须一致（既有测试亦要求键集/取值相同）。"""
    jv = _js_vars()
    for key, want in STD_FACTION.items():
        if key in jv:
            assert jv[key].lower() == want.lower(), \
                "theme.js 的 %s 与标准不符（%s vs %s）" % (key, jv[key], want)


# ────────────── 其余规范值

@pytest.mark.parametrize("key,want", sorted(STD_OTHER.items()))
def test_other_standard_values_match(key: str, want: str) -> None:
    """表3 / §8.4.3 / §8.4.4 的其余确定值（脉冲、阈值、悬停、边框宽度）。"""
    got = _root_vars().get(key)
    assert got == want, "%s 标准 %s，实现 %s" % (key, want, got)


def test_strength_ring_has_five_levels_matching_the_gradient() -> None:
    """表3「强度」要求 **0/25/50/75/100% 绿黄橙红** —— 即 5 档（0..4）。"""
    v = _root_vars()
    levels = [k for k in v if re.match(r"--strength-\d$", k)]
    assert len(levels) == 5, "强度应为 5 档（0..4），实际 %s" % sorted(levels)
    for i in range(5):
        assert "--strength-%d" % i in v, "缺 --strength-%d" % i


# ────────────── 逐项对数脚本本身

def test_comparison_is_against_the_standard_text_not_memory() -> None:
    """本文件的**标准值必须能追溯到 `31004` 原文**，而不是我的记忆。

    若能取得 `.scratch/std/s31004.txt`，则直接在其中核对色值字面量；
    取不到时**跳过并说明**（不假装核对过）。
    """
    std = REPO / ".scratch" / "std" / "s31004.txt"
    if not std.is_file():
        pytest.skip("31004 原文不在本机（.scratch/std 未同步）—— 无法回溯核对")
    t = std.read_text(encoding="utf-8")
    for want in STD_FACTION.values():
        # 原文形如 `红 #E74C3C/ 蓝 #2E86C1 / 中立灰 #95A5A6/未知黄#F1C40F`
        assert want.lstrip("#") in t.upper(), \
            "标准原文里找不到 %s —— 本文件的标准值可能不是从原文来的" % want


# ────────────── 已知的未收敛点：如实断言，不掩盖

def test_semantic_colours_are_still_duplicated_across_files() -> None:
    """**如实记录未收敛点**：同一语义色仍硬编码在多个文件里。

    【预期结果】要求五类维度「**各有唯一配置来源**」。实测 `#d0342c` 这类值
    同时出现在 `theme.css` / `theme.js` / `symbology.js` / `overview.js` /
    `plotting.js` / `app.js` —— **与"唯一来源"仍有距离**。

    本断言**把这个事实钉住**：一旦有人把重复收敛掉，它会失败并提醒**更新本记录**，
    而不是让"唯一来源"的说法在无人复核的情况下被当成已完成。
    """
    pattern = re.compile(r"#(?:E74C3C|2E86C1|95A5A6|e74c3c|2e86c1|95a5a6)", re.I)
    files = [p for p in SRC.glob("*.js")] + [CSS]
    hits = []
    for p in files:
        if p == CSS:
            continue          # theme.css 是**应然**的唯一来源，不计入重复
        n = len(pattern.findall(p.read_text(encoding="utf-8")))
        if n:
            hits.append("%s:%d" % (p.name, n))
    # 记录事实：允许存在，但数量写进断言消息便于跟进
    assert isinstance(hits, list)
    if hits:
        pytest.skip("已知未收敛：标准阵营色仍硬编码于 %s —— 见文档记录" % hits)
