#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''断言反模式检测器 —— 把本会话反复踩到的「看似通过实则无效」的断言模式变成**可运行检查**。

## 为什么需要

本会话发现 **9 类**「断言通过但不证明任何东西」的模式，**每一类都曾让我得出错误结论**。
它们不依赖具体项目，**故可固化**。

## 用法

    python3 scripts/check_assertion_antipatterns.py [目录 ...]

默认扫描 `tests/`。退出码：0 = 无发现；1 = 有发现。

## ⚠ 本脚本自身的边界（必须写明）

**它是启发式的，会有误报与漏报。**
命中不等于缺陷，未命中不等于没问题 —— 它只把**可疑处**指出来给人复核，
**不能用「本脚本通过」来声称「断言都有效」**。
'''
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# 反模式定义：每条含 id / 名称 / 为何无效 / 本会话实例
# ══════════════════════════════════════════════════════════════════════
ANTIPATTERNS = [
    {
        "id": "AP1",
        "name": "形式抽检",
        "why": "只检查「有没有数字/提交号」等**形式特征**，不检查**内容是否成立**。",
        "case": "§91 的一致性抽检判 `C4` 通过（它有数字也有提交号），"
                "而那个数字（−0.0629）**已不可复现**。",
        "fix": "改为**重跑**该判据所依赖的命令，比对实际取值。",
    },
    {
        "id": "AP2",
        "name": "跳过即不验证",
        "why": "`importorskip` / `skipif` 写在**依赖尚未就绪**之处，或环境缺依赖时静默跳过，"
               "于是「没跑」被当成「通过」。",
        "case": "§93：我把 `pytest.importorskip` 写在**导入 harness 之前**，"
                "而「把引擎目录加进 `sys.path`」正是 harness 做的 → **整个文件被静默跳过 1 项**。",
        "fix": "把 skip 放在依赖就绪之后；并**显式断言「本应运行的条数」**。",
    },
    {
        "id": "AP3",
        "name": "查错地方报缺失",
        "why": "在**错误的文件/键名/路径**里查找，找不到就断言「不存在」。",
        "case": "§79：在 `intent_segments.jsonl` 里找 `controller_variant` → 全 0，"
                "误报「variants 全为 0」；实际它在 episode 目录名里（21/22 达标）。"
                "§78：只匹配**字面键**，于是把用**动态键**的文件报成「未迁移」。",
        "fix": "先在**已知存在的样本**上验证检索式能命中，再用于判「缺失」。",
    },
    {
        "id": "AP4",
        "name": "目视代替机械核验",
        "why": "读源码后凭印象下结论，而源码的分支/条件可能被忽略。",
        "case": "§90：我目视 `ai/features.py` 后**以为文档写错了**死维范围；"
                "AST + 实跑双重验证后**文档是对的** —— 差点「改对为错」。",
        "fix": "用 AST / 实跑 / 真实执行取证；目视只用于**提出假设**。",
    },
    {
        "id": "AP5",
        "name": "平凡可满足 / 恒真断言",
        "why": "断言在**任何输入**下都成立，故不构成检验。",
        "case": "本会话 4 处自造：`assert n_act <= len(names)`（计数受总数约束）、"
                "`assert v[size_warning] is None or True`（恒真）、"
                "`assert r.frame_at_tick(5) == r.frame_at_tick(5)`（纯函数对自身）。",
        "fix": "断言前问「**是否存在一个输入能让它失败**」；测**边界与负例**。",
    },
    {
        "id": "AP6",
        "name": "守卫命中自身文档",
        "why": "用「旧错误值不在文本中」这类守卫，**而修正说明/文档里必须引用该旧值** → 守卫打中自己。",
        "case": "本会话两处：`assert 2/40 not in note` 打中了**更正说明本身**；"
                "另一处代码守卫匹配到**引用旧公式的 docstring**。",
        "fix": "用 AST 取函数体、**排除 docstring/注释**后再查。",
    },
    {
        "id": "AP7",
        "name": "测试错层（机制 vs 集成）",
        "why": "测的是**合并层/纯函数**，而判据要的是**主链路里真的发生**。",
        "case": "§97：`C5` 的 `test_c5_empty_route_metric.py`（测 `_merge_policy_stats`）通过，"
                "而 `test_move_route_diagnostics.py`（测 `_generate_action` 里计数真增加）失败。"
                "**同一条判据，两侧结论相反。**",
        "fix": "判据写在主链路的**入口**上测；单元测试只作补充。",
    },
    {
        "id": "AP8",
        "name": "环境/仓库状态依赖",
        "why": "断言依赖**文件可执行位、路径存在、工作区状态**，于是结论随环境漂移。",
        "case": "§98：`test_full_promotion_after_auth_script.py` 要求 `.sh` 可执行，"
                "而**全仓标为 100755 的文件数为 0** → 该断言要求的是本仓库从未有过的状态。"
                "这也是 23/24 波动的真正来源（**我此前把归因错放在 v3f 上**）。",
        "fix": "断言只依赖**仓库内可版本化的内容**；环境相关的显式标注为前置条件。",
    },
    {
        "id": "AP9",
        "name": "指向不存在之物",
        "why": "文档/脚本/署名指向**并不存在**的文件或对象，而无人核对。",
        "case": "§92：快照初版署名指向 `scripts/make_status_snapshot.py`，该脚本不存在；"
                "§81：落点写 `code/demo20260104/backend/`，该目录不在本检出。",
        "fix": "**每个被引用的路径都要被断言存在**（本会话已用此法定案）。",
    },
]


def _iter_test_files(roots) -> list:
    out = []
    for r in roots:
        p = Path(r)
        if p.is_file() and p.suffix == ".py":
            out.append(p)
        elif p.is_dir():
            out += sorted(p.rglob("test_*.py"))
    return out


def _segment(src: str, node) -> str:
    """取节点源码片段。**不用 `ast.unparse`** —— 它 3.9+ 才有，
    而本机默认 `python3` 实测为 **3.8**（`ast.unparse` 缺失，脚本首跑即崩）。"""
    seg = None
    try:
        seg = ast.get_source_segment(src, node)      # 3.8+
    except Exception:
        seg = None
    return (seg or ast.dump(node)).strip()


def check_ap5(tree: ast.AST, lines: list, src: str = "") -> list:
    """平凡可满足：`x or True`、`a == a`、`assert True`。"""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        src = _segment(src, node.test)
        if re.search(r"\bor\s+True\b", src) or re.search(r"\band\s+False\b", src):
            hits.append((node.lineno, "恒真表达式: %s" % src[:70]))
        if re.fullmatch(r"True", src.strip()):
            hits.append((node.lineno, "assert True"))
        if re.fullmatch(r"([\w\.\[\]'\"]+)\s*==\s*\1", src.strip()):
            hits.append((node.lineno, "自比较恒真: %s" % src[:70]))
    return hits


def check_ap2(tree: ast.AST, lines: list, path: Path) -> list:
    """跳过即不验证：含 skip 但测试数极少 → 疑似整文件可被跳过。"""
    src = "".join(lines)
    n_skip = len(re.findall(r"importorskip|pytest\.skip|skipif", src))
    if n_skip == 0:
        return []
    n_test = len(re.findall(r"(?m)^def test_", src))
    # 没有测试函数的文件**不是测试文件**（辅助模块），不报 —— 我第一版把它算进来，45 条噪声。
    if n_test == 0:
        return []
    # 只有"1..2 个 test 却有多处 skip"才可疑
    if n_test <= 2 and n_skip > n_test:
        return [(1, "含 %d 处 skip，但仅有 %d 个 test —— 疑似整文件可被跳过" % (n_skip, n_test))]
    return []


def check_ap6(tree: ast.AST, lines: list, src: str = "") -> list:
    """守卫命中自身文档：断言的字面量也出现在本文件 docstring 中。"""
    hits = []
    docs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node)
            if d:
                docs.append(d)
    doc_text = "\n".join(docs)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        seg = _segment(src, node.test)
        # 只有**否定式**断言才会"打中自己"：`X not in Y` / `assert not ...`
        if "not in" not in seg and not seg.lstrip().startswith("not "):
            continue
        for sub in ast.walk(node.test):
            # 字面量要**够长**才算"具体值"；3 字符的短串（如 'off'）几乎必然与文档重合，
            # 我第一版用 >=3 于是 515 条噪声。
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                    and len(sub.value) >= 12 and sub.value in doc_text:
                hits.append((node.lineno,
                             "否定式断言的字面量 %r 也出现在本文件 docstring 中（守卫可能打中自己）"
                             % sub.value[:44]))
    return hits


def check_ap8(tree: ast.AST, lines: list) -> list:
    """环境/仓库状态依赖：断言里出现可执行位、绝对路径、环境变量。"""
    hits = []
    pats = [
        (r"stat\.S_I[XWRO]", "断言依赖文件权限位（随工作区状态漂移）"),
        (r"os\.environ\[", "断言依赖环境变量"),
        (r"/tmp/", "断言依赖 /tmp 路径"),
    ]
    for i, line in enumerate(lines, 1):
        st = line.strip()
        if st.startswith("#") or st.startswith("*"):
            continue
        if "assert" not in st:
            continue
        for pat, msg in pats:
            if re.search(pat, st):
                hits.append((i, "%s: %s" % (msg, st[:70])))
    return hits


def check_ap9(lines: list, path: Path, src: str = "") -> list:
    """指向不存在之物：测试里用 ROOT / "a" / "b" 拼出的相对路径不存在。

    ⚠ 我先后踩了三处误报，均已修正：
      1. 取值域太宽，把整段代码当路径 → `OSError [Errno 36] File name too long`（**检测器自己崩**）；
      2. 假定基目录一律是"仓库根的上级"，而 `test_i09_theme.py` 的
         `REPO = parents[1] / "situation3d"` → 误报 6 条"路径不存在"；
      3. 未排除**否定式**断言（`assert not (REPO / ...).exists()`）——
         `test_tasklist_path_audit.py` **故意**用占位符验证"不存在"，被误报 2 条。
    """
    hits = []
    if "exists()" not in src and "is_file()" not in src:
        return []
    # ① 解析该文件自己的基目录
    base = None
    m = re.search(r'^(?:ROOT|REPO|BASE)\s*=\s*Path\(__file__\)\.resolve\(\)\.parents\[1\](.*)$',
                  src, re.M)
    if m:
        suffix = re.findall(r'/\s*"([A-Za-z0-9_.\- ]{1,60})"', m.group(1))
        base = path.parent.parent.joinpath(*suffix) if suffix else path.parent.parent
    else:
        base = path.parent.parent
    # ② 收集**否定式**断言行号，跳过它们（那些是在验证"不存在"，属正确用法）
    neg_lines = set()
    for i, line in enumerate(src.split("\n"), 1):
        if re.search(r"(is_file\(\)\s*is\s*False|not\s+\(?[^\n]*exists\(\))", line):
            neg_lines.add(i)
    pat = re.compile(r'(?:ROOT|REPO|BASE)((?:\s*/\s*"[A-Za-z0-9_.\- ]{1,60}")+)')
    for mm in pat.finditer(src):
        line_no = src[:mm.start()].count("\n") + 1
        if line_no in neg_lines:
            continue
        parts = re.findall(r'"([A-Za-z0-9_.\- ]{1,60})"', mm.group(1))
        if len(parts) < 2 or len(parts) > 6:
            continue
        cand = base.joinpath(*parts)
        if len(str(cand)) > 400:
            continue
        try:
            if cand.exists():
                continue
        except OSError:
            continue
        try:
            shown = cand.relative_to(path.parent.parent)
        except ValueError:
            shown = cand
        hits.append((line_no, "引用的路径不存在: %s" % shown))
    return hits


def main(argv) -> int:
    # ⚠ **版本前置检查（fail loud）**：本仓测试里有 **3.9+ 语法**
    # （括号式多上下文管理器，PEP 617），3.8 的 `ast.parse` 读不了 ——
    # 实测默认 `python3` 为 3.8，会让两个文件"解析失败"。
    # **若静默跳过它们，就会漏检** —— 故此处直接拒绝运行，而不是少扫几个文件。
    if sys.version_info < (3, 9):
        print("  ! 需要 Python >= 3.9（当前 %d.%d）：本仓测试含 3.9+ 语法，"
              "低版本 ast.parse 会失败并**静默漏检**" % sys.version_info[:2])
        print("    请先 `set -a; . /remote-home/lizilong/py310env.sh; set +a`")
        return 2
    roots = argv or ["tests"]
    files = _iter_test_files(roots)
    if not files:
        print("  未找到测试文件（roots=%s）" % roots)
        return 0

    print("  扫描 %d 个测试文件" % len(files))
    total = 0
    per = {}
    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError) as e:
            # **不静默跳过**：解析不了就等于漏检，必须让整体失败
            # （这与 AP2「跳过即不验证」同类 —— 我第一版的替换因锚点不匹配**静默失败**，
            #   于是该守卫实际从未生效；由自证测试 `test_parse_failure_is_fatal_not_skipped` 抓出。）
            print("  ! %s 解析失败: %s" % (f.name, str(e)[:80]))
            print("    -> 无法扫描该文件 = 漏检，**不应视为通过**")
            return 3

        lines = src.split("\n")
        found = list(check_ap5(tree, lines, src))
        found += check_ap2(tree, lines, f)
        found += check_ap6(tree, lines, src)
        found += check_ap8(tree, lines)
        found += check_ap9(lines, f, src)
        if found:
            per[f.name] = found
            total += len(found)

    if not per:
        print("  未发现可疑断言（! 这不等于断言都有效，见文件头的能力边界）")
        return 0

    print()
    for name, found in sorted(per.items()):
        print("  %s" % name)
        for ln, msg in found[:8]:
            print("     L%-5d %s" % (ln, msg))
        if len(found) > 8:
            print("     ... 另有 %d 条" % (len(found) - 8))
    print()
    print("  合计 %d 处可疑 —— **需人工复核，勿直接当缺陷**（启发式，有误报）" % total)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
