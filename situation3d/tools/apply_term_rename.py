#!/usr/bin/env python3
"""术语替换：态势 -> TS、战场 -> ZC。

这是**一次性的迁移工具**，保留在仓库里是为了让这次替换可复核、可复现：清单怎么来的、
哪些地方被豁免、豁免理由是什么，都在这里，而不是散在一次性的 sed 里。

三条豁免规则（见 NOTES_term_replacement.md 的风险表）
-----------------------------------------------------
R1  产品名「三维战场态势生成系统」原样保留。
    机械替换会得到「三维ZCTS生成系统」——作为对外名称不可读。
R2  本体论里的「机动态势」（MANEUVER_POSTURE）原样保留。
    它是 2525/庙算术语体系的父类名称，会显示在意图集合里，且与
    code/shenji_agent/Shenji/ontology/ 下的孪生文件必须保持同一口径。
R3  「战场态势」相邻出现时写成 `ZC-TS` 而不是 `ZCTS`——两个缩写黏在一起无法辨读。
    （产品名内的那两处仍按 R1 保留。）

其它不动的类别
--------------
* 英文拼写 `situation`/`Situation`（219 处 / 42 文件）：它是标识符不是中文术语，
  改名会牵动 `SituationHost`/`SituationPanel` 等公开全局名，属另一轮的重命名工作。
* `ts-*` CSS 类名前缀：ASCII，本替换不涉及；但替换后「态势→TS」与该命名空间同形，
  已在注释里注明歧义。
* 二进制与第三方：`tools/shots/`、`web/vendor/`、`web/tiles/`。
* 生成数据集 `web/data/*.json`：不在本脚本范围内，替换后另行重建。

用法
----
    python3 tools/apply_term_rename.py --report          # 只出清单，不改文件
    python3 tools/apply_term_rename.py --apply           # 按同样的规则改写文件
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 只在仓库跟踪的文本文件里改，天然排除 vendor/tiles/shots/二进制。
TEXT_EXT = {".js", ".mjs", ".py", ".css", ".html", ".json", ".md", ".sh",
            ".yml", ".yaml", ".example", ".txt", ".toml", ".cfg"}

PRODUCT = "三维战场态势生成系统"       # R1
ADJACENT = "战场态势"                    # R3
ONTOLOGY_TERM = "机动态势"               # R2

# 逐行豁免标记：夹在这两行之间的内容一律不改。
#
# 为什么需要它：**解释这次替换的注释本身必须引用原词**。比如 index.html 里标注
# "产品名保留原样"的注释会写出「三维战场态势生成系统」，style.css 里说明 `ts-` 前缀
# 歧义的注释会写出「态势」——工具复扫时会把它们当成漏改的地方再次提议改写，形成
# "文档一写就又要改"的循环。用显式标记把这类引用圈起来，语义清楚且可 grep。
KEEP_START = "term-replace:keep-start"
KEEP_END = "term-replace:keep-end"

# 整份豁免的文件：规则定义与它的说明文档天然含有原词。
SKIP_FILES = {
    "tools/apply_term_rename.py",
    "NOTES_term_replacement.md",
}

# 顺序要紧：PRODUCT 含 ADJACENT，必须排在前面才不会先被 ADJACENT 吃掉。
_PAT = re.compile("|".join(re.escape(s) for s in
                           (PRODUCT, ADJACENT, ONTOLOGY_TERM, "态势", "战场")))


def convert(text: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        s = m.group(0)
        if s == PRODUCT:
            return s            # R1
        if s == ADJACENT:
            return "ZC-TS"      # R3
        if s == ONTOLOGY_TERM:
            return s            # R2
        if s == "态势":
            return "TS"
        return "ZC"

    # 逐行处理，以便识别 KEEP 区间。行尾不丢：splitlines(keepends=True)。
    out = []
    in_keep = False
    for line in text.splitlines(keepends=True):
        if KEEP_START in line:
            in_keep = True
        if in_keep or KEEP_END in line:
            out.append(line)
            if KEEP_END in line:
                in_keep = False
            continue
        out.append(_PAT.sub(repl, line))
    return "".join(out)


def _git_bin():
    """
    找到可用的 git。

    这个项目的部署容器里 git **不在 PATH 上**（装在 /remote-home/lizilong/.local/bin），
    直接调 "git" 会 FileNotFoundError。按下面的顺序找：
      1. 环境变量 GIT_BIN —— 容器布局特殊时显式指路，避免把某个人的家目录写死在代码里
      2. PATH
      3. ~/.local/bin 与几个标准位置
    """
    import shutil
    env = os.environ.get("GIT_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    found = shutil.which("git")
    if found:
        return found
    tried = []
    for cand in (os.path.expanduser("~/.local/bin/git"),
                 "/usr/local/bin/git", "/usr/bin/git", "/bin/git"):
        tried.append(cand)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    sys.stderr.write("找不到 git。已尝试 PATH 与：%s\n"
                     "可用 GIT_BIN=/path/to/git 显式指定。\n" % ", ".join(tried))
    sys.exit(1)


def tracked_files():
    try:
        out = subprocess.run([_git_bin(), "ls-files"], cwd=ROOT, check=True,
                             stdout=subprocess.PIPE).stdout.decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("git ls-files 失败：%s\n" % exc)
        sys.exit(1)
    for rel in out.splitlines():
        ext = os.path.splitext(rel)[1].lower()
        if ext in TEXT_EXT or os.path.basename(rel) in (".gitignore", ".env.example"):
            yield rel


def _cell(s: str) -> str:
    """Markdown 表格单元格：竖线要转义，过长要截断，否则清单不成表。"""
    s = s.replace("|", "\\|").strip()
    return s if len(s) <= 150 else s[:147] + "…"


BEGIN = "<!-- BEGIN AUTO-INVENTORY -->"
END = "<!-- END AUTO-INVENTORY -->"


def write_inventory(path: str, entries) -> None:
    """
    把**逐处清单**写进 `path`，只替换两个标记之间的内容。

    为什么不是整份覆盖：这个文档一半是**手写的分析**（替换规则、豁免理由、风险表），
    一半是**机器生成的清单**。整份覆盖会把分析冲掉，而清单又必须能随源码重新生成。
    用标记把两半隔开即可互不干扰——手改分析、重跑生成表格，文件始终完整。

    标记不存在时**不猜**：把表格追加到末尾并提示补标记，避免误以为清单已更新。
    """
    table = ["| 文件:行 | 原文 | 替换后 | 标记 |", "|---|---|---|---|"]
    for rel, lineno, before, after, mark in entries:
        table.append("| `%s:%d` | %s | %s | %s |"
                     % (rel, lineno, _cell(before), _cell(after), mark))
    block = BEGIN + "\n" + "\n".join(table) + "\n" + END + "\n"

    try:
        with open(path, "r", encoding="utf-8") as fh:
            existing = fh.read()
    except OSError:
        existing = ""

    if BEGIN in existing and END in existing:
        head, rest = existing.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        out = head + block + tail.lstrip("\n")
    elif existing:
        out = existing.rstrip("\n") + "\n\n" + block
        print("提示：%s 里没有 %s 标记，清单已追加到末尾——"
              "请把分析文字放进标记之间，免得下次生成把它覆盖。" % (path, BEGIN))
    else:
        out = block
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)


def mark_of(before: str, after: str) -> str:
    if PRODUCT in before and PRODUCT in after:
        return "KEEP (R1 产品名)"
    if ONTOLOGY_TERM in before and ONTOLOGY_TERM in after:
        return "KEEP (R2 本体论术语)"
    if "ZC-TS" in after:
        return "RISK (R3 相邻缩写，写作 ZC-TS)"
    return "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--report", action="store_true", help="只出清单，不改文件")
    g.add_argument("--apply", action="store_true", help="改写文件")
    ap.add_argument("--only", default=None, help="只处理该前缀（便于分批）")
    ap.add_argument("--inventory", default=None,
                    help="把逐处清单写到该路径（markdown）")
    args = ap.parse_args()

    total_files = total_hits = 0
    rows = []
    entries = []
    for rel in sorted(tracked_files()):
        if args.only and not rel.startswith(args.only):
            continue
        if rel in SKIP_FILES:
            continue
        path = os.path.join(ROOT, rel)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                original = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if "态势" not in original and "战场" not in original:
            continue
        updated = convert(original)
        hits = 0
        for i, (a, b) in enumerate(zip(original.splitlines(),
                                      updated.splitlines()), start=1):
            if a != b:
                hits += 1
                entries.append((rel, i, a, b, mark_of(a, b)))
        # 同时记录**未被改写**但含术语的行（R1/R2 豁免），否则清单会漏掉它们。
        #
        # 注意这一遍必须在 "updated == original 就跳过" 之前完成：像
        # engine/data/intent_ontology_v2.json 那样**整份只有豁免词**的文件，
        # 若先跳过就会在清单里彻底消失——而"没有任何一处需要改"与"这个文件被漏掉了"
        # 必须能区分开。
        for i, a in enumerate(original.splitlines(), start=1):
            if ("态势" in a or "战场" in a) and convert(a) == a:
                entries.append((rel, i, a, a, mark_of(a, a)))
        if not hits:
            continue
        total_files += 1
        total_hits += hits
        if args.apply:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(updated)
        rows.append((rel, hits))

    entries.sort(key=lambda e: (e[0], e[1]))

    print("%s：%d 个文件，%d 行被改写，清单共 %d 条"
          % ("已应用" if args.apply else "待改写", total_files, total_hits,
             len(entries)))
    for rel, hits in rows:
        print("  %-42s %d 行" % (rel, hits))

    if args.inventory:
        write_inventory(args.inventory, entries)
        print("逐处清单已写入 %s" % args.inventory)
    return 0


if __name__ == "__main__":
    sys.exit(main())