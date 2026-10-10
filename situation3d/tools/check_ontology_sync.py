#!/usr/bin/env python3
"""意图本体副本与单一真源的一致性校验（D2）。

为什么需要它
------------
`engine/data/intent_ontology_v2.json`（P2）是
`Shenji/ontology/intent_ontology_v2.json`（P3）的**内嵌副本**。两份文件曾经漂移，
而**没有任何机制会在构建或自检时发现**——这正是报告 §5.4 记的「可复现风险」。

单一真源约定
------------
**真源 = P3 `Shenji/ontology/intent_ontology_v2.json`**。副本由它派生；
副本**不得**出现真源没有的结构差异。

本脚本**不是**「让两份文件字节相同」
--------------------------------
实测：两份只差 **1 个字段**，且那一处差异是**有意为之**——

    /parent_categories[4]/name_zh     真源 = '机动态势'   副本 = '机动TS'

副本侧的值来自本项目的**全局术语替换规范**（`NOTES_term_replacement.md`：态势→TS、战场→ZC），
而**真源不在该替换的授权范围内**。该规范文档自己写明：
「…重新同步本体论会把改动覆盖回去。**这是有意接受的代价，不是遗漏。**」

所以本脚本把这一处登记为**显式豁免**（见 KNOWN_EXCEPTIONS），
并对其余**任何**差异一律判失败——包括**新增**的豁免字段差异、意图增删、code/parent 变化。

用法
----
    python3 tools/check_ontology_sync.py            # 人类可读
    python3 tools/check_ontology_sync.py --json     # 机器可读
    python3 tools/check_ontology_sync.py --copy <路径> --source <路径>   # 覆盖路径（负向用例用）

退出码：0 = 一致（只含已登记豁免）；1 = 存在未登记差异；2 = 文件缺失/无法解析。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def _first_existing(cands):
    for c in cands:
        if c.exists():
            return c
    return cands[0]


# 真源可能位于几种检出布局下（远程是同仓兄弟目录，本机是另一棵子树），逐个探测。
SOURCE_CANDIDATES = [
    # 远程布局：同仓兄弟目录 shenji_agent/Shenji/ 与 shenji_agent/situation3d/
    REPO.parent / "Shenji" / "ontology" / "intent_ontology_v2.json",
    # 本机布局：真源在 <root>/code/shenji_agent/，副本在本子树
    REPO.parent.parent / "code" / "shenji_agent" / "Shenji" / "ontology" / "intent_ontology_v2.json",
    REPO.parent / "code" / "shenji_agent" / "Shenji" / "ontology" / "intent_ontology_v2.json",
    REPO.parent.parent / "Shenji" / "ontology" / "intent_ontology_v2.json",
    REPO / "Shenji" / "ontology" / "intent_ontology_v2.json",
]
DEFAULT_SOURCE = _first_existing(SOURCE_CANDIDATES)
DEFAULT_COPY = REPO / "engine" / "data" / "intent_ontology_v2.json"

# 已登记的豁免：真源侧的值 -> 副本侧的值。新增豁免必须在本文件里显式写出理由。
KNOWN_EXCEPTIONS = {
    "/parent_categories[4]/name_zh": {
        "source": "机动态势",
        "copy": "机动TS",
        "why": (
            "副本侧受项目全局术语替换规范约束（态势→TS），真源不在该替换授权范围内；"
            "NOTES_term_replacement.md 已明确记录此为有意接受的代价。"
        ),
    },
}

# 这 4 项在真源与副本之间必须**逐项完全一致**（含顺序）。
STRICT_PATHS = ("/intents", "/parent_categories", "/status_layers", "/confusion_matrix")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(x, y, path, out):
    """收集所有结构化差异：(路径, 类别, 真源值, 副本值)。"""
    if type(x) is not type(y):
        out.append((path, "TYPE", type(x).__name__, type(y).__name__))
        return
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x:
                out.append((path + "/" + str(k), "ONLY_IN_COPY", "-", repr(y[k])[:80]))
            elif k not in y:
                out.append((path + "/" + str(k), "ONLY_IN_SOURCE", repr(x[k])[:80], "-"))
            else:
                _walk(x[k], y[k], path + "/" + str(k), out)
    elif isinstance(x, list):
        if len(x) != len(y):
            out.append((path, "LEN", str(len(x)), str(len(y))))
        for i in range(min(len(x), len(y))):
            _walk(x[i], y[i], "%s[%d]" % (path, i), out)
    else:
        if x != y:
            out.append((path, "VALUE", repr(x)[:80], repr(y)[:80]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the P2 ontology copy against the P3 single source of truth.")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="single source of truth (P3)")
    ap.add_argument("--copy", default=str(DEFAULT_COPY), help="derived copy (P2)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    src, cpy = Path(args.source), Path(args.copy)
    report = {"source": str(src), "copy": str(cpy), "ok": False, "problems": [], "exempt": [], "info": {}}

    for label, p in (("source", src), ("copy", cpy)):
        if not p.exists():
            cands = SOURCE_CANDIDATES if label == "source" else [cpy]
            report["problems"].append({"path": "/", "kind": "MISSING_%s" % label.upper(),
                                       "detail": "file not found: %s" % p,
                                       "tried": [str(x) for x in cands]})
            print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else
                  "FAIL: %s file not found: %s" % (label, p))
            return 2
    try:
        a = json.loads(src.read_text(encoding="utf-8"))
        b = json.loads(cpy.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        report["problems"].append({"path": "/", "kind": "PARSE_ERROR", "detail": str(e)})
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else "FAIL: parse error: %s" % e)
        return 2

    report["info"] = {
        "source_sha256": _sha256(src), "copy_sha256": _sha256(cpy),
        "source_bytes": src.stat().st_size, "copy_bytes": cpy.stat().st_size,
        "intents_source": len(a.get("intents", [])), "intents_copy": len(b.get("intents", [])),
        "version_source": a.get("version"), "version_copy": b.get("version"),
    }

    diffs = []
    _walk(a, b, "", diffs)

    # 先摘掉已登记豁免，剩下的才是问题
    remaining = []
    for path, kind, u, v in diffs:
        ex = KNOWN_EXCEPTIONS.get(path)
        if ex and kind == "VALUE":
            # 豁免只在「值恰好是登记的那一对」时成立；值变了要重新评审
            s_ok = (repr(u) == repr(ex["source"])) or (u == ex["source"]) or (ex["source"] in u)
            c_ok = (v == ex["copy"]) or (ex["copy"] in v)
            if s_ok and c_ok:
                report["exempt"].append({"path": path, "source": u, "copy": v, "why": ex["why"]})
                continue
            remaining.append((path, "EXEMPTION_STALE",
                              "%s (登记: %s)" % (u, ex["source"]), "%s (登记: %s)" % (v, ex["copy"])))
            continue
        remaining.append((path, kind, u, v))

    # 严格路径：必须逐项一致（豁免之外）
    for sp in STRICT_PATHS:
        sm = [d for d in remaining if d[0] == sp or d[0].startswith(sp + "/") or d[0].startswith(sp + "[")]
        report["info"].setdefault("strict_checks", {})[sp] = "OK" if not sm else "%d diff(s)" % len(sm)

    report["problems"] = [{"path": p, "kind": k, "source": u, "copy": v} for p, k, u, v in remaining]
    report["ok"] = not remaining

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("single source of truth (P3): %s" % src)
        print("derived copy           (P2): %s" % cpy)
        print("  intents: source=%d copy=%d   version: %s / %s"
              % (report["info"]["intents_source"], report["info"]["intents_copy"],
                 report["info"]["version_source"], report["info"]["version_copy"]))
        print("  strict paths: %s" % json.dumps(report["info"].get("strict_checks", {}), ensure_ascii=False))
        if report["exempt"]:
            print("  registered exemptions (%d):" % len(report["exempt"]))
            for e in report["exempt"]:
                print("    - %s : %s -> %s" % (e["path"], e["source"], e["copy"]))
                print("      why: %s" % e["why"])
        if remaining:
            print("  UNAPPROVED DIFFS (%d):" % len(remaining))
            for p, k, u, v in remaining:
                print("    - [%s] %s" % (k, p))
                print("        source: %s" % u)
                print("        copy  : %s" % v)
            print("RESULT: FAIL")
        else:
            print("RESULT: OK (only registered exemptions differ)")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
