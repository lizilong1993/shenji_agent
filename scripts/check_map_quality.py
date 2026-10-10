#!/usr/bin/env python3
"""地图数据质量门（C5 步骤 2/6）：邻接表的**结构一致性**、对称性、连通性、孤立节点、坐标越界。

为什么需要它
------------
`reports/shenji_agent_promotion_v3_report.md` 记：空路径 15,506/23,500 = **66%**，集中在 map_id 19/212。
但「空路径多」只是**症状**。本脚本把症状拆成可定位的**结构性事实**，并逐个地图给出判定。

本脚本发现的**关键结构性事实**（本次实测）
--------------------------------------------
1. **16 张地图里有两种 cell schema**：
   - 「规范型」`{pos,elev,cond,roads,rivers,neighbors}` —— 15 张；
   - 「简写型」`{nb,c,r,e,cd,rd,rv}` —— **只有 map_212**。
2. `ai/agent.py` 的 `Map.get_neighbors()` 读的是 **`cell["neighbors"]`**（`divmod(pos,100)` 解坐标），
   `gen_move_route_bfs()` 亦经它取邻接。
   → 对 **map_212**，该键**不存在**，`get_neighbors()` 会抛 `KeyError`。
3. map_212 的 `nb` 与 `cost.pickle` 的边权重**完全对齐**（同一批邻接 id），
   故 `gen_move_route()`（走 cost）能跑，**只有 bfs 回退路径炸** —— 这是「静默产生空路径」的一类。

坐标编码
--------
`divmod(pos, 100)` ⇒ **pos = row*100 + col**。本脚本按此解码，并校验 `pos` 与 `[r][c]` 下标一致。

用法
----
    python3 tools/check_map_quality.py                    # 扫全部 map_*/basic.json
    python3 tools/check_map_quality.py --maps-dir <目录>
    python3 tools/check_map_quality.py --json
    python3 tools/check_map_quality.py --strict           # 有任一 FAIL 即非 0 退出

退出码：0 = 无 FAIL；1 = 存在 FAIL（仅 --strict）；2 = 无法读取。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from pathlib import Path

CANONICAL_CELL_KEY = "neighbors"   # ai/agent.py Map.get_neighbors() 依赖
ALIAS_CELL_KEYS = ("nb",)
STRICT_ALIAS_OK = False            # 别名键不被 get_neighbors() 识别 -> 记为 FAIL


def _decode(pos):
    return divmod(pos, 100)


def inspect_map(path: Path):
    res = {"file": str(path), "fail": [], "warn": [], "info": {}}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        res["fail"].append("JSON 解析失败: %s" % e)
        return res
    md = d.get("map_data")
    if not isinstance(md, list) or not md or not isinstance(md[0], list):
        res["fail"].append("map_data 不是非空二维 list")
        return res
    rows, cols = len(md), len(md[0])

    res["info"] = {
        "map_id": d.get("map_id"), "rows": rows, "cols": cols,
        "coordinate_system": d.get("coordinate_system"),
        "coordinate_number_length": d.get("coordinate_number_length"),
    }

    # --- schema：cell 是否有 get_neighbors() 需要的键
    seen_keys = Counter()
    for r in range(min(rows, 5)):
        for c in range(min(cols, 5)):
            cell = md[r][c]
            if isinstance(cell, dict):
                for k in cell:
                    seen_keys[k] += 1
    res["info"]["cell_keys"] = sorted(seen_keys)
    has_canon = CANONICAL_CELL_KEY in seen_keys
    alias = [k for k in ALIAS_CELL_KEYS if k in seen_keys]
    res["info"]["has_neighbors_key"] = has_canon
    res["info"]["alias_keys"] = alias
    if not has_canon:
        if alias and not STRICT_ALIAS_OK:
            res["fail"].append(
                "缺少 cell['%s'] —— ai/agent.py Map.get_neighbors() 会 KeyError"
                "（本图用的是别名键 %s）" % (CANONICAL_CELL_KEY, alias))
        else:
            res["fail"].append("缺少 cell['%s'] 且无可用别名键" % CANONICAL_CELL_KEY)

    # --- 逐格：坐标自洽 + 邻接对称 + 悬空/自环
    cells = {}
    pos_mismatch = []
    for r, row in enumerate(md):
        for c, cell in enumerate(row):
            if not isinstance(cell, dict):
                continue
            p = cell.get("pos")
            if isinstance(p, int) and p != r * 100 + c:
                pos_mismatch.append(((r, c), p))
            cells[(r, c)] = cell
    if pos_mismatch:
        res["fail"].append("pos 与 [r][c] 不一致: %d 处，样例 %s" % (len(pos_mismatch), pos_mismatch[:3]))

    edges, dangling, selfloop = set(), 0, 0
    for (r, c), cell in cells.items():
        nb = cell.get(CANONICAL_CELL_KEY)
        if nb is None:
            for k in ALIAS_CELL_KEYS:
                if k in cell:
                    nb = cell[k]
                    break
        if not isinstance(nb, list):
            continue
        for t in nb:
            if not isinstance(t, int) or t < 0:
                continue
            rc = _decode(t)
            if rc == (r, c):
                selfloop += 1
                continue
            if rc not in cells:
                dangling += 1
                continue
            edges.add(((r, c), rc))
    res["info"].update(edges=len(edges), dangling=dangling, selfloop=selfloop)
    if dangling:
        res["fail"].append("邻接指向不存在的格子: %d 处" % dangling)
    if selfloop:
        res["warn"].append("自环: %d 处" % selfloop)

    asym = sum(1 for a, b in edges if (b, a) not in edges)
    res["info"]["asymmetric"] = asym
    if asym:
        res["fail"].append("邻接不对称: %d 对（A→B 无 B→A）" % asym)

    # --- 连通性
    und = {}
    for a, b in edges:
        und.setdefault(a, set()).add(b)
        und.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for s in cells:
        if s in seen:
            continue
        q = deque([s]); seen.add(s); n = 0
        while q:
            x = q.popleft(); n += 1
            for y in und.get(x, ()):
                if y not in seen:
                    seen.add(y); q.append(y)
        comps.append(n)
    comps.sort(reverse=True)
    iso = sum(1 for k in cells if not und.get(k))
    res["info"].update(components=len(comps), largest=comps[0] if comps else 0,
                       isolated=iso, total_cells=len(cells))
    if iso:
        res["fail"].append("孤立节点（度=0）: %d / %d" % (iso, len(cells)))
    if len(comps) > 1:
        res["warn"].append("连通分量 %d 个（最大 %d / %d 格）" % (len(comps), comps[0], len(cells)))

    res["ok"] = not res["fail"]
    return res


def main() -> int:
    # 本脚本位于 <repo>/scripts/，故 parent.parent 即仓库根
    repo_root = Path(__file__).resolve().parent.parent
    default_dir = repo_root / "land_wargame_sdk" / "Data" / "Data" / "maps"
    ap = argparse.ArgumentParser(description="Map data quality gate (C5).")
    ap.add_argument("--maps-dir", default=str(default_dir))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any map has FAIL")
    args = ap.parse_args()

    root = Path(args.maps_dir)
    files = sorted(root.glob("map_*/basic.json"))
    if not files:
        print("no map_*/basic.json under %s" % root)
        return 2

    reports = [inspect_map(p) for p in files]
    nfail = sum(1 for r in reports if not r.get("ok"))

    if args.json:
        print(json.dumps({"maps_dir": str(root), "count": len(reports),
                          "failed": nfail, "reports": reports}, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("maps dir: %s   (%d maps)" % (root, len(reports)))
        print()
        print("%-12s %-6s %-9s %-6s %-7s %-8s %-7s %s" %
              ("map", "id", "grid", "cells", "edges", "asym", "iso", "verdict"))
        for r in reports:
            i = r["info"]
            name = Path(r["file"]).parent.name
            verdict = "OK" if r.get("ok") else "FAIL"
            print("%-12s %-6s %-9s %-6s %-7s %-8s %-7s %s" % (
                name, i.get("map_id"), "%sx%s" % (i.get("rows"), i.get("cols")),
                i.get("total_cells"), i.get("edges"), i.get("asymmetric"),
                i.get("isolated"), verdict))
        print()
        for r in reports:
            if r.get("ok"):
                continue
            print("FAIL %s" % Path(r["file"]).parent.name)
            for f in r["fail"]:
                print("     - %s" % f)
        print()
        print("RESULT: %d/%d maps FAIL" % (nfail, len(reports)))
    return 1 if (args.strict and nfail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
