#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D Tiles 体检工具（适配本项目的 zip 直发数据）。

背景
----
项目的 3D Tiles 不摊成文件树，而是以 `<dataset>.zip` 存放在 `tiles3d_db/`，
由 `server/serve.py` 的 `/tiles3d/<dataset>/<路径>` 从 zip 中央目录直发。
因此文章里那套"对目录跑 stats/prune"的脚本不能直接用；这里直接读 zip 条目，
不依赖服务器、不产生网络请求。

用法
----
    python3 tools/tileset_stats.py --zip tiles3d_db/taibei101.zip
    python3 tools/tileset_stats.py --zip tiles3d_db/taibei101.zip --top 15
    python3 tools/tileset_stats.py --zip tiles3d_db/taibei101.zip --prune-depth 12

输出
----
    - 瓦片节点数 / 外部 tileset 数 / 实际内容数（glb、b3dm…）
    - 每层节点数、内容体积、平均/最大单块
    - 体检结论：平均单块过大/过碎、层级过深、缺失引用
    - `--prune-depth N`：演练"保留 N 层"能剪掉多少（不修改 zip）

注意
----
本工具**只读**。真正的剪枝/合并/纹理压缩应在一份离线副本上做，并保留原 zip。
文章里的纹理重压（gltf-transform / gltfpack）针对单个 glb；本项目的 glb 在 zip
内部，需先解包、处理后重新打包（或重新导出 tileset）。
"""

import argparse
import collections
import json
import posixpath
import sys
import zipfile


def find_root_json(zf):
    """找到 zip 内最浅的 tileset.json（兼容带/不带顶层目录两种打包）。"""
    cands = [n for n in zf.namelist() if n.lower().endswith("tileset.json")]
    if not cands:
        raise SystemExit("zip 内没有 tileset.json")
    cands.sort(key=lambda n: (n.count("/"), len(n)))
    return cands[0]


def collect(zf, root_name, max_nodes=500000):
    """遍历 tileset 树（含外部 tileset），返回节点/深度/内容统计。"""
    names = set(zf.namelist())
    depth = collections.Counter()
    bytes_by_depth = collections.Counter()
    count_by_depth = collections.Counter()
    kinds = collections.Counter()
    sizes = []
    missing = []
    visited = set()
    max_depth = 0

    def load(name):
        try:
            return json.loads(zf.read(name).decode("utf-8"))
        except Exception as exc:  # 坏 JSON 必须报出来，静默会让"少一层"无从解释
            print("  [!] 无法解析 %s: %s" % (name, exc), file=sys.stderr)
            return None

    root = load(root_name)
    if not root:
        return None

    # 每个待处理节点：(所在目录, 节点, 深度)
    stack = [(posixpath.dirname(root_name), root.get("root"), 0)]
    nodes = 0
    while stack:
        dirpath, node, d = stack.pop()
        if not node:
            continue
        nodes += 1
        if nodes > max_nodes:
            print("  [!] 节点数超过 %d，提前停止" % max_nodes, file=sys.stderr)
            break
        depth[d] += 1
        max_depth = max(max_depth, d)

        content = node.get("content") or {}
        uri = content.get("uri")
        if uri:
            p = posixpath.normpath(posixpath.join(dirpath, uri))
            if uri.lower().endswith(".json"):
                if p not in visited:
                    visited.add(p)
                    if p not in names:
                        missing.append(p)
                    else:
                        sub = load(p)
                        if sub:
                            stack.append((posixpath.dirname(p), sub.get("root"), d))
            else:
                kind = uri.rsplit(".", 1)[-1].lower()
                if p in names:
                    size = zf.getinfo(p).file_size
                    kinds[kind] += 1
                    sizes.append(size)
                    bytes_by_depth[d] += size
                    count_by_depth[d] += 1
                else:
                    missing.append(p)

        for child in node.get("children") or []:
            stack.append((dirpath, child, d + 1))

    return {
        "nodes": nodes,
        "depth": depth,
        "bytes_by_depth": bytes_by_depth,
        "count_by_depth": count_by_depth,
        "kinds": kinds,
        "sizes": sizes,
        "missing": missing,
        "max_depth": max_depth,
        "externals": len(visited),
    }


def report(st):
    total = sum(st["sizes"])
    avg = total / len(st["sizes"]) if st["sizes"] else 0
    mx = max(st["sizes"]) if st["sizes"] else 0
    print("=== 3D Tiles 体检报告 ===")
    print("瓦片节点          : %d" % st["nodes"])
    print("外部 tileset 引用 : %d" % st["externals"])
    print("内容文件          : %d  %s" % (len(st["sizes"]), dict(st["kinds"])))
    print("内容总大小        : %.2f MB" % (total / 1048576.0))
    print("平均单块          : %.1f KB" % (avg / 1024.0))
    print("最大单块          : %.2f MB" % (mx / 1048576.0))
    print("最大层级          : L%d" % st["max_depth"])
    if st["missing"]:
        print("缺失引用          : %d 个（例：%s）" %
              (len(st["missing"]), st["missing"][0]))
    print("\n层级分布：")
    print("  深度   节点数    内容体积     平均单块")
    for d in sorted(st["depth"]):
        cnt = st["depth"][d]
        b = st["bytes_by_depth"].get(d, 0)
        c = st["count_by_depth"].get(d, 0)
        avg_d = (b / c / 1024.0) if c else 0
        print("  L%-5d %-9d %-12s %s" % (d, cnt, "%.2f MB" % (b / 1048576.0),
                                        ("%.1f KB" % avg_d) if c else "-"))
    print("\n诊断：")
    if avg > 5 * 1024 * 1024:
        print("  [!] 平均单块 > 5 MB：单次请求过久，建议降纹理或切细。")
    elif avg and avg < 200 * 1024:
        print("  [!] 平均单块 < 200 KB：瓦片过碎，请求数会爆炸，建议合并/减少分块。")
    else:
        print("  [ok] 平均单块在 200 KB - 5 MB 之间，粒度合理。")
    if st["max_depth"] > 5:
        print("  [!] 层级 L%d 超过 5 层：最深层在 Web 端很少用到，建议评估剪枝。" %
              st["max_depth"])
    if st["externals"] > 1000:
        print("  [!] 外部 tileset 引用 %d 个：典型的“深链 + 碎片化”，"
              "优先考虑合并/减少目录层级。" % st["externals"])
    if not st["sizes"]:
        print("  [!] 没有找到任何内容文件，检查 content.uri 与 zip 内路径。")
    if st["missing"]:
        print("  [!] 有 %d 个内容引用在 zip 内不存在，先修路径再谈优化。" %
              len(st["missing"]))


def prune_dry_run(st, keep_depth):
    """按外部 tileset 的节点深度做剪枝演练（不修改 zip）。"""
    removed_nodes = 0
    removed_bytes = 0
    for d in sorted(st["depth"]):
        if d > keep_depth:
            removed_nodes += st["depth"][d]
            removed_bytes += st["bytes_by_depth"].get(d, 0)
    print("\n=== 剪枝演练（保留 L0-%d）===" % keep_depth)
    print("可移除节点        : %d" % removed_nodes)
    print("可释放内容体积    : %.2f MB" % (removed_bytes / 1048576.0))
    print("提示：外部 tileset 的深度是逻辑深度；真正剪枝需要在离线副本上"
          "删除对应子节点并重写 tileset.json，建议先用 --top 看清最深层。")


def main():
    ap = argparse.ArgumentParser(description="3D Tiles zip 体检（只读）")
    ap.add_argument("--zip", required=True, help="tiles3d_db 下的 .zip 路径")
    ap.add_argument("--top", type=int, default=0, help="只显示最大的 N 个内容文件")
    ap.add_argument("--prune-depth", type=int, default=None,
                    help="演练保留到该层级可释放多少（不修改文件）")
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as zf:
        root_name = find_root_json(zf)
        print("zip              : %s" % args.zip)
        print("根 tileset       : %s" % root_name)
        st = collect(zf, root_name)
        if not st:
            return 2
        report(st)
        if args.top:
            biggest = sorted(zf.namelist(),
                             key=lambda n: zf.getinfo(n).file_size,
                             reverse=True)[:args.top]
            print("\n最大的 %d 个 zip 条目：" % args.top)
            for n in biggest:
                print("  %8.2f MB  %s" % (zf.getinfo(n).file_size / 1048576.0, n))
        if args.prune_depth is not None:
            prune_dry_run(st, args.prune_depth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
