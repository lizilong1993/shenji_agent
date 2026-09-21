#!/usr/bin/env python3
"""从实际瓦片算出影像/地形的真实覆盖范围，写回 manifest。

为什么需要
----------
"全球层 z0-6"这个名字来自抓取脚本的分级命名，但那批瓦片**只覆盖台海区域**——
实测 z0-z11 的瓦片数是 1/1/1/2/5/7/12/34/99/339/1242/4851，是一条金字塔，不是全球。
provider 若不声明 rectangle（=全球），Cesium 会为域外请求瓦片，
实测打开离线底图会看到 `imagery/2/3/0.jpg`、`1/1/1.jpg`、`6/54/25.jpg` 一连串 404。

为什么不能直接用并集
--------------------
低层级一个瓦片覆盖十几度，**个别离群瓦片就能把并集撑成全球**。实测 z6 里有
`(56,25)`、`(57,29)` 两张（属于热点框，不在台海金字塔里），并集直接变成
[-180,-85,180,85]。所以这里按"这一级里 x 列 / y 行出现的次数"过滤掉稀疏离群值，
再算包围盒。

用法（远程容器）：
    python3 tools/analyze_tile_coverage.py --manifest web/tiles/manifest.json --write
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

MIN_SUPPORT = 3     # 一条 x 列 / y 行至少有这么多瓦片才算"主体覆盖"


def tile_to_lonlat(z, x, y):
    n = 2.0 ** z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return west, south, east, north


def tiles_at(root, z):
    zdir = os.path.join(root, str(z))
    out = []
    if not os.path.isdir(zdir):
        return out
    for xname in os.listdir(zdir):
        if not xname.isdigit():
            continue
        xd = os.path.join(zdir, xname)
        if not os.path.isdir(xd):
            continue
        for f in os.listdir(xd):
            if f.endswith(EXT):
                out.append((int(xname), int(f[: -len(EXT)])))
    return out


def strict_box(z, pairs):
    """过滤稀疏离群值后的包围盒；返回 None 表示这一级没有主体覆盖。"""
    cx = Counter(x for x, _ in pairs)
    cy = Counter(y for _, y in pairs)
    xs = [x for x, n in cx.items() if n >= MIN_SUPPORT]
    ys = [y for y, n in cy.items() if n >= MIN_SUPPORT]
    if not xs or not ys:
        return None
    keep = [(x, y) for x, y in pairs if x in set(xs) and y in set(ys)]
    if not keep:
        return None
    w = min(tile_to_lonlat(z, x, y)[0] for x, y in keep)
    s = min(tile_to_lonlat(z, x, y)[1] for x, y in keep)
    e = max(tile_to_lonlat(z, x, y)[2] for x, y in keep)
    n = max(tile_to_lonlat(z, x, y)[3] for x, y in keep)
    return (w, s, e, n, len(keep), len(pairs))


def scan(root, ext):
    global EXT
    EXT = ext
    per_z = {}
    for zname in sorted(os.listdir(root), key=lambda v: int(v) if v.isdigit() else 0):
        if not zname.isdigit():
            continue
        z = int(zname)
        pairs = tiles_at(root, z)
        if pairs:
            per_z[z] = pairs
    return per_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="web/tiles")
    ap.add_argument("--manifest", default="web/tiles/manifest.json")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    results = {}
    for name, sub, ext in (("imagery", "imagery", ".jpg"), ("terrain", "terrain", ".png")):
        root = os.path.join(args.tiles, sub)
        if not os.path.isdir(root):
            continue
        per_z = scan(root, ext)
        boxes = {}
        for z, pairs in per_z.items():
            b = strict_box(z, pairs)
            if b:
                boxes[z] = b
        if not boxes:
            continue
        w = min(b[0] for b in boxes.values())
        s = min(b[1] for b in boxes.values())
        e = max(b[2] for b in boxes.values())
        n = max(b[3] for b in boxes.values())
        # 底图矩形取**最细那两级**的范围，而不是全部层级的并集。
        #
        # 高层级才带热点框（关岛/冲绳/羽田），所以并集是 108.8–145°E；但 Cesium 会为
        # 同一个地理位置跨层级请求瓦片，如果一个视野只落在 z11 的范围内，它请求的
        # z0–z10 瓦片也就都落在这个范围内。用最细层级算矩形，可以把低层级那些
        # 只服务热点的离群瓦片排除掉，同时不牺牲任何真实视野的覆盖。
        detail_z = sorted(boxes)[-2:]
        dw = min(boxes[z][0] for z in detail_z)
        ds = min(boxes[z][1] for z in detail_z)
        de = max(boxes[z][2] for z in detail_z)
        dn = max(boxes[z][3] for z in detail_z)
        total = sum(len(p) for p in per_z.values())
        kept = sum(b[4] for b in boxes.values())
        results[name] = {
            "rectangle": [round(w, 6), round(s, 6), round(e, 6), round(n, 6)],
            "basemap_rectangle": [round(dw, 6), round(ds, 6), round(de, 6), round(dn, 6)],
            "basemap_from_zoom": detail_z,
            "zoom_min": min(boxes),
            "zoom_max": max(boxes),
            "tiles": total,
            "tiles_in_region": kept,
            "per_zoom": {str(z): len(per_z[z]) for z in sorted(per_z)},
        }
        print("== %s" % name)
        print("   全部层级并集 [%.4f, %.4f, %.4f, %.4f]  z%s-z%s"
              % (w, s, e, n, min(boxes), max(boxes)))
        print("   底图矩形（取自 z%s）[%.4f, %.4f, %.4f, %.4f]"
              % (detail_z, dw, ds, de, dn))
        print("   tiles %d, in-region %d" % (total, kept))
        for z in sorted(boxes):
            b = boxes[z]
            print("   z%-2d kept %4d/%-4d  [%.3f,%.3f,%.3f,%.3f]"
                  % (z, b[4], b[5], b[0], b[1], b[2], b[3]))
        outliers = total - kept
        if outliers:
            print("   离群（不计入 range）: %d 张" % outliers)

    if not args.write:
        print("\n（未写回；加 --write 生效）")
        return 0

    with open(args.manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    man["coverage"] = results
    man["coverage_note"] = (
        "coverage.*.rectangle 由 tools/analyze_tile_coverage.py 从**实际瓦片**算出（并剔除"
        "稀疏离群瓦片），不是抓取时的名义范围。影像层曾按'全球 z0-6'声明覆盖，但瓦片只覆盖"
        "台海区域，声明全球会让域外请求全部 404。")
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)
    print("\nmanifest 已更新")
    for k, v in results.items():
        print("  %s: %s z%s-z%s" % (k, v["rectangle"], v["zoom_min"], v["zoom_max"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
