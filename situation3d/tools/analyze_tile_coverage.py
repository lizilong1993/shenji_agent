#!/usr/bin/env python3
"""从实际瓦片算出影像/地形的真实覆盖范围，写回 manifest。

为什么需要
----------
地基层的 provider 如果不声明 rectangle（=全球），Cesium 会为域外请求瓦片，
实测打开离线底图会看到 `imagery/2/3/0.jpg`、`1/1/1.jpg`、`6/54/25.jpg` 一连串 404。

为什么分两段
------------
补齐了 z0-z6 的全球瓦片之后，覆盖形态变成"**低层全球 + 高层区域**"：
  z0-z6   全球有瓦片（补齐过）
  z7+     只有台海与热点框（一条区域金字塔）
所以不能只算一个矩形。这里对每一级分别算：
  - `full_rectangle`：该级全部瓦片的并集（低层级是全球）
  - `regional_rectangle`：剔除稀疏离群瓦片后的包围盒（高层级是区域）
前端据此对低层级用全球矩形、对高层级用区域矩形。

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

MIN_SUPPORT = 3

WORLD = [-180.0, -85.051129, 180.0, 85.051129]


def tile_to_lonlat(z, x, y):
    n = 2.0 ** z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return west, south, east, north


def box_of(z, pairs):
    if not pairs:
        return None
    w = min(tile_to_lonlat(z, x, y)[0] for x, y in pairs)
    s = min(tile_to_lonlat(z, x, y)[1] for x, y in pairs)
    e = max(tile_to_lonlat(z, x, y)[2] for x, y in pairs)
    n = max(tile_to_lonlat(z, x, y)[3] for x, y in pairs)
    return [round(w, 6), round(s, 6), round(e, 6), round(n, 6)]


def regional_box_of(z, pairs):
    """剔除稀疏离群瓦片后的包围盒。

    低层级一个瓦片覆盖十几度，个别离群瓦片就能把并集撑成全球——实测 z6 里只有
    两张的 (56,25)/(57,29)（属热点框）就会造成这个后果。所以按"这一级里 x 列 /
    y 行出现的次数"过滤。
    """
    cx = Counter(x for x, _ in pairs)
    cy = Counter(y for _, y in pairs)
    xs = {x for x, n in cx.items() if n >= MIN_SUPPORT}
    ys = {y for y, n in cy.items() if n >= MIN_SUPPORT}
    keep = [(x, y) for x, y in pairs if x in xs and y in ys]
    if not keep:
        return None
    return box_of(z, keep)


def read_levels(root, ext):
    out = {}
    if not os.path.isdir(root):
        return out
    for zname in os.listdir(root):
        zdir = os.path.join(root, zname)
        if not zname.isdigit() or not os.path.isdir(zdir):
            continue
        pairs = []
        for xname in os.listdir(zdir):
            if not xname.isdigit():
                continue
            xd = os.path.join(zdir, xname)
            if not os.path.isdir(xd):
                continue
            for f in os.listdir(xd):
                if f.endswith(ext):
                    pairs.append((int(xname), int(f[: -len(ext)])))
        if pairs:
            out[int(zname)] = pairs
    return out


def analyse(root, ext, label):
    levels = read_levels(root, ext)
    if not levels:
        return None
    per_zoom = {}
    full = {}
    regional = {}
    for z, pairs in sorted(levels.items()):
        fb = box_of(z, pairs)
        rb = regional_box_of(z, pairs) or fb
        per_zoom[str(z)] = {
            "tiles": len(pairs),
            "full_rectangle": fb,
            "regional_rectangle": rb,
            "full_is_world": fb == WORLD,
        }
        full[z] = fb
        regional[z] = rb

    ws = sorted(full)
    zmax = ws[-1]
    # 底基层：低层级全球、高层级区域 → 用两个矩形描述
    global_max = 0
    for z in ws:
        if per_zoom[str(z)]["full_is_world"]:
            global_max = z
    region_min = global_max + 1

    reg_boxes = [regional[z] for z in ws if z >= region_min] or [regional[zmax]]
    region_rect = [
        round(min(b[0] for b in reg_boxes), 6),
        round(min(b[1] for b in reg_boxes), 6),
        round(max(b[2] for b in reg_boxes), 6),
        round(max(b[3] for b in reg_boxes), 6),
    ]
    # 最细两级的范围：用于"底图只需要覆盖真实视野"的场景
    detail = [z for z in ws if z >= max(region_min, zmax - 1)]
    detail_boxes = [regional[z] for z in detail] or reg_boxes
    detail_rect = [
        round(min(b[0] for b in detail_boxes), 6),
        round(min(b[1] for b in detail_boxes), 6),
        round(max(b[2] for b in detail_boxes), 6),
        round(max(b[3] for b in detail_boxes), 6),
    ]

    result = {
        "zoom_min": ws[0],
        "zoom_max": zmax,
        "tiles": sum(len(p) for p in levels.values()),
        "global_max_zoom": global_max,
        "global_rectangle": WORLD if global_max else None,
        "region_min_zoom": region_min if region_min <= zmax else None,
        "region_rectangle": region_rect if region_min <= zmax else None,
        "basemap_rectangle": detail_rect,
        "basemap_from_zoom": detail,
        "per_zoom": per_zoom,
    }
    print("== %s" % label)
    print("   层级 z%s-z%s  共 %d 张" % (ws[0], zmax, result["tiles"]))
    print("   全球层: z0-z%d（矩形全球）" % global_max)
    if result["region_rectangle"]:
        print("   区域层: z%d-z%d  矩形 %s" % (region_min, zmax, result["region_rectangle"]))
    print("   底图矩形（取自 z%s）: %s" % (detail, detail_rect))
    for z in ws:
        v = per_zoom[str(z)]
        print("     z%-2d n=%-5d full=%s%s regional=%s"
              % (z, v["tiles"], "WORLD" if v["full_is_world"] else v["full_rectangle"],
                 "" if v["full_is_world"] else "", v["regional_rectangle"]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="web/tiles")
    ap.add_argument("--manifest", default="web/tiles/manifest.json")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    results = {}
    for kind, sub, ext in (("imagery", "imagery", ".jpg"), ("terrain", "terrain", ".png")):
        r = analyse(os.path.join(args.tiles, sub), ext, kind)
        if r:
            results[kind] = r

    if not args.write:
        print("\n（未写回；加 --write 生效）")
        return 0

    with open(args.manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    man["coverage"] = results
    man["coverage_note"] = (
        "coverage.* 由 tools/analyze_tile_coverage.py 从**实际瓦片**算出，"
        "剔除稀疏离群瓦片，并把'低层全球 + 高层区域'两段分开描述。"
        "底部 provider 曾按名义'全球 z0-6'声明覆盖，实际瓦片只覆盖台海，"
        "导致域外请求全部 404。")
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)
    print("\nmanifest 已更新")
    for k, v in results.items():
        print("  %s: global z0-z%s, region z%s-z%s %s"
              % (k, v["global_max_zoom"], v["region_min_zoom"], v["zoom_max"],
                 v["region_rectangle"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
