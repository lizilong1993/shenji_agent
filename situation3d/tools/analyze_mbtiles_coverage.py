#!/usr/bin/env python3
"""从 mbtiles 算出**逐层实测覆盖矩形**，写进 manifest 的 `offline_detail_imagery`。

为什么需要单独一段：`coverage.imagery.per_zoom` 描述的是**静态瓦片树**
（`web/tiles/imagery/`，Esri 那批），而本次发布的谷歌卫星瓦片**不落文件**，
活在 mbtiles 里（`tiles_db/*.mbtiles`，由 serve.py 的 `/tiles_db/` 直发）。
两者是不同的数据面，前端要各按各的矩形去请求，否则会在没有瓦片的区域刷 404。

为什么逐层一个矩形：实测各层范围差得远（z14 覆盖 116.32-121.62E，z16 只有
120.03-120.73E，z17-z20 更是只有 120.11-120.24E 那一小块），用一个矩形包住
整段会让 Cesium 在没有瓦片的地方持续请求。

用法（远程容器）：
    python3 tools/analyze_mbtiles_coverage.py \
        --db tiles_db/google_satellite.mbtiles --source google_satellite \
        --min-zoom 14 --manifest web/tiles/manifest.json --write
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sqlite3
import sys


def bbox_of(z, pairs):
    """(x, y_xyz) 集合 → [西, 南, 东, 北]。"""
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    n = 2.0 ** z
    west = min(xs) / n * 360.0 - 180.0
    east = (max(xs) + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * min(ys) / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (max(ys) + 1) / n))))
    return [round(west, 6), round(south, 6), round(east, 6), round(north, 6)]


def analyze(db, min_zoom, max_zoom):
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    cur = con.cursor()
    per = cur.execute(
        "SELECT zoom_level, count(*) FROM tiles WHERE zoom_level >= ? AND zoom_level <= ?"
        " GROUP BY zoom_level ORDER BY zoom_level",
        (min_zoom, max_zoom)).fetchall()
    bands = []
    for z, n in per:
        rows = cur.execute(
            "SELECT tile_column, tile_row FROM tiles WHERE zoom_level=?", (z,)).fetchall()
        pairs = [(x, (1 << z) - 1 - r) for x, r in rows]
        bbox = bbox_of(z, pairs)
        cols = len({x for x, _ in pairs})
        rows_n = len({y for _, y in pairs})
        fill = n / float(max(cols * rows_n, 1))
        bands.append({"zoom": z, "tiles": n, "bbox": bbox,
                      "grid": [cols, rows_n], "fill": round(fill, 4)})
    con.close()
    return bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--source", required=True, help="库名（不含 .mbtiles），用于 URL 模板")
    ap.add_argument("--min-zoom", type=int, default=14)
    ap.add_argument("--max-zoom", type=int, default=24)
    ap.add_argument("--url-template", default=None)
    ap.add_argument("--manifest", default="web/tiles/manifest.json")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.db):
        sys.stderr.write("找不到库：%s\n" % args.db)
        return 2
    bands = analyze(args.db, args.min_zoom, args.max_zoom)
    if not bands:
        sys.stderr.write("该库里 z%d-z%d 没有瓦片\n" % (args.min_zoom, args.max_zoom))
        return 2

    url = args.url_template or ("/tiles_db/%s/{z}/{x}/{y}.jpg" % args.source)
    block = {
        "source": args.source,
        "url_template": url,
        "min_zoom": bands[0]["zoom"],
        "max_zoom": bands[-1]["zoom"],
        "tiles": sum(b["tiles"] for b in bands),
        "bands": bands,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": ("这些瓦片**不在文件树里**，由 serve.py 从 mbtiles 直发"
                 "（/tiles_db/<source>/<z>/<x>/<y>.jpg）。bands 是**逐层**实测覆盖矩形："
                 "各层范围差异很大，用一个矩形包住整段会在没有瓦片处刷 404。"),
    }

    print("== %s：z%d-z%d，共 %d 张" % (args.source, block["min_zoom"], block["max_zoom"],
                                        block["tiles"]))
    print("   %-4s %8s %-34s %-10s %6s" % ("z", "瓦片", "矩形[西,南,东,北]", "网格", "填充率"))
    for b in bands:
        print("   z%-3d %8d %-34s %-10s %5.1f%%"
              % (b["zoom"], b["tiles"], str(b["bbox"]),
                 "%dx%d" % tuple(b["grid"]), b["fill"] * 100))

    if not args.write:
        print("（未写回 manifest；加 --write 生效）")
        return 0
    with open(args.manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    man["offline_detail_imagery"] = block
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)
    print("manifest 已更新：offline_detail_imagery（其余键未动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
