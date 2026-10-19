#!/usr/bin/env python3
"""GeoTIFF 与已发布瓦片是否**同一份影像**的像素级抽查（瓦片从 mbtiles 读）。

结论的用途：三个 GeoTIFF 是**按瓦片对齐**的墨卡托拼接图（z14 正好 241x241 张、
z15 170x328、z16 129x269），与库里的原始瓦片同源同层。所以"把 tif 也变成瓦片"
有两种做法：
  * 从 tif 重新编码一遍覆盖上去 —— **二次压缩**，只会让图变差；
  * 发布**原始瓦片**（无损），再用 tif 去验证两者确实是同一份影像。
本脚本做后者。抽样比对在 RGB 字节空间做：tif 用 GDAL 读窗口，瓦片经 /vsimem 解 JPEG。
MAE 在 1-3 量级是 JPEG 解码噪声；>15 说明不是同一份图，那才需要以 tif 为准重切。

另外顺带回答一个**只有跑起来才会问的问题**：tif 的"空白格"是上游没有影像，
还是下载漏了？脚本对 tif 网格里 mbtiles 缺失的格子也抽样，报出它们是否纯白填充。

用法（远端，用系统 python3，它有 osgeo）：
    /usr/bin/python3 tools/compare_tif_tiles.py --tif tiles_db/../tif/X.tif --zoom 14 \\
        --mbtiles tiles_db/google_satellite.mbtiles --samples 60
"""
from __future__ import annotations

import argparse
import os
import random
import sqlite3
import statistics
import sys

from osgeo import gdal

gdal.UseExceptions()


def tile_origin(z, gt, world=20037508.342789244):
    span = 2.0 * world / (1 << z)
    return int(round((gt[0] + world) / span)), int(round((world - gt[3]) / span))


def open_tile_bytes(data):
    """把内存里的瓦片字节交给 GDAL 解（/vsimem 是进程内的内存文件系统）。"""
    path = "/vsimem/_tile_%d" % id(data)
    gdal.FileFromMemBuffer(path, data)
    ds = gdal.Open(path)
    return ds, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--zoom", type=int, required=True)
    ap.add_argument("--mbtiles", required=True)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()

    src = gdal.Open(args.tif)
    gt = src.GetGeoTransform()
    W, H = src.RasterXSize, src.RasterYSize
    x0, y0 = tile_origin(args.zoom, gt)
    nx, ny = W // 256, H // 256
    # 单波段（DEM/掩膜）没有 RGB：按实际波段数取，别去读不存在的 2/3 波段
    # （原先硬编码 1,2,3，遇到 DEM GeoTIFF 会直接崩在 ReadRaster）。
    n_tif = src.RasterCount or 1
    band_idx = tuple(range(1, min(3, n_tif) + 1))
    bands = [src.GetRasterBand(k) for k in band_idx]
    if n_tif < 3:
        print("提示：tif 只有 %d 个波段（单波段产品），只比对第 1 波段；"
              "与彩色瓦片的结论仅供参考。" % n_tif)
    print("tif %s：%dx%d px → z%d 起点瓦片 (%d,%d)，网格 %dx%d"
          % (os.path.basename(args.tif), W, H, args.zoom, x0, y0, nx, ny))

    con = sqlite3.connect("file:%s?mode=ro" % args.mbtiles, uri=True)
    cur = con.cursor()
    have = set((x, (1 << args.zoom) - 1 - r) for x, r in
               cur.execute("SELECT tile_column, tile_row FROM tiles WHERE zoom_level=?",
                           (args.zoom,)).fetchall())

    cells = [(x0 + i, y0 + j) for i in range(nx) for j in range(ny)]
    random.seed(args.seed)
    random.shuffle(cells)
    present = [c for c in cells if c in have][: args.samples]
    absent = [c for c in cells if c not in have][: args.samples]

    def tif_window(x, y):
        return [b.ReadRaster((x - x0) * 256, (y - y0) * 256, 256, 256, buf_type=gdal.GDT_Byte)
                for b in bands]

    # ── 1) 已有瓦片：tif 与库里的瓦片是否同一份影像 ──
    stats = []
    for x, y in present:
        row = cur.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=?"
                          " AND tile_row=?", (args.zoom, x, (1 << args.zoom) - 1 - y)).fetchone()
        if not row:
            continue
        ds, path = open_tile_bytes(row[0])
        try:
            tb = []
            for k in band_idx:
                b = ds.GetRasterBand(k)
                tb.append(b.ReadRaster(buf_type=gdal.GDT_Byte) if b is not None else None)
        finally:
            ds = None
            gdal.Unlink(path)
        tw = tif_window(x, y)
        total = count = 0
        mx = 0
        for a, b in zip(tw, tb):
            if a is None or b is None or len(a) != len(b):
                continue
            for pa, pb in zip(a, b):
                d = abs(pa - pb)
                total += d
                count += 1
                if d > mx:
                    mx = d
        if count:
            stats.append((total / count, mx, x, y))

    if not stats:
        print("没有可比对的瓦片")
        return 1
    maes = sorted(s[0] for s in stats)
    over = sum(1 for m in maes if m > args.threshold)
    print("① 与库内原始瓦片比对 %d 张：MAE 中位 %.2f / 均值 %.2f / 最大 %.2f；超阈 %.0f 的 %d 张"
          % (len(maes), maes[len(maes) // 2], sum(maes) / len(maes), maes[-1],
             args.threshold, over))
    worst = sorted(stats, reverse=True)[:3]
    for mae, mx, x, y in worst:
        print("     最差：z%d/%d/%d MAE %.2f 最大差 %d" % (args.zoom, x, y, mae, mx))

    # ── 2) 库里没有的格子：tif 那边是空白填充，还是真有影像？ ──
    white = 0
    nonwhite = []
    for x, y in absent:
        vals = []
        for b in bands:
            vals += list(b.ReadRaster((x - x0) * 256, (y - y0) * 256, 256, 256,
                                      buf_type=gdal.GDT_Byte))
        m = sum(vals) / len(vals)
        sd = statistics.pstdev(vals)
        if m >= 254.999 and sd < 0.01:
            white += 1
        else:
            nonwhite.append((x, y, m, sd))
    print("② 库里缺失的 %d 格：纯白填充 %d 格；非白 %d 格 %s"
          % (len(absent), white, len(nonwhite),
             ("（例：%s）" % str(nonwhite[:3])) if nonwhite else "—— 即 tif 的空白就是上游没有影像"))

    con.close()
    if over == 0 and not nonwhite:
        print("结论：tif 与已发布瓦片**同源**；空白格是上游无影像，不是下载遗漏。")
        return 0
    if over == 0:
        print("结论：影像同源；但有 %d 格 tif 有内容而库里没有 —— 需要判断是否补切。" % len(nonwhite))
        return 1
    print("结论：**存在内容差异**（%d 张超阈）—— 需要人工判断以哪一侧为准。" % over)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
