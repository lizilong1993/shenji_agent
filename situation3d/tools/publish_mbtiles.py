#!/usr/bin/env python3
"""把 mbtiles 发布成 situation3d 的离线瓦片树：`imagery/{z}/{x}/{y}.jpg`。

三个必须说清的点
----------------
1. **坐标翻转**：mbtiles 的 `tile_row` 是 TMS 行（自南向北），前端要的是 XYZ
   （自北向南）——`y_xyz = 2^z - 1 - tile_row`。翻错的后果是整幅图南北镜像，
   而图不会报错，只是"看起来不对"。
2. **覆盖要留备份**：树里已有的瓦片可能是别的来源（Esri）。覆盖前把旧文件
   `move` 到备份目录（保持 z/x/y 结构），保证可回滚；内容逐字节相同的直接跳过，
   不制造无意义的备份。
3. **可验证**：`--verify` 反向逐块比对（mbtiles vs 树），输出 missing/different。
   发布声明必须能被这个模式复核，否则"发布成功"就只是一句话。

用法：
    python3 publish_mbtiles.py --mbtiles X.mbtiles --out web/tiles/imagery \\
        --backup /path/backup --min-zoom 14
    python3 publish_mbtiles.py --mbtiles X.mbtiles --out web/tiles/imagery --verify
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time


def xyz_y(z, row):
    return (1 << z) - 1 - row


def ensure_parent(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)


def run_publish(args):
    con = sqlite3.connect("file:%s?mode=ro" % args.mbtiles, uri=True)
    cur = con.cursor()
    where = "WHERE zoom_level >= ?"
    params = [args.min_zoom]
    if args.max_zoom is not None:
        where += " AND zoom_level <= ?"
        params.append(args.max_zoom)
    total = cur.execute("SELECT count(*) FROM tiles " + where, params).fetchone()[0]
    print("mbtiles 内符合层级条件的瓦片：%d 张（z%s–z%s）"
          % (total, args.min_zoom, args.max_zoom if args.max_zoom is not None else "max"))

    if args.dry_run:
        per = cur.execute(
            "SELECT zoom_level, count(*) FROM tiles " + where +
            " GROUP BY zoom_level ORDER BY zoom_level", params).fetchall()
        for z, n in per:
            print("   z%-3d %7d" % (z, n))
        con.close()
        print("干跑结束（未写任何文件）")
        return 0

    cur2 = con.cursor()
    rows = cur2.execute(
        "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles " + where +
        " ORDER BY zoom_level, tile_column, tile_row", params)

    n_new = n_same = n_replaced = 0
    bytes_new = bytes_replaced = 0
    t0 = time.time()
    done = 0
    for z, x, row, data in rows:
        y = xyz_y(z, row)
        dst = os.path.join(args.out, str(z), str(x), "%d.%s" % (y, args.ext))
        exists = os.path.exists(dst)
        if exists:
            try:
                with open(dst, "rb") as fh:
                    old = fh.read()
            except OSError:
                old = None
            if old == data:
                n_same += 1
                done += 1
                continue
            if args.backup:
                bdst = os.path.join(args.backup, str(z), str(x), "%d.%s" % (y, args.ext))
                ensure_parent(bdst)
                os.replace(dst, bdst)
                bytes_replaced += len(old or b"")
            n_replaced += 1
        ensure_parent(dst)
        tmp = dst + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dst)
        bytes_new += len(data)
        if not exists:
            n_new += 1
        done += 1
        if done % 20000 == 0:
            print("   … %d/%d  (新增 %d / 覆盖 %d / 相同 %d)  %.1f 分钟"
                  % (done, total, n_new, n_replaced, n_same, (time.time() - t0) / 60))
    con.close()

    print()
    print("发布完成：处理 %d 张" % done)
    print("  新增      %7d 张  %10.1f MB" % (n_new, bytes_new / 1024 ** 2))
    print("  覆盖      %7d 张  （旧内容已备份 %d 张 / %.1f MB）"
          % (n_replaced, n_replaced, bytes_replaced / 1024 ** 2))
    print("  已相同    %7d 张  （未动）" % n_same)
    print("  耗时      %.1f 分钟" % ((time.time() - t0) / 60))
    if args.backup and n_replaced:
        print("  备份目录  %s" % args.backup)
    print("  产出目录  %s" % args.out)
    return 0


def run_verify(args):
    con = sqlite3.connect("file:%s?mode=ro" % args.mbtiles, uri=True)
    cur = con.cursor()
    where = "WHERE zoom_level >= ?"
    params = [args.min_zoom]
    if args.max_zoom is not None:
        where += " AND zoom_level <= ?"
        params.append(args.max_zoom)
    total = cur.execute("SELECT count(*) FROM tiles " + where, params).fetchone()[0]
    print("反向核对：mbtiles %d 张 → 树 %s" % (total, args.out))

    cur2 = con.cursor()
    rows = cur2.execute(
        "SELECT zoom_level, tile_column, tile_row, tile_data FROM tiles " + where +
        " ORDER BY zoom_level, tile_column, tile_row", params)
    missing, different, same = 0, 0, 0
    examples = []
    for z, x, row, data in rows:
        y = xyz_y(z, row)
        dst = os.path.join(args.out, str(z), str(x), "%d.%s" % (y, args.ext))
        try:
            with open(dst, "rb") as fh:
                got = fh.read()
        except OSError:
            missing += 1
            if len(examples) < 5:
                examples.append("miss %d/%d/%d" % (z, x, y))
            continue
        if got == data:
            same += 1
        else:
            different += 1
            if len(examples) < 5:
                examples.append("diff %d/%d/%d (%d B vs %d B)"
                                % (z, x, y, len(data), len(got)))
    con.close()
    print("  一致 %d / 缺失 %d / 内容不同 %d" % (same, missing, different))
    for e in examples:
        print("    %s" % e)
    if missing or different:
        print("结论：**不通过**")
        return 1
    print("结论：逐块一致，发布完整")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbtiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backup", default=None)
    ap.add_argument("--ext", default="jpg")
    ap.add_argument("--min-zoom", type=int, default=0)
    ap.add_argument("--max-zoom", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    return run_verify(args) if args.verify else run_publish(args)


if __name__ == "__main__":
    raise SystemExit(main())
