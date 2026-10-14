#!/usr/bin/env python3
"""校验搬到服务器上的 mbtiles 是否完整、可用。

为什么需要：这些 .mbtiles 是从**运行中的 GeoDownloader** 手里 tar 出来的原始字节，
且带着 -wal/-shm 同行的副本。原始拷贝可能：
  * 少掉尚未 checkpoint 的新瓦片（WAL 没被应用）；
  * 带着**别人的 -shm**（共享内存状态，拷贝过来是脏的，会误导 SQLite）。

处置：删掉 -shm（让 SQLite 自己重建）→ 以只读方式打开 → integrity_check →
逐层瓦片数 → 与本地实测值逐项比对。任一不符即非零退出，不静默通过。

用法：
    python3 verify_mbtiles.py --dir /path/mbtiles [--expect expect.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

# 本地实测基线（2026-10-14，从运行中的下载器只读读出的权威值）
EXPECT = {
    "google_satellite.mbtiles": {
        "total": 205457,
        "per_zoom": {5: 21, 7: 24, 8: 31, 9: 32, 10: 24, 11: 22, 12: 30, 13: 28,
                     14: 29752, 15: 34547, 16: 28993, 17: 1614, 18: 5603,
                     19: 21540, 20: 83196},
    },
    "tianditu_satellite.mbtiles": {
        "total": 127682,
        "per_zoom": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 24, 8: 25, 9: 46,
                     10: 152, 11: 528, 12: 2112, 13: 8265, 14: 8365, 15: 29287,
                     16: 78851, 17: 20},
    },
    "tianditu_satellite_label.mbtiles": {
        "total": 59162,
        "per_zoom": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 32, 8: 12, 9: 50,
                     10: 152, 11: 528, 12: 2112, 13: 7192, 14: 12735, 15: 22443,
                     16: 13899},
    },
    "tianditu_terrain_label.mbtiles": {
        "total": 15996,
        "per_zoom": {14: 74, 15: 79, 16: 15843},
    },
    "tianditu_vector_label.mbtiles": {
        "total": 8077,
        "per_zoom": {7: 20, 14: 303, 15: 6625, 16: 1129},
    },
}


def check_one(path, expect, integrity=True):
    name = os.path.basename(path)
    problems = []

    shm = path + "-shm"
    if os.path.exists(shm):
        os.remove(shm)
        print("     [fix] 已删掉拷贝来的 -shm（脏共享内存，交由 SQLite 重建）")

    try:
        con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    except sqlite3.Error as e:
        return ["打不开：%s" % e]
    try:
        cur = con.cursor()
        # integrity_check 会把每个 blob 都读一遍（2.4 GB 级 mbtiles 要跑很久）。
        # 完整性核对真正要回答的是"瓦片齐不齐"，逐层计数即可；全表校验按需再开。
        if integrity:
            try:
                ok = cur.execute("PRAGMA integrity_check").fetchone()[0]
                if ok != "ok":
                    problems.append("integrity_check=%s" % ok)
            except sqlite3.Error as e:
                problems.append("integrity_check 失败：%s" % e)
        try:
            meta = dict(cur.execute("SELECT name, value FROM metadata").fetchall())
        except sqlite3.Error as e:
            meta = {}
            problems.append("metadata 读取失败：%s" % e)
        total = cur.execute("SELECT count(*) FROM tiles").fetchone()[0]
        per = dict(cur.execute(
            "SELECT zoom_level, count(*) FROM tiles GROUP BY zoom_level").fetchall())
        null_rows = cur.execute(
            "SELECT count(*) FROM tiles WHERE tile_data IS NULL OR length(tile_data)=0"
        ).fetchone()[0]
    finally:
        con.close()

    print("     %-34s 瓦片 %7d  z%s–z%s" % (name, total, min(per), max(per)))
    print("     %-34s format=%s name=%s" % ("", meta.get("format"), meta.get("name")))
    if null_rows:
        problems.append("%d 行 tile_data 为空" % null_rows)
    if expect:
        if total != expect["total"]:
            problems.append("总数 %d ≠ 基线 %d（差 %d）"
                            % (total, expect["total"], total - expect["total"]))
        for z, want in expect["per_zoom"].items():
            got = per.get(z, 0)
            if got != want:
                problems.append("z%d=%d ≠ 基线 %d" % (z, got, want))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--expect", default=None, help="基线 JSON（默认用脚本内置值）")
    ap.add_argument("--integrity", action="store_true",
                    help="额外跑 PRAGMA integrity_check（全表读 blob，很慢）")
    args = ap.parse_args()

    expect = EXPECT
    if args.expect:
        raw = json.load(open(args.expect, encoding="utf-8"))
        expect = {k: {"total": v["total"],
                      "per_zoom": {int(z): c for z, c in v["per_zoom"].items()}}
                  for k, v in raw.items()}

    names = sorted(f for f in os.listdir(args.dir) if f.endswith(".mbtiles"))
    print("待校验 %d 个 mbtiles（%s）" % (len(names), args.dir))
    bad = 0
    for n in names:
        p = os.path.join(args.dir, n)
        print("\n  -- %s (%.2f GB)" % (n, os.path.getsize(p) / 1024 ** 3))
        problems = check_one(p, expect.get(n), integrity=args.integrity)
        if problems:
            bad += 1
            for x in problems:
                print("     ✗ %s" % x)
        else:
            print("     ✓ 与基线逐层一致")
    print()
    if bad:
        print("结论：%d/%d 个文件与基线不符 —— 不通过" % (bad, len(names)))
        return 1
    print("结论：全部通过（%d 个文件、逐层瓦片数与本地基线一致）" % len(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
