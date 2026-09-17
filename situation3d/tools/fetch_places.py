#!/usr/bin/env python3
"""把 Natural Earth 的居民点数据抓成前端可直接用的离线地名库。

数据源
------
**Natural Earth 1:10m Populated Places**（`ne_10m_populated_places`）。

选它的理由：

* **公有领域**（Natural Earth 明确声明无需署名、可自由使用），不引入授权风险；
* 覆盖全球约 7300 个居民点，含 `SCALERANK`（0–10 的重要性分级）——这正是分级
  显示需要的 LOD 提示，不必自己按人口重新分档；
* 含 `NAME_ZH` 中文字段，省去翻译；
* 体量小（解压后 shp+dbf 约 2 MB），转换后 JSON 约 600 KB，gzip 后约 200 KB。

为什么自己写 shapefile 解析
---------------------------
远程容器是 python3.8 且**没有 pip**，装不了 `pyshp`/`geopandas`。而 shapefile 与
dBASE 都是很简单的二进制格式，用标准库读一百多行就够，也免得多一份依赖。

输出
----
`web/tiles/places/places.json`，紧凑键名（7300 条 × 约 80 字节）：

    {"schema": "...", "source": "...", "generated": "...", "places": [
      {"n": "北京", "e": "Beijing", "x": 116.407, "y": 39.905,
       "r": 0, "p": 21516000, "c": "中国", "a": "北京市", "k": "Admin-0 capital"},
      ...]}

键：n=中文名 e=英文名 x=经度 y=纬度 r=scalerank p=人口 c=国家 a=一级行政区 k=类别

用法（仓库根目录）：
    python3 situation3d/tools/fetch_places.py
    python3 situation3d/tools/fetch_places.py --bbox 108,18,146,36   # 只保留一个范围
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "web", "tiles", "places")
OUT_FILE = os.path.join(OUT_DIR, "places.json")

SOURCE_URL = ("https://naciscdn.org/naturalearth/10m/cultural/"
              "ne_10m_populated_places.zip")
SOURCE_NAME = "Natural Earth 1:10m Populated Places"
LICENCE = "Public domain (Natural Earth) -- no attribution required."

# 想要的中文名列（不同版本的 NE 字段名略有差异，按优先级取第一个存在的）
NAME_ZH_CANDIDATES = ["NAME_ZH", "NAME_ZHT", "NAME_ZH_HANS"]


# ---------------------------------------------------------------------------
# shapefile / dBASE 读取（标准库实现）
# ---------------------------------------------------------------------------

def read_shp_points(path):
    """读点状 shapefile，返回 [(x, y), ...]（顺序与记录号一致）。"""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 100:
        raise ValueError("shapefile 太短")
    file_code = struct.unpack(">i", data[0:4])[0]
    if file_code != 9994:
        raise ValueError("不是 shapefile（file code=%d）" % file_code)
    shape_type = struct.unpack("<i", data[32:36])[0]
    if shape_type != 1:
        raise ValueError("只支持点状 shapefile，实际类型 %d" % shape_type)

    points = []
    off = 100
    n = len(data)
    while off + 8 <= n:
        _rec_no, content_len = struct.unpack(">ii", data[off:off + 8])
        off += 8
        content = data[off:off + content_len * 2]
        off += content_len * 2
        if len(content) < 4:
            continue
        st = struct.unpack("<i", content[0:4])[0]
        if st == 0:            # null shape
            points.append((None, None))
        elif st == 1 and len(content) >= 20:
            x, y = struct.unpack("<dd", content[4:20])
            points.append((x, y))
        else:
            points.append((None, None))
    return points


def read_dbf(path):
    """读 dBASE 表，返回 list[dict]。"""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 32:
        raise ValueError("dbf 太短")

    num_records = struct.unpack("<I", data[4:8])[0]
    header_size = struct.unpack("<H", data[8:10])[0]
    record_size = struct.unpack("<H", data[10:12])[0]

    fields = []
    off = 32
    while off < header_size and data[off] != 0x0D:
        raw_name = data[off:off + 11].split(b"\x00")[0].decode("ascii", "replace")
        ftype = chr(data[off + 11])
        flen = data[off + 16]
        fields.append((raw_name, ftype, flen))
        off += 32

    rows = []
    pos = header_size
    for _ in range(num_records):
        if pos + record_size > len(data):
            break
        rec = data[pos:pos + record_size]
        pos += record_size
        if rec[0:1] == b"*":       # 已删除
            continue
        cursor = 1
        row = {}
        for name, ftype, flen in fields:
            raw = rec[cursor:cursor + flen]
            cursor += flen
            text = raw.decode("utf-8", "replace").strip()
            if ftype in ("N", "F"):
                try:
                    row[name] = float(text) if text else None
                except ValueError:
                    row[name] = None
            else:
                row[name] = text
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------

def download(url, dest):
    print("下载 %s" % url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "situation3d/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    print("  -> %s (%.1f MB)" % (dest, os.path.getsize(dest) / 1048576.0))


def pick_name_zh(row):
    for key in NAME_ZH_CANDIDATES:
        v = row.get(key)
        if v:
            return v
    return ""


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default=None,
                    help="west,south,east,north —— 只保留范围内的地名")
    ap.add_argument("--cache", default="/tmp/ne_places.zip",
                    help="zip 缓存路径，已存在就不再下载")
    args = ap.parse_args(argv[1:])

    if not os.path.exists(args.cache):
        download(SOURCE_URL, args.cache)
    else:
        print("复用已下载的 %s" % args.cache)

    os.makedirs(OUT_DIR, exist_ok=True)
    with zipfile.ZipFile(args.cache) as z:
        names = z.namelist()
        shp = next(n for n in names if n.endswith(".shp"))
        dbf = next(n for n in names if n.endswith(".dbf"))
        z.extract(shp, "/tmp")
        z.extract(dbf, "/tmp")
        shp_path = os.path.join("/tmp", shp)
        dbf_path = os.path.join("/tmp", dbf)

    points = read_shp_points(shp_path)
    rows = read_dbf(dbf_path)
    print("shapefile 点 %d 个 / dbf 记录 %d 条" % (len(points), len(rows)))

    bbox = None
    if args.bbox:
        bbox = [float(v) for v in args.bbox.split(",")]
        if len(bbox) != 4:
            print("ERROR: --bbox 需要 west,south,east,north", file=sys.stderr)
            return 2

    places = []
    for i, row in enumerate(rows):
        if i >= len(points):
            break
        x, y = points[i]
        if x is None or y is None:
            continue
        if bbox and not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
            continue
        name_zh = pick_name_zh(row)
        name_en = row.get("NAME") or row.get("NAMEASCII") or ""
        # 没有中文名的（大量小城镇）用英文名兜底，总比空标签好
        rank = row.get("SCALERANK")
        pop = row.get("POP_MAX")
        places.append({
            "n": name_zh or name_en,
            "e": name_en,
            "x": round(x, 4),
            "y": round(y, 4),
            "r": int(rank) if rank is not None else 10,
            "p": int(pop) if pop else 0,
            "c": row.get("ADM0NAME") or "",
            "a": row.get("ADM1NAME") or "",
            "k": row.get("FEATURECLA") or "",
        })

    # 按重要性排序：前端按顺序取前 N 个就能得到"最重要的 N 个地名"
    places.sort(key=lambda p: (p["r"], -p["p"]))

    payload = {
        "schema": "situation3d.places.v1",
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "licence": LICENCE,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(places),
        "bbox": bbox,
        "fields": {
            "n": "中文名（缺失时用英文名）", "e": "英文名", "x": "经度", "y": "纬度",
            "r": "scalerank 0-10，越小越重要", "p": "人口",
            "c": "国家/地区名", "a": "一级行政区名", "k": "类别（首都/一级行政中心/居民点）",
        },
        "places": places,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))

    size = os.path.getsize(OUT_FILE)
    hist = {}
    for p in places:
        hist[p["r"]] = hist.get(p["r"], 0) + 1
    print("写出 %s (%.1f KB)" % (OUT_FILE, size / 1024.0))
    print("地名 %d 个；scalerank 分布 %s" % (len(places), dict(sorted(hist.items()))))
    for r in range(0, 5):
        sample = [p["n"] for p in places if p["r"] == r][:8]
        print("  rank %d: %d 个  例：%s" % (r, hist.get(r, 0), "、".join(sample)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
