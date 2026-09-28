#!/usr/bin/env python3
"""补齐底图与高程的分级覆盖。

原来的资产只覆盖台海 AOI（115-125E, 18-29N）到影像 z11 / 高程 z10，后果是：

* 一旦把镜头拉远或平移到别处，球面上就没有影像，只剩底色；
* **三亚、冲绳、关岛三个热点区域整个在 AOI 之外，一张瓦片都没有**；
* 高程 z10 一个瓦片是 153 m/px，而 Terrarium 源数据是 30 m —— 被降采样了 5 倍。

分级覆盖（每一级只在自己需要的范围下载，避免把米级铺满整个战区）：

| 级别 | 范围 | 影像 | 高程 | 影像分辨率 |
|---|---|---|---|---|
| 全球 | 全球 | z0–6 | z0–10（已有） | 2.4 km → 0.7 km/px |
| 战区 | 115–125E, 18–29N | z7–11（已有） | z11–12 | 150 m → 76 m/px |
| 热点区域 | 8 个去重热点框 | z12 | z11–12 | 19 m/px |
| 重点区域 | 4 个城区 AOI | z13–17 | z13–14 | 9.5 m → **1.2 m/px（米级）** |

两点说明：

* **热点框做了包含去重**：平潭、澎湖落在「台湾海峡」框内，重复下载只浪费请求，
  所以只保留不被任何其他框包含的框（实测 8 个 —— 台北框的东边界 121.75 超出
  台湾海峡的 121.00，高雄框的南边界 22.45 低于 23.00，两者其实都不被包含）。
* **高程不需要无限加深**：Terrarium 源是 30 m SRTM，z12 已经是 38 m/px，再往下只是
  上采样、不增加信息。所以高程只补到 z12（热点）与 z14（重点区域，给城区三维留余量）。

全部走 --no-manifest，最后统一把新键合并进 manifest.json —— 直接写会把全域条目
覆盖成某一个框的范围。

用法（仓库根目录，远程容器上）：
    python3 situation3d/tools/fetch_base_imagery.py                # 全部
    python3 situation3d/tools/fetch_base_imagery.py --only global  # 只跑全球
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TILES = os.path.join(ROOT, "web", "tiles")
MANIFEST = os.path.join(TILES, "manifest.json")
FETCH = os.path.join(HERE, "fetch_assets.py")

# Web Mercator 的有效纬度上限；再往外没有瓦片。
MERCATOR_LAT = 85.0511

GLOBAL_IMAGERY_MAX = 6
THEATRE_TERRAIN_MAX = 12      # 战区高程补到 z12（≈38 m/px，已贴近 30 m 源）
# Theatre-wide imagery band. `fetch_assets.py --max-zoom` normally produces this
# at z11; a theatre baked only through this tool needs its own regional imagery or
# it ends up with terrain but no imagery.
THEATRE_IMAGERY_MAX = 11
HOTSPOT_IMAGERY_MAX = 12
HOTSPOT_TERRAIN_MAX = 12
URBAN_IMAGERY_MIN = 13        # 城区影像从 z13 起补到 z17
URBAN_IMAGERY_MAX = 17        # z17 ≈ 1.19 m/px，米级
URBAN_TERRAIN_MAX = 14


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiles_for_bbox(bbox, min_z, max_z):
    """XYZ tiles covering a WGS84 box, for dry-run sizing.

    Local copy rather than an import: `fetch_assets.py` is executed as a
    subprocess by this tool, and importing it would pull in its own argument
    handling. The formula is the standard Web Mercator one.
    """
    import math
    w, s, e, n = bbox
    for z in range(min_z, max_z + 1):
        n_tiles = 2 ** z
        def xt(lon):
            return int((lon + 180.0) / 360.0 * n_tiles)
        def yt(lat):
            lat = max(-MERCATOR_LAT, min(MERCATOR_LAT, lat))
            r = math.radians(lat)
            return int((1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi)
                       / 2.0 * n_tiles)
        x0, x1 = xt(w), xt(e)
        y0, y1 = yt(n), yt(s)
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                yield (z, x, y)


def theatre_spec(theatre):
    """Resolve a theatre name to its bbox and hotspot sources.

    Without this the bake is single-theatre by construction: `hotspot_boxes()`
    reads only `scenario.HOTSPOTS` (Taiwan Strait) and the theatre pass reads the
    manifest's top-level `bbox`. Baking the India-China sector therefore needs an
    explicit selector, or it would re-bake Taiwan and silently report success.

    Returns (bbox, hotspot_boxes, urban_boxes).
    """
    sys.path.insert(0, ROOT)
    from engine import scenario  # noqa: E402

    if theatre == "india_china":
        return (list(scenario.THEATRE_BBOXES["india_china"]),
                [h["bbox"] for h in scenario.INDIA_CHINA_HOTSPOTS],
                [])
    return (list(scenario.THEATRE_BBOXES["taiwan_strait"]),
            hotspot_boxes(), urban_boxes())


def hotspot_boxes():
    """不被任何其他热点框包含的热点框。"""
    sys.path.insert(0, ROOT)
    from engine import scenario  # noqa: E402

    all_boxes = [h["bbox"] for h in scenario.HOTSPOTS]
    kept = []
    for b in all_boxes:
        inside = any(
            b is not o and b[0] >= o[0] and b[1] >= o[1] and b[2] <= o[2] and b[3] <= o[3]
            for o in all_boxes)
        if not inside:
            kept.append(b)
    return kept


def urban_boxes():
    sys.path.insert(0, ROOT)
    from engine import scenario  # noqa: E402
    return [(a["name"], a["bbox"]) for a in scenario.URBAN_AOI]


def run_fetch(family, bbox, min_zoom, max_zoom, workers, label, extra=None):
    cmd = [
        sys.executable, FETCH, family,
        # 必须用 --bbox= 形式：全球范围是 "-180.0,..."，argparse 会把以 "-" 开头的
        # 值当成选项名，报 "expected one argument"。
        "--bbox=" + ",".join("%.4f" % v for v in bbox),
        "--min-zoom", str(min_zoom),
        "--max-zoom", str(max_zoom),
        "--workers", str(workers),
        "--no-manifest",
    ]
    if extra:
        cmd += extra
    print("=== %s  %s  bbox=%s  z%d-%d ===" % (
        family, label, ",".join("%.2f" % v for v in bbox), min_zoom, max_zoom), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print("  (rc=%d)" % rc, flush=True)
    return rc


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="all",
                    choices=["all", "global", "theatre", "hotspot", "urban"])
    ap.add_argument("--theatre", default="taiwan_strait",
                    choices=["taiwan_strait", "india_china"],
                    help="which theatre to bake; selects its bbox and hotspots")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tile counts per pass without downloading")
    args = ap.parse_args(argv[1:])

    only = args.only
    tb, hot, urb = theatre_spec(args.theatre)
    print("theatre      : %s  bbox=%s  hotspots=%d  urban=%d"
          % (args.theatre, tb, len(hot), len(urb)))
    if args.dry_run:
        total = 0
        for fam, boxes, lo, hi, label in (
                ("imagery", [-180.0, -MERCATOR_LAT, 180.0, MERCATOR_LAT], 0,
                 GLOBAL_IMAGERY_MAX, "全球底图"),
                ("terrain", tb, 11, THEATRE_TERRAIN_MAX, "战区高程"),
                ("imagery", tb, 11, THEATRE_IMAGERY_MAX, "战区影像"),
        ):
            n = sum(len(list(tiles_for_bbox(boxes, lo, hi))) for _ in (0,))
            total += n
            print("  %-10s z%d-%d  %6d tiles" % (label, lo, hi, n))
        hn = sum(len(list(tiles_for_bbox(b, 0, HOTSPOT_IMAGERY_MAX))) for b in hot)
        tn = sum(len(list(tiles_for_bbox(b, 11, HOTSPOT_TERRAIN_MAX))) for b in hot)
        total += hn + tn
        print("  %-10s z0-%d  %6d tiles" % ("热点影像", HOTSPOT_IMAGERY_MAX, hn))
        print("  %-10s z11-%d %6d tiles" % ("热点高程", HOTSPOT_TERRAIN_MAX, tn))
        print("  TOTAL (excluding global, which is shared): %d tiles" % total)
        return 0

    # ---- 1. 全球影像 ----
    if only in ("all", "global"):
        run_fetch("imagery", [-180.0, -MERCATOR_LAT, 180.0, MERCATOR_LAT],
                  0, GLOBAL_IMAGERY_MAX, args.workers, "全球底图")

    # ---- 2. 战区影像与高程 ----
    if only in ("all", "theatre"):
        # The theatre's own imagery band. Previously the imagery above z11 was
        # produced by `fetch_assets.py`, so a theatre baked only through this tool
        # had terrain but no regional imagery.
        run_fetch("imagery", tb, 11, THEATRE_IMAGERY_MAX, args.workers, "战区影像")
        run_fetch("terrain", tb, 11, THEATRE_TERRAIN_MAX, args.workers, "战区高程")

    # ---- 3. 热点区域：影像 z12 + 高程 z11-12 ----
    if only in ("all", "hotspot"):
        for b in hot:
            run_fetch("imagery", b, 0, HOTSPOT_IMAGERY_MAX, args.workers, "热点影像")
        for b in hot:
            run_fetch("terrain", b, 11, HOTSPOT_TERRAIN_MAX, args.workers, "热点高程")

    # ---- 4. 重点区域（城区）：影像 z13-17 米级 + 高程 z13-14 ----
    if only in ("all", "urban"):
        for name, b in urb:
            run_fetch("imagery", b, URBAN_IMAGERY_MIN, URBAN_IMAGERY_MAX,
                      args.workers, "重点区域影像 " + name)
        for name, b in urb:
            run_fetch("terrain", b, 13, URBAN_TERRAIN_MAX, args.workers,
                      "重点区域高程 " + name)

    # ---- manifest 合并 ----
    # **只写本次真正跑过的相位**：否则只跑 --only urban 也会把 hotspot_terrain
    # 写成 z12，前端就会去请求一批并不存在的瓦片。
    man = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r", encoding="utf-8") as fh:
            man = json.load(fh)
    if only in ("all", "global"):
        man["imagery_global"] = {"min_zoom": 0, "max_zoom": GLOBAL_IMAGERY_MAX}
    if only in ("all", "theatre"):
        man["theatre_terrain"] = {"min_zoom": 11, "max_zoom": THEATRE_TERRAIN_MAX}
    if only in ("all", "hotspot"):
        # Merge, never replace. The manifest is shared by every theatre, so
        # overwriting this list with one theatre's boxes silently deletes the
        # others. Measured: baking india_china replaced the 9 Taiwan boxes with 7
        # India ones, after which the Taiwan case mounted the *India* hotspots and
        # its own coverage disappeared with no error anywhere.
        prev_boxes = (man.get("hotspot_imagery") or {}).get("boxes") or []
        merged = {tuple(round(float(v), 3) for v in b) for b in prev_boxes if len(b) == 4}
        for b in hot:
            merged.add(tuple(round(float(v), 3) for v in b))
        man["hotspot_imagery"] = {
            "min_zoom": 0, "max_zoom": HOTSPOT_IMAGERY_MAX,
            "boxes": [list(b) for b in sorted(merged)],
        }
        man["hotspot_terrain"] = {"min_zoom": 11, "max_zoom": HOTSPOT_TERRAIN_MAX}
    if only in ("all", "urban"):
        man["urban_imagery_detail"] = {"min_zoom": URBAN_IMAGERY_MIN,
                                       "max_zoom": URBAN_IMAGERY_MAX}
        man["urban_terrain"] = {"min_zoom": 13, "max_zoom": URBAN_TERRAIN_MAX}
    man["coverage_note"] = (
        "影像四级：全球 z0-%d；战区 z0-11；热点区域 %d 个框 z0-%d；"
        "重点区域（4 个城区 AOI）z%d-%d（z%d 约 1.2 m/px，米级）。"
        "高程：全球 z0-10；战区与热点补到 z%d；重点区域补到 z%d。"
        % (GLOBAL_IMAGERY_MAX, len(hot), HOTSPOT_IMAGERY_MAX,
           URBAN_IMAGERY_MIN, URBAN_IMAGERY_MAX, URBAN_IMAGERY_MAX,
           HOTSPOT_TERRAIN_MAX, URBAN_TERRAIN_MAX)
    )
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)

    print("")
    print("全球影像     : z0-%d" % GLOBAL_IMAGERY_MAX)
    print("热点影像     : %d 个框 z0-%d" % (len(hot), HOTSPOT_IMAGERY_MAX))
    for b in hot:
        print("               %.2f,%.2f,%.2f,%.2f" % tuple(b))
    print("重点区域影像 : %d 个 AOI z%d-%d（米级）" % (
        len(urb), URBAN_IMAGERY_MIN, URBAN_IMAGERY_MAX))
    for name, b in urb:
        print("               %s  %.2f,%.2f,%.2f,%.2f" % ((name,) + tuple(b)))
    print("manifest     : %s" % MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
