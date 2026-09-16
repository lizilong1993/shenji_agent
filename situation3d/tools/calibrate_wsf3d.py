#!/usr/bin/env python3
"""Calibrate the WSF3D building-height colour ramp against real values.

The DLR GeoService renders WSF3D as a colour-mapped palette PNG, so a
downloaded tile carries a colour rather than a height. The layer is queryable,
though, and GetFeatureInfo returns the true ``GRAY_INDEX`` in metres -- which
means the mapping can be *measured* instead of guessed.

Two things make the measurement tight:

1. **Pixel-centre pairing.** For each sampled raster pixel the query point is
   the pixel's own centre, so the colour and the value describe the same cell.
   Sampling arbitrary coordinates instead leaves a half-pixel offset that
   dominates the error once the tile resolution approaches the 90 m source.
2. **A lookup table, not a fitted line.** The rasters are palette PNGs with a
   few hundred distinct colours. Each palette entry maps to one height band, so
   the table is exact where a linear fit on the red channel is not -- the ramp
   is not linear in any single channel.

Usage (from the repository root, on the container):
    python3 situation3d/tools/calibrate_wsf3d.py \
        --tiles situation3d/web/tiles/wsf3d/buildingheight \
        --out   situation3d/web/tiles/wsf3d/height_ramp.json \
        --zoom 13 --samples 900
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

R = 6378137.0
WMS = "https://geoservice.dlr.de/eoc/land/wms"
LAYER = "WSF3D_V02_BUILDINGHEIGHT"
NO_DATA = -32767.0
TILE_PX = 256


# --------------------------------------------------------------------------
# PNG decoding (standard library only: the container has no pip, no PIL)
# --------------------------------------------------------------------------


def read_png(path):
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a png: " + path)
    pos = 8
    idat = b""
    palette = None
    width = height = depth = ctype = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctag = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctag == b"IHDR":
            width, height, depth, ctype = struct.unpack(">IIBB", chunk[:10])
        elif ctag == b"IDAT":
            idat += chunk
        elif ctag == b"PLTE":
            palette = chunk
        elif ctag == b"IEND":
            break

    if depth != 8:
        raise ValueError("only 8-bit PNG supported")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    raw = zlib.decompress(idat)
    stride = width * channels
    out = bytearray(height * stride)
    prev = bytearray(stride)
    p = 0
    for y in range(height):
        f = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if f == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:
            for i in range(channels, stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return width, height, channels, bytes(out), palette


def pixel_rgb(png, i, j):
    width, _h, ch, pix, palette = png
    off = (j * width + i) * ch
    if palette is not None:
        idx = pix[off]
        return palette[idx * 3], palette[idx * 3 + 1], palette[idx * 3 + 2]
    if ch >= 3:
        return pix[off], pix[off + 1], pix[off + 2]
    v = pix[off]
    return v, v, v


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------


def tile_merc_bounds(z, x, y):
    world = 2.0 * math.pi * R
    size = world / (2.0 ** z)
    minx = -world / 2.0 + x * size
    maxy = world / 2.0 - y * size
    return minx, maxy - size, minx + size, maxy


def pixel_center_lonlat(z, x, y, i, j):
    """WGS84 of the centre of pixel (i, j) in tile (z, x, y)."""
    minx, miny, maxx, maxy = tile_merc_bounds(z, x, y)
    mx = minx + (i + 0.5) / TILE_PX * (maxx - minx)
    my = maxy - (j + 0.5) / TILE_PX * (maxy - miny)
    lon = math.degrees(mx / R)
    lat = math.degrees(2.0 * math.atan(math.exp(my / R)) - math.pi / 2.0)
    return lon, lat


def merc(lon, lat):
    return math.radians(lon) * R, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R


# --------------------------------------------------------------------------
# WMS query
# --------------------------------------------------------------------------


def query_value(lon, lat, half_deg=0.0002, retries=3):
    x0, y0 = merc(lon - half_deg, lat - half_deg)
    x1, y1 = merc(lon + half_deg, lat + half_deg)
    url = (
        "%s?service=WMS&version=1.3.0&request=GetFeatureInfo"
        "&layers=%s&query_layers=%s&crs=EPSG:3857"
        "&bbox=%.3f,%.3f,%.3f,%.3f&width=101&height=101&info_format=text/plain&i=50&j=50"
        % (WMS, LAYER, LAYER, x0, y0, x1, y1)
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "situation3d-calib/1.0"})
            text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
            for line in text.splitlines():
                if "GRAY_INDEX" in line:
                    return float(line.split("=", 1)[1].strip())
            return None
        except Exception:  # noqa: BLE001 - retried, then reported as None
            time.sleep(0.5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------


def collect_tiles(tiles_dir, zoom):
    out = []
    root = os.path.join(tiles_dir, str(zoom))
    if not os.path.isdir(root):
        return out
    for tx in sorted(os.listdir(root)):
        sub = os.path.join(root, tx)
        if not os.path.isdir(sub):
            continue
        for f in sorted(os.listdir(sub)):
            if f.endswith(".png"):
                out.append((int(tx), int(f[:-4])))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="dir containing <z>/<x>/<y>.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", type=int, default=13)
    ap.add_argument("--samples", type=int, default=900, help="max WMS queries")
    ap.add_argument("--pixel-step", type=int, default=6, help="sample every Nth pixel")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args(argv[1:])

    tiles = collect_tiles(args.tiles, args.zoom)
    if not tiles:
        print("ERROR: no tiles under %s/%d" % (args.tiles, args.zoom), file=sys.stderr)
        return 1
    print("tiles at z%d: %d" % (args.zoom, len(tiles)))

    rng = random.Random(args.seed)
    step = max(1, args.pixel_step)
    candidates = []
    png_cache = {}
    for (tx, ty) in tiles:
        path = os.path.join(args.tiles, str(args.zoom), str(tx), "%d.png" % ty)
        try:
            png = read_png(path)
        except Exception as exc:  # noqa: BLE001
            print("  skip %s: %s" % (path, exc))
            continue
        png_cache[(tx, ty)] = png
        for j in range(step // 2, TILE_PX, step):
            for i in range(step // 2, TILE_PX, step):
                candidates.append((tx, ty, i, j))

    print("pixel candidates: %d" % len(candidates))
    rng.shuffle(candidates)
    picked = candidates[:args.samples]
    print("querying %d of them against GetFeatureInfo" % len(picked))

    def probe(item):
        tx, ty, i, j = item
        lon, lat = pixel_center_lonlat(args.zoom, tx, ty, i, j)
        value = query_value(lon, lat)
        if value is None:
            return None
        rgb = pixel_rgb(png_cache[(tx, ty)], i, j)
        return {"rgb": list(rgb), "value": value, "lon": round(lon, 6), "lat": round(lat, 6)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [r for r in pool.map(probe, picked) if r is not None]

    if not results:
        print("ERROR: no usable samples", file=sys.stderr)
        return 1

    # Group by palette colour.
    by_colour = {}
    for r in results:
        by_colour.setdefault(tuple(r["rgb"]), []).append(r["value"])

    table = []
    inconsistent = []
    for rgb, values in sorted(by_colour.items()):
        values = sorted(values)
        med = values[len(values) // 2]
        spread = values[-1] - values[0]
        entry = {
            "rgb": list(rgb),
            "h": round(med, 2),
            "n": len(values),
            "spread": round(spread, 2),
        }
        table.append(entry)
        if spread > 5.0:
            inconsistent.append(entry)

    # Palette entries that never got sampled are still needed at render time;
    # the front end falls back to the nearest sampled colour, so report how much
    # of the palette is covered.
    all_colours = set()
    for png in png_cache.values():
        _w, _h, ch, pix, palette = png
        if palette is None:
            continue
        for k in range(0, len(palette), 3):
            all_colours.add((palette[k], palette[k + 1], palette[k + 2]))
    covered = len(all_colours & set(by_colour)) if all_colours else len(by_colour)

    print("samples: %d  distinct colours sampled: %d  (palette size %d)"
          % (len(results), len(by_colour), len(all_colours)))
    print("colour/value spread > 5 m: %d entries" % len(inconsistent))
    vals = [e["h"] for e in table if e["h"] != NO_DATA]
    print("height range observed: %.2f - %.2f m" % (min(vals), max(vals)))

    # Monotonic sanity check: the ramp should order colours consistently.
    ramp_sorted = sorted(table, key=lambda e: e["h"])
    print("ramp (lowest 5): %s" % " ".join(
        "%s=%.1f" % (tuple(e["rgb"]), e["h"]) for e in ramp_sorted[:5]))
    print("ramp (highest 5): %s" % " ".join(
        "%s=%.1f" % (tuple(e["rgb"]), e["h"]) for e in ramp_sorted[-5:]))

    nodata_colours = sorted(c for c, v in by_colour.items()
                            if all(x == NO_DATA for x in v))
    print("no-data colours: %d, e.g. %s" % (len(nodata_colours), nodata_colours[:4]))

    out = {
        "layer": LAYER,
        "source": "DLR EOC GeoService WMS, GetFeatureInfo, pixel-centre paired",
        "unit": "m",
        "zoom": args.zoom,
        "no_data_value": NO_DATA,
        "no_data_colours": [list(c) for c in nodata_colours],
        "sample_count": len(results),
        "colour_count": len(by_colour),
        "palette_size": len(all_colours),
        "lookup": [{"rgb": e["rgb"], "h": e["h"], "n": e["n"]} for e in table],
        "inconsistent": inconsistent[:20],
        "value_range_m": [round(min(vals), 2), round(max(vals), 2)],
        "note": (
            "颜色→高度查表，由 WMS GetFeatureInfo 在像元中心实测得到。"
            "查表未命中的颜色取最邻近条目。"
        ),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("written: %s  (%d lookup entries)" % (args.out, len(table)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
