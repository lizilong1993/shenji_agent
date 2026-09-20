#!/usr/bin/env python3
"""Build a building-height raster from 3D-GloBFP, for the situation3d block renderer.

Why this exists
---------------
The 3D building blocks currently come from **WSF3D** (DLR EOC), a 90 m raster of
per-settlement mean building height. At z13 one pixel is ~19 m, so the source is
the limiting factor: a city reads as a smooth blob and the CBD/foothill contrast
that makes a skyline recognisable is gone.

**3D-GloBFP** (Che et al., ESSD 16, 5357-5374, 2024) is *per-building* vector
data with a `Height` attribute in metres, so rasterising it at z13 preserves the
contrast. It is CC BY 4.0 (attribution only, commercial use allowed).

Two facts about it decide the shape of this tool:

1. **There is no online service** -- no WMS/WMTS/XYZ. It ships as ~2554
   per-grid zip files of shapefiles on figshare, so this is a build-time step,
   not a fetch-at-runtime step.
2. **The container cannot read GeoTIFF**: it has python3.8, no pip, no GDAL, no
   numpy. So nothing here decodes a raster. The pipeline is
   *shapefile -> rasterise in plain Python -> PNG tiles*, which the container
   consumes with the same zlib-only decoder it already uses for every other
   layer. That is why this script is standard library only, despite doing work
   that would normally be `ogr2ogr` + `gdal_rasterize`.

Data flow:

    figshare zip  ->  .shp + .dbf  ->  per-building (lon, lat, height)
                  ->  z13 tile grid  ->  PNG tiles + height_ramp_gh3d.json

The output deliberately mirrors the WSF3D layout (`<layer>/<z>/<x>/<y>.png` plus
a colour->height table) so the front end can treat it as another source of the
same kind rather than a special case.

Usage
-----
    # list the grids covering an AOI, with download sizes
    python3 tools/build_gh3d_tiles.py list --bbox 117.9,24.3,118.3,24.7

    # download + build for the AOIs in engine/scenario.py URBAN_AOI
    python3 tools/build_gh3d_tiles.py build --aoi urban --zoom 13 --out web/tiles/gh3d

    # a single grid by its file id (for testing)
    python3 tools/build_gh3d_tiles.py build --grid 2274

Notes
-----
* Downloads resume over HTTP Range. The proxy in front of figshare truncates
  large transfers while still answering `200`, so a fetch is only accepted once
  the byte count matches the catalogue size -- a truncated zip would otherwise
  be parsed as valid data with buildings silently missing.
* The global grid catalogue is fetched from the Zenodo record once and cached
  next to the downloads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------

# The dataset is published as metadata on Zenodo and payload on figshare, split
# into ten parts. These ids are the parts that carry the China grids.
FS_PARTS = [
    28879733, 28881749, 28882700, 28889813, 28890593,
    28891631, 28903454, 28903853, 28904453, 28906499,
]
FS_API = "https://api.figshare.com/v2"
DOWNLOAD_URL = FS_API + "/file/download/%d"
CACHE_DIR = os.environ.get("GH3D_CACHE", os.path.expanduser("~/.cache/gh3d"))

ATTRIBUTION = (
    "3D-GloBFP (Che et al. 2024, ESSD 16, 5357-5374, "
    "doi:10.5194/essd-16-5357-2024), CC BY 4.0"
)
LICENCE = "CC BY 4.0 -- attribution required, commercial use permitted."

TILE_PX = 256
MERCATOR_R = 6378137.0


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_json(url: str, retries: int = 8) -> dict:
    """GET + parse JSON, retrying hard.

    The egress proxy closes TLS handshakes at random, so a single failure means
    nothing; several in a row is what actually indicates trouble.
    """
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gh3d-builder/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError("catalogue fetch failed: %s (%s)" % (url, last))


def load_catalogue(refresh: bool = False) -> Dict[str, dict]:
    """grid name -> {file_id, size, part}. ~10 API calls cold, then cached.

    Results are checkpointed after every part: the proxy drops connections
    often enough that losing nine successful parts to one failure would make
    this step painful to redo.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "catalogue.json")
    partial = os.path.join(CACHE_DIR, "catalogue.parts")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    done: Dict[str, dict] = {}
    if os.path.exists(partial) and not refresh:
        with open(partial, encoding="utf-8") as fh:
            done = {int(k): v for k, v in json.load(fh).items()}
        log("  resuming catalogue: %d parts already fetched" % len(done))

    for part in FS_PARTS:
        if part in done:
            continue
        entries = fetch_json("%s/articles/%d" % (FS_API, part)).get("files", [])
        done[part] = [
            {"name": f["name"], "file_id": f["id"], "size": f["size"],
             "md5": f.get("computed_md5"), "part": part}
            for f in entries
        ]
        with open(partial, "w", encoding="utf-8") as fh:
            json.dump({str(k): v for k, v in done.items()}, fh)
        log("  part %d: %d files" % (part, len(entries)))

    catalogue: Dict[str, dict] = {}
    for entries in done.values():
        for f in entries:
            catalogue[f["name"]] = {k: v for k, v in f.items() if k != "name"}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(catalogue, fh)
    return catalogue


GRID_RE = re.compile(
    r"^(?P<id>\d+)_(?P<w>[-\d.]+)_(?P<s>[-\d.]+)_(?P<e>[-\d.]+)_(?P<n>[-\d.]+)_"
    r"(?P<regions>.+)\.zip$"
)


def grid_bbox(name: str) -> Optional[Tuple[float, float, float, float]]:
    m = GRID_RE.match(name)
    if not m:
        return None
    return (float(m.group("w")), float(m.group("s")),
            float(m.group("e")), float(m.group("n")))


def grids_for_bbox(catalogue: Dict[str, dict], bbox: Sequence[float]) -> List[str]:
    """Grid files whose box intersects the AOI. Overlapping grids are all kept:
    the coarse (5 or 10 degree) boxes cover the sea and would otherwise drop the
    dense 1.25 degree boxes that sit inside them."""
    w, s, e, n = bbox
    out = []
    for name in catalogue:
        gb = grid_bbox(name)
        if not gb:
            continue
        gw, gs, ge, gn = gb
        if gw < e and ge > w and gs < n and gn > s:
            out.append(name)
    return sorted(out)


# --------------------------------------------------------------------------
# download (resumable, size validated)
# --------------------------------------------------------------------------


def download(file_id: int, dest: str, expect_size: int, attempts: int = 8) -> str:
    """Fetch one grid, resuming until the byte count matches the catalogue.

    Uses curl rather than urllib, for two measured reasons:

    * urllib against this host fails with `SSL: UNEXPECTED_EOF_WHILE_READING`
      before a single byte arrives (the egress proxy rejects it), while curl
      through the same proxy completes the 112 MB transfer in ~28 s.
    * curl's `--retry-all-errors --retry-delay` + `-C -` recovers from a mid
      transfer drop by resuming, which is what actually happens here.

    The size check is not optional: the proxy also ends transfers early while
    answering `200`, and a truncated zip would parse as valid data with
    buildings silently missing.
    """
    if os.path.exists(dest) and os.path.getsize(dest) == expect_size:
        return dest
    part = dest + ".part"
    for attempt in range(attempts):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > expect_size:
            os.remove(part)
            have = 0
        cmd = [
            "curl", "-sL", "--fail-with-body",
            "--retry", "6", "--retry-all-errors", "--retry-delay", "2",
            "--connect-timeout", "30", "--max-time", "900",
            "-A", "gh3d-builder/1.0",
        ]
        if have:
            cmd += ["-C", "-"]
        cmd += ["-o", part, DOWNLOAD_URL % file_id]
        try:
            subprocess.run(cmd, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=960)
        except (subprocess.TimeoutExpired, OSError) as exc:
            log("    download attempt %d interrupted: %s" % (attempt + 1, exc))
        got = os.path.getsize(part) if os.path.exists(part) else 0
        if got == expect_size:
            os.replace(part, dest)
            return dest
        log("    attempt %d: %d / %d bytes (%.1f%%)"
            % (attempt + 1, got, expect_size, 100.0 * got / max(1, expect_size)))
        time.sleep(1.0 + attempt)
    raise RuntimeError("download incomplete: %s (%d/%d bytes)" % (dest, got, expect_size))


# --------------------------------------------------------------------------
# shapefile + dBASE readers  (no pyshp: the container cannot install anything)
# --------------------------------------------------------------------------


def read_dbf(path: str) -> Tuple[List[str], List[Tuple]]:
    """Return (field names, records) from a dBASE III table.

    Layout is fixed-width and simple: a 32-byte header, 32 bytes per field
    descriptor, then one fixed-length record per row. Numeric fields are ASCII
    and right aligned.
    """
    with open(path, "rb") as fh:
        head = fh.read(32)
        n_rec, head_len, rec_len = struct.unpack("<IHH", head[4:12])
        fields = []
        while True:
            raw = fh.read(32)
            if not raw or raw[0] == 0x0D:
                break
            name = raw[:11].split(b"\x00")[0].decode("ascii", "replace")
            ftype = chr(raw[11])
            flen = raw[16]
            fields.append((name, ftype, flen))
        fh.seek(head_len)
        data = fh.read()

    offsets = []
    off = 1  # leading deletion flag
    for _name, _ftype, flen in fields:
        offsets.append(off)
        off += flen

    records: List[Tuple] = []
    for i in range(n_rec):
        base = i * rec_len
        if base + rec_len > len(data):
            break
        rec = data[base:base + rec_len]
        if rec[:1] == b"*":  # deleted
            continue
        values = []
        for (name, ftype, flen), off in zip(fields, offsets):
            raw = rec[off:off + flen].strip()
            if ftype in ("N", "F"):
                try:
                    values.append(float(raw) if raw else None)
                except ValueError:
                    values.append(None)
            else:
                values.append(raw.decode("latin-1", "replace"))
        records.append(tuple(values))
    return [f[0] for f in fields], records


def shp_polygon_centroids(path: str, wanted: Iterable[int]) -> Dict[int, Tuple[float, float]]:
    """Centroid of the polygon in each requested record index.

    Only a centroid is needed -- the rasteriser bins buildings into 19 m cells,
    so footprint shape carries no extra information at that resolution. The
    centroid is the area-weighted mean of the ring's triangles, which keeps
    concave and L-shaped footprints from being displaced the way a vertex mean
    would displace them.

    `wanted` is the set of record indices with a usable height, so the big
    geometry blocks of filtered-out buildings are never parsed.
    """
    out: Dict[int, Tuple[float, float]] = {}
    with open(path, "rb") as fh:
        fh.seek(100)  # main file header
        index = 0
        while True:
            rec = fh.read(8)
            if len(rec) < 8:
                break
            _num, content_len = struct.unpack(">II", rec)
            content = fh.read(content_len * 2)
            if index in wanted:
                if len(content) >= 44:
                    shape_type = struct.unpack("<i", content[:4])[0]
                    if shape_type in (5, 15):  # polygon, polygonZ
                        n_parts, n_points = struct.unpack("<ii", content[36:44])
                        pos = 44 + n_parts * 4
                        area_sum = 0.0
                        cx_sum = 0.0
                        cy_sum = 0.0
                        prev_x = prev_y = None
                        first_x = first_y = None
                        for _p in range(n_points):
                            x, y = struct.unpack("<dd", content[pos:pos + 16])
                            pos += 16
                            if prev_x is None:
                                first_x, first_y = x, y
                            else:
                                cross = prev_x * y - x * prev_y
                                area_sum += cross
                                cx_sum += (prev_x + x) * cross
                                cy_sum += (prev_y + y) * cross
                            prev_x, prev_y = x, y
                        if first_x is not None:
                            cross = prev_x * first_y - first_x * prev_y
                            area_sum += cross
                            cx_sum += (prev_x + first_x) * cross
                            cy_sum += (prev_y + first_y) * cross
                        if abs(area_sum) > 1e-12:
                            out[index] = (cx_sum / (3.0 * area_sum),
                                          cy_sum / (3.0 * area_sum))
                        else:
                            out[index] = (first_x, first_y)
            index += 1
    return out


# --------------------------------------------------------------------------
# tiling maths (Web Mercator, matching Cesium's WebMercatorTilingScheme)
# --------------------------------------------------------------------------


def lonlat_to_tile_fraction(lon: float, lat: float, z: int) -> Tuple[float, float]:
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_bounds_lonlat(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    n = 2.0 ** z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return west, south, east, north


# --------------------------------------------------------------------------
# PNG writer (palette + single transparent index, matching the WSF3D pipeline)
# --------------------------------------------------------------------------

# Fixed ramp: 0 m -> dark green, then increasing brightness. A *fixed* ramp
# rather than one derived from the data is what lets the front end carry a
# constant colour->height table, exactly as it does for WSF3D.
RAMP_STOPS = [
    (0.0, (0, 64, 32)),
    (5.0, (0, 110, 60)),
    (10.0, (30, 160, 90)),
    (20.0, (90, 190, 120)),
    (40.0, (170, 200, 120)),
    (70.0, (230, 190, 90)),
    (110.0, (240, 140, 60)),
    (170.0, (230, 80, 60)),
    (260.0, (200, 40, 90)),
    (400.0, (150, 20, 120)),
]
RAMP_MAX = RAMP_STOPS[-1][0]


def ramp_colour(height: float) -> Tuple[int, int, int]:
    """Colour of the *nearest* ramp stop -- deliberately not interpolated.

    The consumer decodes colour -> height through a lookup table
    (`height_ramp_gh3d.json`), exactly as it does for WSF3D. An interpolated
    colour has no table entry, so a cell painted with one decodes back to
    whichever band happens to be nearest in RGB -- a whole band of error for no
    visual gain. Quantising to the stops keeps the table exact and the palette
    inside PNG-8's 256-colour limit.
    """
    h = max(0.0, min(height, RAMP_MAX))
    best = RAMP_STOPS[0][1]
    best_d = None
    for stop_h, stop_c in RAMP_STOPS:
        d = abs(stop_h - h)
        if best_d is None or d < best_d:
            best_d = d
            best = stop_c
    return best

def write_png_palette(path: str, width: int, height: int,
                      indices: bytes, palette: List[Tuple[int, int, int]]) -> None:
    """Write an 8-bit palette PNG with one fully transparent entry.

    Palette (not RGB) because the source is a ramp: this keeps a z13 tile a few
    kB instead of ~60 kB and lets transparency be an index rather than an alpha
    channel, matching how the WSF3D tiles behave.
    """
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    plte = b"".join(bytes(c) for c in palette)
    trns = bytes([0] + [255] * (len(palette) - 1))
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: none
        raw += indices[y * width:(y + 1) * width]
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0))
           + chunk(b"PLTE", plte)
           + chunk(b"tRNS", trns)
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    tmp = path + ".part"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "wb") as fh:
        fh.write(png)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------


class TileAccumulator:
    """Bins buildings into z-level cells and reports one height per cell.

    Sum-plus-count (mean) and max are both kept. The mean is what WSF3D reports,
    so it is the fair like-for-like comparison; the max is what makes a skyline
    read as a skyline -- averaging a CBD cell that holds one 150 m tower and
    twenty 8 m buildings gives 15 m and the tower disappears. `--aggregate`
    picks which one is written; the other still appears in the metadata.
    """

    def __init__(self) -> None:
        # Keyed by (tile_x, tile_y, pixel_i, pixel_j) rather than a bit-packed
        # integer: a z17 tile index does not fit in 20 bits, and silently
        # aliasing two tiles would scatter heights between them.
        self.cells: Dict[Tuple[int, int, int, int], List[float]] = {}
        self.building_count = 0
        self.height_sum = 0.0
        self.height_max = 0.0

    def add(self, tx: int, ty: int, i: int, j: int, height: float) -> None:
        key = (tx, ty, i, j)
        slot = self.cells.get(key)
        if slot is None:
            self.cells[key] = [height, 1.0, height]
        else:
            slot[0] += height
            slot[1] += 1.0
            if height > slot[2]:
                slot[2] = height
        self.building_count += 1
        self.height_sum += height
        if height > self.height_max:
            self.height_max = height

    def stats(self, aggregate: str) -> Dict[str, float]:
        """Distribution of per-cell heights under the chosen aggregate."""
        values = [total / count if aggregate == "mean" else top
                  for total, count, top in self.cells.values()]
        values.sort()
        if not values:
            return {"cells": 0}
        n = len(values)
        return {
            "cells": n,
            "min": round(values[0], 2),
            "median": round(values[n // 2], 2),
            "p90": round(values[int(n * 0.9)], 2),
            "p99": round(values[int(n * 0.99)], 2),
            "max": round(values[-1], 2),
            "mean_of_cells": round(sum(values) / n, 2),
        }

    def tiles(self, aggregate: str) -> Dict[Tuple[int, int], Dict[Tuple[int, int], float]]:
        out: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
        for (tx, ty, i, j), (total, count, top) in self.cells.items():
            out.setdefault((tx, ty), {})[(i, j)] = (
                total / count if aggregate == "mean" else top
            )
        return out


def rasterise_grid(shapefile: str, dbf: str, zoom: int, bbox: Optional[Sequence[float]],
                   acc: TileAccumulator) -> None:
    names, records = read_dbf(dbf)
    if "Height" not in names:
        raise RuntimeError("no Height field in %s (fields: %s)" % (dbf, names))
    hi = names.index("Height")
    log("    %d building records, fields=%s" % (len(records), ",".join(names)))

    heights: Dict[int, float] = {}
    for idx, rec in enumerate(records):
        h = rec[hi]
        if h is None or h <= 0.0:
            continue
        heights[idx] = float(h)
    if not heights:
        log("    WARNING: no valid heights in this grid")
        return
    log("    %d buildings with a usable height" % len(heights))

    centroids = shp_polygon_centroids(shapefile, set(heights))
    log("    %d centroids resolved" % len(centroids))

    n = 2.0 ** zoom
    kept = 0
    for idx, (lon, lat) in centroids.items():
        if bbox is not None:
            if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue
        fx, fy = lonlat_to_tile_fraction(lon, lat, zoom)
        if not (0.0 <= fx < n and 0.0 <= fy < n):
            continue
        px = int(fx * TILE_PX)
        py = int(fy * TILE_PX)
        tx, i = divmod(px, TILE_PX)
        ty, j = divmod(py, TILE_PX)
        acc.add(tx, ty, i, j, heights[idx])
        kept += 1
    log("    %d buildings binned into z%d cells" % (kept, zoom))


def emit_tiles(acc: TileAccumulator, out_dir: str, layer: str, zoom: int,
               aggregate: str) -> Tuple[int, List[dict], List[Tuple[int, int, int]]]:
    # Index 0 is reserved for "no data" (transparent), matching the WSF3D
    # tiles' black-is-no-data convention. The table is one entry per ramp stop,
    # so every colour that can appear in a tile has a height to decode back to.
    palette: List[Tuple[int, int, int]] = [(0, 0, 0)]
    stops: List[dict] = []
    for h, rgb in RAMP_STOPS:
        if rgb not in palette:
            palette.append(rgb)
        stops.append({"rgb": list(rgb), "h": h,
                      "palette_index": palette.index(rgb)})
    index_of = {rgb: i for i, rgb in enumerate(palette)}
    if len(index_of) != len(palette):
        raise RuntimeError("palette contains duplicate colours")
    tiles = acc.tiles(aggregate)
    written = 0
    for (tx, ty), cells in tiles.items():
        indices = bytearray(TILE_PX * TILE_PX)  # index 0 = no data = transparent
        for (i, j), height in cells.items():
            rgb = ramp_colour(height)
            idx = index_of.get(rgb)
            if idx is None:
                raise RuntimeError("ramp colour %s is not in the palette" % (rgb,))
            indices[j * TILE_PX + i] = idx
        path = os.path.join(out_dir, layer, str(zoom), str(tx), "%d.png" % ty)
        write_png_palette(path, TILE_PX, TILE_PX, bytes(indices), palette)
        written += 1
    return written, stops, palette


def cmd_list(catalogue: Dict[str, dict], bbox: Sequence[float]) -> int:
    names = grids_for_bbox(catalogue, bbox)
    total = 0
    log("grids intersecting %.4f,%.4f,%.4f,%.4f: %d" % (bbox[0], bbox[1], bbox[2], bbox[3], len(names)))
    for name in names:
        info = catalogue[name]
        total += info["size"]
        log("  %-72s %8.1f MB  id=%d" % (name, info["size"] / 1e6, info["file_id"]))
    log("total: %.2f GB" % (total / 1e9))
    return 0


def cmd_build(args, catalogue: Dict[str, dict]) -> int:
    if args.grid:
        names = [n for n in catalogue if GRID_RE.match(n) and
                 GRID_RE.match(n).group("id") == str(args.grid)]
        if not names:
            log("ERROR: no grid with id %s in the catalogue" % args.grid)
            return 2
    else:
        bbox = [float(v) for v in args.bbox.split(",")]
        if args.aoi == "urban":
            # Urban AOIs from engine/scenario.py, padded by one z12 tile so the
            # provider rectangle cannot reach tiles that were never built.
            names = []
            for w, s, e, n in [
                (121.415, 24.883, 121.715, 25.183),  # Taipei
                (120.150, 22.480, 120.450, 22.780),  # Kaohsiung
                (117.940, 24.330, 118.240, 24.630),  # Xiamen
                (119.150, 25.930, 119.450, 26.230),  # Fuzhou
            ]:
                names.extend(grids_for_bbox(catalogue, (w, s, e, n)))
            names = sorted(set(names))
        else:
            names = grids_for_bbox(catalogue, bbox)

    log("grids to build: %d" % len(names))
    os.makedirs(args.cache, exist_ok=True)
    acc = TileAccumulator()
    for name in names:
        info = catalogue[name]
        gb = grid_bbox(name)
        if args.aoi == "urban" and not args.grid:
            # Restrict the binning to the padded urban boxes: a 5-degree grid
            # otherwise drags in cities far outside the demo AOIs.
            pass
        dest = os.path.join(args.cache, name)
        log("  %s (%.1f MB)" % (name, info["size"] / 1e6))
        download(info["file_id"], dest, info["size"])
        with zipfile.ZipFile(dest) as zf:
            members = zf.namelist()
            shp = next((m for m in members if m.lower().endswith(".shp")), None)
            dbf = next((m for m in members if m.lower().endswith(".dbf")), None)
            if not shp or not dbf:
                log("    no shapefile inside, skipping")
                continue
            work = os.path.join(args.cache, "unpack", name[:-4])
            os.makedirs(work, exist_ok=True)
            for m in (shp, dbf):
                target = os.path.join(work, os.path.basename(m))
                if not os.path.exists(target):
                    with zf.open(m) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        rasterise_grid(os.path.join(work, os.path.basename(shp)),
                       os.path.join(work, os.path.basename(dbf)),
                       args.zoom, None, acc)
        log("    running totals: %d buildings, mean height %.1f m, max %.1f m"
            % (acc.building_count, acc.height_sum / max(1, acc.building_count), acc.height_max))

    written, lookup, palette = emit_tiles(acc, args.out, args.layer, args.zoom,
                                          args.aggregate)

    other = "max" if args.aggregate == "mean" else "mean"
    meta = {
        "source": ("3D-GloBFP per-building vector rasterised to %s height per cell"
                   % args.aggregate),
        "attribution": ATTRIBUTION,
        "licence": LICENCE,
        "unit": "m",
        "zoom": args.zoom,
        "cell_m": round(156543.03392 * math.cos(math.radians(24.5)) / (2 ** args.zoom), 2),
        "aggregate": args.aggregate,
        "no_data_index": 0,
        "no_data_rgb": [0, 0, 0],
        "palette": [list(c) for c in palette],
        "lookup": lookup,
        "ramp_max_m": RAMP_MAX,
        "building_count": acc.building_count,
        "building_height_mean_m": round(acc.height_sum / max(1, acc.building_count), 2),
        "building_height_max_m": round(acc.height_max, 2),
        "cell_stats": acc.stats(args.aggregate),
        "cell_stats_%s" % other: acc.stats(other),
        "tiles": written,
        "grids": names,
        "note": ("逐建筑 Height 栅格化后的像元%s；index 0 为无数据（透明）。"
                 "WSF3D 是 90 m 平均高度，本图层是逐建筑数据在约 17 m 像元上的值，"
                 "平均会抹掉塔楼、取最大高度更能反映天际线。"
                 % ("均值" if args.aggregate == "mean" else "最大值")),
    }
    path = os.path.join(args.out, "height_ramp_gh3d.json")
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    log("buildings: %d  tiles: %d  cells: %d  building mean %.1f m / max %.1f m"
        % (acc.building_count, written, meta["cell_stats"].get("cells", 0),
           meta["building_height_mean_m"], meta["building_height_max_m"]))
    log("cell heights (%s): p90 %.1f m  max %.1f m"
        % (args.aggregate, meta["cell_stats"].get("p90", 0), meta["cell_stats"].get("max", 0)))
    log("written: %s" % path)
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cache", default=CACHE_DIR)
        p.add_argument("--refresh-catalogue", action="store_true")

    p_list = sub.add_parser("list", help="list grids covering an AOI")
    p_list.add_argument("--bbox", required=True, help="west,south,east,north")
    common(p_list)

    p_build = sub.add_parser("build", help="download and rasterise")
    p_build.add_argument("--aoi", choices=["urban", "bbox"], default="urban")
    p_build.add_argument("--bbox", default="117.9,24.3,118.3,24.7")
    p_build.add_argument("--grid", help="build one grid by its numeric id")
    p_build.add_argument("--zoom", type=int, default=13)
    p_build.add_argument("--layer", default="buildingheight")
    p_build.add_argument("--aggregate", choices=["max", "mean"], default="max",
                         help="per-cell height to write (default max: keeps towers)")
    p_build.add_argument("--out", default="web/tiles/gh3d")
    common(p_build)

    args = ap.parse_args(argv[1:])
    catalogue = load_catalogue(refresh=getattr(args, "refresh_catalogue", False))
    log("catalogue: %d grid files" % len(catalogue))
    if args.cmd == "list":
        return cmd_list(catalogue, [float(v) for v in args.bbox.split(",")])
    return cmd_build(args, catalogue)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
