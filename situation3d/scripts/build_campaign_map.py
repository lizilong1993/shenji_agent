#!/usr/bin/env python3
"""Build a campaign hex map (basic.json/cost.pickle/see.npz) for a theatre.

Inputs are open data already used by situation3d:

* Natural Earth 1:10m land polygons (public domain) -> land/sea mask
* OSM Overpass (ODbL) -> roads and residential areas
* Local Terrarium terrain tiles -> elevation

The output uses the public 庙算 map contract so the same fixtures/tools apply:
``basic.json`` with ``map_data[row][col]`` cells, ``cost.pickle`` (4 movement
modes), ``<map_id>see.npz`` (10 visibility modes).

Only the Python standard library + numpy are required.  PNG is decoded by
hand because the target container has no PIL.

Usage:
    python3 situation3d/scripts/build_campaign_map.py \
        --profile taiwan_south \
        --out situation3d/web/data/campaign_maps/map_9001
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import struct
import sys
import time
import urllib.parse
import urllib.request
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
REPO = os.path.dirname(SITUATION3D)
sys.path.insert(0, SITUATION3D)

from wargame_core import hexmap  # noqa: E402


# ---------------------------------------------------------------------------
# Theatre profiles (demo choices, not real operational areas)
# ---------------------------------------------------------------------------

PROFILES = {
    "taiwan_south": {
        "map_id": 9001,
        "name": "台海南段（虚构演示）",
        "center_lon": 119.55, "center_lat": 23.55,
        "rows": 100, "cols": 100, "scale_m": 3000,
        "elev_grade": 10,
    },
    "india_china_west": {
        "map_id": 9002,
        "name": "中印西段（虚构演示）",
        "center_lon": 79.10, "center_lat": 33.75,
        "rows": 100, "cols": 100, "scale_m": 4000,
        "elev_grade": 20,
    },
}


# ---------------------------------------------------------------------------
# PNG decoding (Terrarium: RGB, elevation = R*256 + G + B/256 - 32768)
# ---------------------------------------------------------------------------

def _png_chunks(data: bytes):
    pos = 8
    while pos < len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        yield ctype, chunk
        pos += 12 + length


def decode_png_rgb(data: bytes) -> Tuple[int, int, bytes]:
    width = height = None
    bit_depth = color_type = None
    idat = []
    for ctype, chunk in _png_chunks(data):
        if ctype == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif ctype == b"IDAT":
            idat.append(chunk)
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError("unsupported PNG: depth=%s type=%s" % (bit_depth, color_type))
    bpp = 3 if color_type == 2 else 4
    raw = zlib.decompress(b"".join(idat))
    stride = width * bpp
    out = bytearray(width * height * 3)
    prev = bytearray(stride)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if ftype == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else b if pb <= pc else c
                line[i] = (line[i] + pr) & 0xFF
        for x in range(width):
            src = x * bpp
            dst = (y * width + x) * 3
            out[dst:dst + 3] = line[src:src + 3]
        prev = line
    return width, height, bytes(out)


class TerrainSampler(object):
    """Reads local Terrarium tiles and samples elevation by lon/lat."""

    def __init__(self, tiles_dir: str, zoom: int = 12):
        self.dir = tiles_dir
        self.zoom = zoom
        self._cache: Dict[Tuple[int, int], Tuple[int, int, bytes]] = {}

    def _tile(self, z: int, x: int, y: int):
        key = (x, y)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.dir, str(z), str(x), "%d.png" % y)
        if not os.path.isfile(path):
            # Walk up to a coarser zoom.  When the coarser tile exists, return
            # its pixel coordinates scaled back to the requested zoom.
            for zz in range(z - 1, -1, -1):
                factor = 2 ** (z - zz)
                px, py = x // factor, y // factor
                p = os.path.join(self.dir, str(zz), str(px), "%d.png" % py)
                if os.path.isfile(p):
                    w, h, rgb = decode_png_rgb(open(p, "rb").read())
                    self._cache[key] = (w, h, rgb, factor)
                    return self._cache[key]
            raise FileNotFoundError(path)
        w, h, rgb = decode_png_rgb(open(path, "rb").read())
        self._cache[key] = (w, h, rgb, 1)
        return self._cache[key]

    def elevation(self, lon: float, lat: float) -> float:
        z = self.zoom
        n = 2 ** z
        xf = (lon + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        yf = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
        x = int(xf)
        y = int(yf)
        w, h, rgb, factor = self._tile(z, x, y)
        # Fraction inside the requested-z tile, mapped onto the coarse tile.
        fx = (xf - x) * factor
        fy = (yf - y) * factor
        px = min(w - 1, max(0, int(fx * w / factor)))
        py = min(h - 1, max(0, int(fy * h / factor)))
        i = (py * w + px) * 3
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        return r * 256.0 + g + b / 256.0 - 32768.0


# ---------------------------------------------------------------------------
# Shapefile (Natural Earth land) — polygon point-in-test with bbox prefilter
# ---------------------------------------------------------------------------

class LandMask(object):
    def __init__(self, shp_path: str):
        self.polys: List[Tuple[float, float, float, float, List[Tuple[float, float]]]] = []
        self._load(shp_path)

    def _load(self, path: str) -> None:
        data = open(path, "rb").read()
        pos = 100
        while pos + 8 <= len(data):
            _num, content_len = struct.unpack(">II", data[pos:pos + 8])
            content = data[pos + 8:pos + 8 + content_len * 2]
            pos += 8 + content_len * 2
            if len(content) < 4:
                continue
            shape_type = struct.unpack("<I", content[:4])[0]
            if shape_type != 5:      # polygon
                continue
            n_parts, n_points = struct.unpack("<II", content[36:44])
            parts = list(struct.unpack("<%dI" % n_parts, content[44:44 + 4 * n_parts]))
            off = 44 + 4 * n_parts
            pts = struct.unpack("<%dd" % (n_points * 2), content[off:off + 16 * n_points])
            ring = [(pts[i * 2], pts[i * 2 + 1]) for i in range(n_points)]
            if not ring:
                continue
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            self.polys.append((min(xs), min(ys), max(xs), max(ys), ring))

    def is_land(self, lon: float, lat: float) -> bool:
        for x0, y0, x1, y1, ring in self.polys:
            if lon < x0 or lon > x1 or lat < y0 or lat > y1:
                continue
            if _point_in_ring(lon, lat, ring):
                return True
        return False


def _point_in_ring(x: float, y: float, ring: Sequence[Tuple[float, float]]) -> bool:
    """Even-odd point-in-polygon test using edge crossings."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y):
            xcross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < xcross:
                inside = not inside
        j = i
    return inside


def _ring_contains(x: float, y: float, ring: Sequence[Tuple[float, float]]) -> bool:
    return _point_in_ring(x, y, ring)


class LandRaster(object):
    """Scanline raster of the land polygons at a fixed set of latitudes.

    Building per-row crossings once makes 10k point tests cheap; the naive
    point-in-polygon loop over Natural Earth rings is far too slow for 100x100.
    """

    def __init__(self, shp_path: str, lats: Sequence[float]):
        self.lats = list(lats)
        self.crossings: List[List[Tuple[float, float]]] = [
            [] for _ in range(len(self.lats))
        ]
        self.rings: List[Tuple[float, float, float, float, List[Tuple[float, float]]]] = []
        self._load(shp_path)
        # Sorting once turns each point test into a scan over a short prefix.
        for row in self.crossings:
            row.sort()

    def _row_index(self, lat: float) -> Optional[int]:
        import bisect
        i = bisect.bisect_left(self.lats, lat)
        best = None
        bestd = 1e9
        for cand in (i - 1, i):
            if 0 <= cand < len(self.lats):
                d = abs(self.lats[cand] - lat)
                if d < bestd:
                    best, bestd = cand, d
        if best is None or bestd > 0.02:
            return None
        return best

    def _load(self, path: str) -> None:
        data = open(path, "rb").read()
        pos = 100
        lats = self.lats
        lo, hi = lats[0], lats[-1]
        while pos + 8 <= len(data):
            _num, content_len = struct.unpack(">II", data[pos:pos + 8])
            content = data[pos + 8:pos + 8 + content_len * 2]
            pos += 8 + content_len * 2
            if len(content) < 44:
                continue
            shape_type = struct.unpack("<I", content[:4])[0]
            if shape_type != 5:
                continue
            n_parts, n_points = struct.unpack("<II", content[36:44])
            parts = list(struct.unpack("<%dI" % n_parts, content[44:44 + 4 * n_parts]))
            off = 44 + 4 * n_parts
            pts = struct.unpack("<%dd" % (n_points * 2), content[off:off + 16 * n_points])
            all_pts = [(pts[i * 2], pts[i * 2 + 1]) for i in range(n_points)]
            # A multi-part polygon must be split into its rings; concatenating
            # the parts would create a fake edge between islands and corrupt
            # the scanline parity.
            bounds = parts + [n_points]
            for k in range(n_parts):
                ring = all_pts[bounds[k]:bounds[k + 1]]
                if len(ring) < 3:
                    continue
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                self.rings.append((min(xs), min(ys), max(xs), max(ys), ring))
                if max(ys) < lo or min(ys) > hi:
                    continue
                n = len(ring)
                j = n - 1
                for i in range(n):
                    xi, yi = ring[i]
                    xj, yj = ring[j]
                    if yi == yj:
                        j = i
                        continue
                    ymin, ymax = (yi, yj) if yi < yj else (yj, yi)
                    for ri, lat in enumerate(lats):
                        if ymin <= lat < ymax:
                            x = xi + (lat - yi) * (xj - xi) / (yj - yi)
                            self.crossings[ri].append((x, 1 if yj > yi else -1))
                    j = i

    def is_land(self, lat: float, lon: float) -> bool:
        # Even-odd rule over the scanline crossings at this latitude.  The
        # crossings are pre-sorted, so only the suffix east of ``lon`` is
        # scanned.
        import bisect
        ri = self._row_index(lat)
        if ri is None:
            return False
        row = self.crossings[ri]
        i = bisect.bisect_right(row, (lon, -1))
        return (len(row) - i) % 2 == 1


# ---------------------------------------------------------------------------
# OSM Overpass
# ---------------------------------------------------------------------------

OVERPASS = "https://overpass-api.de/api/interpreter"
UA = "situation3d-campaign-map/0.1 (research; contact: local)"


def overpass(query: str, timeout: int = 180, retries: int = 4) -> dict:
    body = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(OVERPASS, data=body,
                                     headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:      # 429/502/timeout are common on Overpass
            last = exc
            time.sleep(5 * (attempt + 1))
    raise last


def fetch_roads_urban(profile: dict, timeout: int = 180) -> Tuple[List[dict], List[Tuple[float, float]]]:
    w, s, e, n = profile["west"], profile["south"], profile["east"], profile["north"]
    bbox = "%s,%s,%s,%s" % (s, w, n, e)
    roads: List[dict] = []
    urban: List[Tuple[float, float]] = []
    q_roads = """
    [out:json][timeout:%(t)d];
    way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"](%(b)s);
    out geom;
    """ % {"b": bbox, "t": timeout}
    for el in overpass(q_roads).get("elements") or []:
        tags = el.get("tags") or {}
        geom = el.get("geometry") or []
        if tags.get("highway") and geom:
            roads.append({"class": tags["highway"],
                          "points": [(p["lon"], p["lat"]) for p in geom]})
    # Residential areas only need their centre for the urban terrain flag.
    q_urban = """
    [out:json][timeout:%(t)d];
    way["landuse"="residential"](%(b)s);
    out center;
    """ % {"b": bbox, "t": timeout}
    for el in overpass(q_urban).get("elements") or []:
        c = el.get("center") or {}
        if c:
            urban.append((c["lon"], c["lat"]))
    return roads, urban


# ---------------------------------------------------------------------------
# Hex rasterisation
# ---------------------------------------------------------------------------

ROAD_RANK = {"motorway": 3, "trunk": 3, "primary": 2, "secondary": 1, "tertiary": 1}
ROAD_SPEED = {0: 40.0, 1: 40.0, 2: 60.0, 3: 90.0}


def _prepare_grid(profile: dict) -> dict:
    """Derive an equidistant pixel grid centred on the profile centre.

    Odd-r offset hexes are laid out as
        px = (col + 0.5 * (row & 1)) * width
        py = row * height,   height = sqrt(3)/2 * width
    The pixel rectangle is centred on the profile's centre, then converted to
    lon/lat with local metres-per-degree factors.
    """
    p = dict(profile)
    width_m = float(p["scale_m"])
    height_m = width_m * math.sqrt(3.0) / 2.0
    lat0, lon0 = p["center_lat"], p["center_lon"]
    deg_lon_per_m = 1.0 / (111320.0 * math.cos(math.radians(lat0)))
    deg_lat_per_m = 1.0 / 110540.0

    # Pixel bounds of the grid (same formula as hexmap.pos_to_lonlat).
    cx = ((p["cols"] - 1) * width_m + 0.5 * width_m) / 2.0
    cy = (p["rows"] - 1) * height_m / 2.0
    p["_width_m"] = width_m
    p["_height_m"] = height_m
    p["_cx"] = cx
    p["_cy"] = cy
    p["_deg_lon_per_m"] = deg_lon_per_m
    p["_deg_lat_per_m"] = deg_lat_per_m
    p["_center_lat"] = lat0
    p["_center_lon"] = lon0

    corner_lon = []
    corner_lat = []
    for row in (0, p["rows"] - 1):
        for col in (0, p["cols"] - 1):
            lon, lat = offset_to_lonlat(p, row, col)
            corner_lon.append(lon)
            corner_lat.append(lat)
    # Account for the half-width stagger of odd rows.
    p["west"] = min(corner_lon) - width_m / 2.0 * deg_lon_per_m
    p["east"] = max(corner_lon) + width_m / 2.0 * deg_lon_per_m
    p["south"] = min(corner_lat) - height_m / 2.0 * deg_lat_per_m
    p["north"] = max(corner_lat) + height_m / 2.0 * deg_lat_per_m
    return p


def _crs(lat: float) -> float:
    return 111.32 * math.cos(math.radians(lat))


# ---------------------------------------------------------------------------
# Shapefile (Natural Earth land) — scanline rasteriser
# ---------------------------------------------------------------------------


def lonlat_to_rowcol(profile, lon: float, lat: float) -> Tuple[int, int]:
    """Nearest hex (odd-r) for a lon/lat, using the pixel layout."""
    px = profile["_cx"] + (lon - profile["_center_lon"]) / profile["_deg_lon_per_m"]
    py = profile["_cy"] - (lat - profile["_center_lat"]) / profile["_deg_lat_per_m"]
    row = int(round(py / profile["_height_m"]))
    col = int(round(px / profile["_width_m"] - 0.5 * (row & 1)))
    row = max(0, min(profile["rows"] - 1, row))
    col = max(0, min(profile["cols"] - 1, col))
    return row, col


def offset_to_lonlat(profile, row: int, col: int) -> Tuple[float, float]:
    """Centre of odd-r hex (row, col) in WGS84."""
    px = (col + 0.5 * (row & 1)) * profile["_width_m"]
    py = row * profile["_height_m"]
    lon = profile["_center_lon"] + (px - profile["_cx"]) * profile["_deg_lon_per_m"]
    lat = profile["_center_lat"] - (py - profile["_cy"]) * profile["_deg_lat_per_m"]
    return lon, lat


def build_cells(profile, land: "LandRaster", terrain: TerrainSampler,
                roads: List[dict], urban: List[Tuple[float, float]],
                coast_probe_km: float = 6.0) -> List[List[dict]]:
    rows, cols = profile["rows"], profile["cols"]
    scale_km = profile["scale_m"] / 1000.0
    cells = [[None] * cols for _ in range(rows)]

    # Precompute road point -> cell and neighbour road direction.
    road_edges: Dict[int, List[int]] = {}
    road_rank: Dict[int, int] = {}
    for way in roads:
        rank = ROAD_RANK.get(way["class"], 0)
        pts = way["points"]
        prev = None
        for lon, lat in pts:
            r, c = lonlat_to_rowcol(profile, lon, lat)
            p = hexmap.encode(r, c)
            road_rank[p] = max(road_rank.get(p, 0), rank)
            if prev is not None and prev != p:
                a, b = prev, p
                road_edges.setdefault(a, []).append(b)
                road_edges.setdefault(b, []).append(a)
            prev = p

    urban_set = set()
    for lon, lat in urban:
        r, c = lonlat_to_rowcol(profile, lon, lat)
        urban_set.add(hexmap.encode(r, c))

    for r in range(rows):
        for c in range(cols):
            p = hexmap.encode(r, c)
            lon, lat = offset_to_lonlat(profile, r, c)
            elev = terrain.elevation(lon, lat)
            is_land = land.is_land(lat, lon)
            if not is_land:
                cond = hexmap.RIVER        # water; reused as the sea terrain
            elif p in urban_set:
                cond = hexmap.URBAN
            else:
                cond = hexmap.OPEN
            # Coastline: a land cell whose neighbour is water becomes soft
            # ground (beach) so landing is possible but slower.
            coastal = False
            if is_land:
                for dr, dc in ((0, 1), (-1, 1), (-1, 0), (0, -1), (1, 0), (1, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        nlon, nlat = offset_to_lonlat(profile, nr, nc)
                        if not land.is_land(nlat, nlon):
                            coastal = True
                            break
            if coastal and cond == hexmap.OPEN:
                cond = hexmap.SOFT
            cells[r][c] = {
                "elev": int(round(elev)),
                "cond": cond,
                "roads": [0] * 6,
                "rivers": [0] * 6,
            }

    # Road classes per direction.
    for p, nbs in road_edges.items():
        r, c = hexmap.decode(p)
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        cell = cells[r][c]
        order = hexmap.neighbors(r, c, rows, cols)
        rank = road_rank.get(p, 0)
        for nb in nbs:
            if nb in order:
                idx = order.index(nb)
                cell["roads"][idx] = rank
        # A road cell without a recorded direction still counts as a road for
        # movement cost; mark the first direction too.
        if rank and not any(cell["roads"]):
            cell["roads"][0] = rank
    return cells


def build_neighbors(cells, rows, cols) -> None:
    for r in range(rows):
        for c in range(cols):
            cells[r][c]["pos"] = hexmap.encode(r, c)
            cells[r][c]["r"] = r
            cells[r][c]["c"] = c
            cells[r][c]["neighbors"] = hexmap.neighbors(r, c, rows, cols)
            cells[r][c]["nb"] = cells[r][c]["neighbors"]


def build_cost(cells, rows, cols, scale_m: float) -> list:
    """4 x rows x cols cost dicts matching the public contract."""
    from wargame_core.hexmap import HexMap
    hmap = HexMap({"map_id": 0, "max_row": rows, "max_column": cols,
                   "scale": scale_m, "ele_grade": 10,
                   "map_data": cells})
    modes = ["vehicle", "march", "infantry", "air"]
    out = []
    for i, mode in enumerate(modes):
        grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                p = hexmap.encode(r, c)
                entry = {}
                for nb in hmap.neighbors(p):
                    if nb < 0:
                        continue
                    cost = hmap.move_cost(p, nb, mode)
                    if cost != float("inf"):
                        entry[nb] = float(cost)
                row.append(entry)
            grid.append(row)
        out.append(grid)
    return out


def build_see(cells, rows, cols, scale_m: float) -> "numpy.ndarray":
    import numpy as np
    from wargame_core.los import LineOfSight
    from wargame_core.hexmap import HexMap
    hmap = HexMap({"map_id": 0, "max_row": rows, "max_column": cols,
                   "scale": scale_m, "ele_grade": 10, "map_data": cells})
    los = LineOfSight(hmap)
    # 10 modes: ground-ground + combinations of low/very-low/high bands.
    modes = ["ground"] * 1 + ["低空"] * 4 + ["超低空"] * 3 + ["高空"] * 2
    arr = np.zeros((10, rows, cols, rows, cols), dtype=bool)
    positions = [hexmap.encode(r, c) for r in range(rows) for c in range(cols)]
    for a in positions:
        ar, ac = hexmap.decode(a)
        for b in positions:
            br, bc = hexmap.decode(b)
            if a == b:
                for m in range(10):
                    arr[m, ar, ac, br, bc] = True
                continue
            vis = los.visible(a, b)
            for m in range(10):
                arr[m, ar, ac, br, bc] = vis
    return arr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="taiwan_south", choices=sorted(PROFILES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--land-shp", default="/tmp/ne_land/ne_10m_land.shp")
    ap.add_argument("--terrain-dir", default=os.path.join(
        REPO, "situation3d", "web", "tiles", "terrain"))
    ap.add_argument("--terrain-zoom", type=int, default=10)
    ap.add_argument("--with-see", action="store_true",
                    help="also write the full 10-mode see.npz (large and slow; "
                         "wargame_core computes LOS on the fly)")
    ap.add_argument("--no-osm", action="store_true")
    args = ap.parse_args(argv)

    profile = _prepare_grid(PROFILES[args.profile])
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    print("[map] profile=%s bbox=(%s,%s,%s,%s) grid=%dx%d scale=%sm"
          % (args.profile, profile["west"], profile["south"],
             profile["east"], profile["north"], profile["rows"],
             profile["cols"], profile["scale_m"]), flush=True)
    lats = sorted(set(round(offset_to_lonlat(profile, r, 0)[1], 9)
                      for r in range(profile["rows"])))
    land = LandRaster(args.land_shp, lats)
    print("[map] land raster rows=%d (%.1fs)" % (len(lats), time.time() - t0), flush=True)
    terrain = TerrainSampler(args.terrain_dir, args.terrain_zoom)
    t1 = time.time()
    if args.no_osm:
        roads, urban = [], []
    else:
        roads, urban = fetch_roads_urban(profile)
    print("[map] osm roads=%d urban_points=%d (%.1fs)"
          % (len(roads), len(urban), time.time() - t1), flush=True)

    t2 = time.time()
    cells = build_cells(profile, land, terrain, roads, urban)
    build_neighbors(cells, profile["rows"], profile["cols"])
    print("[map] cells done (%.1fs)" % (time.time() - t2), flush=True)

    doc = {
        "map_id": profile["map_id"],
        "name": profile["name"],
        "start_elev": 0,
        "ele_grade": profile["elev_grade"],
        "coordinate_number_length": 4,
        "coordinate_number_mode": 0,
        "coordinate_system": "oddr",
        "upper_left_longitude": profile["west"],
        "upper_left_latitude": profile["north"],
        "max_row": profile["rows"],
        "max_column": profile["cols"],
        "scale": profile["scale_m"],
        "bbox": [profile["west"], profile["south"], profile["east"], profile["north"]],
        "grid": {
            "center_lon": profile["_center_lon"],
            "center_lat": profile["_center_lat"],
            "width_m": profile["_width_m"],
            "height_m": profile["_height_m"],
            "cols": profile["cols"],
            "rows": profile["rows"],
        },
        "provenance": {
            "land": "Natural Earth 1:10m land (public domain)",
            "roads_urban": "OpenStreetMap contributors (ODbL)",
            "elevation": "Mapzen/AWS Terrarium tiles",
            "note": "虚构演示用六角地图；不是现实作战地图。",
        },
        "map_data": cells,
    }
    map_path = os.path.join(args.out, "basic.json")
    with open(map_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    print("[map] wrote %s (%.1f KB)" % (map_path, os.path.getsize(map_path) / 1024.0), flush=True)

    cost = build_cost(cells, profile["rows"], profile["cols"], profile["scale_m"])
    cost_path = os.path.join(args.out, "cost.pickle")
    with open(cost_path, "wb") as fh:
        pickle.dump(cost, fh, protocol=4)
    print("[map] wrote %s (%.1f KB)" % (cost_path, os.path.getsize(cost_path) / 1024.0), flush=True)

    if args.with_see:
        t3 = time.time()
        see = build_see(cells, profile["rows"], profile["cols"], profile["scale_m"])
        see_path = os.path.join(args.out, "%dsee.npz" % profile["map_id"])
        import numpy as np
        np.savez_compressed(see_path, data=see)
        print("[map] wrote %s (%.1f KB, %.1fs)" % (
            see_path, os.path.getsize(see_path) / 1024.0, time.time() - t3), flush=True)

    print("[map] total %.1fs" % (time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
