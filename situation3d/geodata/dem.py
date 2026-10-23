"""Terrarium DEM source (online fetch + offline cache replay).

Terrarium tiles encode elevation as ``R*256 + G + B/256 - 32768`` metres.
The same PNG decoder as ``build_campaign_map.py`` is used so no Pillow/GDAL
dependency is required.
"""

from __future__ import annotations

import math
import struct
import urllib.request
import zlib
from typing import Dict, List, Optional, Tuple

from geodata.cache import CacheCorrupt, CacheStore, OfflineCacheMiss

TERRARIUM_URLS = [
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/%d/%d/%d.png",
    "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/%d/%d/%d.png",
]


class OnlineDemError(RuntimeError):
    pass


def _png_chunks(data: bytes):
    pos = 8
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        yield ctype, data[pos + 8:pos + 8 + length]
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


class DemSource(object):
    def __init__(self, cache: CacheStore, zoom: int = 10,
                 timeout_s: int = 30, urls: Optional[List[str]] = None):
        self.cache = cache
        self.zoom = int(zoom)
        self.timeout_s = int(timeout_s)
        self.urls = urls or list(TERRARIUM_URLS)
        self._tiles: Dict[Tuple[int, int], tuple] = {}
        self.fetched_network = False

    def _tile(self, x: int, y: int):
        key = (x, y)
        if key in self._tiles:
            return self._tiles[key]
        name = "%d/%d/%d.png" % (self.zoom, x, y)
        if self.cache.mode == "offline" or (self.cache.mode == "auto" and self.cache.has("dem", name)):
            data = self.cache.read_bytes("dem", name)
        else:
            data = None
            last = None
            for tpl in self.urls:
                try:
                    req = urllib.request.Request(tpl % (self.zoom, x, y),
                                                 headers={"User-Agent": "situation3d-geo-engine/1.0"})
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                        data = resp.read()
                    break
                except Exception as exc:
                    last = exc
            if data is None:
                raise OnlineDemError("DEM tile %s failed: %s" % (name, last))
            self.fetched_network = True
            self.cache.write_bytes("dem", name, data)
        try:
            w, h, rgb = decode_png_rgb(data)
        except Exception as exc:
            self.cache.quarantine("dem", name)
            raise CacheCorrupt("corrupt DEM tile %s: %s" % (name, exc))
        self._tiles[key] = (w, h, rgb)
        return self._tiles[key]

    def elevation(self, lon: float, lat: float) -> float:
        z = self.zoom
        n = 2 ** z
        xf = (lon + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        yf = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
        x, y = int(xf), int(yf)
        w, h, rgb = self._tile(x, y)
        px = min(w - 1, max(0, int((xf - x) * w)))
        py = min(h - 1, max(0, int((yf - y) * h)))
        i = (py * w + px) * 3
        return rgb[i] * 256.0 + rgb[i + 1] + rgb[i + 2] / 256.0 - 32768.0

    def sample_grid(self, bbox, step_deg: float):
        """Return a rectangular elevation grid over ``bbox``.

        ``bbox`` is ``(lon_min, lat_min, lon_max, lat_max)``.
        """
        lon0, lat0, lon1, lat1 = [float(v) for v in bbox]
        cols = max(2, int(math.ceil((lon1 - lon0) / step_deg)) + 1)
        rows = max(2, int(math.ceil((lat1 - lat0) / step_deg)) + 1)
        values = []
        for r in range(rows):
            lat = lat0 + (lat1 - lat0) * r / float(rows - 1)
            row = []
            for c in range(cols):
                lon = lon0 + (lon1 - lon0) * c / float(cols - 1)
                try:
                    row.append(round(self.elevation(lon, lat), 1))
                except (OnlineDemError, OfflineCacheMiss):
                    raise
            values.append(row)
        return {"lon0": lon0, "lat0": lat0, "lon1": lon1, "lat1": lat1,
                "cols": cols, "rows": rows, "values": values}
