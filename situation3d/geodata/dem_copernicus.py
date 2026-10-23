"""Copernicus GLO-30 DEM source (1-degree COG tiles on AWS, cached locally).

Online builds extract the requested bbox window from the remote Cloud
Optimized GeoTIFF with ``gdal_translate`` (``/vsicurl/``), convert it to an
ESRI ASCII grid and cache the text; offline rebuilds replay that cache and
never touch the network.  Sampling is pure Python (``geodata.aaigrid``).
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

from geodata.aaigrid import bilinear, parse_aaigrid, sample_regular_grid
from geodata.cache import CacheCorrupt, CacheStore, OfflineCacheMiss

AWS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


class OnlineDemError(RuntimeError):
    pass


def tile_name(lon: float, lat: float) -> str:
    lon_i = int(math.floor(lon))
    lat_i = int(math.floor(lat))
    ns = "N" if lat_i >= 0 else "S"
    ew = "E" if lon_i >= 0 else "W"
    return "Copernicus_DSM_COG_10_%s%02d_00_%s%03d_00_DEM" % (
        ns, abs(lat_i), ew, abs(lon_i))


def tile_url(name: str, base: str = AWS_BASE) -> str:
    return "%s/%s/%s.tif" % (base.rstrip("/"), name, name)


def _tiles_for_bbox(bbox) -> List[str]:
    west, south, east, north = [float(v) for v in bbox]
    names = []
    lon_i = int(math.floor(west))
    while lon_i <= int(math.floor(east)):
        lat_i = int(math.floor(south))
        while lat_i <= int(math.floor(north)):
            names.append(tile_name(lon_i, lat_i))
            lat_i += 1
        lon_i += 1
    return sorted(set(names))


class CopernicusDemSource(object):
    provenance_label = "copernicus-glo30"

    def __init__(self, cache: CacheStore, timeout_s: int = 180,
                 gdal_translate: str = "gdal_translate",
                 base_url: str = AWS_BASE):
        self.cache = cache
        self.timeout_s = int(timeout_s)
        self.gdal_translate = gdal_translate
        self.base_url = base_url
        self.fetched_network = False

    def _run_translate(self, name: str, bbox, out_path: str) -> None:
        lon0, lat0, lon1, lat1 = [float(v) for v in bbox]
        cmd = [self.gdal_translate, "-q", "-of", "AAIGrid",
               "-projwin", "%.8f" % lon0, "%.8f" % lat1,
               "%.8f" % lon1, "%.8f" % lat0,
               "/vsicurl/%s" % tile_url(name, self.base_url), out_path]
        try:
            subprocess.run(cmd, check=True, timeout=self.timeout_s,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.SubprocessError) as exc:
            raise OnlineDemError("gdal_translate failed for %s: %s"
                                 % (name, getattr(exc, "stderr", exc))) from exc

    def _tile_grid(self, name: str, bbox) -> Dict:
        lon0, lat0, lon1, lat1 = [float(v) for v in bbox]
        key = "%s:%.6f,%.6f,%.6f,%.6f" % (name, lon0, lat0, lon1, lat1)
        if self.cache.mode == "offline" or (
                self.cache.mode == "auto" and self.cache.has("dem_cop", key)):
            text = self.cache.read_bytes("dem_cop", key).decode("ascii")
        else:
            fd, path = tempfile.mkstemp(suffix=".asc")
            os.close(fd)
            try:
                self._run_translate(name, bbox, path)
                with open(path, "r", encoding="ascii") as fh:
                    text = fh.read()
            finally:
                if os.path.exists(path):
                    os.unlink(path)
            self.fetched_network = True
            self.cache.write_bytes("dem_cop", key, text.encode("ascii"))
        try:
            return parse_aaigrid(text)
        except ValueError as exc:
            self.cache.quarantine("dem_cop", key)
            raise CacheCorrupt("corrupt Copernicus window %s: %s" % (key, exc))

    def _grids(self, bbox) -> List[Dict]:
        grids = []
        for name in _tiles_for_bbox(bbox):
            lon_i = int(name.split("_")[6][1:])
            lat_i = int(name.split("_")[4][1:])
            pad = 0.0005
            clip = [max(float(bbox[0]) - pad, lon_i),
                    max(float(bbox[1]) - pad, lat_i),
                    min(float(bbox[2]) + pad, lon_i + 1.0),
                    min(float(bbox[3]) + pad, lat_i + 1.0)]
            if clip[0] >= clip[2] or clip[1] >= clip[3]:
                continue
            grids.append(self._tile_grid(name, clip))
        if not grids:
            raise OnlineDemError("no Copernicus tile covers bbox %r" % (bbox,))
        return grids

    def elevation(self, lon: float, lat: float) -> float:
        bbox = [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001]
        for grid in self._grids(bbox):
            value = bilinear(grid, lon, lat)
            if value is not None:
                return float(value)
        raise OnlineDemError("no Copernicus sample at %.6f,%.6f" % (lon, lat))

    def sample_grid(self, bbox, step_deg: float):
        grids = self._grids(bbox)
        lon0, lat0, lon1, lat1 = [float(v) for v in bbox]
        cols = max(2, int(math.ceil((lon1 - lon0) / step_deg)) + 1)
        rows = max(2, int(math.ceil((lat1 - lat0) / step_deg)) + 1)
        values = []
        for r in range(rows):
            lat = lat0 + (lat1 - lat0) * r / float(rows - 1)
            row = []
            for c in range(cols):
                lon = lon0 + (lon1 - lon0) * c / float(cols - 1)
                value = None
                for grid in grids:
                    value = bilinear(grid, lon, lat)
                    if value is not None:
                        break
                row.append(None if value is None else round(value, 1))
            values.append(row)
        return {"lon0": lon0, "lat0": lat0, "lon1": lon1, "lat1": lat1,
                "cols": cols, "rows": rows, "values": values}
