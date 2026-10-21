"""GEBCO 15 arc-second bathymetry source via the download.gebco.net API.

The service accepts a bbox basket, renders a GeoTIFF subset and exposes it
under ``/api/queue/download/<basketId>``.  The window is converted to an
ASCII grid with ``gdal_translate`` and cached, so offline rebuilds replay the
cache.  Grid values are elevation in metres (negative below sea level).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from typing import Dict, Optional

from geodata.aaigrid import parse_aaigrid, sample_regular_grid
from geodata.cache import CacheCorrupt, CacheStore, OfflineCacheMiss

BASE_URL = "https://download.gebco.net"
GRID_ID = 1          # gebco_2026_global, 15 arc-second
DATA_SOURCE_ID = 1   # Bathymetry (land + sea surface elevation)
FORMAT_GEOTIFF = 2


class OnlineBathymetryError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, timeout_s: int) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "User-Agent": "situation3d-geo-engine/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout_s: int) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "situation3d-geo-engine/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bbox_geotiff(bbox, timeout_s: int = 60, poll_s: float = 5.0,
                       base_url: str = BASE_URL) -> bytes:
    """Submit a bbox to the GEBCO queue and return the rendered GeoTIFF bytes."""
    west, south, east, north = [float(v) for v in bbox]
    payload = {
        "id": "0", "email": "situation3d@localhost",
        "submission_date": time.strftime("%Y-%m-%dT%H:%M:%S.000000"),
        "processing_status": "new",
        "items": [{
            "id": 0, "grid_id": GRID_ID, "data_source_ids": [DATA_SOURCE_ID],
            "formats": [FORMAT_GEOTIFF], "left": west, "right": east,
            "bottom": south, "top": north,
        }],
    }
    basket = _post_json(base_url + "/api/queue", payload, timeout_s)
    basket_id = basket.get("basketId")
    if not basket_id:
        raise OnlineBathymetryError("GEBCO queue rejected request: %s" % basket)
    deadline = time.time() + timeout_s * 5
    status = None
    while time.time() < deadline:
        status = _get_json(base_url + "/api/queue/status/" + basket_id, timeout_s)
        if status.get("status") == "finished":
            break
        if status.get("status") in ("failed", "error"):
            raise OnlineBathymetryError("GEBCO job failed: %s" % status)
        time.sleep(poll_s)
    if not status or status.get("status") != "finished":
        raise OnlineBathymetryError("GEBCO job timed out: %s" % status)
    req = urllib.request.Request(
        base_url + "/api/queue/download/" + basket_id,
        headers={"User-Agent": "situation3d-geo-engine/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s * 3) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".tif"):
                return zf.read(name)
    raise OnlineBathymetryError("GEBCO download has no GeoTIFF: %s" % basket_id)


class GebcoSource(object):
    provenance_label = "gebco-2026-15s"

    def __init__(self, cache: CacheStore, timeout_s: int = 60,
                 gdal_translate: str = "gdal_translate",
                 base_url: str = BASE_URL):
        self.cache = cache
        self.timeout_s = int(timeout_s)
        self.gdal_translate = gdal_translate
        self.base_url = base_url
        self.fetched_network = False

    @staticmethod
    def _key(bbox) -> str:
        return "%.6f,%.6f,%.6f,%.6f" % tuple(float(v) for v in bbox)

    def grid(self, bbox) -> Dict:
        key = self._key(bbox)
        if self.cache.mode == "offline" or (
                self.cache.mode == "auto" and self.cache.has("bathymetry", key)):
            text = self.cache.read_bytes("bathymetry", key).decode("ascii")
        else:
            tif = fetch_bbox_geotiff(bbox, timeout_s=self.timeout_s,
                                     base_url=self.base_url)
            fd_tif, path_tif = tempfile.mkstemp(suffix=".tif")
            fd_asc, path_asc = tempfile.mkstemp(suffix=".asc")
            os.close(fd_tif)
            os.close(fd_asc)
            try:
                with open(path_tif, "wb") as fh:
                    fh.write(tif)
                cmd = [self.gdal_translate, "-q", "-of", "AAIGrid",
                       path_tif, path_asc]
                try:
                    subprocess.run(cmd, check=True,
                                   timeout=self.timeout_s,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
                except (OSError, subprocess.SubprocessError) as exc:
                    raise OnlineBathymetryError(
                        "gdal_translate failed: %s" % getattr(exc, "stderr", exc)
                    ) from exc
                with open(path_asc, "r", encoding="ascii") as fh:
                    text = fh.read()
            finally:
                for path in (path_tif, path_asc):
                    if os.path.exists(path):
                        os.unlink(path)
            self.fetched_network = True
            self.cache.write_bytes("bathymetry", key, text.encode("ascii"))
        try:
            return parse_aaigrid(text)
        except ValueError as exc:
            self.cache.quarantine("bathymetry", key)
            raise CacheCorrupt("corrupt GEBCO window %s: %s" % (key, exc))

    def sample_grid(self, bbox, step_deg: float):
        return sample_regular_grid(self.grid(bbox), bbox, step_deg)

    def elevation(self, lon: float, lat: float) -> Optional[float]:
        from geodata.aaigrid import bilinear
        return bilinear(self.grid([lon - 0.005, lat - 0.005,
                                   lon + 0.005, lat + 0.005]), lon, lat)
