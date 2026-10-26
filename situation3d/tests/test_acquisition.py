"""Parallel acquisition, fallback, raster tiling and cache reuse tests."""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from resources.acquire import (AcquireError, acquire_tiles,  # noqa: E402
                               probe_parallel, rank_probes)
from resources.processing import TilesetSpec, verify_tileset  # noqa: E402
from resources.sources import SourceCandidate, raster_candidate  # noqa: E402
from resources.store import ResourceStore  # noqa: E402

try:
    from osgeo import gdal, osr
except ImportError:  # pragma: no cover
    gdal = None
    osr = None

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAF"
    "gwJ/lK3Q6wAAAABJRU5ErkJggg==") + b"pad" * 200
SPEC = TilesetSpec(name="test-tiles", bbox=(119.5, 22.2, 120.8, 23.6),
                   min_zoom=8, max_zoom=9)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        path = self.path
        if path.startswith("/broken/"):
            self.send_error(500)
            return
        if path.startswith("/slow/"):
            time.sleep(0.12)
        if path.startswith("/fast/"):
            time.sleep(0.05)
        if path.startswith("/flaky/"):
            # Valid only for the probe sample tile (z7 x106 y55); every other
            # bulk tile fails, so acquisition must fall back to the next source.
            parts = path.strip("/").split("/")
            if not (parts[1] == "9" and parts[2] == "426" and parts[3].startswith("222.")):
                self.send_error(500)
                return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(PNG)))
        self.end_headers()
        self.wfile.write(PNG)

    def log_message(self, *args):
        pass


def candidate(base, name, priority=10):
    return SourceCandidate(
        source_id=name, method="xyz",
        params={"url": base + "/" + name + "/{z}/{x}/{y}.png",
                "ext": "png", "max_zoom": 10, "license": "test"},
        priority=priority)


class AcquisitionTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.server.server_port
        self.tmp = tempfile.mkdtemp(prefix="acq-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_probe_ranks_fast_before_slow_and_broken(self):
        probes = probe_parallel([
            candidate(self.base, "slow"),
            candidate(self.base, "broken"),
            candidate(self.base, "fast"),
        ], SPEC, timeout_s=2.0)
        ranked = rank_probes(probes)
        self.assertEqual(ranked[0].candidate.source_id, "fast")
        self.assertNotIn("broken", [p.candidate.source_id for p in ranked])

    def test_fallback_when_bulk_fetch_fails(self):
        result = acquire_tiles(
            self.store, SPEC,
            candidates=[candidate(self.base, "flaky", priority=0),
                        candidate(self.base, "fast", priority=10)],
            workers=4, probe_timeout_s=2.0, timeout_s=10)
        self.assertFalse(result.cached)
        self.assertEqual(result.source["source_id"], "fast")
        results = {a["source_id"]: a["result"] for a in result.attempts}
        self.assertEqual(results.get("flaky"), "failed")
        self.assertEqual(results.get("fast"), "ok")
        self.assertTrue(verify_tileset(result.entry.path)["ok"])

    def test_cache_reuse_without_network(self):
        first = acquire_tiles(self.store, SPEC,
                              candidates=[candidate(self.base, "fast")],
                              probe_timeout_s=2.0, timeout_s=10)
        self.server.shutdown()
        self.server.server_close()
        second = acquire_tiles(self.store, SPEC,
                               candidates=[candidate(self.base, "fast")],
                               probe_timeout_s=0.5, timeout_s=2)
        self.assertTrue(second.cached)
        self.assertEqual(second.entry.content_hash, first.entry.content_hash)

    def test_offline_without_cache_fails(self):
        with self.assertRaises(AcquireError):
            acquire_tiles(self.store, SPEC, mode="offline",
                          candidates=[candidate(self.base, "fast")])

    def test_manifest_and_registry(self):
        result = acquire_tiles(self.store, SPEC,
                               candidates=[candidate(self.base, "fast")],
                               probe_timeout_s=2.0, timeout_s=10)
        manifest = result.manifest
        self.assertEqual(manifest["schema"], "s3d.tileset.v1")
        self.assertEqual(manifest["spec_hash"], SPEC.hash())
        self.assertGreater(manifest["tile_count"], 0)
        entry = self.store.get(result.entry.entry_id)
        self.assertEqual(entry.kind, "tile_layer")
        self.assertTrue(self.store.verify(entry.entry_id)["ok"])
        self.assertEqual(entry.notes.get("spec_hash"), SPEC.hash())


@unittest.skipIf(gdal is None, "GDAL python bindings not installed")
class RasterTilingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="raster-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))
        self.raster = os.path.join(self.tmp, "rgb.tif")
        self._make_raster()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_raster(self):
        w, s, e, n = 120.0, 22.8, 120.2, 23.0
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(self.raster, 256, 256, 3, gdal.GDT_Byte)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds.SetGeoTransform((w, (e - w) / 256.0, 0, n, 0, -(n - s) / 256.0))
        for band in range(1, 4):
            data = bytes([(band * 40 + i // 4) % 256 for i in range(256 * 256)])
            ds.GetRasterBand(band).WriteRaster(0, 0, 256, 256, data)
        ds = None

    def test_download_raster_and_tile_locally(self):
        spec = TilesetSpec(name="raster-test", bbox=(120.0, 22.8, 120.2, 23.0),
                           min_zoom=0, max_zoom=1)
        result = acquire_tiles(
            self.store, spec, candidates=[
                raster_candidate("file://" + self.raster)],
            probe_timeout_s=2.0, timeout_s=120)
        self.assertEqual(result.source["method"], "raster")
        self.assertFalse(result.source.get("hillshade"))
        self.assertGreater(result.manifest["tile_count"], 0)
        check = verify_tileset(result.entry.path)
        self.assertTrue(check["ok"])
        self.assertEqual(result.entry.quality, "measured")

    def test_single_band_uses_hillshade(self):
        elevation = os.path.join(self.tmp, "elev.tif")
        w, s, e, n = 120.0, 22.8, 120.2, 23.0
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(elevation, 128, 128, 1, gdal.GDT_Float32)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds.SetGeoTransform((w, (e - w) / 128.0, 0, n, 0, -(n - s) / 128.0))
        data = [float((x * 7 + y * 3) % 400) for y in range(128) for x in range(128)]
        ds.GetRasterBand(1).WriteRaster(0, 0, 128, 128,
                                        b"".join(__import__("struct").pack("<f", v) for v in data))
        ds = None
        spec = TilesetSpec(name="elev-test", bbox=(120.0, 22.8, 120.2, 23.0),
                           min_zoom=0, max_zoom=1, layer="terrain")
        result = acquire_tiles(
            self.store, spec,
            candidates=[raster_candidate("file://" + elevation)],
            probe_timeout_s=2.0, timeout_s=120)
        self.assertTrue(result.source.get("hillshade"))
        self.assertEqual(result.entry.quality, "derived")
        self.assertTrue(verify_tileset(result.entry.path)["ok"])


if __name__ == "__main__":
    unittest.main()
