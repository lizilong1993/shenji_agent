"""Envpack canopy layer conversion and engine sampling."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from env.pack_env import EnvPackEnvironment  # noqa: E402
from resources.envpack_layers import add_canopy_layer, raster_to_grid  # noqa: E402
from resources.scanners import scan_envpack  # noqa: E402
from resources.store import ResourceStore  # noqa: E402

try:
    from osgeo import gdal, osr
except ImportError:  # pragma: no cover
    gdal = None
    osr = None

GRID = {"center_lon": 120.0, "center_lat": 22.9, "cols": 4, "rows": 4,
        "width_m": 2000.0}
BBOX = (120.0, 22.9, 120.02, 22.92)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def make_envpack(root):
    write_json(os.path.join(root, "manifest.json"), {
        "schema": "msgeo.envpack.v2", "pack_id": "p1", "theatre": "tw",
        "built": "2026-10-01", "bbox": list(BBOX), "grid": dict(GRID),
        "layers": {},
        "fallback_basic_json": None,
        "provenance": {"mode": "offline"},
    })
    return root


def make_canopy_tif(path):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, 64, 64, 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform((BBOX[0], (BBOX[2] - BBOX[0]) / 64.0, 0, BBOX[3], 0,
                        -(BBOX[3] - BBOX[1]) / 64.0))
    values = [float(10 + (i % 10)) for i in range(64 * 64)]
    import struct
    ds.GetRasterBand(1).WriteRaster(
        0, 0, 64, 64, b"".join(struct.pack("<f", v) for v in values))
    ds = None


def make_class_tif(path):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, 64, 64, 1, gdal.GDT_Byte)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetGeoTransform((BBOX[0], (BBOX[2] - BBOX[0]) / 64.0, 0, BBOX[3], 0,
                        -(BBOX[3] - BBOX[1]) / 64.0))
    values = []
    for y in range(64):
        for x in range(64):
            if x < 32 and y < 32:
                values.append(50)   # urban
            elif x >= 32 and y < 32:
                values.append(10)   # tree
            else:
                values.append(80)   # water
    ds.GetRasterBand(1).WriteRaster(0, 0, 64, 64, bytes(values))
    ds = None


class CanopyLayerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="canopy-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))
        self.pack = make_envpack(os.path.join(self.tmp, "pack"))
        self.env_entry = self.store.register(scan_envpack(self.pack))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_engine_samples_manual_canopy_layer(self):
        write_json(os.path.join(self.pack, "canopy", "grid.json"), {
            "lon0": BBOX[0], "lat0": BBOX[1], "lon1": BBOX[2],
            "lat1": BBOX[3], "cols": 2, "rows": 2,
            "values": [[6.0, 12.0], [18.0, 24.0]],
        })
        manifest = json.load(open(os.path.join(self.pack, "manifest.json")))
        manifest["layers"]["canopy"] = {"format": "grid-json",
                                        "path": "canopy/grid.json",
                                        "source": "test"}
        write_json(os.path.join(self.pack, "manifest.json"), manifest)
        env = EnvPackEnvironment(self.pack)
        self.assertGreater(env.canopy_height(120.015, 22.915), 0.0)
        self.assertEqual(env.canopy_height(122.0, 23.0), 0.0)
        self.assertGreaterEqual(env.cover_height(120.015, 22.915), 6.0)

    @unittest.skipIf(gdal is None, "GDAL python not available")
    def test_landcover_class_grid(self):
        tif = os.path.join(self.tmp, "lc.tif")
        make_class_tif(tif)
        from resources.envpack_layers import add_landcover_layer
        croot = os.path.join(self.tmp, "lc_dataset")
        os.makedirs(croot)
        shutil.copyfile(tif, os.path.join(croot, "landcover.tif"))
        write_json(os.path.join(croot, "manifest.json"), {
            "schema": "s3d.raster.v1", "raster_file": "landcover.tif",
            "source": {"as_of": "2021-01-01"}, "spec_hash": "x",
        })
        lc_entry = self.store.register(
            type(self.env_entry)(
                entry_id="landcover:test", kind="landcover",
                path=croot, content_hash="a" * 64, size=1,
                schema="s3d.raster.v1", bbox=BBOX, license="CC BY 4.0"))
        result = add_landcover_layer(
            self.store, self.env_entry, lc_entry, name="p1-lc",
            resolution_m=30, cache_root=os.path.join(self.tmp, "out"))
        entry = result["entry"]
        env = EnvPackEnvironment(entry.path)
        self.assertEqual(env.landcover(120.002, 22.918), "urban")
        self.assertEqual(env.landcover(120.018, 22.918), "forest")
        self.assertEqual(env.landcover(120.010, 22.902), "water")

    @unittest.skipIf(gdal is None, "GDAL python not available")
    def test_raster_to_grid_and_add_layer(self):
        tif = os.path.join(self.tmp, "canopy.tif")
        make_canopy_tif(tif)
        grid = raster_to_grid(tif, BBOX, resolution_m=30)
        self.assertGreater(grid["cols"], 0)
        self.assertGreater(grid["rows"], 0)
        self.assertEqual(len(grid["values"]), grid["rows"])
        # Register a canopy resource.
        croot = os.path.join(self.tmp, "canopy_dataset")
        os.makedirs(croot)
        shutil.copyfile(tif, os.path.join(croot, "canopy_height.tif"))
        write_json(os.path.join(croot, "manifest.json"), {
            "schema": "s3d.raster.v1", "raster_file": "canopy_height.tif",
            "source": {"as_of": "2024-01-01"}, "spec_hash": "x",
        })
        canopy_entry = self.store.register(
            type(self.env_entry)(
                entry_id="canopy_height:test", kind="canopy_height",
                path=croot, content_hash="a" * 64, size=1,
                schema="s3d.raster.v1", bbox=BBOX,
                license="CC BY 4.0"))
        result = add_canopy_layer(self.store, self.env_entry, canopy_entry,
                                  name="p1-canopy",
                                  cache_root=os.path.join(self.tmp, "out"))
        entry = result["entry"]
        self.assertEqual(entry.entry_id, "envpack:p1-canopy")
        with open(os.path.join(entry.path, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertIn("canopy", manifest["layers"])
        env = EnvPackEnvironment(entry.path)
        self.assertGreater(env.canopy_height(120.01, 22.91), 0.0)


if __name__ == "__main__":
    unittest.main()
