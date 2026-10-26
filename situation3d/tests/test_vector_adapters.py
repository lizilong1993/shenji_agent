"""Vector/raster dataset adapters, manifests and fallback tests."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from resources import adapters  # noqa: E402
from resources.store import ResourceStore  # noqa: E402
from resources.vectorproc import (DatasetSpec, vector_stats,  # noqa: E402
                                  write_vector)

SPEC = DatasetSpec(name="xiamen-buildings",
                   bbox=(118.0, 24.4, 118.1, 24.5),
                   kind="building_footprints")


def feature(fid, height):
    return {"type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [[[118.01, 24.41], [118.02, 24.41],
                                          [118.02, 24.42], [118.01, 24.41]]]},
            "properties": {"id": fid, "height_m": height, "source": "test"}}


class VectorProcTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vec-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_spec_and_stats(self):
        SPEC.validate()
        self.assertEqual(len(SPEC.hash()), 64)
        stats = vector_stats([feature("a", 10), feature("b", 0),
                              feature("c", None)])
        self.assertEqual(stats["feature_count"], 3)
        self.assertAlmostEqual(stats["height_coverage"], 1.0 / 3.0)

    def test_write_and_register(self):
        root = os.path.join(self.tmp, "dataset")
        stats = vector_stats([feature("a", 12.0)])
        manifest = write_vector(root, [feature("a", 12.0)], SPEC,
                                {"label": "test"}, stats)
        self.assertEqual(manifest["schema"], "s3d.vector.v1")
        with open(os.path.join(root, "features.geojson"), encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(len(doc["features"]), 1)
        entry = adapters.register_dataset(
            self.store, SPEC, root, {"label": "test"}, "test",
            quality="measured", provider="adapter:test")
        self.assertEqual(entry.kind, "building_footprints")
        self.assertTrue(self.store.verify(entry.entry_id)["ok"])

    def test_gba_tiles_and_url(self):
        tiles = adapters._gba_tiles((118.0, 24.4, 118.1, 24.5))
        self.assertEqual(tiles, [(115, 20, 120, 25)])
        url = adapters._gba_url(115, 20, 120, 25)
        self.assertTrue(url.endswith("e115_n25_e120_n20.parquet"))

    def test_worldcover_tiles_and_url(self):
        tiles = adapters._worldcover_tiles((118.0, 24.4, 118.1, 24.5))
        self.assertEqual(tiles, [(117, 24)])
        url = adapters._worldcover_url(117, 24)
        self.assertTrue(url.endswith("N24E117_Map.tif"))

    def test_chmv2_quadkeys_cover_bbox(self):
        keys = adapters._quadkeys((118.0, 24.4, 118.1, 24.5))
        self.assertTrue(keys)
        self.assertTrue(all(len(k) == 10 for k in keys))


class AdapterOrchestrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="adapter-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))
        self.cache = os.path.join(self.tmp, "cache")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_building_fallback(self):
        def boom(*args, **kwargs):
            raise adapters.DatasetError("gba down")

        def ok(*args, **kwargs):
            return ([feature("o1", 15.0), feature("o2", 8.0)],
                    {"label": "Overture", "as_of": "2025-01-01"})

        with mock.patch.object(adapters, "fetch_gba", side_effect=boom), \
                mock.patch.object(adapters, "fetch_overture", side_effect=ok):
            result = adapters.acquire_buildings(
                self.store, SPEC, ["gba", "overture"], self.cache)
        self.assertEqual(result["entry"].kind, "building_footprints")
        self.assertEqual(result["stats"]["feature_count"], 2)
        self.assertIn("gba down", result["errors"][0])
        self.assertEqual(self.store.get(result["entry"].entry_id).notes["adapter"],
                         "overture")

    def test_landcover_registration(self):
        raster = os.path.join(self.tmp, "lc.tif")
        with open(raster, "wb") as fh:
            fh.write(b"II*\x00" + b"\0" * 128)
        info = {"width": 10, "height": 10, "band_count": 1,
                "crs": "EPSG:4326"}
        with mock.patch.object(
                adapters, "fetch_worldcover",
                return_value=(raster, info, {"label": "WorldCover",
                                             "as_of": "2021-01-01"})):
            result = adapters.acquire_landcover(
                self.store, DatasetSpec(name="lc", bbox=SPEC.bbox,
                                        kind="landcover"), self.cache)
        self.assertEqual(result["entry"].kind, "landcover")
        self.assertEqual(result["manifest"]["schema"], "s3d.raster.v1")
        self.assertTrue(self.store.verify(result["entry"].entry_id)["ok"])

    def test_dataset_cache_reuse(self):
        def ok(*args, **kwargs):
            return ([feature("o1", 15.0)], {"label": "Overture", "as_of": ""})

        with mock.patch.object(adapters, "fetch_overture", side_effect=ok):
            first = adapters.acquire_buildings(
                self.store, SPEC, ["overture"], self.cache)
        self.assertFalse(first.get("cached"))
        with mock.patch.object(adapters, "fetch_overture",
                               side_effect=AssertionError("must not run")):
            second = adapters.acquire_buildings(
                self.store, SPEC, ["overture"], self.cache)
        self.assertTrue(second["cached"])
        self.assertEqual(second["entry"].content_hash,
                         first["entry"].content_hash)

    def test_canopy_registration(self):
        raster = os.path.join(self.tmp, "canopy.tif")
        with open(raster, "wb") as fh:
            fh.write(b"II*\x00" + b"\0" * 128)
        info = {"width": 10, "height": 10, "band_count": 1,
                "crs": "EPSG:4326", "stats": {"mean": 4.2}}
        with mock.patch.object(
                adapters, "fetch_chmv2",
                return_value=(raster, info, {"label": "CHMv2",
                                             "as_of": "2024-01-01"})):
            result = adapters.acquire_canopy(
                self.store, DatasetSpec(name="canopy", bbox=SPEC.bbox,
                                        kind="canopy_height"),
                self.cache)
        self.assertEqual(result["entry"].kind, "canopy_height")
        self.assertEqual(result["manifest"]["schema"], "s3d.raster.v1")
        self.assertTrue(self.store.verify(result["entry"].entry_id)["ok"])


if __name__ == "__main__":
    unittest.main()
