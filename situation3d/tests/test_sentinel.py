"""Sentinel-2 STAC search, scene selection and resource registration."""

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
from resources.vectorproc import DatasetSpec  # noqa: E402

BBOX = (118.0, 24.4, 118.1, 24.5)


def stac_feature(fid, when, cloud):
    return {
        "id": fid,
        "bbox": [116.9, 23.4, 118.1, 24.5],
        "properties": {"datetime": when, "eo:cloud_cover": cloud},
        "assets": {"visual": {"href": "https://example/%s/TCI.tif" % fid}},
    }


class SentinelSearchTest(unittest.TestCase):
    def test_selects_low_cloud_scene(self):
        doc = {"features": [
            stac_feature("cloudy", "2025-12-20T00:00:00Z", 70),
            stac_feature("clear-old", "2025-12-01T00:00:00Z", 3),
            stac_feature("clear-new", "2025-12-15T00:00:00Z", 5),
        ]}
        with mock.patch.object(adapters, "_http_json", return_value=doc):
            chosen, query = adapters.search_sentinel2(
                BBOX, "2025-12-31", window_days=60, max_cloud=30)
        self.assertEqual(chosen["id"], "clear-old")
        self.assertEqual(query["clear"], 2)
        self.assertAlmostEqual(query["cloud_cover"], 3.0)

    def test_no_clear_scene_falls_back_to_least_cloudy(self):
        doc = {"features": [stac_feature("a", "2025-12-20T00:00:00Z", 80),
                            stac_feature("b", "2025-12-21T00:00:00Z", 60)]}
        with mock.patch.object(adapters, "_http_json", return_value=doc):
            chosen, query = adapters.search_sentinel2(
                BBOX, "2025-12-31", max_cloud=10)
        self.assertEqual(chosen["id"], "b")
        self.assertEqual(query["clear"], 0)

    def test_no_scene_raises(self):
        with mock.patch.object(adapters, "_http_json",
                               return_value={"features": []}):
            with self.assertRaises(adapters.DatasetError):
                adapters.search_sentinel2(BBOX, "2025-12-31")

    def test_mosaic_picks_best_scene_per_grid(self):
        def feat(fid, code, cloud, when="2025-12-20T00:00:00Z"):
            item = stac_feature(fid, when, cloud)
            item["properties"]["grid:code"] = code
            if code == "MGRS-44SLC":
                item["bbox"] = [78.8, 33.3, 80.0, 34.3]
            else:
                item["bbox"] = [77.7, 32.4, 79.0, 33.4]
            return item

        doc = {"features": [
            feat("a", "MGRS-44SLC", 20),
            feat("b", "MGRS-44SLC", 5),
            feat("c", "MGRS-44SKC", 10),
        ]}
        with mock.patch.object(adapters, "_http_json", return_value=doc):
            scenes, query = adapters.search_sentinel2_mosaic(
                (78.6, 33.3, 79.6, 34.2), "2025-12-31", max_cloud=30)
        self.assertEqual([s["id"] for s in scenes], ["b", "c"])
        self.assertEqual(query["grids"], 2)

    def test_query_changes_spec_hash(self):
        a = DatasetSpec(name="s", bbox=BBOX, kind="satellite_imagery",
                        query={"date": "2025-12-01"})
        b = DatasetSpec(name="s", bbox=BBOX, kind="satellite_imagery",
                        query={"date": "2025-12-02"})
        self.assertNotEqual(a.hash(), b.hash())


class SentinelTilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="st-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))
        self.spec = DatasetSpec(name="x", bbox=BBOX, kind="satellite_imagery",
                                resolution_m=10)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sentinel_to_tiles(self):
        from resources.processing import TilesetSpec
        raster_dir = os.path.join(self.tmp, "scene")
        os.makedirs(raster_dir)
        raster = os.path.join(raster_dir, "sentinel2.tif")
        with open(raster, "wb") as fh:
            fh.write(b"II*\x00" + b"\0" * 128)
        with open(os.path.join(raster_dir, "manifest.json"), "w") as fh:
            json.dump({"raster_file": "sentinel2.tif"}, fh)
        from resources.model import ResourceEntry
        from resources.processing import TilesetSpec
        scene_entry = ResourceEntry(
            entry_id="satellite_imagery:x", kind="satellite_imagery",
            path=raster_dir, content_hash="a" * 64, size=1, bbox=BBOX)
        sentinel = {"entry": scene_entry, "cached": False}
        fake_tiles = type("R", (), {"entry": scene_entry,
                                    "as_dict": lambda self: {"ok": True}})()
        with mock.patch.object(adapters, "acquire_sentinel",
                               return_value=sentinel), \
                mock.patch("resources.acquire.acquire_tiles",
                           return_value=fake_tiles) as tiles:
            result = adapters.acquire_sentinel_tiles(
                self.store, TilesetSpec(name="x", bbox=BBOX, min_zoom=10,
                                        max_zoom=12),
                self.tmp)
        self.assertFalse(result["fallback"])
        self.assertIs(result["tiles"], fake_tiles)
        called = tiles.call_args
        self.assertEqual(called.kwargs["candidates"][0].source_id, "sentinel2")
        self.assertTrue(called.kwargs["candidates"][0].params["url"].startswith("file://"))

    def test_fallback_to_eox(self):
        from resources.processing import TilesetSpec
        fake_tiles = type("R", (), {"entry": None,
                                    "as_dict": lambda self: {}})()
        with mock.patch.object(adapters, "acquire_sentinel",
                               side_effect=adapters.DatasetError("no scene")), \
                mock.patch("resources.acquire.acquire_tiles",
                           return_value=fake_tiles) as tiles:
            result = adapters.acquire_sentinel_tiles(
                self.store, TilesetSpec(name="x", bbox=BBOX, min_zoom=10,
                                        max_zoom=12),
                self.tmp)
        self.assertTrue(result["fallback"])
        self.assertEqual(tiles.call_args.kwargs["source_ids"],
                         ["eox-s2", "esri"])


class SentinelRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sentinel-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_registration_and_cache(self):
        raster = os.path.join(self.tmp, "s2.tif")
        with open(raster, "wb") as fh:
            fh.write(b"II*\\x00" + b"\\0" * 128)
        info = {"width": 10, "height": 10, "band_count": 3,
                "crs": "EPSG:4326"}
        source = {"label": "Sentinel-2", "scene_id": "S2B_TEST",
                  "cloud_cover": 3.0,
                  "scene_datetime": "2025-12-01T00:00:00Z",
                  "as_of": "2025-12-01"}
        chosen = {"id": "S2B_TEST", "cloud": 3.0}
        spec = DatasetSpec(name="s2", bbox=BBOX,
                           kind="satellite_imagery",
                           resolution_m=10,
                           query={"date": "2025-12-31"})
        with mock.patch.object(adapters, "fetch_sentinel2",
                               return_value=(raster, info, source, chosen)):
            first = adapters.acquire_sentinel(
                self.store, spec, os.path.join(self.tmp, "cache"))
        self.assertEqual(first["entry"].kind, "satellite_imagery")
        self.assertFalse(first["cached"])
        with mock.patch.object(adapters, "fetch_sentinel2",
                               side_effect=AssertionError("must not run")):
            second = adapters.acquire_sentinel(
                self.store, spec, os.path.join(self.tmp, "cache"))
        self.assertTrue(second["cached"])
        self.assertEqual(second["entry"].content_hash,
                         first["entry"].content_hash)


if __name__ == "__main__":
    unittest.main()
