"""Tests for the P1 data sources: PBF import, Copernicus DEM, GEBCO depth."""

import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from geodata.aaigrid import bilinear, parse_aaigrid, sample_regular_grid
from geodata.cache import CacheStore
from geodata.dem_copernicus import CopernicusDemSource, _tiles_for_bbox, tile_name
from geodata.osm_pbf import PbfOsmSource
from geodata.pipeline import build_envpack

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 24, "rows": 24,
        "width_m": 200.0}


def make_basic(rows=24, cols=24):
    cells = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append({"pos": r * 100 + c, "elev": 100,
                        "cond": 0, "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID), "map_data": cells}


def make_grid_text(values, xll=120.14, yll=22.97, cell=0.01):
    rows = len(values)
    cols = len(values[0])
    head = "ncols %d\nnrows %d\nxllcorner %.6f\nyllcorner %.6f\ncellsize %.6f\nNODATA_value -9999\n" % (
        cols, rows, xll, yll, cell)
    body = "\n".join(" ".join("%.1f" % v for v in row) for row in values)
    return head + body + "\n"


class TestAaiGrid(unittest.TestCase):
    def test_parse_and_bilinear(self):
        text = make_grid_text([[0, 10], [20, 30]])
        grid = parse_aaigrid(text)
        self.assertEqual(grid["cols"], 2)
        self.assertEqual(grid["rows"], 2)
        # pixel centres at xll+0.005 / yll+0.005; mid point ~ average
        self.assertAlmostEqual(bilinear(grid, 120.15, 22.98), 15.0, delta=0.01)
        self.assertIsNone(bilinear(grid, 120.2, 23.0))

    def test_sample_regular_grid_shape(self):
        text = make_grid_text([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              cell=0.005)
        grid = parse_aaigrid(text)
        out = sample_regular_grid(grid, (120.14, 22.97, 120.16, 22.99), 0.01)
        self.assertEqual((out["cols"], out["rows"]), (3, 3))
        self.assertTrue(all(v == 1.0 for row in out["values"] for v in row))


class TestPbfSource(unittest.TestCase):
    def test_auto_caches_and_offline_replays(self):
        import geodata.osm_pbf as osm_pbf
        calls = {"n": 0}
        real = osm_pbf.read_bbox

        def fake_read(path, bbox):
            calls["n"] += 1
            c = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            return {"elements": [{
                "type": "way", "id": 1, "tags": {"highway": "primary"},
                "geometry": [{"lon": bbox[0], "lat": c[1]},
                             {"lon": bbox[2], "lat": c[1]}]}]}

        osm_pbf.read_bbox = fake_read
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cache = CacheStore(tmp, mode="auto")
                src = PbfOsmSource(cache, "/nonexistent.osm.pbf")
                feats = src.features((120.14, 22.97, 120.19, 23.02))
                self.assertEqual(len(feats["roads"]), 1)
                self.assertTrue(src.fetched_network)
                src2 = PbfOsmSource(CacheStore(tmp, mode="offline"),
                                    "/nonexistent.osm.pbf")
                feats2 = src2.features((120.14, 22.97, 120.19, 23.02))
                self.assertEqual(feats2, feats)
                self.assertFalse(src2.fetched_network)
                self.assertEqual(calls["n"], 1)
        finally:
            osm_pbf.read_bbox = real

    def test_real_pbf_extraction(self):
        try:
            import osmium  # noqa: F401
        except ImportError:
            self.skipTest("pyosmium not installed")
        pbf = os.environ.get("GEO_TEST_PBF")
        if not pbf or not os.path.isfile(pbf):
            self.skipTest("GEO_TEST_PBF not set")
        from geodata.osm_pbf import read_bbox
        raw = read_bbox(pbf, (120.14, 22.97, 120.19, 23.02))
        self.assertGreater(len(raw["elements"]), 100)


class TestCopernicusDem(unittest.TestCase):
    def test_tile_names(self):
        self.assertEqual(tile_name(120.15, 22.99),
                         "Copernicus_DSM_COG_10_N22_00_E120_00_DEM")
        self.assertEqual(tile_name(-118.2, 36.1),
                         "Copernicus_DSM_COG_10_N36_00_W119_00_DEM")
        self.assertEqual(_tiles_for_bbox((120.14, 22.97, 120.19, 23.02)),
                         ["Copernicus_DSM_COG_10_N22_00_E120_00_DEM",
                          "Copernicus_DSM_COG_10_N23_00_E120_00_DEM"])

    def test_sample_grid_with_fake_gdal(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = CopernicusDemSource(CacheStore(tmp, mode="auto"))

            def fake_run(name, bbox, out_path):
                west, south, east, north = bbox
                text = make_grid_text([[round(south * 100, 1),
                                        round(north * 100, 1)]],
                                      xll=west, yll=south,
                                      cell=max(east - west, north - south))
                with open(out_path, "w") as fh:
                    fh.write(text)

            src._run_translate = fake_run
            grid = src.sample_grid((120.14, 22.97, 120.19, 23.02), 0.05)
            self.assertGreaterEqual(grid["cols"], 2)
            self.assertGreaterEqual(grid["rows"], 2)
            self.assertTrue(any(v is not None for row in grid["values"] for v in row))


class TestHeightFallback(unittest.TestCase):
    def _feature(self, ring):
        return {"type": "Feature", "properties": {"h": 21.0},
                "geometry": {"type": "Polygon", "coordinates": [ring]}}

    def test_fill_from_wgs84_polygon(self):
        from geodata.height_fallback import HeightFallback
        ring = [[120.140, 22.970], [120.142, 22.970], [120.142, 22.972],
                [120.140, 22.972], [120.140, 22.970]]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gba.geojson")
            with open(path, "w") as fh:
                json.dump({"type": "FeatureCollection",
                           "features": [self._feature(ring)]}, fh)
            bld = [{"ring": [[120.1405, 22.9705], [120.1415, 22.9705],
                             [120.1415, 22.9715], [120.1405, 22.9705]],
                    "height_m": 6.0, "height_source": "default"},
                   {"ring": [[120.15, 22.98], [120.151, 22.98],
                             [120.151, 22.981], [120.15, 22.98]],
                    "height_m": 6.0, "height_source": "default"},
                   {"ring": [[120.1405, 22.9705], [120.1415, 22.9705],
                             [120.1415, 22.9715], [120.1405, 22.9705]],
                    "height_m": 12.0, "height_source": "tag"}]
            stats = HeightFallback(path).fill(bld)
            self.assertEqual(stats["filled"], 1)
            self.assertAlmostEqual(bld[0]["height_m"], 21.0)
            self.assertEqual(bld[0]["height_source"], "fallback")
            self.assertEqual(bld[2]["height_m"], 12.0)

    def test_mercator_coordinates(self):
        from geodata.height_fallback import HeightFallback
        r = 6378137.0
        def to_merc(lon, lat):
            return [r * math.radians(lon),
                    r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))]
        ring = [to_merc(120.140, 22.970), to_merc(120.142, 22.970),
                to_merc(120.142, 22.972), to_merc(120.140, 22.972),
                to_merc(120.140, 22.970)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gba.geojson")
            with open(path, "w") as fh:
                json.dump({"type": "FeatureCollection",
                           "features": [self._feature(ring)]}, fh)
            fb = HeightFallback(path)
            self.assertAlmostEqual(fb.height_at(120.141, 22.971), 21.0, delta=0.01)



    class FakeBath(object):
        def sample_grid(self, bbox, step_deg):
            lon0, lat0, lon1, lat1 = bbox
            return {"lon0": lon0, "lat0": lat0, "lon1": lon1, "lat1": lat1,
                    "cols": 2, "rows": 2,
                    "values": [[-30.0, -30.0], [-30.0, -30.0]]}

    class FakeOsm(object):
        def features(self, bbox):
            c = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            ring = [[c[0] - 0.002, c[1] - 0.002], [c[0] + 0.002, c[1] - 0.002],
                    [c[0] + 0.002, c[1] + 0.002], [c[0] - 0.002, c[1] + 0.002],
                    [c[0] - 0.002, c[1] - 0.002]]
            return {"roads": [], "water": [{"ring": ring}],
                    "buildings": [], "landuse": []}

    class FakeDem(object):
        def sample_grid(self, bbox, step_deg):
            return {"lon0": bbox[0], "lat0": bbox[1], "lon1": bbox[2],
                    "lat1": bbox[3], "cols": 2, "rows": 2,
                    "values": [[100.0, 100.0], [100.0, 100.0]]}

    def test_pack_includes_depth_grid(self):
        from compat.legacy import LegacyHexGrid
        from env.pack_env import EnvPackEnvironment
        basic = make_basic()
        grid = LegacyHexGrid.from_grid_meta(basic["grid"])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "pack")
            manifest = build_envpack(
                out, basic, grid, mode="offline",
                cache_dir=os.path.join(tmp, "cache"),
                osm_source=self.FakeOsm(), dem_source=self.FakeDem(),
                bathymetry_source=self.FakeBath())
            self.assertIn("bathymetry", manifest["layers"])
            self.assertEqual(manifest["schema"], "msgeo.envpack.v2")
            self.assertIn("chunked", manifest["layers"]["water"])
            self.assertTrue(manifest["layers"]["water"]["chunked"]["chunks"])
            chunk = list(manifest["layers"]["water"]["chunked"]["chunks"].values())[0]
            self.assertTrue(os.path.isfile(os.path.join(out, chunk["path"])))
            from env.pack import EnvPack
            self.assertEqual(EnvPack.load(out).manifest["schema"],
                             "msgeo.envpack.v2")
            self.assertEqual(manifest["layers"]["water"]["depth_grid"],
                             "water/depth.json")
            env = EnvPackEnvironment(out)
            c = manifest["bbox"]
            mid = ((c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0)
            self.assertAlmostEqual(env.water_depth(*mid), 30.0, delta=0.1)

    def test_pack_without_bathymetry_falls_back(self):
        from compat.legacy import LegacyHexGrid
        from env.pack_env import EnvPackEnvironment
        basic = make_basic()
        grid = LegacyHexGrid.from_grid_meta(basic["grid"])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "pack")
            manifest = build_envpack(
                out, basic, grid, mode="offline",
                cache_dir=os.path.join(tmp, "cache"),
                osm_source=self.FakeOsm(), dem_source=self.FakeDem(),
                use_bathymetry=False)
            self.assertNotIn("bathymetry", manifest["layers"])
            env = EnvPackEnvironment(out)
            c = manifest["bbox"]
            mid = ((c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0)
            self.assertGreater(env.water_depth(*mid), 0.0)


if __name__ == "__main__":
    unittest.main()
