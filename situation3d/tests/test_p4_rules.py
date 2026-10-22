"""P4 rule tests: bridge/tunnel, minefields, indirect fire."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from mobility.passability import passable, road_edge_passable
from mobility.profile import PROFILES


class FakeEnv(object):
    def __init__(self, edge=None, cover="open", depth=None):
        self.edge = edge
        self.cover = cover
        self.depth = depth

    def landcover(self, lon, lat):
        return self.cover

    def water_depth(self, lon, lat):
        return self.depth

    def slope_deg(self, lon, lat, bearing_deg=0.0):
        return 0.0

    def on_road(self, lon, lat, tolerance_m=25.0):
        return self.edge is not None

    def road_edge_at(self, lon, lat, tolerance_m=25.0):
        return self.edge

    def elevation(self, lon, lat):
        return 10.0


class TestBridgeConstraints(unittest.TestCase):
    def test_weight_limit_blocks_heavy(self):
        env = FakeEnv(edge={"bridge": True, "maxweight": "20 t"})
        self.assertFalse(passable(PROFILES["tracked"], 120.1, 23.0, env))
        self.assertTrue(passable(PROFILES["wheeled"], 120.1, 23.0, env))

    def test_height_limit_blocks_tall(self):
        env = FakeEnv(edge={"tunnel": True, "maxheight": "2.5 m"})
        self.assertFalse(passable(PROFILES["tracked"], 120.1, 23.0, env))
        self.assertTrue(passable(PROFILES["wheeled"], 120.1, 23.0, env))

    def test_width_limit_blocks_wide(self):
        env = FakeEnv(edge={"bridge": True, "width": "3"})
        self.assertFalse(passable(PROFILES["tracked"], 120.1, 23.0, env))

    def test_no_edge_no_constraint(self):
        env = FakeEnv(edge=None)
        self.assertTrue(passable(PROFILES["tracked"], 120.1, 23.0, env))

    def test_edge_passable_helper(self):
        self.assertFalse(road_edge_passable(PROFILES["tracked"],
                                            {"maxweight": "30"}))
        self.assertTrue(road_edge_passable(PROFILES["tracked"],
                                           {"maxweight": "50", "maxheight": "4"}))

    def test_pack_road_edge_lookup(self):
        from compat.legacy import LegacyHexGrid
        from env.pack_env import EnvPackEnvironment
        from geodata.pipeline import build_envpack

        class FakeOsm(object):
            def features(self, bbox):
                c = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                road = [[bbox[0], c[1]], [bbox[2], c[1]]]
                return {"roads": [{"coords": road, "class": "primary",
                                   "bridge": True, "maxheight": "2.5",
                                   "maxweight": "10", "width": "3"}],
                        "water": [], "buildings": [], "landuse": []}

        class FakeDem(object):
            def sample_grid(self, bbox, step_deg):
                return {"lon0": bbox[0], "lat0": bbox[1], "lon1": bbox[2],
                        "lat1": bbox[3], "cols": 2, "rows": 2,
                        "values": [[10.0, 10.0], [10.0, 10.0]]}

        with tempfile.TemporaryDirectory() as tmp:
            basic = {"map_id": 1, "ele_grade": 10,
                     "grid": {"center_lon": 119.9, "center_lat": 23.1,
                              "cols": 24, "rows": 24, "width_m": 200.0},
                     "map_data": [[{"pos": r * 100 + c, "elev": 10, "cond": 0,
                                    "roads": [0] * 6, "rivers": [0] * 6}
                                   for c in range(24)] for r in range(24)]}
            grid = LegacyHexGrid.from_grid_meta(basic["grid"])
            out = os.path.join(tmp, "pack")
            manifest = build_envpack(out, basic, grid, mode="offline",
                                     cache_dir=os.path.join(tmp, "cache"),
                                     use_bathymetry=False,
                                     osm_source=FakeOsm(), dem_source=FakeDem())
            env = EnvPackEnvironment(out)
            c = manifest["bbox"]
            edge = env.road_edge_at((c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0)
            self.assertIsNotNone(edge)
            self.assertEqual(edge["maxheight"], "2.5")
            self.assertTrue(edge["bridge"])
            self.assertFalse(passable(PROFILES["tracked"],
                                      (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0,
                                      env))


if __name__ == "__main__":
    unittest.main()
