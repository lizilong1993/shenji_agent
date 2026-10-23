"""Detection-cache correctness on sub-cell cover (envpack) data."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from compat.legacy import LegacyHexGrid  # noqa: E402
from env.pack_env import EnvPackEnvironment  # noqa: E402
from geodata.pipeline import build_envpack  # noqa: E402
from sim.context import WorldRuleContext  # noqa: E402
from sim.entity import Pose  # noqa: E402
from sim.scenario import build_world  # noqa: E402

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 24, "rows": 24,
        "width_m": 200.0}


def make_basic():
    cells = []
    for r in range(24):
        row = []
        for c in range(24):
            row.append({"pos": r * 100 + c, "elev": 100, "cond": 0,
                        "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID), "map_data": cells}


def _ring(lon, lat, half_deg):
    return [[lon - half_deg, lat - half_deg], [lon + half_deg, lat - half_deg],
            [lon + half_deg, lat + half_deg], [lon - half_deg, lat + half_deg],
            [lon - half_deg, lat - half_deg]]


class TestDetectionCacheSubCell(unittest.TestCase):
    class FakeDem(object):
        def sample_grid(self, bbox, step_deg):
            return {"lon0": bbox[0], "lat0": bbox[1], "lon1": bbox[2],
                    "lat1": bbox[3], "cols": 2, "rows": 2,
                    "values": [[100.0, 100.0], [100.0, 100.0]]}

    def test_cover_change_within_cell_not_stale(self):
        basic = make_basic()
        grid = LegacyHexGrid.from_grid_meta(basic["grid"])
        target_hex = 5 * 100 + 19
        tlon, tlat = grid.to_lonlat(target_hex)
        forest_ring = _ring(tlon, tlat, 0.00025)   # ~55 m box inside one cell
        open_lon, open_lat = tlon, tlat - 0.0006  # ~67 m south, same cell

        class FakeOsm(object):
            def features(self, bbox):
                return {"roads": [],
                        "water": [],
                        "buildings": [],
                        "landuse": [{"ring": forest_ring, "class": "forest"}]}

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "pack")
            build_envpack(out, basic, grid, mode="offline",
                          cache_dir=os.path.join(tmp, "cache"),
                          use_bathymetry=False,
                          osm_source=FakeOsm(), dem_source=self.FakeDem())
            env = EnvPackEnvironment(out)
            self.assertEqual(env.landcover(tlon, tlat), "forest")
            self.assertNotEqual(env.landcover(open_lon, open_lat), "forest")

            scenario = {"scenario_id": 1,
                        "operators": [
                            {"obj_id": 1, "color": 0, "type": 2,
                             "name": "重型坦克", "sub_type": 0, "blood": 3,
                             "max_blood": 3, "armor": 3,
                             "cur_hex": 5 * 100 + 5,
                             "carry_weapon_ids": [36], "value": 10},
                            {"obj_id": 2, "color": 1, "type": 2,
                             "name": "重型坦克", "sub_type": 0, "blood": 3,
                             "max_blood": 3, "armor": 3,
                             "cur_hex": target_hex,
                             "carry_weapon_ids": [36], "value": 10}],
                        "cities": [],
                        "time": {"cur_step": 0, "tick": 10.0, "max_time": 999}}
            world = build_world(scenario, basic, seed=1, envpack=out)
            world.scan_interval_s = 0.0  # rescan on every _update_sensors call
            target = world.entities[2]
            # Same rule/env cell, different sub-cell cover.
            target.pose = Pose(open_lon, open_lat,
                               world.env.elevation(open_lon, open_lat))
            world._refresh_cell_cache()
            ctx = WorldRuleContext(world)
            world._update_sensors(ctx)
            self.assertIn(2, world.seen.get(1, set()),
                          "open position must be observed")

            target.pose = Pose(tlon, tlat, world.env.elevation(tlon, tlat))
            world._refresh_cell_cache()
            ctx = WorldRuleContext(world)
            world._update_sensors(ctx)
            self.assertNotIn(2, world.seen.get(1, set()),
                             "forest position must not reuse the open result")

            target.pose = Pose(open_lon, open_lat,
                               world.env.elevation(open_lon, open_lat))
            world._refresh_cell_cache()
            ctx = WorldRuleContext(world)
            world._update_sensors(ctx)
            self.assertIn(2, world.seen.get(1, set()),
                          "moving back into the open must restore detection")


if __name__ == "__main__":
    unittest.main()
