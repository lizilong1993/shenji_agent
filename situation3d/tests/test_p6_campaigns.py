"""P6 campaign cases: landing chain and India theatre replay."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from scripts.run_geo_demo import _assign_orders, _case_inputs  # noqa: E402
from sim.scenario import build_world  # noqa: E402


def _add_objectives(world, raw):
    objectives = []
    for o in raw:
        lon, lat = world.env.grid.to_lonlat(int(o["coord"]))
        cell = world.rule_grid.cell_of(lon, lat)
        owner = o.get("flag", -1)
        world.add_objective(cell, value=int(o.get("value") or 0),
                            name=str(o.get("name")), owner=owner)
        objectives.append({"lon": lon, "lat": lat, "owner": owner,
                           "value": o.get("value"), "name": o.get("name")})
    return objectives


class TestP6Campaigns(unittest.TestCase):
    def _require_map(self, map_id):
        path = os.path.join(SITUATION3D, "web", "data", "campaign_maps",
                            "map_%d" % map_id, "basic.json")
        if not os.path.isfile(path):
            self.skipTest("map_%d not generated in this checkout" % map_id)

    def test_taiwan_landing_chain(self):
        self._require_map(9001)
        scenario, basic, raw = _case_inputs("taiwan_landing")
        world = build_world(scenario, basic, seed=20261021, dt_s=30.0)
        objectives = _add_objectives(world, raw)
        _assign_orders(world, objectives)
        disembarked = []
        for i in range(60):
            events = world.step()
            disembarked.extend(e for e in events if e.get("type") == "disembark")
            if i % 30 == 29:
                _assign_orders(world, objectives)
        self.assertGreaterEqual(len(disembarked), 1,
                                "landing units must disembark")
        self.assertFalse(world.entities[1030].on_board)

    def test_india_case_czml(self):
        self._require_map(9002)
        scenario, basic, raw = _case_inputs("india_china_west")
        self.assertEqual(scenario["scenario_id"], 9002)
        world = build_world(scenario, basic, seed=20261022, dt_s=30.0)
        objectives = _add_objectives(world, raw)
        _assign_orders(world, objectives)
        for _ in range(60):
            world.step()
        from geoio.czml import write_czml
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.czml")
            write_czml(world, path)
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            packets = doc["packets"]
            self.assertGreater(sum(1 for p in packets if p.get("position")), 5)
            self.assertGreater(sum(1 for p in packets if p.get("id") == "document"), 0)


if __name__ == "__main__":
    unittest.main()
