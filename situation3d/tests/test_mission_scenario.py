"""Mission runner and real-coordinate scenario tests (TW2 D1)."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim.mission import MissionRunner  # noqa: E402
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


def operator(uid, color, lon, lat, name="ZTD-05 两栖突击车",
             domain="land", subclass="land.vehicle.amphibious", **extra):
    op = {"obj_id": uid, "name": name, "color": color, "type": 2,
          "domain": domain, "lon": lon, "lat": lat, "subclass_id": subclass,
          "blood": 2, "max_blood": 2}
    op.update(extra)
    op.setdefault("observe_distance", {"1": 10, "2": 25, "3": 1})
    return op


def scenario(ops, missions=(), roe=None):
    return {"engine": {"detection_model": "table", "scan_interval_s": 30,
                       "use_role_cards": False},
            "operators": list(ops), "missions": list(missions),
            "roe": roe or {}}


class MissionScenarioTest(unittest.TestCase):
    def test_real_lonlat_position(self):
        lon, lat = 119.905, 23.095
        world = build_world(scenario([operator(1, 0, lon, lat)]), make_basic(), seed=1)
        entity = world.entities[1]
        self.assertAlmostEqual(entity.pose.lon, lon, places=6)
        self.assertAlmostEqual(entity.pose.lat, lat, places=6)

    def test_explicit_carrier(self):
        ship = operator(10, 0, 119.90, 23.10, name="071 型船坞登陆舰",
                        domain="surface", subclass="sea.surface.landing_ship")
        vehicle = operator(1, 0, 119.90, 23.10, carrier=10)
        world = build_world(scenario([ship, vehicle]), make_basic(), seed=1)
        self.assertTrue(world.entities[1].on_board)
        self.assertEqual(world.entities[1].carrier, 10)
        self.assertIn(1, world.entities[10].passengers)

    def test_move_mission_completes_and_unlocks_dependent(self):
        start = (119.90, 23.10)
        target = (119.91, 23.10)
        missions = [
            {"id": "M1", "side": 0, "units": [1], "task": "move_to",
             "lon": target[0], "lat": target[1], "pace": "dash",
             "arrive_m": 260.0},
            {"id": "M2", "side": 0, "units": [1], "task": "hold",
             "lon": target[0], "lat": target[1], "after": ["M1"]},
        ]
        world = build_world(scenario([operator(1, 0, *start)], missions),
                            make_basic(), seed=1)
        runner = MissionRunner(world, missions, {
            "halt_on_contact": False}, interval_s=30.0)
        runner.update(force=True)
        self.assertEqual(runner.status()[0]["status"], "active")
        self.assertEqual(runner.status()[1]["status"], "pending")
        for _ in range(30):
            world.step(30.0)
            runner.update()
        status = {m["id"]: m["status"] for m in runner.status()}
        self.assertEqual(status["M1"], "complete")
        self.assertIn(status["M2"], ("active", "complete"))

    def test_contact_rule_issues_stop(self):
        red = operator(1, 0, 119.90, 23.10)
        blue = operator(2, 1, 119.901, 23.10)
        missions = [{"id": "M1", "side": 0, "units": [1], "task": "hold",
                     "lon": 119.93, "lat": 23.10, "pace": "normal"}]
        world = build_world(scenario([red, blue], missions), make_basic(), seed=1)
        runner = MissionRunner(world, missions, {
            "halt_on_contact": True, "engage_m": 2500.0,
            "resume_after_s": 60.0}, interval_s=30.0)
        runner.update(force=True)
        self.assertIn(1, runner.engaged)
        self.assertEqual((world.orders.get(1) or {}).get("type"), "stop")
        world.step(30.0)
        self.assertIsNone(world.entities[1].route)


if __name__ == "__main__":
    unittest.main()
