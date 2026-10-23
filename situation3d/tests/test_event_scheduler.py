"""Event-driven time fast-forward and event gating (P4).

Run from ``situation3d/``:
    python3 -m unittest tests/test_event_scheduler.py
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim import perception  # noqa: E402
from sim.entity import Pose  # noqa: E402
from sim.scenario import build_world

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 24, "rows": 24,
        "width_m": 200.0}


def make_basic():
    cells = [[{"pos": r * 100 + c, "elev": 100, "cond": 0,
               "roads": [0] * 6, "rivers": [0] * 6}
              for c in range(24)] for r in range(24)]
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID),
            "map_data": cells}


def operator(uid, cell, color=0):
    return {"obj_id": uid, "name": "重型坦克", "color": color, "type": 2,
            "sub_type": 0, "cur_hex": cell, "blood": 999, "max_blood": 999,
            "armor": 2, "value": 10, "domain": "land",
            "carry_weapon_ids": [], "move_state": 0, "stop": 1,
            "readiness": 0.85, "fuel": 1.0}


def build(ops, engine=None, seed=7):
    sc = {"scenario_id": 1, "cities": [], "operators": ops,
          "time": {"cur_step": 0, "tick": 1.0, "max_time": 9999},
          "engine": engine or {}}
    return build_world(sc, make_basic(), seed=seed)


class EventTimeTest(unittest.TestCase):
    def _world(self):
        return build([operator(1, 1000), operator(2, 1001)])

    def test_fresh_world_advance_runs_initial_sweep(self):
        world = self._world()
        world.advance_to(1.0)
        self.assertTrue(world._scan_due)
        self.assertEqual(world.time_s, 1.0)

    def test_advance_fast_forwards_idle_to_end(self):
        world = self._world()
        world.step()
        scans = []
        orig = world._advance_scan_due

        def wrapped(uid, due_map=None, interval=None):
            scans.append(uid)
            return orig(uid, due_map, interval)

        world._advance_scan_due = wrapped
        world.advance_to(10.0)
        self.assertEqual(world.time_s, 10.0)
        self.assertEqual(scans, [])        # no event lies inside 2..10 s

    def test_advance_wakes_at_scan_due(self):
        world = self._world()
        world.step()
        scans = []
        orig = world._advance_scan_due

        def wrapped(uid, due_map=None, interval=None):
            scans.append(uid)
            return orig(uid, due_map, interval)

        world._advance_scan_due = wrapped
        due = world._scan_due[1]
        world.advance_to(due + 5.0)
        self.assertIn(1, scans)
        self.assertLessEqual(world.time_s, due + 5.0 + 1e-9)
        self.assertEqual(world.time_s, due + 5.0)

    def test_weather_timeline_applied_on_jump(self):
        world = self._world()
        world.weather_enabled = True
        world.weather_timeline = [{"at_s": 100.0, "visibility_m": 1234.0,
                                   "sea_state": 2, "wind_ms": 7.0}]
        world.step()
        world.advance_to(100.0)
        self.assertEqual(world.time_s, 100.0)
        self.assertAlmostEqual(
            world.effective_weather["visibility_m"], 1234.0)
        self.assertEqual(world.effective_weather["sea_state"], 2)

    def test_pending_impact_fires_at_due_time(self):
        world = self._world()
        target = world.entities[2]
        world._pending_impacts.append({
            "due_s": 50.0, "attacker": 1, "target": 2, "weapon": "test",
            "kind": "missile", "damage": 3, "suppress_s": 0.0,
            "intercepted": False, "planned_hit": True, "pk": 1.0,
            "intercept_pk": 0.0, "impact_at_s": 50.0, "model": "flight",
            "launch_tick_s": 0.0})
        world._next_impact_s = 50.0
        world.step()
        world.advance_to(49.0)
        self.assertEqual(target.blood, 999)
        world.advance_to(51.0)
        self.assertEqual(target.blood, 996)
        self.assertTrue(any(e.get("type") == "missile_impact"
                            for e in world.events))

    def test_orders_disable_fast_forward(self):
        world = self._world()
        world.step()
        world.order(1, {"type": "move_to", "lon": 119.9, "lat": 23.1})
        world.advance_to(3.0)
        self.assertEqual(world.time_s, 3.0)
        self.assertFalse(world.idle())


class RadioRelayTest(unittest.TestCase):
    def test_radio_share_does_not_multihop(self):
        ops = [operator(1, 1000), operator(2, 1000), operator(4, 1000),
               operator(3, 1000, color=1)]
        world = build(ops, engine={"detection_model": "physical",
                                    "enable_radio_share": True})
        base_lon, lat = 119.9, 23.1
        deg = 1.0 / 111320.0
        lat_m = 1.0 / 110540.0
        positions = {1: (base_lon, lat),
                     3: (base_lon + 50 * deg, lat),
                     2: (base_lon + 600 * deg, lat),
                     4: (base_lon + 1200 * deg, lat)}
        for uid, (lon, lat2) in positions.items():
            world.entities[uid].pose = Pose(lon, lat2, 100.0)
            world.entities[uid].comms_range_m = 800.0
        original = perception.specs_for
        perception.specs_for = lambda entity: [perception.SensorSpec(
            "radar", 500.0, 360.0, 90.0, 1.0, 1.0e6)]
        try:
            world.step()
            self.assertIn(3, world.seen.get(1, set()))
            self.assertIn(3, world.seen.get(2, set()))   # one-hop from A
            self.assertNotIn(3, world.seen.get(4, set()))
            world.step()
            # B must not rebroadcast the contact it only received by radio.
            self.assertNotIn(3, world.seen.get(4, set()))
        finally:
            perception.specs_for = original


if __name__ == "__main__":
    unittest.main()
