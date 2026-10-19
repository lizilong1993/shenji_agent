"""Compatibility smoke tests against public 庙算 scenario/map fixtures."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
REPO = os.path.dirname(SITUATION3D)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from wargame_core import engine, scenario  # noqa: E402


DATA = os.path.join(REPO, "land_wargame_sdk", "Data", "Data")
SCENARIO = os.path.join(DATA, "scenarios", "1231.json")
MAP = os.path.join(DATA, "maps", "map_123", "basic.json")


@unittest.skipUnless(os.path.isfile(SCENARIO) and os.path.isfile(MAP),
                     "public scenario/map fixtures not present")
class PublicFixtureTests(unittest.TestCase):
    def test_load_scenario(self):
        state = scenario.load_scenario(SCENARIO, MAP)
        self.assertEqual(len(state.units), 12)
        self.assertEqual(len(state.objectives), 2)
        self.assertEqual(state.map.rows, 100)
        self.assertEqual(state.map.cols, 100)
        # Public scenario 1231 has red and blue vehicles.
        self.assertTrue(any(u.color == 0 for u in state.units.values()))
        self.assertTrue(any(u.color == 1 for u in state.units.values()))
        for u in state.units.values():
            self.assertGreaterEqual(u.blood, 1)
            self.assertTrue(0 <= u.pos < 10000)

    def test_demo_runs_and_is_deterministic(self):
        s1 = scenario.load_scenario(SCENARIO, MAP, seed=7)
        s2 = scenario.load_scenario(SCENARIO, MAP, seed=7)
        r1 = engine.run_demo(s1, max_ticks=15)
        r2 = engine.run_demo(s2, max_ticks=15)
        self.assertEqual(r1["tick"], r2["tick"])
        self.assertEqual(r1["score"], r2["score"])
        self.assertEqual([e["type"] for e in r1["events"]],
                         [e["type"] for e in r2["events"]])
        self.assertGreaterEqual(len(r1["events"]), 1)

    def test_other_public_fixtures_load(self):
        pairs = [
            ("1930331196.json", "map_96"),
            ("2120131194.json", "map_212"),
        ]
        for name, map_dir in pairs:
            sp = os.path.join(DATA, "scenarios", name)
            mp = os.path.join(DATA, "maps", map_dir, "basic.json")
            if not (os.path.isfile(sp) and os.path.isfile(mp)):
                continue
            state = scenario.load_scenario(sp, mp, seed=3)
            self.assertGreater(len(state.units), 0, name)
            self.assertGreater(len(state.objectives), 0, name)
            result = engine.run_demo(state, max_ticks=3)
            self.assertLessEqual(result["tick"], 3, name)


if __name__ == "__main__":
    unittest.main()
