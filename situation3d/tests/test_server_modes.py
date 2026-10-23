"""Mode isolation tests for the dual-mode server (TW2 D0)."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
SERVER = os.path.join(SITUATION3D, "server")
for p in (SITUATION3D, SERVER):
    if p not in sys.path:
        sys.path.insert(0, p)

from modes import (  # noqa: E402
    assert_no_simulation, capabilities, loaded_simulation_modules, normalize_mode,
)


class ServerModeTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_mode(None), "demo")
        self.assertEqual(normalize_mode(" TEST "), "test")
        with self.assertRaises(ValueError):
            normalize_mode("prod")

    def test_capabilities(self):
        demo = capabilities("demo")
        test = capabilities("test")
        self.assertFalse(demo["simulate"])
        self.assertFalse(demo["runs"])
        self.assertFalse(demo["promote"])
        self.assertTrue(demo["replay"])
        self.assertTrue(test["simulate"])
        self.assertTrue(test["runs"])
        self.assertTrue(test["promote"])

    def test_loaded_simulation_modules_uses_package_prefixes(self):
        mods = {"sim", "sim.world", "rules", "rules.physics_direct_fire",
                "simulation_tools", "engine.inference"}
        found = loaded_simulation_modules(mods)
        self.assertEqual(found, ["rules", "rules.physics_direct_fire",
                                 "sim", "sim.world"])

    def test_demo_refuses_loaded_simulator(self):
        with self.assertRaises(RuntimeError):
            assert_no_simulation("demo", {"sim.world": object(), "engine": object()})
        with self.assertRaises(RuntimeError):
            assert_no_simulation("demo", {"rules.engagement": object()})

    def test_demo_allows_clean_process_and_test_mode_ignores(self):
        assert_no_simulation("demo", {"engine.inference": object(), "os": object()})
        assert_no_simulation("test", {"sim.world": object()})


if __name__ == "__main__":
    unittest.main()
