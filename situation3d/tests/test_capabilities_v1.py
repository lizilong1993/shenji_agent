"""Capability catalog (P2 slice): projection, provenance, gates, dual read.

Run from ``situation3d/``:
    python3 -m unittest tests/test_capabilities_v1.py
"""

from __future__ import annotations

import copy
import os
import sys
import unittest
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from capabilities import (  # noqa: E402
    CapabilityCatalog, EntityType, evaluate, evaluate_all, resolve_type,
)
from engine.rolecards import load_registry  # noqa: E402

UNITS = ("m", "m/s", "m/s2", "deg", "level", "class", "hp", "t")


class CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()
        cls.catalog = CapabilityCatalog(cls.registry)

    def test_every_subclass_becomes_a_type(self):
        self.assertEqual(len(self.catalog), len(self.registry.subclasses))
        for type_id in self.registry.subclasses:
            entity = self.catalog.get(type_id)
            self.assertIsInstance(entity, EntityType)
            self.assertEqual(entity.type_id, type_id)
            self.assertTrue(entity.domain)
            self.assertIn(entity.unknown_policy,
                          ("deny", "advisory", "allow_with_penalty", "allow"))

    def test_quantities_have_units_and_sources(self):
        for type_id in self.catalog.ids():
            for key, quantity in self.catalog.get(type_id).quantities.items():
                self.assertTrue(quantity.unit, key)
                self.assertIn(quantity.unit, UNITS)
                if quantity.known:
                    self.assertTrue(quantity.source.primary(),
                                    "%s/%s missing source" % (type_id, key))
                    self.assertIn(quantity.source.confidence,
                                  ("high", "medium", "low", "unknown"))

    def test_tank_projection(self):
        mbt = self.catalog.get("land.vehicle.mbt")
        self.assertEqual(mbt.class_id, "land.vehicle")
        self.assertEqual(mbt.domain, "land")
        self.assertAlmostEqual(mbt.quantity("speed.max_ms").value, 22.22)
        self.assertAlmostEqual(mbt.quantity("terrain.gradient_max_deg").value, 30.0)
        self.assertAlmostEqual(mbt.quantity("terrain.wading_m").value, 1.2)
        self.assertAlmostEqual(mbt.quantity("dim.width_m").value, 3.4)
        self.assertAlmostEqual(mbt.quantity("dim.height_m").value, 2.4)
        self.assertFalse(mbt.flags["amphibious"])
        self.assertFalse(mbt.quantity("mass.t").known)
        self.assertEqual(mbt.quantity("mass.t").unit, "t")
        self.assertEqual(mbt.quantity("dim.width_m").unit, "m")

    def test_ship_projection(self):
        ship = self.catalog.get("sea.surface.destroyer")
        self.assertEqual(ship.domain, "sea")
        self.assertAlmostEqual(ship.quantity("sea.draft_m").value, 6.5)
        self.assertAlmostEqual(ship.quantity("sea.ukc_m").value, 0.5)
        self.assertAlmostEqual(ship.quantity("sea.beam_m").value, 17.2)
        self.assertAlmostEqual(ship.quantity("sea.sea_state_max").value, 6.0)
        self.assertAlmostEqual(ship.quantity("dim.length_m").value, 157.0)

    def test_catalog_does_not_mutate_registry(self):
        registry = load_registry()
        tables = ("classes", "subclasses", "equipment", "types",
                  "aliases", "sdk_map")
        before = {name: copy.deepcopy(getattr(registry, name)) for name in tables}
        catalog = CapabilityCatalog(registry)
        for name in tables:
            self.assertEqual(before[name], getattr(registry, name), name)
        entity = catalog.get("land.vehicle.mbt")
        entity.sdk.setdefault("observe_distance", {})["infantry"] = 999
        self.assertNotEqual(
            registry.subclasses["land.vehicle.mbt"]["sdk"].get(
                "observe_distance", {}).get("infantry"), 999)

    def test_unknown_policy_from_gates_block(self):
        registry = load_registry()
        registry.subclasses["land.vehicle.mbt"]["values"]["gates"] = {
            "unknown_policy": "bogus"}
        entity = CapabilityCatalog(registry).get("land.vehicle.mbt")
        self.assertEqual(entity.unknown_policy, "deny")

        registry = load_registry()
        registry.subclasses["land.vehicle.mbt"]["values"]["gates"] = {
            "unknown_policy": "advisory"}
        entity = CapabilityCatalog(registry).get("land.vehicle.mbt")
        self.assertEqual(entity.unknown_policy, "advisory")

    def test_dual_read_modes(self):
        legacy = resolve_type("land.vehicle.mbt", mode="legacy", registry=self.registry)
        self.assertIsInstance(legacy, dict)
        self.assertIn("motion", legacy)
        modern = resolve_type("land.vehicle.mbt", mode="new", registry=self.registry)
        self.assertIsInstance(modern, EntityType)
        self.assertEqual(modern.type_id, "land.vehicle.mbt")
        self.assertIsNone(resolve_type("no.such.type", mode="new",
                                       registry=self.registry))
        with self.assertRaises(ValueError):
            resolve_type("land.vehicle.mbt", mode="bogus", registry=self.registry)


class GateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()
        cls.catalog = CapabilityCatalog(cls.registry)
        cls.tank = cls.catalog.get("land.vehicle.mbt")
        cls.ship = cls.catalog.get("sea.surface.destroyer")
        cls.fighter = cls.catalog.get("air.aircraft.fighter")
        cls.amp = cls.catalog.get("land.vehicle.amphibious")
        cls.sub = cls.catalog.get("sea.subsurface.attack")

    def test_slope_boundary(self):
        self.assertEqual(evaluate(self.tank, "slope", {"gradient_deg": 30.0}).decision,
                         "pass")
        denied = evaluate(self.tank, "slope", {"gradient_deg": 30.1})
        self.assertEqual(denied.decision, "deny")
        self.assertTrue(denied.blocking)
        side = evaluate(self.tank, "slope", {"side_slope_deg": 18.0})
        self.assertEqual(side.decision, "deny")
        self.assertEqual(evaluate(self.tank, "slope", {}).decision, "unknown")
        # A missing quantity must not mask a confirmed violation elsewhere.
        quantities = dict(self.tank.quantities)
        del quantities["terrain.gradient_max_deg"]
        partial = replace(self.tank, quantities=quantities)
        result = evaluate(partial, "slope",
                          {"gradient_deg": 10.0, "side_slope_deg": 30.0})
        self.assertEqual(result.decision, "deny")

    def test_wading(self):
        self.assertEqual(evaluate(self.tank, "wading", {"water_depth_m": 1.0}).decision,
                         "pass")
        self.assertEqual(evaluate(self.tank, "wading", {"water_depth_m": 1.3}).decision,
                         "deny")
        # Amphibious cards use swim depth (4.0 m) over wading depth (1.2 m).
        self.assertEqual(evaluate(self.amp, "wading",
                                  {"water_depth_m": 3.9}).decision, "pass")
        self.assertEqual(evaluate(self.amp, "wading",
                                  {"water_depth_m": 4.1}).decision, "deny")

    def test_bridge_limits(self):
        self.assertEqual(
            evaluate(self.tank, "bridge", {"bridge_max_width_m": 3.0}).decision,
            "deny")
        self.assertEqual(
            evaluate(self.tank, "bridge", {"bridge_max_width_m": 3.4}).decision,
            "pass")
        self.assertEqual(
            evaluate(self.tank, "bridge", {"bridge_max_height_m": 3.0}).decision,
            "pass")
        weight = evaluate(self.tank, "bridge", {"bridge_max_weight_t": 40.0})
        self.assertEqual(weight.decision, "unknown")
        self.assertTrue(weight.blocking)
        relaxed = evaluate(self.tank, "bridge", {"bridge_max_weight_t": 40.0},
                           unknown_policy="allow")
        self.assertFalse(relaxed.blocking)
        # Unknown weight must not mask a confirmed width violation.
        mixed = evaluate(self.tank, "bridge",
                         {"bridge_max_weight_t": 40.0, "bridge_max_width_m": 2.0},
                         unknown_policy="allow")
        self.assertEqual(mixed.decision, "deny")
        self.assertTrue(mixed.blocking)
        bad = evaluate(self.tank, "bridge", {"bridge_max_weight_t": 40.0},
                       unknown_policy="bogus")
        self.assertTrue(bad.blocking)
        self.assertEqual(evaluate(self.tank, "bridge", {}).decision, "unknown")

    def test_sea_state(self):
        self.assertEqual(evaluate(self.ship, "sea_state", {"sea_state": 6}).decision,
                         "pass")
        self.assertEqual(evaluate(self.ship, "sea_state", {"sea_state": 7}).decision,
                         "deny")
        self.assertEqual(evaluate(self.tank, "sea_state", {"sea_state": 2}).decision,
                         "unknown")

    def test_ukc(self):
        self.assertEqual(evaluate(self.ship, "ukc", {"water_depth_m": 7.0}).decision,
                         "pass")
        self.assertEqual(evaluate(self.ship, "ukc", {"water_depth_m": 6.9}).decision,
                         "deny")
        self.assertEqual(evaluate(self.tank, "ukc", {"water_depth_m": 10.0}).decision,
                         "unknown")
        self.assertEqual(evaluate(self.sub, "ukc",
                                  {"water_depth_m": 100.0}).decision, "pass")
        self.assertEqual(evaluate(self.sub, "ukc",
                                  {"water_depth_m": 301.0}).decision, "deny")

    def test_runway(self):
        self.assertEqual(
            evaluate(self.fighter, "runway", {"runway_length_m": 1200}).decision,
            "pass")
        self.assertEqual(
            evaluate(self.fighter, "runway", {"runway_length_m": 1000}).decision,
            "deny")
        self.assertEqual(
            evaluate(self.tank, "runway", {"runway_length_m": 1000}).decision,
            "unknown")

    def test_evaluate_all_and_bad_gate(self):
        results = evaluate_all(self.tank, {"gradient_deg": 10.0,
                                           "water_depth_m": 0.5})
        self.assertEqual(set(results), {"slope", "wading", "bridge",
                                        "sea_state", "ukc", "runway"})
        with self.assertRaises(ValueError):
            evaluate(self.tank, "warp", {})


if __name__ == "__main__":
    unittest.main()
