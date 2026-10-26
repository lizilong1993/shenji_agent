"""Capability catalog wiring: card -> profile, scenario catalog_mode.

Run from ``situation3d/``:
    python3 -m unittest tests/test_catalog_engine.py
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from capabilities import CapabilityCatalog  # noqa: E402
from engine.rolecards import load_registry  # noqa: E402
from mobility.card_profile import (DOMAIN_MAP, FOOTPRINT_DEFAULT,  # noqa: E402
                                   profile_from_card)
from sim.rolecards_bridge import entity_type_for  # noqa: E402
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


def tank_scenario(engine):
    return {
        "engine": dict({"detection_model": "table", "scan_interval_s": 30,
                        "use_role_cards": True}, **engine),
        "operators": [{
            "obj_id": 1, "name": "ZTZ-99A", "color": 0, "type": 2,
            "domain": "land", "lon": 119.905, "lat": 23.095,
            "subclass_id": "land.vehicle.mbt", "blood": 4, "max_blood": 4,
            "observe_distance": {"1": 10, "2": 25, "3": 1},
        }],
        "missions": [],
        "roe": {},
    }


class CardProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()
        cls.catalog = CapabilityCatalog(cls.registry)
        cls.card = cls.registry.effective("land.vehicle.mbt")

    def test_legacy_profile_is_unchanged(self):
        legacy = profile_from_card(self.card, key="land.vehicle.mbt")
        self.assertEqual(legacy.length_m, 6.0)   # domain default
        self.assertEqual(legacy.width_m, 2.8)
        self.assertEqual(legacy.vehicle_height_m, 0.0)
        self.assertEqual(legacy.weight_t, 0.0)
        self.assertEqual(legacy.beam_m, 0.0)

    def test_legacy_fields_match_old_rules_for_all_cards(self):
        for type_id in self.registry.subclasses:
            card = self.registry.effective(type_id)
            motion = card.get("motion") or {}
            sea = motion.get("sea") or {}
            domain = DOMAIN_MAP.get(
                str((card.get("identity") or {}).get("primary_domain") or "land"),
                "land")
            length_d, width_d, _, _ = FOOTPRINT_DEFAULT[domain]
            profile = profile_from_card(card, key=type_id)
            self.assertEqual(profile.length_m,
                             float(motion.get("length_m") or length_d), type_id)
            self.assertEqual(profile.width_m,
                             float(motion.get("width_m") or width_d), type_id)
            self.assertEqual(profile.beam_m,
                             float(sea.get("beam_m") or 0.0), type_id)
            self.assertEqual(profile.vehicle_height_m, 0.0, type_id)
            self.assertEqual(profile.weight_t, 0.0, type_id)

    def test_typed_profile_uses_card_dimensions(self):
        typed = profile_from_card(self.card, key="land.vehicle.mbt",
                                  entity=self.catalog.get("land.vehicle.mbt"))
        self.assertAlmostEqual(typed.length_m, 11.0)
        self.assertAlmostEqual(typed.width_m, 3.4)
        self.assertAlmostEqual(typed.vehicle_height_m, 2.4)
        self.assertAlmostEqual(typed.beam_m, 3.4)
        self.assertEqual(typed.weight_t, 0.0)  # no mass in the card yet

    def test_ship_typed_dimensions(self):
        ship = self.catalog.get("sea.surface.destroyer")
        typed = profile_from_card(self.registry.effective("sea.surface.destroyer"),
                                  key="sea.surface.destroyer", entity=ship)
        self.assertAlmostEqual(typed.length_m, 157.0)
        self.assertAlmostEqual(typed.beam_m, 17.2)
        self.assertAlmostEqual(typed.vehicle_height_m, 30.0)

    def test_air_typed_beam_stays_zero(self):
        fighter = self.catalog.get("air.aircraft.fighter")
        typed = profile_from_card(self.registry.effective("air.aircraft.fighter"),
                                  key="air.aircraft.fighter", entity=fighter)
        self.assertEqual(typed.beam_m, 0.0)
        self.assertAlmostEqual(typed.width_m, 14.7)

    def test_entity_type_for_is_cached(self):
        self.assertIs(entity_type_for("land.vehicle.mbt"),
                      entity_type_for("land.vehicle.mbt"))


class ScenarioCatalogModeTest(unittest.TestCase):
    def test_new_mode_sets_typed_dimensions(self):
        world = build_world(tank_scenario({"catalog_mode": "new"}),
                            make_basic(), seed=1)
        profile = world.entities[1].profile
        self.assertAlmostEqual(profile.length_m, 11.0)
        self.assertAlmostEqual(profile.width_m, 3.4)
        self.assertAlmostEqual(profile.vehicle_height_m, 2.4)

    def test_default_and_invalid_modes_stay_legacy(self):
        for engine in ({}, {"catalog_mode": "bogus"}):
            world = build_world(tank_scenario(engine), make_basic(), seed=1)
            profile = world.entities[1].profile
            self.assertEqual(profile.length_m, 6.0)
            self.assertEqual(profile.width_m, 2.8)
            self.assertEqual(profile.vehicle_height_m, 0.0)

    def test_glb_override_keeps_typed_beam_in_sync(self):
        scen = tank_scenario({"catalog_mode": "new"})
        scen["operators"][0].update({"length_m": 12.0, "width_m": 4.5})
        profile = build_world(scen, make_basic(), seed=1).entities[1].profile
        self.assertAlmostEqual(profile.length_m, 12.0)
        self.assertAlmostEqual(profile.width_m, 4.5)
        self.assertAlmostEqual(profile.beam_m, 4.5)

        legacy = tank_scenario({})
        legacy["operators"][0].update({"length_m": 12.0, "width_m": 4.5})
        profile = build_world(legacy, make_basic(), seed=1).entities[1].profile
        self.assertAlmostEqual(profile.width_m, 4.5)
        self.assertEqual(profile.beam_m, 0.0)


if __name__ == "__main__":
    unittest.main()
