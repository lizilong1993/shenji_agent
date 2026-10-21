"""P2 smoke tests: role cards -> PlatformProfile / Entity fields.

Run from the geo worktree:
    python3 situation3d/tests/test_geo_rolecards.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # situation3d/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.rolecards import load_registry  # noqa: E402
from mobility.card_profile import profile_from_card  # noqa: E402
from sim.scenario import build_world  # noqa: E402
from sim.rolecards_bridge import (  # noqa: E402
    card_for_operator,
    sensors_from_card,
    subclass_for_operator,
    unit_kind_from_card,
    unit_type_from_card,
    weapons_from_card,
)


class GeoRoleCardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()

    def test_profile_from_land_card(self):
        card = self.registry.effective("land.vehicle.mbt")
        p = profile_from_card(card, "land.vehicle.mbt")
        self.assertEqual(p.domain, "land")
        self.assertAlmostEqual(p.max_speed_ms, 22.22, places=2)
        self.assertEqual(p.armor_class, 4)
        self.assertEqual(p.size_class, "large")
        self.assertAlmostEqual(p.terrain_factor("open"), 1.0)
        self.assertAlmostEqual(p.terrain_factor("water"), 0.0)
        self.assertFalse(p.road_bound)
        self.assertAlmostEqual(p.max_gradient_deg, 30.0)

    def test_profile_from_sea_and_air_cards(self):
        sea = profile_from_card(self.registry.effective("sea.surface.destroyer"),
                                "sea.surface.destroyer")
        self.assertEqual(sea.domain, "surface")
        self.assertAlmostEqual(sea.draft_m, 6.5)
        self.assertAlmostEqual(sea.max_speed_ms, 15.43, places=2)
        air = profile_from_card(self.registry.effective("air.aircraft.fighter"),
                                "air.aircraft.fighter")
        self.assertEqual(air.domain, "air")
        self.assertAlmostEqual(air.service_ceiling_m, 17300.0)
        self.assertGreater(air.climb_rate_ms, 0)

    def test_operator_resolution_is_opt_in(self):
        op = {"type": 2, "sub_type": 0, "name": "重型坦克"}
        self.assertIsNone(subclass_for_operator(op, "land", infer=False))
        self.assertEqual(subclass_for_operator(op, "land", infer=True),
                         "land.vehicle.mbt")
        explicit = dict(op, subclass_id="land.vehicle.light_tank")
        self.assertEqual(subclass_for_operator(explicit, "land", infer=False),
                         "land.vehicle.light_tank")

    def test_unknown_fallback(self):
        op = {"type": 2, "name": "某种未登记车辆"}
        self.assertEqual(subclass_for_operator(op, "land", infer=True),
                         "land.vehicle.unknown")
        sid, card = card_for_operator(op, "land", infer=True)
        self.assertEqual(sid, "land.vehicle.unknown")
        self.assertEqual(card["protection"]["max_blood"], 1)

    def test_sensors_and_weapons_from_card(self):
        card = self.registry.effective("land.vehicle.mbt")
        sensors = sensors_from_card(card)
        self.assertTrue(sensors)
        self.assertTrue(any(s.kind in ("radar", "ir") for s in sensors))
        weapons, ammo = weapons_from_card(card)
        self.assertEqual(len(weapons), 3)
        self.assertEqual(ammo["36"], 42)
        self.assertEqual(ammo["84"], 6)
        self.assertTrue(any(w.tables for w in weapons))

    def test_unit_type_mapping(self):
        self.assertEqual(unit_type_from_card("land.infantry.rifle"), "infantry")
        self.assertEqual(unit_type_from_card("land.vehicle.spg"), "artillery")
        self.assertEqual(unit_type_from_card("land.vehicle.sam"), "aa")
        self.assertEqual(unit_type_from_card("sea.surface.destroyer"), "ship")
        self.assertEqual(unit_type_from_card("air.helicopter.attack"), "helicopter")
        self.assertEqual(unit_kind_from_card("sea.subsurface.attack"), "sub")
        self.assertEqual(unit_kind_from_card("land.vehicle.recon"), "recon")

    def test_build_world_with_explicit_card(self):
        basic = {
            "max_row": 2, "max_column": 2, "scale": 200.0,
            "map_data": [[{"cond": 0, "elev": 0}, {"cond": 0, "elev": 0}],
                         [{"cond": 0, "elev": 0}, {"cond": 0, "elev": 0}]],
        }
        scenario = {
            "operators": [{
                "obj_id": 1, "color": 0, "type": 2, "name": "重型坦克",
                "cur_hex": 101, "blood": 4, "subclass_id": "land.vehicle.mbt",
            }]
        }
        world = build_world(scenario, basic)
        entity = world.entities[1]
        self.assertEqual(entity.domain, "land")
        self.assertEqual(entity.profile_id, "land.vehicle.mbt")
        self.assertAlmostEqual(entity.profile.max_speed_ms, 22.22, places=2)
        self.assertEqual(entity.armor, 4)
        self.assertEqual(entity.max_blood, 4)
        self.assertEqual(len(entity.sensors), 2)
        self.assertEqual(len(entity.weapons), 3)
        self.assertIsNotNone(entity.role_card)

    def test_build_world_legacy_path_unchanged(self):
        basic = {
            "max_row": 2, "max_column": 2, "scale": 200.0,
            "map_data": [[{"cond": 0, "elev": 0}, {"cond": 0, "elev": 0}],
                         [{"cond": 0, "elev": 0}, {"cond": 0, "elev": 0}]],
        }
        scenario = {
            "operators": [{
                "obj_id": 2, "color": 0, "type": 2, "name": "重型坦克",
                "cur_hex": 100, "blood": 4,
            }]
        }
        world = build_world(scenario, basic)
        entity = world.entities[2]
        self.assertIsNone(entity.role_card)
        self.assertEqual(entity.profile_id, "vehicle")
        self.assertAlmostEqual(entity.profile.max_speed_ms, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
