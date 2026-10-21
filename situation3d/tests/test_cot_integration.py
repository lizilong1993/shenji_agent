"""Tests for the COT-derived spotting, physics and commander layers.

Run from ``situation3d/``:
    python3 -m unittest tests/test_cot_integration.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from compat.legacy import LegacyHexGrid  # noqa: E402
from env.grid import MapGridEnvironment  # noqa: E402
from env.service import LandCover  # noqa: E402
from sim import spotting  # noqa: E402
from sim.commander import (CommanderOrder, CommanderRegistry, RuleCommander,
                           RemoteCommander)  # noqa: E402
from sim.entity import Entity, Pose  # noqa: E402
from sim.scenario import build_world  # noqa: E402
from rules import physics_direct_fire as physics  # noqa: E402

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 24, "rows": 24,
        "width_m": 200.0}


def make_basic(rows=24, cols=24, water_cols=(), forest_cols=(), urban_cols=(),
               roads_row=None, slope=0.0):
    cells = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cond = 0
            if c in water_cols:
                cond = 4
            elif c in forest_cols:
                cond = 1
            elif c in urban_cols:
                cond = 2
            roads = [1] * 6 if (roads_row is not None and r == roads_row) else [0] * 6
            row.append({"pos": r * 100 + c, "elev": int(100 + slope * r),
                        "cond": cond, "roads": roads, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID), "map_data": cells}


def operator(oid, color, hexpos, name="重型坦克", type_=2, blood=3, armor=4,
             weapons=(36,), sub_type=0, value=10, **extra):
    item = {"obj_id": oid, "color": color, "type": type_, "name": name,
            "sub_type": sub_type, "blood": blood, "max_blood": blood,
            "armor": armor, "cur_hex": hexpos, "carry_weapon_ids": list(weapons),
            "value": value, "move_state": 0}
    item.update(extra)
    return item


def scenario(ops, cities=(), engine=None):
    doc = {"scenario_id": 1, "operators": list(ops), "cities": list(cities),
           "time": {"cur_step": 0, "tick": 1.0, "max_time": 3600}}
    if engine:
        doc["engine"] = engine
    return doc


def lonlat(hexpos):
    return LegacyHexGrid.from_grid_meta(GRID).cell_center_lonlat(
        hexpos // 100, hexpos % 100)


class TestSpottingModel(unittest.TestCase):
    def test_view_range_uses_observe_distance(self):
        e = Entity(1, "A", 0, "land", "tracked", Pose(*lonlat(1010)))
        e.observe_distance = {"vehicle": 25, "infantry": 10}
        self.assertAlmostEqual(spotting.view_range_m(e, 200.0), 5000.0)

    def test_concealment_and_firing_bloom(self):
        basic = make_basic(forest_cols=(10,))
        grid = LegacyHexGrid.from_grid_meta(GRID)
        env = MapGridEnvironment(basic, grid)
        target = Entity(2, "B", 1, "land", "tracked", Pose(*lonlat(1011)))
        target.concealment = 0.20
        open_camo = spotting.camo_of(target, env, 0.0)
        self.assertGreaterEqual(open_camo, 0.20)
        forest_target = Entity(3, "C", 1, "land", "tracked",
                               Pose(*grid.cell_center_lonlat(10, 10)))
        forest_target.concealment = 0.20
        forest_camo = spotting.camo_of(forest_target, env, 0.0)
        self.assertGreater(forest_camo, open_camo)
        target.last_fired_s = 0.0
        hot = spotting.camo_of(target, env, 0.0)
        cold = spotting.camo_of(target, env, 10.0)
        self.assertLess(hot, cold)

    def test_can_spot_uses_spot_range_and_los(self):
        basic = make_basic()
        grid = LegacyHexGrid.from_grid_meta(GRID)
        env = MapGridEnvironment(basic, grid)
        observer = Entity(1, "A", 0, "land", "tracked", Pose(*grid.cell_center_lonlat(10, 10)))
        observer.observe_distance = {"vehicle": 25}
        target = Entity(2, "B", 1, "land", "tracked", Pose(*grid.cell_center_lonlat(10, 20)))
        target.concealment = 0.95
        self.assertFalse(spotting.can_spot(observer, target, env, 0.0))
        target.last_fired_s = 0.0
        self.assertTrue(spotting.can_spot(observer, target, env, 0.0))

    def test_shadow_mode_records_diff_and_keeps_table_gameplay(self):
        basic = make_basic()
        world = build_world(scenario([
            operator(1, 0, 1010),
            operator(2, 1, 1011, blood=5),
        ], engine={"detection_model": "shadow"}), basic, seed=1)
        world.step()
        self.assertIsNotNone(world.detection_shadow)
        self.assertIn("table_pairs", world.detection_shadow)
        self.assertIn(2, world.seen.get(1, set()))
        self.assertEqual(world.detection_model, "shadow")

    def test_radio_share(self):
        basic = make_basic()
        world = build_world(scenario([
            operator(1, 0, 1010), operator(2, 0, 1011),
            operator(3, 1, 1020, blood=5),
        ]), basic, seed=1)
        seen = {1: {3}}
        shared = spotting.radio_share(world, seen)
        self.assertIn(3, shared.get(2, set()))


    def test_shadow_does_not_consume_rng(self):
        basic = make_basic()
        base = scenario([operator(1, 0, 1010),
                         operator(2, 1, 1011, blood=5)])
        shadow_sc = scenario([operator(1, 0, 1010),
                              operator(2, 1, 1011, blood=5)],
                             engine={"direct_fire_model": "shadow"})
        a = build_world(base, basic, seed=77)
        b = build_world(shadow_sc, basic, seed=77)
        for _ in range(12):
            ea = a.step()
            eb = b.step()
            self.assertEqual(
                [(e["type"], e.get("actors"), e.get("damage")) for e in ea],
                [(e["type"], e.get("actors"), e.get("damage")) for e in eb])
        self.assertEqual(a.rng.getstate(), b.rng.getstate())
        shadow_fire = [e for e in b.events
                       if e["type"] == "direct_fire" and "shadow_physics" in e]
        self.assertTrue(shadow_fire)
        self.assertEqual(shadow_fire[0]["shadow_physics"]["weapon"],
                         shadow_fire[0]["weapon"])

    def test_plate_angle_conversion(self):
        self.assertAlmostEqual(physics._plate_angle("front", 0.0), 0.0)
        self.assertAlmostEqual(physics._plate_angle("side", 90.0), 0.0)
        self.assertAlmostEqual(physics._plate_angle("rear", 180.0), 0.0)
        self.assertFalse(physics._would_ricochet(
            {"shell_type": "AP", "caliber_mm": 100}, 250.0, 0.0))

    def test_observe_distance_is_per_target_kind(self):
        observer = Entity(1, "A", 0, "land", "tracked", Pose(*lonlat(1010)))
        observer.observe_distance = {"infantry": 10, "vehicle": 25}
        infantry = Entity(2, "I", 1, "land", "tracked", Pose(*lonlat(1011)))
        infantry.unit_type = "infantry"
        infantry.unit_kind = "infantry"
        vehicle = Entity(3, "V", 1, "land", "tracked", Pose(*lonlat(1011)))
        self.assertLessEqual(spotting.view_range_m(observer, 200.0, infantry), 2000.0)
        self.assertGreaterEqual(spotting.view_range_m(observer, 200.0, vehicle), 5000.0)

    def test_spotting_keeps_sea_air_detection(self):
        basic = make_basic()
        world = build_world(scenario([
            operator(1, 0, 1010, name="驱逐舰"),
            operator(2, 1, 1012, name="无人机", type_=3, armor=0, blood=1),
        ], engine={"detection_model": "spotting"}), basic, seed=1)
        world.step()
        self.assertTrue(world.seen.get(1, set()))


class TestPhysicsDirectFire(unittest.TestCase):
    def test_penetration_falloff(self):
        profile = physics.profile_for_weapon(type("W", (), {"weapon_id": 36})())
        self.assertAlmostEqual(physics.pen_at_distance(profile, 100.0), 600.0)
        self.assertAlmostEqual(physics.pen_at_distance(profile, 1000.0), 550.0)
        self.assertAlmostEqual(physics.pen_at_distance(profile, 2000.0), 500.0)
        self.assertGreater(physics.pen_at_distance(profile, 500.0),
                           physics.pen_at_distance(profile, 1500.0))

    def test_ricochet_and_overmatch(self):
        ap = {"shell_type": "AP", "caliber_mm": 100}
        apfsds = {"shell_type": "APFSDS", "caliber_mm": 125}
        self.assertTrue(physics._would_ricochet(ap, 100.0, 75.0))
        self.assertFalse(physics._would_ricochet(apfsds, 20.0, 80.0))
        self.assertEqual(physics._face_for_angle(0.0), "front")
        self.assertEqual(physics._face_for_angle(90.0), "side")
        self.assertEqual(physics._face_for_angle(180.0), "rear")

    def test_physics_event_and_determinism(self):
        basic = make_basic()
        sc = scenario([operator(1, 0, 1010), operator(2, 1, 1011, blood=5)],
                      engine={"direct_fire_model": "physics"})
        a = build_world(sc, basic, seed=99)
        b = build_world(sc, basic, seed=99)
        for _ in range(8):
            ea = a.step()
            eb = b.step()
            self.assertEqual(
                [(e["type"], e.get("model"), e.get("damage"), e.get("pen_mm")) for e in ea],
                [(e["type"], e.get("model"), e.get("damage"), e.get("pen_mm")) for e in eb])
        fire = [e for e in a.events if e["type"] == "direct_fire" and e.get("model") == "physics"]
        self.assertTrue(fire)
        ev = fire[0]
        for key in ("distance_m", "tof_s", "impact_angle_deg", "armor_face",
                    "effective_mm", "pen_mm", "result"):
            self.assertIn(key, ev)

    def test_default_is_crt(self):
        basic = make_basic()
        world = build_world(scenario([operator(1, 0, 1010),
                                      operator(2, 1, 1011, blood=5)]), basic, seed=1)
        for _ in range(2):
            world.step()
        fire = [e for e in world.events if e["type"] == "direct_fire"]
        self.assertTrue(fire)
        self.assertIsNone(fire[0].get("model"))


class TestCommanderAndTactics(unittest.TestCase):
    def test_rule_commander_postures(self):
        state = {
            "units": [
                {"uid": 1, "affiliation": 0, "blood": 1, "max_blood": 4,
                 "lon": 119.9, "lat": 23.1, "in_cover": False},
                {"uid": 2, "affiliation": 0, "blood": 4, "max_blood": 4,
                 "lon": 119.9, "lat": 23.1, "in_cover": True},
            ],
            "enemies": [{"uid": 9, "distance_m": 300.0}],
            "objectives": [{"lon": 119.91, "lat": 23.1, "owner": 1},
                           {"lon": 119.89, "lat": 23.1, "owner": 0}],
        }
        orders = {o.uid: o for o in RuleCommander().decide(state)}
        self.assertEqual(orders[1].posture, "retreat")
        self.assertEqual(orders[2].posture, "hold")
        self.assertEqual(orders[2].target_uid, 9)

    def test_registry_expiry_and_fallback(self):
        class Broken(object):
            def decide(self, state):
                raise RuntimeError("model down")
        registry = CommanderRegistry(Broken(), interval_s=1.0)
        orders = registry.decide({"units": [], "enemies": [], "objectives": []}, 10.0)
        self.assertEqual(orders, [])
        self.assertIn("model down", registry.last_error or "")

        registry2 = CommanderRegistry(RuleCommander(), interval_s=1.0)
        orders2 = registry2.decide({"units": [
            {"uid": 1, "affiliation": 0, "blood": 4, "max_blood": 4,
             "lon": 119.9, "lat": 23.1, "in_cover": False}],
            "enemies": [], "objectives": []}, 1.0)
        self.assertEqual(len(orders2), 1)
        self.assertIsNotNone(registry2.active(1, 2.0))
        self.assertIsNone(registry2.active(1, 100.0))

    def test_remote_commander_failure_is_empty(self):
        client = RemoteCommander("http://127.0.0.1:9", timeout_s=0.2)
        self.assertEqual(client.decide({"units": []}), [])

    def test_tactics_move_toward_objective(self):
        basic = make_basic()
        world = build_world(scenario([
            operator(1, 0, 1005),
        ], engine={"enable_tactics": True}), basic, seed=1)
        cell = (10, 15)
        lon, lat = world.rule_grid.cell_center_lonlat(*cell)
        world.add_objective(cell, value=10, name="OBJ", owner=1)
        start = (world.entities[1].pose.lon, world.entities[1].pose.lat)
        for _ in range(20):
            world.step()
        moved = math.hypot(
            (world.entities[1].pose.lon - start[0]) * 111320.0 * math.cos(math.radians(start[1])),
            (world.entities[1].pose.lat - start[1]) * 111320.0)
        self.assertGreater(moved, 50.0)

    def test_world_commander_attaches(self):
        basic = make_basic()
        world = build_world(scenario([operator(1, 0, 1010)]), basic, seed=1)
        world.set_commander(RuleCommander(), interval_s=1.0)
        self.assertIsNotNone(world.commander)
        world.step()
        self.assertIn(1, world.commander.orders)

    def test_engine_config_is_safe(self):
        basic = make_basic()
        world = build_world(scenario([operator(1, 0, 1010)], engine={
            "detection_model": "typo",
            "direct_fire_model": "typo",
            "enable_tactics": "false",
            "commander_interval_s": None,
        }), basic, seed=1)
        self.assertEqual(world.detection_model, "table")
        self.assertEqual(world.direct_fire_model, "crt")
        self.assertIsNone(world.tactics)
        self.assertEqual(world.commander_interval_s, 2.0)

    def test_engine_can_attach_rule_commander(self):
        basic = make_basic()
        world = build_world(scenario([operator(1, 0, 1010)], engine={
            "enable_commander": True,
        }), basic, seed=1)
        self.assertIsNotNone(world.commander)

    def test_commander_order_expires(self):
        class OneShot(object):
            def __init__(self):
                self.done = False

            def decide(self, state):
                if self.done:
                    return []
                self.done = True
                return [CommanderOrder(uid=1, posture="push",
                                       lon=119.91, lat=23.1, expires_s=1.0)]

        basic = make_basic()
        world = build_world(scenario([operator(1, 0, 1010)]), basic, seed=1)
        world.set_commander(OneShot(), interval_s=1.0)
        world.step()
        self.assertIn(1, world._commander_owned)
        world.step()
        self.assertIsNotNone(world.entities[1].route)
        for _ in range(3):
            world.step()
        self.assertNotIn(1, world._commander_owned)
        self.assertIsNone(world.entities[1].route)


if __name__ == "__main__":
    unittest.main()
