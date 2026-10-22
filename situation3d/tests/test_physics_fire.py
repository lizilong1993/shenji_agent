"""P1.5 physical fire tests: data coverage, monotonicity, components, indirect.

Run from ``situation3d/``:
    python3 -m unittest tests/test_physics_fire.py
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
from geo.rule_grid import RuleGrid  # noqa: E402
from mobility.profile import PROFILES  # noqa: E402
from rules import physics_direct_fire as pdir  # noqa: E402
from rules import physics_indirect as pind  # noqa: E402
from rules.weapons import WeaponTables, load_wpdata  # noqa: E402
from sim import perception  # noqa: E402
from sim.entity import Entity, Pose  # noqa: E402
from sim.motion import integrate  # noqa: E402
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


def operator(uid, name, cell, color=0, weapons=None, armor=0, blood=3):
    return {"obj_id": uid, "name": name, "color": color, "type": 2,
            "sub_type": 0, "cur_hex": cell, "blood": blood,
            "max_blood": blood, "armor": armor, "value": 10,
            "carry_weapon_ids": weapons or [], "domain": "land",
            "basic_speed": 30, "observe_distance": {"1": 10, "2": 25, "3": 1},
            "move_state": 0, "stop": 1, "readiness": 0.85, "fuel": 1.0,
            "emitting": True}


def world_for(ops, engine=None, seed=11):
    scenario = {"scenario_id": 1, "cities": [], "operators": ops,
                "time": {"cur_step": 0, "tick": 1.0, "max_time": 999},
                "engine": engine or {}}
    return build_world(scenario, make_basic(), seed=seed)


class DataCoverageTest(unittest.TestCase):
    def test_all_direct_weapons_have_ballistics(self):
        w = load_wpdata()
        cols = w["columns"]
        ids = set()
        for row in w["rows"]:
            rec = dict(zip(cols, row)) if isinstance(row, list) else row
            if rec.get("weapon_id") is not None:
                ids.add(int(rec["weapon_id"]))
        indirect = {"72", "88", "89"}
        direct = {str(i) for i in ids} - indirect
        table = pdir.ballistics_doc()["by_weapon_id"]
        missing = sorted(direct - set(table))
        self.assertEqual(missing, [])
        for wid in indirect:
            self.assertIn(wid, pind.doc()["by_weapon_id"])

    def test_penetration_and_thickness_monotonic(self):
        profile = pdir.profile_for_weapon(type("W", (), {"weapon_id": 36})())
        near = pdir.pen_at_distance(profile, 300.0)
        far = pdir.pen_at_distance(profile, 3000.0)
        self.assertGreater(near, far)
        flat, _ = pdir._effective_thickness(profile, 100.0, 0.0)
        sloped, _ = pdir._effective_thickness(profile, 100.0, 55.0)
        self.assertLess(flat, sloped)


class DirectPhysicsTest(unittest.TestCase):
    def test_component_damage_effects(self):
        world = world_for([operator(1, "T-1", 1010, weapons=[36]),
                           operator(2, "Target", 1011, color=1)])
        e = world.entities[1]
        self.assertAlmostEqual(e.components["mobility"], 1.0)
        world.apply_component_damage(1, "mobility", 0.6)
        self.assertAlmostEqual(e.components["mobility"], 0.4)
        world.apply_component_damage(1, "sensors", 0.5)
        self.assertLess(e.components["sensors"], 0.6)
        # Sensor damage shrinks the sensor suite.
        plain = Entity(9, "x", 0, "land", "tracked", Pose(120.0, 23.0))
        plain.profile = PROFILES["tracked"]
        full = perception.specs_for(plain)
        injured = Entity(8, "y", 0, "land", "tracked", Pose(120.0, 23.0))
        injured.profile = PROFILES["tracked"]
        injured.components["sensors"] = 0.5
        damaged = perception.specs_for(injured)
        self.assertLess(damaged[0].range_m, full[0].range_m)

    def test_hit_probability_decreases_with_distance(self):
        import random
        profile = pdir.profile_for_weapon(type("W", (), {"weapon_id": 36})())

        def hit_rate(distance):
            rng = random.Random(5)
            hits = 0
            for _ in range(2000):
                if pdir._aim_error_m(profile, distance, rng) <= 3.0:
                    hits += 1
            return hits / 2000.0

        self.assertGreater(hit_rate(500.0), hit_rate(2500.0))

    def test_dry_run_does_not_mutate_components(self):
        world = world_for([operator(1, "Heavy", 1010, weapons=[36]),
                           operator(2, "Soft", 1011, color=1, armor=0)])
        world.direct_fire_model = "shadow"
        ctx = __import__("sim.context", fromlist=["WorldRuleContext"]).WorldRuleContext(world)
        before = dict(world.entities[2].components)
        for _ in range(30):
            world.cooldowns.pop((1, "36"), None)
            world.entities[1].ammo["36"] = 10
            world.seen[1] = {2}
            world._engage(ctx)
            if any("shadow_physics" in e for e in world.events):
                break
        self.assertEqual(world.entities[2].components, before)

    def test_aa_only_weapon_cannot_fire_ground(self):
        world = world_for([operator(1, "SAM", 1010, weapons=[3]),
                           operator(2, "Inf", 1011, color=1, armor=0)])
        world.direct_fire_model = "physics"
        world.seen[1] = {2}
        ctx = __import__("sim.context", fromlist=["WorldRuleContext"]).WorldRuleContext(world)
        self.assertFalse(pdir.can_engage(ctx.view(world.entities[1]),
                                         ctx.view(world.entities[2]), ctx))

    def test_rifle_penetrates_unarmored(self):
        profile = pdir.profile_for_weapon(type("W", (), {"weapon_id": 29})())
        pen = pdir.pen_at_distance(profile, 100.0)
        self.assertGreater(pen, 10.0)  # armor class 0 front = 10 mm

    def test_cover_reduces_hit_rate(self):
        import random

        class CoverCtx(object):
            def __init__(self, cover):
                from rules.context import Cover
                self.rng = random.Random(5)
                self._cover = Cover.URBAN if cover else Cover.NONE

            def cover_at(self, target):
                return self._cover

            def bearing_deg(self, a, b):
                return 0.0

            def distance_m(self, a, b):
                return 800.0

            def target_kind(self, t):
                return "vehicle"

            def weapon_ready(self, uid, key):
                return True

            def consume_ammo(self, *a):
                return True

            def mark_fired(self, *a):
                pass

            def apply_damage(self, *a):
                pass

            def apply_suppression(self, *a):
                pass

        world = world_for([operator(1, "Heavy", 1010, weapons=[36]),
                           operator(2, "Soft", 1011, color=1, armor=0)])
        view_a = world.entities[1].to_rule_view()
        view_b = world.entities[2].to_rule_view()
        open_ctx = CoverCtx(False)
        cover_ctx = CoverCtx(True)
        ev_open = pdir.resolve_physics_direct_fire(view_a, view_b, open_ctx,
                                                   dry_run=True)
        ev_cover = pdir.resolve_physics_direct_fire(view_a, view_b, cover_ctx,
                                                    dry_run=True)
        self.assertIsNotNone(ev_open)
        self.assertIsNotNone(ev_cover)
        self.assertLess(ev_cover["target_radius_m"],
                        ev_open["target_radius_m"])

    def test_component_zero_disables(self):
        world = world_for([operator(1, "Heavy", 1010, weapons=[36]),
                           operator(2, "Soft", 1011, color=1, armor=0)])
        ctx = __import__("sim.context", fromlist=["WorldRuleContext"]).WorldRuleContext(world)
        world.apply_component_damage(1, "firepower", 1.0)
        self.assertFalse(pdir.can_engage(ctx.view(world.entities[1]),
                                         ctx.view(world.entities[2]), ctx))
        world.apply_component_damage(1, "sensors", 1.0)
        self.assertEqual(perception.specs_for(world.entities[1]), [])

    def test_sensor_component_reduces_table_detection(self):
        from sim import spotting
        from sim.sensors import Sensor
        world = world_for([operator(1, "Radar", 1010),
                           operator(2, "Target", 1011, color=1)])
        e = world.entities[1]
        e.sensors = [Sensor("radar", 5.0, 1.0)]
        target = world.entities[2]
        full = spotting.view_range_m(e, 200.0, target)
        world.apply_component_damage(1, "sensors", 0.9)  # 0.1 -> 20% range
        reduced = spotting.view_range_m(e, 200.0, target)
        self.assertLess(reduced, full)
        world = world_for([operator(1, "Heavy", 1010, weapons=[36]),
                           operator(2, "Soft", 1011, color=1, armor=0)])
        world.direct_fire_model = "physics"
        world.seen[1] = {2}
        fire = []
        for _ in range(4):
            world.step()
            fire = [e for e in world.events
                    if e.get("type") == "direct_fire" and e.get("model") == "physics"]
            if fire:
                break
        self.assertTrue(fire)
        ev = fire[0]
        self.assertIn("component", ev)
        self.assertIn("component_health", ev)
        self.assertEqual(ev["model"], "physics")

    def test_default_model_is_physics(self):
        world = world_for([operator(1, "A", 1010, weapons=[36])])
        self.assertEqual(world.direct_fire_model, "physics")
        self.assertEqual(world.indirect_fire_model, "physics")
        legacy = world_for([], engine={"direct_fire_model": "crt",
                                       "indirect_fire_model": "table"})
        self.assertEqual(legacy.direct_fire_model, "crt")
        self.assertEqual(legacy.indirect_fire_model, "table")


class IndirectPhysicsTest(unittest.TestCase):
    def _profile(self):
        return pind.doc()["by_weapon_id"]["72"]

    def test_tof_increases_with_range(self):
        prof = self._profile()
        t1 = pind.time_of_flight(prof, 4000.0)
        t2 = pind.time_of_flight(prof, 10000.0)
        self.assertIsNotNone(t1)
        self.assertGreater(t2, t1)

    def test_observed_cep_is_smaller(self):
        prof = self._profile()
        import random
        unobserved = pind.solve_impact(prof, 8000.0, False,
                                       random.Random(3))
        observed = pind.solve_impact(prof, 8000.0, True,
                                     random.Random(3))
        self.assertLess(observed["cep_m"], unobserved["cep_m"])
        self.assertLess(observed["sigma_m"], unobserved["sigma_m"])

    def test_impact_is_deterministic(self):
        prof = self._profile()
        import random
        a = pind.solve_impact(prof, 8000.0, False, random.Random(7))
        b = pind.solve_impact(prof, 8000.0, False, random.Random(7))
        self.assertEqual(a, b)

    def test_blast_falloff_and_kind(self):
        prof = self._profile()
        near = pind.blast_damage(prof, 5.0, "infantry", 0)
        far = pind.blast_damage(prof, 50.0, "infantry", 0)
        outside = pind.blast_damage(prof, 200.0, "infantry", 0)
        armored = pind.blast_damage(prof, 5.0, "vehicle", 4)
        soft = pind.blast_damage(prof, 5.0, "vehicle", 0)
        self.assertGreater(near, far)
        self.assertEqual(outside, 0)
        self.assertGreater(near, armored)
        self.assertGreaterEqual(soft, armored)

    def test_blast_armor_min_damage(self):
        prof = self._profile()
        self.assertGreaterEqual(pind.blast_damage(prof, 0.0, "vehicle", 4), 1)

    def test_physics_min_range_does_not_fall_back(self):
        world = world_for([operator(1, "Arty", 202, weapons=[72]),
                           operator(2, "Close", 204, color=1, armor=0)],
                          engine={"indirect_fire_model": "physics"})
        ctx = __import__("sim.context", fromlist=["WorldRuleContext"]).WorldRuleContext(world)
        aim = ctx.rule_cell(ctx.view(world.entities[2]))
        self.assertIsNone(world._resolve_indirect(
            ctx.view(world.entities[1]), aim, ctx, observed=True))

    def test_side_fused_indirect_end_to_end(self):
        world = world_for([operator(1, "Arty", 202, weapons=[72]),
                           operator(2, "FO", 2020),
                           operator(3, "Tank", 2021, color=1, armor=3)],
                          engine={"indirect_fire_model": "physics"})
        world.seen.pop(1, None)
        world.seen[2] = {3}
        ctx = __import__("sim.context", fromlist=["WorldRuleContext"]).WorldRuleContext(world)
        world._engage(ctx)
        events = [e for e in world.events
                  if e.get("type") == "indirect_fire"
                  and e.get("model") == "physics"]
        self.assertTrue(events)
        self.assertEqual(events[0]["actors"][0], 1)

    def test_world_indirect_physics_event(self):
        world = world_for([operator(1, "Arty", 202, weapons=[72]),
                           operator(2, "Tank", 2020, color=1, armor=3)],
                          engine={"indirect_fire_model": "physics"})
        ctx = __import__("sim.context", fromlist=["WorldRuleContext"]).WorldRuleContext(world)
        aim = ctx.rule_cell(ctx.view(world.entities[2]))
        ev = world._resolve_indirect(ctx.view(world.entities[1]), aim, ctx,
                                     observed=True)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["model"], "physics")
        self.assertGreater(ev["tof_s"], 0.0)
        self.assertIn("impact_geo", ev)
        self.assertGreaterEqual(ev["blast_radius_m"], 0.0)
        # Same seed -> same solution.
        world2 = world_for([operator(1, "Arty", 202, weapons=[72]),
                            operator(2, "Tank", 2020, color=1, armor=3)],
                           engine={"indirect_fire_model": "physics"})
        ctx2 = __import__("sim.context", fromlist=["WorldRuleContext"]).WorldRuleContext(world2)
        ev2 = world2._resolve_indirect(ctx2.view(world2.entities[1]), aim, ctx2,
                                       observed=True)
        self.assertEqual(ev["impact_geo"], ev2["impact_geo"])
        self.assertEqual(ev["damage"], ev2["damage"])


if __name__ == "__main__":
    unittest.main()
