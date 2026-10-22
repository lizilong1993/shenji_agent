"""P4 rule tests: bridge/tunnel, minefields, indirect fire."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from mobility.passability import passable, road_edge_passable
from mobility.profile import PROFILES


class FakeEnv(object):
    def __init__(self, edge=None, cover="open", depth=None):
        self.edge = edge
        self.cover = cover
        self.depth = depth

    def landcover(self, lon, lat):
        return self.cover

    def water_depth(self, lon, lat):
        return self.depth

    def slope_deg(self, lon, lat, bearing_deg=0.0):
        return 0.0

    def on_road(self, lon, lat, tolerance_m=25.0):
        return self.edge is not None

    def road_edge_at(self, lon, lat, tolerance_m=25.0):
        return self.edge

    def elevation(self, lon, lat):
        return 10.0


class TestBridgeConstraints(unittest.TestCase):
    def test_weight_limit_blocks_heavy(self):
        env = FakeEnv(edge={"bridge": True, "maxweight": "20 t"})
        self.assertFalse(passable(PROFILES["tracked"], 120.1, 23.0, env))
        self.assertTrue(passable(PROFILES["wheeled"], 120.1, 23.0, env))

    def test_height_limit_blocks_tall(self):
        env = FakeEnv(edge={"tunnel": True, "maxheight": "2.5 m"})
        self.assertFalse(passable(PROFILES["tracked"], 120.1, 23.0, env))
        self.assertTrue(passable(PROFILES["wheeled"], 120.1, 23.0, env))

    def test_width_limit_blocks_wide(self):
        env = FakeEnv(edge={"bridge": True, "width": "3"})
        self.assertFalse(passable(PROFILES["tracked"], 120.1, 23.0, env))

    def test_no_edge_no_constraint(self):
        env = FakeEnv(edge=None)
        self.assertTrue(passable(PROFILES["tracked"], 120.1, 23.0, env))

    def test_amphibious_land_respects_weight(self):
        env = FakeEnv(edge={"bridge": True, "maxweight": "10"})
        self.assertFalse(passable(PROFILES["amphibious"], 120.1, 23.0, env))
        water_env = FakeEnv(edge={"bridge": True, "maxweight": "10"}, depth=1.0)
        self.assertTrue(passable(PROFILES["amphibious"], 120.1, 23.0, water_env))

    def test_edge_passable_helper(self):
        self.assertFalse(road_edge_passable(PROFILES["tracked"],
                                            {"maxweight": "30"}))
        self.assertTrue(road_edge_passable(PROFILES["tracked"],
                                           {"maxweight": "50", "maxheight": "4"}))

    def test_pack_road_edge_lookup(self):
        from compat.legacy import LegacyHexGrid
        from env.pack_env import EnvPackEnvironment
        from geodata.pipeline import build_envpack

        class FakeOsm(object):
            def features(self, bbox):
                c = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                road = [[bbox[0], c[1]], [bbox[2], c[1]]]
                return {"roads": [{"coords": road, "class": "primary",
                                   "bridge": True, "maxheight": "2.5",
                                   "maxweight": "10", "width": "3"}],
                        "water": [], "buildings": [], "landuse": []}

        class FakeDem(object):
            def sample_grid(self, bbox, step_deg):
                return {"lon0": bbox[0], "lat0": bbox[1], "lon1": bbox[2],
                        "lat1": bbox[3], "cols": 2, "rows": 2,
                        "values": [[10.0, 10.0], [10.0, 10.0]]}

        with tempfile.TemporaryDirectory() as tmp:
            basic = {"map_id": 1, "ele_grade": 10,
                     "grid": {"center_lon": 119.9, "center_lat": 23.1,
                              "cols": 24, "rows": 24, "width_m": 200.0},
                     "map_data": [[{"pos": r * 100 + c, "elev": 10, "cond": 0,
                                    "roads": [0] * 6, "rivers": [0] * 6}
                                   for c in range(24)] for r in range(24)]}
            grid = LegacyHexGrid.from_grid_meta(basic["grid"])
            out = os.path.join(tmp, "pack")
            manifest = build_envpack(out, basic, grid, mode="offline",
                                     cache_dir=os.path.join(tmp, "cache"),
                                     use_bathymetry=False,
                                     osm_source=FakeOsm(), dem_source=FakeDem())
            env = EnvPackEnvironment(out)
            c = manifest["bbox"]
            edge = env.road_edge_at((c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0)
            self.assertIsNotNone(edge)
            self.assertEqual(edge["maxheight"], "2.5")
            self.assertTrue(edge["bridge"])
            self.assertFalse(passable(PROFILES["tracked"],
                                      (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0,
                                      env))


class TestMinefields(unittest.TestCase):
    def test_damage_distribution_matches_table(self):
        import random
        from rules.minefield import MinefieldTables
        tables = MinefieldTables()
        rng = random.Random(4242)
        n = 120000
        counts = {}
        for _ in range(n):
            d = tables.resolve("vehicle", 2, rng)["damage"]
            counts[d] = counts.get(d, 0) + 1
        # Exact distribution: 2d6 counts × 中型装甲 correction.
        base = {2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 2, 12: 1}
        rect = {2: -1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 1}
        expected = {}
        total = 36.0
        for roll in range(2, 13):
            ways = roll - 1 if roll <= 7 else 13 - roll
            damage = max(0, base[roll] + rect[roll])
            expected[damage] = expected.get(damage, 0.0) + ways / total
        for damage, prob in expected.items():
            observed = counts.get(damage, 0) / float(n)
            self.assertLess(abs(observed - prob), 0.01,
                            "damage %s: %.3f vs %.3f" % (damage, observed, prob))

    def test_armour_calibration_changes_damage(self):
        from rules.minefield import MinefieldTables

        class FixedRng(object):
            def __init__(self):
                self.calls = 0

            def randint(self, a, b):
                self.calls += 1
                return 5 if self.calls % 2 else 6   # 5 + 6 = 11

        tables = MinefieldTables()
        heavy = tables.resolve("vehicle", 4, FixedRng())
        light = tables.resolve("vehicle", 0, FixedRng())
        self.assertEqual(heavy["roll"], 11)
        self.assertLess(heavy["damage"], light["damage"])
        inf = tables.resolve("infantry", 0, FixedRng())
        self.assertEqual(inf["damage"], 2)

    def _world_with_field(self, obstacle=None):
        from sim.scenario import build_world
        basic = {"map_id": 1, "ele_grade": 10,
                 "grid": {"center_lon": 119.9, "center_lat": 23.1,
                          "cols": 24, "rows": 24, "width_m": 200.0},
                 "map_data": [[{"pos": r * 100 + c, "elev": 10, "cond": 0,
                                "roads": [0] * 6, "rivers": [0] * 6}
                               for c in range(24)] for r in range(24)]}
        grid_lon, grid_lat = 119.9, 23.1
        field = {"name": "MF-1", "kind": "临时雷场", "owner": -1,
                 "ring": [[grid_lon - 0.02, grid_lat - 0.02],
                          [grid_lon + 0.02, grid_lat - 0.02],
                          [grid_lon + 0.02, grid_lat + 0.02],
                          [grid_lon - 0.02, grid_lat + 0.02],
                          [grid_lon - 0.02, grid_lat - 0.02]]}
        scenario = {"scenario_id": 1,
                    "operators": [{"obj_id": 1, "color": 0, "type": 2, "name": "重型坦克",
                                   "sub_type": 0, "blood": 5, "max_blood": 5, "armor": 3,
                                   "cur_hex": 505, "carry_weapon_ids": [36], "value": 10}],
                    "cities": [], "time": {"cur_step": 0, "tick": 10.0, "max_time": 999},
                    "minefields": [field]}
        if obstacle:
            scenario["obstacles"] = [obstacle]
        return build_world(scenario, basic, seed=7)

    def test_minefield_triggers_once(self):
        world = self._world_with_field()
        events = world.step()
        strikes = [e for e in events if e.get("type") == "mine_strike"]
        self.assertEqual(len(strikes), 1)
        self.assertIn(strikes[0]["roll"], range(2, 13))
        events2 = world.step()
        self.assertFalse([e for e in events2 if e.get("type") == "mine_strike"])

    def test_obstacles_block_ground(self):
        ring = [[119.88, 23.08], [119.92, 23.08],
                [119.92, 23.12], [119.88, 23.12], [119.88, 23.08]]
        world = self._world_with_field(
            {"name": "RB-1", "kind": "roadblock", "ring": ring})
        ent = world.entities[1]
        self.assertFalse(world._can_enter_cell(ent, 119.90, 23.10))
        self.assertTrue(world._can_enter_cell(ent, 119.95, 23.10))

    def test_fortification_blocks_vehicles_only(self):
        ring = [[119.88, 23.08], [119.92, 23.08],
                [119.92, 23.12], [119.88, 23.12], [119.88, 23.08]]
        world = self._world_with_field(
            {"name": "FT-1", "kind": "fortification", "ring": ring})
        ent = world.entities[1]
        self.assertFalse(world._can_enter_cell(ent, 119.90, 23.10))
        ent.unit_type = "infantry"
        self.assertTrue(world._can_enter_cell(ent, 119.90, 23.10))

    def test_planner_respects_blocked(self):
        from compat.legacy import LegacyHexGrid
        from env.grid import MapGridEnvironment
        from mobility.planner import GridPlanner
        from mobility.profile import PROFILES
        basic = {"map_id": 1, "ele_grade": 10,
                 "grid": {"center_lon": 119.9, "center_lat": 23.1,
                          "cols": 24, "rows": 24, "width_m": 200.0},
                 "map_data": [[{"pos": r * 100 + c, "elev": 10, "cond": 0,
                                "roads": [0] * 6, "rivers": [0] * 6}
                               for c in range(24)] for r in range(24)]}
        grid = LegacyHexGrid.from_grid_meta(basic["grid"])
        env = MapGridEnvironment(basic, grid)
        planner = GridPlanner(env, grid)
        start = grid.cell_center_lonlat(1, 1)
        goal = grid.cell_center_lonlat(10, 10)
        self.assertIsNone(planner.plan(PROFILES["tracked"], start, goal,
                                       blocked=lambda lon, lat: True))
        route = planner.plan(PROFILES["tracked"], start, goal,
                             blocked=lambda lon, lat: False)
        self.assertIsNotNone(route)


class TestIndirectFire(unittest.TestCase):
    def test_scatter_table_lookup(self):
        from rules.indirect_fire import tables
        t = tables()
        self.assertEqual(t.lookup(False, 10, 4), ("hit", 0))
        self.assertEqual(t.lookup(False, 10, 2), ("scatter", 1))
        self.assertEqual(t.lookup(False, 70, 2), ("scatter", 4))
        self.assertEqual(t.lookup(True, 10, 5), ("hit", 0))
        self.assertEqual(t.lookup(True, 70, 2), ("scatter", 3))

    def test_hit_rate_matches_table(self):
        import random
        from rules.indirect_fire import tables
        t = tables()
        rng = random.Random(99)
        n = 120000
        hits = 0
        rolls = 0
        for roll in range(2, 13):
            ways = roll - 1 if roll <= 7 else 13 - roll
            rolls += ways
        # Exact hit probability at distance 10 (0-20 band): rolls 4, 7, 10.
        expected = (3 + 6 + 3) / 36.0
        for _ in range(n):
            roll = rng.randint(1, 6) + rng.randint(1, 6)
            if t.lookup(False, 10, roll)[0] == "hit":
                hits += 1
        self.assertLess(abs(hits / float(n) - expected), 0.01)

    def test_scatter_tables_match_raw_sheet(self):
        import json
        import re
        from rules.indirect_fire import tables
        doc = json.load(open(os.path.join(SITUATION3D, "rules", "data", "crt.json")))
        parsed = tables()
        bands = ((0, 20), (21, 40), (41, 60), (61, 100000))
        for sheet_name, table in (("无校射火力散布裁决", parsed.unobserved),
                                  ("格内校射火力散布裁决", parsed.adjusted)):
            sheet = doc["sheets"][sheet_name]
            for idx, (lo, hi) in enumerate(bands, start=1):
                expected = {}
                for row in sheet[2:]:
                    try:
                        roll = int(float(row[0]))
                    except (TypeError, ValueError):
                        continue
                    if not 2 <= roll <= 12:
                        continue
                    text = str(row[idx] or "").strip()
                    if "命中" in text:
                        expected[roll] = ("hit", 0)
                    elif text:
                        m = re.search(r"散布(\d+)格", text)
                        expected[roll] = ("scatter", int(m.group(1)) if m else 1)
                for roll, value in expected.items():
                    self.assertEqual(parsed.lookup(table is parsed.adjusted, lo, roll),
                                     value,
                                     "%s band %s roll %s" % (sheet_name, lo, roll))

    def _world(self):
        from sim.scenario import build_world
        basic = {"map_id": 1, "ele_grade": 10,
                 "grid": {"center_lon": 119.9, "center_lat": 23.1,
                          "cols": 24, "rows": 24, "width_m": 200.0},
                 "map_data": [[{"pos": r * 100 + c, "elev": 10, "cond": 0,
                                "roads": [0] * 6, "rivers": [0] * 6}
                               for c in range(24)] for r in range(24)]}
        scenario = {"scenario_id": 1,
                    "operators": [
                        {"obj_id": 1, "color": 0, "type": 2, "name": "炮兵",
                         "sub_type": 0, "blood": 2, "max_blood": 2, "armor": 0,
                         "cur_hex": 505, "carry_weapon_ids": [72], "value": 10},
                        {"obj_id": 2, "color": 1, "type": 2, "name": "重型坦克",
                         "sub_type": 0, "blood": 3, "max_blood": 3, "armor": 3,
                         "cur_hex": 1010, "carry_weapon_ids": [36], "value": 10}],
                    "cities": [], "engine": {"indirect_fire_model": "table"},
                    "time": {"cur_step": 0, "tick": 10.0,
                             "max_time": 999}}
        return build_world(scenario, basic, seed=11)

    def test_engage_indirect_event(self):
        from sim.context import WorldRuleContext
        world = self._world()
        ctx = WorldRuleContext(world)
        world.seen[1] = {2}
        ammo_before = world.entities[1].ammo["72"]
        world._engage(ctx)
        events = [e for e in world.events if e.get("type") == "indirect_fire"]
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["actors"][0], 1)
        self.assertEqual(world.entities[1].ammo["72"], ammo_before - 1)
        self.assertIn(event["outcome"], ("hit", "scatter"))
        aim = tuple(event["aim_cell"])
        impact = tuple(event["impact_cell"])
        if event["outcome"] == "scatter":
            self.assertLessEqual(ctx.rule_distance_cells(aim, impact),
                                 event["scatter_cells"])
        else:
            self.assertEqual(aim, impact)

    def test_fire_indirect_action(self):
        from geoio.actions import apply_action
        world = self._world()
        target = world.entities[2].pose
        self.assertTrue(apply_action(world, {"type": "fire_indirect", "entity": 1,
                                             "target": {"lon": target.lon,
                                                        "lat": target.lat}}))
        world.step()
        events = [e for e in world.events if e.get("type") == "indirect_fire"]
        self.assertEqual(len(events), 1)

    def test_indirect_fuzz_no_illegal_state(self):
        import random
        from rules.indirect_fire import resolve_indirect_fire
        from sim.context import WorldRuleContext
        world = self._world()
        ctx = WorldRuleContext(world)
        rng = random.Random(5)
        shots = 0
        attacker_cell = ctx.rule_cell(ctx.view(world.entities[1]))
        for _ in range(200):
            world.cooldowns.pop((1, "72"), None)
            world.entities[1].ammo["72"] = 10
            while True:
                aim = (rng.randrange(0, 24), rng.randrange(0, 24))
                if ctx.rule_distance_cells(attacker_cell, aim) <= 18:
                    break
            event = resolve_indirect_fire(ctx.view(world.entities[1]), aim, ctx,
                                          observed=bool(rng.getrandbits(1)))
            self.assertIsNotNone(event)
            self.assertIn("impact_cell", event)
            self.assertLessEqual(event["scatter_cells"], 4)
            impact = tuple(event["impact_cell"])
            self.assertLessEqual(ctx.rule_distance_cells(aim, impact),
                                 max(1, event["scatter_cells"]))
            shots += 1
        self.assertEqual(shots, 200)


if __name__ == "__main__":
    unittest.main()
