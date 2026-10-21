"""Core engine tests: environment, mobility, motion, rules, world, IO.

Run from ``situation3d/``:
    python3 -m unittest tests/test_geo_engine_core.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from compat.legacy import LegacyHexGrid  # noqa: E402
from env.grid import MapGridEnvironment  # noqa: E402
from env.service import LandCover  # noqa: E402
from geo.geodesy import haversine_m  # noqa: E402
from geo.rule_grid import RuleGrid  # noqa: E402
from mobility.passability import passable, speed_factor  # noqa: E402
from mobility.planner import GridPlanner  # noqa: E402
from mobility.profile import PROFILES  # noqa: E402
from rules.crt import CRT, load_crt  # noqa: E402
from rules.elevation import elevation_modifier, load_elevation  # noqa: E402
from sim.entity import Entity, Pose  # noqa: E402
from sim.motion import integrate  # noqa: E402
from sim.scenario import build_world  # noqa: E402
from env.pack import EnvPack, EnvPackError  # noqa: E402
from geoio.actions import apply_action, legacy_move_path_to_orders  # noqa: E402
from geoio.czml import world_to_czml  # noqa: E402
from geoio.observation import observation  # noqa: E402

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
             weapons=(36,), sub_type=0, value=10):
    return {"obj_id": oid, "color": color, "type": type_, "name": name,
            "sub_type": sub_type, "blood": blood, "max_blood": blood,
            "armor": armor, "cur_hex": hexpos, "carry_weapon_ids": list(weapons),
            "value": value, "move_state": 0}


def scenario(ops, cities=()):
    return {"scenario_id": 1, "operators": list(ops), "cities": list(cities),
            "time": {"cur_step": 0, "tick": 1.0, "max_time": 3600}}


def make_env(basic=None, water_depth=20.0):
    basic = basic or make_basic()
    grid = LegacyHexGrid.from_grid_meta(basic["grid"])
    return MapGridEnvironment(basic, grid, default_water_depth_m=water_depth), grid


class TestEnvironment(unittest.TestCase):
    def test_elevation_landcover_water(self):
        env, grid = make_env(make_basic(water_cols=(5,), forest_cols=(6,),
                                        urban_cols=(7,), roads_row=3))
        lon, lat = grid.cell_center_lonlat(10, 5)
        self.assertEqual(env.landcover(lon, lat), LandCover.WATER)
        self.assertAlmostEqual(env.water_depth(lon, lat), 20.0)
        self.assertEqual(env.landcover(*grid.cell_center_lonlat(10, 6)), LandCover.FOREST)
        self.assertEqual(env.landcover(*grid.cell_center_lonlat(10, 7)), LandCover.URBAN)
        self.assertTrue(env.on_road(*grid.cell_center_lonlat(3, 10)))
        self.assertFalse(env.on_road(*grid.cell_center_lonlat(10, 10)))

    def test_line_of_sight_flat_and_cover(self):
        env, grid = make_env(make_basic())
        a = grid.cell_center_lonlat(10, 2)
        b = grid.cell_center_lonlat(10, 9)
        self.assertTrue(env.line_of_sight(a, env.elevation(*a) + 2, b, env.elevation(*b) + 2))
        env2, grid2 = make_env(make_basic(urban_cols=(5,)))
        a2 = grid2.cell_center_lonlat(10, 2)
        b2 = grid2.cell_center_lonlat(10, 9)
        self.assertFalse(env2.line_of_sight(a2, env2.elevation(*a2) + 0.5,
                                            b2, env2.elevation(*b2) + 0.5))

    def test_shallow_water_wading(self):
        env, grid = make_env(make_basic(water_cols=(5,)), water_depth=1.0)
        lon, lat = grid.cell_center_lonlat(10, 5)
        self.assertTrue(passable(PROFILES["tracked"], lon, lat, env))
        self.assertAlmostEqual(speed_factor(PROFILES["tracked"], lon, lat, env), 0.35)
        deep, grid2 = make_env(make_basic(water_cols=(5,)), water_depth=2.0)
        lon2, lat2 = grid2.cell_center_lonlat(10, 5)
        self.assertFalse(passable(PROFILES["tracked"], lon2, lat2, deep))
        self.assertTrue(passable(PROFILES["amphibious"], lon2, lat2, deep))

    def test_slope(self):
        env, grid = make_env(make_basic(slope=5.0))
        lon, lat = grid.cell_center_lonlat(10, 10)
        self.assertGreater(env.slope_deg(lon, lat, bearing_deg=0.0), 0.0)


class TestMobility(unittest.TestCase):
    def test_water_blocks_tracked_allows_infantry_wading(self):
        env, grid = make_env(make_basic(water_cols=(5,)))
        lon, lat = grid.cell_center_lonlat(10, 5)
        self.assertFalse(passable(PROFILES["tracked"], lon, lat, env))
        self.assertFalse(passable(PROFILES["infantry"], lon, lat, env))  # 5 m > 1.2 m wading
        self.assertTrue(passable(PROFILES["ship"], lon, lat, env))
        self.assertFalse(passable(PROFILES["ship"], *grid.cell_center_lonlat(10, 10), env))

    def test_slope_blocks_tracked(self):
        env, grid = make_env(make_basic(slope=200.0))
        lon, lat = grid.cell_center_lonlat(20, 10)
        self.assertFalse(passable(PROFILES["tracked"], lon, lat, env))
        self.assertTrue(passable(PROFILES["infantry"], lon, lat, env))

    def test_air_clearance(self):
        env, grid = make_env(make_basic(slope=60.0))
        lon, lat = grid.cell_center_lonlat(20, 10)
        terrain = env.elevation(lon, lat)
        self.assertFalse(passable(PROFILES["uav"], lon, lat, env, altitude_m=terrain + 5.0))
        self.assertTrue(passable(PROFILES["uav"], lon, lat, env, altitude_m=terrain + 40.0))

    def test_planner_routes_around_water(self):
        env, grid = make_env(make_basic(water_cols=(10,)))
        planner = GridPlanner(env, RuleGrid.from_grid_meta(GRID))
        start = grid.cell_center_lonlat(10, 5)
        goal = grid.cell_center_lonlat(10, 15)
        route = planner.plan(PROFILES["tracked"], start, goal)
        # A full water column splits the map; no route exists.
        self.assertIsNone(route)
        goal2 = grid.cell_center_lonlat(10, 8)
        route2 = planner.plan(PROFILES["tracked"], start, goal2)
        self.assertIsNotNone(route2)
        self.assertGreater(route2.total_m, 0.0)


class TestMotion(unittest.TestCase):
    def test_route_integration_moves_entity(self):
        env, grid = make_env()
        start = grid.cell_center_lonlat(10, 5)
        goal = grid.cell_center_lonlat(10, 15)
        e = Entity(1, "A", 0, "land", "tracked", Pose(start[0], start[1]))
        e.profile = PROFILES["tracked"]
        route = GridPlanner(env, RuleGrid.from_grid_meta(GRID)).plan(e.profile, start, goal)
        e.route = route
        for _ in range(320):
            integrate(e, 1.0, env)
        self.assertLess(haversine_m((e.pose.lon, e.pose.lat), goal), 50.0)
        self.assertLessEqual(e.velocity_ms, PROFILES["tracked"].max_speed_ms + 1e-6)

    def test_blocked_motion_stops(self):
        env, grid = make_env(make_basic(water_cols=(6,)))
        start = grid.cell_center_lonlat(10, 5)
        goal = grid.cell_center_lonlat(10, 7)
        e = Entity(1, "A", 0, "land", "tracked", Pose(start[0], start[1]))
        e.profile = PROFILES["tracked"]
        from geo.rule_grid import HexGrid  # noqa: F401
        from sim.entity import Route
        e.route = Route(points=[Pose(start[0], start[1]),
                                Pose(*grid.cell_center_lonlat(10, 7))],
                        speed_limits=[8.0], modes=["offroad"])
        for _ in range(60):
            integrate(e, 1.0, env)
        self.assertEqual(e.velocity_ms, 0.0)
        self.assertIsNone(e.route)


class TestCRT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crt = CRT(load_crt())

    def test_vehicle_raw_matches_probe(self):
        observed = [3, 3, 1, 1, 1, 3, 2, 2, 2, 3, 3]
        for i, r1 in enumerate(range(2, 13)):
            self.assertEqual(self.crt.vehicle_raw(3, 10, r1), observed[i])

    def test_vehicle_correction_armor4(self):
        # Observed rect values at armour 4 (composite) in the probe.
        self.assertEqual(self.crt.vehicle_correction(5, 4), -1)
        self.assertEqual(self.crt.vehicle_correction(7, 4), 0)
        self.assertEqual(self.crt.vehicle_correction(9, 4), 0)
        self.assertEqual(self.crt.vehicle_correction(10, 4), 0)
        self.assertEqual(self.crt.vehicle_correction(11, 4), 1)
        self.assertEqual(self.crt.vehicle_correction(12, 4), 2)

    def test_correction_accepts_combined_modifier(self):
        self.assertEqual(self.crt.vehicle_correction(7, 4, 0), 0)
        self.assertEqual(self.crt.vehicle_correction(7, 4, -2), -1)
        self.assertEqual(self.crt.personnel_correction(7, 0), 0)
        self.assertEqual(self.crt.personnel_correction(7, -7), -1)

    def test_personnel_suppression(self):
        self.assertEqual(self.crt.personnel_result(1, 2), ("suppressed", 0))


class TestEnvPackPipeline(unittest.TestCase):
    class FakeOsm(object):
        def features(self, bbox):
            c = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            ring = [[c[0] - 0.002, c[1] - 0.002], [c[0] + 0.002, c[1] - 0.002],
                    [c[0] + 0.002, c[1] + 0.002], [c[0] - 0.002, c[1] + 0.002],
                    [c[0] - 0.002, c[1] - 0.002]]
            road = [[bbox[0], c[1]], [bbox[2], c[1]]]
            return {
                "roads": [{"coords": road, "class": "primary"}],
                "water": [{"ring": ring}],
                "buildings": [{"ring": [[c[0] + 0.003, c[1] + 0.003],
                                         [c[0] + 0.004, c[1] + 0.003],
                                         [c[0] + 0.004, c[1] + 0.004],
                                         [c[0] + 0.003, c[1] + 0.004],
                                         [c[0] + 0.003, c[1] + 0.003]],
                                "height_m": 12.0}],
                "landuse": [{"ring": [[c[0] - 0.008, c[1] - 0.008],
                                       [c[0] - 0.006, c[1] - 0.008],
                                       [c[0] - 0.006, c[1] - 0.006],
                                       [c[0] - 0.008, c[1] - 0.006],
                                       [c[0] - 0.008, c[1] - 0.008]],
                             "class": "forest"}],
            }

    class FakeDem(object):
        def sample_grid(self, bbox, step_deg):
            return {"lon0": bbox[0], "lat0": bbox[1], "lon1": bbox[2],
                    "lat1": bbox[3], "cols": 2, "rows": 2,
                    "values": [[100.0, 100.0], [100.0, 100.0]]}

    def test_offline_pack_with_injected_sources(self):
        import tempfile
        from geodata.pipeline import build_envpack
        from env.pack_env import EnvPackEnvironment
        basic = make_basic()
        grid = LegacyHexGrid.from_grid_meta(basic["grid"])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "pack")
            manifest = build_envpack(out, basic, grid, theatre="test",
                                     mode="offline", cache_dir=os.path.join(tmp, "cache"),
                                     osm_source=self.FakeOsm(), dem_source=self.FakeDem())
            self.assertEqual(manifest["layers"]["terrain"]["source"], "offline-cache")
            env = EnvPackEnvironment(out)
            c = manifest["bbox"]
            mid = ((c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0)
            self.assertAlmostEqual(env.elevation(*mid), 100.0, delta=0.1)
            self.assertGreater(env.water_depth(*mid), 0.0)
            self.assertGreaterEqual(env.building_at(mid[0] + 0.0035,
                                                    mid[1] + 0.0035).height_m, 12.0)
            self.assertEqual(env.landcover(mid[0] - 0.007, mid[1] - 0.007), "forest")
            self.assertTrue(env.on_road(c[0] + 0.001, mid[1], tolerance_m=50.0))

    def test_offline_cache_miss_raises(self):
        import tempfile
        from geodata.cache import CacheStore, OfflineCacheMiss
        from geodata.osm import OsmSource
        with tempfile.TemporaryDirectory() as tmp:
            source = OsmSource(CacheStore(tmp, mode="offline"))
            with self.assertRaises(OfflineCacheMiss):
                source.features((119.8, 23.0, 119.9, 23.1))

    def test_normalize_osm_shapes(self):
        from geodata.osm import normalize
        raw = {"elements": [
            {"tags": {"highway": "primary"},
             "geometry": [{"lon": 120.0, "lat": 23.0}, {"lon": 120.01, "lat": 23.0}]},
            {"tags": {"building": "yes", "height": "12"},
             "geometry": [{"lon": 120.0, "lat": 23.0}, {"lon": 120.001, "lat": 23.0},
                          {"lon": 120.001, "lat": 23.001}, {"lon": 120.0, "lat": 23.0}]},
            {"tags": {"natural": "water"},
             "geometry": [{"lon": 120.0, "lat": 23.0}, {"lon": 120.002, "lat": 23.0},
                          {"lon": 120.002, "lat": 23.002}, {"lon": 120.0, "lat": 23.0}]},
        ]}
        features = normalize(raw)
        self.assertEqual(len(features["roads"]), 1)
        self.assertEqual(len(features["buildings"]), 1)
        self.assertEqual(features["buildings"][0]["height_m"], 12.0)
        self.assertEqual(len(features["water"]), 1)


class TestWorldAndRules(unittest.TestCase):
    def test_direct_fire_event(self):
        basic = make_basic()
        world = build_world(scenario([
            operator(1, 0, 1010, blood=3),
            operator(2, 1, 1011, blood=5),
        ]), basic, seed=20261021)
        events = []
        for _ in range(3):
            events.extend(world.step())
        fires = [e for e in events if e["type"] == "direct_fire"]
        self.assertTrue(fires)
        ev = fires[0]
        self.assertEqual(ev["distance"], 1)
        self.assertEqual(ev["att_level"], 10)
        self.assertIn(ev["random1"], range(2, 13))
        expected_raw = CRT(load_crt()).vehicle_raw(3, 10, ev["random1"])
        self.assertEqual(ev["ori_damage"], expected_raw)
        self.assertEqual(ev["damage"], max(-1, expected_raw + ev["rect_damage"]))

    def test_stacking_limit(self):
        basic = make_basic()
        ops = [operator(i, 0, 1010 + i, blood=3) for i in range(1, 7)]
        world = build_world(scenario(ops), basic, seed=1)
        goal = world.env.grid.cell_center_lonlat(10, 10)
        for uid in range(1, 7):
            world.order(uid, {"type": "move_to", "lon": goal[0], "lat": goal[1]})
        for _ in range(80):
            world.step()
        cells = {}
        for e in world.entities.values():
            if e.alive:
                cells.setdefault(world.rule_cell(e), 0)
                cells[world.rule_cell(e)] += 1
        self.assertLessEqual(max(cells.values()), 4)

    def test_objective_capture(self):
        basic = make_basic()
        lonlat = LegacyHexGrid.from_grid_meta(GRID).cell_center_lonlat(10, 10)
        world = build_world(scenario([operator(1, 0, 1010)],
                                     [{"coord": 1010, "value": 50, "flag": -1, "name": "OBJ"}]),
                            basic, seed=1)
        for _ in range(2):
            world.step()
        owners = [o["owner"] for o in world.objectives.values()]
        self.assertEqual(owners, [0])

    def test_observation_filters_enemies(self):
        basic = make_basic()
        world = build_world(scenario([operator(1, 0, 1010),
                                      operator(2, 1, 1011, blood=5)]), basic, seed=1)
        world.step()
        red = observation(world, seat=0)
        ids = {e["id"] for e in red["entities"]}
        self.assertIn(1, ids)
        self.assertIn(2, ids)  # adjacent, observable
        # A far enemy is not visible.
        far = operator(3, 1, 2323, blood=5)
        far["move_state"] = 4  # concealed: observation range halves
        world2 = build_world(scenario([operator(1, 0, 1010), far]), basic, seed=1)
        world2.step()
        red2 = observation(world2, seat=0)
        self.assertNotIn(3, {e["id"] for e in red2["entities"]})

    def test_campaign_grid_los_uses_env_grid(self):
        basic = make_basic(urban_cols=(5,))
        basic["grid"]["width_m"] = 1500.0
        grid = LegacyHexGrid.from_grid_meta(basic["grid"])
        env = MapGridEnvironment(basic, grid)
        from sim.context import WorldRuleContext
        from sim.world import World
        from geo.rule_grid import RuleGrid
        world = World(env, RuleGrid(grid.center_lon, grid.center_lat, grid.cols, grid.cols),
                      plan_grid=grid)
        a = Entity(1, "A", 0, "land", "tracked",
                   Pose(*grid.cell_center_lonlat(10, 2)))
        b = Entity(2, "B", 1, "land", "tracked",
                   Pose(*grid.cell_center_lonlat(10, 9)))
        world.add_entity(a)
        world.add_entity(b)
        ctx = WorldRuleContext(world)
        ca = grid.cell_of(a.pose.lon, a.pose.lat)
        cb = grid.cell_of(b.pose.lon, b.pose.lat)
        self.assertEqual(ctx.line_of_sight(ctx.view(a), ctx.view(b)),
                         env.cell_line_of_sight(ca, cb))
        self.assertFalse(env.cell_line_of_sight(ca, cb))

    def test_forest_cover_applied_only_in_correction(self):
        basic = make_basic(forest_cols=(11,))
        world = build_world(scenario([operator(1, 0, 1010),
                                      operator(2, 1, 1011, blood=5)]), basic, seed=7)
        for _ in range(3):
            world.step()
        fires = [e for e in world.events if e["type"] == "direct_fire"]
        self.assertTrue(fires)
        self.assertEqual(fires[0]["att_level"], 10)  # no attack-level cover double count

    def test_deterministic_events(self):
        basic = make_basic()
        sc = scenario([operator(1, 0, 1010), operator(2, 1, 1011, blood=5)])
        a = build_world(sc, basic, seed=99)
        b = build_world(sc, basic, seed=99)
        for _ in range(5):
            ea = a.step()
            eb = b.step()
            self.assertEqual([(e["type"], e.get("actors"), e.get("damage")) for e in ea],
                             [(e["type"], e.get("actors"), e.get("damage")) for e in eb])

    def test_amphibious_unit_fights_as_ground(self):
        op = operator(1, 0, 1010)
        op["name"] = "两栖战车"
        world = build_world(scenario([op, operator(2, 1, 1011, blood=5)]), make_basic(), seed=5)
        self.assertEqual(world.entities[1].domain, "amphibious")
        for _ in range(2):
            world.step()
        self.assertTrue(any(e["type"] == "direct_fire" for e in world.events))

    def test_carrier_loss_kills_passengers(self):
        basic = make_basic(water_cols=(10,))
        ops = [operator(1, 0, 1009, name="登陆舰"),
               operator(2, 0, 1010, name="步兵小队", type_=1, sub_type=2, armor=0)]
        # Put the landing ship and passenger on the same water cell.
        world = build_world(scenario(ops), basic, seed=1)
        self.assertTrue(world.entities[2].on_board)
        world.apply_damage(1, 99)
        self.assertFalse(world.entities[1].alive)
        self.assertFalse(world.entities[2].alive)

    def test_seen_is_refreshed(self):
        basic = make_basic()
        world = build_world(scenario([operator(1, 0, 1010),
                                      operator(2, 1, 1011, blood=5)]), basic, seed=1)
        world.step()
        self.assertIn(2, world.seen[1])
        far = world.env.grid.cell_center_lonlat(23, 23)
        world.entities[2].pose = Pose(far[0], far[1], world.env.elevation(*far))
        world.entities[2].concealed = True
        world.step()
        self.assertNotIn(2, world.seen.get(1, set()))

    def test_envpack_manifest(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {
                "schema": "msgeo.envpack.v1", "theatre": "t",
                "bbox": [119.0, 22.0, 120.0, 23.0], "tile_size_m": 5000,
                "layers": {"terrain": {}, "landcover": {}},
                "fallback_basic_json": "base_map.json",
            }
            with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh)
            with open(os.path.join(tmp, "base_map.json"), "w", encoding="utf-8") as fh:
                json.dump(make_basic(), fh)
            pack = EnvPack.load(tmp)
            self.assertEqual(pack.bbox, (119.0, 22.0, 120.0, 23.0))
            self.assertEqual(len(pack.load_basic_map()["map_data"]), 24)
            manifest["schema"] = "bad"
            with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh)
            with self.assertRaises(EnvPackError):
                EnvPack.load(tmp)

    def test_envpack_builder(self):
        import importlib.util
        import tempfile
        path = os.path.join(SITUATION3D, "scripts", "build_envpack.py")
        spec = importlib.util.spec_from_file_location("build_envpack", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            basic_path = os.path.join(tmp, "basic.json")
            with open(basic_path, "w", encoding="utf-8") as fh:
                json.dump(make_basic(), fh)
            out = os.path.join(tmp, "pack")
            self.assertEqual(mod.main(["--basic", basic_path, "--out", out,
                                       "--theatre", "test",
                                       "--no-dem", "--no-osm"]), 0)
            pack = EnvPack.load(out)
            self.assertEqual(pack.manifest["theatre"], "test")
            self.assertEqual(pack.manifest["layers"]["terrain"]["format"], "grid-json")
            self.assertTrue(os.path.isfile(os.path.join(out, "terrain", "grid.json")))

    def test_legacy_move_and_fire_actions(self):
        basic = make_basic()
        world = build_world(scenario([operator(1, 0, 1010),
                                      operator(2, 1, 1011, blood=5)]), basic, seed=3)
        grid = world.env.grid
        self.assertTrue(legacy_move_path_to_orders(world, 1, [1011], grid))
        world.step()
        self.assertIsNotNone(world.entities[1].route)
        self.assertTrue(apply_action(world, {"type": "stop", "entity": 1}))
        world.step()
        self.assertIsNone(world.entities[1].route)
        self.assertTrue(apply_action(world, {"type": "fire", "entity": 1, "target": 2}))
        for _ in range(2):
            world.step()
        self.assertTrue(any(e["type"] == "direct_fire" for e in world.events))

    def test_czml_packets(self):
        basic = make_basic()
        world = build_world(scenario([operator(1, 0, 1010)]), basic, seed=1)
        doc = world_to_czml(world, "2026-10-21T00:00:00Z")
        self.assertEqual(doc["packets"][0]["id"], "document")
        self.assertIn("/", doc["packets"][0]["clock"]["interval"])
        self.assertTrue(doc["packets"][0]["clock"]["interval"].startswith("2026-10-21T00:00:00Z"))
        ent = [p for p in doc["packets"] if p["id"] == "entity-1"][0]
        self.assertEqual(ent["position"]["epoch"], "2026-10-21T00:00:00Z")
        self.assertTrue(any(p["id"] == "entity-1" for p in doc["packets"]))
        # A planned route adds a polyline packet field.
        e = world.entities[1]
        route = world.planner.plan(e.profile, (e.pose.lon, e.pose.lat),
                                   world.env.grid.cell_center_lonlat(10, 15))
        e.route = route
        packet = [p for p in world_to_czml(world)["packets"] if p["id"] == "entity-1"][0]
        self.assertIn("polyline", packet)

    def test_fuzz_invariants(self):
        rng = random.Random(7)
        basic = make_basic(water_cols=(8,), forest_cols=(12,))
        ops = [operator(i, i % 2, 500 + i * 3) for i in range(1, 7)]
        world = build_world(scenario(ops), basic, seed=11)
        for _ in range(300):
            for uid in list(world.entities):
                if rng.random() < 0.05:
                    cell = (rng.randrange(4, 20), rng.randrange(4, 20))
                    lon, lat = world.rule_grid.cell_center_lonlat(*cell)
                    world.order(uid, {"type": "move_to", "lon": lon, "lat": lat})
            world.step()
            for e in world.entities.values():
                self.assertTrue(math.isfinite(e.pose.lon) and math.isfinite(e.pose.lat))
                self.assertLessEqual(e.velocity_ms, e.profile.max_speed_ms + 1e-6)
                depth = world.env.water_depth(e.pose.lon, e.pose.lat)
                if depth is not None and e.domain == "land":
                    self.assertLessEqual(depth, e.profile.wading_depth_m + 1e-9)


if __name__ == "__main__":
    unittest.main()
