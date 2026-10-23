"""P2 tests: missile flight/terminal defence, air/sea dynamics, weather.

Run from ``situation3d/``:
    python3 -m unittest tests/test_p2_environment.py
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

from env.service import Weather  # noqa: E402
from geo.geodesy import haversine_m  # noqa: E402
from mobility.profile import PROFILES  # noqa: E402
from rules import missile  # noqa: E402
from rules.engagement import SEA_AIR_PROFILES  # noqa: E402
from sim.entity import Entity, Pose, Route  # noqa: E402
from sim.motion import integrate  # noqa: E402
from sim.scenario import build_world  # noqa: E402


def make_basic(rows=20, cols=20, cell_m=10000.0, center=(119.9, 23.1),
               cond=0):
    cells = [[{"pos": r * 100 + c, "elev": 100, "cond": cond,
               "roads": [0] * 6, "rivers": [0] * 6}
              for c in range(cols)] for r in range(rows)]
    return {"map_id": 1, "ele_grade": 10,
            "grid": {"center_lon": center[0], "center_lat": center[1],
                     "cols": cols, "rows": rows, "width_m": cell_m},
            "map_data": cells}


def ship(uid, name, cell, color=0, blood=6):
    return {"obj_id": uid, "name": name, "color": color, "type": 2,
            "sub_type": 0, "cur_hex": cell, "blood": blood,
            "max_blood": blood, "armor": 4, "value": 10, "domain": "sea",
            "observe_distance": {"1": 1, "2": 25, "3": 1},
            "move_state": 0, "stop": 1, "readiness": 0.85, "fuel": 1.0,
            "emitting": True}


def scenario(ops, engine=None, extra=None):
    doc = {"scenario_id": 1, "cities": [], "operators": ops,
           "time": {"cur_step": 0, "tick": 1.0, "max_time": 9999}}
    if engine:
        doc["engine"] = engine
    if extra:
        doc.update(extra)
    return doc


class MissileTest(unittest.TestCase):
    def test_tof_monotonic(self):
        near = missile.tof_s("asm", 30.0)
        far = missile.tof_s("asm", 160.0)
        self.assertGreater(far, near)
        self.assertGreaterEqual(near, 5.0)

    def test_intercept_probability(self):
        target = Entity(2, "DDG", 1, "surface", "ship", Pose(120.0, 23.0))
        target.sea_air_weapons = [SEA_AIR_PROFILES["sam_long"]]
        target.ammo = {"sam_long": 8}

        class Ctx(object):
            def weapon_ready(self, uid, key):
                return True

        p = missile.intercept_probability(target.to_rule_view(), "asm",
                                          50.0, Ctx())
        self.assertGreater(p, 0.0)
        bare = Entity(3, "Boat", 1, "surface", "ship", Pose(120.0, 23.0))
        self.assertEqual(missile.intercept_probability(
            bare.to_rule_view(), "asm", 50.0, Ctx()), 0.0)

    def _world(self, missile_model="flight"):
        ops = [ship(1, "驱逐舰", 1002, color=0),
               ship(2, "护卫舰", 1018, color=1, blood=4)]
        return build_world(scenario(ops, engine={"missile_model": missile_model,
                                                 "detection_model": "physical"}),
                           make_basic(), seed=5)

    def test_flight_schedules_impact(self):
        world = self._world()
        ctx = __import__("sim.context", fromlist=["WorldRuleContext"]).WorldRuleContext(world)
        world.seen[1] = {2}
        world._engage(ctx)
        launches = [e for e in world.events if e.get("type") == "sea_air_fire"]
        self.assertTrue(launches)
        launch = launches[0]
        self.assertEqual(launch["result"], "launched")
        self.assertGreater(launch["tof_s"], 0.0)
        self.assertIsNotNone(launch["impact_at_s"])
        self.assertTrue(world._pending_impacts)
        world.time_s = float(launch["impact_at_s"])
        world._resolve_pending_impacts()
        impacts = [e for e in world.events if e.get("type") == "missile_impact"]
        self.assertEqual(len(impacts), 1)
        self.assertIn(impacts[0]["result"], ("hit", "miss", "intercepted"))

    def test_instant_model_still_available(self):
        world = self._world(missile_model="instant")
        ctx = __import__("sim.context", fromlist=["WorldRuleContext"]).WorldRuleContext(world)
        world.seen[1] = {2}
        world._engage(ctx)
        launches = [e for e in world.events if e.get("type") == "sea_air_fire"]
        self.assertTrue(launches)
        self.assertEqual(launches[0]["model"], "instant")
        self.assertFalse(world._pending_impacts)


class AirSeaDynamicsTest(unittest.TestCase):
    def test_aircraft_climb_rate(self):
        world = build_world(scenario([{
            "obj_id": 1, "name": "战斗机", "color": 0, "type": 3,
            "sub_type": 0, "cur_hex": 1010, "blood": 3, "max_blood": 3,
            "armor": 1, "value": 10, "domain": "air",
            "observe_distance": {"1": 1, "2": 25, "3": 1},
            "move_state": 0, "stop": 1, "readiness": 0.85, "fuel": 1.0,
            "emitting": True}]), make_basic(), seed=1)
        e = world.entities[1]
        start = e.pose
        e.pose = Pose(start.lon, start.lat, 1000.0)
        e.route = Route(points=[Pose(start.lon, start.lat, 1000.0),
                                Pose(start.lon, start.lat + 0.2, 2000.0)],
                        speed_limits=[200.0], modes=["air"])
        e.moving = True
        climb = PROFILES["aircraft"].climb_rate_ms
        last = e.pose.alt_m
        for _ in range(10):
            integrate(e, 10.0, world.env)
            self.assertLessEqual(e.pose.alt_m - last, climb * 10.0 + 1e-6)
            self.assertGreaterEqual(e.pose.alt_m, last)
            last = e.pose.alt_m
        self.assertGreater(last, 1100.0)

    def test_aircraft_level_has_no_climb_penalty(self):
        world = build_world(scenario([{
            "obj_id": 1, "name": "战斗机", "color": 0, "type": 3,
            "sub_type": 0, "cur_hex": 1010, "blood": 3, "max_blood": 3,
            "armor": 1, "value": 10, "domain": "air",
            "observe_distance": {"1": 1, "2": 25, "3": 1},
            "move_state": 0, "stop": 1, "readiness": 0.85, "fuel": 1.0,
            "emitting": True}]), make_basic(), seed=1)
        e = world.entities[1]
        e.pose = Pose(e.pose.lon, e.pose.lat, 2000.0)
        e.route = Route(points=[Pose(e.pose.lon, e.pose.lat, 2000.0),
                                Pose(e.pose.lon, e.pose.lat + 0.5, 2000.0)],
                        speed_limits=[100.0], modes=["air"])
        e.moving = True
        integrate(e, 1.0, world.env)
        self.assertAlmostEqual(e.vertical_ms, 0.0, delta=0.2)
        for _ in range(60):
            integrate(e, 1.0, world.env)
        self.assertGreater(e.velocity_ms, 90.0)

    def test_aircraft_descent_rate(self):
        world = build_world(scenario([{
            "obj_id": 1, "name": "战斗机", "color": 0, "type": 3,
            "sub_type": 0, "cur_hex": 1010, "blood": 3, "max_blood": 3,
            "armor": 1, "value": 10, "domain": "air",
            "observe_distance": {"1": 1, "2": 25, "3": 1},
            "move_state": 0, "stop": 1, "readiness": 0.85, "fuel": 1.0,
            "emitting": True}]), make_basic(), seed=1)
        e = world.entities[1]
        e.pose = Pose(e.pose.lon, e.pose.lat, 2000.0)
        e.route = Route(points=[Pose(e.pose.lon, e.pose.lat, 2000.0),
                                Pose(e.pose.lon, e.pose.lat + 0.2, 1000.0)],
                        speed_limits=[200.0], modes=["air"])
        e.moving = True
        descent = PROFILES["aircraft"].descent_rate_ms
        for _ in range(20):
            last = e.pose.alt_m
            integrate(e, 5.0, world.env)
            self.assertGreaterEqual(e.pose.alt_m, 1000.0 - 1e-6)
            self.assertLessEqual(last - e.pose.alt_m, descent * 5.0 + 1e-6)
        self.assertLess(e.pose.alt_m, 1900.0)

    def test_ciws_intercepts_long_range_asm(self):
        target = Entity(2, "DDG", 1, "surface", "ship", Pose(120.0, 23.0))
        target.sea_air_weapons = [SEA_AIR_PROFILES["ciws"]]
        target.ammo = {"ciws": 30}

        class Ctx(object):
            def weapon_ready(self, uid, key):
                return True

        p = missile.intercept_probability(target.to_rule_view(), "asm",
                                          160.0, Ctx())
        self.assertGreater(p, 0.0)

    def test_weather_through_step(self):
        sc = scenario([ship(1, "驱逐舰", 1010)], extra={
            "start_hour_utc": 12.0,
            "weather": [{"at_s": 0.0, "visibility_m": 3000.0,
                         "sea_state": 3}]})
        world = build_world(sc, make_basic(cond=4), seed=3)
        self.assertTrue(world.weather_enabled)
        world.step()
        self.assertAlmostEqual(world.effective_weather["visibility_m"],
                               3000.0)
        self.assertEqual(world.env.weather(0.0, 0.0).sea_state, 3)

    def test_no_weather_config_leaves_env_alone(self):
        world = build_world(scenario([ship(1, "驱逐舰", 1010)]),
                            make_basic(cond=4), seed=3)
        self.assertFalse(world.weather_enabled)

    def test_sea_state_slows_ship(self):
        def run(sea_state):
            world = build_world(scenario([ship(1, "驱逐舰", 1010)]),
                                make_basic(cond=4), seed=2)
            world.env.weather(0.0, 0.0).sea_state = sea_state
            e = world.entities[1]
            e.route = Route(points=[Pose(e.pose.lon, e.pose.lat),
                                    Pose(e.pose.lon, e.pose.lat + 0.5)],
                            speed_limits=[15.0], modes=["water"])
            for _ in range(10):
                integrate(e, 5.0, world.env)
            return e.velocity_ms
        calm = run(0)
        rough = run(8)
        self.assertLess(rough, calm)


class WeatherTest(unittest.TestCase):
    def test_day_night_and_timeline(self):
        sc = scenario([ship(1, "驱逐舰", 1010)], extra={
            "start_hour_utc": 0.0,
            "weather_base": {"visibility_m": 10000.0, "sea_state": 1},
            "weather": [{"at_s": 3600.0, "visibility_m": 2000.0,
                         "sea_state": 4}],
        })
        world = build_world(sc, make_basic(), seed=3)
        world.time_s = 0.0
        world._update_environment()
        self.assertLess(world.effective_weather["visibility_m"], 3000.0)
        self.assertEqual(world.effective_weather["sea_state"], 1)
        world.time_s = 3600.0
        world._update_environment()
        self.assertLess(world.effective_weather["visibility_m"], 500.0)
        self.assertEqual(world.effective_weather["sea_state"], 4)
        night_weather = world.env.weather(0.0, 0.0)
        self.assertLess(night_weather.visibility_m, 500.0)

    def test_daytime_visibility(self):
        sc = scenario([ship(1, "驱逐舰", 1010)],
                      extra={"start_hour_utc": 12.0})
        world = build_world(sc, make_basic(), seed=3)
        world._update_environment()
        self.assertAlmostEqual(world.effective_weather["visibility_m"],
                               10000.0)


if __name__ == "__main__":
    unittest.main()
