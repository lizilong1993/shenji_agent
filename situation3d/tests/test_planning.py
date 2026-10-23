"""P1.6 kinematic planning tests: fillets, one-way roads, intents.

Run from ``situation3d/``:
    python3 -m unittest tests/test_planning.py
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from env.service import LandCover, Weather  # noqa: E402
from geo.geodesy import haversine_m, initial_bearing_deg  # noqa: E402
from mobility.kinematics import wrap180  # noqa: E402
from mobility.nav_field import NavField  # noqa: E402
from mobility.profile import PROFILES  # noqa: E402
from sim.entity import Entity, Pose  # noqa: E402
from sim.intent import Intent  # noqa: E402
from sim.motion import integrate  # noqa: E402
from sim.scenario import build_world  # noqa: E402

BBOX = (120.0, 23.0, 120.02, 23.02)


class ClearEnv(object):
    """Flat 2 km box with optional water rectangle / one-way road."""

    def __init__(self, roads=None, water=None):
        self.roads = roads or ({}, [])
        self.water = water
        self._weather = Weather()

    def nav_bbox(self):
        return BBOX

    def elevation(self, lon, lat):
        return 0.0

    def slope_deg(self, lon, lat, bearing_deg=0.0):
        return 0.0

    def water_depth(self, lon, lat):
        if self.water:
            x0, y0, x1, y1 = self.water
            if x0 <= lon <= x1 and y0 <= lat <= y1:
                return 5.0
        return None

    def landcover(self, lon, lat):
        return (LandCover.WATER if self.water_depth(lon, lat)
                else LandCover.OPEN)

    def building_footprint_at(self, lon, lat):
        return None

    def weather(self, lon, lat):
        return self._weather

    def road_graph(self):
        return self.roads

    def on_road(self, lon, lat, tolerance_m=30.0):
        return False

    def road_class(self, lon, lat, tolerance_m=30.0):
        return 0

    def building_intersects(self, a, h0, b, h1):
        return False


def make_basic():
    cells = [[{"pos": r * 100 + c, "elev": 100, "cond": 0,
               "roads": [0] * 6, "rivers": [0] * 6}
              for c in range(24)] for r in range(24)]
    return {"map_id": 1, "ele_grade": 10,
            "grid": {"center_lon": 119.9, "center_lat": 23.1, "cols": 24,
                     "rows": 24, "width_m": 200.0},
            "map_data": cells}


def operator(uid, name, cell, color=0, domain="land"):
    return {"obj_id": uid, "name": name, "color": color, "type": 2,
            "sub_type": 0, "cur_hex": cell, "blood": 3, "max_blood": 3,
            "armor": 1, "value": 10, "domain": domain,
            "observe_distance": {"1": 10, "2": 25, "3": 1},
            "move_state": 0, "stop": 1, "readiness": 0.85, "fuel": 1.0,
            "emitting": True}


class FilletTest(unittest.TestCase):
    def _plan(self, prof, start, goal, res=30.0, water=None):
        nav = NavField(ClearEnv(water=water), res_m=res, use_cache=False)
        return nav, nav.plan(prof, start, goal)

    @staticmethod
    def _turns(route):
        turns = []
        pts = route.points
        for i in range(1, len(pts) - 1):
            h1 = initial_bearing_deg((pts[i - 1].lon, pts[i - 1].lat),
                                     (pts[i].lon, pts[i].lat))
            h2 = initial_bearing_deg((pts[i].lon, pts[i].lat),
                                     (pts[i + 1].lon, pts[i + 1].lat))
            turns.append(abs(wrap180(h2 - h1)))
        return turns

    def test_fillet_angle_geometry(self):
        wheeled = PROFILES["wheeled"]
        nav = NavField(ClearEnv(), res_m=30.0, use_cache=False)
        mask = nav.profile_mask(wheeled)[0]
        b = (120.01, 23.01)
        kx = 111320.0 * math.cos(math.radians(b[1]))
        ky = 110540.0

        def corner(deflection_deg):
            a = (b[0] - 200.0 / kx, b[1])
            br = math.radians(90.0 + deflection_deg)
            c = (b[0] + 200.0 * math.sin(br) / kx,
                 b[1] + 200.0 * math.cos(br) / ky)
            return [Pose(a[0], a[1]), Pose(b[0], b[1]), Pose(c[0], c[1])]

        def signed_turns(points):
            out = []
            for i in range(1, len(points) - 1):
                h1 = initial_bearing_deg(
                    (points[i - 1].lon, points[i - 1].lat),
                    (points[i].lon, points[i].lat))
                h2 = initial_bearing_deg(
                    (points[i].lon, points[i].lat),
                    (points[i + 1].lon, points[i + 1].lat))
                out.append(wrap180(h2 - h1))
            return out

        for deflection, sign in ((45.0, 1), (-45.0, -1), (90.0, 1),
                                 (135.0, 1)):
            out = nav._fillet(wheeled, corner(deflection), mask, None)
            self.assertGreater(len(out), 3)
            turns = signed_turns(out)
            self.assertGreater(len(turns), 1)
            self.assertTrue(all(sign * t >= -1e-6 for t in turns),
                            "turn direction flips: %s" % turns)
            total = sum(turns)
            self.assertAlmostEqual(abs(total), abs(deflection), delta=5.0)
            self.assertLess(max(abs(t) for t in turns), 30.0)

        # Near U-turn: leave the vertex for the motion layer.
        out = nav._fillet(wheeled, corner(170.0), mask, None)
        self.assertEqual(len(out), 3)

    def test_wheeled_corners_are_filleted(self):
        wheeled = PROFILES["wheeled"]
        nav, route = self._plan(wheeled, (120.002, 23.008),
                                (120.018, 23.008),
                                water=(120.008, 23.002, 120.012, 23.016))
        self.assertIsNotNone(route)
        turns = self._turns(route)
        self.assertTrue(turns)
        self.assertLess(max(turns), 40.0)
        # And the motion layer can execute the planned path without blocking.
        e = Entity(1, "W", 0, "land", "wheeled", Pose(120.002, 23.008))
        e.profile = wheeled
        e.route = route
        e.moving = True
        for _ in range(2000):
            if e.route is None:
                break
            self.assertNotEqual(integrate(e, 2.0, nav.env, nav=nav),
                                "blocked")
        self.assertIsNone(e.route)
        self.assertGreater(haversine_m((120.002, 23.008),
                                       (e.pose.lon, e.pose.lat)), 500.0)


class OneWayTest(unittest.TestCase):
    def _nav(self):
        nodes = {"A": (120.002, 23.010), "B": (120.018, 23.010)}
        edges = [{"a": "A", "b": "B", "class": "primary", "bridge": False,
                  "tunnel": False, "oneway": True, "maxweight": None,
                  "maxheight": None, "width": None}]
        return NavField(ClearEnv(roads=(nodes, edges)), res_m=30.0,
                        use_cache=False)

    def test_oneway_factor_direction(self):
        nav = self._nav()
        west = nav.cell_index(120.004, 23.010)
        east = nav.cell_index(120.016, 23.010)
        i_west = west[0] * nav.cols + west[1]
        i_east = east[0] * nav.cols + east[1]
        self.assertTrue(nav.oneway[i_west])
        tracked = PROFILES["tracked"]
        self.assertEqual(nav._oneway_factor(tracked, i_west, i_east), 1.0)
        self.assertEqual(nav._oneway_factor(tracked, i_east, i_west), 3.0)
        road_bound = dataclasses.replace(PROFILES["wheeled"], road_bound=True)
        self.assertEqual(nav._oneway_factor(road_bound, i_east, i_west), 0.0)
        self.assertEqual(nav._oneway_factor(road_bound, i_west, i_east), 1.0)


class IntentTest(unittest.TestCase):
    def _world(self):
        return build_world({"scenario_id": 1, "cities": [],
                            "operators": [operator(1, "Unit", 1010)],
                            "time": {"cur_step": 0, "tick": 1.0,
                                     "max_time": 999}},
                           make_basic(), seed=3)

    def test_assign_intent_resolves_cell_and_pace(self):
        world = self._world()
        e = world.entities[1]
        cell = (10, 15)
        ok = world.assign_intent(e, Intent(cell=cell, pace="dash"))
        self.assertTrue(ok)
        self.assertIsNotNone(e.route)
        self.assertAlmostEqual(e.pace_scale, 1.0)
        goal_cell = world.rule_grid.cell_of(*e.goal)
        self.assertEqual(goal_cell, cell)
        world.assign_intent(e, Intent(cell=cell, pace="cautious"))
        self.assertAlmostEqual(e.pace_scale, 0.65)

    def test_pace_scales_motion(self):
        world = self._world()
        e = world.entities[1]
        world.assign_intent(e, Intent(cell=(10, 15), pace="normal"))
        route_normal = e.route
        route_normal.speed_limits[:] = [4.0] * len(route_normal.speed_limits)
        e2 = Entity(2, "Fast", 0, "land", "tracked", Pose(e.pose.lon, e.pose.lat))
        e2.profile = e.profile
        e2.route = route_normal
        e2.pace_scale = 1.0
        for _ in range(8):
            integrate(e, 1.0, world.env)
            integrate(e2, 1.0, world.env)
        self.assertGreater(e2.velocity_ms, e.velocity_ms)
        self.assertLessEqual(e2.velocity_ms, 4.0 + 1e-6)
        self.assertLessEqual(e.velocity_ms, 4.0 + 1e-6)

    def test_intent_rejects_empty(self):
        world = self._world()
        self.assertFalse(world.assign_intent(world.entities[1], Intent()))


if __name__ == "__main__":
    unittest.main()
