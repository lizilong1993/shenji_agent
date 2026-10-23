"""Navigation field (P0.1): raster authority, fine A*, hybrid refinement.

Run from ``situation3d/``:
    python3 -m unittest tests/test_nav_field.py
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
from env.service import Building, FlatEnvironment, LandCover  # noqa: E402
from geo.geodesy import haversine_m  # noqa: E402
from mobility.nav_field import NavField  # noqa: E402
from mobility.planner import GridPlanner  # noqa: E402
from mobility.profile import PROFILES  # noqa: E402
from sim.entity import Entity, Pose  # noqa: E402
from sim.motion import integrate  # noqa: E402

BBOX = (120.0, 23.0, 120.02, 23.02)
WATER = (120.008, 23.002, 120.012, 23.016)
BUILDING = (120.014, 23.009, 120.016, 23.011, 12.0)


class FakeFineEnv(object):
    """Envpack-like env: flat terrain, one water rect, one building block."""

    def __init__(self, water=WATER, buildings=(BUILDING,), roads=None):
        self.water = water
        self.buildings = list(buildings)
        self.roads = roads or ({}, [])

    def nav_bbox(self):
        return BBOX

    def elevation(self, lon, lat):
        return 0.0

    def slope_deg(self, lon, lat, bearing_deg=0.0):
        return 0.0

    def water_depth(self, lon, lat):
        if self.water is None:
            return None
        x0, y0, x1, y1 = self.water
        return 5.0 if (x0 <= lon <= x1 and y0 <= lat <= y1) else None

    def landcover(self, lon, lat):
        return LandCover.WATER if self.water_depth(lon, lat) else LandCover.OPEN

    def building_footprint_at(self, lon, lat):
        for x0, y0, x1, y1, height in self.buildings:
            if x0 <= lon <= x1 and y0 <= lat <= y1:
                return Building(footprint=[], height_m=height)
        return None

    def road_graph(self):
        return self.roads

    def on_road(self, lon, lat, tolerance_m=30.0):
        return False

    def road_class(self, lon, lat, tolerance_m=30.0):
        return 0


class FlatEnvWithRoads(FlatEnvironment):
    def on_road(self, lon, lat, tolerance_m=30.0):
        return False

    def road_class(self, lon, lat, tolerance_m=30.0):
        return 0

    def nearest_road(self, lon, lat, max_m=3000.0):
        return None


def _in_rect(lon, lat, rect):
    return rect[0] <= lon <= rect[2] and rect[1] <= lat <= rect[3]


def _straight(a, b):
    return haversine_m(a, b)


class NavFieldTest(unittest.TestCase):
    def setUp(self):
        self.env = FakeFineEnv()
        self.nav = NavField(self.env, res_m=50.0, use_cache=False)
        self.prof = PROFILES["tracked"]

    def test_routes_around_water_and_building(self):
        start = (120.002, 23.008)
        goal = (120.018, 23.008)
        route = self.nav.plan(self.prof, start, goal)
        self.assertIsNotNone(route)
        self.assertGreater(route.total_m, _straight(start, goal) * 1.1)
        for p in route.points:
            self.assertTrue(self.nav.passable(self.prof, p.lon, p.lat),
                            "route vertex inside blocked cell")
            self.assertFalse(_in_rect(p.lon, p.lat, WATER))
            self.assertFalse(_in_rect(p.lon, p.lat, BUILDING[:4]))

    def test_nav_blocks_buildings_env_does_not(self):
        # The raster treats footprints as movement obstacles; the coarse
        # point predicate only uses landcover/slope/water.
        lon, lat = 120.015, 23.010
        self.assertIsNotNone(self.env.building_footprint_at(lon, lat))
        self.assertFalse(self.nav.passable(self.prof, lon, lat))

    def test_goal_inside_building_snaps_to_street(self):
        start = (120.002, 23.008)
        goal = (120.015, 23.010)
        route = self.nav.plan(self.prof, start, goal)
        self.assertIsNotNone(route)
        end = route.points[-1]
        self.assertLess(haversine_m((end.lon, end.lat), goal), 150.0)
        self.assertTrue(self.nav.passable(self.prof, end.lon, end.lat))

    def test_plan_is_deterministic(self):
        a = self.nav.plan(self.prof, (120.002, 23.008), (120.018, 23.008))
        b = self.nav.plan(self.prof, (120.002, 23.008), (120.018, 23.008))
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual([(p.lon, p.lat) for p in a.points],
                         [(p.lon, p.lat) for p in b.points])

    def test_motion_follows_nav_route_without_blocking(self):
        route = self.nav.plan(self.prof, (120.002, 23.008), (120.018, 23.008))
        self.assertIsNotNone(route)
        entity = Entity(1, "T", 0, "land", "tracked", Pose(120.002, 23.008))
        entity.profile = self.prof
        entity.route = route
        entity.moving = True
        steps = 0
        blocked = 0
        while entity.route is not None and steps < 2000:
            reason = integrate(entity, 5.0, self.env, nav=self.nav)
            if reason == "blocked":
                blocked += 1
                break
            steps += 1
        self.assertEqual(blocked, 0)
        self.assertIsNone(entity.route, "route should complete")
        moved = haversine_m((120.002, 23.008),
                            (entity.pose.lon, entity.pose.lat))
        self.assertGreater(moved, route.total_m * 0.7)

    def test_hybrid_refines_inside_pack(self):
        coarse_env = FlatEnvWithRoads()
        grid = LegacyHexGrid(120.0115, 23.01, 12, 12, 500.0)
        coarse = GridPlanner(coarse_env, grid)
        start = (119.996, 23.008)
        goal = (120.026, 23.008)
        base = coarse.plan(self.prof, start, goal)
        self.assertIsNotNone(base)
        hybrid = self.nav.plan_hybrid(coarse, self.prof, start, goal)
        self.assertIsNotNone(hybrid)
        inside = [p for p in hybrid.points
                  if self.nav.covers(p.lon, p.lat)]
        self.assertGreaterEqual(len(inside), 2)
        for p in inside:
            self.assertTrue(self.nav.passable(self.prof, p.lon, p.lat))
        # The refinement must not leave a vertex inside the water rectangle.
        self.assertFalse(any(_in_rect(p.lon, p.lat, WATER)
                             for p in hybrid.points
                             if self.nav.covers(p.lon, p.lat)))

    def test_speed_factor_matches_mask(self):
        lon, lat = 120.002, 23.008
        self.assertAlmostEqual(self.nav.speed_factor(self.prof, lon, lat), 1.0)
        lon, lat = 120.010, 23.010  # water
        self.assertLess(self.nav.speed_factor(self.prof, lon, lat), 0.5)

    def _dense_ok(self, nav, route, step=5.0):
        for i in range(1, len(route.points)):
            a, b = route.points[i - 1], route.points[i]
            total = haversine_m((a.lon, a.lat), (b.lon, b.lat))
            n = max(1, int(math.ceil(total / step)))
            for k in range(n + 1):
                f = k / float(n)
                lon = a.lon + (b.lon - a.lon) * f
                lat = a.lat + (b.lat - a.lat) * f
                if nav.passable(self.prof, lon, lat) is False:
                    return False
        return True

    def test_default_resolution_segments_stay_passable(self):
        # Regression: decimating A* vertices at >45 m used to shortcut a
        # blocked cell at the default 30 m resolution.
        nav = NavField(self.env, res_m=30.0, use_cache=False)
        route = nav.plan(self.prof, (120.002, 23.008), (120.018, 23.008))
        self.assertIsNotNone(route)
        self.assertTrue(self._dense_ok(nav, route))

    def test_start_equals_goal(self):
        p = (120.002, 23.008)
        route = self.nav.plan(self.prof, p, p)
        self.assertIsNotNone(route)
        self.assertEqual(route.points[-1].lon, p[0] + 1e-7)

    def test_blocked_start_escapes(self):
        nav = NavField(self.env, res_m=30.0, use_cache=False)
        route = nav.plan(self.prof, (120.015, 23.010), (120.002, 23.008))
        self.assertIsNotNone(route)
        entity = Entity(9, "E", 0, "land", "tracked", Pose(120.015, 23.010))
        entity.profile = self.prof
        entity.route = route
        entity.moving = True
        start = (entity.pose.lon, entity.pose.lat)
        for _ in range(500):
            if entity.route is None:
                break
            self.assertNotEqual(integrate(entity, 5.0, self.env, nav=nav),
                                "blocked")
        self.assertGreater(
            haversine_m(start, (entity.pose.lon, entity.pose.lat)), 30.0)
        self.assertTrue(nav.passable(self.prof, entity.pose.lon, entity.pose.lat))

    def test_hybrid_rejects_walled_route(self):
        env = FakeFineEnv(water=(120.008, 23.0, 120.012, 23.02))
        nav = NavField(env, res_m=50.0, use_cache=False)
        coarse = GridPlanner(FlatEnvWithRoads(),
                             LegacyHexGrid(120.0115, 23.01, 12, 12, 500.0))
        route = nav.plan_hybrid(coarse, self.prof, (119.996, 23.008),
                                (120.026, 23.008))
        self.assertIsNone(route)

    def test_goal_snaps_into_start_cell_is_validated(self):
        # Goal inside the building next to a street; the snapped goal cell
        # may equal the start cell.  Any returned route must be clear.
        nav = NavField(self.env, res_m=30.0, use_cache=False)
        route = nav.plan(self.prof, (120.0135, 23.010), (120.015, 23.010))
        if route is not None:
            self.assertTrue(nav.route_ok(self.prof, route))

    def test_bridge_deck_is_passable_over_water(self):
        nodes = {"A": (120.007, 23.010), "B": (120.013, 23.010)}
        edges = [{"a": "A", "b": "B", "class": "residential",
                  "bridge": True, "tunnel": False,
                  "maxweight": None, "maxheight": None, "width": None}]
        nav = NavField(FakeFineEnv(roads=(nodes, edges)), res_m=30.0,
                       use_cache=False)
        self.assertTrue(nav.passable(self.prof, 120.010, 23.010))
        # Off the bridge the water still blocks.
        self.assertFalse(nav.passable(self.prof, 120.010, 23.006))


PACK = os.path.join(SITUATION3D, "envpacks", "anping_v2c")


@unittest.skipUnless(os.path.isdir(PACK), "anping_v2c envpack not present")
class EnvPackNavFieldTest(unittest.TestCase):
    def test_pack_nav_is_connected_and_plans(self):
        from env.pack_env import EnvPackEnvironment
        env = EnvPackEnvironment(PACK)
        nav = NavField(env)
        prof = PROFILES["tracked"]
        mask, _ = nav.profile_mask(prof)
        total = sum(mask)
        self.assertGreater(total, 1000)
        # Largest 8-connected component must cover most passable cells.
        seen = bytearray(len(mask))
        largest = 0
        for i in range(len(mask)):
            if not mask[i] or seen[i]:
                continue
            stack = [i]
            seen[i] = 1
            size = 0
            while stack:
                j = stack.pop()
                size += 1
                r, c = divmod(j, nav.cols)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < nav.rows and 0 <= cc < nav.cols:
                        k = rr * nav.cols + cc
                        if mask[k] and not seen[k]:
                            seen[k] = 1
                            stack.append(k)
            largest = max(largest, size)
        self.assertGreater(largest / float(total), 0.9)
        route = nav.plan(prof, (120.1845, 23.0199), (120.1895, 23.0113))
        self.assertIsNotNone(route)
        for p in route.points:
            self.assertTrue(nav.passable(prof, p.lon, p.lat))


if __name__ == "__main__":
    unittest.main()
