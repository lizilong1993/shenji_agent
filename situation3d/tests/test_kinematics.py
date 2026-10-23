"""P0.2 kinematics and physical collision tests.

Run from ``situation3d/``:
    python3 -m unittest tests/test_kinematics.py
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
from env.service import FlatEnvironment  # noqa: E402
from geo.geodesy import haversine_m  # noqa: E402
from mobility import kinematics  # noqa: E402
from mobility.profile import PROFILES  # noqa: E402
from sim.collision import obbs_overlap  # noqa: E402
from sim.entity import Entity, Pose, Route  # noqa: E402
from sim.motion import integrate  # noqa: E402
from sim.scenario import build_world  # noqa: E402

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 24, "rows": 24,
        "width_m": 200.0}


def make_basic(rows=24, cols=24):
    cells = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append({"pos": r * 100 + c, "elev": 100, "cond": 0,
                        "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID), "map_data": cells}


def operator(uid, name, cell, domain="land", unit_type="vehicle", heading=None):
    op = {"obj_id": uid, "name": name, "color": 0, "type": 2, "sub_type": 0,
          "cur_hex": cell, "blood": 3, "max_blood": 3, "armor": 2,
          "basic_speed": 30, "value": 10, "domain": domain,
          "observe_distance": {"1": 10, "2": 25, "3": 1},
          "move_state": 0, "stop": 1, "readiness": 0.85, "fuel": 1.0,
          "emitting": True}
    if heading is not None:
        op["heading_deg"] = heading
    return op


def scenario(ops):
    return {"scenario_id": 1, "time": {"cur_step": 0, "tick": 1.0, "max_time": 1000},
            "cities": [], "operators": ops, "supply_points": []}


def _world(ops):
    return build_world(scenario(ops), make_basic(), seed=11)


class KinematicsTest(unittest.TestCase):
    def test_heading_turn_rate_limited(self):
        env = FlatEnvironment()
        e = Entity(1, "T", 0, "land", "tracked", Pose(119.9, 23.1))
        e.profile = PROFILES["tracked"]
        e.route = Route(points=[Pose(119.9, 23.1), Pose(119.9, 23.102),
                                Pose(119.901, 23.102)],
                        speed_limits=[10.0, 10.0], modes=["offroad", "offroad"])
        limit = PROFILES["tracked"].turn_rate_dps + 1e-6
        prev = e.pose.heading_deg
        turned = 0.0
        for _ in range(200):
            integrate(e, 1.0, env)
            delta = abs(kinematics.wrap180(e.pose.heading_deg - prev))
            self.assertLessEqual(delta, limit)
            turned = max(turned, e.pose.heading_deg)
            prev = e.pose.heading_deg
            if e.route is None:
                break
        self.assertGreater(turned, 80.0)   # the 90 degree turn was completed

    def test_braking_stops_at_route_end(self):
        env = FlatEnvironment()
        e = Entity(1, "T", 0, "land", "tracked", Pose(119.9, 23.1))
        e.profile = PROFILES["tracked"]
        e.route = Route(points=[Pose(119.9, 23.1), Pose(119.9, 23.1045)],
                        speed_limits=[10.0], modes=["offroad"])
        vmax = 0.0
        for _ in range(400):
            integrate(e, 1.0, env)
            vmax = max(vmax, e.velocity_ms)
            if e.route is None:
                break
        self.assertIsNone(e.route)
        self.assertEqual(e.velocity_ms, 0.0)
        self.assertLessEqual(vmax, 10.0 + 1e-6)
        self.assertLess(haversine_m((e.pose.lon, e.pose.lat),
                                    (119.9, 23.1045)), 2.0)

    def test_corner_speed_cap(self):
        route = Route(points=[Pose(119.9, 23.1), Pose(119.9, 23.11),
                              Pose(119.9001, 23.11)],
                      speed_limits=[10.0, 10.0], modes=["offroad", "offroad"])
        caps = kinematics.speed_caps(route, PROFILES["tracked"])
        # A 90 degree turn over a ~10 m segment must lower the vertex cap.
        self.assertLess(caps[1], 10.0)

    def test_wheeled_low_speed_yaw(self):
        wheeled = PROFILES["wheeled"]
        base = math.radians(wheeled.turn_rate_dps)
        self.assertAlmostEqual(kinematics.max_yaw_rate(wheeled, 0.0), base * 0.25)
        self.assertLess(kinematics.max_yaw_rate(wheeled, 4.0), base)

    def test_mixed_segment_limits_are_honoured(self):
        route = Route(points=[Pose(119.9, 23.1), Pose(119.9, 23.102),
                              Pose(119.9, 23.104)],
                      speed_limits=[25.0, 5.0], modes=["offroad", "offroad"])
        caps = kinematics.speed_caps(route, PROFILES["tracked"])
        # Second segment is slow; the cap entering it must not stay at 25.
        self.assertLessEqual(caps[1], 5.0 + 1e-6)
        self.assertLessEqual(caps[0], 25.0 + 1e-6)

    def test_zero_length_leg_no_fake_corner(self):
        route = Route(points=[Pose(119.9, 23.1), Pose(119.9, 23.101),
                              Pose(119.9, 23.101), Pose(119.9, 23.102)],
                      speed_limits=[10.0, 10.0, 10.0],
                      modes=["offroad", "offroad", "offroad"])
        caps = kinematics.speed_caps(route, PROFILES["tracked"])
        self.assertEqual(caps[0], 10.0)   # no artificial 90-degree corner

    def test_short_speed_limits_do_not_crash(self):
        route = Route(points=[Pose(119.9, 23.1), Pose(119.9, 23.101),
                              Pose(119.9, 23.102)],
                      speed_limits=[10.0], modes=["offroad"])
        caps = kinematics.speed_caps(route, PROFILES["tracked"])
        self.assertEqual(len(caps), 3)
        self.assertTrue(all(0.0 <= c <= 10.0 + 1e-6 for c in caps))

    def test_yield_keeps_heading(self):
        env = FlatEnvironment()
        e = Entity(1, "T", 0, "land", "tracked", Pose(119.9, 23.1))
        e.profile = PROFILES["tracked"]
        e.route = Route(points=[Pose(119.9, 23.1), Pose(119.902, 23.1)],
                        speed_limits=[10.0], modes=["offroad"])
        heading0 = e.pose.heading_deg
        for _ in range(20):
            reason = integrate(e, 1.0, env,
                               traffic_check=lambda lon, lat, hd: False)
            if reason == "yield":
                break
        self.assertEqual(reason, "yield")
        self.assertEqual(e.pose.heading_deg, heading0)
        self.assertIsNotNone(e.route)


class CollisionTest(unittest.TestCase):
    def test_queue_keeps_gap_and_both_move(self):
        grid = LegacyHexGrid.from_grid_meta(GRID)
        env = MapGridEnvironment(make_basic(), grid)
        start = grid.cell_center_lonlat(10, 5)
        goal = grid.cell_center_lonlat(10, 14)
        world = build_world(scenario([operator(1, "lead", 1005),
                                      operator(2, "follow", 1005)]),
                            make_basic(), seed=3)
        lead = world.entities[1]
        follow = world.entities[2]
        lead.pose = Pose(start[0], start[1])
        follow.pose = Pose(start[0], start[1] + 0.00009)  # ~10 m south
        for uid in (1, 2):
            world.order(uid, {"type": "move_to", "lon": goal[0], "lat": goal[1]})
        p_lead, p_follow = PROFILES["tracked"], PROFILES["tracked"]
        overlap_seen = False
        yield_seen = False
        for _ in range(400):
            world.step()
            if obbs_overlap(p_lead, lead.pose.lon, lead.pose.lat,
                            lead.pose.heading_deg, p_follow,
                            follow.pose.lon, follow.pose.lat,
                            follow.pose.heading_deg, margin=0.0):
                overlap_seen = True
            if follow.velocity_ms == 0.0 and follow.route is not None:
                yield_seen = True
            if lead.route is None and follow.route is None:
                break
        self.assertFalse(overlap_seen)
        self.assertTrue(yield_seen)
        self.assertGreater(haversine_m((start[0], start[1]),
                                       (follow.pose.lon, follow.pose.lat)), 200.0)

    def test_five_units_in_one_rule_cell_can_move(self):
        grid = LegacyHexGrid.from_grid_meta(GRID)
        center = grid.cell_center_lonlat(10, 10)
        ops = [operator(i + 1, "U%d" % i, 1010) for i in range(5)]
        world = build_world(scenario(ops), make_basic(), seed=5)
        for i, uid in enumerate(sorted(world.entities)):
            e = world.entities[uid]
            e.pose = Pose(center[0], center[1] + i * 0.00011)  # ~12 m apart
        goals = {}
        for uid in sorted(world.entities):
            e = world.entities[uid]
            goals[uid] = (e.pose.lon, e.pose.lat + 0.003)
            world.order(uid, {"type": "move_to", "lon": goals[uid][0],
                              "lat": goals[uid][1]})
        starts = {uid: (world.entities[uid].pose.lon, world.entities[uid].pose.lat)
                  for uid in world.entities}
        for _ in range(250):
            world.step()
        self.assertFalse(any(ev.get("type") == "stack_blocked"
                             for ev in world.events))
        moved = {uid: haversine_m(starts[uid],
                                  (world.entities[uid].pose.lon,
                                   world.entities[uid].pose.lat))
                 for uid in world.entities}
        for uid, dist in moved.items():
            self.assertGreater(dist, 5.0, "unit %s did not move" % uid)
        # The whole group progressed (no four-per-cell deadlock).
        self.assertGreater(max(moved.values()), 200.0)

    def test_spawn_deoverlap_preserves_heading(self):
        ops = [operator(1, "A", 1010, heading=30.0),
               operator(2, "B", 1010, heading=30.0)]
        world = build_world(scenario(ops), make_basic(), seed=9)
        a, b = world.entities[1], world.entities[2]
        self.assertFalse(obbs_overlap(
            PROFILES["tracked"], a.pose.lon, a.pose.lat, a.pose.heading_deg,
            PROFILES["tracked"], b.pose.lon, b.pose.lat, b.pose.heading_deg))
        self.assertEqual(a.pose.heading_deg, 30.0)
        self.assertEqual(b.pose.heading_deg, 30.0)

    def test_head_on_yield_escalates(self):
        ops = [operator(1, "A", 1005), operator(2, "B", 1006)]
        world = build_world(scenario(ops), make_basic(), seed=13)
        a, b = world.entities[1], world.entities[2]
        grid = world.env.grid
        ca, cb = grid.cell_center_lonlat(10, 5), grid.cell_center_lonlat(10, 6)
        a.pose = Pose(ca[0], ca[1], heading_deg=90.0)
        b.pose = Pose(cb[0], cb[1], heading_deg=270.0)
        world.order(1, {"type": "move_to", "lon": cb[0], "lat": cb[1]})
        world.order(2, {"type": "move_to", "lon": ca[0], "lat": ca[1]})
        for _ in range(45):
            world.step()
        self.assertTrue(any(ev.get("type") == "traffic_stuck"
                            for ev in world.events))
        self.assertIsNone(a.route)
        self.assertIsNone(b.route)

    def test_deterministic_with_kinematics(self):
        ops = [operator(1, "A", 1005), operator(2, "B", 1006, unit_type="vehicle")]
        a = _world(ops)
        b = _world(ops)
        for uid in (1, 2):
            goal_hex = 1012
            for w in (a, b):
                lon, lat = w.env.grid.to_lonlat(goal_hex)
                w.order(uid, {"type": "move_to", "lon": lon, "lat": lat})
        for _ in range(80):
            ea = a.step()
            eb = b.step()
            self.assertEqual([(e.get("type"), e.get("actors")) for e in ea],
                             [(e.get("type"), e.get("actors")) for e in eb])
        for uid in (1, 2):
            self.assertAlmostEqual(a.entities[uid].pose.lon,
                                   b.entities[uid].pose.lon, places=9)
            self.assertAlmostEqual(a.entities[uid].pose.lat,
                                   b.entities[uid].pose.lat, places=9)


if __name__ == "__main__":
    unittest.main()
