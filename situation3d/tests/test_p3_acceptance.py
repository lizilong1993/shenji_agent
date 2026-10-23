"""P3 acceptance: scale-invariant cadence, invariants, kinematic/fire validation.

Run from ``situation3d/``:
    python3 -m unittest tests/test_p3_acceptance.py
"""

from __future__ import annotations

import hashlib
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

from env.service import LandCover, Weather  # noqa: E402
from geo.geodesy import haversine_m, initial_bearing_deg  # noqa: E402
from mobility import kinematics  # noqa: E402
from mobility.nav_field import NavField  # noqa: E402
from mobility.profile import PROFILES  # noqa: E402
from sim import perception  # noqa: E402
from sim.collision import obbs_overlap  # noqa: E402
from sim.entity import Entity, Pose, Route  # noqa: E402
from sim.motion import integrate  # noqa: E402
from sim.scenario import build_world  # noqa: E402

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 24, "rows": 24,
        "width_m": 200.0}


def make_basic(water=False):
    cond = 4 if water else 0
    cells = [[{"pos": r * 100 + c, "elev": 100, "cond": cond,
               "roads": [0] * 6, "rivers": [0] * 6}
              for c in range(24)] for r in range(24)]
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID),
            "map_data": cells}


def operator(uid, cell, color=0):
    return {"obj_id": uid, "name": "重型坦克", "color": color, "type": 2,
            "sub_type": 0, "cur_hex": cell, "blood": 4, "max_blood": 4,
            "armor": 2, "value": 10, "domain": "land",
            "carry_weapon_ids": [36],
            "observe_distance": {"1": 10, "2": 25, "3": 1},
            "move_state": 0, "stop": 1, "readiness": 0.85, "fuel": 1.0,
            "emitting": True}


def build(ops, seed=7, engine=None):
    sc = {"scenario_id": 1, "cities": [], "operators": ops,
          "time": {"cur_step": 0, "tick": 1.0, "max_time": 9999},
          "engine": engine or {}}
    return build_world(sc, make_basic(), seed=seed)


class FlatBox(object):
    """Flat obstacle-free box for nav planning tests."""

    def __init__(self, water=None):
        self.water = water
        self._weather = Weather()

    def nav_bbox(self):
        return (120.0, 23.0, 120.02, 23.02)

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
        return {}, []

    def on_road(self, lon, lat, tolerance_m=30.0):
        return False

    def road_class(self, lon, lat, tolerance_m=30.0):
        return 0

    def building_intersects(self, a, h0, b, h1):
        return False


class CadenceTest(unittest.TestCase):
    def _due(self, n_entities):
        ops = [operator(i + 1, 1000 + (i % 20) * 100 + (i // 20)) for i in
               range(n_entities)]
        world = build(ops)
        world.step()
        return world, dict(world._scan_due)

    def test_scan_cadence_independent_of_force_size(self):
        small, due_small = self._due(10)
        large, due_large = self._due(70)
        shared = set(due_small) & set(due_large)
        self.assertEqual(shared, set(range(1, 11)))
        for uid in shared:
            # Same per-observer phase regardless of force size; the first
            # full sweep is followed by a phase offset below one interval.
            self.assertEqual(due_small[uid], due_large[uid])
            self.assertGreaterEqual(due_small[uid], 30.0)
            self.assertLess(due_small[uid], 60.0)
        for _ in range(60):
            small.step()
            large.step()
        for uid in shared:
            # Rescan happens on the first tick at/after the phase due; the
            # next due is anchored to that tick plus exactly one interval.
            self.assertGreaterEqual(small._scan_due[uid], due_small[uid] + 30.0)
            self.assertLess(small._scan_due[uid], due_small[uid] + 31.0)
            self.assertGreaterEqual(large._scan_due[uid], due_large[uid] + 30.0)
            self.assertLess(large._scan_due[uid], due_large[uid] + 31.0)


class InvariantTest(unittest.TestCase):
    def test_no_overlap_teleport_or_heading_jump(self):
        rng = random.Random(11)
        # Cluster onto 16 cells so four+ units share a spawn cell and the
        # spawn de-overlap pass is actually exercised.
        ops = [operator(i + 1, 1000 + (i % 4) * 100 + (i // 4) % 4, 0)
               for i in range(40)]
        world = build(ops, seed=11)
        initial_overlap = set()
        ground = [e for e in world.sorted_entities()
                  if e.alive and e.domain == "land"]
        for i in range(len(ground)):
            for j in range(i + 1, len(ground)):
                a, b = ground[i], ground[j]
                if obbs_overlap(a.profile, a.pose.lon, a.pose.lat,
                                a.pose.heading_deg, b.profile,
                                b.pose.lon, b.pose.lat,
                                b.pose.heading_deg, margin=0.0):
                    initial_overlap.add((a.uid, b.uid))
        # Spawn de-overlap leaves no overlapping pairs; fail if it regresses.
        self.assertEqual(initial_overlap, set())
        for tick in range(120):
            if tick % 10 == 0:
                for e in world.sorted_entities():
                    cell = (rng.randrange(4, 20), rng.randrange(4, 20))
                    lon, lat = world.rule_grid.cell_center_lonlat(*cell)
                    world.order(e.uid, {"type": "move_to", "lon": lon,
                                        "lat": lat})
            before = {e.uid: (e.pose.lon, e.pose.lat, e.velocity_ms,
                              e.pose.heading_deg)
                      for e in world.sorted_entities()
                      if e.alive and e.domain == "land"}
            world.step()
            entities = [e for e in world.sorted_entities()
                        if e.alive and e.domain == "land"]
            for e in entities:
                prev = before.get(e.uid)
                if prev is None:
                    continue
                kx = 111320.0 * math.cos(math.radians(e.pose.lat))
                moved = math.hypot((e.pose.lon - prev[0]) * kx,
                                   (e.pose.lat - prev[1]) * 110540.0)
                self.assertLessEqual(moved, e.profile.max_speed_ms + 2.0)
                jump = abs(kinematics.wrap180(e.pose.heading_deg - prev[3]))
                self.assertLessEqual(
                    jump, kinematics.max_yaw_rate(
                        e.profile, max(prev[2], e.velocity_ms)) * 180.0
                    / math.pi + 3.0)
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    a, b = entities[i], entities[j]
                    self.assertFalse(obbs_overlap(
                        a.profile, a.pose.lon, a.pose.lat,
                        a.pose.heading_deg, b.profile, b.pose.lon,
                        b.pose.lat, b.pose.heading_deg, margin=0.0),
                        "overlap %s/%s at tick %s" % (a.uid, b.uid, tick))


class KinematicsValidationTest(unittest.TestCase):
    def test_turn_radius_feasible(self):
        nav = NavField(FlatBox(water=(120.008, 23.002, 120.012, 23.016)),
                       res_m=30.0, use_cache=False)
        prof = PROFILES["wheeled"]
        route = nav.plan(prof, (120.002, 23.008), (120.018, 23.008))
        self.assertIsNotNone(route)
        pts = route.points
        radius = prof.min_turn_radius_m
        for i in range(1, len(pts) - 1):
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            h1 = initial_bearing_deg((a.lon, a.lat), (b.lon, b.lat))
            h2 = initial_bearing_deg((b.lon, b.lat), (c.lon, c.lat))
            turn = abs(kinematics.wrap180(h2 - h1))
            self.assertLessEqual(turn, 35.0)
            if turn < 1.0:
                continue
            chord = max(1.0, haversine_m((a.lon, a.lat), (b.lon, b.lat)))
            curvature = 2.0 * math.sin(math.radians(turn) / 2.0) / chord
            self.assertLessEqual(curvature, 1.0 / radius * 1.1,
                                 "vertex %d curvature %.3f > 1/R" % (i, curvature))

    def test_braking_distance(self):
        env = FlatBox()
        prof = PROFILES["tracked"]
        e = Entity(1, "B", 0, "land", "tracked", Pose(120.0, 23.0))
        e.profile = prof
        e.route = Route(points=[Pose(120.0, 23.0), Pose(120.0, 23.02)],
                        speed_limits=[10.0], modes=["offroad"])
        prev_v = 0.0
        braking = False
        distance = 0.0
        max_dv = 0.0
        last = (e.pose.lon, e.pose.lat)
        for _ in range(2000):
            if e.route is None:
                break
            integrate(e, 0.5, env)
            moved = haversine_m(last, (e.pose.lon, e.pose.lat))
            last = (e.pose.lon, e.pose.lat)
            dv = abs(e.velocity_ms - prev_v)
            if braking or (e.velocity_ms < prev_v - 0.01 and prev_v > 5.0):
                braking = True
                distance += moved
                max_dv = max(max_dv, dv)
            prev_v = e.velocity_ms
        self.assertTrue(braking)
        ideal = 10.0 ** 2 / (2.0 * prof.decel_ms2)
        self.assertGreaterEqual(distance, 0.8 * ideal)
        self.assertLessEqual(distance, 1.3 * ideal + 3.0)
        self.assertLessEqual(max_dv, prof.decel_ms2 * 0.5 + 0.25)


class FireValidationTest(unittest.TestCase):
    def test_detection_probability_calibration(self):
        env = FlatBox()
        obs = Entity(1, "O", 0, "land", "tracked", Pose(120.0, 23.0))
        obs.profile = PROFILES["tracked"]
        tgt = Entity(2, "T", 1, "land", "tracked", Pose(120.0, 23.02))
        tgt.profile = PROFILES["tracked"]
        spec = perception.specs_for(obs)[0]
        distance = 1500.0
        p = perception.detection_probability(spec, obs, tgt, distance, env)
        self.assertGreater(p, 0.0)
        hits = sum(1 for seed in range(600)
                   if perception.detection_roll(seed, 0, 1, 2) < p)
        self.assertAlmostEqual(hits / 600.0, p, delta=0.06)

    def test_roc_auc_monotonic(self):
        env = FlatBox()
        obs = Entity(1, "O", 0, "land", "tracked", Pose(120.0, 23.0))
        obs.profile = PROFILES["tracked"]
        tgt = Entity(2, "T", 1, "land", "tracked", Pose(120.0, 23.0))
        tgt.profile = PROFILES["tracked"]
        spec = perception.specs_for(obs)[0]
        scores = []
        labels = []
        for distance in range(100, 4600, 100):
            p = perception.detection_probability(spec, obs, tgt,
                                                 float(distance), env)
            scores.append(p)
            labels.append(1 if distance <= 3000 else 0)
        pos = [s for s, y in zip(scores, labels) if y == 1]
        neg = [s for s, y in zip(scores, labels) if y == 0]
        wins = sum(1 for a in pos for b in neg if a > b)
        ties = sum(1 for a in pos for b in neg if a == b)
        auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
        self.assertGreater(auc, 0.9)


class CadenceSemanticsTest(unittest.TestCase):
    def _world(self, engine=None):
        ops = []
        for uid, cell, color in ((1, 1000, 0), (2, 1001, 1), (3, 1002, 1)):
            op = operator(uid, cell, color=color)
            op["blood"] = op["max_blood"] = 999   # survive any exchange
            ops.append(op)
        return build(ops, engine=engine)

    def test_unexpired_observer_keeps_prior_seen(self):
        world = self._world()
        world.step()                      # t=1, first full scan
        self.assertIn(2, world.seen.get(1, set()))
        world.entities[2].pose.lon += 1.0  # moved far out of sight
        world.step()                      # t=2, observer not due
        self.assertIn(2, world.seen.get(1, set()))
        due = world._scan_due[1]
        while world.time_s < due - 1.0:
            world.step()
        world.entities[2].pose.lon += 1.0  # still out of sight at rescan
        world.time_s = due
        world.step()                      # processes at the observer due
        self.assertNotIn(2, world.seen.get(1, set()))

    def test_time_jump_scans_once_and_reanchors_due(self):
        world = self._world()
        world.step()
        calls = []
        orig = world._advance_scan_due

        def wrapped(uid, due_map=None, interval=None):
            calls.append(uid)
            return orig(uid, due_map, interval)

        world._advance_scan_due = wrapped
        world.time_s = 100.0
        world.step()                      # jump across 3 periods -> one scan
        self.assertEqual(sorted(calls), [1, 2, 3])
        due = world._scan_due[1]
        phase = world._scan_phase(1)
        self.assertGreater(due, 100.0)
        self.assertLessEqual(due, 130.0 + 1e-9)
        # Phase survives the jump: the next due is phase-aligned.
        self.assertAlmostEqual((due - phase) % 30.0, 0.0, places=6)

    def test_zero_interval_rescans_every_step(self):
        self.assertEqual(self._world(engine={"scan_interval_s": -1})
                         .scan_interval_s, 30.0)
        world = self._world(engine={"scan_interval_s": 0})
        world.step()
        self.assertIn(2, world.seen.get(1, set()))
        world.entities[2].pose.lon += 1.0
        world.step()
        self.assertNotIn(2, world.seen.get(1, set()))

    def test_dead_or_boarded_observer_due_removed(self):
        world = self._world()
        world.step()
        self.assertIn(3, world._scan_due)
        world._on_destroyed(world.entities[3])
        world.step()
        self.assertNotIn(3, world._scan_due)
        self.assertNotIn(3, world.seen)

    def test_boarded_observer_due_removed(self):
        world = self._world()
        world.step()
        self.assertIn(3, world._scan_due)
        world.entities[3].on_board = True
        world._seen_dirty = True
        world.step()
        self.assertNotIn(3, world._scan_due)
        self.assertNotIn(3, world.seen)

    def test_scan_interval_controls_refresh_rate(self):
        counts = {}
        for interval in (3.0, 30.0):
            world = self._world(engine={"scan_interval_s": interval})
            calls = []
            orig = world._advance_scan_due

            def wrapped(uid, due_map=None, interval=None,
                        _orig=orig, _calls=calls):
                _calls.append(uid)
                return _orig(uid, due_map, interval)

            world._advance_scan_due = wrapped
            for _ in range(31):
                world.step()
            counts[interval] = calls.count(1)
        self.assertGreaterEqual(counts[3.0], 9)
        self.assertLessEqual(counts[3.0], 11)
        self.assertEqual(counts[30.0], 1)   # only the initial full sweep

    def test_physical_roll_bucket_follows_interval(self):
        seen = []
        original = perception.detection_roll

        def fake(seed, bucket, observer_uid, target_uid):
            seen.append(bucket)
            return 1.0

        original_specs = perception.specs_for
        perception.detection_roll = fake
        perception.specs_for = lambda entity: [perception.SensorSpec(
            "radar", 5000.0, 360.0, 90.0, 0.95, 100.0)]
        try:
            ops = [operator(1, 1000), operator(2, 1001, color=1)]
            world = build(ops, engine={"detection_model": "physical",
                                        "scan_interval_s": 3.0})
            for _ in range(7):
                world.step()
        finally:
            perception.detection_roll = original
            perception.specs_for = original_specs
        self.assertGreaterEqual(len(set(seen)), 2)

    def test_physical_scan_due_tracks_interval(self):
        world = self._world(engine={"detection_model": "physical",
                                     "scan_interval_s": 12.0})
        world.step()
        self.assertTrue(world._phys_scan_due)
        due = world._phys_scan_due.get(1)
        self.assertGreaterEqual(due, 12.0)
        self.assertLess(due, 24.0)
        world.step()
        self.assertGreater(world._phys_scan_due.get(1), world.time_s)


class DeterminismTest(unittest.TestCase):
    def test_event_and_position_hash(self):
        ops = [operator(i + 1, 1000 + i) for i in range(12)]

        def run():
            world = build(ops, seed=21)
            rng = random.Random(3)
            for _ in range(120):
                for e in world.sorted_entities():
                    if rng.random() < 0.05:
                        cell = (rng.randrange(4, 20), rng.randrange(4, 20))
                        lon, lat = world.rule_grid.cell_center_lonlat(*cell)
                        world.order(e.uid, {"type": "move_to", "lon": lon,
                                            "lat": lat})
                world.step()
            events = hashlib.sha256(json.dumps(
                world.events, sort_keys=True, ensure_ascii=False
            ).encode()).hexdigest()
            positions = hashlib.sha256(json.dumps(
                [(uid, round(e.pose.lon, 9), round(e.pose.lat, 9),
                  e.alive) for uid, e in sorted(world.entities.items())]
            ).encode()).hexdigest()
            return events, positions

        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
