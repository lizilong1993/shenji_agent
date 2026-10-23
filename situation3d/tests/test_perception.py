"""P1.4 physical perception tests: geometry, FoV, probability, tracks.

Run from ``situation3d/``:
    python3 -m unittest tests/test_perception.py
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

from env.service import LandCover, Weather  # noqa: E402
from geo.rule_grid import RuleGrid  # noqa: E402
from mobility.profile import PROFILES  # noqa: E402
from sim import perception  # noqa: E402
from sim.entity import Entity, Pose  # noqa: E402
from sim.sensors import Sensor  # noqa: E402
from sim.world import World  # noqa: E402


class PhysicalEnv(object):
    def __init__(self, ridge=None, building=None, visibility=10000.0):
        self.ridge = ridge          # (lat0, lat1, top_m)
        self.building = building    # (lon0, lat0, lon1, lat1, top_m)
        self.visibility = visibility
        self._weather = Weather(visibility_m=visibility)

    def elevation(self, lon, lat):
        if self.ridge:
            lat0, lat1, top = self.ridge
            if lat0 <= lat <= lat1:
                return top
        return 0.0

    def landcover(self, lon, lat):
        return LandCover.OPEN

    def weather(self, lon, lat):
        return self._weather

    def building_intersects(self, a, h0, b, h1):
        if not self.building:
            return False
        x0, y0, x1, y1, top = self.building
        n = 20
        for i in range(n + 1):
            f = i / float(n)
            lon = a[0] + (b[0] - a[0]) * f
            lat = a[1] + (b[1] - a[1]) * f
            if x0 <= lon <= x1 and y0 <= lat <= y1:
                sight = h0 + (h1 - h0) * f
                if sight < top:
                    return True
        return False


def _world(env, seed=7):
    grid = RuleGrid(120.0, 23.0, 40, 40)
    return World(env, grid, seed=seed, detection_model="physical")


def _observer(lon, lat, heading=0.0, alt=0.0):
    e = Entity(1, "OBS", 0, "land", "tracked",
               Pose(lon, lat, alt, heading_deg=heading))
    e.profile = PROFILES["tracked"]
    return e


def _target(lon, lat, uid=2, unit_type="tracked", alt=0.0):
    return Entity(uid, "TGT", 1, "land", unit_type,
                  Pose(lon, lat, alt))


def _spec(entity, kind="optics"):
    for spec in perception.specs_for(entity):
        if spec.kind == kind:
            return spec
    raise AssertionError("no %s sensor" % kind)


def _spec_of_kind(entity, kind):
    base = _spec(entity, "optics")
    defaults = perception._KINDS[kind]
    return perception.SensorSpec(
        kind=kind, range_m=base.range_m * 4.0,
        fov_deg=defaults["fov_deg"],
        elev_fov_deg=defaults["elev_fov_deg"], p0=defaults["p0"],
        scale_m=base.range_m * 4.0 * defaults["scale_frac"])


class GeometryTest(unittest.TestCase):
    def test_ridge_blocks_then_elevation_clears(self):
        env = PhysicalEnv(ridge=(23.0095, 23.0105, 200.0))
        obs = _observer(120.0, 23.0)
        obs.profile = PROFILES["tracked"]
        tgt = _target(120.0, 23.02)
        spec = _spec(obs)
        ground = perception.detection_probability(
            spec, obs, tgt, 2200.0, env)
        self.assertEqual(ground, 0.0)
        obs.pose.alt_m = 500.0
        elevated = perception.detection_probability(
            spec, obs, tgt, 2200.0, env)
        self.assertGreater(elevated, 0.0)

    def test_building_blocks_line_of_sight(self):
        env = PhysicalEnv(building=(119.9995, 23.004, 120.0005, 23.006, 40.0))
        obs = _observer(120.0, 23.0)
        obs.profile = PROFILES["tracked"]
        tgt = _target(120.0, 23.01)
        spec = _spec(obs)
        self.assertEqual(perception.detection_probability(
            spec, obs, tgt, 1100.0, env), 0.0)
        tgt.pose = Pose(120.01, 23.0)
        obs.pose.heading_deg = 90.0
        self.assertGreater(perception.detection_probability(
            spec, obs, tgt, 1000.0, env), 0.0)

    def test_smoke_attenuates_optics(self):
        env = PhysicalEnv()
        obs = _observer(120.0, 23.0)
        obs.profile = PROFILES["tracked"]
        tgt = _target(120.0, 23.002)
        spec = _spec(obs)
        clear = perception.detection_probability(spec, obs, tgt, 220.0, env)
        smoke = [{"lon": 120.0, "lat": 23.001, "radius_m": 300.0,
                  "top_m": 100.0, "from_s": 0.0, "to_s": 1e18,
                  "attenuation": 0.0}]
        obscured = perception.detection_probability(
            spec, obs, tgt, 220.0, env, now_s=10.0, obscurants=smoke)
        self.assertGreater(clear, 0.3)
        self.assertEqual(obscured, 0.0)

    def test_sensor_fov(self):
        env = PhysicalEnv()
        obs = _observer(120.0, 23.0, heading=90.0)  # facing east
        obs.profile = PROFILES["tracked"]
        tgt = _target(120.0, 23.002)  # due north
        spec = _spec(obs)
        self.assertEqual(perception.detection_probability(
            spec, obs, tgt, 220.0, env), 0.0)
        obs.pose.heading_deg = 0.0
        self.assertGreater(perception.detection_probability(
            spec, obs, tgt, 220.0, env), 0.0)


class ProbabilityTest(unittest.TestCase):
    def test_distance_monotonic(self):
        env = PhysicalEnv()
        obs = _observer(120.0, 23.0)
        obs.profile = PROFILES["tracked"]
        tgt = _target(120.0, 23.002)
        spec = _spec(obs)
        near = perception.detection_probability(spec, obs, tgt, 300.0, env)
        far = perception.detection_probability(spec, obs, tgt, 3000.0, env)
        self.assertGreater(near, far)
        self.assertGreater(far, 0.0)

    def test_signature_scales_probability(self):
        env = PhysicalEnv()
        obs = _observer(120.0, 23.0)
        small = _target(120.0, 23.002, unit_type="vehicle")
        small.profile = PROFILES["infantry"]
        small.size_class = "small"
        large = _target(120.0, 23.002)
        large.profile = PROFILES["tracked"]
        large.size_class = "large"
        spec = _spec(obs)
        p_small = perception.detection_probability(
            spec, obs, small, 1000.0, env)
        p_large = perception.detection_probability(
            spec, obs, large, 1000.0, env)
        self.assertGreater(p_large, p_small)

    def test_visibility_reduces_probability(self):
        clear = PhysicalEnv(visibility=20000.0)
        hazy = PhysicalEnv(visibility=500.0)
        obs = _observer(120.0, 23.0)
        obs.profile = PROFILES["tracked"]
        tgt = _target(120.0, 23.002)
        spec = _spec(obs)
        p_clear = perception.detection_probability(spec, obs, tgt, 2000.0,
                                                   clear)
        p_hazy = perception.detection_probability(spec, obs, tgt, 2000.0,
                                                  hazy)
        self.assertLess(p_hazy, p_clear)

    def test_ship_heights_detect_over_sea(self):
        # Seabed elevation is negative; ships must use surface + mast.
        env = PhysicalEnv()
        obs = Entity(1, "DDG", 0, "surface", "ship", Pose(120.0, 23.0, -50.0))
        obs.sensors = [Sensor("radar", 50.0, 1.0)]
        ship = Entity(2, "Boat", 1, "surface", "ship", Pose(120.0, 23.05, -40.0))
        spec = perception.specs_for(obs)[0]
        self.assertGreater(perception.detection_probability(
            spec, obs, ship, 5000.0, env), 0.0)

    def test_domain_gating(self):
        env = PhysicalEnv()
        obs = _observer(120.0, 23.0)
        air = Entity(2, "Air", 1, "air", "aircraft", Pose(120.0, 23.002, 800.0))
        sub = Entity(3, "Sub", 1, "subsurface", "sub", Pose(120.0, 23.002, -30.0))
        sonar = _spec_of_kind(obs, "sonar")
        optics = _spec(obs, "optics")
        self.assertEqual(perception.detection_probability(
            sonar, obs, air, 500.0, env), 0.0)
        self.assertEqual(perception.detection_probability(
            optics, obs, sub, 500.0, env), 0.0)
        radar = _spec_of_kind(obs, "radar")
        self.assertEqual(perception.detection_probability(
            radar, obs, sub, 500.0, env), 0.0)

    def test_zero_distance_still_detects(self):
        env = PhysicalEnv()
        obs = _observer(120.0, 23.0)
        tgt = _target(120.0, 23.0)
        self.assertGreater(perception.detection_probability(
            _spec(obs), obs, tgt, 0.0, env), 0.0)

    def test_building_exception_fails_closed(self):
        class Broken(PhysicalEnv):
            def building_intersects(self, a, h0, b, h1):
                raise RuntimeError("broken footprints")
        env = Broken(building=(119.9, 23.0, 120.1, 23.1, 10.0))
        obs = _observer(120.0, 23.0)
        tgt = _target(120.0, 23.002)
        self.assertEqual(perception.detection_probability(
            _spec(obs), obs, tgt, 220.0, env), 0.0)

    def test_narrow_ridge_at_5km_is_seen(self):
        # 100 m wide, 150 m high ridge between observer and target.
        env = PhysicalEnv(ridge=(23.0225, 23.0235, 150.0))
        obs = _observer(120.0, 23.0)
        tgt = _target(120.0, 23.045)
        self.assertEqual(perception.detection_probability(
            _spec(obs), obs, tgt, 5000.0, env), 0.0)

    def test_smoke_time_window_and_esm(self):
        env = PhysicalEnv()
        obs = _observer(120.0, 23.0)
        obs.sensors = [Sensor("esm", 50.0, 1.0, True)]
        tgt = _target(120.0, 23.002)
        spec = perception.specs_for(obs)[0]
        smoke = [{"lon": 120.0, "lat": 23.001, "radius_m": 300.0,
                  "top_m": 100.0, "from_s": 100.0, "to_s": 200.0,
                  "attenuation": 0.0}]
        self.assertGreater(perception.detection_probability(
            spec, obs, tgt, 220.0, env, now_s=50.0, obscurants=smoke), 0.0)
        self.assertGreater(perception.detection_probability(
            spec, obs, tgt, 220.0, env, now_s=150.0, obscurants=smoke), 0.0)

    def test_platform_sensor_fallback_and_override(self):
        plain = _observer(120.0, 23.0)
        specs = perception.specs_for(plain)
        record = perception._record_for(plain)
        expected = record["sensors"][0]["range_m"]
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].kind, "optics")
        self.assertEqual(specs[0].range_m, expected)
        radar = _observer(120.0, 23.0)
        radar.sensors = [Sensor("radar", 50.0, 1.0)]
        specs = perception.specs_for(radar)
        self.assertEqual(specs[0].kind, "radar")
        self.assertEqual(specs[0].range_m, 50000.0)
        self.assertEqual(specs[0].fov_deg, 360.0)


class WorldPerceptionTest(unittest.TestCase):
    def _world_with_pair(self, env=None):
        env = env or PhysicalEnv()
        world = _world(env)
        obs = _observer(120.0, 23.0)
        tgt = _target(120.0, 23.001)  # ~110 m north
        world.add_entity(obs)
        world.add_entity(tgt)
        return world, obs, tgt

    def test_seen_and_tracks(self):
        world, obs, tgt = self._world_with_pair()
        seen = perception.compute_seen(world)
        self.assertIn(tgt.uid, seen[obs.uid])
        tracks = world.tracks_for(0)
        self.assertTrue(any(t["uid"] == tgt.uid for t in tracks))
        self.assertTrue(tracks[0]["sensors"])

    def test_hold_expires_without_renewal(self):
        world, obs, tgt = self._world_with_pair()
        original = perception.detection_roll
        perception.detection_roll = lambda *args: 0.999
        try:
            world._phys_last_seen[(obs.uid, tgt.uid)] = world.time_s
            seen = perception.compute_seen(world)
            self.assertIn(tgt.uid, seen[obs.uid])   # held
            world.time_s += 61.0
            seen = perception.compute_seen(world)
            self.assertNotIn(tgt.uid, seen[obs.uid])
            world.time_s += 10.0
            seen = perception.compute_seen(world)
            self.assertNotIn(tgt.uid, seen[obs.uid])  # not renewed
        finally:
            perception.detection_roll = original

    def test_geometry_changes_detection(self):
        env = PhysicalEnv(building=(119.9995, 23.0002, 120.0005, 23.0004,
                                    40.0))
        world, obs, tgt = self._world_with_pair(env)
        seen = perception.compute_seen(world)
        self.assertNotIn(tgt.uid, seen[obs.uid])

    def test_deterministic_across_worlds(self):
        a, obs_a, tgt_a = self._world_with_pair()
        b, obs_b, tgt_b = self._world_with_pair()
        seen_a = perception.compute_seen(a)
        seen_b = perception.compute_seen(b)
        self.assertEqual(seen_a, seen_b)

    def test_tracks_expire(self):
        world, obs, tgt = self._world_with_pair()
        perception.compute_seen(world)
        self.assertTrue(world.tracks_for(0))
        tgt.pose = Pose(121.0, 24.0)  # far out of range
        world.time_s += 400.0
        perception.compute_seen(world)
        self.assertEqual(world.tracks_for(0), [])

    def test_world_step_physical_smoke(self):
        world, obs, tgt = self._world_with_pair()
        world.obscurants = [{"lon": 120.0, "lat": 23.0005, "radius_m": 300.0,
                             "top_m": 100.0, "from_s": 0.0, "to_s": 1e18,
                             "attenuation": 0.0}]
        original = perception.detection_roll
        perception.detection_roll = lambda *args: 0.0
        try:
            for _ in range(3):
                world.step()
            self.assertNotIn(tgt.uid, world.seen.get(obs.uid, set()))
            world.obscurants = []
            world.time_s = max(world.time_s, world.scan_interval_s)
            world.step()
            self.assertIn(tgt.uid, world.seen.get(obs.uid, set()))
        finally:
            perception.detection_roll = original


if __name__ == "__main__":
    unittest.main()
