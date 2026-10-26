"""Tests for the P1 v3 environment vertical slice."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from env.pack_env import load_envpack_environment
from envmodel import (SpatialQueryV3, authorize_engagement, evaluate_gates,
                      load_manifest, load_policy, open_pack, validate_route)
from envmodel.manifest import ManifestError
from envmodel.policy import PolicyError
from mobility.profile import profile_for
from sim.entity import Pose, Route
from sim.world import World

CENTER = (120.16, 23.0)
BUILDING_RING = [
    [120.1595, 22.9995],
    [120.1605, 22.9995],
    [120.1605, 23.0005],
    [120.1595, 23.0005],
    [120.1595, 22.9995],
]
WATER_RING = [
    [120.1505, 22.9905],
    [120.1525, 22.9905],
    [120.1525, 22.9925],
    [120.1505, 22.9925],
    [120.1505, 22.9905],
]
FOREST_RING = [
    [120.1640, 23.0040],
    [120.1660, 23.0040],
    [120.1660, 23.0060],
    [120.1640, 23.0060],
    [120.1640, 23.0040],
]


def _write_json(path: str, value) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False)


def make_pack(root: str) -> None:
    _write_json(os.path.join(root, "terrain", "grid.json"), {
        "lon0": 120.15, "lat0": 22.99, "lon1": 120.17, "lat1": 23.01,
        "cols": 3, "rows": 3,
        "values": [[0.0, 10.0, 20.0], [5.0, 10.0, 15.0], [10.0, 20.0, 30.0]],
    })
    _write_json(os.path.join(root, "landcover", "polygons.json"), [
        {"ring": FOREST_RING, "class": "forest"},
    ])
    _write_json(os.path.join(root, "water", "polygons.json"), [
        {"ring": WATER_RING, "name": "test-water"},
    ])
    _write_json(os.path.join(root, "buildings", "buildings.json"), [
        {"ring": BUILDING_RING, "height_m": 20.0, "height_source": "test"},
    ])
    _write_json(os.path.join(root, "roads", "graph.json"), {
        "nodes": {
            "n1": [120.15, 22.99], "n2": [120.17, 23.01],
            "n3": [120.1495, 22.9915], "n4": [120.1535, 22.9915],
        },
        "edges": [{
            "a": "n1", "b": "n2", "length_m": 3000.0,
            "class": "primary", "surface": "asphalt",
            "bridge": False, "tunnel": False, "oneway": False,
            "maxheight": None, "maxweight": "1", "width": None,
        }, {
            "a": "n3", "b": "n4", "length_m": 400.0,
            "class": "secondary", "surface": "asphalt",
            "bridge": True, "tunnel": False, "oneway": False,
            "maxheight": "5", "maxweight": None, "width": None,
        }],
    })
    manifest = {
        "schema": "msgeo.envpack.v3",
        "pack_id": "test-pack",
        "version": "0.1",
        "theatre": "test",
        "bbox": [120.15, 22.99, 120.17, 23.01],
        "crs": "EPSG:4326",
        "vertical_datum": "unknown",
        "tile_size_m": 1000.0,
        "resolution_m": 100.0,
        "ele_grade": 10.0,
        "grid": {
            "center_lon": 120.16, "center_lat": 23.0,
            "cols": 10, "rows": 10, "width_m": 200.0,
        },
        "layers": {
            "terrain": {
                "format": "grid-json", "path": "terrain/grid.json",
                "source": "test-dem", "license": "test", "confidence": "high",
            },
            "hydrography": {
                "format": "json", "path": "water/polygons.json",
                "source": "test-water", "license": "test", "confidence": "low",
                "depth_kind": "unknown",
            },
            "water": {
                "format": "json", "path": "water/polygons.json",
                "source": "test-water", "license": "test", "confidence": "low",
                "depth_kind": "unknown",
            },
            "landcover": {
                "format": "json", "path": "landcover/polygons.json",
                "source": "test-landcover", "license": "test", "confidence": "medium",
            },
            "buildings": {
                "format": "json", "path": "buildings/buildings.json",
                "source": "test-buildings", "license": "test", "confidence": "medium",
                "height_kind": "measured",
            },
            "roads": {
                "format": "json", "path": "roads/graph.json",
                "source": "test-roads", "license": "test", "confidence": "high",
            },
            "obstacles": {
                "format": "json", "path": "buildings/buildings.json",
                "source": "test-buildings", "license": "test", "confidence": "medium",
            },
        },
        "nav": {"global_res_m": 30.0, "local_res_m": 2.0, "graph_version": "test"},
        "provenance": {"built_at": "2026-10-24T00:00:00Z"},
        "license": {
            "spdx": "TEST", "redistribute": False,
            "commercial_use": False, "attribution": "test",
        },
        "fallback_basic_json": None,
    }
    _write_json(os.path.join(root, "manifest.json"), manifest)


class EnvModelTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        make_pack(self.root)
        self.manifest = load_manifest(self.root)
        self.env, _grid = load_envpack_environment(self.root)
        self.query = SpatialQueryV3(self.env, self.manifest, time_s=0.0)

    def tearDown(self):
        self._tmp.cleanup()

    def test_open_pack(self):
        query = open_pack(self.root)
        self.assertEqual(query.manifest.pack_id, "test-pack")
        self.assertEqual(query.elevation(*CENTER).quality, "measured")

    def test_manifest_v3(self):
        self.assertEqual(self.manifest.schema, "msgeo.envpack.v3")
        self.assertEqual(self.manifest.pack_id, "test-pack")
        self.assertAlmostEqual(self.manifest.bbox[0], 120.15)

    def test_manifest_rejects_wrong_schema(self):
        path = os.path.join(self.root, "manifest.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["schema"] = "msgeo.envpack.v2"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        with self.assertRaises(ManifestError):
            load_manifest(self.root)

    def test_time_binding(self):
        later = self.query.at(10.0)
        self.assertEqual(later.time_s, 10.0)
        self.assertEqual(self.query.elevation(*CENTER).as_of, 0.0)
        self.assertEqual(later.elevation(*CENTER).as_of, 10.0)

    def test_elevation_envelope(self):
        env = self.query.elevation(*CENTER)
        self.assertEqual(env.quality, "measured")
        self.assertEqual(env.source_id, "test-dem")
        self.assertGreater(env.value, 0.0)

    def test_building_and_obstacle(self):
        env = self.query.building(*CENTER)
        self.assertEqual(env.quality, "measured")
        self.assertAlmostEqual(env.value["height_m"], 20.0)
        self.assertEqual(len(self.query.obstacles().value), 1)

    def test_water_unknown_depth(self):
        water = self.query.water(120.1515, 22.9915)
        self.assertTrue(water.value["is_water"])
        self.assertIsNone(water.value["depth_m"])
        self.assertEqual(water.quality, "unknown")
        land = self.query.water(*CENTER)
        self.assertFalse(land.value["is_water"])
        self.assertEqual(land.quality, "not_applicable")

    def test_los_building(self):
        a = (120.1580, 23.0)
        b = (120.1620, 23.0)
        low = self.query.los(a, 5.0, b, 5.0)
        self.assertFalse(low.value["visible"])
        self.assertTrue(low.value["building_blocked"])
        high = self.query.los(a, 50.0, b, 50.0)
        self.assertTrue(high.value["visible"])

    def test_batch_and_road(self):
        points = [CENTER, (120.158, 23.0)]
        batch = self.query.query_batch(points, "landcover")
        self.assertEqual(len(batch), 2)
        self.assertIsNotNone(self.query.road(120.16, 23.0).value)

    def test_tile(self):
        tile = self.query.tile("0/0/0", 0)
        self.assertEqual(tile.value["path"], "terrain/grid.json")

    def test_gates_building_and_water(self):
        profile = profile_for("vehicle", "land")
        b = self.query.passability(profile, *CENTER)
        self.assertFalse(b.value["allowed"])
        self.assertIn("building", b.value["reason_codes"])
        w = self.query.passability(profile, 120.1508, 22.9908)
        self.assertFalse(w.value["allowed"])
        self.assertIn("water_depth_unknown", w.value["reason_codes"])
        self.assertTrue(w.value["unknown"])
        self.assertEqual(w.quality, "unknown")

    def test_bridge_deck_skips_water(self):
        profile = profile_for("vehicle", "land")
        # Bridge edge n3->n4 crosses the test water polygon at this point.
        b = self.query.passability(profile, 120.1515, 22.9915)
        self.assertTrue(b.value["allowed"])
        self.assertTrue(b.value["unknown"])
        self.assertIn("bridge_mlc_unknown_advisory", b.value["reason_codes"])
        self.assertEqual(len(self.query._bridges), 1)

    def test_gates_road_limit(self):
        profile = profile_for("vehicle", "land")
        r = self.query.passability(profile, 120.155, 22.995)
        self.assertFalse(r.value["allowed"])
        self.assertIn("road_maxweight", r.value["reason_codes"])

    def test_gates_landcover_matches_nav_mask(self):
        from env.service import LandCover

        class _CoverEnv(object):
            def __init__(self, cover):
                self.cover = cover

            def building_at(self, lon, lat):
                return None

            def is_water(self, lon, lat):
                return False

            def water_depth(self, lon, lat):
                return None

            def slope_deg(self, lon, lat):
                return 0.0

            def landcover(self, lon, lat):
                return self.cover

            def on_road(self, lon, lat):
                return True

            def road_edge_at(self, lon, lat, tolerance_m=25.0):
                return None

        profile = profile_for("vehicle", "land")
        denied = evaluate_gates(profile, 0.0, 0.0, _CoverEnv(LandCover.WATER),
                                depth_kind="unknown")
        self.assertFalse(denied.allowed)
        self.assertIn("landcover", denied.reason_codes)
        opened = evaluate_gates(profile, 0.0, 0.0, _CoverEnv(LandCover.OPEN),
                                depth_kind="unknown")
        self.assertTrue(opened.allowed)

    def test_gates_unknown_policy_advisory(self):
        profile = profile_for("vehicle", "land")
        r = evaluate_gates(
            profile, 120.1515, 22.9915, self.env,
            depth_kind="unknown",
            policy={"water_depth": "advisory"},
        )
        self.assertTrue(r.allowed)
        self.assertTrue(r.unknown)
        self.assertIn("water_depth_unknown_advisory", r.reason_codes)

    def test_policy_loader_and_override(self):
        policy = {
            "schema": "situation3d.policy.v1",
            "policy_id": "test-policy",
            "revision": "0.1",
            "approval_status": "approved",
            "approved_by": "test",
            "roe": "return_fire",
            "posture": "defensive",
            "authorization_matrix": {},
            "gates": {"water_depth": {"unknown_policy": "advisory"}},
        }
        path = os.path.join(self.root, "policy.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(policy, fh)
        self.assertEqual(load_policy(path)["policy_id"], "test-policy")
        query = open_pack(self.root, policy_path=path)
        profile = profile_for("vehicle", "land")
        r = query.passability(profile, 120.1508, 22.9908)
        self.assertTrue(r.value["allowed"])
        self.assertIn("water_depth_unknown_advisory", r.value["reason_codes"])

    def test_policy_requires_approval(self):
        policy = {
            "schema": "situation3d.policy.v1",
            "policy_id": "draft",
            "revision": "0.1",
            "approval_status": "draft",
            "approved_by": "test",
            "roe": "hold",
            "posture": "defensive",
            "authorization_matrix": {},
        }
        path = os.path.join(self.root, "policy.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(policy, fh)
        with self.assertRaises(PolicyError):
            load_policy(path)

    def test_authorization_matrix(self):
        policy = {
            "policy_id": "auth-test",
            "approval_status": "approved",
            "roe": "return_fire",
            "authorization_matrix": {"blue": {"land": ["cannon"]}},
        }
        ok = authorize_engagement(policy, "blue", "land", "cannon", under_fire=True)
        self.assertTrue(ok.allowed)
        self.assertEqual(ok.reason, "authorized")
        self.assertFalse(
            authorize_engagement(policy, "blue", "land", "cannon").allowed
        )
        self.assertFalse(
            authorize_engagement(policy, "blue", "land", "atgm", under_fire=True).allowed
        )
        self.assertFalse(
            authorize_engagement(policy, "red", "land", "cannon", under_fire=True).allowed
        )
        hold = dict(policy, roe="hold")
        self.assertFalse(
            authorize_engagement(hold, "blue", "land", "cannon", under_fire=True).allowed
        )

    def test_policy_raw_dict_requires_approval(self):
        with self.assertRaises(PolicyError):
            open_pack(self.root, policy={
                "policy_id": "draft",
                "approval_status": "draft",
                "roe": "hold",
                "authorization_matrix": {},
            })

    def test_surface_vessel_on_land_denied(self):
        profile = profile_for("ship", "surface")
        r = self.query.passability(profile, 120.165, 23.007)
        self.assertFalse(r.value["allowed"])
        self.assertIn("not_water", r.value["reason_codes"])

    def test_invalid_unknown_policy_denies(self):
        profile = profile_for("vehicle", "land")
        r = evaluate_gates(
            profile, 120.1508, 22.9908, self.env,
            depth_kind="unknown",
            policy={"water_depth": "bogus"},
        )
        self.assertFalse(r.allowed)
        self.assertIn("water_depth_unknown", r.reason_codes)

    def test_world_sea_air_authorization(self):
        world = World.__new__(World)
        world.time_s = 0.0
        world.weapon_ready = lambda uid, key: True
        world.policy = {
            "policy_id": "sea-auth",
            "approval_status": "approved",
            "roe": "weapons_free",
            "authorization_matrix": {"blue": {"surface": ["sam"]}},
        }
        attacker = SimpleNamespace(
            uid=2, affiliation=1, last_damaged_s=-1.0, weapons=[],
            sea_air_weapons=[
                SimpleNamespace(key="s1", weapon_id=9999, kind="sam", ammo=4)
            ],
        )
        target = SimpleNamespace(domain="surface")
        self.assertTrue(world.authorize_engagement(attacker, target))
        world.policy = {}
        self.assertFalse(world.authorize_engagement(attacker, target))

    def test_world_indirect_authorization(self):
        world = World.__new__(World)
        world.time_s = 0.0
        world.weapon_ready = lambda uid, key: True
        attacker = SimpleNamespace(
            uid=3, affiliation=1, last_damaged_s=-1.0,
            weapons=[SimpleNamespace(key="i1", weapon_id=9998,
                                     kind="indirect", ammo=2)],
            sea_air_weapons=[],
        )
        world.policy = {
            "policy_id": "indirect-auth",
            "approval_status": "approved",
            "roe": "weapons_free",
            "authorization_matrix": {"blue": {"land": ["indirect"]}},
        }
        self.assertTrue(world.authorize_indirect(attacker))
        world.policy = dict(world.policy, roe="hold")
        self.assertFalse(world.authorize_indirect(attacker))

    def test_surface_vessel_on_bridge_still_water_gated(self):
        profile = profile_for("ship", "surface")
        r = self.query.passability(profile, 120.1515, 22.9915)
        self.assertFalse(r.value["allowed"])
        self.assertIn("water_depth_unknown", r.value["reason_codes"])

    def test_route_validation_rejects_building(self):
        profile = profile_for("vehicle", "land")
        through = Route(
            points=[Pose(120.158, 23.0), Pose(120.162, 23.0)],
            speed_limits=[5.0], modes=["offroad"],
        )
        self.assertFalse(validate_route(self.query, profile, through))
        clear = Route(
            points=[Pose(120.164, 23.007), Pose(120.1655, 23.007)],
            speed_limits=[5.0], modes=["offroad"],
        )
        self.assertTrue(validate_route(self.query, profile, clear))

    def test_world_authorization_gate(self):
        world = World.__new__(World)
        world.time_s = 0.0
        world.weapon_ready = lambda uid, key: True
        world.policy = {
            "policy_id": "world-auth",
            "approval_status": "approved",
            "roe": "return_fire",
            "authorization_matrix": {"blue": {"land": ["cannon"]}},
        }
        attacker = SimpleNamespace(
            uid=1, affiliation=1, last_damaged_s=0.0,
            weapons=[SimpleNamespace(key="w1", weapon_id=36, kind="direct")],
        )
        target = SimpleNamespace(domain="land")
        self.assertTrue(world.authorize_engagement(attacker, target))
        world.policy = dict(world.policy, roe="hold")
        self.assertFalse(world.authorize_engagement(attacker, target))
        world.policy = None
        self.assertTrue(world.authorize_engagement(attacker, target))

    def test_gates_land_open(self):
        profile = profile_for("vehicle", "land")
        r = self.query.passability(profile, 120.165, 23.007)
        self.assertTrue(r.value["allowed"])
        self.assertEqual(r.value["reason_codes"], [])


if __name__ == "__main__":
    unittest.main()
