"""Tests for the P1 v3 environment vertical slice."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from env.pack_env import load_envpack_environment
from envmodel import SpatialQueryV3, load_manifest, open_pack
from envmodel.manifest import ManifestError

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
        "nodes": {"n1": [120.15, 22.99], "n2": [120.17, 23.01]},
        "edges": [{
            "a": "n1", "b": "n2", "length_m": 3000.0,
            "class": "primary", "surface": "asphalt",
            "bridge": False, "tunnel": False, "oneway": False,
            "maxheight": None, "maxweight": None, "width": None,
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


if __name__ == "__main__":
    unittest.main()
