"""Online envpack refresh and provided-file registration tests (P3)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from resources import providers  # noqa: E402
from resources.scanners import scan_envpack  # noqa: E402
from resources.store import ResourceStore  # noqa: E402

GRID = {"center_lon": 120.1, "center_lat": 22.9, "cols": 4, "rows": 4,
        "width_m": 2000.0}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


def make_basic():
    cells = []
    for r in range(GRID["rows"]):
        row = []
        for c in range(GRID["cols"]):
            row.append({"pos": r * 10 + c, "elev": 10, "cond": 0,
                        "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10.0, "grid": dict(GRID),
            "map_data": cells}


def make_envpack(root):
    basic = make_basic()
    write_json(os.path.join(root, "base_map.json"), basic)
    write_json(os.path.join(root, "terrain", "grid.json"), {"values": [[1]]})
    write_json(os.path.join(root, "roads", "graph.json"),
               {"nodes": {}, "edges": []})
    write_json(os.path.join(root, "manifest.json"), {
        "schema": "msgeo.envpack.v2", "pack_id": "p1", "theatre": "tw",
        "built": "2026-10-01", "bbox": [120.0, 22.8, 120.2, 23.0],
        "grid": dict(GRID), "resolution_m": 100.0, "dem_zoom": 10,
        "fallback_basic_json": "base_map.json",
        "layers": {
            "terrain": {"format": "grid-json", "path": "terrain/grid.json",
                        "source": "fallback-basic"},
            "roads": {"format": "graph-json", "path": "roads/graph.json",
                      "source": "fallback-basic"},
        },
        "provenance": {"mode": "offline", "fallback": True},
    })
    return root


FEATURES = {
    "roads": [{"coords": [[120.05, 22.85], [120.1, 22.9]],
               "class": "primary", "bridge": False, "tunnel": False}],
    "water": [{"ring": [[120.05, 22.85], [120.06, 22.85],
                        [120.06, 22.86], [120.05, 22.85]]}],
    "buildings": [{"ring": [[120.09, 22.89], [120.10, 22.89],
                            [120.10, 22.90], [120.09, 22.89]],
                   "height": 12.0}],
    "landuse": [{"ring": [[120.07, 22.87], [120.08, 22.87],
                          [120.08, 22.88], [120.07, 22.87]],
                 "kind": "residential"}],
}


class FakeOsm(object):
    fetched_network = True
    provenance_label = "fake-osm"

    def __init__(self, features=None):
        self.features_data = features if features is not None else FEATURES

    def features(self, bbox):
        return self.features_data


class FakeDem(object):
    fetched_network = True
    provenance_label = "fake-dem"

    def sample_grid(self, bbox, step_deg):
        return {"lon0": bbox[0], "lat0": bbox[1], "lon1": bbox[2],
                "lat1": bbox[3], "cols": 2, "rows": 2,
                "values": [[10.0, 10.0], [10.0, 10.0]]}


EMPTY = {"roads": [], "water": [], "buildings": [], "landuse": []}


class RefreshTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="res-fetch-")
        self.pack = make_envpack(os.path.join(self.tmp, "pack"))
        self.store = ResourceStore(os.path.join(self.tmp, "index"))
        self.entry = self.store.register(scan_envpack(self.pack))
        self.cache = os.path.join(self.tmp, "cache")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_refresh_registers_versioned_entry(self):
        new = providers.refresh_envpack(
            self.store, self.entry, self.cache,
            osm_source=FakeOsm(), dem_source=FakeDem())
        self.assertTrue(new.entry_id.startswith("envpack:p1@"))
        self.assertIn("/envpacks/", new.path.replace(os.sep, "/"))
        self.assertEqual(new.provider, "online-refresh")
        self.assertEqual(new.quality, "measured")
        self.assertEqual(new.as_of,
                         datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        self.assertEqual(new.notes.get("refreshed_from"), self.entry.entry_id)
        self.assertEqual(len(self.store.entries("envpack")), 2)
        # The old entry remains pinned to its original path/hash.
        old = self.store.get(self.entry.entry_id)
        self.assertEqual(old.content_hash, self.entry.content_hash)

    def test_incomplete_online_fetch_is_rejected(self):
        with self.assertRaises(providers.ProviderError) as ctx:
            providers.refresh_envpack(self.store, self.entry, self.cache,
                                      osm_source=FakeOsm(EMPTY),
                                      dem_source=FakeDem())
        self.assertIn("fallback-basic", str(ctx.exception))
        self.assertEqual(len(self.store.entries("envpack")), 1)

    def test_allow_degraded_registers_derived_pack(self):
        new = providers.refresh_envpack(
            self.store, self.entry, self.cache, allow_degraded=True,
            osm_source=FakeOsm(EMPTY), dem_source=FakeDem())
        self.assertEqual(new.quality, "derived")

    def test_provide_file_marks_manual_override(self):
        entry = providers.register_provided_envpack(
            self.store, self.pack, entry_id="envpack:manual")
        self.assertEqual(entry.entry_id, "envpack:manual")
        self.assertTrue(entry.notes.get("manual_override"))
        self.assertEqual(entry.provider, "provided-file")

    def test_same_day_refresh_is_reused(self):
        first = providers.refresh_envpack(
            self.store, self.entry, self.cache,
            osm_source=FakeOsm(), dem_source=FakeDem())
        second = providers.refresh_envpack(
            self.store, self.entry, self.cache,
            osm_source=FakeOsm(EMPTY), dem_source=FakeDem())
        self.assertEqual(first.entry_id, second.entry_id)
        self.assertEqual(len(self.store.entries("envpack")), 2)

    def test_provide_path_allowlist(self):
        with self.assertRaises(providers.ProviderError):
            providers.register_provided_envpack(
                self.store, self.pack, entry_id="envpack:blocked",
                allowed_roots=[self.cache])
        entry = providers.register_provided_envpack(
            self.store, self.pack, entry_id="envpack:allowed",
            allowed_roots=[os.path.dirname(self.pack)])
        self.assertEqual(entry.entry_id, "envpack:allowed")

    def test_probe_failure(self):
        self.assertFalse(providers.probe_online("127.0.0.1", 9, 0.2))


if __name__ == "__main__":
    unittest.main()
