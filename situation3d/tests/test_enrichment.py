"""Envpack building-height enrichment from a footprint resource."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from resources import adapters  # noqa: E402
from resources.enrichment import enrich_envpack  # noqa: E402
from resources.scanners import scan_envpack  # noqa: E402
from resources.store import ResourceStore  # noqa: E402
from resources.vectorproc import DatasetSpec, write_vector  # noqa: E402

RING1 = [[120.000, 22.900], [120.002, 22.900], [120.002, 22.902],
         [120.000, 22.902], [120.000, 22.900]]
RING2 = [[120.010, 22.910], [120.012, 22.910], [120.012, 22.912],
         [120.010, 22.912], [120.010, 22.910]]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def make_envpack(root):
    write_json(os.path.join(root, "manifest.json"), {
        "schema": "msgeo.envpack.v2", "pack_id": "p1", "theatre": "tw",
        "built": "2026-10-01", "bbox": [120.0, 22.9, 120.02, 22.92],
        "grid": {"center_lon": 120.0, "center_lat": 22.9, "cols": 4,
                 "rows": 4, "width_m": 2000.0},
        "layers": {"buildings": {"format": "polygons-json",
                                 "path": "buildings/buildings.json"}},
        "fallback_basic_json": None,
        "provenance": {"mode": "offline", "osm": "offline-cache"},
    })
    write_json(os.path.join(root, "buildings", "buildings.json"), [
        {"ring": RING1, "height_m": 6.0, "height_source": "default"},
        {"ring": RING2, "height_m": 6.0, "height_source": "default"},
    ])
    return root


def feature(ring, height, fid):
    return {"type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"id": fid, "height_m": height, "source": "gba"}}


class EnrichmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="enrich-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))
        self.pack = make_envpack(os.path.join(self.tmp, "pack"))
        self.env_entry = self.store.register(scan_envpack(self.pack))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fill_default_heights(self):
        root = os.path.join(self.tmp, "buildings")
        spec = DatasetSpec(name="test-buildings",
                           bbox=(120.0, 22.9, 120.02, 22.92),
                           kind="building_footprints")
        features = [feature(RING1, 20.0, "a"), feature(RING2, 31.0, "b")]
        stats = {"feature_count": 2, "height_coverage": 1.0,
                 "height_mean_m": 25.5}
        write_vector(root, features, spec, {"label": "test"}, stats)
        building_entry = adapters.register_dataset(
            self.store, spec, root, {"label": "test"}, "test",
            provider="adapter:test")
        result = enrich_envpack(self.store, self.env_entry, building_entry,
                                cache_root=os.path.join(self.tmp, "out"))
        self.assertEqual(result["stats"]["filled"], 2)
        new_entry = result["entry"]
        with open(os.path.join(new_entry.path, "buildings", "buildings.json"),
                  encoding="utf-8") as fh:
            buildings = json.load(fh)
        self.assertEqual([b["height_m"] for b in buildings], [20.0, 31.0])
        self.assertEqual(buildings[0]["height_source"], "gbd_overlap")
        # Original pack is untouched.
        with open(os.path.join(self.pack, "buildings", "buildings.json"),
                  encoding="utf-8") as fh:
            old = json.load(fh)
        self.assertEqual([b["height_m"] for b in old], [6.0, 6.0])
        self.assertTrue(self.store.verify(new_entry.entry_id)["ok"])
        self.assertEqual(new_entry.notes["enriched_from"],
                         self.env_entry.entry_id)


if __name__ == "__main__":
    unittest.main()
