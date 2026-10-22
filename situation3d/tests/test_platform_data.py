"""Platform data v1 (P0.3): schema validation and profile reconstruction.

Run from ``situation3d/``:
    python3 -m unittest tests/test_platform_data.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from mobility import platform_data as pd  # noqa: E402
from mobility.profile import (  # noqa: E402
    PROFILES, profile_for, profile_key_for)


class PlatformDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = pd.load_platforms()

    def test_all_platforms_present_and_valid(self):
        self.assertEqual(sorted(self.registry), sorted(PROFILES))
        for key, record in self.registry.items():
            self.assertEqual(record["doc"]["schema"], pd.SCHEMA)
            self.assertEqual(record["doc"]["key"], key)
            self.assertTrue(record["doc"]["provenance"])

    def test_json_rebuilds_builtin_profiles(self):
        for key, builtin in PROFILES.items():
            rebuilt = self.registry[key]["profile"]
            for field in dataclasses.fields(builtin):
                self.assertEqual(
                    getattr(rebuilt, field.name), getattr(builtin, field.name),
                    "%s.%s differs between JSON and constants"
                    % (key, field.name))

    def test_profile_for_prefers_data(self):
        cases = [
            ("vehicle", "land", "tracked"),
            ("infantry", "land", "infantry"),
            ("artillery", "land", "artillery"),
            ("amphibious", "amphibious", "amphibious"),
            ("ship", "sea", "ship"),
            ("sub", "sub", "submarine"),
            ("helicopter", "air", "helicopter"),
            ("uav", "air", "uav"),
            ("aircraft", "air", "aircraft"),
            ("wheeled", "land", "wheeled"),
        ]
        for unit_type, domain, key in cases:
            self.assertEqual(profile_key_for(unit_type, domain), key)
            prof = profile_for(unit_type, domain)
            self.assertEqual(prof, PROFILES[key])
            self.assertEqual(prof.key, key)

    def test_metadata_is_carried(self):
        doc = pd.record_by_key("tracked")
        self.assertEqual(doc["mass"]["combat_t"], 40.0)
        self.assertEqual(doc["armor"]["faces_mm_rha"]["front"], 500)
        self.assertTrue(doc["sensors"])
        self.assertTrue(doc["weapons"])
        self.assertGreater(doc["signature"]["rcs_m2"], 0.0)
        self.assertEqual(doc["motion"]["speed_curve"][-1][0], 10.0)

    def test_profile_for_prefers_data_not_constants(self):
        import mobility.platform_data as module
        sentinel = dataclasses.replace(PROFILES["tracked"], max_speed_ms=1.23)
        original = module.profile_by_key
        module.profile_by_key = (
            lambda key: sentinel if key == "tracked" else None)
        try:
            self.assertIs(profile_for("vehicle", "land"), sentinel)
        finally:
            module.profile_by_key = original

    def test_invalid_records_raise(self):
        base = pd.record_by_key("tracked")
        cases = []
        bad = dict(base)
        bad["schema"] = "msplatform.v0"
        cases.append(bad)
        bad = dict(base)
        bad["dimensions"] = {"length_m": 7.0}
        cases.append(bad)
        bad = dict(base)
        bad["identity"] = dict(base["identity"], domain="space")
        cases.append(bad)
        bad = dict(base)
        bad["mass"] = {"combat_t": -5}
        cases.append(bad)
        bad = dict(base)
        bad["motion"] = dict(base["motion"],
                             speed_curve=[[10, 1.0], [5, 1.0]])
        cases.append(bad)
        for doc in cases:
            with self.assertRaises(pd.PlatformDataError):
                pd.validate_platform(doc)

    def test_non_object_blocks_rejected(self):
        import json
        base = pd.record_by_key("tracked")
        cases = []
        for field, value in (("identity", ["land"]), ("motion", "fast"),
                             ("terrain", 5), ("sea", "deep"), ("mass", [1]),
                             ("dimensions", None), ("provenance", [])):
            doc = json.loads(json.dumps(base))
            doc[field] = value
            cases.append(doc)
        for doc in cases:
            with self.assertRaises(pd.PlatformDataError):
                pd.validate_platform(doc)

    def test_nan_values_rejected(self):
        import copy
        base = copy.deepcopy(pd.record_by_key("tracked"))
        base["motion"]["max_speed_ms"] = float("nan")
        with self.assertRaises(pd.PlatformDataError):
            pd.validate_platform(base)
        base = copy.deepcopy(pd.record_by_key("tracked"))
        base["motion"]["terrain_factors"] = {"open": float("inf")}
        with self.assertRaises(pd.PlatformDataError):
            pd.validate_platform(base)

    def test_provenance_requires_source_and_confidence(self):
        import copy
        base = copy.deepcopy(pd.record_by_key("tracked"))
        del base["provenance"]["confidence"]
        with self.assertRaises(pd.PlatformDataError):
            pd.validate_platform(base)

    def test_bad_registry_falls_back_to_constants(self):
        import tempfile
        import warnings
        old_dir = pd.PLATFORM_DIR
        old_reg = pd._REGISTRY
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "broken.json"), "w",
                      encoding="utf-8") as fh:
                fh.write("{not json")
            pd.PLATFORM_DIR = tmp
            pd._REGISTRY = None
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.assertIsNone(pd.profile_by_key("tracked"))
                    self.assertEqual(profile_for("vehicle", "land"),
                                     PROFILES["tracked"])
                    self.assertTrue(caught)
            finally:
                pd.PLATFORM_DIR = old_dir
                pd._REGISTRY = old_reg

    def test_reload_registry(self):
        import json
        import tempfile
        doc = pd.record_by_key("ship")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, "ship.json"), "w",
                          encoding="utf-8") as fh:
                    json.dump(doc, fh)
                pd.reload_registry(tmp)
                self.assertEqual(sorted(pd.registry()), ["ship"])
        finally:
            pd.reload_registry()
        self.assertEqual(sorted(pd.registry()), sorted(PROFILES))

    def test_missing_directory_is_empty_registry(self):
        self.assertEqual(pd.load_platforms(os.path.join(HERE, "no-such-dir")),
                         {})

    def test_duplicate_key_rejected(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            doc = pd.record_by_key("ship")
            with open(os.path.join(tmp, "a.json"), "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            with open(os.path.join(tmp, "b.json"), "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            with self.assertRaises(pd.PlatformDataError):
                pd.load_platforms(tmp)


if __name__ == "__main__":
    unittest.main()
