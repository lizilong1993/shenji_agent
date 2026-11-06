"""Tests for exposing per-item location uncertainty and freshness."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.management import IntelManager  # noqa: E402


def _write_bundle(root, case_id, observations):
    directory = os.path.join(root, case_id)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "bundle.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"bundle_id": "b1", "scenario_id": case_id,
                   "scenario_start": "2026-10-24T00:00:00Z",
                   "synthetic": False, "hash": "abc",
                   "observations": observations}, handle)


def _obs(obs_id="usgs:1", **extra):
    payload = {
        "observation_id": obs_id, "source_id": "usgs",
        "source_native_id": obs_id, "domain": "hazard",
        "type": "earthquake", "event_time": "2026-10-24T12:00:00Z",
        "ingest_time": "2026-10-24T12:05:00Z",
        "lat": 23.5, "lon": 120.5, "title": obs_id, "confidence": 0.9,
        "status": "active",
    }
    payload.update(extra)
    return payload


class LocationUncertaintyTest(unittest.TestCase):
    def test_uncertainty_and_centroid_are_exposed(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [
                _obs("usgs:1", geo_uncertainty_m=50000.0, is_centroid=True),
            ])
            items = IntelManager(root).overview("c1")["items"]
            self.assertEqual(items[0]["geo_uncertainty_m"], 50000.0)
            self.assertTrue(items[0]["is_centroid"])

    def test_missing_uncertainty_is_zero_not_none(self):
        # 缺值必须是 0：前端拿 None 算 ±X km 会得到 NaN。
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [_obs("usgs:1")])
            item = IntelManager(root).overview("c1")["items"][0]
            self.assertEqual(item["geo_uncertainty_m"], 0.0)
            self.assertFalse(item["is_centroid"])

    def test_bad_uncertainty_values_fall_back_to_zero(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [
                _obs("usgs:1", geo_uncertainty_m="not-a-number"),
                _obs("usgs:2", geo_uncertainty_m=-5.0),
                _obs("usgs:3", geo_uncertainty_m=float("nan")),
            ])
            items = {row["id"]: row
                     for row in IntelManager(root).overview("c1")["items"]}
            for obs_id in ("usgs:1", "usgs:2", "usgs:3"):
                self.assertEqual(items[obs_id]["geo_uncertainty_m"], 0.0,
                                 obs_id)

    def test_ingest_time_is_exposed_for_freshness(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [_obs("usgs:1")])
            item = IntelManager(root).overview("c1")["items"][0]
            self.assertEqual(item["ingest_time"], "2026-10-24T12:05:00Z")

    def test_manual_items_carry_the_same_fields(self):
        with tempfile.TemporaryDirectory() as root:
            manager = IntelManager(root)
            manager.manual.create({
                "case_id": "c1", "title": "人工", "domain": "hazard",
                "intel_type": "earthquake",
                "event_time": "2026-10-24T12:10:00Z",
                "lat": 23.5, "lon": 120.5, "place": "港口",
                "content": "观察员报告",
            })
            item = [row for row in manager.overview("c1")["items"]
                    if row["priority"] == "manual"][0]
            self.assertEqual(item["geo_uncertainty_m"], 0.0)
            self.assertFalse(item["is_centroid"])
            self.assertIn("ingest_time", item)


if __name__ == "__main__":
    unittest.main()
