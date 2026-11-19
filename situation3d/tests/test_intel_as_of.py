"""历史时点查询（只读筛选）——口径见 goals/intel_asof_query_design.md。

不传 ``as_of`` 时返回体逐键不变（除新增的四个固定字段）；
传 ``as_of`` 时只筛 ``items``，``links`` 在筛选后集合上重算。
"""

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

from intel.management import IntelManager, InvalidAsOf  # noqa: E402


def _write_bundle(root, case_id, observations):
    directory = os.path.join(root, case_id)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "bundle.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"bundle_id": "b1", "scenario_id": case_id,
                   "scenario_start": "2026-10-24T00:00:00Z",
                   "synthetic": False, "hash": "abc",
                   "observations": observations}, handle)


def _obs(obs_id, event_time="2026-10-24T12:00:00Z",
         ingest_time="2026-10-24T12:05:00Z", lat=23.5, lon=120.5,
         type_="earthquake"):
    record = {"observation_id": obs_id, "source_id": "usgs",
              "source_native_id": obs_id, "domain": "hazard", "type": type_,
              "event_time": event_time, "lat": lat, "lon": lon,
              "title": "自动地震", "confidence": 0.9, "status": "active"}
    if ingest_time is not None:
        record["ingest_time"] = ingest_time
    return record


class AsOfQueryTest(unittest.TestCase):
    def test_two_record_kinds_are_filtered(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [
                _obs("usgs:in_time_before", "2026-10-24T11:00:00Z",
                     "2026-10-24T11:05:00Z"),
                _obs("usgs:in_time_late", "2026-10-24T11:30:00Z",
                     "2026-10-24T12:30:00Z"),
                _obs("usgs:event_after", "2026-10-24T13:00:00Z",
                     "2026-10-24T12:10:00Z"),
            ])
            view = IntelManager(root).overview("c1", as_of="2026-10-24T12:00:00Z")
            self.assertEqual([row["id"] for row in view["items"]],
                             ["usgs:in_time_before"])
            self.assertEqual(view["late_records_skipped"], 1)
            self.assertTrue(view["as_of_applied"])

    def test_missing_ingest_time_is_excluded_and_named(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [
                _obs("usgs:old", "2026-10-24T11:00:00Z", ingest_time=None)])
            view = IntelManager(root).overview("c1", as_of="2026-10-24T12:00:00Z")
            self.assertEqual(view["items"], [])
            self.assertTrue(any("ingest_time" in warning
                                for warning in view["warnings"]))

    def test_same_as_of_twice_does_not_hit_the_current_view(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [
                _obs("usgs:a", "2026-10-24T11:00:00Z", "2026-10-24T11:05:00Z"),
                _obs("usgs:b", "2026-10-24T11:20:00Z", "2026-10-24T11:25:00Z"),
            ])
            manager = IntelManager(root)
            current = manager.overview("c1")
            first = manager.overview("c1", as_of="2026-10-24T11:10:00Z")
            second = manager.overview("c1", as_of="2026-10-24T11:10:00Z")
            self.assertEqual(len(current["items"]), 2)
            self.assertEqual([row["id"] for row in first["items"]], ["usgs:a"])
            self.assertEqual(first["links"], second["links"])
            self.assertEqual(len(second["items"]), 1)

    def test_manual_created_after_reference_is_hidden(self):
        with tempfile.TemporaryDirectory() as root:
            manager = IntelManager(root)
            manager.manual.create({"case_id": "c1", "title": "人工记录",
                                   "domain": "hazard",
                                   "intel_type": "earthquake",
                                   "event_time": "2026-10-24T12:10:00Z",
                                   "lat": 23.5, "lon": 120.5,
                                   "content": "观察员报告"})
            view = manager.overview("c1", as_of="2026-10-24T12:00:00Z")
            self.assertEqual(view["counts"]["manual_active"], 0)
            now = manager.overview("c1")
            self.assertEqual(now["counts"]["manual_active"], 1)

    def test_no_as_of_keeps_legacy_result(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [_obs("usgs:a")])
            manager = IntelManager(root)
            legacy = manager.overview("c1")
            self.assertFalse(legacy["as_of_applied"])
            self.assertIsNone(legacy["as_of"])
            self.assertEqual(legacy["late_records_skipped"], 0)
            self.assertFalse(legacy["manual_status_not_historical"])
            self.assertEqual(len(legacy["items"]), 1)

    def test_bad_and_future_as_of_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            manager = IntelManager(root)
            with self.assertRaises(InvalidAsOf):
                manager.overview("c1", as_of="yesterday")
            with self.assertRaises(InvalidAsOf):
                manager.overview("c1", as_of="2099-01-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
