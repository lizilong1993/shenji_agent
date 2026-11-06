"""Tests for analyst-confirmed cross-source correlations."""

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

from intel.correlations import CorrelationStore, pair_key  # noqa: E402
from intel.management import _apply_verdicts, IntelManager  # noqa: E402


def _write_bundle(root, case_id, observations):
    directory = os.path.join(root, case_id)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "bundle.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"bundle_id": "b1", "scenario_id": case_id,
                   "scenario_start": "2026-10-24T00:00:00Z",
                   "synthetic": False, "hash": "abc",
                   "observations": observations}, handle)


def _obs(obs_id, source_id="usgs", lat=23.5, lon=120.5, type_="earthquake"):
    return {
        "observation_id": obs_id, "source_id": source_id,
        "source_native_id": obs_id, "domain": "hazard",
        "type": type_, "event_time": "2026-10-24T12:00:00Z",
        "ingest_time": "2026-10-24T12:05:00Z",
        "lat": lat, "lon": lon, "title": obs_id, "confidence": 0.9,
        "status": "active",
    }


class PairKeyTest(unittest.TestCase):
    def test_order_independent(self):
        self.assertEqual(pair_key("b", "a"), pair_key("a", "b"))

    def test_requires_both_ids(self):
        with self.assertRaises(ValueError):
            pair_key("", "a")


class CorrelationStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CorrelationStore(os.path.join(self.tmp.name, "corr.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_set_and_list(self):
        item = self.store.set({"case_id": "c1", "a_id": "x", "b_id": "y",
                               "verdict": "confirm", "note": "同一辆指挥车"})
        self.assertTrue(item["correlation_id"].startswith("corr:"))
        self.assertEqual(len(self.store.list("c1")), 1)
        self.assertEqual(self.store.list("c2"), [])

    def test_upsert_replaces_verdict(self):
        first = self.store.set({"case_id": "c1", "a_id": "x", "b_id": "y",
                                "verdict": "confirm"})
        second = self.store.set({"case_id": "c1", "a_id": "y", "b_id": "x",
                                 "verdict": "reject", "note": "误判"})
        # 方向无关，第二次是覆盖而非新增
        self.assertEqual(first["correlation_id"], second["correlation_id"])
        self.assertEqual(len(self.store.list("c1")), 1)
        self.assertEqual(second["verdict"], "reject")
        self.assertEqual(second["revision"], first["revision"] + 1)

    def test_validation(self):
        for bad in (
            {"case_id": "c1", "a_id": "x", "b_id": "x", "verdict": "confirm"},
            {"case_id": "c1", "a_id": "x", "b_id": "y", "verdict": "maybe"},
            {"case_id": "", "a_id": "x", "b_id": "y", "verdict": "confirm"},
        ):
            with self.assertRaises(ValueError):
                self.store.set(bad)

    def test_clear_removes_verdict(self):
        item = self.store.set({"case_id": "c1", "a_id": "x", "b_id": "y",
                               "verdict": "confirm"})
        self.assertTrue(self.store.clear(item["correlation_id"]))
        self.assertFalse(self.store.clear(item["correlation_id"]))
        self.assertEqual(self.store.list("c1"), [])

    def test_revision_changes_on_write(self):
        before = self.store.revision()
        self.store.set({"case_id": "c1", "a_id": "x", "b_id": "y",
                        "verdict": "confirm"})
        self.assertNotEqual(before, self.store.revision())


class ApplyVerdictsTest(unittest.TestCase):
    def _link(self, kind="co_location", score=0.5):
        return {"from_id": "a", "to_id": "b", "kind": kind, "score": score,
                "reasons": ["距离 10.0 km"]}

    def test_confirm_promotes_to_manual(self):
        out = _apply_verdicts([self._link()], {
            "a|b": {"verdict": "confirm", "note": "同一目标",
                    "correlation_id": "corr:1", "updated_at": "t"},
        })
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["kind"], "manual")
        self.assertEqual(out[0]["score"], 0.99)
        self.assertTrue(any("人工确认" in r for r in out[0]["reasons"]))
        self.assertEqual(out[0]["correlation_id"], "corr:1")

    def test_reject_drops_link(self):
        out = _apply_verdicts([self._link()], {"a|b": {"verdict": "reject"}})
        self.assertEqual(out, [])

    def test_unrelated_links_untouched(self):
        out = _apply_verdicts([self._link()], {"x|y": {"verdict": "reject"}})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["kind"], "co_location")

    def test_empty_verdicts_is_a_passthrough(self):
        links = [self._link()]
        self.assertIs(_apply_verdicts(links, {}), links)


class OverviewCorrelationTest(unittest.TestCase):
    def test_reject_removes_link_from_overview(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [_obs("usgs:1"), _obs("gdacs:2",
                                                            source_id="gdacs")])
            manager = IntelManager(root)
            before = manager.overview("c1")["links"]
            self.assertTrue(before, "两个近距离同类型条目应产生链接")
            pair = before[0]
            manager.correlations.set({
                "case_id": "c1", "a_id": pair["from_id"],
                "b_id": pair["to_id"], "verdict": "reject",
            })
            after = manager.overview("c1")
            self.assertEqual(after["links"], [])
            self.assertEqual(len(after["correlations"]["items"]), 1)

    def test_confirm_marks_link_manual(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [_obs("usgs:1"), _obs("gdacs:2",
                                                            source_id="gdacs")])
            manager = IntelManager(root)
            pair = manager.overview("c1")["links"][0]
            manager.correlations.set({
                "case_id": "c1", "a_id": pair["from_id"],
                "b_id": pair["to_id"], "verdict": "confirm",
                "note": "影像已确认",
            })
            links = manager.overview("c1")["links"]
            self.assertEqual([link["kind"] for link in links], ["manual"])
            self.assertEqual(links[0]["score"], 0.99)

    def test_verdict_invalidates_cached_overview(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "c1", [_obs("usgs:1"), _obs("gdacs:2",
                                                            source_id="gdacs")])
            manager = IntelManager(root)
            first = manager.overview("c1")["links"]
            pair = first[0]
            manager.correlations.set({
                "case_id": "c1", "a_id": pair["from_id"],
                "b_id": pair["to_id"], "verdict": "reject",
            })
            # 缓存键含 correlations.revision()：不重建就会返回旧的 links
            self.assertEqual(manager.overview("c1")["links"], [])


if __name__ == "__main__":
    unittest.main()
