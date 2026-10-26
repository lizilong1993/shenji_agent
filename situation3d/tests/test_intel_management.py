"""Tests for manual intel, source registry, priority and association."""

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
        json.dump({
            "bundle_id": "b1",
            "scenario_id": case_id,
            "scenario_start": "2026-10-24T00:00:00Z",
            "synthetic": False,
            "hash": "abc",
            "observations": observations,
        }, handle)


def _auto(obs_id="usgs:auto1", lat=23.5, lon=120.5,
          event_time="2026-10-24T12:00:00Z", type_="earthquake"):
    return {
        "observation_id": obs_id,
        "source_id": "usgs",
        "source_native_id": obs_id,
        "domain": "hazard",
        "type": type_,
        "event_time": event_time,
        "ingest_time": "2026-10-24T12:05:00Z",
        "lat": lat,
        "lon": lon,
        "title": "自动地震",
        "confidence": 0.91,
        "status": "active",
        "license": "U.S. public domain",
        "attribution": "USGS",
    }


def _manual_payload(case_id="case1", **overrides):
    payload = {
        "case_id": case_id,
        "title": "人工核实：港口附近爆炸",
        "domain": "hazard",
        "intel_type": "earthquake",
        "event_time": "2026-10-24T12:10:00Z",
        "lat": 23.501,
        "lon": 120.501,
        "place": "港口",
        "source_id": "manual",
        "source_name": "人工录入",
        "content": "观察员报告港口附近发生爆炸，等待核实。",
        "verification": "unverified",
        "tags": ["港口", "爆炸"],
    }
    payload.update(overrides)
    return payload


class ManualStoreTest(unittest.TestCase):
    def test_create_update_revoke_and_persist(self):
        with tempfile.TemporaryDirectory() as root:
            manager = IntelManager(root)
            item = manager.manual.create(_manual_payload())
            self.assertTrue(item["intel_id"].startswith("manual:"))
            self.assertEqual(item["priority"], "manual")
            self.assertEqual(item["verification"], "unverified")
            self.assertEqual(item["lat"], 23.501)
            self.assertEqual(item["lon"], 120.501)
            updated = manager.manual.update(
                item["intel_id"], {"title": "人工核实：港口爆炸",
                                   "verification": "verified"})
            self.assertEqual(updated["revision"], 2)
            self.assertEqual(updated["verification"], "verified")
            again = IntelManager(root).manual.get(item["intel_id"])
            self.assertEqual(again["title"], "人工核实：港口爆炸")
            revoked = manager.manual.revoke(item["intel_id"])
            self.assertEqual(revoked["status"], "revoked")
            self.assertEqual(revoked["revision"], 3)

    def test_validation(self):
        with tempfile.TemporaryDirectory() as root:
            manager = IntelManager(root)
            with self.assertRaises(ValueError):
                manager.manual.create(_manual_payload(title=""))
            with self.assertRaises(ValueError):
                manager.manual.create(_manual_payload(lat=99))
            with self.assertRaises(ValueError):
                manager.manual.create(_manual_payload(event_time="yesterday"))
            with self.assertRaises(ValueError):
                manager.manual.create(_manual_payload(case_id="bad case"))
            with self.assertRaises(ValueError):
                manager.manual.create(_manual_payload(domain="military"))


class SourceStoreTest(unittest.TestCase):
    def test_builtins_add_search_delete(self):
        with tempfile.TemporaryDirectory() as root:
            manager = IntelManager(root)
            sources = manager.sources.list()
            ids = {item["source_id"] for item in sources}
            self.assertIn("usgs", ids)
            self.assertIn("manual", ids)
            added = manager.sources.add({
                "source_id": "local_news",
                "name": "本地新闻 RSS",
                "kind": "rss",
                "domains": ["news", "society"],
                "url": "https://example.invalid/rss.xml",
                "description": "地方新闻与自媒体",
                "tags": ["新闻", "RSS"],
            })
            self.assertFalse(added["builtin"])
            found = manager.sources.list(q="本地")
            self.assertEqual([item["source_id"] for item in found],
                             ["local_news"])
            by_domain = {item["source_id"]
                         for item in manager.sources.list(q="society")}
            self.assertIn("local_news", by_domain)
            self.assertIn("gdelt", by_domain)
            by_kind = manager.sources.list(kind="rss")
            self.assertEqual([item["source_id"] for item in by_kind],
                             ["local_news"])
            manager.sources.update("local_news", {"enabled": "false"})
            self.assertEqual(manager.sources.list(enabled=False)[0]["source_id"],
                             "local_news")
            removed = manager.sources.delete("local_news")
            self.assertEqual(removed["source_id"], "local_news")
            with self.assertRaises(ValueError):
                manager.sources.delete("usgs")


class OverviewTest(unittest.TestCase):
    def test_manual_overrides_auto_but_keeps_original(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "case1", [_auto()])
            manager = IntelManager(root)
            item = manager.manual.create(_manual_payload())
            overview = manager.overview("case1")
            self.assertEqual(overview["counts"]["manual_active"], 1)
            self.assertEqual(overview["counts"]["auto"], 1)
            self.assertEqual(overview["counts"]["superseded"], 1)
            manual = [row for row in overview["items"]
                      if row["priority"] == "manual"][0]
            auto = [row for row in overview["items"]
                    if row["priority"] == "auto"][0]
            self.assertTrue(manual["effective"])
            self.assertEqual(manual["verification"], "unverified")
            self.assertFalse(auto["effective"])
            self.assertEqual(auto["superseded_by"], item["intel_id"])
            self.assertEqual(auto["record"]["observation_id"], "usgs:auto1")
            self.assertTrue(overview["links"])
            self.assertTrue(overview["key_items"])
            self.assertEqual(overview["key_items"][0]["id"], item["intel_id"])
            self.assertIn("人工情报优先", overview["key_items"][0]["reasons"][0])

    def test_revoke_restores_auto_effective(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "case1", [_auto()])
            manager = IntelManager(root)
            item = manager.manual.create(_manual_payload())
            manager.manual.revoke(item["intel_id"])
            overview = manager.overview("case1")
            auto = [row for row in overview["items"]
                    if row["priority"] == "auto"][0]
            manual = [row for row in overview["items"]
                      if row["priority"] == "manual"][0]
            self.assertTrue(auto["effective"])
            self.assertFalse(manual["effective"])
            self.assertEqual(overview["counts"]["superseded"], 0)

    def test_different_domain_does_not_supersede(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "case1", [_auto()])
            manager = IntelManager(root)
            manager.manual.create(_manual_payload(domain="weather",
                                                  intel_type="weather"))
            overview = manager.overview("case1")
            auto = [row for row in overview["items"]
                    if row["priority"] == "auto"][0]
            self.assertTrue(auto["effective"])
            self.assertEqual(overview["counts"]["superseded"], 0)

    def test_generic_target_does_not_supersede_different_type(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "case1", [_auto()])
            manager = IntelManager(root)
            manager.manual.create(_manual_payload(intel_type="target"))
            overview = manager.overview("case1")
            auto = [row for row in overview["items"]
                    if row["priority"] == "auto"][0]
            self.assertTrue(auto["effective"])
            self.assertEqual(overview["counts"]["superseded"], 0)

    def test_missing_bundle_is_warning_not_failure(self):
        with tempfile.TemporaryDirectory() as root:
            manager = IntelManager(root)
            manager.manual.create(_manual_payload())
            overview = manager.overview("case1")
            self.assertEqual(overview["counts"]["manual_active"], 1)
            self.assertEqual(overview["counts"]["auto"], 0)
            self.assertTrue(overview["warnings"])

    def test_mismatched_bundle_is_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            _write_bundle(root, "case1", [_auto()])
            path = os.path.join(root, "case1", "bundle.json")
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            data["scenario_id"] = "global"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            overview = IntelManager(root).overview("case1")
            self.assertEqual(overview["counts"]["auto"], 0)
            self.assertTrue(any("不一致" in warning
                                for warning in overview["warnings"]))


if __name__ == "__main__":
    unittest.main()
