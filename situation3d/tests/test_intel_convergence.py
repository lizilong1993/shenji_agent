"""Tests for grid-cell convergence events (task F)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.convergence import (  # noqa: E402
    cell_of, cell_size_for, convergence_events, summarize)
from intel.management import IntelManager  # noqa: E402

TAIWAN_BBOX = (119.0, 21.5, 123.0, 25.5)


def item(item_id, type_, lat=24.0, lon=120.5, event_time="2026-10-24T12:00:00Z",
         source_id="usgs", domain="hazard", confidence=0.9, **extra):
    payload = {
        "id": item_id,
        "type": type_,
        "lat": lat,
        "lon": lon,
        "event_time": event_time,
        "source_id": source_id,
        "domain": domain,
        "confidence": confidence,
        "status": "active",
        "effective": True,
        "title": item_id,
    }
    payload.update(extra)
    return payload


class ConvergenceTest(unittest.TestCase):
    def test_three_types_in_one_cell_is_an_event(self):
        events = convergence_events([
            item("a", "earthquake"),
            item("b", "fire", source_id="firms"),
            item("c", "conflict", source_id="gdelt", domain="conflict"),
        ], bbox=TAIWAN_BBOX)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["types"], ["conflict", "earthquake", "fire"])
        self.assertEqual(event["member_count"], 3)
        self.assertEqual([row["id"] for row in event["members"]],
                         ["a", "b", "c"])
        self.assertTrue(event["reasons"][0].startswith("同格 3 类信号"))
        self.assertEqual(event["t_start"], "2026-10-24T12:00:00Z")

    def test_two_types_do_not_trigger(self):
        events = convergence_events([
            item("a", "earthquake"),
            item("b", "fire", source_id="firms"),
        ], bbox=TAIWAN_BBOX)
        self.assertEqual(events, [])

    def test_same_type_repeats_do_not_count_as_diversity(self):
        events = convergence_events([
            item("a", "earthquake"),
            item("b", "eq", source_id="gdacs"),
            item("c", "earthquakes", source_id="eonet"),
        ], bbox=TAIWAN_BBOX)
        self.assertEqual(events, [])

    def test_different_cells_do_not_converge(self):
        events = convergence_events([
            item("a", "earthquake", lat=24.0, lon=120.5),
            item("b", "fire", lat=24.0, lon=120.5),
            item("c", "conflict", lat=25.2, lon=121.8),
        ], bbox=TAIWAN_BBOX)
        self.assertEqual(events, [])

    def test_window_splits_the_burst(self):
        events = convergence_events([
            item("a", "earthquake", event_time="2026-10-24T00:00:00Z"),
            item("b", "fire", event_time="2026-10-24T01:00:00Z"),
            item("c", "conflict", event_time="2026-10-26T00:00:00Z"),
        ], bbox=TAIWAN_BBOX, window_h=24.0)
        self.assertEqual(events, [])

    def test_revoked_and_superseded_items_are_ignored(self):
        events = convergence_events([
            item("a", "earthquake"),
            item("b", "fire"),
            item("c", "conflict", status="revoked"),
            item("d", "flood", effective=False),
        ], bbox=TAIWAN_BBOX)
        self.assertEqual(events, [])

    def test_missing_coordinates_are_ignored(self):
        events = convergence_events([
            item("a", "earthquake"),
            item("b", "fire"),
            item("c", "conflict", lat=None),
        ], bbox=TAIWAN_BBOX)
        self.assertEqual(events, [])

    def test_cell_center_and_distance_are_reported(self):
        events = convergence_events([
            item("a", "earthquake", lat=24.1, lon=120.6),
            item("b", "fire", lat=24.2, lon=120.7),
            item("c", "conflict", lat=24.3, lon=120.8),
        ], bbox=TAIWAN_BBOX, cell_deg=0.5)
        event = events[0]
        self.assertEqual(event["cell"]["center_lat"], 24.25)
        self.assertEqual(event["cell"]["center_lon"], 120.75)
        for row in event["members"]:
            self.assertGreaterEqual(row["distance_km"], 0.0)
            self.assertLessEqual(row["distance_km"], 40.0)
        self.assertEqual(event["members"][0]["offset_h"], 0.0)

    def test_event_is_deterministic(self):
        rows = [item("a", "earthquake"), item("b", "fire"),
                item("c", "conflict")]
        first = convergence_events(rows, bbox=TAIWAN_BBOX)
        second = convergence_events(list(reversed(rows)), bbox=TAIWAN_BBOX)
        self.assertEqual([e["id"] for e in first], [e["id"] for e in second])

    def test_overlapping_windows_report_one_event(self):
        events = convergence_events([
            item("a", "earthquake", event_time="2026-10-24T00:00:00Z"),
            item("b", "fire", event_time="2026-10-24T01:00:00Z"),
            item("c", "conflict", event_time="2026-10-24T02:00:00Z"),
            item("d", "flood", event_time="2026-10-24T03:00:00Z"),
        ], bbox=TAIWAN_BBOX)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["member_count"], 4)

    def test_summary_counts_types_and_sources(self):
        events = convergence_events([
            item("a", "earthquake"),
            item("b", "fire", source_id="firms"),
            item("c", "conflict", source_id="gdelt"),
        ], bbox=TAIWAN_BBOX)
        summary = summarize(events)
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["by_source"],
                         {"firms": 1, "gdelt": 1, "usgs": 1})

    def test_adaptive_cell_size_and_floor(self):
        self.assertAlmostEqual(cell_size_for(TAIWAN_BBOX), 0.25, places=6)
        self.assertEqual(cell_size_for((120.0, 24.0, 120.01, 24.01)), 0.05)
        self.assertEqual(cell_of(-0.5, -0.5, 1.0), (-1, -1))
        self.assertEqual(cell_of(24.9, 120.9, 1.0), (24, 120))

    def test_empty_input(self):
        self.assertEqual(convergence_events([], bbox=TAIWAN_BBOX), [])
        self.assertEqual(summarize([])["count"], 0)


class ConvergenceOverviewTest(unittest.TestCase):
    def _write_bundle(self, root, case_id, observations):
        import json
        directory = os.path.join(root, case_id)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "bundle.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"bundle_id": "b1", "scenario_id": case_id,
                       "scenario_start": "2026-10-24T00:00:00Z",
                       "synthetic": False, "hash": "abc",
                       "observations": observations}, handle)

    def _auto(self, obs_id, type_, source_id, lat=24.0, lon=120.5):
        return {
            "observation_id": obs_id, "source_id": source_id,
            "source_native_id": obs_id,
            "domain": "conflict" if type_ == "conflict" else "hazard",
            "type": type_, "event_time": "2026-10-24T12:00:00Z",
            "ingest_time": "2026-10-24T12:05:00Z",
            "lat": lat, "lon": lon, "title": obs_id,
            "confidence": 0.9, "status": "active",
        }

    def test_overview_exposes_events(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_bundle(root, "case1", [
                self._auto("a", "earthquake", "usgs"),
                self._auto("b", "fire", "firms"),
                self._auto("c", "conflict", "gdelt"),
            ])
            overview = IntelManager(root).overview("case1")
            block = overview["convergence"]
            self.assertEqual(block["summary"]["count"], 1)
            self.assertEqual(len(block["events"]), 1)
            self.assertEqual(overview["counts"]["convergence"], 1)
            self.assertEqual(block["rule"]["min_types"], 3)

    def test_overview_without_convergence_reports_zero(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_bundle(root, "case1", [
                self._auto("a", "earthquake", "usgs"),
                self._auto("b", "fire", "firms"),
            ])
            overview = IntelManager(root).overview("case1")
            self.assertEqual(overview["convergence"]["summary"]["count"], 0)
            self.assertEqual(overview["counts"]["convergence"], 0)


if __name__ == "__main__":
    unittest.main()
