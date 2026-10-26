"""Cross-source fusion tests: space-time clustering, anchors, determinism."""

from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel import fuse  # noqa: E402
from intel.normalize import normalize_firms, normalize_gdacs, normalize_usgs  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "intel")
FETCHED = "2026-10-24T00:00:00Z"


def load(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def all_observations():
    return (normalize_usgs(load("usgs.json"), FETCHED)
            + normalize_gdacs(load("gdacs.json"), FETCHED)
            + normalize_firms(load("firms.json"), FETCHED))


class FusionTest(unittest.TestCase):
    def test_same_event_merges_across_sources(self):
        events = fuse(all_observations())
        merged = [e for e in events if len(e.member_ids) > 1]
        self.assertEqual(len(merged), 1)
        event = merged[0]
        self.assertEqual(event.member_ids, ("gdacs:gd1", "usgs:us1"))
        self.assertEqual(event.sources, ("gdacs", "usgs"))
        self.assertEqual(event.type, "earthquake")
        self.assertEqual(event.representative_id, "usgs:us1")
        members = [o for o in all_observations()
                   if o.observation_id in event.member_ids]
        self.assertGreater(event.confidence, max(o.confidence for o in members))
        self.assertLessEqual(event.confidence, 0.999)

    def test_distinct_events_stay_separate(self):
        events = fuse(all_observations())
        ids = {e.representative_id for e in events}
        self.assertIn("usgs:us2", ids)
        self.assertIn("usgs:us3", ids)
        self.assertEqual(len(events), 5)  # us1+gd1, us2, us3, f1, f2

    def test_radius_and_window_thresholds(self):
        obs = all_observations()
        tight_radius = fuse(obs, radius_m=10.0)
        self.assertFalse(any(len(e.member_ids) > 1 for e in tight_radius))
        tight_time = fuse(obs, window_s=10.0)
        self.assertFalse(any(len(e.member_ids) > 1 for e in tight_time))

    def test_fusion_is_deterministic(self):
        obs = all_observations()
        forward = [e.as_dict() for e in fuse(obs)]
        backward = [e.as_dict() for e in fuse(list(reversed(obs)))]
        self.assertEqual(forward, backward)

    def test_cluster_id_is_content_addressed(self):
        events = fuse(all_observations())
        ids = [e.cluster_id for e in events]
        self.assertEqual(len(ids), len(set(ids)))
        for event in events:
            self.assertTrue(event.cluster_id.startswith("cluster:"))
            self.assertEqual(len(event.cluster_id), len("cluster:") + 16)


if __name__ == "__main__":
    unittest.main()
