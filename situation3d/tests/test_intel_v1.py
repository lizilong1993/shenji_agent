"""VGI P1 slice: normalization, trust decay, bundle hash, cue gateway.

Run from ``situation3d/``:
    python3 -m unittest tests/test_intel_v1.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel import IntelBundle, IntelGateway, components, normalize, recency  # noqa: E402
from intel.normalize import normalize_firms, normalize_gdacs, normalize_usgs  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "intel")
START = "2026-10-24T00:00:00Z"
FETCHED = "2026-10-24T00:00:00Z"


def load(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


class IntelTest(unittest.TestCase):
    def test_normalizers(self):
        us = normalize_usgs(load("usgs.json"), FETCHED)
        self.assertEqual(len(us), 3)
        self.assertEqual(us[0].type, "earthquake")
        self.assertEqual(us[0].domain, "hazard")
        self.assertAlmostEqual(us[0].lat, 23.0)
        self.assertEqual(us[0].properties["magnitude"], 5.2)
        gd = normalize_gdacs(load("gdacs.json"), FETCHED)
        self.assertEqual(len(gd), 1)
        self.assertEqual(gd[0].source_id, "gdacs")
        fi = normalize_firms(load("firms.json"), FETCHED)
        self.assertEqual(len(fi), 2)
        self.assertEqual(fi[0].type, "fire")
        self.assertEqual(normalize("usgs", load("usgs.json"), FETCHED)[0].source_id,
                         "usgs")

    def test_trust_recency_and_components(self):
        now = datetime(2026, 10, 24, tzinfo=timezone.utc)
        recent = recency(datetime(2026, 10, 23, 23, tzinfo=timezone.utc), now, "hazard")
        old = recency(datetime(2026, 10, 20, tzinfo=timezone.utc), now, "hazard")
        self.assertGreater(recent, old)
        c = components("usgs", datetime(2026, 10, 23, 23, tzinfo=timezone.utc),
                       now, "hazard")
        self.assertIn("source_reliability", c)
        self.assertIn("confidence", c)
        self.assertGreater(c["confidence"], 0.0)

    def test_bundle_hash_is_deterministic(self):
        obs = normalize_usgs(load("usgs.json"), FETCHED)
        b1 = IntelBundle("b1", "scenario", obs, START)
        b2 = IntelBundle("b1", "scenario", list(reversed(obs)), START)
        self.assertEqual(b1.hash, b2.hash)
        self.assertEqual(len(b1.as_dict()["observations"]), 3)

    def test_gateway_future_only_and_no_duplicates(self):
        obs = normalize_usgs(load("usgs.json"), FETCHED)
        for item in obs:
            item.properties["side"] = 1
        bundle = IntelBundle("b1", "scenario", obs, START)
        gw = IntelGateway(bundle, ttl_s=600.0)
        batch = gw.consume(side=1, sim_time=0.0, watermark=0.0, lookahead=180.0)
        self.assertEqual(len(batch.cues), 2)
        self.assertEqual(batch.cues[0].cue_kind, "observation")
        self.assertEqual(batch.cues[0].side, 1)
        self.assertGreaterEqual(batch.cues[0].valid_from, 0.0)
        again = gw.consume(side=1, sim_time=0.0, watermark=0.0, lookahead=180.0)
        self.assertEqual(len(again.cues), 0)
        batch2 = gw.consume(side=1, sim_time=200.0, watermark=180.0, lookahead=200.0)
        self.assertEqual(len(batch2.cues), 1)
        self.assertEqual(batch2.cues[0].observation_id, "usgs:us3")

    def test_gateway_late_policy(self):
        obs = normalize_usgs(load("usgs.json"), FETCHED)
        bundle = IntelBundle("b1", "scenario", obs, START)
        gw = IntelGateway(bundle)
        batch = gw.consume(side=1, sim_time=0.0, watermark=100.0, lookahead=400.0)
        self.assertEqual(len(batch.cues), 2)  # 120 s and 300 s; 60 s is late
        self.assertEqual(gw.late, ["usgs:us1"])

    def test_gateway_log_hash_stable(self):
        obs = normalize_usgs(load("usgs.json"), FETCHED)
        b1 = IntelBundle("b1", "scenario", obs, START)
        b2 = IntelBundle("b1", "scenario", list(reversed(obs)), START)
        h1 = IntelGateway(b1).consume(1, 0.0, 0.0, 180.0).log_hash
        h2 = IntelGateway(b2).consume(1, 0.0, 0.0, 180.0).log_hash
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
