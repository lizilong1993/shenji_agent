"""Cross-source fusion tests: space-time clustering, anchors, determinism."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel import (IntelBundle, IntelGateway, Observation, fuse,  # noqa: E402
                   iso)
from intel.normalize import normalize_firms, normalize_gdacs, normalize_usgs  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "intel")
FETCHED = "2026-10-24T00:00:00Z"
BASE_MS = int(datetime(2026, 10, 24, tzinfo=timezone.utc).timestamp() * 1000)


def load(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def all_observations():
    return (normalize_usgs(load("usgs.json"), FETCHED)
            + normalize_gdacs(load("gdacs.json"), FETCHED)
            + normalize_firms(load("firms.json"), FETCHED))


def make(oid, lat, lon, offset_s, conf, source="usgs"):
    event = datetime.fromtimestamp((BASE_MS + offset_s * 1000) / 1000.0,
                                   tz=timezone.utc)
    return Observation(
        observation_id=oid, source_id=source, source_native_id=oid,
        domain="hazard", type="earthquake", lat=lat, lon=lon,
        event_time=iso(event), ingest_time=FETCHED, ingest_seq=0,
        confidence=conf, source_url="")


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
        expected = round(1.0 - (1.0 - members[0].confidence)
                         * (1.0 - members[1].confidence), 3)
        self.assertEqual(event.confidence, expected)
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
        obs = all_observations()
        events = fuse(obs)
        ids = [e.cluster_id for e in events]
        self.assertEqual(len(ids), len(set(ids)))
        for event in events:
            members = [o for o in obs if o.observation_id in event.member_ids]
            payload = "|".join(sorted(m.content_hash() for m in members))
            expected = "cluster:" + hashlib.sha256(
                payload.encode("utf-8")).hexdigest()[:16]
            self.assertEqual(event.cluster_id, expected)

    def test_anchor_does_not_drift_with_chained_points(self):
        # A 0 km (c .5), B 9 km (c .9), C 18 km (c .95): C anchors, B joins,
        # A is outside C's radius and must not ride in through B.
        events = fuse([
            make("A", 23.0, 120.0, 0, 0.5),
            make("B", 23.0814, 120.0, 0, 0.9),
            make("C", 23.1628, 120.0, 0, 0.95),
        ], radius_m=10000.0, window_s=3600.0)
        groups = sorted(sorted(e.member_ids) for e in events)
        self.assertEqual(groups, [["A"], ["B", "C"]])
        merged = [e for e in events if len(e.member_ids) == 2][0]
        self.assertEqual(merged.representative_id, "C")

    def test_duplicate_ids_are_deduplicated(self):
        events = fuse([
            make("firms:x", 22.9, 120.0, 0, 0.8, source="firms"),
            make("firms:x", 22.9, 120.0, 0, 0.9, source="firms"),
        ])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].member_ids, ("firms:x",))
        self.assertEqual(events[0].confidence, 0.9)

    def test_non_finite_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            fuse([make("bad", 23.0, 120.0, 0, float("nan"))])
        with self.assertRaises(ValueError):
            fuse([make("bad", float("nan"), 120.0, 0, 0.9)])
        with self.assertRaises(ValueError):
            fuse([make("bad", 100.0, 120.0, 0, 0.9)])

    def test_gateway_fused_cues_deduplicate_sources(self):
        bundle = IntelBundle("b1", "s1", all_observations(),
                             "2026-10-24T00:00:00Z")
        gateway = IntelGateway(bundle, fusion=(10000.0, 3600.0))
        batch = gateway.consume(1, 0.0, -90000.0, 400.0)
        self.assertEqual(len(batch.cues), 5)
        self.assertTrue(all(c.cue_kind == "fused" for c in batch.cues))
        fused = [c for c in batch.cues
                 if len(c.provenance["member_ids"]) > 1]
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].provenance["member_ids"],
                         ["gdacs:gd1", "usgs:us1"])
        self.assertEqual(fused[0].provenance["sources"], ["gdacs", "usgs"])
        self.assertEqual(fused[0].observation_id, fused[0].provenance["cluster_id"])
        again = gateway.consume(1, 0.0, -90000.0, 400.0)
        self.assertEqual(len(again.cues), 0)

    def test_gateway_without_fusion_keeps_each_source(self):
        bundle = IntelBundle("b1", "s1", all_observations(),
                             "2026-10-24T00:00:00Z")
        batch = IntelGateway(bundle).consume(1, 0.0, -90000.0, 400.0)
        self.assertEqual(len(batch.cues), 6)
        self.assertTrue(all(c.cue_kind == "observation" for c in batch.cues))

    def test_fused_gateway_never_exposes_future_members(self):
        bundle = IntelBundle("b1", "s1", all_observations(),
                             "2026-10-24T00:00:00Z")
        gateway = IntelGateway(bundle, fusion=(10000.0, 3600.0))
        early = gateway.consume(1, 0.0, 0.0, 60.0)
        self.assertEqual(len(early.cues), 1)
        self.assertEqual(early.cues[0].provenance["member_ids"], ["usgs:us1"])
        later = gateway.consume(1, 0.0, 0.0, 180.0)
        ids = {tuple(c.provenance["member_ids"]) for c in later.cues}
        self.assertIn(("gdacs:gd1",), ids)

    def test_fused_gateway_watermark_is_member_level(self):
        bundle = IntelBundle("b1", "s1", all_observations(),
                             "2026-10-24T00:00:00Z")
        gateway = IntelGateway(bundle, fusion=(10000.0, 3600.0))
        batch = gateway.consume(1, 0.0, 90.0, 400.0)
        ids = {tuple(c.provenance["member_ids"]) for c in batch.cues}
        self.assertIn(("gdacs:gd1",), ids)
        self.assertIn("usgs:us1", gateway.late)

    def test_fused_log_hash_is_stable(self):
        def run():
            bundle = IntelBundle("b1", "s1", all_observations(),
                                 "2026-10-24T00:00:00Z")
            return IntelGateway(bundle, fusion=(10000.0, 3600.0)).consume(
                1, 0.0, -90000.0, 400.0).log_hash
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
