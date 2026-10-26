"""Intel scheduler tests: per-side cadence, manifest, world hook."""

from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from datetime import datetime, timezone  # noqa: E402

from intel import (IntelBundle, IntelGateway, IntelScheduler,  # noqa: E402
                   Observation, iso)
from intel.normalize import normalize_firms, normalize_gdacs, normalize_usgs  # noqa: E402
from sim.scenario import build_world  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "intel")
START = "2026-10-24T00:00:00Z"
FETCHED = "2026-10-24T00:00:00Z"

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 24, "rows": 24,
        "width_m": 200.0}


def load(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def observations():
    return (normalize_usgs(load("usgs.json"), FETCHED)
            + normalize_gdacs(load("gdacs.json"), FETCHED)
            + normalize_firms(load("firms.json"), FETCHED))


def bundle():
    return IntelBundle("b1", "s1", observations(), START)


def make_basic():
    cells = []
    for r in range(24):
        row = []
        for c in range(24):
            row.append({"pos": r * 100 + c, "elev": 100, "cond": 0,
                        "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID), "map_data": cells}


def scenario():
    return {"engine": {"detection_model": "table", "scan_interval_s": 30},
            "operators": [{
                "obj_id": 1, "name": "ZTZ-99A", "color": 0, "type": 2,
                "domain": "land", "lon": 119.905, "lat": 23.095,
                "blood": 4, "max_blood": 4,
                "observe_distance": {"1": 10, "2": 25, "3": 1},
            }],
            "missions": [], "roe": {}}


class SchedulerTest(unittest.TestCase):
    def test_advance_pulls_future_only_cues(self):
        scheduler = IntelScheduler(IntelGateway(bundle()), sides=(1,),
                                   lookahead_s=180.0)
        first = scheduler.advance(0.0)
        ids = sorted(cue.observation_id for cue in first[1].cues)
        self.assertEqual(ids, ["gdacs:gd1", "usgs:us1", "usgs:us2"])
        second = scheduler.advance(180.0)
        self.assertEqual([c.observation_id for c in second[1].cues],
                         ["usgs:us3"])
        manifest = scheduler.manifest()
        self.assertEqual(manifest["cue_total"], 4)
        self.assertEqual(manifest["bundle_hash"], bundle().hash)
        self.assertEqual(len(manifest["batches"]), 2)
        self.assertEqual(manifest["batches"][0]["side"], 1)
        self.assertTrue(manifest["late_ids"])

    def test_fused_scheduler_emits_one_cluster_cue(self):
        scheduler = IntelScheduler(
            IntelGateway(bundle(), fusion=(10000.0, 3600.0)),
            sides=(1,), lookahead_s=180.0)
        batch = scheduler.advance(0.0)[1]
        self.assertEqual(len(batch.cues), 2)
        merged = [c for c in batch.cues
                  if len(c.provenance["member_ids"]) > 1]
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].provenance["member_ids"],
                         ["gdacs:gd1", "usgs:us1"])

    def test_manifest_is_deterministic(self):
        def manifest():
            scheduler = IntelScheduler(IntelGateway(bundle()), sides=(1,),
                                       lookahead_s=180.0)
            scheduler.advance(0.0)
            return scheduler.manifest()
        self.assertEqual(manifest(), manifest())

    def test_manifest_does_not_alias_internal_state(self):
        scheduler = IntelScheduler(IntelGateway(bundle()), sides=(1,),
                                   lookahead_s=180.0)
        scheduler.advance(0.0)
        manifest = scheduler.manifest()
        manifest["batches"][0]["cue_ids"].append("INJECTED")
        manifest["batches"][0]["cue_count"] = 999
        fresh = scheduler.manifest()
        self.assertNotIn("INJECTED", fresh["batches"][0]["cue_ids"])
        self.assertNotEqual(fresh["batches"][0]["cue_count"], 999)

    def test_world_publishes_cues_only_when_due(self):
        world = build_world(scenario(), make_basic(), seed=1)
        world.set_intel_gateway(IntelGateway(bundle()), sides=(1,),
                                lookahead_s=400.0)
        self.assertEqual(world.intel_for(1)["cues"], [])
        world.step(10.0)
        self.assertEqual(world.intel_for(1)["cues"], [])  # all events > 10 s
        self.assertEqual(world.intel_manifest()["cue_total"], 4)  # prefetched
        world.step(100.0)
        cues = world.intel_for(1)["cues"]
        self.assertEqual(len(cues), 1)  # usgs:us1 at 60 s
        self.assertTrue(all(c["valid_from"] <= world.time_s for c in cues))
        world.step(200.0)
        self.assertEqual(len(world.intel_for(1)["cues"]), 4)

    def test_world_multi_side_gets_neutral_observations(self):
        world = build_world(scenario(), make_basic(), seed=1)
        world.set_intel_gateway(IntelGateway(bundle()), sides=(0, 1),
                                lookahead_s=400.0)
        world.step(310.0)
        self.assertEqual(len(world.intel_for(0)["cues"]), 4)
        self.assertEqual(len(world.intel_for(1)["cues"]), 4)
        batches = world.intel_manifest()["batches"]
        self.assertEqual({b["side"] for b in batches}, {0, 1})

    def test_world_early_event_is_not_lost(self):
        event = datetime(2026, 10, 24, 0, 0, 5, tzinfo=timezone.utc)
        obs = Observation(
            observation_id="usgs:early", source_id="usgs",
            source_native_id="early", domain="hazard", type="earthquake",
            lat=23.0, lon=120.1, event_time=iso(event), ingest_time=FETCHED,
            ingest_seq=0, confidence=0.9, source_url="")
        world = build_world(scenario(), make_basic(), seed=1)
        world.set_intel_gateway(
            IntelGateway(IntelBundle("b2", "s1", [obs], START)),
            sides=(1,), lookahead_s=300.0)
        world.step(10.0)
        self.assertEqual(len(world.intel_for(1)["cues"]), 1)
        self.assertEqual(world.intel_manifest()["late_ids"], [])

    def test_world_advance_to_keeps_intel_in_sync(self):
        world = build_world(scenario(), make_basic(), seed=1)
        world.set_intel_gateway(IntelGateway(bundle()), sides=(1,),
                                lookahead_s=300.0)
        world.advance_to(400.0)
        self.assertEqual(len(world.intel_for(1)["cues"]), 4)
        late = world.intel_manifest()["late_ids"]
        self.assertNotIn("usgs:us1", late)

    def test_world_without_scheduler_has_no_cues_key(self):
        world = build_world(scenario(), make_basic(), seed=1)
        self.assertNotIn("cues", world.intel_for(1))
        self.assertEqual(world.intel_manifest(), {"enabled": False})


if __name__ == "__main__":
    unittest.main()
