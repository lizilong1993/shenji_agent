"""Resource resolution, preflight summary and waiting-state tests (P2)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)
for extra in (os.path.join(SITUATION3D, "server"),):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from resources import preflight  # noqa: E402
from resources.model import (ResourceDecision, ResourceEntry,  # noqa: E402
                             ResourceRequirement, hash_path)
from resources.resolver import requirements_for, resolve_requirements  # noqa: E402
from resources.store import ResourceStore  # noqa: E402
from run_store import RunConflict, RunStore  # noqa: E402

ENV_REQ = ResourceRequirement("envpack", "critical", "引擎空间权威",
                              {"contains": [120.1, 22.9],
                               "points": [(120.12, 22.92)]})


def add_entry(store, entry_id, kind, content, as_of, bbox=None, notes=None):
    os.makedirs(str(store.root), exist_ok=True)
    path = os.path.join(str(store.root),
                        entry_id.replace(":", "_") + ".bin")
    with open(path, "wb") as fh:
        fh.write(content.encode("utf-8"))
    digest, size = hash_path(path)
    entry = ResourceEntry(
        entry_id=entry_id, kind=kind, path=path, content_hash=digest,
        size=size, version=digest[:12], schema="test", provider="test",
        source="test", fetched_at=(as_of or "") + "T00:00:00Z",
        as_of=as_of or "", bbox=tuple(bbox) if bbox else None,
        license="test", quality="measured", notes=dict(notes or {}))
    store.register(entry)
    return entry


class ResolverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="res-plan-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_newest_snapshot_is_selected(self):
        add_entry(self.store, "envpack:old", "envpack", "old", "2026-10-20",
                  bbox=[120.0, 22.8, 120.2, 23.0])
        add_entry(self.store, "envpack:new", "envpack", "new", "2026-10-23",
                  bbox=[120.0, 22.8, 120.2, 23.0])
        plan = resolve_requirements(self.store, [ENV_REQ], "2026-10-26")
        self.assertEqual(plan.status, "ready")
        self.assertEqual([e.entry_id for e in plan.entries], ["envpack:new"])

    def test_future_snapshot_is_never_used(self):
        add_entry(self.store, "envpack:future", "envpack", "future",
                  "2026-11-01", bbox=[120.0, 22.8, 120.2, 23.0])
        plan = resolve_requirements(self.store, [ENV_REQ], "2026-10-26")
        self.assertEqual(plan.status, "blocked")
        self.assertIn("no_point_in_time", plan.gaps[0].reason)

    def test_stale_critical_blocks_then_decision_accepts(self):
        add_entry(self.store, "envpack:stale", "envpack", "stale",
                  "2026-08-01", bbox=[120.0, 22.8, 120.2, 23.0])
        plan = resolve_requirements(self.store, [ENV_REQ], "2026-10-26")
        self.assertEqual(plan.status, "blocked")
        self.assertIn("stale", plan.gaps[0].reason)
        decision = ResourceDecision(action="use_cached_stale",
                                    entry_id="envpack:stale")
        plan = resolve_requirements(self.store, [ENV_REQ], "2026-10-26",
                                    decisions=[decision])
        self.assertEqual(plan.status, "degraded")
        self.assertEqual([e.entry_id for e in plan.entries], ["envpack:stale"])
        self.assertIn("stale_accepted", plan.gaps[0].reason)

    def test_advisory_stale_degrades_but_continues(self):
        req = ResourceRequirement("intel_bundle", "advisory", "情报")
        add_entry(self.store, "intel_bundle:x", "intel_bundle", "x",
                  "2026-08-01")
        plan = resolve_requirements(self.store, [req], "2026-10-26")
        self.assertEqual(plan.status, "degraded")
        self.assertEqual(len(plan.entries), 1)

    def test_points_outside_bbox_are_rejected(self):
        add_entry(self.store, "envpack:small", "envpack", "small",
                  "2026-10-23", bbox=[120.0, 22.8, 120.11, 22.91])
        plan = resolve_requirements(self.store, [ENV_REQ], "2026-10-26")
        self.assertEqual(plan.status, "blocked")

    def test_multi_requires_every_entry(self):
        req = ResourceRequirement("platforms", "critical", "平台", multi=True)
        add_entry(self.store, "platforms:a", "platforms", "a", "2026-10-01")
        good = add_entry(self.store, "platforms:b", "platforms", "b",
                         "2026-10-01")
        plan = resolve_requirements(self.store, [req], "2026-10-26")
        self.assertEqual(plan.status, "ready")
        self.assertEqual(len(plan.entries), 2)
        with open(good.path, "ab") as fh:
            fh.write(b"tamper")
        plan = resolve_requirements(self.store, [req], "2026-10-26")
        self.assertEqual(plan.status, "blocked")
        self.assertIn("integrity", plan.gaps[0].reason)
        self.assertEqual(len(plan.entries), 1)

    def test_requirements_follow_scenario_and_tasks(self):
        scenario = {
            "geo_anchor": {"lon": 120.1, "lat": 22.9},
            "objectives": [{"name": "D", "lon": 120.15, "lat": 22.95}],
            "engine": {"use_role_cards": True},
        }
        missions = [{"id": "M1", "objective": "D", "lon": None, "lat": None}]
        reqs = requirements_for(scenario, missions, theatre="tw",
                                engine=scenario["engine"])
        kinds = [r.kind for r in reqs]
        self.assertEqual(kinds, ["envpack", "case_roads", "rolecards",
                                 "weapons", "platforms"])
        env = [r for r in reqs if r.kind == "envpack"][0]
        self.assertIn((120.15, 22.95), env.scope["points"])
        self.assertTrue([r for r in reqs if r.kind == "platforms"][0].multi)

    def test_plan_summary_messages(self):
        plan = resolve_requirements(self.store, [ENV_REQ], "2026-10-26")
        summary = preflight.plan_summary(plan)
        self.assertEqual(summary["status"], "blocked")
        self.assertEqual(len(summary["blockers"]), 1)
        self.assertIn("envpack", summary["blockers"][0])


class WaitingStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="res-wait-")
        self.store = RunStore(os.path.join(self.tmp, "runs"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_waiting_lifecycle(self):
        rid = self.store.create_run("case1", seed=1, config={})
        self.store.update_run("case1", rid, status="waiting_resources",
                              resource_gaps=[{"kind": "envpack"}])
        with self.assertRaises(RunConflict):
            self.store.create_run("case1", seed=2, config={},
                                  reject_active=True)
        self.store.update_run("case1", rid, status="queued")
        self.store.update_run("case1", rid, status="running")
        self.store.complete_run("case1", rid, {"summary.json": b"{}"})
        self.assertEqual(self.store.get_run("case1", rid)["status"],
                         "completed")

    def test_waiting_can_be_cancelled(self):
        rid = self.store.create_run("case1", seed=1, config={})
        self.store.update_run("case1", rid, status="waiting_resources")
        self.store.update_run("case1", rid, status="cancelled")
        self.assertEqual(self.store.get_run("case1", rid)["status"],
                         "cancelled")


if __name__ == "__main__":
    unittest.main()
