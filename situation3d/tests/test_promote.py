"""Promotion integrity tests (TW2 D3)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
for p in (SITUATION3D, os.path.join(SITUATION3D, "server"),
          os.path.join(SITUATION3D, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import promote_run  # noqa: E402
from run_store import RunError, RunStore  # noqa: E402

CASE = "taiwan_dense_transfer_v1"


class PromoteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = os.path.join(self.tmp.name, "runs")
        self.demo = os.path.join(self.tmp.name, "demo")
        os.makedirs(os.path.join(self.demo, "situation3d", "web", "data", "situations"))
        self.store = RunStore(self.runs)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, payload=b"{\"frames\":[]}", validation=True, select=True):
        rid = self.store.create_run(CASE, seed=1)
        self.store.complete_run(CASE, rid, {"trajectory.json": payload},
                                metrics={"events": 1},
                                validation={"passed": validation})
        if select:
            self.store.set_best(CASE, rid)
        return rid

    def test_promote_selected_run_copies_bytes_and_manifest(self):
        rid = self._run()
        result = promote_run.promote(CASE, rid, self.runs, self.demo)
        target = os.path.join(self.demo, "situation3d", "web", "data",
                              "situations", CASE + ".json")
        with open(target, "rb") as fh:
            self.assertEqual(fh.read(), b"{\"frames\":[]}")
        with open(result["release"], encoding="utf-8") as fh:
            release = json.load(fh)
        self.assertEqual(release["run_id"], rid)
        self.assertEqual(release["trajectory_sha256"], result["sha256"])
        # The manifest must live outside the situations directory; a JSON with
        # the same case_id there would shadow the trajectory in the index.
        situations = os.path.dirname(result["target"])
        self.assertFalse(os.path.exists(os.path.join(situations, "demo_release.json")))
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(situations),
                                                    "demo_release.json")))

    def test_unselected_run_rejected(self):
        rid = self._run(select=False)
        with self.assertRaises(RunError):
            promote_run.promote(CASE, rid, self.runs, self.demo)

    def test_failed_validation_rejected(self):
        rid = self._run(validation=False, select=False)
        # set_best refuses invalid runs; force the role for the promotion check.
        self.store.pin(CASE, rid)
        with self.assertRaises(RunError):
            promote_run.promote(CASE, rid, self.runs, self.demo)

    def test_tampered_artifact_rejected(self):
        rid = self._run()
        path = os.path.join(self.runs, CASE, rid, "trajectory.json")
        with open(path, "wb") as fh:
            fh.write(b"tampered")
        with self.assertRaises(RunError):
            promote_run.promote(CASE, rid, self.runs, self.demo)

    def test_backup_created_for_existing_demo_file(self):
        target_dir = os.path.join(self.demo, "situation3d", "web", "data", "situations")
        with open(os.path.join(target_dir, CASE + ".json"), "w", encoding="utf-8") as fh:
            fh.write("old")
        rid = self._run()
        promote_run.promote(CASE, rid, self.runs, self.demo)
        backups = [n for n in os.listdir(target_dir) if ".bak_prev_" in n]
        self.assertEqual(len(backups), 1)


if __name__ == "__main__":
    unittest.main()
