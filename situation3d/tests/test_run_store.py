"""Run-store retention and integrity tests (TW2 dual-mode, D0)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
SERVER = os.path.join(SITUATION3D, "server")
for p in (SITUATION3D, SERVER):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_store import RunConflict, RunError, RunStore  # noqa: E402

CASE = "taiwan_dense_transfer_v1"


class RunStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RunStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _complete(self, seed=1, validation=True, case=CASE):
        rid = self.store.create_run(case, seed=seed)
        self.store.complete_run(
            case, rid,
            {"trajectory.json": json.dumps({"seed": seed}).encode("utf-8"),
             "nodes.json": b"{\"nodes\":[]}",
             "summary.json": b"{}"},
            metrics={"seed": seed, "events": seed},
            validation={"passed": validation},
        )
        return rid

    def _roles(self, case=CASE):
        return {r["run_id"]: sorted(r["roles"]) for r in self.store.list_runs(case)}

    def test_latest_rotates_and_previous_latest_is_dropped(self):
        a = self._complete(1)
        b = self._complete(2)
        rows = self.store.list_runs(CASE)
        self.assertEqual([r["run_id"] for r in rows], [b])
        self.assertEqual(self.store.get_run(CASE, b)["roles"], ["latest"])
        self.assertFalse(os.path.isdir(os.path.join(self.tmp.name, CASE, a)))

    def test_three_slots_latest_best_pinned(self):
        a = self._complete(1)
        self.store.pin(CASE, a)
        b = self._complete(2)
        self.store.set_best(CASE, b)
        c = self._complete(3)
        rows = self.store.list_runs(CASE)
        self.assertEqual(len(rows), 3)
        roles = self._roles()
        self.assertEqual(roles[a], ["pinned"])
        self.assertEqual(roles[b], ["best"])
        self.assertEqual(roles[c], ["latest"])

    def test_same_run_can_hold_all_roles_without_duplicates(self):
        a = self._complete(1)
        self.store.set_best(CASE, a)
        self.store.pin(CASE, a)
        rows = self.store.list_runs(CASE)
        self.assertEqual(len(rows), 1)
        self.assertEqual(sorted(rows[0]["roles"]), ["best", "latest", "pinned"])

    def test_new_latest_does_not_evict_best_or_pinned(self):
        a = self._complete(1)
        self.store.pin(CASE, a)
        b = self._complete(2)
        self.store.set_best(CASE, b)
        c = self._complete(3)
        roles = self._roles()
        self.assertIn(a, roles)
        self.assertIn(b, roles)
        self.assertIn(c, roles)

    def test_set_best_replaces_old_best_and_prunes_it(self):
        a = self._complete(1)
        self.store.set_best(CASE, a)
        b = self._complete(2)
        self.store.set_best(CASE, b)
        self.assertFalse(os.path.isdir(os.path.join(self.tmp.name, CASE, a)))
        self.assertEqual(sorted(self.store.get_run(CASE, b)["roles"]), ["best", "latest"])

    def test_pin_requires_explicit_replace(self):
        a = self._complete(1)
        self.store.pin(CASE, a)
        b = self._complete(2)
        with self.assertRaises(RunConflict) as ctx:
            self.store.pin(CASE, b)
        self.assertEqual(ctx.exception.detail.get("pinned"), a)
        self.store.pin(CASE, b, replace=a)
        self.assertFalse(os.path.isdir(os.path.join(self.tmp.name, CASE, a)))
        self.assertEqual(self.store.get_run(CASE, b)["roles"], ["latest", "pinned"])

    def test_delete_rules(self):
        a = self._complete(1)
        self.store.set_best(CASE, a)
        with self.assertRaises(RunConflict):
            self.store.delete_run(CASE, a)
        b = self._complete(2)
        self.store.pin(CASE, b)
        with self.assertRaises(RunConflict):
            self.store.delete_run(CASE, b)
        self.store.unpin(CASE, b)
        # b is still latest, and deleting latest is allowed.
        self.store.delete_run(CASE, b)
        self.assertFalse(os.path.isdir(os.path.join(self.tmp.name, CASE, b)))

    def test_validation_must_pass_to_be_best(self):
        a = self._complete(1, validation=False)
        with self.assertRaises(RunConflict):
            self.store.set_best(CASE, a)

    def test_verify_detects_tampering(self):
        a = self._complete(1)
        self.assertTrue(self.store.verify(CASE, a)["ok"])
        path = os.path.join(self.tmp.name, CASE, a, "trajectory.json")
        with open(path, "wb") as fh:
            fh.write(b"{\"tampered\":true}")
        result = self.store.verify(CASE, a)
        self.assertFalse(result["ok"])
        self.assertIn("trajectory.json", result["mismatches"])

    def test_failed_runs_keep_only_newest_and_do_not_take_a_slot(self):
        live = self._complete(1)
        f1 = self.store.create_run(CASE, seed=91)
        self.store.fail_run(CASE, f1, "boom 1")
        f2 = self.store.create_run(CASE, seed=92)
        self.store.fail_run(CASE, f2, "boom 2")
        rows = self.store.list_runs(CASE)
        ids = [r["run_id"] for r in rows]
        self.assertIn(live, ids)
        self.assertIn(f2, ids)
        self.assertNotIn(f1, ids)
        self.assertEqual(len(ids), 2)

    def test_state_rebuilds_from_run_json(self):
        a = self._complete(1)
        self.store.pin(CASE, a)
        os.remove(os.path.join(self.tmp.name, CASE, "state.json"))
        rows = self.store.list_runs(CASE)
        self.assertEqual(rows[0]["run_id"], a)
        self.assertEqual(sorted(rows[0]["roles"]), ["latest", "pinned"])

    def test_explicit_run_id_reservation(self):
        rid = "20261023T000000Z-abcdef12"
        got = self.store.create_run(CASE, seed=5, run_id=rid)
        self.assertEqual(got, rid)
        self.assertEqual(self.store.get_run(CASE, rid)["status"], "queued")
        with self.assertRaises(RunConflict):
            self.store.create_run(CASE, seed=6, run_id=rid)

    def test_audit_note_is_appended(self):
        rid = self._complete(1)
        self.store.audit_note(CASE, rid, "select", "smoke note")
        path = os.path.join(self.tmp.name, CASE, "audit.jsonl")
        with open(path, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        self.assertTrue(any(x.get("action") == "select" and x.get("note") == "smoke note"
                            for x in lines))

    def test_unpin_does_not_resurrect_removed_latest(self):
        a = self._complete(1)
        self.store.pin(CASE, a)
        b = self._complete(2)
        self.store.unpin(CASE, a)
        latest = [rid for rid, roles in self._roles().items() if "latest" in roles]
        self.assertEqual(latest, [b])

    def test_active_run_cannot_be_deleted(self):
        rid = self.store.create_run(CASE, seed=1)
        with self.assertRaises(RunConflict):
            self.store.delete_run(CASE, rid)

    def test_invalid_ids_rejected(self):
        with self.assertRaises(RunError):
            self.store.create_run("../escape")
        with self.assertRaises(RunError):
            self.store.get_run(CASE, "../escape")


if __name__ == "__main__":
    unittest.main()
