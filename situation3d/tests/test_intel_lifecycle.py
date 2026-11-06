"""Tests for the five-stage target lifecycle state machine."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.lifecycle import (  # noqa: E402
    LifecycleStore, STAGES, TRANSITIONS, can_transition)
from intel.management import IntelManager  # noqa: E402


class LifecycleStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = LifecycleStore(os.path.join(self.tmp.name, "lc.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_target_starts_at_its_stage(self):
        event = self.store.transit({"target_id": "T1", "to_stage": "identified"})
        self.assertEqual(event["from"], "identified")
        self.assertEqual(event["to"], "identified")
        self.assertEqual(self.store.get("T1")["stage"], "identified")

    def test_new_target_cannot_skip_stages(self):
        with self.assertRaises(ValueError):
            self.store.transit({"target_id": "T1", "to_stage": "complete",
                                "from_stage": "identified"})

    def test_walks_the_full_chain(self):
        self.store.transit({"target_id": "T1", "to_stage": "identified"})
        for stage in ("prioritized", "in_coordination", "in_execution",
                      "complete"):
            self.store.transit({"target_id": "T1", "to_stage": stage})
        self.assertEqual(self.store.get("T1")["stage"], "complete")
        self.assertEqual(len(self.store.history("T1")), 5)

    def test_illegal_transition_is_rejected(self):
        self.store.transit({"target_id": "T1", "to_stage": "identified"})
        with self.assertRaises(ValueError):
            self.store.transit({"target_id": "T1", "to_stage": "in_execution"})

    def test_backward_transition_allowed_where_declared(self):
        for stage in ("identified", "prioritized", "in_coordination"):
            self.store.transit({"target_id": "T1", "to_stage": stage})
        event = self.store.transit({"target_id": "T1", "to_stage": "prioritized"})
        self.assertEqual(event["from"], "in_coordination")
        self.assertEqual(event["to"], "prioritized")

    def test_complete_can_be_reopened(self):
        for stage in ("identified", "prioritized", "in_coordination",
                      "in_execution", "complete"):
            self.store.transit({"target_id": "T1", "to_stage": stage})
        event = self.store.transit({"target_id": "T1", "to_stage": "identified"})
        self.assertEqual(event["from"], "complete")

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):
            self.store.transit({"target_id": "T1", "to_stage": "archived"})

    def test_missing_target_id_rejected(self):
        with self.assertRaises(ValueError):
            self.store.transit({"to_stage": "identified"})

    def test_board_groups_by_stage(self):
        for tid, stages in (("A", ("identified",)),
                            ("B", ("identified", "prioritized")),
                            ("C", ("identified", "prioritized",
                                   "in_coordination"))):
            for stage in stages:
                self.store.transit({"target_id": tid, "to_stage": stage})
        board = self.store.board()
        self.assertEqual(sorted(row["target_id"] for row in board["identified"]),
                         ["A"])
        self.assertEqual(sorted(row["target_id"] for row in board["prioritized"]),
                         ["B"])
        self.assertEqual(sorted(row["target_id"]
                                for row in board["in_coordination"]), ["C"])
        self.assertEqual(board["complete"], [])

    def test_history_is_append_only(self):
        self.store.transit({"target_id": "T1", "to_stage": "identified",
                            "actor": "analyst-a", "note": "首次登记"})
        self.store.transit({"target_id": "T1", "to_stage": "prioritized",
                            "actor": "analyst-b"})
        rows = self.store.history("T1")
        self.assertEqual([row["seq"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["note"], "首次登记")
        self.assertEqual(rows[1]["actor"], "analyst-b")

    def test_revision_tracks_history_length(self):
        before = self.store.revision()
        self.store.transit({"target_id": "T1", "to_stage": "identified"})
        self.assertNotEqual(before, self.store.revision())

    def test_transition_table_is_closed(self):
        # 每一段都在表里，且 complete 之后的去处在表里明确写了。
        self.assertEqual(set(TRANSITIONS), set(STAGES))
        self.assertTrue(can_transition("identified", "prioritized"))
        self.assertFalse(can_transition("identified", "complete"))


class BdaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = LifecycleStore(os.path.join(self.tmp.name, "lc.json"))
        for stage in ("identified", "prioritized", "in_coordination",
                      "in_execution"):
            self.store.transit({"target_id": "T1", "to_stage": stage})

    def tearDown(self):
        self.tmp.cleanup()

    def test_assess_closes_the_target(self):
        event = self.store.assess({"target_id": "T1", "result": "destroyed",
                                   "notes": "炮塔已掀翻", "media": ["a.jpg"]})
        self.assertEqual(event["from"], "in_execution")
        self.assertEqual(event["to"], "complete")
        row = self.store.get("T1")
        self.assertEqual(row["stage"], "complete")
        self.assertEqual(row["assessment"]["result"], "destroyed")
        self.assertEqual(row["assessment"]["media"], ["a.jpg"])

    def test_bda_requires_in_execution(self):
        self.store.transit({"target_id": "T1", "to_stage": "complete"})
        with self.assertRaises(ValueError):
            self.store.assess({"target_id": "T1", "result": "destroyed"})

    def test_bda_result_must_be_known(self):
        with self.assertRaises(ValueError):
            self.store.assess({"target_id": "T1", "result": "vaporized"})

    def test_unknown_target_rejected(self):
        with self.assertRaises(ValueError):
            self.store.assess({"target_id": "nope", "result": "destroyed"})

    def test_bda_is_in_the_audit_trail(self):
        self.store.assess({"target_id": "T1", "result": "damaged",
                           "actor": "analyst-a"})
        last = self.store.history("T1")[-1]
        self.assertIn("BDA: damaged", last["note"])
        self.assertEqual(last["actor"], "analyst-a")


class LifecycleOverviewTest(unittest.TestCase):
    def test_overview_exposes_board_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as root:
            manager = IntelManager(root)
            manager.lifecycle.transit({"target_id": "PJ1",
                                       "to_stage": "identified"})
            manager.lifecycle.transit({"target_id": "PJ1",
                                       "to_stage": "prioritized"})
            block = manager.overview("c1")["lifecycle"]
            self.assertEqual([stage["key"] for stage in block["stages"]],
                             list(STAGES))
            self.assertEqual(block["stages"][0]["label"], "已识别")
            self.assertEqual([row["target_id"] for row in
                              block["board"]["prioritized"]], ["PJ1"])
            self.assertEqual(block["transitions"]["identified"],
                             ["prioritized"])


if __name__ == "__main__":
    unittest.main()
