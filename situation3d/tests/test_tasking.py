"""User task validation and scenario-date tests (P1)."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim import tasking  # noqa: E402
from sim.tasking import TaskError  # noqa: E402

SCENARIO = {
    "duration_s": 3600,
    "objectives": [{"name": "D区"}, {"name": "E区"}],
    "operators": [
        {"obj_id": 1, "color": 0, "name": "red-1"},
        {"obj_id": 2, "color": 0, "name": "red-2"},
        {"obj_id": 9, "color": 1, "name": "blue-1"},
    ],
}

DEFAULTS = [
    {"id": "D0", "side": 0, "units": [1], "task": "hold",
     "objective": "D区", "lon": 120.1, "lat": 22.9},
    {"id": "B0", "side": 1, "units": [9], "task": "hold",
     "objective": "E区", "lon": 120.2, "lat": 22.95},
]


def task(**kw):
    base = {"id": "U1", "side": 0, "units": [1], "task": "move_to",
            "lon": 120.1, "lat": 22.9}
    base.update(kw)
    return base


class ParseRequestTest(unittest.TestCase):
    def test_none_is_default_mode(self):
        req = tasking.parse_request(None)
        self.assertEqual(req["mode"], "default")
        self.assertEqual(req["tasks"], [])

    def test_bare_list_is_user_mode(self):
        req = tasking.parse_request([task()])
        self.assertEqual(req["mode"], "user")
        self.assertEqual(len(req["tasks"]), 1)

    def test_dict_roundtrip(self):
        req = tasking.parse_request({"mode": "mixed", "side": 1,
                                     "tasks": [task(side=1, units=[9])]})
        self.assertEqual((req["mode"], req["side"]), ("mixed", 1))

    def test_bad_mode_and_side(self):
        with self.assertRaises(TaskError):
            tasking.parse_request({"mode": "all", "tasks": []})
        with self.assertRaises(TaskError):
            tasking.parse_request({"side": 2, "tasks": []})


class ValidateTasksTest(unittest.TestCase):
    def test_defaults_are_filled(self):
        result = tasking.validate_tasks([task()], SCENARIO)
        item = result[0]
        self.assertEqual(item["id"], "U1")
        self.assertEqual(item["pace"], "normal")
        self.assertEqual(item["risk"], "normal")
        self.assertEqual(item["arrive_m"], 250.0)
        self.assertEqual(item["units"], [1])
        self.assertEqual(item["task"], "move_to")

    def test_auto_id(self):
        result = tasking.validate_tasks([task(id=None)], SCENARIO)
        self.assertEqual(result[0]["id"], "U01")

    def test_objective_only(self):
        result = tasking.validate_tasks(
            [task(id="U2", units=[2], lon=None, lat=None, objective="E区")],
            SCENARIO)
        self.assertIsNone(result[0]["lon"])
        self.assertEqual(result[0]["objective"], "E区")

    def test_rejections(self):
        cases = [
            task(units=[404]),                       # unknown unit
            task(units=[9]),                         # foreign affiliation
            task(task="attack"),                     # unsupported task
            task(lon=None),                          # missing point/objective
            task(lon=None, lat=None, objective="X"), # unknown objective
            task(after=["NOPE"]),                    # dangling dependency
            task(start_s=7200),                      # beyond duration
            task(pace="fast"),                       # bad pace
            task(risk="extreme"),                    # bad risk
            task(arrive_m=0),                        # bad arrival radius
            task(id="bad id"),                       # invalid id
        ]
        for raw in cases:
            with self.assertRaises(TaskError, msg=str(raw)):
                tasking.validate_tasks([raw], SCENARIO)

    def test_nan_start_is_rejected(self):
        with self.assertRaises(TaskError):
            tasking.validate_tasks([task(start_s=float("nan"))], SCENARIO)

    def test_duplicate_unit_across_tasks(self):
        tasks = [task(id="U1"), task(id="U2", units=[1])]
        with self.assertRaises(TaskError):
            tasking.validate_tasks(tasks, SCENARIO)

    def test_duplicate_id(self):
        with self.assertRaises(TaskError):
            tasking.validate_tasks([task(id="U1"), task(id="U1", units=[2])],
                                   SCENARIO)

    def test_cycle(self):
        tasks = [task(id="A", after=["B"]), task(id="B", units=[2], after=["A"])]
        with self.assertRaises(TaskError):
            tasking.validate_tasks(tasks, SCENARIO)

    def test_bbox_enforced(self):
        with self.assertRaises(TaskError):
            tasking.validate_tasks([task(lon=120.5, lat=22.9)], SCENARIO,
                                   bbox=[120.0, 22.9, 120.2, 23.0])


class MergeTest(unittest.TestCase):
    def test_default_mode_keeps_defaults(self):
        request = tasking.parse_request(None)
        missions, summary = tasking.merge_missions(DEFAULTS, request, SCENARIO)
        self.assertEqual([m["id"] for m in missions], ["D0", "B0"])
        self.assertEqual(summary["mode"], "default")

    def test_default_mode_rejects_tasks(self):
        request = {"mode": "default", "side": 0, "tasks": [task()]}
        with self.assertRaises(TaskError):
            tasking.merge_missions(DEFAULTS, request, SCENARIO)

    def test_user_mode_keeps_unrelated_defaults(self):
        request = tasking.parse_request([task(id="U1", units=[2])])
        missions, summary = tasking.merge_missions(DEFAULTS, request, SCENARIO)
        self.assertEqual([m["id"] for m in missions], ["D0", "B0", "U1"])
        self.assertEqual(summary["user_tasks"], 1)
        self.assertEqual(summary["kept_default_missions"], 2)
        self.assertEqual(summary["dropped_default_missions"], [])
        self.assertEqual(len(summary["request_hash"]), 64)

    def test_user_mode_overrides_same_units(self):
        request = tasking.parse_request([task(id="U1", units=[1])])
        missions, summary = tasking.merge_missions(DEFAULTS, request, SCENARIO)
        self.assertEqual([m["id"] for m in missions], ["B0", "U1"])
        self.assertEqual(summary["dropped_default_missions"], ["D0"])

    def test_user_mode_trims_partial_overlap(self):
        defaults = [{"id": "D0", "side": 0, "units": [1, 2], "task": "hold",
                     "objective": "D区", "lon": 120.1, "lat": 22.9}]
        request = tasking.parse_request([task(id="U1", units=[1])])
        missions, summary = tasking.merge_missions(defaults, request, SCENARIO)
        self.assertEqual([m["id"] for m in missions], ["D0", "U1"])
        self.assertEqual(missions[0]["units"], [2])
        self.assertEqual(summary["trimmed_default_missions"], ["D0"])

    def test_dependency_on_dropped_default_is_rejected(self):
        defaults = [
            {"id": "D0", "side": 0, "units": [1], "task": "hold",
             "objective": "D区", "lon": 120.1, "lat": 22.9},
            {"id": "D1", "side": 0, "units": [2], "task": "hold",
             "objective": "E区", "lon": 120.2, "lat": 22.95,
             "after": ["D0"]},
        ]
        request = tasking.parse_request([task(id="U1", units=[1])])
        with self.assertRaises(TaskError):
            tasking.merge_missions(defaults, request, SCENARIO)

    def test_mixed_mode_keeps_all_but_rejects_unit_overlap(self):
        request = {"mode": "mixed", "side": 0, "tasks": [task(id="U1")]}
        with self.assertRaises(TaskError):
            tasking.merge_missions(DEFAULTS, request, SCENARIO)
        request = {"mode": "mixed", "side": 0,
                   "tasks": [task(id="U1", units=[2])]}
        missions, _summary = tasking.merge_missions(DEFAULTS, request, SCENARIO)
        self.assertEqual([m["id"] for m in missions], ["D0", "B0", "U1"])

    def test_per_task_side_overrides_that_sides_defaults(self):
        request = {"mode": "user", "side": 0,
                   "tasks": [task(id="U1", side=1, units=[9],
                                  objective="E区", lon=None, lat=None)]}
        missions, summary = tasking.merge_missions(DEFAULTS, request, SCENARIO)
        self.assertEqual([m["id"] for m in missions], ["D0", "U1"])
        self.assertEqual(summary["dropped_default_missions"], ["B0"])

    def test_mixed_per_task_side_conflict(self):
        request = {"mode": "mixed", "side": 0,
                   "tasks": [task(id="U1", side=1, units=[9],
                                  objective="E区", lon=None, lat=None)]}
        with self.assertRaises(TaskError):
            tasking.merge_missions(DEFAULTS, request, SCENARIO)

    def test_user_mode_can_depend_on_kept_default(self):
        request = {"mode": "user", "side": 0,
                   "tasks": [task(id="U1", after=["B0"])]}
        missions, _summary = tasking.merge_missions(DEFAULTS, request, SCENARIO)
        self.assertEqual([m["id"] for m in missions], ["B0", "U1"])


def load_case_plugin():
    path = os.path.join(SITUATION3D, "campaigns", "taiwan_dense.py")
    spec = importlib.util.spec_from_file_location("case_taiwan_dense", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ScenarioDateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = load_case_plugin()

    def test_explicit_date_with_offset(self):
        scenario = self.plugin.scenario_dict(
            scenario_date="2026-03-01T08:30:00+08:00")
        self.assertEqual(scenario["start_time_utc"], "2026-03-01T00:30:00Z")
        meta = self.plugin.case_meta(scenario_date="2026-03-01T08:30:00+08:00")
        self.assertEqual(meta["start_time_utc"], "2026-03-01T00:30:00Z")

    def test_date_only(self):
        scenario = self.plugin.scenario_dict(scenario_date="2026-03-01")
        self.assertEqual(scenario["start_time_utc"], "2026-03-01T00:00:00Z")

    def test_default_is_now(self):
        before = datetime.now(timezone.utc).replace(microsecond=0)
        scenario = self.plugin.scenario_dict()
        after = datetime.now(timezone.utc).replace(microsecond=0)
        started = datetime.strptime(scenario["start_time_utc"],
                                    "%Y-%m-%dT%H:%M:%SZ").replace(
                                        tzinfo=timezone.utc)
        self.assertGreaterEqual(started, before.replace(second=0))
        self.assertLessEqual(started, after)

    def test_invalid_date(self):
        with self.assertRaises(ValueError):
            self.plugin.scenario_dict(scenario_date="not-a-date")

    def test_missions_override(self):
        scenario = self.plugin.scenario_dict(missions=[])
        self.assertEqual(scenario["missions"], [])
        scenario = self.plugin.scenario_dict(tasking={"mode": "user",
                                                      "side": 0,
                                                      "user_tasks": 1})
        self.assertEqual(scenario["tasking"]["mode"], "user")

    def test_default_scenario_still_has_missions(self):
        scenario = self.plugin.scenario_dict()
        self.assertTrue(scenario["missions"])
        self.assertEqual(scenario["start_time_utc"][-1], "Z")


if __name__ == "__main__":
    unittest.main()
