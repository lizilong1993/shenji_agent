"""推演中途任务注入（live/retask.json → runner）。

只测文件通道这一层：写文件、被读取、结果写回、非法内容不打断推演。
真实的 ``run_situation_case.py`` 主循环不在这里跑（它依赖 envpack）。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

sys.path.insert(0, os.path.join(SITUATION3D, "scripts"))
from run_situation_case import _poll_retask  # noqa: E402


class FakeLive(object):
    def __init__(self, root):
        self.dir = root
        os.makedirs(root, exist_ok=True)


class FakeRunner(object):
    def __init__(self):
        self.calls = []

    def retask(self, missions):
        self.calls.append(list(missions))
        applied, rejected = [], []
        for item in missions:
            if item.get("objective") == "missing":
                rejected.append((item.get("id", ""), "找不到目标点"))
            else:
                applied.append(item.get("id", ""))
        return applied, rejected


class PollRetaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.live = FakeLive(self.root)
        self.runner = FakeRunner()
        self.state = {"mtime": None}

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, payload):
        path = os.path.join(self.root, "retask.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    def _result(self):
        with open(os.path.join(self.root, "retask-result.json"),
                  encoding="utf-8") as fh:
            return json.load(fh)

    def test_missing_file_is_a_noop(self):
        _poll_retask(self.runner, self.live, self.state, 5)
        self.assertEqual(self.runner.calls, [])
        self.assertFalse(os.path.exists(
            os.path.join(self.root, "retask-result.json")))

    def test_applied_is_written_back(self):
        self._write({"missions": [{"id": "R1", "units": [1],
                                   "objective": "甲点"}]})
        _poll_retask(self.runner, self.live, self.state, 7)
        self.assertEqual(len(self.runner.calls), 1)
        result = self._result()
        self.assertEqual(result["tick"], 7)
        self.assertEqual(result["applied"], ["R1"])
        self.assertEqual(result["rejected"], [])

    def test_unchanged_file_is_not_reapplied(self):
        self._write({"missions": [{"id": "R1", "units": [1],
                                   "objective": "甲点"}]})
        _poll_retask(self.runner, self.live, self.state, 7)
        _poll_retask(self.runner, self.live, self.state, 8)
        self.assertEqual(len(self.runner.calls), 1, "同一份文件不该重复下发")

    def test_rejected_reasons_survive(self):
        self._write({"missions": [{"id": "R1", "units": [1],
                                   "objective": "missing"}]})
        _poll_retask(self.runner, self.live, self.state, 3)
        self.assertEqual(self._result()["rejected"],
                         [["R1", "找不到目标点"]])

    def test_bare_list_is_accepted(self):
        self._write([{"id": "R1", "units": [1], "objective": "甲点"}])
        _poll_retask(self.runner, self.live, self.state, 4)
        self.assertEqual(self.runner.calls[0], [{"id": "R1", "units": [1],
                                                 "objective": "甲点"}])

    def test_malformed_json_does_not_raise(self):
        path = os.path.join(self.root, "retask.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        _poll_retask(self.runner, self.live, self.state, 4)
        self.assertEqual(self.runner.calls, [])
        result = self._result()
        self.assertEqual(result["applied"], [])
        self.assertIn("读取失败", result["rejected"][0][1])

    def test_non_list_missions_rejected(self):
        self._write({"missions": {"id": "R1"}})
        _poll_retask(self.runner, self.live, self.state, 4)
        self.assertEqual(self.runner.calls, [])
        self.assertIn("必须是列表", self._result()["rejected"][0][1])


if __name__ == "__main__":
    unittest.main()
