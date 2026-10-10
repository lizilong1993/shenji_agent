from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shenji_dataset.recorder import EpisodeRecorder


class EpisodeRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "episode_id": "episode_001",
            "scenario_id": "scenario_1231",
            "map_id": "map_123",
            "seed": 7,
            "policy_red": "rule_v1",
            "policy_blue": "rule_v2",
        }
        self.truth = {"time": {"cur_step": 3}, "operators": [{"obj_id": 1001}]}
        self.red = {"time": {"cur_step": 3}, "operators": [{"obj_id": 1001}]}
        self.blue = {"time": {"cur_step": 3}, "operators": []}

    def test_records_truth_red_blue_and_actions_without_mutating_inputs(self) -> None:
        action = {"type": 1, "actor": 1, "obj_id": 1001, "move_path": [501, 502]}
        recorder = EpisodeRecorder(self.metadata)

        recorder.record({-1: self.truth, 0: self.red, 1: self.blue}, [action])
        raw = recorder.to_raw_episode()

        self.assertEqual(raw["metadata"]["dataset"], "Shenji")
        self.assertEqual(raw["metadata"]["simulation_backend"], "MiaoSuan")
        self.assertEqual(raw["observations"]["truth"], [self.truth])
        self.assertEqual(raw["observations"]["red"], [self.red])
        self.assertEqual(raw["observations"]["blue"], [self.blue])
        self.assertEqual(raw["actions"][0]["cur_step"], 3)
        self.assertNotIn("cur_step", action)

    def test_accepts_named_side_keys(self) -> None:
        recorder = EpisodeRecorder(self.metadata)

        recorder.record({"truth": self.truth, "red": self.red, "blue": self.blue})

        self.assertEqual(len(recorder.to_raw_episode()["observations"]["truth"]), 1)

    def test_rejects_missing_observation_view(self) -> None:
        recorder = EpisodeRecorder(self.metadata)

        with self.assertRaisesRegex(ValueError, "blue"):
            recorder.record({-1: self.truth, 0: self.red})

    def test_saves_deterministic_utf8_json(self) -> None:
        recorder = EpisodeRecorder(self.metadata)
        recorder.record({-1: self.truth, 0: self.red, 1: self.blue})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode" / "raw_episode.json"
            recorder.save(path)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, recorder.to_raw_episode())


if __name__ == "__main__":
    unittest.main()
