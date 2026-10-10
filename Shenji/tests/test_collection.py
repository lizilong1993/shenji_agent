from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from shenji_dataset.collection import CollectionDependencies, run_collection_job
from shenji_dataset.jobs import CollectionJob
from shenji_dataset.release import validate_release, verify_release


def make_job(max_steps: int = 3) -> CollectionJob:
    return CollectionJob(
        episode_id="shenji_fake_001",
        scenario_id="fake_scenario",
        map_id="fake_map",
        scenario_path="scenarios/fake.json",
        basic_path="maps/fake/basic.json",
        cost_path="maps/fake/cost.pickle",
        see_path="maps/fake/see.npz",
        see_key="data",
        seed=7,
        red_policy_code="red_fake",
        red_policy_profile="baseline",
        red_agent_class="fake:Agent",
        red_artifacts_dir=None,
        blue_policy_code="blue_fake",
        blue_policy_profile="baseline",
        blue_agent_class="fake:Agent",
        blue_artifacts_dir=None,
        max_steps=max_steps,
    )


class FakeAgent:
    def __init__(self, actor: int, obj_id: int) -> None:
        self.actor = actor
        self.obj_id = obj_id
        self.setup_info = None
        self.reset_called = False

    def setup(self, setup_info: dict) -> None:
        self.setup_info = setup_info

    def step(self, observation: dict) -> list[dict]:
        return [
            {
                "actor": self.actor,
                "obj_id": self.obj_id,
                "type": 5,
                "target_pos": 700,
            }
        ]

    def reset(self) -> None:
        self.reset_called = True


class FakeEnv:
    def __init__(self, fail_on_step: bool = False) -> None:
        self.step_count = 0
        self.fail_on_step = fail_on_step
        self.reset_called = False

    def _state(self, accepted_actions: list[dict] | None = None) -> dict:
        operators = [
            {"obj_id": 101, "color": 0, "type": 1, "sub_type": 0, "cur_hex": 500 + self.step_count, "blood": 10},
            {"obj_id": 201, "color": 1, "type": 1, "sub_type": 1, "cur_hex": 700, "blood": 10},
        ]
        truth = {
            "time": {"cur_step": self.step_count, "stage": 2},
            "operators": operators,
            "communication": [],
            "actions": copy.deepcopy(accepted_actions or []),
        }
        red = {"time": truth["time"], "operators": [operators[0]]}
        blue = {"time": truth["time"], "operators": [operators[1]]}
        return {-1: truth, 0: red, 1: blue}

    def setup(self, env_step_info: dict) -> dict:
        self.env_step_info = env_step_info
        return self._state()

    def step(self, actions: list[dict]) -> tuple[dict, bool]:
        if self.fail_on_step:
            raise RuntimeError("synthetic environment failure")
        self.step_count += 1
        state = self._state(actions)
        state[0]["reward"] = 2.0
        state[1]["reward"] = 1.0
        return state, self.step_count >= 2

    def reset(self) -> None:
        self.reset_called = True


class CollectionRunnerTests(unittest.TestCase):
    def test_successful_job_writes_raw_labels_release_and_status(self) -> None:
        env = FakeEnv()
        red = FakeAgent(1, 101)
        blue = FakeAgent(11, 201)
        dependencies = CollectionDependencies(
            env_factory=lambda: env,
            red_agent_factory=lambda: red,
            blue_agent_factory=lambda: blue,
            asset_loader=lambda job: {
                "scenario_data": {"scenario_id": job.scenario_id},
                "basic_data": {"map": job.map_id},
                "cost_data": [[], [], [], []],
                "see_data": [[1]],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = run_collection_job(
                make_job(),
                dependencies,
                root / "raw",
                canonical_root=root / "canonical",
            )
            episode_dir = root / "raw" / "shenji_fake_001"
            raw = json.loads((episode_dir / "raw_episode.json").read_text(encoding="utf-8"))
            plan = json.loads((episode_dir / "intent_plan.json").read_text(encoding="utf-8"))
            release_dir = root / "canonical" / "shenji_fake_001"

            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["termination_reason"], "environment_done")
            self.assertEqual(len(raw["observations"]["truth"]), 3)
            self.assertEqual(len(raw["observations"]["red"]), 3)
            self.assertEqual(len(raw["observations"]["blue"]), 3)
            self.assertEqual(len(raw["actions"]), 4)
            self.assertEqual(len(plan["segments"]), 8)
            self.assertEqual(verify_release(release_dir), [])
            self.assertEqual(validate_release(release_dir), [])

        self.assertTrue(env.reset_called)
        self.assertTrue(red.reset_called)
        self.assertTrue(blue.reset_called)

    def test_completed_job_is_skipped_without_constructing_runtime(self) -> None:
        calls = {"env": 0}

        def env_factory() -> FakeEnv:
            calls["env"] += 1
            return FakeEnv()

        dependencies = CollectionDependencies(
            env_factory=env_factory,
            red_agent_factory=lambda: FakeAgent(1, 101),
            blue_agent_factory=lambda: FakeAgent(11, 201),
            asset_loader=lambda job: {
                "scenario_data": {},
                "basic_data": {},
                "cost_data": [],
                "see_data": [],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_collection_job(make_job(), dependencies, root / "raw", root / "canonical")
            second = run_collection_job(make_job(), dependencies, root / "raw", root / "canonical")

        self.assertEqual(first["state"], "completed")
        self.assertEqual(second["state"], "completed")
        self.assertTrue(second["skipped"])
        self.assertEqual(calls["env"], 1)

    def test_failure_is_recorded_and_does_not_create_immutable_raw_episode(self) -> None:
        env = FakeEnv(fail_on_step=True)
        dependencies = CollectionDependencies(
            env_factory=lambda: env,
            red_agent_factory=lambda: FakeAgent(1, 101),
            blue_agent_factory=lambda: FakeAgent(11, 201),
            asset_loader=lambda job: {
                "scenario_data": {},
                "basic_data": {},
                "cost_data": [],
                "see_data": [],
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = run_collection_job(make_job(), dependencies, root / "raw", root / "canonical")
            episode_dir = root / "raw" / "shenji_fake_001"
            persisted = json.loads((episode_dir / "status.json").read_text(encoding="utf-8"))

            self.assertEqual(status["state"], "failed")
            self.assertEqual(persisted["error_type"], "RuntimeError")
            self.assertFalse((episode_dir / "raw_episode.json").exists())
            self.assertTrue((episode_dir / "partial_raw_episode.json").exists())

        self.assertTrue(env.reset_called)


if __name__ == "__main__":
    unittest.main()
