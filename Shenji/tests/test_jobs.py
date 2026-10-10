from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from shenji_dataset.jobs import expand_jobs, load_collection_manifest


ROOT = Path(__file__).resolve().parents[1]


def make_manifest() -> dict:
    return {
        "dataset": "Shenji",
        "version": "1.0.0",
        "simulation_backend": "MiaoSuan",
        "data_root": "../land_wargame_sdk/Data/Data",
        "output_root": "raw/pilot_v1",
        "max_steps": 100,
        "scenarios": [
            {
                "scenario_id": "1231",
                "map_id": "map_123",
                "scenario_path": "scenarios/1231.json",
                "basic_path": "maps/map_123/basic.json",
                "cost_path": "maps/map_123/cost.pickle",
                "see_path": "maps/map_123/123see.npz",
                "see_key": "data",
            },
            {
                "scenario_id": "1531",
                "map_id": "map_153",
                "scenario_path": "scenarios/1531.json",
                "basic_path": "maps/map_153/basic.json",
                "cost_path": "maps/map_153/cost.pickle",
                "see_path": "maps/map_153/153see.npz",
                "see_key": "data",
            },
        ],
        "policies": [
            {
                "code": "baseline_v1",
                "profile": "baseline",
                "agent_class": "ai.agent:Agent",
                "artifacts_dir": "../ai/weights",
            },
            {
                "code": "challenger_v1",
                "profile": "challenger",
                "agent_class": "ai.agent:Agent",
                "artifacts_dir": "../ai/weights",
            },
        ],
        "matchups": [
            {"red": "baseline_v1", "blue": "baseline_v1"},
            {"red": "challenger_v1", "blue": "baseline_v1"},
        ],
        "seeds": [22, 11],
    }


class CollectionJobTests(unittest.TestCase):
    def test_expands_matrix_with_stable_unique_episode_ids(self) -> None:
        jobs = expand_jobs(make_manifest())

        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({job.episode_id for job in jobs}), 8)
        self.assertEqual([job.episode_id for job in jobs], sorted(job.episode_id for job in jobs))
        self.assertTrue(all(job.episode_id.startswith("shenji_") for job in jobs))

    def test_manifest_order_does_not_change_job_identity_or_order(self) -> None:
        original = make_manifest()
        reordered = deepcopy(original)
        reordered["scenarios"].reverse()
        reordered["policies"].reverse()
        reordered["matchups"].reverse()
        reordered["seeds"].reverse()

        first = [job.episode_id for job in expand_jobs(original)]
        second = [job.episode_id for job in expand_jobs(reordered)]

        self.assertEqual(first, second)

    def test_rejects_matchup_with_unknown_policy(self) -> None:
        manifest = make_manifest()
        manifest["matchups"][0]["red"] = "missing_policy"

        with self.assertRaisesRegex(ValueError, "unknown red policy"):
            expand_jobs(manifest)

    def test_loads_utf8_manifest_and_actual_pilot_config_is_valid(self) -> None:
        manifest = make_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            loaded = load_collection_manifest(path)

        self.assertEqual(loaded, manifest)
        actual = load_collection_manifest(ROOT / "configs" / "pilot_manifest_v1.json")
        self.assertEqual(len(expand_jobs(actual)), 4)


if __name__ == "__main__":
    unittest.main()
