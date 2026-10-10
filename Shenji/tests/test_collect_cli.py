from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from shenji_dataset.collect_cli import main
from shenji_dataset.coverage import build_coverage_report


ROOT = Path(__file__).resolve().parents[1]


def write_completed_episode(root: Path, episode_id: str, intent: str) -> None:
    episode = root / episode_id
    episode.mkdir(parents=True)
    (episode / "status.json").write_text(
        json.dumps({"episode_id": episode_id, "state": "completed"}), encoding="utf-8"
    )
    (episode / "job.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "scenario_id": "1231",
                "red_policy_code": "red_v1",
                "blue_policy_code": "blue_v1",
            }
        ),
        encoding="utf-8",
    )
    (episode / "intent_plan.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "primary_intent": intent,
                        "label_kind": "realized",
                        "evidence_grade": "A",
                        "status": "completed",
                        "outcome": "achieved",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class CollectionCliTests(unittest.TestCase):
    def test_coverage_counts_jobs_labels_policies_and_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            write_completed_episode(raw_root, "episode_1", "SEIZE")
            write_completed_episode(raw_root, "episode_2", "SEIZE")
            failed = raw_root / "episode_3"
            failed.mkdir()
            (failed / "status.json").write_text(
                json.dumps({"episode_id": "episode_3", "state": "failed"}),
                encoding="utf-8",
            )

            report = build_coverage_report(raw_root)

        self.assertEqual(report["jobs"]["total"], 3)
        self.assertEqual(report["jobs"]["completed"], 2)
        self.assertEqual(report["jobs"]["failed"], 1)
        self.assertEqual(report["segments"]["total"], 2)
        self.assertEqual(report["segments"]["by_intent"]["SEIZE"], 2)
        self.assertEqual(report["scenarios"]["1231"], 2)
        self.assertEqual(report["policies"]["red_v1"], 2)

    def test_dry_run_expands_actual_manifest_without_importing_runtime(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "run",
                    "--manifest",
                    str(ROOT / "configs" / "pilot_manifest_v1.json"),
                    "--dry-run",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["job_count"], 4)
        self.assertEqual(len(payload["jobs"]), 4)
        self.assertEqual(stderr.getvalue(), "")

    def test_probe_returns_machine_readable_failure_instead_of_mock_fallback(self) -> None:
        stdout = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            with redirect_stdout(stdout):
                code = main(
                    [
                        "probe",
                        "--manifest",
                        str(ROOT / "configs" / "pilot_manifest_v1.json"),
                        "--project-root",
                        str(project_root),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(payload["ready"])
        self.assertIn("environment", payload["checks"])
        self.assertNotIn("MockTrainEnv", json.dumps(payload))

    def test_run_uses_runner_and_respects_limit(self) -> None:
        stdout = StringIO()
        completed = {
            "dataset": "Shenji",
            "episode_id": "patched",
            "state": "completed",
            "skipped": False,
        }
        with patch(
            "shenji_dataset.collect_cli.dependencies_for_job", return_value=object()
        ) as dependency_builder, patch(
            "shenji_dataset.collect_cli.run_collection_job", return_value=completed
        ) as runner, redirect_stdout(stdout):
            code = main(
                [
                    "run",
                    "--manifest",
                    str(ROOT / "configs" / "pilot_manifest_v1.json"),
                    "--limit",
                    "1",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(dependency_builder.call_count, 1)
        self.assertEqual(runner.call_count, 1)
        self.assertIn('"state": "completed"', stdout.getvalue())

    def test_coverage_command_writes_json_report(self) -> None:
        stdout = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            write_completed_episode(raw_root, "episode_1", "RECON")
            output = root / "coverage.json"
            with redirect_stdout(stdout):
                code = main(
                    [
                        "coverage",
                        "--raw-root",
                        str(raw_root),
                        "--output",
                        str(output),
                    ]
                )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(persisted["segments"]["by_intent"]["RECON"], 1)
        self.assertEqual(json.loads(stdout.getvalue()), persisted)


if __name__ == "__main__":
    unittest.main()
