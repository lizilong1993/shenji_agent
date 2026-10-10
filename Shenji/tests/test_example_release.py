from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from shenji_dataset.cli import main
from shenji_dataset.normalizer import CANONICAL_TABLES, normalize_episode
from shenji_dataset.ontology import load_ontology
from shenji_dataset.validation import validate_tables


ROOT = Path(__file__).resolve().parents[1]


class ExampleReleaseTests(unittest.TestCase):
    def test_example_normalizes_and_validates(self) -> None:
        raw = json.loads((ROOT / "examples" / "raw_episode.json").read_text(encoding="utf-8"))
        plan = json.loads((ROOT / "examples" / "intent_plan.json").read_text(encoding="utf-8"))

        tables = normalize_episode(raw, plan)

        self.assertEqual(tuple(tables), CANONICAL_TABLES)
        self.assertEqual(len(tables["intent_segments"]), 3)
        self.assertEqual(validate_tables(tables, load_ontology()), [])

    def test_cli_build_verify_and_validate_example(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                build_code = main(
                    [
                        "build",
                        "--raw",
                        str(ROOT / "examples" / "raw_episode.json"),
                        "--intent-plan",
                        str(ROOT / "examples" / "intent_plan.json"),
                        "--output",
                        str(output),
                    ]
                )
                verify_code = main(["verify", str(output)])
                validate_code = main(["validate", str(output)])

            self.assertEqual((build_code, verify_code, validate_code), (0, 0, 0))
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("release verification passed", stdout.getvalue())
            self.assertIn("dataset validation passed", stdout.getvalue())

    def test_schema_and_benchmark_metadata_cover_all_tables_and_core_tasks(self) -> None:
        schema = json.loads(
            (ROOT / "schema" / "shenji_schema_v1.json").read_text(encoding="utf-8")
        )
        benchmarks = json.loads(
            (ROOT / "benchmarks" / "tasks_v1.json").read_text(encoding="utf-8")
        )

        self.assertEqual(tuple(schema["table_order"]), CANONICAL_TABLES)
        task_codes = {task["code"] for task in benchmarks["tasks"]}
        self.assertTrue(
            {
                "state_estimation",
                "trajectory_prediction",
                "activity_recognition",
                "intent_recognition",
                "intent_transition_forecasting",
            }.issubset(task_codes)
        )


if __name__ == "__main__":
    unittest.main()
