from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shenji_dataset.release import load_release, verify_release, write_release


TABLE_NAMES = (
    "episodes",
    "entities",
    "states",
    "observations",
    "actions",
    "events",
    "relations",
    "intent_segments",
)


def make_tables() -> dict[str, list[dict]]:
    tables = {name: [] for name in TABLE_NAMES}
    tables["episodes"] = [
        {
            "episode_id": "episode_001",
            "dataset": "Shenji",
            "schema_version": "shenji_schema_v1",
        }
    ]
    tables["entities"] = [
        {"episode_id": "episode_001", "entity_id": "1001", "side": "red"}
    ]
    return tables


class ReleaseTests(unittest.TestCase):
    def test_writes_deterministic_jsonl_and_hashed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = write_release(make_tables(), output, formats=("jsonl",))

            self.assertEqual(manifest["dataset"], "Shenji")
            self.assertEqual(manifest["schema_version"], "shenji_schema_v1")
            self.assertEqual(len(manifest["artifacts"]), 8)
            self.assertEqual(verify_release(output), [])
            rows = load_release(output)

        self.assertEqual(rows, make_tables())

    def test_verifier_detects_tampered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_release(make_tables(), output, formats=("jsonl",))
            path = output / "entities.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

            errors = verify_release(output)

        self.assertTrue(any("SHA-256" in error for error in errors))
        self.assertTrue(any("row count" in error for error in errors))

    def test_rejects_unknown_format_without_partial_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            with self.assertRaisesRegex(ValueError, "unsupported release format"):
                write_release(make_tables(), output, formats=("csv",))

            self.assertFalse((output / "manifest.json").exists())

    def test_manifest_is_valid_utf8_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_release(make_tables(), output, formats=("jsonl",))

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["table_order"], list(TABLE_NAMES))


if __name__ == "__main__":
    unittest.main()
