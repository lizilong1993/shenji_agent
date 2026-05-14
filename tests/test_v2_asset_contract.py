from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from ai.artifacts import missing_required_assets, naming_mismatches, resolve_manifest_assets
from scripts.run_round005_smoke_pipeline import build_report, manifest_missing_files


class V2AssetContractTests(unittest.TestCase):
    def test_candidate_manifest_resolves_restored_checkpoint_aliases(self) -> None:
        artifact_dir = Path("ai/weights")
        manifest = json.loads((artifact_dir / "model_manifest.json").read_text(encoding="utf-8"))

        resolutions = resolve_manifest_assets(artifact_dir, manifest)

        self.assertTrue(resolutions["intent_lstm"].exists)
        self.assertEqual(resolutions["intent_lstm"].path.name, "intention_lstm.pth")
        self.assertFalse(resolutions["intent_lstm"].resolved_by_alias)
        self.assertTrue(resolutions["situation_gcn"].exists)
        self.assertEqual(resolutions["situation_gcn"].path.name, "situation_gcn.pth")

    def test_validators_report_naming_mismatch_instead_of_false_missing_assets(self) -> None:
        artifact_dir = Path("ai/weights")
        manifest = {
            "intent_lstm": "intent_lstm.pt",
            "situation_gcn": "situation_gcn.pt",
            "tactical_model": "tactical_model.pt",
        }

        resolutions = resolve_manifest_assets(artifact_dir, manifest)
        missing = missing_required_assets(resolutions)
        mismatches = naming_mismatches(resolutions)

        self.assertNotIn("intent_lstm", missing)
        self.assertNotIn("situation_gcn", missing)
        self.assertIn("intent_lstm", mismatches)
        self.assertIn("situation_gcn", mismatches)

    def test_tactical_checkpoint_restored_status_is_ready(self) -> None:
        missing = manifest_missing_files(Path("ai/weights"))

        self.assertNotIn("ai/weights/tactical_model.pt", missing)

    def test_rule_incumbent_assets_are_not_required_for_candidate_smoke(self) -> None:
        missing = manifest_missing_files(Path("ai/weights_incumbent_rule"))

        self.assertEqual(missing, [])

    def test_smoke_preflight_reports_asset_contract_details(self) -> None:
        report = build_report(Path("ai/weights"), Path("ai/weights_incumbent_rule"))

        self.assertEqual(report["status"], "ready")
        self.assertIn("candidate_assets", report)
        self.assertIn("incumbent_assets", report)
        self.assertFalse(report["incumbent_assets"]["models_required"])
        self.assertNotIn("tactical_model", report["candidate_assets"]["missing_required"])
        self.assertEqual(report["candidate_assets"]["naming_mismatches"], {})

    def test_manifest_missing_file_remains_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = manifest_missing_files(Path(temp_dir))

        self.assertEqual(len(missing), 1)
        self.assertTrue(missing[0].endswith("model_manifest.json"))


if __name__ == "__main__":
    unittest.main()
