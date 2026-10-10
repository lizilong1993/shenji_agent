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
        # 第 52 轮：完整训练后 manifest 声明的是**规范名** `.pt`
        # （I07 的注释已说明：命名解析在**发布时**处理，manifest 只声明本次训练写出的规范名）。
        # 故这里断言规范名；历史 `.pth` 的兼容性由下面的别名用例负责。
        self.assertEqual(resolutions["intent_lstm"].path.name, "intent_lstm.pt")
        self.assertFalse(resolutions["intent_lstm"].resolved_by_alias)
        self.assertTrue(resolutions["situation_gcn"].exists)
        self.assertEqual(resolutions["situation_gcn"].path.name, "situation_gcn.pt")

    def test_validators_report_naming_mismatch_instead_of_false_missing_assets(self) -> None:
        """校验器要能区分**「缺文件」**与**「命名不一致」**两件事。

        ⚠ 第 52 轮调整：原用例要求 `ai/weights/` 里"有别名、没规范名"才能报 mismatch。
        完整训练后 `ai/weights/` **同时有**规范名与历史别名，故那里**本就不该**报 mismatch。
        若把那两条 `assertIn` 直接删掉，就等于放弃了这个校验职责 ——
        故改为在**临时目录**里只放历史别名，用它继续守住"命名不一致要被报出来"。
        另有断言保证：命名不一致**不会**被误报成"缺文件"（这正是本用例的原意）。
        """
        manifest = {
            "intent_lstm": "intent_lstm.pt",
            "situation_gcn": "situation_gcn.pt",
            "tactical_model": "tactical_model.pt",
        }

        # (a) 规范名齐备时：不得报缺、不得报 mismatch
        ok_dir = Path("ai/weights")
        ok = resolve_manifest_assets(ok_dir, manifest)
        self.assertNotIn("intent_lstm", missing_required_assets(ok))
        self.assertNotIn("situation_gcn", missing_required_assets(ok))
        self.assertNotIn("intent_lstm", naming_mismatches(ok))
        self.assertNotIn("situation_gcn", naming_mismatches(ok))

        # (b) 只放**历史别名**时：应报 mismatch，且**不得**报缺文件
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp)
            for name in ("intention_lstm.pth", "situation_gcn.pth", "tactical_model.pt"):
                (legacy / name).write_bytes(b"x")
            res = resolve_manifest_assets(legacy, manifest)
            missing = missing_required_assets(res)
            mismatches = naming_mismatches(res)
            self.assertNotIn("intent_lstm", missing,
                             "别名可解析时不应报缺文件")
            self.assertNotIn("situation_gcn", missing,
                             "别名可解析时不应报缺文件")
            self.assertIn("intent_lstm", mismatches, "别名未解析成 mismatch")
            self.assertIn("situation_gcn", mismatches, "别名未解析成 mismatch")

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
