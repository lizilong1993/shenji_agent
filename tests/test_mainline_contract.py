from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_fake_torch() -> None:
    if "torch" in sys.modules:
        return
    torch_module = types.ModuleType("torch")
    torch_module.float32 = "float32"
    torch_module.long = "long"
    torch_module.tensor = lambda *args, **kwargs: None
    utils_module = types.ModuleType("torch.utils")
    data_module = types.ModuleType("torch.utils.data")

    class Dataset:
        pass

    data_module.Dataset = Dataset
    utils_module.data = data_module
    torch_module.utils = utils_module
    sys.modules["torch"] = torch_module
    sys.modules["torch.utils"] = utils_module
    sys.modules["torch.utils.data"] = data_module


class MainlineContractTest(unittest.TestCase):
    def test_selfplay_builds_ordered_unique_first_map_list(self) -> None:
        from training.selfplay import _build_paired_map_list, _collect_scenario_candidates

        with tempfile.TemporaryDirectory() as tmp:
            scenario_root = Path(tmp)
            bucket_lookup = {}
            for idx in range(1, 121):
                (scenario_root / f"scenario_{idx}.json").write_text("{}", encoding="utf-8")
                bucket_lookup[idx] = "complex" if idx % 3 == 0 else ("medium" if idx % 3 == 1 else "simple")
            candidates = _collect_scenario_candidates(scenario_root, bucket_lookup)

        random.seed(42)
        payload = _build_paired_map_list(candidates, 100)
        items = payload["items"]

        self.assertEqual(payload["sampling_protocol"], "unique_first_then_replacement")
        self.assertEqual(payload["candidate_pool_size"], 120)
        self.assertFalse(payload["duplicates_used"])
        self.assertEqual(len(items), 100)
        self.assertEqual([item["order_index"] for item in items], list(range(100)))
        self.assertEqual(len({item["scenario"] for item in items}), 100)
        self.assertTrue(
            all({"scenario", "map_id", "bucket", "order_index"}.issubset(item.keys()) for item in items)
        )

    def test_selfplay_backfills_duplicates_only_after_unique_pool(self) -> None:
        from training.selfplay import _build_paired_map_list, _collect_scenario_candidates

        with tempfile.TemporaryDirectory() as tmp:
            scenario_root = Path(tmp)
            bucket_lookup = {1: "simple", 2: "medium", 3: "complex"}
            for idx in bucket_lookup:
                (scenario_root / f"scenario_{idx}.json").write_text("{}", encoding="utf-8")
            candidates = _collect_scenario_candidates(scenario_root, bucket_lookup)

        random.seed(7)
        payload = _build_paired_map_list(candidates, 10)
        items = payload["items"]

        self.assertTrue(payload["duplicates_used"])
        self.assertEqual(payload["candidate_pool_size"], 3)
        self.assertEqual(payload["unique_item_count_before_replacement"], 3)
        self.assertEqual(len(items), 10)
        self.assertEqual(len({item["scenario"] for item in items[:3]}), 3)
        self.assertEqual(
            {item["scenario"] for item in items},
            {"scenario_1.json", "scenario_2.json", "scenario_3.json"},
        )

    def test_selfplay_resolves_incumbent_profile_from_snapshot_role(self) -> None:
        from training.selfplay import _participant_payload, _resolve_incumbent_profile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            learned_dir = root / "learned"
            learned_dir.mkdir()
            (learned_dir / "model_manifest.json").write_text(
                json.dumps({"snapshot_role": "accepted_learned_mainline", "generated_at": "20260423_000000"}),
                encoding="utf-8",
            )
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "model_manifest.json").write_text(
                json.dumps({"snapshot_role": "bootstrap_rule_baseline"}),
                encoding="utf-8",
            )

            self.assertEqual(_resolve_incumbent_profile(str(learned_dir), None, "baseline"), "incumbent")
            self.assertEqual(_resolve_incumbent_profile(str(baseline_dir), None, "baseline"), "baseline")
            participant = _participant_payload(str(learned_dir), "incumbent")
            self.assertEqual(participant["profile"], "incumbent")
            self.assertEqual(participant["snapshot_role"], "accepted_learned_mainline")

    def test_selfplay_prefers_train_env_backend_over_mock(self) -> None:
        from training.selfplay import _resolve_train_env_class

        class TrainEnv:
            pass

        def fake_import(name: str) -> object:
            if name == "land_wargame_train_env":
                raise ImportError("linux alias missing")
            if name == "train_env":
                return types.SimpleNamespace(TrainEnv=TrainEnv)
            raise AssertionError(f"unexpected import: {name}")

        env_class, backend = _resolve_train_env_class(fake_import)
        self.assertIs(env_class, TrainEnv)
        self.assertEqual(backend, "train_env.TrainEnv")

    def test_selfplay_falls_back_to_mock_only_when_real_env_missing(self) -> None:
        from training.selfplay import _resolve_train_env_class

        class MockTrainEnv:
            pass

        def fake_import(name: str) -> object:
            if name in {"land_wargame_train_env", "train_env"}:
                raise ImportError("missing")
            if name == "mock_train_env":
                return types.SimpleNamespace(MockTrainEnv=MockTrainEnv)
            raise AssertionError(f"unexpected import: {name}")

        env_class, backend = _resolve_train_env_class(fake_import)
        self.assertIs(env_class, MockTrainEnv)
        self.assertEqual(backend, "mock_train_env.MockTrainEnv")

    def test_tactical_dataset_uses_explicit_label_column(self) -> None:
        _install_fake_torch()
        from training.data import TacticalPositionDataset

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "toy.csv"
            csv_path.write_text(
                "ID,样本标签,整体移动方向,数值特征\n"
                "0,1,9,0.5\n"
                "1,0,8,1.5\n",
                encoding="utf-8",
            )
            dataset = TacticalPositionDataset(csv_path, label_column="样本标签")
        self.assertEqual(dataset.labels.tolist(), [1.0, 0.0])
        self.assertEqual(dataset.label_column, "样本标签")

    def test_map_id_resolution_can_be_scoped_to_filename_segment(self) -> None:
        _install_fake_torch()
        from training.data import resolve_map_id_by_strategy

        self.assertEqual(
            resolve_map_id_by_strategy(
                "20210109101817255-201033039601-Liquidator-DemoAI-t83lvwyW.zip",
                [83, 9601],
                "suffix",
                filename_segment_index=1,
            ),
            9601,
        )
        self.assertEqual(
            resolve_map_id_by_strategy(
                "20210730145902084-2110431553-zBv9bHRr.json",
                [53, 84],
                "suffix",
                filename_segment_index=1,
            ),
            53,
        )

    def test_selfplay_merge_policy_stats_keeps_model_load_diagnostics(self) -> None:
        from training.selfplay import _merge_policy_stats

        merged = _merge_policy_stats(
            {
                "policy_profile": "incumbent",
                "loaded_from": "/tmp/strongest",
                "intent_model_loaded": True,
                "situation_model_loaded": True,
                "counts": {
                    "move": {"total": 1, "learned": 1},
                    "shoot": {"total": 0, "learned": 0},
                },
                "score_means": {
                    "move": {"samples": 1, "intent_bias": 1.0, "situation_score": 0.4, "combined_score": 2.0},
                    "shoot": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "combined_score": 0.0},
                },
            },
            {
                "policy_profile": "incumbent",
                "loaded_from": "/tmp/strongest",
                "intent_model_loaded": True,
                "situation_model_loaded": True,
                "counts": {
                    "move": {"total": 1, "learned": 1},
                    "shoot": {"total": 1, "learned": 1},
                },
                "score_means": {
                    "move": {"samples": 1, "intent_bias": 3.0, "situation_score": 0.6, "combined_score": 4.0},
                    "shoot": {"samples": 1, "intent_bias": 2.0, "situation_score": 0.7, "combined_score": 5.0},
                },
            },
        )
        self.assertEqual(merged["policy_profiles"], ["incumbent"])
        self.assertEqual(merged["loaded_from_dirs"], ["/tmp/strongest"])
        self.assertEqual(merged["intent_model_loaded_count"], 2)
        self.assertEqual(merged["situation_model_loaded_count"], 2)

    def test_summarize_matches_reports_policy_coverage(self) -> None:
        _install_fake_torch()
        from training.evaluate import summarize_matches

        matches = [
            {
                "paired_score": 0.75,
                "bucket": "complex",
                "candidate_policy_stats": {
                    "counts": {
                        "move": {"total": 2, "learned": 1},
                        "shoot": {"total": 2, "learned": 2},
                    },
                    "score_means": {
                        "move": {"samples": 1, "intent_bias": 1.0, "situation_score": 0.5, "combined_score": 3.0},
                        "shoot": {"samples": 1, "intent_bias": 2.0, "situation_score": 0.7, "combined_score": 5.0},
                    },
                },
            }
        ]
        summary = summarize_matches(matches)
        self.assertAlmostEqual(summary["policy"]["learned_path_coverage"], 0.75)
        self.assertAlmostEqual(summary["policy"]["move_path_coverage"], 0.5)
        self.assertAlmostEqual(summary["policy"]["shoot_path_coverage"], 1.0)

    def test_promotion_decision_enforces_sota_stop_guards(self) -> None:
        _install_fake_torch()
        from training.evaluate import promotion_decision

        summary = {
            "overall": {"point_estimate": 0.62, "ci95": (0.56, 0.67), "elo": 84.0, "count": 100},
            "buckets": {
                "complex": {"point_estimate": 0.61, "ci95": (0.56, 0.66), "count": 50},
                "medium": {"point_estimate": 0.58, "ci95": (0.52, 0.63), "count": 30},
                "simple": {"point_estimate": 0.57, "ci95": (0.50, 0.64), "count": 20},
            },
            "policy": {
                "learned_path_coverage": 0.34,
                "move_path_coverage": 0.24,
                "shoot_path_coverage": 0.21,
            },
        }
        decision = promotion_decision(
            summary,
            {
                "overall_point_estimate_min": 0.60,
                "overall_ci_lower_min": 0.55,
                "complex_point_estimate_min": 0.60,
                "complex_ci_lower_min": 0.55,
                "min_learned_path_coverage": 0.30,
                "min_move_path_coverage": 0.20,
                "min_shoot_path_coverage": 0.20,
                "elo_delta_lower_min": 35.0,
                "max_bucket_regression": 0.03,
                "max_exploiter_win_rate": 0.45,
            },
            {
                "aggressive_exploiter": {"win_rate_vs_candidate": 0.42},
                "occupy_exploiter": {"win_rate_vs_candidate": 0.40},
            },
        )
        self.assertTrue(decision["accepted"])
        self.assertGreaterEqual(decision["elo_delta_lower_bound"], 35.0)
        self.assertAlmostEqual(decision["max_exploiter_win_rate"], 0.42)

        blocked = promotion_decision(
            summary,
            {
                "overall_point_estimate_min": 0.60,
                "overall_ci_lower_min": 0.55,
                "complex_point_estimate_min": 0.60,
                "complex_ci_lower_min": 0.55,
                "min_learned_path_coverage": 0.30,
                "min_move_path_coverage": 0.20,
                "min_shoot_path_coverage": 0.20,
                "elo_delta_lower_min": 35.0,
                "max_bucket_regression": 0.03,
                "max_exploiter_win_rate": 0.45,
            },
            {
                "aggressive_exploiter": {"win_rate_vs_candidate": 0.48},
            },
        )
        self.assertFalse(blocked["accepted"])
        self.assertIn("exploiter_win_rate", blocked["failures"])

    def test_selfplay_rejects_same_candidate_and_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "output_root": "experiments/shenji_agent",
                        "league_dir": "league",
                        "paired_matches": 2,
                        "max_steps": 10,
                        "promotion_gate": {
                            "overall_point_estimate_min": 0.5,
                            "overall_ci_lower_min": 0.4,
                            "complex_point_estimate_min": 0.5,
                            "complex_ci_lower_min": 0.4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.selfplay",
                    "--config",
                    str(config_path),
                    "--candidate",
                    "ai/weights",
                    "--incumbent",
                    "ai/weights",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("candidate and incumbent must point to different artifact directories", proc.stdout + proc.stderr)

    def test_report_progress_returns_none_for_missing_root(self) -> None:
        missing_root = REPO_ROOT / "experiments" / "missing_for_test"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/report_progress.py",
                "--experiments-root",
                str(missing_root),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["event"], "none")
        self.assertIsNone(payload["run_dir"])


if __name__ == "__main__":
    unittest.main()
