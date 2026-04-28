from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_test_stubs() -> None:
    from tests.test_mainline_contract import _install_fake_numpy, _install_fake_torch

    _install_fake_numpy()
    _install_fake_torch()


class PromotionArtifactContractTests(unittest.TestCase):
    def test_promotion_decision_reports_coverage_gate_failures_explicitly(self) -> None:
        _install_test_stubs()
        from training.evaluate import promotion_decision

        summary = {
            "overall": {"point_estimate": 0.62, "ci95": (0.57, 0.67), "elo": 84.0, "count": 100},
            "buckets": {
                "complex": {"point_estimate": 0.61, "ci95": (0.56, 0.66), "count": 50},
            },
            "policy": {
                "learned_path_coverage": 0.10,  # below 0.30
                "move_path_coverage": 0.05,  # below 0.20
                "shoot_path_coverage": 0.08,  # below 0.20
            },
        }
        gate_config = {
            "overall_point_estimate_min": 0.60,
            "overall_ci_lower_min": 0.55,
            "complex_point_estimate_min": 0.60,
            "complex_ci_lower_min": 0.55,
            "min_learned_path_coverage": 0.30,
            "min_move_path_coverage": 0.20,
            "min_shoot_path_coverage": 0.20,
            "elo_delta_lower_min": 35.0,
            "max_bucket_regression": 1.0,
            "max_exploiter_win_rate": 1.0,
        }

        decision = promotion_decision(summary, gate_config, None)

        self.assertFalse(decision["accepted"])
        self.assertIn("learned_path_coverage", decision["failures"])
        self.assertIn("move_path_coverage", decision["failures"])
        self.assertIn("shoot_path_coverage", decision["failures"])
        # coverage values should be reported in the decision
        self.assertAlmostEqual(decision["learned_path_coverage"], 0.10)
        self.assertAlmostEqual(decision["move_path_coverage"], 0.05)
        self.assertAlmostEqual(decision["shoot_path_coverage"], 0.08)

    def test_promotion_decision_with_missing_diagnostics_is_not_accepted(self) -> None:
        _install_test_stubs()
        from training.evaluate import promotion_decision

        summary = {
            "overall": {"point_estimate": 0.62, "ci95": (0.57, 0.67), "elo": 84.0, "count": 100},
            "buckets": {
                "complex": {"point_estimate": 0.61, "ci95": (0.56, 0.66), "count": 50},
            },
        }
        gate_config = {
            "overall_point_estimate_min": 0.60,
            "overall_ci_lower_min": 0.55,
            "complex_point_estimate_min": 0.60,
            "complex_ci_lower_min": 0.55,
            "min_learned_path_coverage": 0.30,
            "min_move_path_coverage": 0.20,
            "min_shoot_path_coverage": 0.20,
            "elo_delta_lower_min": 35.0,
            "max_bucket_regression": 1.0,
            "max_exploiter_win_rate": 1.0,
        }

        decision = promotion_decision(summary, gate_config, None)

        self.assertFalse(decision["accepted"])
        self.assertIn("learned_path_coverage", decision["failures"])
        self.assertIn("move_path_coverage", decision["failures"])
        self.assertIn("shoot_path_coverage", decision["failures"])

    def test_promotion_decision_with_coverage_zero_threshold_is_no_op(self) -> None:
        _install_test_stubs()
        from training.evaluate import promotion_decision

        summary = {
            "overall": {"point_estimate": 0.62, "ci95": (0.57, 0.67), "elo": 84.0, "count": 100},
            "buckets": {
                "complex": {"point_estimate": 0.61, "ci95": (0.56, 0.66), "count": 50},
            },
            "policy": {
                "learned_path_coverage": 0.0,
                "move_path_coverage": 0.0,
                "shoot_path_coverage": 0.0,
            },
        }
        gate_config = {
            "overall_point_estimate_min": 0.60,
            "overall_ci_lower_min": 0.55,
            "complex_point_estimate_min": 0.60,
            "complex_ci_lower_min": 0.55,
            "min_learned_path_coverage": 0.0,
            "min_move_path_coverage": 0.0,
            "min_shoot_path_coverage": 0.0,
            "elo_delta_lower_min": 35.0,
            "max_bucket_regression": 1.0,
            "max_exploiter_win_rate": 1.0,
        }

        decision = promotion_decision(summary, gate_config, None)

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["failures"], [])

    def test_promotion_decision_reports_bucket_regressions(self) -> None:
        _install_test_stubs()
        from training.evaluate import promotion_decision

        summary = {
            "overall": {"point_estimate": 0.62, "ci95": (0.57, 0.67), "elo": 84.0, "count": 100},
            "buckets": {
                "complex": {"point_estimate": 0.46, "ci95": (0.40, 0.52), "count": 50},
            },
            "policy": {
                "learned_path_coverage": 0.40,
                "move_path_coverage": 0.30,
                "shoot_path_coverage": 0.25,
            },
        }
        gate_config = {
            "overall_point_estimate_min": 0.60,
            "overall_ci_lower_min": 0.55,
            "complex_point_estimate_min": 0.60,
            "complex_ci_lower_min": 0.55,
            "min_learned_path_coverage": 0.30,
            "min_move_path_coverage": 0.20,
            "min_shoot_path_coverage": 0.20,
            "elo_delta_lower_min": 35.0,
            "max_bucket_regression": 0.03,
            "max_exploiter_win_rate": 1.0,
        }

        decision = promotion_decision(summary, gate_config, None)

        self.assertFalse(decision["accepted"])
        self.assertIn("complex_point_estimate", decision["failures"])
        self.assertIn("complex_ci_lower", decision["failures"])
        self.assertIn("bucket_regression", decision["failures"])

    def test_promotion_decision_passes_with_clean_coverage(self) -> None:
        _install_test_stubs()
        from training.evaluate import promotion_decision

        summary = {
            "overall": {"point_estimate": 0.62, "ci95": (0.57, 0.67), "elo": 84.0, "count": 100},
            "buckets": {
                "complex": {"point_estimate": 0.61, "ci95": (0.56, 0.66), "count": 50},
            },
            "policy": {
                "learned_path_coverage": 0.34,
                "move_path_coverage": 0.24,
                "shoot_path_coverage": 0.21,
            },
        }
        gate_config = {
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
        }

        exploiter_results = {
            "aggressive_exploiter": {"win_rate_vs_candidate": 0.42},
        }

        decision = promotion_decision(summary, gate_config, exploiter_results)

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["failures"], [])
        self.assertGreaterEqual(decision["elo_delta_lower_bound"], 35.0)
        self.assertAlmostEqual(decision["max_exploiter_win_rate"], 0.42)

    def test_summarize_matches_includes_policy_coverage_in_output(self) -> None:
        _install_test_stubs()
        from training.evaluate import summarize_matches

        matches = [
            {
                "paired_score": 0.75,
                "bucket": "complex",
                "candidate_policy_stats": {
                    "counts": {
                        "move": {"total": 4, "learned": 3},
                        "shoot": {"total": 2, "learned": 2},
                    },
                    "score_means": {
                        "move": {"samples": 3, "intent_bias": 1.0, "situation_score": 0.5, "combined_score": 3.0},
                        "shoot": {"samples": 1, "intent_bias": 2.0, "situation_score": 0.7, "combined_score": 5.0},
                    },
                },
                "candidate_policy_diagnostics": {
                    "artifact_dir": "/tmp/candidate",
                    "manifest_path": "/tmp/candidate/model_manifest.json",
                    "manifest_loaded": True,
                    "intent_model_loaded": True,
                    "situation_model_loaded": True,
                    "tactical_model_loaded": False,
                    "learned_policy_enabled": True,
                    "profile": "challenger",
                    "fallback_reason": [],
                },
            }
        ]
        summary = summarize_matches(matches)

        self.assertIn("policy", summary)
        policy = summary["policy"]
        self.assertIn("learned_path_coverage", policy)
        self.assertIn("move_path_coverage", policy)
        self.assertIn("shoot_path_coverage", policy)
        self.assertGreater(policy["learned_path_coverage"], 0.0)
        self.assertGreater(policy["move_path_coverage"], 0.0)
        self.assertGreater(policy["shoot_path_coverage"], 0.0)
        self.assertIn("manifest_loaded_match_count", policy)
        self.assertIn("intent_model_loaded_match_count", policy)
        self.assertIn("situation_model_loaded_match_count", policy)
        self.assertIn("tactical_model_loaded_match_count", policy)
        self.assertIn("learned_policy_enabled_match_count", policy)
        self.assertIn("profiles", policy)
        self.assertIn("artifact_dirs", policy)

    def test_build_match_policy_diagnostics_reports_fallback_when_models_missing(self) -> None:
        _install_test_stubs()
        from training.selfplay import _build_match_policy_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / "model_manifest.json").write_text(
                json.dumps(
                    {
                        "generated_at": "20260427_000000",
                        "source_commit_sha": "abc123",
                        "snapshot_role": "accepted_learned_mainline",
                        "managed_files": ["model_manifest.json"],
                    }
                ),
                encoding="utf-8",
            )

            diagnostics_with_fallback = _build_match_policy_diagnostics(
                artifacts=str(artifacts),
                participant={"user_name": "TestAI", "source_commit_sha": "abc123", "code_origin": "git:abc123"},
                profile="challenger",
                stats={
                    "intent_model_loaded_count": 0,
                    "situation_model_loaded_count": 0,
                    "tactical_model_loaded_count": 0,
                    "learned_policy_enabled_count": 0,
                },
            )

        self.assertIn("policy_models_unavailable", diagnostics_with_fallback["fallback_reason"])
        self.assertFalse(diagnostics_with_fallback["intent_model_loaded"])
        self.assertFalse(diagnostics_with_fallback["situation_model_loaded"])
        self.assertFalse(diagnostics_with_fallback["learned_policy_enabled"])

    def test_build_match_policy_diagnostics_reports_fallback_for_missing_manifest(self) -> None:
        _install_test_stubs()
        from training.selfplay import _build_match_policy_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)

            diagnostics = _build_match_policy_diagnostics(
                artifacts=str(artifacts),
                participant={"user_name": "TestAI", "source_commit_sha": "abc123", "code_origin": "git:abc123"},
                profile="challenger",
                stats={
                    "intent_model_loaded_count": 0,
                    "situation_model_loaded_count": 0,
                },
            )

        self.assertIn("manifest_missing", diagnostics["fallback_reason"])
        self.assertFalse(diagnostics["manifest_loaded"])
        self.assertFalse(diagnostics["learned_policy_enabled"])

    def test_build_match_policy_diagnostics_reports_baseline_profile_fallback(self) -> None:
        _install_test_stubs()
        from training.selfplay import _build_match_policy_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / "model_manifest.json").write_text(
                json.dumps(
                    {
                        "snapshot_role": "bootstrap_rule_baseline",
                    }
                ),
                encoding="utf-8",
            )

            diagnostics = _build_match_policy_diagnostics(
                artifacts=str(artifacts),
                participant={"user_name": "IncumbentAI", "source_commit_sha": "incsha", "code_origin": "git:incsha"},
                profile="baseline",
                stats={
                    "intent_model_loaded_count": 0,
                    "situation_model_loaded_count": 0,
                },
            )

        self.assertIn("baseline_profile", diagnostics["fallback_reason"])
        self.assertFalse(diagnostics["learned_policy_requested"])
        self.assertFalse(diagnostics["learned_policy_enabled"])

    def test_selfplay_main_writes_complete_promotion_artifact_on_failed_compare(self) -> None:
        _install_test_stubs()
        from training import selfplay

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_root = root / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
            scenario_root.mkdir(parents=True)
            (scenario_root / "scenario_7.json").write_text("{}", encoding="utf-8")
            (scenario_root / "scenario_8.json").write_text("{}", encoding="utf-8")

            candidate_dir = root / "candidate_dir"
            incumbent_dir = root / "incumbent_dir"
            fixed_tag = "20260428_000000"
            run_dir = root / "experiments" / f"league_{fixed_tag}"
            config = {
                "seed": 42,
                "output_root": "experiments",
                "league_dir": "league",
                "paired_matches": 1,
                "max_steps": 10,
                "promotion_gate": {
                    "overall_point_estimate_min": 0.5,
                    "overall_ci_lower_min": 0.4,
                    "complex_point_estimate_min": 0.5,
                    "complex_ci_lower_min": 0.4,
                    "consecutive_windows": 2,
                },
            }

            def fake_build_participant(artifacts, profile, fallback_commit_sha, user_name):
                source_commit_sha = "candsha" if user_name == "CurrentAI" else "incsha"
                return {
                    "artifacts": artifacts,
                    "profile": profile,
                    "user_name": user_name,
                    "source_commit_sha": source_commit_sha,
                    "code_origin": f"git:{source_commit_sha}",
                    "agent_class": object,
                }

            with (
                mock.patch.object(selfplay, "repo_root", return_value=root),
                mock.patch.object(selfplay, "load_config", return_value=config),
                mock.patch.object(selfplay, "_configure_torch_cpu_inference_threads", return_value=None),
                mock.patch.object(selfplay, "set_seed", return_value=None),
                mock.patch.object(selfplay, "git_head_sha", return_value="headsha"),
                mock.patch.object(selfplay, "_resolve_incumbent_profile", return_value="baseline"),
                mock.patch.object(selfplay, "_build_participant_spec", side_effect=fake_build_participant),
                mock.patch.object(selfplay, "_train_env_runtime", return_value=(object, "train_env.TrainEnv")),
                mock.patch.object(selfplay, "build_map_catalog", return_value=[]),
                mock.patch.object(selfplay, "map_bucket_lookup", return_value={7: "complex", 8: "complex"}),
                mock.patch.object(
                    selfplay,
                    "_collect_scenario_candidates",
                    return_value=[
                        {"scenario": "scenario_7.json", "map_id": 7, "bucket": "complex"},
                        {"scenario": "scenario_8.json", "map_id": 8, "bucket": "complex"},
                    ],
                ),
                mock.patch.object(
                    selfplay,
                    "_build_paired_map_list",
                    return_value={
                        "items": [
                            {"scenario": "scenario_7.json", "map_id": 7, "bucket": "complex", "order_index": 0},
                            {"scenario": "scenario_8.json", "map_id": 8, "bucket": "complex", "order_index": 1},
                        ],
                        "duplicates_used": False,
                    },
                ),
                mock.patch.object(
                    selfplay,
                    "_bootstrap_league",
                    return_value={
                        "champion": str(incumbent_dir),
                        "historical_mains": [],
                        "exploiters": [],
                        "specialists": [],
                        "consecutive_windows": 0,
                    },
                ),
                mock.patch.object(selfplay, "_run_single", side_effect=RuntimeError("explode")),
                mock.patch.object(selfplay, "now_tag", return_value=fixed_tag),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "training.selfplay",
                        "--config",
                        "ignored.json",
                        "--candidate",
                        str(candidate_dir),
                        "--incumbent",
                        str(incumbent_dir),
                    ],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "explode"):
                    selfplay.main()

            promotion = json.loads((run_dir / "promotion.json").read_text(encoding="utf-8"))

        self.assertEqual(promotion["status"], "failed")
        self.assertFalse(promotion["accepted"])
        self.assertIn("compare_failed", promotion["failures"])


if __name__ == "__main__":
    unittest.main()
