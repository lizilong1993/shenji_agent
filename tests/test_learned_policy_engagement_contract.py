from __future__ import annotations

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


class LearnedPolicyEngagementContractTests(unittest.TestCase):
    def test_resolve_formal_compare_artifacts_prefers_cli_then_config_then_defaults(self) -> None:
        _install_test_stubs()
        from training.selfplay import _resolve_formal_compare_artifacts

        candidate, incumbent = _resolve_formal_compare_artifacts(
            {
                "candidate_artifacts": "cfg/candidate",
                "incumbent_artifacts": "cfg/incumbent",
            },
            candidate_arg=None,
            incumbent_arg=None,
        )
        self.assertEqual(candidate, "cfg/candidate")
        self.assertEqual(incumbent, "cfg/incumbent")

        candidate, incumbent = _resolve_formal_compare_artifacts(
            {},
            candidate_arg="cli/candidate",
            incumbent_arg="cli/incumbent",
        )
        self.assertEqual(candidate, "cli/candidate")
        self.assertEqual(incumbent, "cli/incumbent")

        candidate, incumbent = _resolve_formal_compare_artifacts({}, candidate_arg=None, incumbent_arg=None)
        self.assertEqual(candidate, "ai/weights")
        self.assertEqual(incumbent, "ai/weights_incumbent_rule")

    def test_selfplay_main_uses_configured_formal_artifact_paths_when_cli_omits_them(self) -> None:
        _install_test_stubs()
        from training import selfplay

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments").mkdir()
            (root / "league").mkdir()
            config = {
                "seed": 42,
                "output_root": "experiments",
                "league_dir": "league",
                "paired_matches": 1,
                "max_steps": 10,
                "candidate_artifacts": "ai/weights",
                "incumbent_artifacts": "ai/weights_incumbent_rule",
                "candidate_profile": "challenger",
                "incumbent_profile": "baseline",
                "promotion_gate": {
                    "overall_point_estimate_min": 0.5,
                    "overall_ci_lower_min": 0.4,
                    "complex_point_estimate_min": 0.5,
                    "complex_ci_lower_min": 0.4,
                    "consecutive_windows": 1,
                },
            }
            participant_calls: list[tuple[str, str, str]] = []

            def fake_build_participant(artifacts: str, profile: str, fallback_commit_sha: str, user_name: str) -> dict[str, object]:
                participant_calls.append((artifacts, profile, user_name))
                return {
                    "artifacts": artifacts,
                    "profile": profile,
                    "user_name": user_name,
                    "source_commit_sha": fallback_commit_sha,
                    "code_origin": f"git:{fallback_commit_sha}",
                    "agent_class": object,
                }

            with (
                mock.patch.object(selfplay, "repo_root", return_value=root),
                mock.patch.object(selfplay, "load_config", return_value=config),
                mock.patch.object(selfplay, "_configure_torch_cpu_inference_threads", return_value=None),
                mock.patch.object(selfplay, "set_seed", return_value=None),
                mock.patch.object(selfplay, "git_head_sha", return_value="headsha"),
                mock.patch.object(
                    selfplay,
                    "_bootstrap_league",
                    return_value={"champion": "ai/weights", "historical_mains": [], "exploiters": [], "specialists": [], "consecutive_windows": 0},
                ),
                mock.patch.object(selfplay, "_resolve_incumbent_profile", return_value="baseline") as resolve_profile,
                mock.patch.object(selfplay, "_build_participant_spec", side_effect=fake_build_participant),
                mock.patch.object(selfplay, "_train_env_runtime", side_effect=RuntimeError("stop_after_participant_resolution")),
                mock.patch.object(sys, "argv", ["training.selfplay", "--config", "ignored.yaml"]),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop_after_participant_resolution"):
                    selfplay.main()

        self.assertEqual(
            participant_calls,
            [
                ("ai/weights", "challenger", "CurrentAI"),
                ("ai/weights_incumbent_rule", "baseline", "IncumbentAI"),
            ],
        )
        resolve_profile.assert_called_once_with("ai/weights_incumbent_rule", "baseline", "baseline")

    def test_verify_contract_reports_candidate_enablement_failures(self) -> None:
        _install_test_stubs()
        from scripts.verify_learned_policy_engagement import _evaluate_contract

        payload = {
            "candidate": {
                "manifest_loaded": True,
                "participant": {
                    "user_name": "CurrentAI",
                },
                "policy_diagnostics_after_setup": {
                    "profile": "challenger",
                    "manifest_loaded": True,
                    "intent_model_loaded": True,
                    "situation_model_loaded": False,
                    "learned_policy_enabled": False,
                    "use_ai_optimization": False,
                },
            },
            "incumbent": {
                "participant": {
                    "user_name": "IncumbentAI",
                },
                "policy_diagnostics_after_setup": {
                    "profile": "baseline",
                },
            },
        }

        verdict = _evaluate_contract(payload)
        self.assertFalse(verdict["passed"])
        self.assertIn("candidate_situation_model_not_loaded", verdict["failures"])
        self.assertIn("candidate_learned_policy_not_enabled", verdict["failures"])
        self.assertIn("candidate_ai_optimization_disabled", verdict["failures"])


if __name__ == "__main__":
    unittest.main()
