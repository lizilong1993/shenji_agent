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


class SelfplayParticipantWiringTests(unittest.TestCase):
    def test_candidate_and_incumbent_point_to_different_artifact_dirs(self) -> None:
        _install_test_stubs()
        from training import selfplay

        with mock.patch.object(
            selfplay,
            "load_config",
            return_value={
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
            },
        ), mock.patch.object(
            sys,
            "argv",
            [
                "training.selfplay",
                "--config",
                "ignored.json",
                "--candidate",
                "ai/weights",
                "--incumbent",
                "ai/weights",
            ],
        ):
            with self.assertRaises(SystemExit) as ctx:
                selfplay.main()
        self.assertIn(
            "candidate and incumbent must point to different artifact directories",
            str(ctx.exception),
        )

    def test_incumbent_with_bootstrap_rule_baseline_snapshot_role_resolves_to_baseline(self) -> None:
        _install_test_stubs()
        from training.selfplay import _resolve_incumbent_profile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "baseline_artifacts"
            artifacts.mkdir()
            (artifacts / "model_manifest.json").write_text(
                json.dumps({"snapshot_role": "bootstrap_rule_baseline"}),
                encoding="utf-8",
            )

            profile = _resolve_incumbent_profile(str(artifacts), None, "baseline")
            self.assertEqual(profile, "baseline")

    def test_incumbent_with_no_snapshot_role_falls_back_to_default(self) -> None:
        _install_test_stubs()
        from training.selfplay import _resolve_incumbent_profile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "plain_artifacts"
            artifacts.mkdir()
            (artifacts / "model_manifest.json").write_text(
                json.dumps({"history_len": 8}),
                encoding="utf-8",
            )

            profile = _resolve_incumbent_profile(str(artifacts), None, "baseline")
            self.assertEqual(profile, "baseline")

    def test_incumbent_with_learned_snapshot_resolves_as_incumbent(self) -> None:
        _install_test_stubs()
        from training.selfplay import _resolve_incumbent_profile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "learned_artifacts"
            artifacts.mkdir()
            (artifacts / "model_manifest.json").write_text(
                json.dumps({"snapshot_role": "accepted_learned_mainline"}),
                encoding="utf-8",
            )

            profile = _resolve_incumbent_profile(str(artifacts), None, "baseline")
            self.assertEqual(profile, "incumbent")

    def test_explicit_incumbent_profile_overrides_snapshot_role(self) -> None:
        _install_test_stubs()
        from training.selfplay import _resolve_incumbent_profile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "model_manifest.json").write_text(
                json.dumps({"snapshot_role": "bootstrap_rule_baseline"}),
                encoding="utf-8",
            )

            profile = _resolve_incumbent_profile(str(artifacts), "custom_profile", "baseline")
            self.assertEqual(profile, "custom_profile")

    def test_participant_payload_includes_manifest_and_snapshot_role(self) -> None:
        _install_test_stubs()
        from training.selfplay import _participant_payload

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "model_manifest.json").write_text(
                json.dumps(
                    {
                        "snapshot_role": "accepted_learned_mainline",
                        "generated_at": "20260427_000000",
                        "source_commit_sha": "abc123",
                        "managed_files": ["model_manifest.json"],
                    }
                ),
                encoding="utf-8",
            )

            participant = _participant_payload(
                str(artifacts),
                "challenger",
                "abc123",
                "git:abc123",
            )

        self.assertEqual(participant["profile"], "challenger")
        self.assertEqual(participant["snapshot_role"], "accepted_learned_mainline")
        self.assertEqual(participant["source_commit_sha"], "abc123")
        self.assertTrue(participant["manifest_loaded"])
        self.assertTrue(str(participant["manifest_path"]).endswith("model_manifest.json"))
        self.assertEqual(participant["artifacts"], str(artifacts.resolve()))

    def test_participant_json_sidecar_writes_candidate_and_incumbent(self) -> None:
        _install_test_stubs()
        from training.selfplay import _write_participants_sidecar

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            candidate = {"source_commit_sha": "candsha", "code_origin": "git:candsha", "user_name": "CurrentAI"}
            incumbent = {"source_commit_sha": "incsha", "code_origin": "git:incsha", "user_name": "IncumbentAI"}

            _write_participants_sidecar(
                run_dir,
                env_backend="train_env.TrainEnv",
                candidate_artifacts="/tmp/candidate",
                incumbent_artifacts="/tmp/incumbent",
                candidate_profile="challenger",
                incumbent_profile="baseline",
                candidate=candidate,
                incumbent=incumbent,
            )

            participants = json.loads((run_dir / "participants.json").read_text(encoding="utf-8"))

        self.assertEqual(participants["env_backend"], "train_env.TrainEnv")
        self.assertEqual(participants["candidate"]["profile"], "challenger")
        self.assertEqual(participants["candidate"]["artifacts"], "/tmp/candidate")
        self.assertEqual(participants["candidate"]["code_origin"], "git:candsha")
        self.assertEqual(participants["candidate"]["user_name"], "CurrentAI")
        self.assertEqual(participants["incumbent"]["profile"], "baseline")
        self.assertEqual(participants["incumbent"]["artifacts"], "/tmp/incumbent")
        self.assertEqual(participants["incumbent"]["code_origin"], "git:incsha")
        self.assertEqual(participants["incumbent"]["user_name"], "IncumbentAI")

    def test_candidate_artifacts_differs_from_incumbent_artifacts_in_sidecar(self) -> None:
        _install_test_stubs()
        from training.selfplay import _write_participants_sidecar

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            candidate = {"source_commit_sha": "candsha", "code_origin": "git:candsha"}
            incumbent = {"source_commit_sha": "incsha", "code_origin": "git:incsha"}

            _write_participants_sidecar(
                run_dir,
                env_backend="train_env.TrainEnv",
                candidate_artifacts="/data/candidate_weights",
                incumbent_artifacts="/data/incumbent_baseline",
                candidate_profile="challenger",
                incumbent_profile="baseline",
                candidate=candidate,
                incumbent=incumbent,
            )

            participants = json.loads((run_dir / "participants.json").read_text(encoding="utf-8"))

        self.assertNotEqual(
            participants["candidate"]["artifacts"],
            participants["incumbent"]["artifacts"],
        )

    def test_selfplay_main_resolves_incumbent_as_baseline_for_rule_artifacts(self) -> None:
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

            called_profiles: list[str] = []

            def fake_resolve_incumbent_profile(artifacts, requested, fallback):
                called_profiles.append(str(requested))
                return str(requested) if requested else str(fallback)

            participant_calls: list[str] = []

            def fake_build_participant(artifacts, profile, fallback_commit_sha, user_name):
                participant_calls.append(profile)
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
                    return_value={
                        "champion": "ai/weights",
                        "historical_mains": [],
                        "exploiters": [],
                        "specialists": [],
                        "consecutive_windows": 0,
                    },
                ),
                mock.patch.object(selfplay, "_resolve_incumbent_profile", side_effect=fake_resolve_incumbent_profile),
                mock.patch.object(selfplay, "_build_participant_spec", side_effect=fake_build_participant),
                mock.patch.object(selfplay, "_train_env_runtime", side_effect=RuntimeError("stop_early")),
                mock.patch.object(sys, "argv", ["training.selfplay", "--config", "ignored.yaml"]),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop_early"):
                    selfplay.main()

        self.assertIn("challenger", participant_calls)
        self.assertIn("baseline", participant_calls)


if __name__ == "__main__":
    unittest.main()
