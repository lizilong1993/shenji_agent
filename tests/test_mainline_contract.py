from __future__ import annotations

import json
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
                        "map_sampling": {"complex": 1.0},
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
