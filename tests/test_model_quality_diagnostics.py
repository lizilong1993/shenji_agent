"""Tests for scripts/diagnose_model_quality.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _import_script():
    from importlib import import_module
    import scripts.diagnose_model_quality as dq
    return dq


class TestDiagnoseJsonSchema:
    """Verify the output JSON has stable, expected schema."""

    def test_empty_artifacts_produces_stable_schema(self, tmp_path):
        dq = _import_script()
        artifacts = tmp_path / "weights"
        artifacts.mkdir()
        result = dq.diagnose(artifacts, datasets_dir=None)

        required_keys = [
            "git_head", "manifest_path", "manifest_loaded",
            "intent_model_path", "situation_model_path", "tactical_model_path",
            "situation_checkpoint_loaded", "intent_checkpoint_loaded",
            "tactical_checkpoint_loaded", "normalizer_available", "normalizer_loaded",
            "warnings", "recommended_next_actions",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        assert isinstance(result["warnings"], list)
        assert isinstance(result["recommended_next_actions"], list)

    def test_missing_models_produces_warnings(self, tmp_path):
        dq = _import_script()
        artifacts = tmp_path / "weights"
        artifacts.mkdir()
        result = dq.diagnose(artifacts, datasets_dir=None)

        missing_warnings = [
            w for w in result["warnings"]
            if "model_missing" in w.lower() or "checkpoint_loaded" in str(w).lower()
            or "tactical" in str(w).lower()
        ]
        # At minimum tactical_model_missing should trigger
        assert any("tactical" in w.lower() for w in result["warnings"])

    def test_dead_situation_model_triggers_warning(self, tmp_path, monkeypatch):
        dq = _import_script()
        artifacts = tmp_path / "weights"
        artifacts.mkdir()

        import torch
        import torch.nn as nn
        sys.path.insert(0, str(REPO_ROOT))
        from ai.models import SituationGCN

        class DeadBackbone(SituationGCN):
            def forward(self, x, adj):
                return torch.zeros(x.shape[0], 1)

        class DeadWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = DeadBackbone(num_features=108, hidden_dim=64, output_dim=1)

            def forward(self, x, adj):
                return self.backbone(x, adj).mean(dim=0)

        model = DeadWrapper()
        torch.save({"model_state": model.state_dict(), "input_dim": 108, "hidden_dim": 64},
                    artifacts / "situation_gcn.pt")

        result = dq.diagnose(artifacts, datasets_dir=None)
        assert result["situation_per_node_variance_mean"] < 1e-4

    def test_missing_normalizer_triggers_warning(self, tmp_path):
        dq = _import_script()
        artifacts = tmp_path / "weights"
        artifacts.mkdir()
        result = dq.diagnose(artifacts, datasets_dir=None)

        assert result["normalizer_loaded"] is False
        assert any("normalizer" in w.lower() for w in result["warnings"])

    def test_missing_tactical_trigger_warning(self, tmp_path):
        dq = _import_script()
        artifacts = tmp_path / "weights"
        artifacts.mkdir()
        result = dq.diagnose(artifacts, datasets_dir=None)

        assert result["tactical_available"] is False
        assert any("tactical" in w.lower() for w in result["warnings"])

    def test_normalizer_loaded_detected(self, tmp_path):
        dq = _import_script()
        artifacts = tmp_path / "weights"
        artifacts.mkdir()
        normalizer = {
            "feature_mean": [0.0] * 108,
            "feature_std": [1.0] * 108,
            "feature_dim": 108,
        }
        (artifacts / "situation_feature_normalizer.json").write_text(json.dumps(normalizer))
        result = dq.diagnose(artifacts, datasets_dir=None)

        assert result["normalizer_available"] is True
        assert result["normalizer_loaded"] is True

    def test_output_json_roundtrips(self, tmp_path):
        dq = _import_script()
        artifacts = tmp_path / "weights"
        artifacts.mkdir()

        class MockModel:
            pass  # no real model, just empty

        result = dq.diagnose(artifacts, datasets_dir=None)
        output_path = tmp_path / "output.json"
        output_path.write_text(json.dumps(result, indent=2))

        roundtripped = json.loads(output_path.read_text())
        assert roundtripped["git_head"] == result["git_head"]
        assert roundtripped["warnings"] == result["warnings"]


class TestDiagnoseWithMockModel:
    """Verify warnings trigger correctly with mock models that produce bad outputs."""

    def test_negative_move_bias_warning(self, tmp_path):
        dq = _import_script()
        artifacts = tmp_path / "weights"
        artifacts.mkdir()

        import torch
        import torch.nn as nn
        sys.path.insert(0, str(REPO_ROOT))
        from ai.models import MultiHorizonIntentionModel

        model = MultiHorizonIntentionModel(input_dim=108, hidden_dim=128, horizons=5, num_classes=7)
        # Set final head layer to produce class_0 dominance for all horizons
        # head is Sequential: LayerNorm, Linear, ReLU, Linear
        # The last Linear maps hidden_dim*2 -> horizons*num_classes = 5*7 = 35
        with torch.no_grad():
            last_linear = model.head[-1]  # nn.Linear(256, 35)
            last_linear.weight.zero_()
            last_linear.bias.zero_()
            # Set bias for class_0 in each horizon to a high value
            for h in range(5):
                last_linear.bias[h * 7 + 0] = 10.0  # class 0 = stay

        torch.save({
            "model_state": model.state_dict(),
            "feature_dim": 108, "hidden_dim": 128,
            "horizons": 5, "num_classes": 7,
        }, artifacts / "intent_lstm.pt")

        result = dq.diagnose(artifacts, datasets_dir=None)
        assert result.get("move_bias_estimate", 0) < -0.1
        assert any("negative_move_bias" in w for w in result["warnings"])
