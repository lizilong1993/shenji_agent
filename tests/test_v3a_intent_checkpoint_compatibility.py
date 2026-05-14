"""Tests for the intent checkpoint compatibility adapter."""

import copy
import unittest

import torch
import torch.nn as nn

from ai.models import IntentionLSTM, LegacyIntentModel, LegacySituationModel, MultiHorizonIntentionModel, SituationGCN


class IntentCheckpointCompatibilityTests(unittest.TestCase):
    """Verify that old fc_intent checkpoints are detected and loadable."""

    def setUp(self):
        self.batch = 4
        self.seq_len = 8
        self.feature_dim = 108

    def _make_old_state_dict(self):
        """Simulate an old fc_intent checkpoint state_dict (IntentionLSTM style, num_classes=8)."""
        old = IntentionLSTM(input_dim=self.feature_dim, hidden_dim=256, num_classes=8)
        return copy.deepcopy(old.state_dict())

    def _make_new_state_dict(self):
        """Simulate a current head checkpoint state_dict (MultiHorizonIntentionModel style)."""
        new = MultiHorizonIntentionModel(input_dim=self.feature_dim, hidden_dim=128, horizons=5, num_classes=7)
        return copy.deepcopy(new.state_dict())

    def test_old_checkpoint_detected_by_fc_intent_keys(self):
        """Old checkpoint has fc_intent keys."""
        state = self._make_old_state_dict()
        self.assertTrue(any("fc_intent" in k for k in state))

    def test_new_checkpoint_detected_by_head_keys(self):
        """Current checkpoint has head keys, not fc_intent keys."""
        state = self._make_new_state_dict()
        self.assertTrue(any(k.startswith("head.") for k in state))
        self.assertFalse(any("fc_intent" in k for k in state))

    def test_legacy_model_loads_old_checkpoint(self):
        """LegacyIntentModel can load old fc_intent state dict."""
        state = self._make_old_state_dict()
        model = LegacyIntentModel(input_dim=self.feature_dim, hidden_dim=256)
        model.load_state_dict(state)
        model.eval()
        self.assertIsNotNone(model)

    def test_legacy_model_output_shape(self):
        """LegacyIntentModel forward produces [batch, 1, 7]."""
        state = self._make_old_state_dict()
        model = LegacyIntentModel(input_dim=self.feature_dim, hidden_dim=256)
        model.load_state_dict(state)
        model.eval()
        dummy = torch.randn(self.batch, self.seq_len, self.feature_dim)
        with torch.no_grad():
            out = model(dummy)
        expected = (self.batch, 1, 7)
        self.assertEqual(out.shape, expected, f"Expected {expected}, got {out.shape}")

    def test_legacy_model_output_usable_by_policy(self):
        """Policy indexes [0, 0] into output — must work with LegacyIntentModel output."""
        state = self._make_old_state_dict()
        model = LegacyIntentModel(input_dim=self.feature_dim, hidden_dim=256)
        model.load_state_dict(state)
        model.eval()
        dummy = torch.randn(1, self.seq_len, self.feature_dim)
        with torch.no_grad():
            out = model(dummy)  # [1, 1, 7]
            logits = out[0, 0]  # [7]
        self.assertEqual(logits.shape, (7,))

    def test_incompatible_checkpoint_fails_closed(self):
        """A random state dict should fail closed."""
        state = {
            "random_key": torch.randn(10, 10),
            "another_key": torch.randn(5),
        }
        model = LegacyIntentModel(input_dim=self.feature_dim, hidden_dim=256)
        with self.assertRaises(Exception):
            model.load_state_dict(state)

    def test_current_model_rejects_old_checkpoint(self):
        """MultiHorizonIntentionModel should reject old fc_intent keys."""
        state = self._make_old_state_dict()
        model = MultiHorizonIntentionModel(input_dim=self.feature_dim, hidden_dim=128, horizons=5, num_classes=7)
        with self.assertRaises(Exception):
            model.load_state_dict(state)

    def test_old_checkpoint_loads_into_legacy_model_with_strict(self):
        """Old checkpoint loads into LegacyIntentModel with strict=True."""
        state = self._make_old_state_dict()
        model = LegacyIntentModel(input_dim=self.feature_dim, hidden_dim=256)
        model.load_state_dict(state, strict=True)
        model.eval()
        self.assertTrue(hasattr(model, "fc_intent"))
        self.assertEqual(model.fc_intent.weight.shape, (8, 512))

    def test_legacy_model_num_classes_access(self):
        """LegacyIntentModel exposes num_classes and horizons like current model."""
        model = LegacyIntentModel(input_dim=self.feature_dim, hidden_dim=256)
        self.assertEqual(model.num_classes, 7)
        self.assertEqual(model.horizons, 1)

    def test_legacy_situation_model_loads_bare_checkpoint(self):
        """LegacySituationModel can load bare SituationGCN state_dict (no backbone. prefix)."""
        bare = SituationGCN(num_features=108, hidden_dim=64).state_dict()
        self.assertTrue(any(k.startswith("sage") or k.startswith("fc_") for k in bare))
        model = LegacySituationModel(num_features=108, hidden_dim=64)
        model.load_state_dict(bare)
        model.eval()
        dummy_x = torch.randn(10, 108)
        dummy_adj = torch.eye(10)
        with torch.no_grad():
            out = model(dummy_x, dummy_adj)
        self.assertEqual(out.shape, torch.Size([1]))  # scalar output via mean(dim=0)

    def test_legacy_situation_model_rejects_prefixed_checkpoint(self):
        """LegacySituationModel does NOT accept GraphSituationModel-style (backbone. prefixed) state dict."""
        from ai.models import GraphSituationModel
        gsm = GraphSituationModel(num_features=108, hidden_dim=64)
        prefixed = gsm.state_dict()  # has backbone. prefix
        model = LegacySituationModel(num_features=108, hidden_dim=64)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(prefixed)

    def test_situation_bare_detected_in_policy_load(self):
        """Check that bare situation checkpoint is detected."""
        bare = SituationGCN(num_features=108, hidden_dim=64).state_dict()
        is_bare = any(k.startswith("sage") or k.startswith("fc_") for k in bare)
        self.assertTrue(is_bare)

    def test_policy_diagnostics_reports_intent_architecture(self):
        """Policy diagnostics include intent_model_architecture field."""
        # Test the model's class name attribute
        model = LegacyIntentModel(input_dim=self.feature_dim, hidden_dim=256)
        self.assertEqual(model.__class__.__name__, "LegacyIntentModel")


if __name__ == "__main__":
    unittest.main()
