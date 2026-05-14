"""Tests for tactical on/off switch.

Verifies that tactical_mode flag correctly controls tactical score usage
while preserving all non-tactical behaviour.
"""

from __future__ import annotations

import pytest

from ai.policy import LearnedPolicy, TORCH_AVAILABLE


class TestTacticalDefaultMode:
    """Default tactical_mode should be 'on'."""

    def test_default_tactical_mode_is_on(self):
        p = LearnedPolicy()
        assert p.tactical_mode == "on"

    def test_diagnostics_includes_tactical_mode(self):
        p = LearnedPolicy()
        diag = p.get_diagnostics()
        assert "tactical_mode" in diag
        assert diag.get("tactical_mode") == "on"


class TestTacticalOn:
    """tactical-mode=on: tactical model is used."""

    def test_tactical_on_config(self):
        p = LearnedPolicy()
        p.tactical_mode = "on"
        assert p.tactical_mode == "on"

    def test_diagnostics_records_on_mode(self):
        p = LearnedPolicy()
        p.tactical_mode = "on"
        diag = p.get_diagnostics()
        assert diag.get("tactical_mode") == "on"


class TestTacticalOff:
    """tactical-mode=off: tactical score always 0, model still loaded."""

    def test_tactical_off_config(self):
        p = LearnedPolicy()
        p.tactical_mode = "off"
        assert p.tactical_mode == "off"

    def test_diagnostics_records_off_mode(self):
        p = LearnedPolicy()
        p.tactical_mode = "off"
        diag = p.get_diagnostics()
        assert diag.get("tactical_mode") == "off"

    def test_tactical_off_returns_zero_score(self):
        p = LearnedPolicy()
        p.tactical_mode = "off"
        p.tactical_model = object()  # model loaded but disabled
        score = p._tactical_score({"obj_id": 0, "cur_hex": 101}, {"target_obj_id": 1})
        assert score == 0.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestTacticalMissingFailClosed:
    """tactical-mode=missing_fail_closed raises when model absent."""

    def test_raises_when_model_missing(self):
        p = LearnedPolicy()
        p.tactical_mode = "missing_fail_closed"
        p.tactical_model = None
        with pytest.raises(RuntimeError, match="missing_fail_closed"):
            p._tactical_score({"obj_id": 0, "cur_hex": 101}, {"target_obj_id": 1})

    def test_on_mode_graceful_when_model_missing(self):
        p = LearnedPolicy()
        p.tactical_mode = "on"
        p.tactical_model = None
        score = p._tactical_score({"obj_id": 0, "cur_hex": 101}, {"target_obj_id": 1})
        assert score == 0.0


class TestTacticalModeViaConfigure:
    """tactical_mode is set correctly when passed to configure()."""

    def test_configure_baseline_default_mode(self):
        p = LearnedPolicy()
        p.configure(faction=1, policy_profile="baseline")
        assert p.tactical_mode == "on"

    def test_configure_baseline_explicit_on(self):
        p = LearnedPolicy()
        p.configure(faction=1, policy_profile="baseline", tactical_mode="on")
        assert p.tactical_mode == "on"

    def test_configure_baseline_explicit_off(self):
        p = LearnedPolicy()
        p.configure(faction=1, policy_profile="baseline", tactical_mode="off")
        assert p.tactical_mode == "off"

    def test_configure_baseline_explicit_missing_fail_closed(self):
        p = LearnedPolicy()
        p.configure(faction=1, policy_profile="baseline", tactical_mode="missing_fail_closed")
        assert p.tactical_mode == "missing_fail_closed"

    def test_configure_non_baseline_sets_mode_before_loading(self):
        """tactical_mode is set before model loading, so even if models fail
        to load the mode flag is preserved.
        This test is xfail due to a pre-existing model-architecture mismatch
        between saved weights and current model definitions."""
        p = LearnedPolicy()
        try:
            p.configure(faction=1, policy_profile="challenger", tactical_mode="off")
        except Exception:
            pass
        assert p.tactical_mode == "off"


class TestConfigureWithoutTacticalMode:
    """configure() calls without tactical_mode must not crash
    and must default to 'on'."""

    def test_configure_defaults_on(self):
        p = LearnedPolicy()
        p.configure(faction=1, policy_profile="baseline")
        assert p.tactical_mode == "on"
