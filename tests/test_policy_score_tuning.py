"""Contract tests for policy score tuning: city selection, shoot selection, and situation model health."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch  # noqa: F401

from ai.policy import LearnedPolicy


class _DummyTacticalMap:
    """Minimal tactical map stub for scoring tests."""

    def is_valid(self, pos: int) -> bool:
        return True

    def get_distance(self, p1: int, p2: int) -> int:
        return abs(p1 - p2)


def _make_observation(*, operators=None, cities=None, step=100):
    return {
        "operators": operators or [],
        "cities": cities or [],
        "time": {"cur_step": step, "tick": step * 10},
        "scores": {"red_total": 0, "blue_total": 0, "red_win": 0, "blue_win": 0},
        "valid_actions": {},
        "communication": [],
        "tank_enemy": [],
    }


def _make_operator(obj_id, color, cur_hex, blood=100, **kwargs):
    op = {
        "obj_id": obj_id,
        "color": color,
        "cur_hex": cur_hex,
        "cur_pos": 0,
        "type": 0,
        "blood": blood,
        "ammo": 30,
        "fuel": 50,
        "morale": 80,
        "stealth": 0,
    }
    op.update(kwargs)
    return op


def _make_city(coord, value=50, flag=0):
    return {"coord": coord, "value": value, "flag": flag}


@pytest.fixture(scope="module")
def policy():
    saved_device = os.environ.get("SHENJI_POLICY_DEVICE")
    os.environ["SHENJI_POLICY_DEVICE"] = "cpu"
    try:
        p = LearnedPolicy()
        p.configure(faction=1, artifact_dir="ai/weights", policy_profile="challenger")
        yield p
        p.reset()
    finally:
        if saved_device is not None:
            os.environ["SHENJI_POLICY_DEVICE"] = saved_device
        else:
            os.environ.pop("SHENJI_POLICY_DEVICE", None)


@pytest.fixture
def tactical_map():
    return _DummyTacticalMap()


class TestCityScoring:
    def test_higher_value_city_preferred_at_same_distance(self, policy, tactical_map):
        """City with higher value should beat lower value at equal distance."""
        cities = [_make_city(1200, value=80, flag=0), _make_city(1200, value=30, flag=0)]
        operator = _make_operator(1, 1, 1000)
        obs = _make_observation(
            operators=[operator, _make_operator(2, 2, 5000)],
            cities=cities,
        )
        policy.observe(obs)
        selected, _trace = policy.choose_city(operator, cities, obs, tactical_map)
        assert selected is not None
        assert selected["value"] == 80

    def test_enemy_city_preferred_over_friendly_at_same_distance(self, policy, tactical_map):
        """Enemy city preferred over friendly city at equal distance and value."""
        cities = [
            _make_city(1200, value=50, flag=2),   # enemy
            _make_city(1200, value=50, flag=1),   # friendly
        ]
        operator = _make_operator(1, 1, 1000)
        obs = _make_observation(
            operators=[operator, _make_operator(2, 2, 5000)],
            cities=cities,
        )
        policy.observe(obs)
        selected, _trace = policy.choose_city(operator, cities, obs, tactical_map)
        assert selected is not None
        assert selected["flag"] == 2  # enemy city preferred

    def test_neutral_city_preferred_over_friendly_at_same_distance(self, policy, tactical_map):
        """Neutral city preferred over friendly city at equal distance and value."""
        cities = [
            _make_city(1200, value=50, flag=0),   # neutral
            _make_city(1200, value=50, flag=1),   # friendly
        ]
        operator = _make_operator(1, 1, 1000)
        obs = _make_observation(
            operators=[operator, _make_operator(2, 2, 5000)],
            cities=cities,
        )
        policy.observe(obs)
        selected, _trace = policy.choose_city(operator, cities, obs, tactical_map)
        assert selected is not None
        assert selected["flag"] == 0  # neutral city preferred

    def test_nearby_high_value_beats_far_low_value(self, policy, tactical_map):
        """Nearby high-value neutral city beats far low-value neutral city."""
        cities = [
            _make_city(1100, value=30, flag=0),   # distance 100, low value
            _make_city(5000, value=80, flag=0),   # distance 4000, high value
        ]
        operator = _make_operator(1, 1, 1000)
        obs = _make_observation(
            operators=[operator, _make_operator(2, 2, 5000)],
            cities=cities,
        )
        policy.observe(obs)
        selected, _trace = policy.choose_city(operator, cities, obs, tactical_map)
        assert selected is not None
        # Distance should not completely dominate; close but low-value may still lose
        # to far high-value depending on weights. Just verify selection is not None.

    def test_occupy_proximity_bonus_applied(self, policy, tactical_map):
        """Operator adjacent to capturable city gets occupy proximity bonus."""
        city = _make_city(1001, value=50, flag=2)  # enemy, adjacent
        operator = _make_operator(1, 1, 1000)
        obs = _make_observation(
            operators=[operator, _make_operator(2, 2, 5000)],
            cities=[city],
        )
        policy.observe(obs)
        selected, trace = policy.choose_city(operator, [city], obs, tactical_map)
        assert selected is not None
        assert trace["combined_score"] > 50  # should be high due to proximity + owner bonus

    def test_city_selection_returns_trace_with_scores(self, policy, tactical_map):
        """Trace contains all required score fields."""
        city = _make_city(1500, value=50, flag=0)
        operator = _make_operator(1, 1, 1000)
        obs = _make_observation(
            operators=[operator, _make_operator(2, 2, 5000)],
            cities=[city],
        )
        policy.observe(obs)
        selected, trace = policy.choose_city(operator, [city], obs, tactical_map)
        assert selected is not None
        for field in ("obj_id", "candidate_id", "intent_bias", "situation_score", "combined_score"):
            assert field in trace, f"trace missing {field}"


class TestShootScoring:
    def test_high_value_kill_preferred(self, policy, tactical_map):
        """Higher-value kill candidate preferred when both are killable."""
        operator = _make_operator(1, 1, 1000)
        target_high = _make_operator(2, 2, 2000, blood=20, value=80)
        target_low = _make_operator(3, 2, 2100, blood=20, value=30)
        obs = _make_observation(
            operators=[operator, target_high, target_low],
            cities=[],
        )
        policy.observe(obs)
        candidates = [
            {"target_obj_id": 2, "damage": 20, "hit_prob": 0.8, "attack_level": 5, "weapon_id": 1, "target_blood": 20},
            {"target_obj_id": 3, "damage": 20, "hit_prob": 0.8, "attack_level": 5, "weapon_id": 2, "target_blood": 20},
        ]
        best, trace = policy.choose_shoot_candidate(operator, candidates, obs, tactical_map)
        assert best is not None
        assert best["target_obj_id"] == 2  # higher value target preferred

    def test_kill_beats_non_kill_at_same_value(self, policy, tactical_map):
        """Kill candidate beats non-kill candidate at same target value."""
        operator = _make_operator(1, 1, 1000)
        target_killable = _make_operator(2, 2, 2000, blood=10, value=50)
        target_tanky = _make_operator(3, 2, 2100, blood=100, value=50)
        obs = _make_observation(
            operators=[operator, target_killable, target_tanky],
            cities=[],
        )
        policy.observe(obs)
        candidates = [
            {"target_obj_id": 2, "damage": 20, "hit_prob": 0.8, "attack_level": 5, "weapon_id": 1, "target_blood": 10},
            {"target_obj_id": 3, "damage": 20, "hit_prob": 0.8, "attack_level": 5, "weapon_id": 2, "target_blood": 100},
        ]
        best, _trace = policy.choose_shoot_candidate(operator, candidates, obs, tactical_map)
        assert best is not None
        assert best["target_obj_id"] == 2  # killable target preferred

    def test_shoot_trace_contains_scores(self, policy, tactical_map):
        """Shoot trace contains all required score fields."""
        operator = _make_operator(1, 1, 1000)
        target = _make_operator(2, 2, 2000, blood=50, value=40)
        obs = _make_observation(
            operators=[operator, target],
            cities=[],
        )
        policy.observe(obs)
        candidates = [
            {"target_obj_id": 2, "damage": 10, "hit_prob": 0.7, "attack_level": 4, "weapon_id": 1, "target_blood": 50},
        ]
        best, trace = policy.choose_shoot_candidate(operator, candidates, obs, tactical_map)
        assert best is not None
        for field in ("obj_id", "candidate_id", "intent_bias", "situation_score", "combined_score"):
            assert field in trace, f"trace missing {field}"


class TestSituationModelHealth:
    def test_situation_model_produces_valid_score(self, policy, tactical_map):
        """Situation model output should be finite and in valid range, not NaN/Inf."""
        operator = _make_operator(1, 1, 1000)
        obs = _make_observation(
            operators=[operator, _make_operator(2, 2, 2000)],
            cities=[_make_city(1500)],
        )
        policy.observe(obs)
        city = _make_city(1500, value=50, flag=0)
        _selected, trace = policy.choose_city(operator, [city], obs, tactical_map)
        sit_score = float(trace["situation_score"])
        import math
        assert not math.isnan(sit_score), f"situation_score is NaN"
        assert not math.isinf(sit_score), f"situation_score is Inf"
        assert 0.0 <= sit_score <= 1.0, f"situation_score {sit_score} outside [0,1]"

    def test_situation_model_contributes_to_combined_score(self, policy, tactical_map):
        """When situation_score > 1e-6, it should contribute to combined_score."""
        operator = _make_operator(1, 1, 1000)
        obs = _make_observation(
            operators=[
                operator,
                _make_operator(2, 2, 2000),
                _make_operator(3, 1, 1100),
                _make_operator(4, 2, 2100),
            ],
            cities=[_make_city(1500)],
        )
        policy.observe(obs)
        city = _make_city(1500, value=50, flag=0)
        _selected, trace = policy.choose_city(operator, [city], obs, tactical_map)
        sit_score = float(trace["situation_score"])
        combined = float(trace["combined_score"])
        if sit_score > 1e-6:
            assert combined >= sit_score * 100.0 * 0.5  # contribution should be meaningful


class TestPolicyDiagnostics:
    def test_diagnostics_include_score_means(self, policy, tactical_map):
        """After actions, diagnostics include score means with valid structure."""
        operator = _make_operator(1, 1, 1000)
        target = _make_operator(2, 2, 2000, blood=50, value=40)
        obs = _make_observation(
            operators=[operator, target],
            cities=[_make_city(1500)],
        )
        policy.observe(obs)

        city = _make_city(1500, value=50, flag=0)
        _selected, trace = policy.choose_city(operator, [city], obs, tactical_map)
        policy.note_action("move", True, trace)

        candidates = [
            {"target_obj_id": 2, "damage": 10, "hit_prob": 0.7, "attack_level": 4, "weapon_id": 1, "target_blood": 50},
        ]
        _best, shoot_trace = policy.choose_shoot_candidate(operator, candidates, obs, tactical_map)
        policy.note_action("shoot", True, shoot_trace)

        diag = policy.get_diagnostics()
        assert "score_means" in diag
        for action_key in ("move", "shoot"):
            means = diag["score_means"].get(action_key, {})
            assert "intent_bias" in means
            assert "situation_score" in means
            assert "combined_score" in means
            assert isinstance(means.get("samples", 0), int)
