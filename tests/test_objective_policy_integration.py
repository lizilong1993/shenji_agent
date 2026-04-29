from __future__ import annotations

import os

from ai.policy import LearnedPolicy


class _Map:
    def get_distance(self, left: int, right: int) -> int:
        return abs(left - right)

    def is_valid(self, pos: int) -> bool:
        return pos >= 0


def _operator(obj_id: int, color: int, cur_hex: int, blood: int = 100):
    return {"obj_id": obj_id, "color": color, "cur_hex": cur_hex, "cur_pos": 0, "type": 2, "blood": blood}


def _obs(operators, cities):
    return {"operators": operators, "cities": cities, "time": {"cur_step": 1}}


def _policy() -> LearnedPolicy:
    saved_device = os.environ.get("SHENJI_POLICY_DEVICE")
    os.environ["SHENJI_POLICY_DEVICE"] = "cpu"
    policy = LearnedPolicy()
    policy.configure(faction=1, artifact_dir="ai/weights", policy_profile="challenger")
    if saved_device is None:
        os.environ.pop("SHENJI_POLICY_DEVICE", None)
    else:
        os.environ["SHENJI_POLICY_DEVICE"] = saved_device
    return policy


def test_policy_loads_with_objective_fallback_diagnostics() -> None:
    policy = _policy()
    diagnostics = policy.get_diagnostics()

    assert diagnostics["objective_model_loaded"] is False
    assert diagnostics["objective_fallback_reason"] == "objective_model_missing_using_heuristic"


def test_objective_score_affects_city_action_ranking() -> None:
    policy = _policy()
    operator = _operator(1, 1, 1000)
    cities = [
        {"coord": 1001, "value": 80, "flag": -1},
        {"coord": 1009, "value": 10, "flag": 1},
    ]
    observation = _obs([operator, _operator(2, 0, 2000)], cities)
    policy.observe(observation)

    selected, trace = policy.choose_city(operator, cities, observation, _Map())

    assert selected["coord"] == 1001
    assert trace["objective_score"] > 0.0
    assert trace["combined_score"] >= trace["objective_score"]


def test_occupy_city_progress_action_ranks_above_friendly_no_score_action() -> None:
    policy = _policy()
    operator = _operator(1, 1, 1000)
    valuable_enemy = {"coord": 1001, "value": 50, "flag": 0}
    friendly = {"coord": 1000, "value": 80, "flag": 1}
    observation = _obs([operator, _operator(2, 0, 2000)], [valuable_enemy, friendly])
    policy.observe(observation)

    selected, trace = policy.choose_city(operator, [valuable_enemy, friendly], observation, _Map())

    assert selected["coord"] == 1001
    assert trace["objective_score"] >= 100.0


def test_objective_score_is_recorded_in_diagnostics() -> None:
    policy = _policy()
    operator = _operator(1, 1, 1000)
    cities = [{"coord": 1001, "value": 80, "flag": -1}]
    observation = _obs([operator, _operator(2, 0, 2000)], cities)
    policy.observe(observation)
    _selected, trace = policy.choose_city(operator, cities, observation, _Map())
    policy.note_action("move", True, trace)

    diagnostics = policy.get_diagnostics()

    assert diagnostics["objective_score_used"] is True
    assert diagnostics["objective_score_by_action_type"]["move"]["samples"] == 1
    assert diagnostics["selected_action_objective_score"] == trace["objective_score"]
