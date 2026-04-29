from __future__ import annotations

from training.selfplay import _build_step_objective_records, _collect_objective_sidecars


def _obs(red_reward: float = 0.0, blue_reward: float = 0.0):
    base = {
        "cities": [{"coord": 1001, "value": 50, "flag": -1}],
        "operators": [
            {"obj_id": 1, "color": 0, "type": 2, "cur_hex": 1001, "blood": 3},
            {"obj_id": 2, "color": 1, "type": 2, "cur_hex": 1002, "blood": 3},
        ],
        "reward": red_reward,
    }
    blue = {**base, "reward": blue_reward}
    return [base, blue]


def test_step_objective_records_capture_score_damage_and_occupy() -> None:
    before = _obs(red_reward=0.0, blue_reward=0.0)
    after = _obs(red_reward=5.0, blue_reward=0.0)
    after[0]["cities"][0]["flag"] = 0
    after[1]["cities"][0]["flag"] = 0
    after[0]["operators"][1]["blood"] = 1
    after[1]["operators"][1]["blood"] = 1

    progress, outcomes, timeline = _build_step_objective_records(
        match_id="m1",
        turn=3,
        before_state=before,
        after_state=after,
        actions=[{"actor": 1, "obj_id": 1, "type": 5}],
    )

    assert progress[0]["red"]["score_delta"] == 5.0
    assert progress[0]["red"]["damage_dealt"] == 2.0
    assert outcomes[0]["occupy_success"] is True
    assert outcomes[0]["target_owner_before"] == -1
    assert outcomes[0]["target_owner_after"] == 0
    assert timeline["red_city_value_controlled"] == 50.0


def test_collect_objective_sidecars_summarizes_records() -> None:
    matches = [
        {
            "candidate_policy_stats": {
                "objective_progress_turns": [{"red": {"score_delta": 1.0, "damage_dealt": 0.0}, "blue": {}}],
                "action_outcomes": [{"occupy_available": True, "occupy_success": True, "score_delta": 1.0}],
                "city_control_timeline": [
                    {"red_city_value_controlled": 0.0, "blue_city_value_controlled": 0.0},
                    {"red_city_value_controlled": 50.0, "blue_city_value_controlled": 0.0},
                ],
            }
        }
    ]

    sidecars = _collect_objective_sidecars(matches)

    assert sidecars["objective_progress"]["summary"]["turn_records"] == 1
    assert sidecars["action_outcomes"]["summary"]["occupy_opportunities"] == 1
    assert sidecars["action_outcomes"]["summary"]["occupy_successes"] == 1
    assert sidecars["city_control"]["summary"]["city_value_change_records"] == 1
