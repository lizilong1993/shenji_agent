from __future__ import annotations

from training.selfplay import (
    _build_step_objective_records,
    _collect_objective_sidecars,
    _paired_exploiter_result,
    _paired_score,
    _terminal_objective_score,
)


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


def test_terminal_objective_score_uses_reward_when_available() -> None:
    observation = _obs(red_reward=7.0)[0]

    assert _terminal_objective_score(observation, 0) == 7.0


def test_terminal_objective_score_falls_back_to_real_objective_state() -> None:
    observation = _obs(red_reward=0.0)[0]
    observation["cities"][0]["flag"] = 0
    observation["operators"][0]["blood"] = 5
    observation["operators"][1]["blood"] = 1

    assert _terminal_objective_score(observation, 0) > _terminal_objective_score(observation, 1)


def test_paired_score_uses_objective_tiebreak_for_reward_ties() -> None:
    candidate_stats = {
        "score_means": {
            "move": {"samples": 2, "objective_score": 10.0},
            "shoot": {"samples": 1, "objective_score": 5.0},
        }
    }
    opponent_stats = {
        "score_means": {
            "move": {"samples": 2, "objective_score": 2.0},
            "shoot": {"samples": 0, "objective_score": 0.0},
        }
    }

    assert (
        _paired_score(
            0.0,
            0.0,
            0.0,
            0.0,
            candidate_stats=candidate_stats,
            opponent_stats=opponent_stats,
        )
        == 1.0
    )


def test_paired_exploiter_result_is_side_symmetric_for_reward_ties() -> None:
    candidate_stats = {
        "counts": {"move": {"total": 1, "learned": 1}, "shoot": {"total": 0, "learned": 0}},
        "score_means": {
            "move": {"samples": 1, "objective_score": 25.0},
            "shoot": {"samples": 0, "objective_score": 0.0},
        },
    }
    exploiter_stats = {
        "counts": {"move": {"total": 1, "learned": 0}, "shoot": {"total": 0, "learned": 0}},
        "score_means": {
            "move": {"samples": 1, "objective_score": 5.0},
            "shoot": {"samples": 0, "objective_score": 0.0},
        },
    }

    exploiter_win_rate, paired_candidate_stats = _paired_exploiter_result(
        candidate_red_score=93.0,
        exploiter_blue_score=137.0,
        exploiter_red_score=93.0,
        candidate_blue_score=137.0,
        candidate_red_stats=candidate_stats,
        exploiter_blue_stats=exploiter_stats,
        exploiter_red_stats=exploiter_stats,
        candidate_blue_stats=candidate_stats,
    )

    assert exploiter_win_rate == 0.0
    assert paired_candidate_stats["learned_path_coverage"] == 1.0


def test_paired_exploiter_result_counts_draw_as_not_exploiter_win() -> None:
    stats = {
        "counts": {"move": {"total": 1, "learned": 1}, "shoot": {"total": 0, "learned": 0}},
        "score_means": {
            "move": {"samples": 1, "objective_score": 10.0},
            "shoot": {"samples": 0, "objective_score": 0.0},
        },
    }

    exploiter_win_rate, _ = _paired_exploiter_result(
        candidate_red_score=10.0,
        exploiter_blue_score=20.0,
        exploiter_red_score=10.0,
        candidate_blue_score=20.0,
        candidate_red_stats=stats,
        exploiter_blue_stats=stats,
        exploiter_red_stats=stats,
        candidate_blue_stats=stats,
    )

    assert exploiter_win_rate == 0.0
