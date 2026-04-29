from __future__ import annotations

import json

from training.selfplay import (
    _build_phase_action_records,
    _build_phase_action_sidecars,
    _summarize_phase_action_records,
)


def _obs(*, stage: int, valid_actions: dict[int, dict[int, object]], reward: float = 0.0) -> dict[str, object]:
    return {
        "time": {"stage": stage, "cur_step": 0},
        "reward": reward,
        "operators": [{"obj_id": 1, "color": 0, "type": 2, "cur_hex": 1001, "blood": 10}],
        "cities": [{"coord": 1001, "value": 50, "flag": -1}],
        "valid_actions": valid_actions,
    }


def test_setup_only_turns_are_recorded_with_reason() -> None:
    before = [_obs(stage=1, valid_actions={1: {303: [], 333: []}}), _obs(stage=1, valid_actions={2: {303: []}})]
    after = [_obs(stage=1, valid_actions={1: {303: []}}), _obs(stage=1, valid_actions={2: {303: []}})]

    records = _build_phase_action_records(
        match_id="m1",
        turn=0,
        before_state=before,
        after_state=after,
        actions=[{"actor": 1, "type": 333}, {"actor": 11, "type": 333}],
        done=False,
        env_backend="train_env.TrainEnv",
        auth_status={"status": "mac_match_unverified"},
    )

    assert records[0]["phase"] == "deployment"
    assert records[0]["deployment_action_count"] == 1
    assert records[0]["no_battle_action_reason"] == "deployment_or_setup_phase"


def test_battle_turns_with_move_shoot_occupy_are_recorded() -> None:
    before = [
        _obs(stage=2, valid_actions={1: {1: None, 2: [{}], 5: None}}),
        _obs(stage=2, valid_actions={2: {1: None}}),
    ]
    after = [_obs(stage=2, valid_actions={}, reward=1.0), _obs(stage=2, valid_actions={}, reward=0.0)]

    records = _build_phase_action_records(
        match_id="m1",
        turn=1,
        before_state=before,
        after_state=after,
        actions=[{"actor": 1, "obj_id": 1, "type": 1, "move_path": [1002]}],
        done=False,
        env_backend="train_env.TrainEnv",
        auth_status={"status": "mac_match_unverified"},
    )

    assert records[0]["phase"] == "battle"
    assert records[0]["move_action_count"] == 1
    assert records[0]["shoot_action_count"] == 1
    assert records[0]["occupy_action_count"] == 1
    assert records[0]["no_battle_action_reason"] is None


def test_summary_aggregate_counts_phase_actions() -> None:
    records = [
        {
            "match_id": "m1",
            "turn": 0,
            "phase": "deployment",
            "auth_status": "failed_mac_mismatch",
            "move_action_count": 0,
            "shoot_action_count": 0,
            "occupy_action_count": 0,
            "selected_action_type_counts": {"333": 1},
            "no_battle_action_reason": "auth_failed_or_unverified_stage_stuck",
        },
        {
            "match_id": "m1",
            "turn": 1,
            "phase": "battle",
            "auth_status": "mac_match_unverified",
            "move_action_count": 1,
            "shoot_action_count": 0,
            "occupy_action_count": 0,
            "selected_action_type_counts": {"Move": 1},
            "no_battle_action_reason": None,
        },
    ]

    summary = _summarize_phase_action_records(records)

    assert summary["total_turn_records"] == 2
    assert summary["battle_phase_entered"] is True
    assert summary["first_battle_turn"] == 1
    assert summary["turns_with_move"] == 1
    assert summary["selected_move_actions"] == 1
    assert summary["auth_failure_count"] == 1


def test_no_move_shoot_occupy_has_explicit_reason() -> None:
    before = [_obs(stage=2, valid_actions={1: {6: [{}]}}), _obs(stage=2, valid_actions={2: {6: [{}]}})]
    after = [_obs(stage=2, valid_actions={}), _obs(stage=2, valid_actions={})]

    records = _build_phase_action_records(
        match_id="m2",
        turn=4,
        before_state=before,
        after_state=after,
        actions=[{"actor": 1, "obj_id": 1, "type": 6, "target_state": 4}],
        done=False,
        env_backend="train_env.TrainEnv",
        auth_status={"status": "mac_match_unverified"},
    )

    assert records[0]["no_battle_action_reason"] == "no_move_shoot_occupy_valid_actions"


def test_phase_diagnostics_do_not_pollute_match_stats_and_serialize() -> None:
    records = _build_phase_action_sidecars(
        [
            {
                "match_id": "m3",
                "turn": 0,
                "side": 0,
                "phase": "deployment",
                "stage": 1,
                "phase_transition": "1->1",
                "valid_action_type_counts": {"303": 1},
                "selected_action_type_counts": {"333": 1},
                "move_action_count": 0,
                "shoot_action_count": 0,
                "occupy_action_count": 0,
                "auth_status": "failed_mac_mismatch",
                "no_battle_action_reason": "auth_failed_or_unverified_stage_stuck",
            }
        ]
    )

    payload = json.dumps(records)
    assert "phase_action" in records
    assert "valid_action_timeline" in records
    assert len(payload) < 20000
