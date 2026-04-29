from __future__ import annotations

from training.selfplay import _apply_battle_phase_gate, _build_phase_action_sidecars


def test_setup_only_match_is_not_valid_promotion_evidence() -> None:
    decision = {"accepted": True, "failures": []}
    phase_summary = {
        "battle_phase_entered": False,
        "selected_move_actions": 0,
        "selected_shoot_actions": 0,
        "selected_occupy_actions": 0,
    }

    gated = _apply_battle_phase_gate(decision, phase_summary)

    assert gated["accepted"] is False
    assert "battle_phase_not_entered" in gated["failures"]


def test_battle_phase_entered_false_blocks_promotion() -> None:
    gated = _apply_battle_phase_gate(
        {"accepted": True, "failures": []},
        {
            "battle_phase_entered": False,
            "selected_move_actions": 3,
            "selected_shoot_actions": 0,
            "selected_occupy_actions": 0,
        },
    )

    assert gated["accepted"] is False
    assert gated["battle_phase_entered"] is False


def test_no_move_shoot_occupy_selected_is_invalid_evidence_reason() -> None:
    gated = _apply_battle_phase_gate(
        {"accepted": True, "failures": []},
        {
            "battle_phase_entered": True,
            "selected_move_actions": 0,
            "selected_shoot_actions": 0,
            "selected_occupy_actions": 0,
        },
    )

    assert gated["accepted"] is False
    assert "no_move_shoot_occupy_selected_actions" in gated["failures"]


def test_runner_can_mark_invalid_scenario_without_reducing_existing_failures() -> None:
    gated = _apply_battle_phase_gate(
        {"accepted": False, "failures": ["overall_ci_lower"]},
        {
            "battle_phase_entered": False,
            "selected_move_actions": 0,
            "selected_shoot_actions": 0,
            "selected_occupy_actions": 0,
        },
    )

    assert gated["accepted"] is False
    assert gated["failures"][0] == "overall_ci_lower"
    assert "battle_phase_not_entered" in gated["failures"]


def test_phase_action_diagnostics_sidecar_contract() -> None:
    sidecars = _build_phase_action_sidecars(
        [
            {
                "match_id": "m1",
                "turn": 0,
                "side": 0,
                "phase": "deployment",
                "stage": 1,
                "phase_transition": "1->1",
                "valid_action_type_counts": {"Move": 2},
                "selected_action_type_counts": {"333": 1},
                "move_action_count": 2,
                "shoot_action_count": 0,
                "occupy_action_count": 0,
                "auth_status": "failed_mac_mismatch",
                "no_battle_action_reason": "auth_failed_or_unverified_stage_stuck",
            }
        ]
    )

    assert sidecars["phase_action"]["summary"]["battle_phase_entered"] is False
    assert sidecars["valid_action_timeline"]["turns"][0]["valid_action_type_counts"] == {"Move": 2}
