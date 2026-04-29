from __future__ import annotations

import json
from pathlib import Path

from scripts.build_objective_labels import build_labels, normalize_label, write_outputs


def test_synthetic_match_diagnostics_construct_objective_labels(tmp_path: Path) -> None:
    scenario_root = tmp_path / "scenarios"
    scenario_root.mkdir()
    (scenario_root / "case.json").write_text(
        json.dumps({"cities": [{"coord": 1005, "value": 80, "flag": -1}]}),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "match_results.json").write_text(
        json.dumps(
            {
                "matches": [
                    {
                        "scenario": "case.json",
                        "first_leg": {"candidate": 0.0, "opponent": 0.0},
                        "second_leg": {"candidate": 0.0, "opponent": 0.0},
                        "candidate_policy_stats": {
                            "move_route_samples": [
                                {
                                    "operator_id": 7,
                                    "operator_type": 2,
                                    "current_hex": 1000,
                                    "target_city_id": 1005,
                                    "target_hex": 1005,
                                    "route_len": 3,
                                }
                            ],
                            "trace_samples": [],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    labels = build_labels([run_dir], scenario_root=scenario_root)

    assert labels
    assert labels[0]["operator_id"] == 7
    assert labels[0]["target_city_id"] == 1005
    assert labels[0]["city_value"] == 80
    assert labels[0]["label_confidence"] == "medium"


def test_score_delta_takes_priority_over_heuristic() -> None:
    label = normalize_label(
        {
            "score_before": 3.0,
            "score_after": 8.0,
            "distance_before": 10,
            "distance_after": 1,
            "city_value": 80,
        },
        source_file="synthetic.json",
    )

    assert label["reward_proxy"] == 5.0
    assert label["label_kind"] == "score_delta"
    assert label["label_confidence"] == "high"


def test_occupy_success_generates_positive_reward() -> None:
    label = normalize_label(
        {"occupy_success": True, "city_value": 50},
        source_file="synthetic.json",
    )

    assert label["reward_proxy"] == 50.0
    assert label["label_kind"] == "occupy_success"
    assert label["label_confidence"] == "high"


def test_moving_closer_to_valuable_city_is_positive() -> None:
    label = normalize_label(
        {"distance_before": 8, "distance_after": 3, "city_value": 80},
        source_file="synthetic.json",
    )

    assert label["reward_proxy"] > 0
    assert label["label_kind"] == "distance_to_objective_delta"
    assert label["label_confidence"] == "medium"


def test_moving_away_from_city_is_negative() -> None:
    label = normalize_label(
        {"distance_before": 3, "distance_after": 8, "city_value": 80},
        source_file="synthetic.json",
    )

    assert label["reward_proxy"] < 0
    assert label["label_kind"] == "distance_to_objective_delta"


def test_missing_fields_do_not_crash_and_lower_confidence() -> None:
    label = normalize_label({"action_type": "Move"}, source_file="synthetic.json")

    assert label["reward_proxy"] == 0.0
    assert label["label_confidence"] == "low"
    assert label["distance_before"] is None


def test_summary_records_counts_and_confidence_distribution(tmp_path: Path) -> None:
    labels = [
        normalize_label({"score_delta": 1.0, "action_type": "Occupy"}, source_file="a.json"),
        normalize_label({"distance_before": 4, "distance_after": 2, "action_type": "Move"}, source_file="b.json"),
    ]

    summary = write_outputs(labels, tmp_path)

    assert summary["label_count"] == 2
    assert summary["confidence_counts"] == {"high": 1, "medium": 1}
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["action_type_counts"]["Move"] == 1
