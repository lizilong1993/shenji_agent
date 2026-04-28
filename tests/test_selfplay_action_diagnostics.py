from __future__ import annotations

from ai.agent import ActionType, Agent
from training.selfplay import _merge_policy_stats


def test_agent_policy_stats_include_action_type_counts() -> None:
    agent = Agent(seed=0, log_level=50)
    agent.valid_action_type_counts = {"Move": 3, "Occupy": 2}
    agent.generated_action_type_counts = {"Move": 1, "Occupy": 1}

    stats = agent.get_policy_stats()

    assert stats["valid_action_type_counts"] == {"Move": 3, "Occupy": 2}
    assert stats["generated_action_type_counts"] == {"Move": 1, "Occupy": 1}


def test_action_type_name_uses_symbolic_names() -> None:
    agent = Agent(seed=0, log_level=50)

    assert agent._action_name(ActionType.Move) == "Move"
    assert agent._action_name(ActionType.Occupy) == "Occupy"


def test_selfplay_merge_preserves_action_type_counts() -> None:
    merged = _merge_policy_stats(
        {
            "valid_action_type_counts": {"Move": 2, "Occupy": 1},
            "generated_action_type_counts": {"Move": 1},
        },
        {
            "valid_action_type_counts": {"Move": 3, "Shoot": 1},
            "generated_action_type_counts": {"Occupy": 1},
        },
    )

    assert merged["valid_action_type_counts"] == {"Move": 5, "Occupy": 1, "Shoot": 1}
    assert merged["generated_action_type_counts"] == {"Move": 1, "Occupy": 1}
