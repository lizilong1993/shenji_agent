from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_test_stubs() -> None:
    from tests.test_mainline_contract import _install_fake_numpy, _install_fake_torch

    _install_fake_numpy()
    _install_fake_torch()


def _fake_map_dims(rows: int = 10, cols: int = 10) -> tuple[dict, list, object]:
    """Build a fake basic_data, cost_data, see_data for a simple map."""
    map_data = []
    for r in range(rows):
        row = []
        for c in range(cols):
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append(nr * 100 + nc)
                else:
                    neighbors.append(-1)
            row.append(
                {
                    "pos": r * 100 + c,
                    "elev": 200,
                    "cond": 0,
                    "roads": [0, 0, 0, 0, 0, 0],
                    "rivers": [0, 0, 0, 0, 0, 0],
                    "neighbors": neighbors,
                }
            )
        map_data.append(row)
    basic_data = {"map_id": 999, "ele_grade": [], "map_data": map_data}

    cost_data = []
    for mode in range(4):
        mode_cost = []
        for r in range(rows):
            row_cost = []
            for c in range(cols):
                cell: dict[int, float] = {}
                neighbors = map_data[r][c]["neighbors"]
                for n_hex in neighbors:
                    if n_hex >= 0:
                        cell[n_hex] = 1.0
                row_cost.append(cell)
            mode_cost.append(row_cost)
        cost_data.append(mode_cost)

    see_data = None
    return basic_data, cost_data, see_data


def _fake_operators() -> list:
    return [
        {"obj_id": 0, "type": 2, "sub_type": 1, "cur_hex": 0, "color": 0, "blood": 100, "move_state": 0},
        {"obj_id": 1, "type": 2, "sub_type": 1, "cur_hex": 101, "color": 1, "blood": 100, "move_state": 0},
    ]


def _fake_cities() -> list:
    return [
        {"coord": 505, "value": 80, "name": "CityA", "flag": -1},
        {"coord": 909, "value": 50, "name": "CityB", "flag": -1},
    ]


def _fake_observation(operators=None, cities=None, valid_actions=None):
    operators = operators if operators is not None else _fake_operators()
    cities = cities if cities is not None else _fake_cities()
    if valid_actions is None:
        valid_actions = {
            0: {1: [], 2: [], 3: []},
            1: {1: [], 2: [], 3: []},
        }
    return {
        "operators": operators,
        "cities": cities,
        "valid_actions": valid_actions,
        "time": {"cur_step": 100, "stage": 2},
        "reward": 0.0,
        "done": False,
        "role_and_grouping_info": {1: {"operators": [0, 1]}},
        "communication": [],
    }


class MoveRouteDiagnosticsContractTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_test_stubs()
        from ai.agent import Agent, ActionType

        self.ActionType = ActionType

    def _make_agent(self, use_optimization: bool = True) -> "Agent":
        from ai.agent import Agent
        from ai.policy import LearnedPolicy

        agent = Agent(seed=42, log_level=50)
        basic_data, cost_data, see_data = _fake_map_dims(10, 10)
        with mock.patch.object(LearnedPolicy, "configure", return_value=None):
            agent.setup(
                {
                    "scenario": "test",
                    "faction": 0,
                    "seat": 1,
                    "role": 1,
                    "user_name": "CurrentAI",
                    "user_id": 1,
                    "policy_profile": "challenger" if use_optimization else "baseline",
                    "policy_artifacts_dir": None,
                    "use_ai_optimization": use_optimization,
                    "basic_data": basic_data,
                    "cost_data": cost_data,
                    "see_data": see_data,
                    "state": {},
                }
            )
        if use_optimization:
            agent.use_ai_optimization = True
            agent.policy.intent_model = mock.MagicMock()
            agent.policy.situation_model = mock.MagicMock()
        return agent

    def test_empty_route_recorded_with_route_empty_true_and_failure_reason(self) -> None:
        """Route empty -> diagnostics record route_empty=True and route_failure_reason."""
        agent = self._make_agent(use_optimization=True)
        obs = _fake_observation(
            cities=[{"coord": 99999, "value": 80}],  # unreachable city
            valid_actions={0: {2: []}},
        )
        agent.observation = obs
        agent.controllable_ops = [0]
        agent.team_info = {1: {"operators": [0]}}

        agent._generate_action(0, self.ActionType.Move, [{}])

        agg = agent.policy.get_diagnostics().get("move_route_aggregate", {})
        self.assertEqual(agg.get("total_attempts"), 1)
        self.assertEqual(agg.get("empty_route_count"), 1)
        self.assertIn("empty_route", agg.get("route_failure_reasons", {}))
        samples = agent.policy.get_diagnostics().get("move_route_samples", [])
        self.assertGreater(len(samples), 0)
        self.assertTrue(samples[0].get("route_empty"))

    def test_fallback_neighbor_not_counted_as_learned_success(self) -> None:
        """Move fallback (neighbor step) is not counted as learned move success."""
        agent = self._make_agent(use_optimization=True)
        obs = _fake_observation(
            # City very far, A* should find a route but if map is small, it will
            # For a truly isolated case, use a city outside map
            cities=[{"coord": 555, "value": 80}],
            valid_actions={0: {2: []}},
        )
        agent.observation = obs
        agent.controllable_ops = [0]
        agent.team_info = {1: {"operators": [0]}}

        action = agent._generate_action(0, self.ActionType.Move, [])
        diag = agent.policy.get_diagnostics()
        counts = diag.get("counts", {}).get("move", {})
        # If fallback_neighbor was used, it should NOT be counted as learned
        self.assertEqual(counts.get("learned", 0), 0)
        # But it SHOULD be counted as a total move
        if action is not None:
            self.assertEqual(counts.get("total", 0), 1)

    def test_selected_by_learned_false_when_rule_fallback(self) -> None:
        """selected_by_learned is false when rule/fallback move is selected."""
        agent = self._make_agent(use_optimization=False)
        obs = _fake_observation(
            valid_actions={0: {2: []}},
        )
        agent.observation = obs
        agent.controllable_ops = [0]
        agent.team_info = {1: {"operators": [0]}}

        agent._generate_action(0, self.ActionType.Move, [])
        samples = agent.policy.get_diagnostics().get("move_route_samples", [])
        self.assertGreater(len(samples), 0)
        for sample in samples:
            self.assertFalse(sample.get("selected_by_learned", True))

    def test_move_path_coverage_not_inflated_by_shoot(self) -> None:
        """move_path_coverage is not inflated by shoot actions."""
        agent = self._make_agent(use_optimization=True)
        diag = agent.policy.get_diagnostics()
        move_cov = diag.get("move_path_coverage", 0.0)
        shoot_cov = diag.get("shoot_path_coverage", 0.0)
        # Coverage should be 0 when no actions taken
        self.assertEqual(move_cov, 0.0)
        self.assertEqual(shoot_cov, 0.0)

        # Simulate a shoot action
        agent.policy.note_action("shoot", True, {"obj_id": 0, "candidate_id": 1})
        diag = agent.policy.get_diagnostics()
        shoot_cov = diag.get("shoot_path_coverage", 0.0)
        move_cov = diag.get("move_path_coverage", 0.0)
        self.assertGreater(shoot_cov, 0.0)
        self.assertEqual(move_cov, 0.0)  # shoot should not affect move coverage

    def test_missing_target_recorded_as_failure_reason(self) -> None:
        """Missing target city results in diagnostic recording with route_empty=True."""
        agent = self._make_agent(use_optimization=True)
        # City at hex 99999 is invalid/unreachable
        obs = _fake_observation(
            cities=[{"coord": 99999, "value": 80}],
            valid_actions={0: {2: []}},
        )
        agent.observation = obs
        agent.controllable_ops = [0]
        agent.team_info = {1: {"operators": [0]}}

        result = agent._generate_action(0, self.ActionType.Move, [])
        self.assertIsNone(result)
        samples = agent.policy.get_diagnostics().get("move_route_samples", [])
        self.assertGreater(len(samples), 0)
        self.assertTrue(samples[0].get("route_empty"))
        self.assertIn(samples[0].get("route_failure_reason"), ("empty_route", "no_valid_move_candidate"))

    def test_empty_route_planner_increases_empty_route_count(self) -> None:
        """When route planner returns empty, aggregate empty_route count increases."""
        agent = self._make_agent(use_optimization=True)
        obs = _fake_observation(
            cities=[{"coord": 99999, "value": 80}],
            valid_actions={0: {2: []}},
        )
        agent.observation = obs
        agent.controllable_ops = [0]
        agent.team_info = {1: {"operators": [0]}}

        for _ in range(3):
            agent._generate_action(0, self.ActionType.Move, [])

        agg = agent.policy.get_diagnostics().get("move_route_aggregate", {})
        self.assertEqual(agg.get("total_attempts"), 3)
        self.assertEqual(agg.get("empty_route_count"), 3)

    def test_record_move_route_persists_in_diagnostics(self) -> None:
        """record_move_route data is present in get_diagnostics output."""
        agent = self._make_agent(use_optimization=True)
        agent.policy.record_move_route(
            {
                "operator_id": 0,
                "operator_type": 2,
                "current_hex": 100,
                "move_type": 0,
                "move_type_name": "Maneuver",
                "target_kind": "city",
                "target_source": "learned",
                "target_city_id": 505,
                "target_hex": 505,
                "reachable_city_count": 2,
                "total_cities": 2,
                "route_len": 5,
                "route_empty": False,
                "route_failure_reason": "none",
                "selected_by_learned": True,
            }
        )
        diag = agent.policy.get_diagnostics()
        samples = diag.get("move_route_samples", [])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["operator_id"], 0)
        self.assertEqual(samples[0]["target_city_id"], 505)
        self.assertEqual(samples[0]["route_len"], 5)
        self.assertFalse(samples[0]["route_empty"])
        agg = diag.get("move_route_aggregate", {})
        self.assertEqual(agg.get("total_attempts"), 1)
        self.assertEqual(agg.get("empty_route_count"), 0)
        self.assertEqual(agg.get("reachable_city_total"), 2)

    def test_move_type_distribution_recorded(self) -> None:
        """Move route diagnostics record by_move_type distribution."""
        agent = self._make_agent(use_optimization=True)
        for mt_name in ["Maneuver", "March", "Maneuver", "Walk"]:
            agent.policy.record_move_route(
                {
                    "operator_id": 0,
                    "operator_type": 2,
                    "current_hex": 100,
                    "move_type": 0,
                    "move_type_name": mt_name,
                    "target_kind": "city",
                    "target_source": "test",
                    "target_city_id": 505,
                    "target_hex": 505,
                    "reachable_city_count": 1,
                    "total_cities": 3,
                    "route_len": 3,
                    "route_empty": False,
                    "route_failure_reason": "none",
                    "selected_by_learned": False,
                }
            )
        agg = agent.policy.get_diagnostics().get("move_route_aggregate", {})
        by_mt = agg.get("by_move_type", {})
        self.assertEqual(by_mt["Maneuver"]["total"], 2)
        self.assertEqual(by_mt["March"]["total"], 1)
        self.assertEqual(by_mt["Walk"]["total"], 1)

    def test_reset_clears_move_route_diagnostics(self) -> None:
        """reset() clears move route diagnostics."""
        agent = self._make_agent(use_optimization=True)
        agent.policy.record_move_route(
            {
                "operator_id": 0,
                "operator_type": 2,
                "current_hex": 100,
                "move_type": 0,
                "move_type_name": "Maneuver",
                "target_kind": "city",
                "target_source": "test",
                "target_city_id": 505,
                "target_hex": 505,
                "reachable_city_count": 1,
                "total_cities": 3,
                "route_len": 3,
                "route_empty": False,
                "route_failure_reason": "none",
                "selected_by_learned": False,
            }
        )
        self.assertEqual(len(agent.policy.get_diagnostics().get("move_route_samples", [])), 1)

        agent.policy.reset()
        diag = agent.policy.get_diagnostics()
        self.assertEqual(len(diag.get("move_route_samples", [])), 0)
        agg = diag.get("move_route_aggregate", {})
        self.assertEqual(agg.get("total_attempts"), 0)

    def test_fallback_move_type_recorded_in_target_source(self) -> None:
        """When a fallback move type is used, target_source records it in diagnostics."""
        agent = self._make_agent(use_optimization=True)
        # Directly record a diagnostic simulating a fallback move type scenario
        agent.policy.record_move_route({
            "operator_id": 0,
            "operator_type": 2,
            "current_hex": 0,
            "move_type": 0,
            "move_type_name": "Maneuver",
            "target_kind": "city",
            "target_source": "learned_fallback_move_type_0",
            "target_city_id": 505,
            "target_hex": 505,
            "reachable_city_count": 1,
            "total_cities": 1,
            "route_len": 10,
            "route_empty": False,
            "route_failure_reason": "none",
            "selected_by_learned": True,
        })
        samples = agent.policy.get_diagnostics().get("move_route_samples", [])
        self.assertGreater(len(samples), 0)
        target_source = samples[0].get("target_source", "")
        self.assertIn("learned_fallback", target_source)
        self.assertEqual(samples[0]["move_type_name"], "Maneuver")
        self.assertTrue(samples[0]["selected_by_learned"])


if __name__ == "__main__":
    unittest.main()
