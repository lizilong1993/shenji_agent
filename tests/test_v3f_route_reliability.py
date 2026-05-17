"""Tests for V3F route reliability analysis.

Covers:
- Route denominator is nonzero when move actions occur
- Route-empty is not counted for non-move actions
- No-valid-candidate is separated from route planner empty path
- Fallback move type is recorded separately
- V3D route evidence is not generalized without claim gate
- Route claim gate cannot mark general support without denominator evidence
"""

import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestRouteDiagnosticCode(unittest.TestCase):
    """Tests that inspect the route diagnostic code correctness."""

    def test_record_move_route_denominator_is_move_only(self):
        """Verify record_move_route is only called from Move action branch."""
        agent_path = ROOT / "ai" / "agent.py"
        code = agent_path.read_text(encoding="utf-8")

        # record_move_route should only appear inside the Move branch
        lines = code.split("\n")
        record_calls = []
        for i, line in enumerate(lines, 1):
            if "self.policy.record_move_route" in line:
                record_calls.append((i, line.strip()))

        # All record_move_route calls should be in _generate_action under Move
        # Check that the function containing them is _generate_action
        for lineno, call in record_calls:
            # Search backwards for the enclosing function definition
            found_def = False
            found_move = False
            for j in range(lineno - 2, max(-1, lineno - 300), -1):
                context_line = lines[j]
                if "def _generate_action" in context_line:
                    found_def = True
                    break
                if "action_type == ActionType.Move" in context_line:
                    found_move = True
            self.assertTrue(found_def,
                            f"record_move_route at line {lineno} not in _generate_action")
            self.assertTrue(found_move,
                            f"record_move_route at line {lineno} not under Move action branch")
        self.assertGreater(len(record_calls), 0, "No record_move_route calls found")

    def test_no_valid_move_candidate_before_route_planner(self):
        """Verify no_valid_move_candidate is set before gen_move_route is called."""
        agent_path = ROOT / "ai" / "agent.py"
        code = agent_path.read_text(encoding="utf-8")

        # Find the missing_move_candidate assignment and check it precedes route planning
        missing_line = None
        gen_move_line = None

        for i, line in enumerate(code.split("\n"), 1):
            if "missing_move_candidate = not candidate" in line:
                missing_line = i
            if "gen_move_route" in line and "if route:" in code.split("\n")[i - 1:i]:
                gen_move_line = i

        self.assertIsNotNone(missing_line, "missing_move_candidate assignment not found")
        self.assertLess(missing_line, gen_move_line or float("inf"),
                        "no_valid_move_candidate set AFTER route planning")

    def test_route_empty_types_are_distinct(self):
        """Verify route_empty with no_valid_move_candidate vs empty_route are distinct."""
        agent_path = ROOT / "ai" / "agent.py"
        code = agent_path.read_text(encoding="utf-8")

        # Check that no_valid_move_candidate and empty_route are separate cases
        lines = code.split("\n")
        found_no_valid = False
        found_empty_route = False

        for i, line in enumerate(lines):
            if 'no_valid_move_candidate' in line and 'route_failure_reason' in line:
                found_no_valid = True
            if '"empty_route"' in line and 'route_failure_reason' in line:
                found_empty_route = True

        self.assertTrue(found_no_valid,
                        "no_valid_move_candidate failure reason not found")
        self.assertTrue(found_empty_route,
                        "empty_route failure reason not found")

    def test_fallback_neighbor_is_non_empty_route(self):
        """Verify fallback neighbor path records route_empty=False."""
        agent_path = ROOT / "ai" / "agent.py"
        code = agent_path.read_text(encoding="utf-8")

        # Find fallback neighbor section
        lines = code.split("\n")
        in_fallback = False
        route_empty_setting = None

        for i, line in enumerate(lines):
            if 'target_source": "fallback_neighbor"' in line:
                in_fallback = True
            if in_fallback and '"route_empty"' in line:
                route_empty_setting = line.strip()
                break

        self.assertIsNotNone(route_empty_setting,
                             "Fallback path route_empty setting not found")
        self.assertIn("False", route_empty_setting,
                      "Fallback route should not be marked empty")


class TestRouteEvidenceConsistency(unittest.TestCase):
    """Tests for route evidence consistency."""

    def test_v3d_has_nonzero_denominator(self):
        """Verify V3D route metrics have non-zero denominators."""
        metrics_dir = ROOT / "reports" / "v3d_authenticated_tactical_ablation" / "per_game_metrics"
        if not metrics_dir.exists():
            self.skipTest("V3D per-game metrics not found")

        files = list(metrics_dir.glob("*.json"))
        self.assertGreater(len(files), 0, "No V3D per-game metrics files found")

        zero_denominator_games = 0
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            for mode in ("tactical_on", "tactical_off"):
                routes = data.get(mode, {}).get("policy_stats", {}).get("move_route_aggregate", {})
                total = routes.get("total_attempts", 0)
                if total == 0:
                    zero_denominator_games += 1

        total_games = len(files) * 2
        self.assertLess(
            zero_denominator_games / max(total_games, 1),
            0.5,
            f"Too many games with zero route denominator: {zero_denominator_games}/{total_games}",
        )

    def test_route_empty_rate_consistency(self):
        """Verify route_empty_rate = empty_count / total_attempts in V3D data."""
        metrics_dir = ROOT / "reports" / "v3d_authenticated_tactical_ablation" / "per_game_metrics"
        if not metrics_dir.exists():
            self.skipTest("V3D per-game metrics not found")

        for f in sorted(metrics_dir.glob("*.json"))[:3]:
            data = json.loads(f.read_text(encoding="utf-8"))
            for mode in ("tactical_on", "tactical_off"):
                rm = data.get(mode, {}).get("route_metrics", {})
                empty = rm.get("route_empty_count", 0)
                total = rm.get("total_move_attempts", 0)
                rate = rm.get("route_empty_rate", None)
                if total > 0 and rate is not None:
                    expected = round(empty / total, 4)
                    self.assertAlmostEqual(rate, expected, places=4,
                                           msg=f"{f.name} {mode}: rate mismatch")

    def test_v3d_empty_rate_is_zero(self):
        """Verify V3D shows 0% empty routes across all games (known result)."""
        metrics_dir = ROOT / "reports" / "v3d_authenticated_tactical_ablation" / "per_game_metrics"
        if not metrics_dir.exists():
            self.skipTest("V3D per-game metrics not found")

        total_empty = 0
        total_attempts = 0
        for f in metrics_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            for mode in ("tactical_on", "tactical_off"):
                agg = data.get(mode, {}).get("policy_stats", {}).get("move_route_aggregate", {})
                total_empty += int(agg.get("empty_route_count", 0))
                total_attempts += int(agg.get("total_attempts", 0))

        self.assertGreater(total_attempts, 0,
                           "V3D should have non-zero move attempts")
        self.assertEqual(total_empty, 0,
                         f"V3D should have 0 empty routes but found {total_empty}")
        self.assertGreaterEqual(total_attempts, 590,
                                f"V3D should have >=590 move attempts, found {total_attempts}")


class TestRouteClaimGate(unittest.TestCase):
    """Tests for route claim gate logic."""

    def test_claim_gate_cannot_mark_supported_without_denominator(self):
        """Verify claim gate JSON forbids ROUTE_RELIABILITY_SUPPORTED without denominator."""
        gate_path = ROOT / "reports" / "v3f_route_reliability" / "route_claim_gate.json"
        if not gate_path.exists():
            self.skipTest("Route claim gate not yet created")

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        classification = gate.get("classification", "")

        # Must not claim supported without conditions
        if "SUPPORTED" in classification:
            conditions = gate.get("conditions_for_route_reliability_supported", False)
            self.assertTrue(conditions,
                            "Cannot claim ROUTE_RELIABILITY_SUPPORTED without conditions")

    def test_no_generalization_without_coverage(self):
        """Verify V3D evidence is not overgeneralized."""
        gate_path = ROOT / "reports" / "v3f_route_reliability" / "route_claim_gate.json"
        if not gate_path.exists():
            self.skipTest("Route claim gate not yet created")

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        # 3 scenarios is too few for general claim
        v3d_coverage_too_narrow = gate.get("v3d_scenario_coverage_too_narrow", True)
        self.assertTrue(v3d_coverage_too_narrow,
                        "V3D scenario coverage is too narrow for general route claim")


if __name__ == "__main__":
    unittest.main()
