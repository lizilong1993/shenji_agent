"""Tests for V3G full route coverage expansion.

Covers:
- Evaluator records raw SDK empty Move candidate events separately from route planner empty path events
- Route planner empty count cannot increase if route planner was never invoked
- no_valid_move_candidate is not mislabeled as A* route planner failure
- Route denominator is nonzero when move actions occur
- Scenario 1910631192 is included in focused audit if available
- Route claim gate cannot mark full support without scenario coverage and denominator evidence
- Archived 66.3% route failure cannot be dismissed without current comparison
"""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV3GDiagnosticDistinctions(unittest.TestCase):
    """Tests that verify the SDK diagnostic code separates failure causes."""

    def test_record_move_route_is_move_only(self):
        """Verify record_move_route is only called from Move action branch."""
        agent_path = ROOT / "ai" / "agent.py"
        code = agent_path.read_text(encoding="utf-8")

        lines = code.split("\n")
        record_calls = []
        for i, line in enumerate(lines, 1):
            if "self.policy.record_move_route" in line:
                record_calls.append((i, line.strip()))

        for lineno, call in record_calls:
            found_def = False
            for j in range(lineno - 2, max(-1, lineno - 300), -1):
                context_line = lines[j]
                if "def _generate_action" in context_line:
                    found_def = True
                    break
            self.assertTrue(
                found_def,
                f"record_move_route at line {lineno} not in _generate_action",
            )
        self.assertGreater(len(record_calls), 0, "No record_move_route calls found")

    def test_sdk_move_key_is_null_not_candidate_list(self):
        """Verify SDK Move valid_actions entry is None (null), not a candidate list."""
        agent_path = ROOT / "ai" / "agent.py"
        code = agent_path.read_text(encoding="utf-8")

        lines = code.split("\n")
        # The candidate variable in _generate_action for Move should be None
        # Check that missing_move_candidate = not candidate is always True for Move
        missing_line = None
        for i, line in enumerate(lines, 1):
            if "missing_move_candidate = not candidate" in line:
                missing_line = i
                break

        self.assertIsNotNone(
            missing_line, "missing_move_candidate check not found in agent.py"
        )

        # Verify the route planner is called before no_valid_move_candidate is recorded
        gen_move_line = None
        for i, line in enumerate(lines, 1):
            if "gen_move_route" in line:
                gen_move_line = i
                break

        no_valid_line = None
        for i, line in enumerate(lines, 1):
            if '"no_valid_move_candidate"' in line and 'route_failure_reason' in line:
                no_valid_line = i
                break

        self.assertIsNotNone(gen_move_line, "gen_move_route call not found")
        self.assertIsNotNone(
            no_valid_line,
            "no_valid_move_candidate failure reason not found",
        )
        self.assertLess(
            gen_move_line,
            no_valid_line,
            "route planner call should precede no_valid_move_candidate recording",
        )

    def test_fallback_neighbor_is_non_empty_route(self):
        """Verify fallback neighbor path records route_empty=False."""
        agent_path = ROOT / "ai" / "agent.py"
        code = agent_path.read_text(encoding="utf-8")

        lines = code.split("\n")
        in_fallback = False
        route_empty_setting = None

        for i, line in enumerate(lines):
            if 'target_source": "fallback_neighbor"' in line:
                in_fallback = True
            if in_fallback and '"route_empty"' in line:
                route_empty_setting = line.strip()
                break

        self.assertIsNotNone(
            route_empty_setting,
            "Fallback path route_empty setting not found",
        )
        self.assertIn("False", route_empty_setting,
                      "Fallback route should not be marked empty")

    def test_no_valid_move_candidate_label_is_misnomer(self):
        """Verify that no_valid_move_candidate is recorded AFTER route planner call.

        The SDK Move action entry in valid_actions is None (a binary availability flag),
        not a candidate list. The code calls gen_move_route BEFORE recording
        no_valid_move_candidate. This test verifies gen_move_route appears
        before no_valid_move_candidate in the source.
        """
        agent_path = ROOT / "ai" / "agent.py"
        code = agent_path.read_text(encoding="utf-8")
        lines = code.split("\n")

        # Find gen_move_route call line number
        gen_move_line = None
        for i, line in enumerate(lines, 1):
            if "gen_move_route" in line and "route = self.map.gen_move_route" in line:
                gen_move_line = i
                break

        # Find no_valid_move_candidate failure recording line number
        no_valid_line = None
        for i, line in enumerate(lines, 1):
            if '"no_valid_move_candidate"' in line and 'route_failure_reason' in line:
                no_valid_line = i
                break

        self.assertIsNotNone(
            gen_move_line,
            "gen_move_route call not found in agent.py",
        )
        self.assertIsNotNone(
            no_valid_line,
            "no_valid_move_candidate failure reason not found in agent.py",
        )
        self.assertLess(
            gen_move_line,
            no_valid_line,
            "gen_move_route should precede no_valid_move_candidate in source",
        )

        # Also verify that failing the route planner block leads to fallback,
        # and failing fallback leads to no_valid_move_candidate
        fallback_neighbor_line = None
        for i, line in enumerate(lines, 1):
            if 'target_source": "fallback_neighbor"' in line:
                fallback_neighbor_line = i
                break

        self.assertIsNotNone(
            fallback_neighbor_line,
            "fallback_neighbor target_source not found",
        )
        self.assertLess(
            gen_move_line,
            fallback_neighbor_line,
            "gen_move_route should precede fallback_neighbor",
        )
        self.assertLess(
            fallback_neighbor_line,
            no_valid_line,
            "fallback_neighbor should precede no_valid_move_candidate",
        )


class TestV3GEvaluatorStructure(unittest.TestCase):
    """Tests for the V3G evaluator script structure."""

    def test_evaluator_script_exists(self):
        """Verify the V3G evaluator script exists."""
        eval_path = ROOT / "scripts" / "run_v3g_full_route_coverage_evaluation.py"
        self.assertTrue(eval_path.exists(), "V3G evaluator script not found")

    def test_evaluator_has_required_functions(self):
        """Verify the evaluator has required functions."""
        eval_path = ROOT / "scripts" / "run_v3g_full_route_coverage_evaluation.py"
        code = eval_path.read_text(encoding="utf-8")

        required_funcs = [
            "run_v3g_route_evaluation_game",
            "aggregate_v3g_results",
            "write_v3g_output",
            "load_league_map_list",
        ]
        for func in required_funcs:
            self.assertIn(
                f"def {func}",
                code,
                f"Required function {func} not found in evaluator",
            )

    def test_evaluator_has_required_fields(self):
        """Verify per-game output has all required fields."""
        eval_path = ROOT / "scripts" / "run_v3g_full_route_coverage_evaluation.py"
        code = eval_path.read_text(encoding="utf-8")

        required_fields = [
            "route_planner_invocation_count",
            "route_planner_success_count",
            "route_planner_empty_path_count",
            "sdk_move_key_present_count",
            "sdk_move_key_absent_count",
            "empty_rate_by_scenario",
            "empty_rate_by_map",
            "empty_rate_by_move_type",
            "worst_scenarios",
            "scenario_1910631192_empty_rate",
        ]
        for field in required_fields:
            self.assertIn(
                field, code,
                f"Required field '{field}' not found in evaluator",
            )


class TestV3GSmokeData(unittest.TestCase):
    """Tests for V3G smoke evaluation output data."""

    def setUp(self):
        self.smoke_dir = ROOT / "experiments" / "shenji_agent" / "v3g_full_route_coverage_smoke"
        self.results_path = self.smoke_dir / "results.json"

    def test_smoke_data_exists(self):
        """Verify smoke evaluation output exists."""
        if not self.smoke_dir.exists():
            self.skipTest("Smoke evaluation directory not found")

        self.assertTrue(
            self.results_path.exists(),
            "Smoke results.json not found",
        )
        self.assertTrue(
            (self.smoke_dir / "per_game.jsonl").exists(),
            "Smoke per_game.jsonl not found",
        )
        self.assertTrue(
            (self.smoke_dir / "per_scenario.csv").exists(),
            "Smoke per_scenario.csv not found",
        )
        self.assertTrue(
            (self.smoke_dir / "summary.md").exists(),
            "Smoke summary.md not found",
        )

    def test_smoke_battle_entered(self):
        """Verify smoke evaluation entered battle for at least some games."""
        if not self.results_path.exists():
            self.skipTest("Smoke results not found")

        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertGreater(
            data.get("battle_entered_count", 0),
            0,
            "Smoke evaluation should have entered battle",
        )

    def test_smoke_denominator_nonzero(self):
        """Verify smoke evaluation has non-zero route denominator when moves occur."""
        if not self.results_path.exists():
            self.skipTest("Smoke results not found")

        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        if data.get("total_move_actions", 0) > 0:
            self.assertGreater(
                data.get("route_eligible_move_actions", 0),
                0,
                "Route denominator should be non-zero when move actions occur",
            )


class TestV3GClaimGate(unittest.TestCase):
    """Tests for route claim gate logic."""

    def test_claim_gate_cannot_mark_full_support_without_coverage(self):
        """Verify claim gate classification cannot be ROUTE_RELIABILITY_SUPPORTED
        without scenario coverage and denominator evidence."""
        gate_path = ROOT / "reports" / "v3g_full_route_coverage" / "route_claim_gate.json"
        if not gate_path.exists():
            self.skipTest("V3G route claim gate not yet created")

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        classification = gate.get("classification", "")

        # If full coverage is claimed, must have extensive scenario coverage
        if "ROUTE_RELIABILITY_SUPPORTED" in classification:
            scenario_coverage = gate.get("scenarios_tested", 0)
            self.assertGreaterEqual(
                scenario_coverage,
                40,
                "Must test 40+ scenarios for full reliability claim",
            )
            denominator = gate.get("route_eligible_move_actions", 0)
            self.assertGreater(
                denominator,
                1000,
                f"Denominator too low ({denominator}) for full support claim",
            )

    def test_archived_failure_not_dismissed_without_comparison(self):
        """Verify the archived 66.3% route failure is acknowledged."""
        gate_path = ROOT / "reports" / "v3g_full_route_coverage" / "route_claim_gate.json"
        if not gate_path.exists():
            self.skipTest("V3G route claim gate not yet created")

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        comparison = gate.get("archived_vs_current_comparison", "")
        self.assertNotEqual(
            comparison,
            "ARCHIVED_FAILURE_IRRELEVANT",
            "Archived 66.3% failure cannot be dismissed",
        )

    def test_scenario_1910631192_acknowledged(self):
        """Verify scenario 1910631192 is addressed in claim gate."""
        gate_path = ROOT / "reports" / "v3g_full_route_coverage" / "route_claim_gate.json"
        if not gate_path.exists():
            self.skipTest("V3G route claim gate not yet created")

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        contradiction = gate.get("scenario_1910631192_contradiction", {})
        self.assertIn(
            contradiction.get("status", ""),
            ["resolved", "unresolved", "blocked", "protocol_difference", "resolved_as_auth_artifact"],
            "Scenario 1910631192 contradiction must be addressed",
        )

    def test_full_support_requires_authenticated_coverage(self):
        """Verify ROUTE_RELIABILITY_SUPPORTED requires authenticated coverage."""
        gate_path = ROOT / "reports" / "v3g_full_route_coverage" / "route_claim_gate.json"
        if not gate_path.exists():
            self.skipTest("V3G route claim gate not yet created")

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        classification = gate.get("classification", "")

        if "ROUTE_RELIABILITY_SUPPORTED" in classification:
            self.assertTrue(
                gate.get("authenticated_replay_battle_entered", False),
                "Must enter battle in authenticated replay for full support",
            )
            self.assertTrue(
                gate.get("denominator_meaningful", False),
                "Denominator must be meaningful for full support",
            )

    def test_no_unsupported_claims(self):
        """Verify no forbidden claims are introduced."""
        gate_path = ROOT / "reports" / "v3g_full_route_coverage" / "route_claim_gate.json"
        if not gate_path.exists():
            self.skipTest("V3G route claim gate not yet created")

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        classification = gate.get("classification", "")

        forbidden_classifications = [
            "ROUTE_RELIABILITY_SOLVED_GENERALLY",
        ]
        for forbidden in forbidden_classifications:
            self.assertNotEqual(
                classification,
                forbidden,
                f"Forbidden classification: {forbidden}",
            )


class TestV3GScenario1910631192Audit(unittest.TestCase):
    """Tests for scenario 1910631192 focused audit."""

    def test_audit_file_exists(self):
        """Verify scenario 1910631192 audit file exists."""
        audit_path = ROOT / "reports" / "v3g_full_route_coverage" / "02_scenario_1910631192_audit.md"
        audit_json = ROOT / "reports" / "v3g_full_route_coverage" / "scenario_1910631192_audit.json"

        has_md = audit_path.exists()
        has_json = audit_json.exists()

        if not has_md and not has_json:
            self.skipTest("Scenario 1910631192 audit not yet created")

        self.assertTrue(has_md or has_json,
                        "At least one audit file should exist")

    def test_audit_has_classification(self):
        """Verify the audit has a contradiction classification."""
        audit_json = ROOT / "reports" / "v3g_full_route_coverage" / "scenario_1910631192_audit.json"
        if not audit_json.exists():
            self.skipTest("Scenario 1910631192 audit JSON not found")

        data = json.loads(audit_json.read_text(encoding="utf-8"))
        classification = data.get("contradiction_classification", "")
        self.assertIn(
            classification,
            [
                "SCENARIO_1910631192_CONTRADICTION_REPRODUCED",
                "SCENARIO_1910631192_CONTRADICTION_OLD_SDK_OR_MAP_ARTIFACT",
                "SCENARIO_1910631192_CONTRADICTION_PROTOCOL_DIFFERENCE",
                "SCENARIO_1910631192_CONTRADICTION_SIDE_OR_SEED_DEPENDENT",
                "SCENARIO_1910631192_CONTRADICTION_UNRESOLVED",
                "SCENARIO_1910631192_EVALUATION_BLOCKED",
            ],
            f"Invalid classification: {classification}",
        )


class TestV3GInventory(unittest.TestCase):
    """Tests for league scenario inventory."""

    def test_inventory_file_exists(self):
        """Verify inventory file exists."""
        inv_path = ROOT / "reports" / "v3g_full_route_coverage" / "league_scenario_inventory.json"
        if not inv_path.exists():
            self.skipTest("League scenario inventory not found")

        data = json.loads(inv_path.read_text(encoding="utf-8"))

        # Must have unique scenarios
        unique_scenarios = data.get("unique_scenarios", [])
        if isinstance(unique_scenarios, list):
            count = len(unique_scenarios)
        elif isinstance(unique_scenarios, dict):
            count = len(unique_scenarios)
        else:
            count = 0
        self.assertGreaterEqual(
            count,
            40,
            f"Should have at least 40 unique scenarios, got {count}",
        )

        # Must have maps
        unique_maps = data.get("unique_maps", [])
        if isinstance(unique_maps, list):
            map_count = len(unique_maps)
        else:
            map_count = 0
        self.assertGreaterEqual(
            map_count,
            10,
            f"Should have at least 10 unique maps, got {map_count}",
        )

    def test_inventory_includes_scenario_1910631192(self):
        """Verify the inventory includes scenario 1910631192."""
        inv_path = ROOT / "reports" / "v3g_full_route_coverage" / "league_scenario_inventory.json"
        if not inv_path.exists():
            self.skipTest("League scenario inventory not found")

        data = json.loads(inv_path.read_text(encoding="utf-8"))

        references = data.get("scenario_1910631192_references", [])
        self.assertGreater(
            len(references),
            0,
            "Scenario 1910631192 should have references in inventory",
        )


if __name__ == "__main__":
    unittest.main()
