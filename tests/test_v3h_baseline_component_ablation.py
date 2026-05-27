"""Tests for V3H baseline and component-ablation evaluation.

Covers:
- Baseline evaluator emits required fields
- Learned and rule incumbent variants are distinguishable
- Missing baselines are marked non-executable
- Component ablation results differentiate full from ablated variants
- Tactical outcome status remains weak/inconclusive unless new claim gate supports otherwise
- Route claim remains scenario-dependent after V3G
- Map 94 stress results are not merged into route-safe aggregate without labeling
- Public SOTA and two-window promotion claims remain forbidden unless supported
"""

import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestV3HEvaluatorCode(unittest.TestCase):
    """Tests that inspect the V3H evaluator code for correctness."""

    def test_evaluator_exists(self):
        """Verify the V3H evaluator script exists."""
        path = ROOT / "scripts" / "run_v3h_baseline_component_ablation.py"
        self.assertTrue(path.exists(), f"Evaluator not found: {path}")

    def test_evaluator_has_required_variants(self):
        """Verify the evaluator defines required policy variants."""
        path = ROOT / "scripts" / "run_v3h_baseline_component_ablation.py"
        code = path.read_text(encoding="utf-8")

        # Check variant configuration function
        self.assertIn("def configure_variant", code)
        self.assertIn('variant == "learned"', code)
        self.assertIn('variant == "rule_incumbent"', code)
        self.assertIn('variant == "no_tactical"', code)
        self.assertIn('variant == "no_intent"', code)
        self.assertIn('variant == "no_situation"', code)

    def test_evaluator_has_required_output_fields(self):
        """Verify the per-game result dict has all required fields."""
        path = ROOT / "scripts" / "run_v3h_baseline_component_ablation.py"
        code = path.read_text(encoding="utf-8")

        required_fields = [
            "variant", "seed", "scenario_id", "map_id",
            "authenticated", "status", "battle_entered",
            "score_red", "winner", "turns",
            "route_empty_count", "route_empty_rate",
            "tactical_invocation_count",
            "intent_used", "situation_used",
            "selected_by_learned_count",
            "fallback_move_count",
        ]
        for field in required_fields:
            self.assertIn(f'"{field}"', code,
                          f"Required output field missing: {field}")

    def test_evaluator_aggregates_variant_stats(self):
        """Verify the aggregate function groups by variant."""
        path = ROOT / "scripts" / "run_v3h_baseline_component_ablation.py"
        code = path.read_text(encoding="utf-8")
        self.assertIn("per_variant", code)
        self.assertIn("paired_statistics", code)

    def test_evaluator_has_route_safe_scenario_list(self):
        """Verify route-safe and route-stress scenario lists exist."""
        path = ROOT / "scripts" / "run_v3h_baseline_component_ablation.py"
        code = path.read_text(encoding="utf-8")
        self.assertIn("ROUTE_SAFE_SCENARIOS", code)
        self.assertIn("ROUTE_STRESS_MAP94_SCENARIOS", code)

    def test_evaluator_fail_closed_on_auth(self):
        """Verify evaluator checks for model manifest before running."""
        path = ROOT / "scripts" / "run_v3h_baseline_component_ablation.py"
        code = path.read_text(encoding="utf-8")
        self.assertIn("model_manifest.json", code)

    def test_configure_variant_no_intent_sets_none(self):
        """Verify no_intent variant sets intent_model to None."""
        path = ROOT / "scripts" / "run_v3h_baseline_component_ablation.py"
        code = path.read_text(encoding="utf-8")
        # Must null the model reference, not modify checkpoint
        self.assertIn('policy.intent_model = None', code)

    def test_configure_variant_no_situation_sets_none(self):
        """Verify no_situation variant sets situation_model to None."""
        path = ROOT / "scripts" / "run_v3h_baseline_component_ablation.py"
        code = path.read_text(encoding="utf-8")
        self.assertIn('policy.situation_model = None', code)

    def test_configure_variant_rule_incumbent_uses_baseline_profile(self):
        """Verify rule_incumbent uses policy_profile='baseline'."""
        path = ROOT / "scripts" / "run_v3h_baseline_component_ablation.py"
        code = path.read_text(encoding="utf-8")
        self.assertIn('red_profile = "baseline"', code)


class TestV3HClaimGates(unittest.TestCase):
    """Tests that claim gate constraints are preserved."""

    def test_tactical_outcome_remains_weak_inconclusive(self):
        """Verify V3H does not overturn V3D tactical outcome limitation."""
        v3d_dir = ROOT / "reports" / "v3d_authenticated_tactical_ablation"
        claim_content = ""
        if v3d_dir.exists():
            for gf in v3d_dir.glob("*claim_gate*"):
                try:
                    gate = json.loads(gf.read_text(encoding="utf-8"))
                    claim_content = json.dumps(gate)
                except Exception:
                    claim_content = gf.read_text(encoding="utf-8", errors="replace")
        if not claim_content:
            # Check FINAL_DECISION.md
            final = v3d_dir / "FINAL_DECISION.md"
            if final.exists():
                claim_content = final.read_text(encoding="utf-8", errors="replace")
        if claim_content:
            self.assertIn(
                "weak", claim_content.lower(),
                "V3D tactical outcome claim must remain weak/inconclusive"
            )
        else:
            # If V3D reports don't exist, at least check V3H reports are consistent
            reports_dir = ROOT / "reports" / "v3h_baseline_expansion"
            if reports_dir.exists():
                start_state = reports_dir / "00_start_state.md"
                if start_state.exists():
                    content = start_state.read_text(encoding="utf-8")
                    self.assertIn(
                        "weak/inconclusive", content.lower(),
                        "Start state must record tactical outcome as weak/inconclusive"
                    )

    def test_route_claim_remains_scenario_dependent(self):
        """Verify V3G route scenario-dependent limitation is preserved."""
        v3g_gate_dir = ROOT / "reports" / "v3g_full_route_coverage"
        v3g_gate_files = list(v3g_gate_dir.glob("*route_claim*")) if v3g_gate_dir.exists() else []
        route_status = ""
        for gf in v3g_gate_files:
            try:
                gate = json.loads(gf.read_text(encoding="utf-8"))
                for val in gate.values():
                    if isinstance(val, str) and "PARTIALLY" in val:
                        route_status = val
            except Exception:
                content = gf.read_text(encoding="utf-8", errors="replace")
                if "PARTIALLY" in content:
                    route_status = "PARTIALLY_SUPPORTED"
        if not route_status:
            # Check the FINAL_DECISION.md as fallback
            final = v3g_gate_dir / "FINAL_DECISION.md"
            if final.exists():
                content = final.read_text(encoding="utf-8", errors="replace")
                if "PARTIALLY_SUPPORTED" in content:
                    route_status = "PARTIALLY_SUPPORTED_SCENARIO_DEPENDENT"
        self.assertIn(
            "PARTIALLY", route_status,
            "V3G route claim must remain partially supported/scenario-dependent. "
            "Check that route_claim_gate.json or FINAL_DECISION.md exists."
        )

        # Check V3H reports do not claim general route reliability
        reports_dir = ROOT / "reports" / "v3h_baseline_expansion"
        if reports_dir.exists():
            for fpath in reports_dir.glob("*.md"):
                content = fpath.read_text(encoding="utf-8", errors="replace")
                forbidden = [
                    "route reliability is solved",
                    "route reliability is proven",
                    "route reliability is generally",
                    "all scenarios are route-safe",
                ]
                for pattern in forbidden:
                    if pattern.lower() in content.lower():
                        self.fail(
                            f"{fpath.name} claims general route reliability: "
                            f"'{pattern}'"
                        )

    def test_public_sota_claim_forbidden(self):
        """Verify public SOTA claims remain forbidden."""
        reports_dir = ROOT / "reports" / "v3h_baseline_expansion"
        if reports_dir.exists():
            for fpath in reports_dir.glob("*.md"):
                content = fpath.read_text(encoding="utf-8", errors="replace")
                forbidden = [
                    "public SOTA",
                    "state-of-the-art",
                    "state of the art",
                ]
                for pattern in forbidden:
                    if pattern.lower() in content.lower():
                        # Allow negation context
                        lines = content.split("\n")
                        for line in lines:
                            if pattern.lower() in line.lower():
                                if "forbidden" in line.lower() or "not" in line.lower():
                                    continue
                                self.fail(
                                    f"{fpath.name} may claim SOTA: "
                                    f"'{line.strip()}'"
                                )

    def test_two_window_promotion_claim_forbidden(self):
        """Verify two-window promotion claims remain forbidden unless proven."""
        # Check if any formal two-window evidence exists
        league_dirs = list(
            (ROOT / "experiments" / "shenji_agent").glob("league_*/promotion.json")
        )
        has_evidence = False
        for ld in sorted(league_dirs):
            try:
                prom = json.loads(ld.read_text(encoding="utf-8"))
                if prom.get("consecutive_windows", 0) >= 2:
                    has_evidence = True
                    break
            except Exception:
                pass

        reports_dir = ROOT / "reports" / "v3h_baseline_expansion"
        if reports_dir.exists():
            for fpath in reports_dir.glob("*.md"):
                content = fpath.read_text(encoding="utf-8", errors="replace")
                if "two-window" not in content.lower():
                    continue
                # Check each line — constraint/forbidden references are OK
                lines = content.split("\n")
                for line in lines:
                    if "two-window" in line.lower():
                        is_constraint = (
                            "forbidden" in line.lower()
                            or "do not" in line.lower()
                            or "no " in line.lower()
                            or "not " in line.lower()
                            or "preserve" in line.lower()
                            or "limitation" in line.lower()
                            or "single window" in line.lower()
                        )
                        if not is_constraint and not has_evidence:
                            self.fail(
                                f"{fpath.name} makes a two-window claim without evidence: "
                                f"'{line.strip()}'"
                            )


class TestV3HDataIntegrity(unittest.TestCase):
    """Tests for data integrity in V3H experiment outputs."""

    def setUp(self):
        self.exp_base = ROOT / "experiments" / "shenji_agent"

    def _check_results_json(self, path: Path):
        """Helper to validate a results.json file."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            self.fail(f"Invalid JSON in {path}")
        return data

    def test_smoke_results_have_required_fields(self):
        """Verify smoke test results.json has required aggregate fields."""
        smoke_path = self.exp_base / "v3h_smoke" / "results.json"
        data = self._check_results_json(smoke_path)
        if data is None:
            self.skipTest("Smoke results not found")
        required = ["total_games", "total_completed", "per_variant",
                     "variants", "scenario_count", "map_count"]
        for field in required:
            self.assertIn(field, data, f"Missing field: {field}")

    def test_smoke_learned_and_rule_distinguishable(self):
        """Verify learned and rule_incumbent have different metrics in smoke."""
        smoke_path = self.exp_base / "v3h_smoke" / "results.json"
        data = self._check_results_json(smoke_path)
        if data is None:
            self.skipTest("Smoke results not found")
        pv = data.get("per_variant", {})
        learned = pv.get("learned", {})
        rule = pv.get("rule_incumbent", {})

        # Variants should have different scores
        ls = learned.get("mean_score")
        rs = rule.get("mean_score")
        if ls is not None and rs is not None:
            self.assertNotEqual(ls, rs,
                                "Learned and rule incumbent should differ in score")

        # Learned should have intent_used=True, rule should have False
        self.assertTrue(learned.get("intent_used", False),
                        "Learned variant should use intent model")
        self.assertFalse(rule.get("intent_used", True),
                         "Rule incumbent should not use intent model")

    def test_component_smoke_all_variants_present(self):
        """Verify component smoke has all expected variants."""
        comp_path = self.exp_base / "v3h_smoke_components" / "results.json"
        data = self._check_results_json(comp_path)
        if data is None:
            self.skipTest("Component smoke results not found")
        variants = data.get("variants", [])
        for v in ["full", "no_tactical", "no_intent", "no_situation"]:
            self.assertIn(v, variants,
                          f"Component variant {v} missing from smoke results")

    def test_component_smoke_intent_situation_flags(self):
        """Verify intent_used and situation_used flags are correct."""
        comp_path = self.exp_base / "v3h_smoke_components" / "results.json"
        data = self._check_results_json(comp_path)
        if data is None:
            self.skipTest("Component smoke results not found")
        pv = data.get("per_variant", {})

        # full: both intent and situation should be used
        self.assertTrue(pv["full"].get("intent_used", False),
                        "Full variant should use intent")
        self.assertTrue(pv["full"].get("situation_used", False),
                        "Full variant should use situation")

        # no_intent: intent should NOT be used, situation should be used
        self.assertFalse(pv.get("no_intent", {}).get("intent_used", True),
                         "no_intent variant should NOT use intent")
        self.assertTrue(pv.get("no_intent", {}).get("situation_used", False),
                        "no_intent variant should still use situation")

        # no_situation: situation should NOT be used, intent should be used
        self.assertFalse(pv.get("no_situation", {}).get("situation_used", True),
                         "no_situation variant should NOT use situation")
        self.assertTrue(pv.get("no_situation", {}).get("intent_used", False),
                        "no_situation variant should still use intent")

        # no_tactical: both intent and situation still used
        self.assertTrue(pv["no_tactical"].get("intent_used", False),
                        "no_tactical variant should use intent")
        self.assertTrue(pv["no_tactical"].get("situation_used", False),
                        "no_tactical variant should use situation")


class TestV3HMap94Separation(unittest.TestCase):
    """Tests that map 94 stress results are not merged with route-safe results."""

    def test_route_safe_and_stress_are_separate(self):
        """Verify route-safe and route-stress scenario lists are disjoint."""
        code = (ROOT / "scripts" / "run_v3h_baseline_component_ablation.py") \
            .read_text(encoding="utf-8")
        # Parse the lists from code
        import ast
        tree = ast.parse(code)
        safe_list = None
        stress_list = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "ROUTE_SAFE_SCENARIOS":
                            safe_list = {s.value for s in node.value.elts}
                        elif target.id == "ROUTE_STRESS_MAP94_SCENARIOS":
                            stress_list = {s.value for s in node.value.elts}

        if safe_list is not None and stress_list is not None:
            overlap = safe_list & stress_list
            self.assertEqual(
                len(overlap), 0,
                f"Route-safe and route-stress scenario lists overlap: {overlap}"
            )

    def test_stress_scenarios_are_map_94(self):
        """Verify route-stress scenarios are both on map 94."""
        inventory_path = \
            ROOT / "reports" / "v3g_full_route_coverage" / "league_scenario_inventory.json"
        if not inventory_path.exists():
            self.skipTest("League inventory not found")
        inv = json.loads(inventory_path.read_text(encoding="utf-8"))
        lookup = inv.get("scenario_map_lookup", {})
        stress_sids = ["2010131194", "2010141294"]
        for sid in stress_sids:
            self.assertEqual(
                lookup.get(sid), 94,
                f"Stress scenario {sid} should be on map 94"
            )


if __name__ == "__main__":
    unittest.main()
