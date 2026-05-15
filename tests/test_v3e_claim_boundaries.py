"""V3E claim-boundary consistency tests.

Verify that frozen claim boundaries are consistent with evidence and
that forbidden claims are not accidentally supported by the evidence package.
"""

import json
import os
import re
import unittest

REPORTS_DIR = "reports/v3e_claim_freeze"
EVIDENCE_DIR = "paper/v3_evidence_package"


def read_json(rel_path):
    with open(rel_path) as f:
        return json.load(f)


def read_md(rel_path):
    with open(rel_path) as f:
        return f.read()


class TestV3ETacticalClaimBoundary(unittest.TestCase):
    """Tactical outcome contribution must not be marked as supported."""

    def setUp(self):
        self.claim_gate = read_json(
            "reports/v3d_authenticated_tactical_ablation/claim_gate.json"
        )

    def test_tactical_claim_gate_is_not_supported(self):
        gate = self.claim_gate["claim_gate"]
        self.assertIn(
            gate,
            ["TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE",
             "TACTICAL_OUTCOME_CONTRIBUTION_INCONCLUSIVE"],
            f"Claim gate is {gate}, expected weak/inconclusive",
        )

    def test_tactical_outcome_not_marked_supported_in_matrix(self):
        matrix = read_json(f"{EVIDENCE_DIR}/results_tables_v3.json")
        gate = matrix["v3d_authenticated_ablation"]["claim_gate"]
        self.assertIn("WEAK", gate.upper())

    def test_tactical_freeze_file_consistent(self):
        freeze = read_json(f"{REPORTS_DIR}/tactical_claim_freeze.json")
        self.assertEqual(freeze["action_level"], "TACTICAL_ACTION_LEVEL_SUPPORTED")
        self.assertEqual(freeze["outcome_level"], "TACTICAL_OUTCOME_WEAK_OR_INCONCLUSIVE")
        self.assertIn("Tactical improves win rate", freeze["forbidden_claims"])
        self.assertIn("Tactical improves outcome performance", freeze["forbidden_claims"])


class TestV3EForbiddenClaims(unittest.TestCase):
    """Public SOTA and two-window success must remain forbidden."""

    def test_sota_forbidden_in_v3_readiness(self):
        readiness = read_md(f"{EVIDENCE_DIR}/V3_READINESS.md")
        self.assertIn("public SOTA", readiness)
        self.assertIn("forbidden", readiness.lower())

    def test_two_window_forbidden_in_v3_readiness(self):
        readiness = read_md(f"{EVIDENCE_DIR}/V3_READINESS.md")
        self.assertIn("two-window", readiness)
        self.assertIn("forbidden", readiness.lower())

    def test_top_tier_forbidden_in_v3_readiness(self):
        readiness = read_md(f"{EVIDENCE_DIR}/V3_READINESS.md")
        self.assertIn("top-tier", readiness)
        self.assertIn("forbidden", readiness.lower())

    def test_sota_forbidden_in_claim_matrix_json(self):
        matrix = read_json(f"{REPORTS_DIR}/claim_matrix_freeze.json")
        self.assertEqual(matrix["claims"]["public_sota"], "FORBIDDEN")

    def test_two_window_forbidden_in_claim_matrix_json(self):
        matrix = read_json(f"{REPORTS_DIR}/claim_matrix_freeze.json")
        self.assertEqual(matrix["claims"]["two_window_success"], "FORBIDDEN")


class TestV3EPromotionClaims(unittest.TestCase):
    """Two-window success must not be claimed without sidecar evidence."""

    def test_no_two_window_claim_in_readiness(self):
        readiness = read_md(f"{EVIDENCE_DIR}/V3_READINESS.md")
        two_window_claim = re.search(
            r"two-window\s+success.*(?:supported|proven|achieved)",
            readiness,
            re.IGNORECASE,
        )
        if two_window_claim:
            line = two_window_claim.group(0)
            self.assertIn("forbidden", line.lower(), f"Found two-window claim: {line}")

    def test_promotion_window_audit_no_two_window(self):
        audit = read_json(f"{REPORTS_DIR}/promotion_window_audit.json")
        self.assertIn("SINGLE_WINDOW", audit["classification"])

    def test_consecutive_windows_not_two(self):
        """Verify league_state shows consecutive_windows != 2."""
        import glob
        league_dirs = sorted(
            glob.glob("experiments/shenji_agent/league_2026*/league_state.json")
        )
        self.assertGreater(len(league_dirs), 0, "No league_state.json found")
        latest = league_dirs[-1]
        state = read_json(latest)
        cw = state.get("consecutive_windows", -1)
        self.assertNotEqual(
            cw, 2,
            f"Latest league has consecutive_windows=2 — two-window success may be proven"
        )


class TestV3ERouteClaims(unittest.TestCase):
    """Route reliability must not be claimed as solved."""

    def test_route_not_solved_in_readiness(self):
        readiness = read_md(f"{EVIDENCE_DIR}/V3_READINESS.md")
        route_solved = re.search(
            r"route\s+(?:reliability|empty|issue)\s+(?:is\s+)?(?:fixed|solved|resolved|repaired)",
            readiness,
            re.IGNORECASE,
        )
        if route_solved:
            line = route_solved.group(0)
            self.assertIn("unproven", line.lower(), f"Found solved claim: {line}")

    def test_route_freeze_not_supported(self):
        freeze = read_json(f"{REPORTS_DIR}/route_reliability_audit.json")
        self.assertNotEqual(
            freeze["classification"],
            "ROUTE_RELIABILITY_SUPPORTED",
        )

    def test_route_forbidden_claims_listed(self):
        freeze = read_json(f"{REPORTS_DIR}/route_reliability_audit.json")
        for forbidden in freeze["forbidden_claims"]:
            self.assertIsInstance(forbidden, str)
            self.assertGreater(len(forbidden), 5)

    def test_route_claim_matrix_not_solved(self):
        matrix = read_json(f"{REPORTS_DIR}/claim_matrix_freeze.json")
        self.assertEqual(matrix["claims"]["route_empty_issue_solved"], "UNSUPPORTED")


class TestV3EConsistency(unittest.TestCase):
    """Cross-file consistency checks."""

    def test_readiness_and_claim_matrix_tactical_consistent(self):
        """Both files must agree tactical is not an outcome contributor."""
        readiness = read_md(f"{EVIDENCE_DIR}/V3_READINESS.md")
        matrix = read_md(f"{EVIDENCE_DIR}/claim_to_evidence_matrix_v3.md")

        readiness_weak = "weak/inconclusive" in readiness.lower()
        matrix_weak = "weak/inconclusive" in matrix.lower()

        self.assertTrue(
            readiness_weak or matrix_weak,
            "Neither V3_READINESS nor claim matrix describes tactical as weak/inconclusive",
        )

    def test_source_evidence_references_v3d(self):
        source = read_md("paper/source_evidence.md")
        self.assertIn("V3D", source)
        self.assertIn("v3d_authenticated_tactical_ablation", source)

    def test_source_evidence_references_v3e(self):
        source = read_md("paper/source_evidence.md")
        self.assertIn("V3E", source)

    def test_results_tables_claim_gate_consistent(self):
        tables = read_json(f"{EVIDENCE_DIR}/results_tables_v3.json")
        gate = tables["v3d_authenticated_ablation"]["claim_gate"]
        claim_gate = read_json(
            "reports/v3d_authenticated_tactical_ablation/claim_gate.json"
        )
        self.assertEqual(gate, claim_gate["claim_gate"])


class TestV3EBaseLineBoundaries(unittest.TestCase):
    """Baseline coverage must not be overstated."""

    def test_baseline_not_strong_in_matrix(self):
        matrix = read_json(f"{REPORTS_DIR}/claim_matrix_freeze.json")
        self.assertEqual(matrix["claims"]["baseline_coverage_strong"], "UNSUPPORTED")

    def test_baseline_coverage_not_sufficient_for_strong_paper(self):
        audit = read_json(f"{REPORTS_DIR}/baseline_ablation_audit.json")
        self.assertIn(
            "INSUFFICIENT",
            audit["classification"],
        )


if __name__ == "__main__":
    unittest.main()
