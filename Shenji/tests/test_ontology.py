from __future__ import annotations

import unittest
from copy import deepcopy

from shenji_dataset.ontology import load_ontology
from shenji_dataset.validation import validate_intent_segment, validate_ontology


class OntologyTests(unittest.TestCase):
    def test_ontology_has_six_parents_and_twenty_two_intents(self) -> None:
        ontology = load_ontology()

        self.assertEqual(len(ontology["parent_categories"]), 6)
        self.assertEqual(len(ontology["intents"]), 22)
        self.assertEqual(validate_ontology(ontology), [])

    def test_feint_requires_explicit_plan_or_human_review(self) -> None:
        ontology = load_ontology()
        feint = next(item for item in ontology["intents"] if item["code"] == "FEINT")

        self.assertTrue(feint["requires_plan_tag"])
        self.assertEqual(feint["auto_label_grade"], "C")

    def test_ontology_rejects_duplicate_intent_code(self) -> None:
        ontology = deepcopy(load_ontology())
        ontology["intents"][1]["code"] = ontology["intents"][0]["code"]

        errors = validate_ontology(ontology)

        self.assertTrue(any("unique" in error for error in errors))

    def test_valid_multilabel_segment_passes(self) -> None:
        segment = {
            "segment_id": "segment_001",
            "label_kind": "commanded",
            "subject_scope": "company",
            "subject_id": "red_company_01",
            "start_step": 10,
            "end_step": 40,
            "parent_intent": "AREA_CONTROL",
            "primary_intent": "SEIZE",
            "secondary_intents": ["FLANK", "FIRE_SUPPORT"],
            "target_type": "objective_point",
            "target_id": "city_07",
            "status": "executing",
            "outcome": "unknown",
            "evidence_source": "scenario_plan",
            "evidence_grade": "A",
            "evidence_refs": ["plan:red_company_01:1"],
            "confidence": 1.0,
        }

        self.assertEqual(validate_intent_segment(segment, load_ontology()), [])

    def test_segment_rejects_wrong_parent_and_duplicate_primary(self) -> None:
        segment = {
            "segment_id": "segment_002",
            "label_kind": "realized",
            "subject_scope": "group",
            "subject_id": "blue_group_01",
            "start_step": 50,
            "end_step": 20,
            "parent_intent": "OFFENSIVE_ACTION",
            "primary_intent": "RETREAT",
            "secondary_intents": ["RETREAT"],
            "target_type": "own_survival",
            "target_id": None,
            "status": "completed",
            "outcome": "achieved",
            "evidence_source": "deterministic_rule",
            "evidence_grade": "B",
            "evidence_refs": [],
            "confidence": 1.2,
        }

        errors = validate_intent_segment(segment, load_ontology())

        self.assertTrue(any("start_step" in error for error in errors))
        self.assertTrue(any("parent" in error for error in errors))
        self.assertTrue(any("secondary" in error for error in errors))
        self.assertTrue(any("confidence" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
