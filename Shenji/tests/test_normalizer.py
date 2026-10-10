from __future__ import annotations

import unittest

from shenji_dataset.normalizer import CANONICAL_TABLES, normalize_episode
from shenji_dataset.ontology import load_ontology
from shenji_dataset.validation import validate_tables


def make_raw_episode() -> dict:
    red_1001_0 = {
        "obj_id": 1001,
        "color": 0,
        "type": 1,
        "sub_type": 0,
        "cur_hex": 501,
        "speed": 0.0,
        "blood": 10.0,
        "fatigue": 0.1,
        "suppression": 0.0,
        "move_state": 0,
    }
    blue_2001_0 = {
        "obj_id": 2001,
        "color": 1,
        "type": 1,
        "sub_type": 1,
        "cur_hex": 701,
        "speed": 0.0,
        "blood": 10.0,
    }
    red_1001_1 = {**red_1001_0, "cur_hex": 502, "speed": 3.5}
    blue_2001_1 = {**blue_2001_0, "cur_hex": 700, "speed": 2.0}
    return {
        "metadata": {
            "dataset": "Shenji",
            "schema_version": "shenji_schema_v1",
            "episode_id": "shenji_example_001",
            "scenario_id": "scenario_1231",
            "map_id": "map_123",
            "seed": 7,
            "simulation_backend": "MiaoSuan",
            "policy_red": "rule_v1",
            "policy_blue": "rule_v2",
        },
        "observations": {
            "truth": [
                {"time": {"cur_step": 0}, "operators": [red_1001_0, blue_2001_0]},
                {"time": {"cur_step": 1}, "operators": [red_1001_1, blue_2001_1]},
            ],
            "red": [
                {"time": {"cur_step": 0}, "operators": [red_1001_0]},
                {"time": {"cur_step": 1}, "operators": [red_1001_1, blue_2001_1]},
            ],
            "blue": [
                {"time": {"cur_step": 0}, "operators": [blue_2001_0]},
                {"time": {"cur_step": 1}, "operators": [blue_2001_1]},
            ],
        },
        "actions": [
            {
                "cur_step": 1,
                "actor": 1,
                "obj_id": 1001,
                "type": 1,
                "move_path": [502, 503],
            },
            {
                "cur_step": 1,
                "actor": 1,
                "obj_id": 1002,
                "type": 9,
                "target_obj_id": 2001,
                "guided_obj_id": 1003,
                "weapon_id": 88,
            },
        ],
    }


def make_intent_plan() -> dict:
    return {
        "segments": [
            {
                "segment_id": "commanded_001",
                "label_kind": "commanded",
                "subject_scope": "company",
                "subject_id": "red_company_01",
                "start_step": 0,
                "end_step": 1,
                "parent_intent": "AREA_CONTROL",
                "primary_intent": "SEIZE",
                "secondary_intents": ["FLANK", "FIRE_SUPPORT"],
                "target_type": "objective_point",
                "target_id": "city_07",
                "status": "completed",
                "outcome": "achieved",
                "evidence_source": "scenario_plan",
                "evidence_grade": "A",
                "evidence_refs": ["plan:red_company_01:1"],
                "confidence": 1.0,
            }
        ]
    }


class NormalizerTests(unittest.TestCase):
    def test_produces_eight_canonical_tables_and_categorical_ids(self) -> None:
        tables = normalize_episode(make_raw_episode(), make_intent_plan())

        self.assertEqual(tuple(tables), CANONICAL_TABLES)
        self.assertEqual(len(tables["episodes"]), 1)
        self.assertEqual(len(tables["entities"]), 2)
        self.assertEqual(len(tables["states"]), 4)
        self.assertEqual(len(tables["observations"]), 9)
        self.assertEqual(tables["states"][0]["entity_id"], "1001")
        self.assertEqual(tables["states"][0]["hex_id"], "501")
        self.assertEqual(tables["states"][0]["side"], "red")

    def test_maps_actions_events_and_guidance_relation(self) -> None:
        tables = normalize_episode(make_raw_episode(), make_intent_plan())

        self.assertEqual(tables["actions"][0]["action_name"], "MOVE")
        self.assertEqual(tables["actions"][0]["path"], ["502", "503"])
        self.assertEqual(tables["actions"][1]["action_name"], "GUIDED_FIRE")
        self.assertEqual(tables["events"][1]["event_type"], "GUIDED_FIRE")
        self.assertEqual(tables["relations"][0]["relation_type"], "GUIDES")
        self.assertEqual(tables["relations"][0]["subject_id"], "1002")
        self.assertEqual(tables["relations"][0]["object_id"], "1003")

    def test_preserves_commanded_intent_as_segment_not_frame_label(self) -> None:
        tables = normalize_episode(make_raw_episode(), make_intent_plan())
        segment = tables["intent_segments"][0]

        self.assertEqual(segment["episode_id"], "shenji_example_001")
        self.assertEqual(segment["primary_intent"], "SEIZE")
        self.assertEqual(segment["secondary_intents"], ["FLANK", "FIRE_SUPPORT"])
        self.assertNotIn("intent", tables["states"][0])

    def test_rejects_non_shenji_dataset_identity(self) -> None:
        raw = make_raw_episode()
        raw["metadata"]["dataset"] = "MiaoSuan"

        with self.assertRaisesRegex(ValueError, "dataset must be Shenji"):
            normalize_episode(raw, make_intent_plan())

    def test_output_passes_cross_table_validation(self) -> None:
        tables = normalize_episode(make_raw_episode(), make_intent_plan())

        self.assertEqual(validate_tables(tables, load_ontology()), [])


if __name__ == "__main__":
    unittest.main()
