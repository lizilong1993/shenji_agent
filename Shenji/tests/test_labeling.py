from __future__ import annotations

import unittest

from shenji_dataset.labeling import derive_grade_a_intent_plan, derive_grade_b_intent_plan, derive_grade_c_intent_plan
from shenji_dataset.ontology import load_ontology
from shenji_dataset.validation import validate_intent_segment


def make_raw_episode() -> dict:
    communications = [
        {
            "type": 200,
            "info": {
                "company_id": 1,
                "mission_type": 0,
                "target_pos": 700,
                "start_time": 1,
                "end_time": 20,
            },
        },
        {
            "type": 200,
            "info": {
                "company_id": 2,
                "mission_type": 1,
                "target_pos": 701,
                "start_time": 2,
                "end_time": 30,
            },
        },
        {
            "type": 200,
            "info": {
                "company_id": 3,
                "mission_type": 2,
                "target_pos": 702,
                "start_time": 3,
                "end_time": 40,
            },
        },
    ]
    accepted_actions = [
        {"cur_step": 5, "type": 5, "obj_id": 101, "target_pos": 700},
        {
            "cur_step": 6,
            "type": 9,
            "obj_id": 102,
            "target_obj_id": 201,
            "guided_obj_id": 103,
        },
        {"cur_step": 7, "type": 8, "obj_id": 104, "jm_pos": 705},
        {"cur_step": 8, "type": 15, "obj_id": 105, "target_obj_id": 106},
        {"cur_step": 9, "type": 14, "obj_id": 107},
        {"cur_step": 10, "type": 18, "obj_id": 108, "target_obj_id": 9001},
        {"cur_step": 11, "type": 6, "obj_id": 109, "target_state": 4},
        {"cur_step": 12, "type": 20, "obj_id": 110},
        {"cur_step": 13, "type": 2, "obj_id": 111, "target_obj_id": 202},
        {
            "cur_step": 14,
            "type": 5,
            "obj_id": 112,
            "target_pos": 706,
            "error": "action rejected",
        },
    ]
    return {
        "metadata": {"episode_id": "episode_grade_a"},
        "observations": {
            "truth": [
                {
                    "time": {"cur_step": 0},
                    "operators": [],
                    "communication": communications,
                    "actions": [],
                },
                {
                    "time": {"cur_step": 1},
                    "operators": [],
                    "communication": communications,
                    "actions": accepted_actions,
                },
            ],
            "red": [],
            "blue": [],
        },
        "actions": [
            {"cur_step": 99, "type": 5, "obj_id": 999, "target_pos": 999, "error": "rejected in test"}
        ],
    }


class GradeALabelingTests(unittest.TestCase):
    def test_derives_all_supported_grade_a_intents(self) -> None:
        plan = derive_grade_a_intent_plan(make_raw_episode())
        codes = [segment["primary_intent"] for segment in plan["segments"]]

        self.assertEqual(codes.count("ASSAULT"), 1)
        self.assertEqual(codes.count("HOLD"), 1)
        self.assertEqual(codes.count("RECON"), 1)
        self.assertEqual(codes.count("SEIZE"), 1)
        self.assertEqual(codes.count("TARGET_GUIDANCE"), 1)
        self.assertEqual(codes.count("FIRE_SUPPORT"), 2)
        self.assertEqual(codes.count("ASSEMBLE"), 1)
        self.assertEqual(codes.count("DISPERSE"), 1)
        self.assertEqual(codes.count("FORTIFY"), 2)

    def test_deduplicates_repeated_commands_and_ignores_rejected_or_submitted_actions(self) -> None:
        plan = derive_grade_a_intent_plan(make_raw_episode())
        segments = plan["segments"]

        self.assertEqual(len(segments), 11)
        self.assertFalse(any(segment["subject_id"] == "112" for segment in segments))
        self.assertFalse(any(segment["subject_id"] == "999" for segment in segments))

    def test_never_invents_grade_b_grade_c_or_feint(self) -> None:
        plan = derive_grade_a_intent_plan(make_raw_episode())
        prohibited = {
            "BREAKTHROUGH",
            "FLANK",
            "ENCIRCLEMENT",
            "PURSUIT",
            "AMBUSH",
            "INTERDICT",
            "SURVEILLANCE",
            "REINFORCE",
            "ESCORT",
            "REDEPLOY",
            "RETREAT",
            "EVADE",
            "FEINT",
        }

        self.assertTrue(prohibited.isdisjoint(segment["primary_intent"] for segment in plan["segments"]))
        self.assertTrue(all(segment["evidence_grade"] == "A" for segment in plan["segments"]))

    def test_every_derived_segment_passes_the_shared_validator(self) -> None:
        ontology = load_ontology()
        plan = derive_grade_a_intent_plan(make_raw_episode())

        for segment in plan["segments"]:
            self.assertEqual(validate_intent_segment(segment, ontology), [])


if __name__ == "__main__":
    unittest.main()
