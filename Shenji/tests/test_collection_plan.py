from __future__ import annotations

import json
import unittest
from pathlib import Path

from shenji_dataset.ontology import load_ontology


ROOT = Path(__file__).resolve().parents[1]


class CollectionPlanTests(unittest.TestCase):
    def test_every_intent_has_a_collection_target_and_multiple_controllers(self) -> None:
        plan = json.loads(
            (ROOT / "configs" / "collection_plan_v1.json").read_text(encoding="utf-8")
        )
        ontology_codes = {item["code"] for item in load_ontology()["intents"]}

        self.assertEqual(set(plan["intent_targets"]), ontology_codes)
        for target in plan["intent_targets"].values():
            self.assertGreaterEqual(target["pilot_min_segments"], 30)
            self.assertGreaterEqual(target["v1_min_segments"], 300)
            self.assertGreaterEqual(target["min_controller_variants"], 3)

    def test_collection_plan_requires_failure_switch_and_ood_examples(self) -> None:
        plan = json.loads(
            (ROOT / "configs" / "collection_plan_v1.json").read_text(encoding="utf-8")
        )

        self.assertGreaterEqual(plan["quality_targets"]["failure_or_switch_min_share"], 0.20)
        self.assertGreaterEqual(plan["quality_targets"]["held_out_map_families"], 2)
        self.assertGreaterEqual(plan["quality_targets"]["seeds_per_scenario_variant"], 5)


if __name__ == "__main__":
    unittest.main()
