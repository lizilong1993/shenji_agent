"""P0 acceptance tests for engine/rolecards.

Runs with the standard library only:

    python3 -m unittest tests.test_rolecards_p0 -v

P0 scope: the registry and the legacy adapter exist and validate, every legacy
target maps to a subclass (or ``<class>.unknown``), and adding the module does
not change generated frames.  Behaviour paths are intentionally untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "situation3d"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from engine import cases, scenario  # noqa: E402
from engine.rolecards import card_from_legacy, load_registry  # noqa: E402
from engine.rolecards import validate as role_validate  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "data" / "rolecards_frame_hashes.json")
                    .read_text(encoding="utf-8"))


def _frame_hash(frames) -> str:
    blob = json.dumps(frames, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _walk_keys(doc, path=""):
    if isinstance(doc, dict):
        for key, value in doc.items():
            here = path + "." + key if path else key
            yield here, key, value
            if isinstance(value, dict):
                yield from _walk_keys(value, here)


class RoleCardP0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry()
        cls.scenario_frames = {}
        for case in cases.CASES:
            if case.generator != "scenario":
                continue
            targets = scenario.targets_for(case)
            cls.scenario_frames[case.case_id] = (
                targets, scenario.build_frames(targets, theatre=case.theatre))
        from engine.campaigns import taiwan_dense
        cls.dense_doc, _meta = taiwan_dense.build_case()

    # -- registry ----------------------------------------------------------
    def test_registry_valid(self):
        self.assertEqual(self.registry.validate(), [])
        self.assertEqual(len(self.registry.classes), 8)
        self.assertEqual(len(self.registry.subclasses), 61)
        self.assertEqual(len(self.registry.equipment["weapons"]), 40)
        self.assertEqual(len(self.registry.equipment["sensors"]), 32)
        self.assertEqual(len(self.registry.equipment["comms"]), 13)

    def test_class_cards_do_not_carry_subclass_values(self):
        forbidden = {
            "max_speed_ms", "cruise_speed_ms", "accel_ms2", "decel_ms2",
            "armor_class", "max_blood", "signature", "sensors", "weapons",
            "terrain", "sea", "air",
        }
        for cls in self.registry.classes.values():
            for path, key, _value in _walk_keys(cls.get("shared") or {}):
                self.assertNotIn(key, forbidden,
                                 "%s carries %s" % (cls["class_id"], path))
            self.assertNotIn("max_speed_ms", json.dumps(cls.get("shared") or {}))

    def test_registry_rejects_class_numeric_value(self):
        classes = copy.deepcopy(self.registry.classes)
        classes["land.vehicle"].setdefault("shared", {})["motion"] = {"max_speed_ms": 1.0}
        errors = role_validate.validate_registry(
            classes, self.registry.subclasses, self.registry.equipment, self.registry.types)
        self.assertTrue(any("max_speed_ms" in e for e in errors), errors)

    def test_registry_rejects_unknown_equipment_ref(self):
        subclasses = copy.deepcopy(self.registry.subclasses)
        subclasses["land.vehicle.mbt"]["values"]["weapons"][0]["ref"] = "weapon.no.such"
        errors = role_validate.validate_registry(
            self.registry.classes, subclasses, self.registry.equipment, self.registry.types)
        self.assertTrue(any("unknown weapon ref" in e for e in errors), errors)

    def test_every_class_has_conservative_unknown(self):
        for class_id, cls in self.registry.classes.items():
            unknown = self.registry.subclasses[cls["unknown_subclass"]]
            self.assertEqual(unknown["class_id"], class_id)
            self.assertEqual(unknown["confidence"], "unknown")
            self.assertFalse((unknown.get("unknown_policy") or {}).get("engage", True))

    def test_type_resolution(self):
        self.assertEqual(self.registry.resolve_type("land.vehicle.mbt"), "land.vehicle.mbt")
        self.assertEqual(self.registry.resolve_type("icon.armor"), "land.vehicle.mbt")
        self.assertEqual(self.registry.resolve_type("armor"), "land.vehicle.mbt")
        self.assertEqual(self.registry.resolve_type("2.0"), "land.vehicle.mbt")
        self.assertEqual(self.registry.resolve_type("icon.generic_land"),
                         "land.vehicle.unknown")
        self.assertIsNone(self.registry.resolve_type("no.such.type"))

    def test_effective_card_merge(self):
        card = self.registry.effective("land.vehicle.mbt")
        self.assertAlmostEqual(card["motion"]["max_speed_ms"], 22.22, places=2)
        self.assertAlmostEqual(card["motion"]["cruise_speed_ms"], 13.89, places=2)
        self.assertEqual(card["protection"]["armor_class"], 4)
        self.assertEqual(card["behavior"]["default_roe"], "return_fire")
        self.assertEqual(card["identity"]["primary_domain"], "land")
        overridden = self.registry.effective(
            "land.vehicle.mbt", {"motion": {"cruise_speed_ms": 7.0}})
        self.assertEqual(overridden["motion"]["cruise_speed_ms"], 7.0)
        self.assertAlmostEqual(overridden["motion"]["max_speed_ms"], 22.22, places=2)

    # -- legacy adapter ----------------------------------------------------
    def test_legacy_scenario_targets_map_and_preserve_speed(self):
        seen = set()
        by_id = {}
        for case_id, (targets, _frames) in self.scenario_frames.items():
            for target in targets:
                snapshot = card_from_legacy(target, self.registry)
                by_id[target.target_id] = snapshot
                seen.add(snapshot["subclass_id"])
                self.assertIn(snapshot["class_id"], self.registry.classes)
                self.assertIn(snapshot["subclass_id"], self.registry.subclasses)
                self.assertEqual(snapshot["confidence"], "legacy")
                if snapshot["effective"]["motion"].get("immobile"):
                    self.assertLess(target.speed_kn, 0.1)
                    self.assertEqual(snapshot["projection"]["speed_kn"], 0.0)
                    self.assertEqual(
                        snapshot["effective"]["motion"]["cruise_speed_ms"], 0.0)
                else:
                    self.assertAlmostEqual(snapshot["projection"]["speed_kn"],
                                           target.speed_kn, delta=1e-9)
                    self.assertAlmostEqual(
                        snapshot["effective"]["motion"]["cruise_speed_ms"],
                        target.speed_kn * (1852.0 / 3600.0), delta=1e-9)
                self.assertEqual(snapshot["legacy"]["icon"], target.icon)
                if target.domain == "land":
                    self.assertEqual(snapshot["projection"]["road_bound"],
                                     bool(target.road_bound))
        # Refined mappings for the concrete order of battle.
        expected = {
            "R01": "sea.surface.destroyer",
            "R04": "land.installation.coastal_battery",
            "R17": "land.vehicle.coastal_launcher",
            "R09": "land.vehicle.amphibious",
            "R15": "land.vehicle.mbt",
            "R16": "land.vehicle.spg",
            "R08": "sea.surface.landing_ship",
            "H01": "land.vehicle.light_tank",
            "H02": "land.vehicle.truck",
            "H08": "land.vehicle.towed_artillery",
            "H10": "land.vehicle.mlrs",
        }
        for target_id, subclass_id in expected.items():
            self.assertEqual(by_id[target_id]["subclass_id"], subclass_id, target_id)

    def test_dense_targets_map_and_preserve_speed(self):
        targets = self.dense_doc.get("targets") or []
        self.assertEqual(len(targets), 80)
        by_id = {}
        for target in targets:
            snapshot = card_from_legacy(target, self.registry)
            by_id[target["id"]] = snapshot
            self.assertIn(snapshot["class_id"], self.registry.classes)
            self.assertAlmostEqual(snapshot["projection"]["speed_kn"],
                                   target["speed_kn"], delta=1e-9)
            if target.get("domain") == "land":
                self.assertEqual(snapshot["projection"]["road_bound"],
                                 bool(target.get("road_bound", True)))
        self.assertEqual(by_id["TRS001"]["subclass_id"], "sea.surface.landing_ship")
        self.assertEqual(by_id["TRV001"]["subclass_id"], "land.vehicle.amphibious")
        self.assertEqual(by_id["TRV021"]["subclass_id"], "land.vehicle.truck")
        self.assertEqual(by_id["TRV033"]["subclass_id"], "land.vehicle.engineer")

    def test_legacy_projection_for_mbt(self):
        from engine.scenario import Target
        target = Target(
            target_id="T1", name="坦克", platform_class="坦克", domain="land",
            affiliation="red", waypoints=[], speed_kn=18.0, icon="armor",
            sensor_range_km=5.0, engage_km=2.5, influence_km=10.0, road_bound=False,
        )
        snapshot = card_from_legacy(target, self.registry)
        self.assertEqual(snapshot["subclass_id"], "land.vehicle.mbt")
        self.assertAlmostEqual(snapshot["projection"]["speed_kn"], 18.0, delta=1e-9)
        self.assertAlmostEqual(snapshot["projection"]["sensor_range_km"], 5.0)
        self.assertAlmostEqual(snapshot["projection"]["engage_km"], 2.5)
        self.assertAlmostEqual(snapshot["projection"]["influence_km"], 10.0)
        self.assertFalse(snapshot["projection"]["road_bound"])

    # -- P1 preview: card-driven movement / dataset contract ---------------
    def test_build_frames_uses_attached_card_speed(self):
        """With a card attached, zeroing legacy speed must not move anything.

        Immobile installations are excluded from this comparison; their legacy
        micro-speed comes from the known geo.kmh_to_knots bug and P1 will not
        make positional batteries march.
        """
        for case_id, (targets, _frames) in self.scenario_frames.items():
            fresh = copy.deepcopy(scenario.targets_for(cases.get_case(case_id)))
            scenario.attach_role_cards(fresh)
            moving = [t for t in fresh
                      if not (t.role_card["effective"]["motion"].get("immobile"))]
            reference = scenario.build_frames(moving, theatre=cases.get_case(case_id).theatre)
            for target in moving:
                target.speed_kn = 0.0
            driven = scenario.build_frames(moving, theatre=cases.get_case(case_id).theatre)
            self.assertEqual(_frame_hash(driven), _frame_hash(reference), case_id)

    def test_frames_unchanged_with_cards_attached(self):
        """Full P1 path: attach cards, then rebuild frames from scratch."""
        for case_id in ("taiwan_landing", "india_china_west"):
            case = cases.get_case(case_id)
            targets = copy.deepcopy(scenario.targets_for(case))
            scenario.attach_role_cards(targets)
            frames = scenario.build_frames(targets, theatre=case.theatre)
            self.assertEqual(_frame_hash(frames), GOLDEN[case_id]["sha256"], case_id)

    def test_dataset_entries_include_role_card(self):
        for case_id in ("taiwan_landing", "india_china_west"):
            targets = copy.deepcopy(scenario.targets_for(cases.get_case(case_id)))
            scenario.attach_role_cards(targets)
            for target in targets:
                entry = target.to_dataset_entry()
                self.assertIn("role_card", entry)
                card = entry["role_card"]
                self.assertEqual(card["class_id"], self.registry.class_of(
                    card["subclass_id"])["class_id"])
                self.assertIn("effective", card)
                self.assertEqual(card["confidence"], "legacy")

    def test_dense_dataset_entries_include_role_card(self):
        targets = self.dense_doc.get("targets") or []
        self.assertTrue(targets)
        for entry in targets:
            self.assertIn("role_card", entry)
            self.assertIn("effective", entry["role_card"])
        by_id = {entry["id"]: entry for entry in targets}
        self.assertEqual(by_id["TRS001"]["role_card"]["subclass_id"],
                         "sea.surface.landing_ship")
        self.assertEqual(by_id["TRV001"]["role_card"]["subclass_id"],
                         "land.vehicle.amphibious")

    # -- frame parity ------------------------------------------------------
    def test_frames_unchanged_when_speed_read_from_card(self):
        """Rebuild frames using the card cruise speed (P1 preview).

        Immobile installations are skipped: their legacy micro-speed comes from
        the known ``geo.kmh_to_knots`` unit bug and P1 must decide whether a
        positional battery follows it at all.
        """
        import dataclasses
        knot_to_ms = 1852.0 / 3600.0
        for case_id, (targets, _frames) in self.scenario_frames.items():
            clones = []
            for target in targets:
                snapshot = card_from_legacy(target, self.registry)
                motion = snapshot["effective"]["motion"]
                if motion.get("immobile"):
                    # Keep the legacy micro-speed for the hash comparison; P1
                    # decides how positional batteries should behave.
                    speed_kn = target.speed_kn
                else:
                    speed_kn = motion["cruise_speed_ms"] / knot_to_ms
                clones.append(dataclasses.replace(target, speed_kn=speed_kn))
            frames = scenario.build_frames(clones, theatre=cases.get_case(case_id).theatre)
            self.assertEqual(_frame_hash(frames), GOLDEN[case_id]["sha256"], case_id)

    def test_validator_rejects_cross_field_violations(self):
        cases = []

        def mutate(fn):
            subs = copy.deepcopy(self.registry.subclasses)
            fn(subs)
            cases.append(subs)

        mutate(lambda s: s["land.vehicle.mbt"]["values"]["motion"].update(
            {"cruise_speed_ms": 999.0}))
        mutate(lambda s: s["land.vehicle.mbt"]["values"]["weapons"][0].update(
            {"ammo_default": 9999}))
        mutate(lambda s: s["land.vehicle.amphibious"]["values"]["motion"]["terrain"].pop(
            "swim_speed_ms", None))
        mutate(lambda s: s["sea.surface.destroyer"]["values"]["motion"]["sea"].pop(
            "draft_m", None))
        mutate(lambda s: s["air.aircraft.fighter"]["values"]["motion"]["air"].pop(
            "service_ceiling_m", None))
        mutate(lambda s: s["land.vehicle.unknown"].update(
            {"unknown_policy": {"engage": True}}))
        mutate(lambda s: s["land.vehicle.unknown"]["values"].update(
            {"weapons": [{"ref": "weapon.30mm.auto"}]}))

        for idx, subs in enumerate(cases):
            errors = role_validate.validate_registry(
                self.registry.classes, subs, self.registry.equipment, self.registry.types)
            self.assertTrue(errors, "mutation %d should fail validation" % idx)

    def test_validator_rejects_bad_alias(self):
        types = copy.deepcopy(self.registry.types)
        types["aliases"]["icon.broken"] = "no.such.subclass"
        errors = role_validate.validate_registry(
            self.registry.classes, self.registry.subclasses, self.registry.equipment, types)
        self.assertTrue(any("unknown subclass" in e for e in errors), errors)

    def test_frames_unchanged_scenario_cases(self):
        for case_id, (_targets, frames) in self.scenario_frames.items():
            self.assertEqual(len(frames), GOLDEN[case_id]["frames"])
            self.assertEqual(_frame_hash(frames), GOLDEN[case_id]["sha256"], case_id)

    def test_frames_unchanged_dense_case(self):
        case_id = "taiwan_dense_transfer_v1"
        frames = self.dense_doc["frames"]
        self.assertEqual(len(frames), GOLDEN[case_id]["frames"])
        self.assertEqual(len(self.dense_doc.get("targets") or []),
                         GOLDEN[case_id]["targets"])
        self.assertEqual(_frame_hash(frames), GOLDEN[case_id]["sha256"])


if __name__ == "__main__":
    unittest.main()
