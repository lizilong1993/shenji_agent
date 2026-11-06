"""ROE 手册的判定逻辑。

只测 ``sim/roe.py`` 本身：它是纯函数，不依赖世界状态，因此可以把每条
判据的边界都覆盖到。世界侧的接入用 ``test_roe_world.py``。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim.roe import POLICIES, Rules, describe, evaluate  # noqa: E402


def check(rules, **kwargs):
    base = dict(distance_m=1000.0, now_s=100.0, first_seen_s=90.0,
                damaged_at_s=-1.0, in_area=True, no_strike=False,
                manual_fire=False)
    base.update(kwargs)
    return evaluate(rules, **base)


class ParseTest(unittest.TestCase):
    def test_defaults_do_not_change_behaviour(self):
        rules = Rules.from_dict(None)
        self.assertEqual(rules.policy, "weapons_free")
        self.assertEqual(rules.confirm_s, 0.0)
        self.assertTrue(rules.respect_no_strike)
        allowed, reason = check(rules)
        self.assertTrue(allowed)
        self.assertEqual(reason, "weapons_free")

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            Rules.from_dict({"policy": "shoot_everything"})

    def test_empty_dict_uses_defaults(self):
        self.assertEqual(Rules.from_dict({}).policy, "weapons_free")

    def test_numbers_are_coerced_and_clamped(self):
        rules = Rules.from_dict({
            "policy": "weapons_tight", "area_radius_m": "-500",
            "confirm_s": "abc", "return_fire_s": "60"})
        self.assertEqual(rules.area_radius_m, 0.0)
        self.assertEqual(rules.confirm_s, 0.0)
        self.assertEqual(rules.return_fire_s, 60.0)

    def test_string_false_is_not_truthy(self):
        rules = Rules.from_dict({"respect_no_strike": "false"})
        self.assertFalse(rules.respect_no_strike)

    def test_round_trip(self):
        rules = Rules.from_dict({"policy": "hold_fire", "confirm_s": 30})
        again = Rules.from_dict(rules.as_dict())
        self.assertEqual(rules, again)


class PolicyTest(unittest.TestCase):
    def test_weapons_free_allows_anything(self):
        allowed, _ = check(Rules(policy="weapons_free"))
        self.assertTrue(allowed)

    def test_hold_fire_blocks_unprovoked(self):
        allowed, reason = check(Rules(policy="hold_fire"))
        self.assertFalse(allowed)
        self.assertIn("未遭攻击", reason)

    def test_hold_fire_allows_return_fire(self):
        allowed, reason = check(Rules(policy="hold_fire"),
                                damaged_at_s=90.0, now_s=120.0)
        self.assertTrue(allowed)
        self.assertIn("还击", reason)

    def test_return_fire_window_expires(self):
        allowed, _ = check(Rules(policy="hold_fire", return_fire_s=60.0),
                           damaged_at_s=10.0, now_s=200.0)
        self.assertFalse(allowed)

    def test_weapons_tight_blocks_outside_area(self):
        allowed, reason = check(Rules(policy="weapons_tight",
                                      area_radius_m=5000.0), in_area=False)
        self.assertFalse(allowed)
        self.assertIn("责任区", reason)

    def test_weapons_tight_allows_inside_area(self):
        allowed, _ = check(Rules(policy="weapons_tight",
                                 area_radius_m=5000.0), in_area=True)
        self.assertTrue(allowed)

    def test_weapons_tight_without_radius_degrades(self):
        # 半径 0 等于不限制，避免"选了 tight 但没填半径"变成永不开火。
        allowed, _ = check(Rules(policy="weapons_tight", area_radius_m=0.0),
                           in_area=False)
        self.assertTrue(allowed)


class ConfirmGateTest(unittest.TestCase):
    def test_below_threshold_blocks(self):
        allowed, reason = check(Rules(confirm_s=60.0), first_seen_s=90.0,
                                now_s=100.0)
        self.assertFalse(allowed)
        self.assertIn("确认门槛", reason)

    def test_at_threshold_allows(self):
        allowed, _ = check(Rules(confirm_s=10.0), first_seen_s=90.0, now_s=100.0)
        self.assertTrue(allowed)

    def test_missing_first_seen_blocks(self):
        allowed, reason = check(Rules(confirm_s=10.0), first_seen_s=None)
        self.assertFalse(allowed)
        self.assertIn("尚未被持续探测", reason)

    def test_no_gate_when_zero(self):
        allowed, _ = check(Rules(confirm_s=0.0), first_seen_s=None)
        self.assertTrue(allowed)


class NoStrikeTest(unittest.TestCase):
    def test_nsl_blocks_automatic_fire(self):
        allowed, reason = check(Rules(), no_strike=True)
        self.assertFalse(allowed)
        self.assertIn("禁止打击", reason)

    def test_nsl_blocks_manual_fire_too(self):
        # 法律层面的禁止连人工命令也不能越过。
        allowed, _ = check(Rules(), no_strike=True, manual_fire=True)
        self.assertFalse(allowed)

    def test_manual_fire_bypasses_policy(self):
        allowed, reason = check(Rules(policy="hold_fire"), manual_fire=True)
        self.assertTrue(allowed)
        self.assertEqual(reason, "人工开火命令")

    def test_respect_can_be_disabled(self):
        allowed, _ = check(Rules(respect_no_strike=False), no_strike=True)
        self.assertTrue(allowed)


class DescribeTest(unittest.TestCase):
    def test_free(self):
        self.assertIn("自由开火", describe(Rules()))

    def test_tight_with_radius(self):
        text = describe(Rules(policy="weapons_tight", area_radius_m=15000.0))
        self.assertIn("责任区", text)
        self.assertIn("15", text)

    def test_hold_fire(self):
        self.assertIn("还击", describe(Rules(policy="hold_fire")))

    def test_extras_appear(self):
        text = describe(Rules(confirm_s=30.0))
        self.assertIn("确认门槛", text)
        self.assertIn("禁止打击清单", text)


if __name__ == "__main__":
    unittest.main()
