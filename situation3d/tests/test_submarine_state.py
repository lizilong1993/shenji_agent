"""潜艇浮出 / 下潜状态机。

``Entity.depth`` 字段此前存在但从未被读写：所有潜艇在感知层一律按潜航
处理，等于永远隐身，也不能浮出。这里覆盖状态机本身与它在感知层的效果。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim import submarine  # noqa: E402


class _Entity(object):
    def __init__(self, domain="subsurface", depth="surface"):
        self.domain = domain
        self.depth = depth


class StateTest(unittest.TestCase):
    def test_default_is_submerged_for_submarines(self):
        # Entity.depth 的默认值是 "surface"（为水面舰艇设的），潜艇要显式
        # 归到 submerged —— 否则新潜艇一出生就能被雷达看见。
        entity = _Entity(depth="surface")
        self.assertEqual(submarine.state_of(entity), "submerged")

    def test_only_submarines_can_change_state(self):
        ship = _Entity(domain="surface")
        self.assertFalse(submarine.can_surface(ship))
        with self.assertRaises(ValueError):
            submarine.set_state(ship, "surfaced")

    def test_set_valid_states(self):
        entity = _Entity()
        for state in submarine.STATES:
            self.assertEqual(submarine.set_state(entity, state), state)
            self.assertEqual(submarine.state_of(entity), state)

    def test_invalid_state_rejected(self):
        entity = _Entity()
        with self.assertRaises(ValueError):
            submarine.set_state(entity, "flying")


class DetectionTest(unittest.TestCase):
    def test_submerged_is_invisible_to_radar(self):
        entity = _Entity()
        submarine.set_state(entity, "submerged")
        self.assertFalse(submarine.is_detectable_by(entity, "radar"))
        self.assertFalse(submarine.is_detectable_by(entity, "optics"))
        self.assertTrue(submarine.is_detectable_by(entity, "sonar"))

    def test_surfaced_is_visible_to_radar(self):
        entity = _Entity()
        submarine.set_state(entity, "surfaced")
        self.assertTrue(submarine.is_detectable_by(entity, "radar"))
        self.assertEqual(submarine.detection_factor(entity), 1.0)

    def test_periscope_shortens_effective_range(self):
        entity = _Entity()
        submarine.set_state(entity, "periscope")
        self.assertTrue(submarine.is_detectable_by(entity, "radar"))
        self.assertLess(submarine.detection_factor(entity), 1.0)
        self.assertGreater(submarine.detection_factor(entity), 0.0)

    def test_non_submarine_is_always_detectable(self):
        ship = _Entity(domain="surface")
        self.assertTrue(submarine.is_detectable_by(ship, "radar"))
        self.assertEqual(submarine.detection_factor(ship), 1.0)


class TorpedoTest(unittest.TestCase):
    def test_torpedo_requires_submerged(self):
        entity = _Entity()
        submarine.set_state(entity, "submerged")
        self.assertTrue(submarine.can_use_torpedo(entity))
        submarine.set_state(entity, "surfaced")
        self.assertFalse(submarine.can_use_torpedo(entity))
        submarine.set_state(entity, "periscope")
        self.assertFalse(submarine.can_use_torpedo(entity))

    def test_non_submarine_unaffected(self):
        self.assertTrue(submarine.can_use_torpedo(_Entity(domain="surface")))


class LabelTest(unittest.TestCase):
    def test_chinese_labels(self):
        self.assertEqual(submarine.label("submerged"), "潜航")
        self.assertEqual(submarine.label("periscope"), "潜望镜")
        self.assertEqual(submarine.label("surfaced"), "浮出")


if __name__ == "__main__":
    unittest.main()
