"""ROE 接入世界后的行为：默认不改结果，显式配置才收紧。

``test_roe.py`` 只测纯判定；这里验证接进 ``World._engage`` 之后
默认路径仍然照旧开火、配置后确实被拦住，且拒绝理由可查。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim.scenario import build_world  # noqa: E402

GRID = {"center_lon": 120.1, "center_lat": 22.9, "cols": 24, "rows": 24,
        "width_m": 2000.0}


def _basic():
    cells = []
    for r in range(24):
        row = []
        for c in range(24):
            row.append({"pos": r * 100 + c, "elev": 100, "cond": 0,
                        "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID), "map_data": cells}


def _op(uid, color, hexpos, name, weapons=(36,)):
    return {"obj_id": uid, "color": color, "type": 2, "sub_type": 0,
            "name": name, "blood": 3, "max_blood": 3, "armor": 4,
            "cur_hex": hexpos, "carry_weapon_ids": list(weapons),
            "value": 10, "move_state": 0}


def _scenario(roe=None):
    doc = {
        "scenario_id": 1,
        # 两队挨着放，保证能互相探测到
        "operators": [_op(1, 0, 10 * 100 + 10, "红1"),
                      _op(2, 1, 10 * 100 + 11, "蓝1")],
        "cities": [],
        "time": {"cur_step": 0, "tick": 30.0, "max_time": 3600},
        "engine": {"enable_tactics": False, "enable_commander": False},
    }
    if roe is not None:
        doc["roe"] = roe
    return doc


class RoeWiringTest(unittest.TestCase):
    def test_no_roe_config_keeps_old_behaviour(self):
        world = build_world(_scenario(), _basic(), seed=3)
        self.assertIsNone(world.roe, "未配置时不应安装手册")

    def test_roe_config_is_installed(self):
        world = build_world(_scenario({"policy": "hold_fire"}), _basic(), seed=3)
        self.assertIsNotNone(world.roe)
        self.assertEqual(world.roe.policy, "hold_fire")

    def test_invalid_policy_falls_back_without_failing(self):
        world = build_world(_scenario({"policy": "nonsense"}), _basic(), seed=3)
        self.assertIsNone(world.roe)
        self.assertTrue(getattr(world, "roe_warning", ""),
                        "非法政策名要留下警告，不能静默")

    def test_set_roe_accepts_dict_and_none(self):
        world = build_world(_scenario(), _basic(), seed=3)
        rules = world.set_roe({"policy": "weapons_tight",
                               "area_radius_m": 5000})
        self.assertEqual(rules.area_radius_m, 5000.0)
        world.set_roe(None)
        self.assertIsNone(world.roe)

    def test_contact_since_is_recorded(self):
        world = build_world(_scenario(), _basic(), seed=3)
        # 刷新一次感知，两个单位应互相进入接触表
        world.seen
        self.assertTrue(world.contact_since,
                        "感知刷新后应记录首次接触时刻")
        stamps = list(world.contact_since.values())
        self.assertTrue(all(t == world.time_s for t in stamps))

    def test_hold_fire_blocks_opening_shot(self):
        world = build_world(_scenario({"policy": "hold_fire"}), _basic(), seed=3)
        before = len([e for e in world.events
                      if e.get("type") in ("fire", "direct_fire")])
        for _ in range(5):
            world.step()
        after = len([e for e in world.events
                     if e.get("type") in ("fire", "direct_fire")])
        self.assertEqual(before, after, "hold_fire 下不该主动开火")
        self.assertTrue(world.roe_rejections, "被拒的交战要留下理由")

    def test_weapons_free_still_fires(self):
        world = build_world(_scenario({"policy": "weapons_free"}), _basic(), seed=3)
        for _ in range(5):
            world.step()
        fires = [e for e in world.events
                 if e.get("type") in ("fire", "direct_fire")]
        self.assertTrue(fires, "weapons_free 应照常开火")
        self.assertFalse(world.roe_rejections)

    def test_confirm_gate_delays_engagement(self):
        # confirm_s 很大时，刚接触就不该开火
        world = build_world(
            _scenario({"policy": "weapons_free", "confirm_s": 100000.0}),
            _basic(), seed=3)
        for _ in range(3):
            world.step()
        fires = [e for e in world.events
                 if e.get("type") in ("fire", "direct_fire")]
        self.assertFalse(fires, "未达确认门槛不应开火")
        reasons = list(world.roe_rejections.values())
        self.assertTrue(any("确认门槛" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
