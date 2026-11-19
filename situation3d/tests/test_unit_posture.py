"""GP-05 单位级交战姿态：只覆盖被点名的单位，默认一行不差。

姿态词汇三种（``sim/roe.py`` ``UNIT_POSTURES``）：
* ``free``          —— 不受全场手册的收紧项约束（仍受 NSL 与人工命令规则）；
* ``hold_fire``     —— 只在还击窗口内开火；
* ``hold_position`` —— 原地警戒：与 hold_fire 同判据，语义上强调不挪窝。

全场手册仍是 ``roe.policy``；这张表只替换 ``policy`` 一项。
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


def _op(uid, color, hexpos, name):
    return {"obj_id": uid, "color": color, "type": 2, "sub_type": 0,
            "name": name, "blood": 3, "max_blood": 3, "armor": 4,
            "cur_hex": hexpos, "carry_weapon_ids": [36],
            "value": 10, "move_state": 0}


def _scenario(roe=None):
    doc = {
        "scenario_id": 1,
        # 红 1/红 2 与蓝 1 都挨着，默认都会打起来。
        "operators": [_op(1, 0, 10 * 100 + 10, "红1"),
                      _op(2, 0, 10 * 100 + 12, "红2"),
                      _op(3, 1, 10 * 100 + 11, "蓝1")],
        "cities": [],
        "time": {"cur_step": 0, "tick": 30.0, "max_time": 3600},
        "engine": {"enable_tactics": False, "enable_commander": False},
    }
    if roe is not None:
        doc["roe"] = roe
    return doc


def _fires(world):
    return [e for e in world.events if e.get("type") in ("fire", "direct_fire")]


class UnitPostureTest(unittest.TestCase):
    def test_no_posture_table_leaves_default_path(self):
        world = build_world(_scenario(), _basic(), seed=3)
        self.assertEqual(world.unit_postures, {})
        self.assertIsNone(world.roe)
        for _ in range(3):
            world.step()
        self.assertTrue(_fires(world), "没有配置时照常开火")

    def test_table_installs_without_global_policy(self):
        world = build_world(_scenario({"unit_postures": {"1": "hold_fire"}}),
                            _basic(), seed=3)
        self.assertEqual(world.unit_postures, {1: "hold_fire"})
        self.assertIsNone(world.roe, "只配单位姿态不应顺带装全场手册")

    def test_one_unit_holds_other_still_fires(self):
        world = build_world(_scenario({"unit_postures": {"1": "hold_fire"}}),
                            _basic(), seed=3)
        for _ in range(4):
            world.step()
        shooters = {int(e.get("actor", e.get("uid", -1)) or -1)
                    for e in _fires(world)}
        shooter_ids = set()
        for e in _fires(world):
            for key in ("actor", "uid", "attacker"):
                if e.get(key) is not None:
                    shooter_ids.add(int(e[key]))
        self.assertNotIn(1, shooter_ids, "hold_fire 的单位不该主动开火")
        self.assertTrue(any(world.roe_rejections.get((1, t)) for t in (3,)),
                        "被拒的交战要留下理由")

    def test_manual_fire_still_overrides_posture(self):
        world = build_world(_scenario({"unit_postures": {"1": "hold_position"}}),
                            _basic(), seed=3)
        world.order(1, {"type": "fire", "target": 3}, source="manual")
        world.step()
        self.assertEqual(world._order_source.get(1), "manual")
        # 人工开火不该被姿态挡下：要么开火，要么至少没有被 ROE 拒的记录
        self.assertNotIn("hold_fire", str(world.roe_rejections.get((1, 3), "")))

    def test_invalid_posture_warns_and_is_ignored(self):
        world = build_world(_scenario({"unit_postures": {
            "1": "nonsense", "99": "free"}}), _basic(), seed=3)
        self.assertEqual(world.unit_postures, {})
        warning = getattr(world, "roe_warning", "")
        self.assertIn("无效", warning)
        self.assertIn("不在本案编成内", warning)

    def test_posture_does_not_change_global_policy(self):
        world = build_world(_scenario({"policy": "weapons_tight",
                                       "area_radius_m": 50.0,
                                       "unit_postures": {"1": "free"}}),
                            _basic(), seed=3)
        self.assertEqual(world.roe.policy, "weapons_tight")
        self.assertEqual(world.unit_postures, {1: "free"})


if __name__ == "__main__":
    unittest.main()
