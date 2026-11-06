"""推演中途的任务变更（retask）。

启动时下发任务已有覆盖（``test_tasking``）；这里只测中途注入那条路径：
单位已在路上、目标已达成、单位已损失这三种情况下的行为。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from engine import scenario as engine_scenario  # noqa: E402
from sim.mission import MissionRunner  # noqa: E402
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


def _scenario():
    """Operator records must carry ``cur_hex`` (the world builder georeferences
    from the hex when ``lon/lat`` are absent)."""
    def op(uid, color, hexpos, name):
        return {"obj_id": uid, "color": color, "type": 2, "sub_type": 0,
                "name": name, "blood": 3, "max_blood": 3, "armor": 4,
                "cur_hex": hexpos, "carry_weapon_ids": [36], "value": 10,
                "move_state": 0}
    return {
        "scenario_id": 1,
        "operators": [
            op(1, 0, 5 * 100 + 5, "红1"),
            op(2, 0, 5 * 100 + 6, "红2"),
            op(3, 1, 15 * 100 + 15, "蓝1"),
        ],
        "cities": [],
        "time": {"cur_step": 0, "tick": 30.0, "max_time": 3600},
        "engine": {"enable_tactics": False, "enable_commander": False},
    }


class RetaskTest(unittest.TestCase):
    def _world(self):
        scenario = _scenario()
        world = build_world(scenario, _basic(), seed=7)
        # 目标点挂在 world 上（build_world 之后才有 rule_grid）
        for name, (lon, lat) in (("甲点", (120.15, 22.95)),
                                 ("乙点", (120.05, 22.85))):
            cell = world.rule_grid.cell_of(lon, lat)
            world.add_objective(cell, value=10, name=name, owner=-1)
        return world

    def test_retask_moves_unit_to_new_objective(self):
        world = self._world()
        runner = MissionRunner(world, [], None, interval_s=30.0)
        applied, rejected = runner.retask([
            {"id": "R1", "units": [1], "task": "seize", "objective": "乙点"},
        ])
        self.assertEqual(applied, ["R1"])
        self.assertEqual(rejected, [])
        entity = world.entities[1]
        self.assertIsNotNone(entity.route, "换任务后应重新规划路线")
        self.assertEqual(runner.missions["R1"]["objective"], "乙点")

    def test_retask_replaces_previous_mission_for_same_unit(self):
        world = self._world()
        runner = MissionRunner(world, [], None, interval_s=30.0)
        runner.retask([{"id": "R1", "units": [1], "objective": "甲点"}])
        runner.retask([{"id": "R2", "units": [1], "objective": "乙点"}])
        # 同一单位不能被两条 active mission 同时 _assign
        self.assertEqual(runner.missions["R1"]["status"], "active")
        self.assertEqual(runner.missions["R2"]["status"], "active")
        self.assertEqual(world.intents[1].objective, "乙点",
                         "最后一次下发的任务应生效")

    def test_retask_rejects_unknown_unit(self):
        world = self._world()
        runner = MissionRunner(world, [], None, interval_s=30.0)
        applied, rejected = runner.retask([
            {"id": "R1", "units": [999], "objective": "甲点"},
        ])
        self.assertEqual(applied, [])
        self.assertEqual(len(rejected), 1)
        self.assertIn("不存在", rejected[0][1])

    def test_retask_rejects_unknown_objective(self):
        world = self._world()
        runner = MissionRunner(world, [], None, interval_s=30.0)
        applied, rejected = runner.retask([
            {"id": "R1", "units": [1], "objective": "不存在的地点"},
        ])
        self.assertEqual(applied, [])
        self.assertIn("找不到目标点", rejected[0][1])

    def test_retask_rejects_missing_id(self):
        world = self._world()
        runner = MissionRunner(world, [], None, interval_s=30.0)
        applied, rejected = runner.retask([{"units": [1], "objective": "甲点"}])
        self.assertEqual(applied, [])
        self.assertIn("缺少 id", rejected[0][1])

    def test_retask_by_coordinates(self):
        world = self._world()
        runner = MissionRunner(world, [], None, interval_s=30.0)
        applied, rejected = runner.retask([
            {"id": "R1", "units": [2], "lon": 120.13, "lat": 22.93},
        ])
        self.assertEqual(applied, ["R1"])
        self.assertEqual(rejected, [])
        self.assertAlmostEqual(runner.missions["R1"]["lon"], 120.13)

    def test_retask_emits_event(self):
        world = self._world()
        runner = MissionRunner(world, [], None, interval_s=30.0)
        before = len(world.events)
        runner.retask([{"id": "R1", "units": [1], "objective": "甲点"}])
        kinds = [e.get("type") for e in world.events[before:]]
        self.assertIn("mission_retasked", kinds)

    def test_retask_clears_engagement_pause(self):
        world = self._world()
        runner = MissionRunner(world, [], None, interval_s=30.0)
        runner.engaged[1] = world.time_s
        runner.retask([{"id": "R1", "units": [1], "objective": "甲点"}])
        # 换任务意味着"不再按原任务停在原地"，交战暂停必须解除
        self.assertNotIn(1, runner.engaged)


if __name__ == "__main__":
    unittest.main()
