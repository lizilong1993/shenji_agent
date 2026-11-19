"""GP-06 队形落点：默认仍是逐单位 17 候选试探；显式 formation 才成层。

队形只决定 **去哪**（Intent 的目的点），不改交战、不改运动积分。
"""

from __future__ import annotations

import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from geo.geodesy import haversine_m  # noqa: E402
from sim.mission import MissionRunner  # noqa: E402
from sim.scenario import build_world  # noqa: E402


GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 24, "rows": 24,
        "width_m": 200.0}


def _basic():
    cells = []
    for r in range(24):
        row = []
        for c in range(24):
            row.append({"pos": r * 100 + c, "elev": 100, "cond": 0,
                        "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID), "map_data": cells}


def _op(uid, lon, lat):
    return {"obj_id": uid, "name": "ZTD-05 两栖突击车", "color": 0, "type": 2,
            "domain": "land", "lon": lon, "lat": lat,
            "subclass_id": "land.vehicle.amphibious",
            "blood": 2, "max_blood": 2,
            "observe_distance": {"1": 10, "2": 25, "3": 1}}


def _world(formation=None):
    ops = [_op(1, 119.900, 23.100), _op(2, 119.901, 23.100),
           _op(3, 119.900, 23.101), _op(4, 119.901, 23.101)]
    mission = {"id": "M1", "side": 0, "units": [1, 2, 3, 4],
               "task": "move_to", "lon": 119.905, "lat": 23.105,
               "pace": "normal", "arrive_m": 100.0}
    if formation is not None:
        mission["formation"] = formation
    doc = {"engine": {"detection_model": "table", "scan_interval_s": 30,
                      "use_role_cards": False},
           "operators": ops, "missions": [mission], "roe": {}}
    world = build_world(doc, _basic(), seed=1)
    runner = MissionRunner(world, [mission], {"halt_on_contact": False})
    return world, runner, mission


class FormationSlotTest(unittest.TestCase):
    def test_no_formation_keeps_existing_single_point(self):
        world, runner, mission = _world()
        runner.update(force=True)
        goals = {uid: (world.entities[uid].goal or ()) for uid in (1, 2, 3, 4)}
        # 默认路径：所有单位先试同一个目标点（17 候选也以该点为首）。
        points = {(round(p[0], 9), round(p[1], 9)) for p in goals.values() if p}
        self.assertEqual(len(points), 1, points)

    def test_formation_spreads_destinations(self):
        world, runner, mission = _world({"spacing_m": 40.0})
        runner.update(force=True)
        goals = {}
        for uid in (1, 2, 3, 4):
            goal = world.entities[uid].goal
            self.assertIsNotNone(goal, "uid %d 没有目标点" % uid)
            goals[uid] = tuple(goal)
        points = list(goals.values())
        self.assertGreaterEqual(len({(round(p[0], 8), round(p[1], 8))
                                     for p in points}), 3,
                                "四个单位至少占三个不同落点")
        # 落点都在目标附近，而非原地。
        for p in points:
            self.assertLess(haversine_m(p, (mission["lon"], mission["lat"])), 100.0)

    def test_formation_slot_spacing(self):
        world, runner, mission = _world({"spacing_m": 50.0})
        slot1 = runner._formation_point(mission, 1, mission["formation"])
        slot2 = runner._formation_point(mission, 2, mission["formation"])
        self.assertAlmostEqual(haversine_m(slot1, slot2), 50.0, delta=1.0)

    def test_retask_preserves_formation(self):
        world, runner, mission = _world({"spacing_m": 40.0})
        applied, rejected = runner.retask([{
            "id": "M2", "side": 0, "units": [1, 2], "task": "move_to",
            "lon": 119.910, "lat": 23.110, "formation": {"spacing_m": 60.0},
        }])
        self.assertEqual(applied, ["M2"])
        self.assertEqual(rejected, [])
        # retask 新建的是 M2，M1 仍在表里（formation 40.0），按 id 取。
        status = {m["id"]: m for m in runner.status()}
        self.assertEqual(status["M2"]["formation"], {"spacing_m": 60.0})

    def test_malformed_formation_falls_back(self):
        world, runner, mission = _world({"spacing_m": -1.0})
        runner.update(force=True)
        # 非法间距退回旧路径：所有单位共用同一个目标点。
        points = {(round(world.entities[uid].goal[0], 9),
                   round(world.entities[uid].goal[1], 9))
                  for uid in (1, 2, 3, 4) if world.entities[uid].goal}
        self.assertEqual(len(points), 1, points)


if __name__ == "__main__":
    unittest.main()
