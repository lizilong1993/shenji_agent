"""两栖卸载节拍。

此前 ``_sync_passengers`` 在同一 tick 把整船乘客全部放上岸：舱门、艇位与
登陆艇数量都是有限的，一次放一个波次才贴近实际。这里锁住节拍与顺序。
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


def _basic(cond=None):
    cells = []
    for r in range(24):
        row = []
        for c in range(24):
            row.append({"pos": r * 100 + c, "elev": 100,
                        "cond": (cond if cond is not None else 0),
                        "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": dict(GRID), "map_data": cells}


def _op(uid, color, hexpos, name, domain="land", carrier=None):
    item = {"obj_id": uid, "color": color, "type": 2, "sub_type": 0,
            "name": name, "blood": 3, "max_blood": 3, "armor": 4,
            "cur_hex": hexpos, "carry_weapon_ids": [], "value": 10,
            "move_state": 0, "domain": domain}
    if carrier is not None:
        item["carrier"] = carrier
    return item


def _scenario():
    return {
        "scenario_id": 1,
        "operators": [
            _op(1, 0, 10 * 100 + 10, "登陆舰", domain="surface"),
            _op(101, 0, 10 * 100 + 10, "陆战1", carrier=1),
            _op(102, 0, 10 * 100 + 10, "陆战2", carrier=1),
            _op(103, 0, 10 * 100 + 10, "陆战3", carrier=1),
        ],
        "cities": [],
        "time": {"cur_step": 0, "tick": 30.0, "max_time": 3600},
        "engine": {"enable_tactics": False, "enable_commander": False},
    }


class UnloadCadenceTest(unittest.TestCase):
    def _world(self):
        # cond=0 全陆地：登陆舰一开局就在"能上岸"的条件下，便于观察节拍
        return build_world(_scenario(), _basic(), seed=5)

    def test_passengers_are_registered(self):
        world = self._world()
        carrier = world.entities[1]
        self.assertEqual(sorted(carrier.passengers), [101, 102, 103])

    def test_unload_interval_defaults_to_20s(self):
        world = self._world()
        self.assertEqual(world.unload_interval_s, 20.0)

    def test_one_wave_per_interval(self):
        world = self._world()
        carrier = world.entities[1]
        before = len(carrier.passengers)
        self.assertEqual(before, 3)
        # 同一个 tick 内只放一个
        world._sync_passengers()
        after_first = len(carrier.passengers)
        self.assertEqual(after_first, before - 1,
                         "一次只应放出一个波次，而不是整船同时下")
        # 到时间之前不再放
        world._sync_passengers()
        self.assertEqual(len(carrier.passengers), after_first)
        # 时间推进后放第二个
        world.time_s += world.unload_interval_s
        world._sync_passengers()
        self.assertEqual(len(carrier.passengers), after_first - 1)

    def test_unload_order_is_stable(self):
        world = self._world()
        carrier = world.entities[1]
        released = []
        for _ in range(3):
            first = carrier.passengers[0]
            world._sync_passengers()
            released.append(first)
            world.time_s += world.unload_interval_s
        self.assertEqual(released, [101, 102, 103],
                         "卸载顺序应按 uid 固定，保证同 seed 可复现")

    def test_event_is_emitted_per_wave(self):
        world = self._world()
        before = len([e for e in world.events if e.get("type") == "disembark"])
        world._sync_passengers()
        after = len([e for e in world.events if e.get("type") == "disembark"])
        self.assertEqual(after - before, 1)


if __name__ == "__main__":
    unittest.main()
