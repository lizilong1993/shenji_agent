"""P0 regression fixture: time.tick > 1 must not silently mean seconds.

Locks the current (pre-geo) behaviour so later refactors cannot change it
without an explicit decision:

* ``scenario.build_state`` ignores ``time.tick``; the engine step is driven
  by the caller's ``tick_seconds`` (30 s for the campaign case).
* One engine step advances the tick counter by exactly one, whatever
  ``time.tick`` says.
* The SDK formulas (frames per step, 1 s per frame) are centralised in
  ``compat.timebase``.
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from compat.timebase import (  # noqa: E402
    sdk_max_step,
    sdk_step_seconds,
    sdk_total_seconds,
)
from wargame_core import engine, scenario  # noqa: E402
from wargame_core.hexmap import HexMap  # noqa: E402


def _tiny_map():
    cells = []
    for r in range(3):
        row = []
        for c in range(3):
            row.append({"pos": r * 100 + c, "elev": 0, "cond": 0,
                        "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return HexMap({"map_id": 1, "ele_grade": 20, "map_data": cells})


def _scenario(tick_value):
    return {
        "scenario_id": 424242,
        "operators": [{
            "obj_id": 1, "color": 0, "type": 2, "name": "重型坦克",
            "sub_type": 0, "basic_speed": 36, "armor": 4, "blood": 3,
            "max_blood": 3, "cur_hex": 101, "carry_weapon_ids": [36],
            "observe_distance": {"1": 10, "2": 25, "3": 1},
        }],
        "time": {"cur_step": 0, "tick": tick_value, "max_time": 100},
        "cities": [],
    }


class TestTimeBaseFormulas(unittest.TestCase):
    def test_sdk_step_count(self):
        self.assertEqual(sdk_max_step(1800, 1.0), 1800)
        self.assertEqual(sdk_max_step(1800, 5.0), 360)
        self.assertEqual(sdk_max_step(100, 3.0), 33)  # floor, frames rounded down
        self.assertEqual(sdk_max_step(100, 0), 100)

    def test_sdk_seconds(self):
        self.assertEqual(sdk_total_seconds(1800), 1800.0)
        self.assertEqual(sdk_step_seconds(5.0), 5.0)


class TestTickAboveOneFixture(unittest.TestCase):
    def test_scenario_tick_is_not_read_as_seconds(self):
        state = scenario.build_state(_scenario(5.0), _tiny_map(),
                                     seed=7, tick_seconds=30.0)
        # Current behaviour (baseline 6982407): time.tick is ignored and the
        # caller-supplied tick_seconds wins.  A future geo engine that honours
        # the SDK frame semantics must update this fixture explicitly.
        self.assertEqual(state.tick_seconds, 30.0)
        self.assertEqual(state.max_time, 100)
        self.assertEqual(state.tick, 0)

    def test_one_engine_step_is_one_tick_for_any_tick_value(self):
        for tick_value in (1.0, 2.0, 5.0):
            state = scenario.build_state(_scenario(tick_value), _tiny_map(),
                                         seed=7, tick_seconds=30.0)
            eng = engine.WargameEngine(state)
            for _ in range(3):
                eng.step([])
            self.assertEqual(state.tick, 3, "tick=%s" % tick_value)
            self.assertEqual(state.tick * state.tick_seconds, 90.0)
            self.assertEqual(state.max_time, 100)


if __name__ == "__main__":
    unittest.main()
