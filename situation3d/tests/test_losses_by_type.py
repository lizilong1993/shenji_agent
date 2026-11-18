"""GP-07：按兵种的战损明细。

只测统计口径本身，不跑推演：``_losses_by_type()`` 的输入就是 world.entities
与插件 meta，两者都能用轻量替身构造。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
for p in (SITUATION3D, os.path.join(SITUATION3D, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_situation_case import _losses_by_type  # noqa: E402


class _Entity(object):
    def __init__(self, unit_kind, value, alive):
        self.unit_kind = unit_kind
        self.value = value
        self.alive = alive


class _World(object):
    def __init__(self, entities):
        self.entities = entities


def _meta(affiliation, uid):
    return {"uid": uid, "affiliation": affiliation}


class LossesByTypeTest(unittest.TestCase):
    def test_only_dead_units_are_counted(self):
        world = _World({1: _Entity("vehicle", 10, True),
                        2: _Entity("vehicle", 10, False)})
        meta = {"A": _meta(0, 1), "B": _meta(0, 2)}
        out = _losses_by_type(world, meta)
        self.assertEqual(out["red"], {"vehicle": {"count": 1, "value": 10}})
        self.assertEqual(out["blue"], {})

    def test_kind_is_the_key_and_value_sums(self):
        world = _World({1: _Entity("ship", 30, False),
                        2: _Entity("ship", 30, False),
                        3: _Entity("aircraft", 40, False)})
        meta = {"A": _meta(0, 1), "B": _meta(0, 2), "C": _meta(1, 3)}
        out = _losses_by_type(world, meta)
        self.assertEqual(out["red"]["ship"], {"count": 2, "value": 60})
        self.assertEqual(out["blue"]["aircraft"], {"count": 1, "value": 40})

    def test_sides_are_separated(self):
        world = _World({1: _Entity("vehicle", 10, False),
                        2: _Entity("vehicle", 10, False)})
        meta = {"A": _meta(0, 1), "B": _meta(1, 2)}
        out = _losses_by_type(world, meta)
        self.assertEqual(out["red"]["vehicle"]["count"], 1)
        self.assertEqual(out["blue"]["vehicle"]["count"], 1)

    def test_units_outside_meta_are_ignored(self):
        # 运行时生成的单位（不在插件 meta 里）不计入：本项目用它排除临时单位。
        world = _World({1: _Entity("vehicle", 10, False),
                        999: _Entity("vehicle", 10, False)})
        meta = {"A": _meta(0, 1)}
        out = _losses_by_type(world, meta)
        self.assertEqual(out["red"]["vehicle"], {"count": 1, "value": 10})

    def test_missing_entity_and_no_dead_units(self):
        world = _World({1: _Entity("vehicle", 10, True)})
        meta = {"A": _meta(0, 1), "B": _meta(0, 2)}   # 2 不在 world 里
        out = _losses_by_type(world, meta)
        self.assertEqual(out, {"red": {}, "blue": {}})

    def test_totals_match_the_plain_losses_counter(self):
        # 明细的 count 之和必须等于同一批 meta 上 old 口径数出来的总数。
        world = _World({float(i): _Entity("vehicle", 10, i % 2 == 0)
                        for i in range(1, 7)})
        meta = {str(i): _meta(0, float(i)) for i in range(1, 7)}
        out = _losses_by_type(world, meta)
        counted = sum(rec["count"] for rec in out["red"].values())
        expected = sum(1 for m in meta.values()
                       if not world.entities[m["uid"]].alive)
        self.assertEqual(counted, expected)


if __name__ == "__main__":
    unittest.main()
