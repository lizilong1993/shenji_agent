"""空中加油最小规则：只有案例显式写 ``refuel`` 才有行为。

口径（``sim/refuel.py`` 文档字符串）：配对靠 uid，会合靠距离+高度差，
输油量 = min(速率×时长, 受油缺口, 加油机余额)，窗口到期或任一条件不满足即关窗。
未配置时 ``World.refuel`` 为 None，``_logistics`` 一行不变。
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

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 24, "rows": 24,
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


def _air(uid, lon, lat, alt, color=0, fuel=1.0, name=None):
    return {"obj_id": uid, "name": name or "air-%d" % uid, "color": color,
            "type": 3, "sub_type": 0, "domain": "air",
            "lon": lon, "lat": lat, "alt_m": alt,
            "blood": 2, "max_blood": 2, "fuel": fuel,
            "observe_distance": {"1": 1, "2": 25, "3": 1}}


def _world(refuel=None, tanker_fuel=1.0, receiver_fuel=0.4):
    ops = [_air(1, 119.900, 23.100, 3000.0, fuel=tanker_fuel, name="加油机"),
           _air(2, 119.900, 23.100, 3050.0, fuel=receiver_fuel, name="战斗机")]
    doc = {"engine": {"detection_model": "table", "scan_interval_s": 0},
           "operators": ops, "cities": [], "time": {"tick": 1.0},
           "fuel_ok": True}
    if refuel is not None:
        doc["refuel"] = refuel
    return build_world(doc, _basic(), seed=5)


class AirRefuelTest(unittest.TestCase):
    def test_no_config_leaves_legacy_path(self):
        world = _world()
        self.assertIsNone(world.refuel)
        before = world.entities[2].fuel
        for _ in range(5):
            world.step()
        self.assertEqual(world.entities[2].fuel, before, "未配置不该凭空加油")

    def test_rendezvous_transfers_at_rate(self):
        world = _world({"tanker": 1, "receiver": 2, "rate_per_s": 0.1,
                        "range_m": 500.0, "window_s": 60.0,
                        "supply": 0.5})
        self.assertEqual(world.entities[2].fuel, 0.4)
        world.step()                       # 开窗（首步只判定）
        self.assertTrue(world.refuel.active())
        world.step()
        self.assertAlmostEqual(world.entities[2].fuel, 0.5, delta=1e-6)
        self.assertAlmostEqual(world.refuel.supply, 0.4, delta=1e-6)

    def test_out_of_range_does_not_open(self):
        world = _world({"tanker": 1, "receiver": 2, "rate_per_s": 0.1,
                        "range_m": 100.0, "window_s": 60.0})
        world.entities[2].pose.lat += 0.01          # 约 1.1 km，超出 100 m
        for _ in range(3):
            world.step()
        self.assertFalse(world.refuel.active())
        self.assertEqual(world.entities[2].fuel, 0.4)

    def test_full_receiver_does_not_open(self):
        world = _world({"tanker": 1, "receiver": 2, "rate_per_s": 0.1,
                        "range_m": 500.0, "window_s": 60.0},
                       receiver_fuel=1.0)
        world.step()
        self.assertFalse(world.refuel.active())

    def test_window_expiry_closes_once(self):
        world = _world({"tanker": 1, "receiver": 2, "rate_per_s": 0.01,
                        "range_m": 500.0, "window_s": 10.0, "supply": 0.5})
        for _ in range(30):
            world.step()
        self.assertFalse(world.refuel.active(), "窗口到期应关闭")
        ends = [e for e in world.events if e.get("type") == "refuel_end"]
        self.assertEqual(len(ends), 1, "只关一次，不重复报")
        # 关窗后不再重开。
        for _ in range(5):
            world.step()
        self.assertFalse(world.refuel.active())

    def test_supply_exhausts(self):
        world = _world({"tanker": 1, "receiver": 2, "rate_per_s": 0.2,
                        "range_m": 500.0, "window_s": 60.0, "supply": 0.1},
                       receiver_fuel=0.1)
        for _ in range(10):
            world.step()
        self.assertAlmostEqual(world.refuel.supply, 0.0, delta=1e-6)
        self.assertAlmostEqual(world.entities[2].fuel, 0.2, delta=1e-6)
        self.assertFalse(world.refuel.active())

    def test_window_caps_large_dt(self):
        """快进路径一次给大 dt：只按窗口剩余时长输油，不超窗。"""
        world = _world({"tanker": 1, "receiver": 2, "rate_per_s": 0.01,
                        "range_m": 500.0, "window_s": 20.0, "supply": 1.0},
                       receiver_fuel=0.0)
        world.step()
        self.assertTrue(world.refuel.active())
        world._logistics(600.0)            # 远大于剩余窗口
        # step 已推进 1 s，窗口只剩 19 s → 0.01 × 19，而不是 0.01 × 600。
        self.assertAlmostEqual(world.entities[2].fuel, 0.19, delta=1e-6)
        world._advance_clock(600.0)        # 时钟跳到窗口之后
        world._logistics(0.0)              # 零时长也要判一次关窗
        self.assertFalse(world.refuel.active())

    def test_cross_side_pair_rejected(self):
        ops = [_air(1, 119.900, 23.100, 3000.0, color=0),
               _air(2, 119.900, 23.100, 3050.0, color=1, fuel=0.4)]
        doc = {"operators": ops, "engine": {"scan_interval_s": 0},
               "refuel": {"tanker": 1, "receiver": 2}}
        world = build_world(doc, _basic(), seed=5)
        self.assertIsNone(world.refuel)
        self.assertIn("不同阵营", world.refuel_warning)

    def test_missing_uid_warns(self):
        world = _world({"tanker": 1, "receiver": 99})
        self.assertIsNone(world.refuel)
        self.assertIn("不存在", world.refuel_warning)

    def test_bad_rate_warns(self):
        world = _world({"tanker": 1, "receiver": 2, "rate_per_s": 0})
        self.assertIsNone(world.refuel)
        self.assertIn("rate_per_s", world.refuel_warning)

    def test_ground_pair_does_not_open(self):
        world = _world({"tanker": 1, "receiver": 2, "rate_per_s": 0.1,
                        "range_m": 500.0, "window_s": 60.0})
        world.entities[1].domain = "land"
        for _ in range(3):
            world.step()
        self.assertFalse(world.refuel.active(), "加油机必须在空域")


if __name__ == "__main__":
    unittest.main()
