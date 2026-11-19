"""指挥链中断：默认不配置时一行不差；配置后只锁指定的那个动作。

两个窗口分别对着 TODOLIST §5.4 的三问：
* upper —— 上级新命令送达；
* peer  —— 同级接触共享（共享本身仍是单跳，见 test_event_scheduler）。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim.commander import CommanderOrder  # noqa: E402
from sim.command_links import CommandLinks, Cut  # noqa: E402
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


def _op(uid, color, hexpos, name, comms=600.0):
    return {"obj_id": uid, "color": color, "type": 2, "sub_type": 0,
            "name": name, "blood": 3, "max_blood": 3, "armor": 4,
            "cur_hex": hexpos, "carry_weapon_ids": [36],
            "value": 10, "move_state": 0, "comms_range_m": comms}


def _scenario(links=None, commander=True):
    doc = {
        "scenario_id": 1,
        "operators": [_op(1, 0, 10 * 100 + 10, "红1"),
                      _op(2, 0, 10 * 100 + 12, "红2"),
                      _op(3, 0, 10 * 100 + 14, "红3"),
                      _op(4, 1, 10 * 100 + 11, "蓝1")],
        "cities": [],
        "time": {"cur_step": 0, "tick": 30.0, "max_time": 3600},
        "engine": {"enable_tactics": False, "enable_commander": commander},
    }
    if links is not None:
        doc["command_links"] = links
    return doc


class _PushAll(object):
    """Every call orders uid 1 and uid 2 to push to the same point."""

    def decide(self, state):
        return [CommanderOrder(uid=1, posture="push", lon=120.15, lat=22.95,
                               expires_s=5.0),
                CommanderOrder(uid=2, posture="push", lon=120.16, lat=22.96,
                               expires_s=5.0)]


class CommandLinkParseTest(unittest.TestCase):
    def test_absent_block_is_none(self):
        world = build_world(_scenario(), _basic(), seed=3)
        self.assertIsNone(world.command_links)

    def test_upper_cut_parsed(self):
        world = build_world(_scenario({"upper_cuts": [
            {"side": 0, "units": [1, 2], "cut_at_s": 10.0, "restore_at_s": 60.0}
        ]}), _basic(), seed=3)
        self.assertIsNotNone(world.command_links)
        cuts = world.command_links.describe()
        self.assertEqual(cuts[0]["kind"], "upper")
        self.assertEqual(cuts[0]["units"], [1, 2])

    def test_peer_pair_must_be_same_side(self):
        world = build_world(_scenario({"peer_cuts": [
            {"units": [1, 4], "cut_at_s": 10.0, "restore_at_s": 60.0}
        ]}), _basic(), seed=3)
        self.assertIsNone(world.command_links)
        self.assertIn("不同阵营", world.command_links_warning)

    def test_peer_pair_needs_two_uids(self):
        world = build_world(_scenario({"peer_cuts": [
            {"units": [1, 2, 3], "cut_at_s": 10.0, "restore_at_s": 60.0}
        ]}), _basic(), seed=3)
        self.assertIsNone(world.command_links)
        self.assertIn("恰好两个 uid", world.command_links_warning)

    def test_all_alive_restoration(self):
        # "恢复 = 相关单位全部存活"：全员活着时本来就没断，所以窗口关着；
        # 有人阵亡才进入中断，伤亡补齐（复活）后自动恢复。
        world = build_world(_scenario({"upper_cuts": [
            {"side": 0, "units": [1, 2], "cut_at_s": 0.0,
             "restore_when": {"all_alive": True}}
        ]}), _basic(), seed=3)
        cut = world.command_links.cuts[0]
        self.assertFalse(cut.active(0.0, world), "全员存活 = 链路正常")
        world.entities[2].alive = False
        self.assertTrue(cut.active(0.0, world), "有人阵亡即进入中断")
        world.entities[2].alive = True
        self.assertFalse(cut.active(0.0, world), "全员复活即恢复")
        # 两个判据同时写：谁先到谁生效。
        target = Cut(kind="upper", side=0, units=frozenset({1}), from_s=0.0,
                     until_s=5.0, all_alive=True)
        world.entities[1].alive = False
        self.assertTrue(target.active(1.0, world), "两判据都没到时窗口开着")
        world.entities[1].alive = True
        self.assertFalse(target.active(1.0, world), "存活判据先到即恢复")
        world.entities[1].alive = False
        self.assertFalse(target.active(6.0, world), "时间判据先到即恢复")


class UpperCutTest(unittest.TestCase):
    def _world(self, links):
        world = build_world(_scenario(links), _basic(), seed=3)
        world.set_commander(_PushAll(), interval_s=1.0)
        return world

    def test_no_cut_issues_commander_orders(self):
        world = self._world(None)
        world.step()
        self.assertTrue(any(v == "commander" for v in world._order_source.values()))

    def test_cut_blocks_new_commander_orders(self):
        world = self._world({"upper_cuts": [
            {"side": 0, "units": [1], "cut_at_s": 0.0, "restore_at_s": 100.0}
        ]})
        world.step()
        self.assertNotIn(1, world._commander_owned, "断链单位不该收到新命令")
        self.assertIn(2, world._commander_owned, "同侧未受影响的单位照常收到命令")

    def test_cut_does_not_stop_automatic_return_fire(self):
        # 还击不经 commander 路径：断链只锁新命令，不该影响交战。
        world = build_world(_scenario({"upper_cuts": [
            {"side": 0, "units": [1], "cut_at_s": 0.0, "restore_at_s": 100.0}
        ]}, commander=False), _basic(), seed=3)
        for _ in range(4):
            world.step()
        fires = [e for e in world.events
                 if e.get("type") in ("fire", "direct_fire")]
        self.assertTrue(fires, "断链不该让单位停火")

    def test_restored_link_resumes_orders(self):
        world = self._world({"upper_cuts": [
            {"side": 0, "units": [1], "cut_at_s": 0.0, "restore_at_s": 2.0}
        ]})
        world.step()
        self.assertNotIn(1, world._commander_owned)
        world.advance_to(3.0)
        world.step()
        self.assertIn(1, world._commander_owned, "窗口结束后应恢复收令")

    def test_manual_order_wins_regardless(self):
        world = self._world({"upper_cuts": [
            {"side": 0, "units": [1], "cut_at_s": 0.0, "restore_at_s": 100.0}
        ]})
        world.order(1, {"type": "move_to", "lon": 120.2, "lat": 22.9},
                    source="manual")
        self.assertEqual(world._order_source.get(1), "manual")


class PeerCutTest(unittest.TestCase):
    def test_peer_cut_stops_sharing_not_sharing_itself(self):
        # 直接调 CommandLinks：同一对在窗口内外给出不同答案。
        world = build_world(_scenario(), _basic(), seed=3)
        links = CommandLinks([Cut(kind="peer", pair=(1, 2), from_s=0.0,
                                  until_s=10.0)])
        self.assertTrue(links.peer_blocked(1, 2, 0.0, world))
        self.assertTrue(links.peer_blocked(2, 1, 1.0, world), "顺序无关")
        self.assertFalse(links.peer_blocked(1, 2, 11.0, world), "窗口结束即恢复")
        self.assertFalse(links.peer_blocked(1, 3, 0.0, world), "没声明的对不受影响")


if __name__ == "__main__":
    unittest.main()
