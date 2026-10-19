"""Rule tests for the clean-room land wargame core (stdlib unittest)."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from wargame_core import combat, hexmap, movement, weapons  # noqa: E402
from wargame_core.los import LineOfSight  # noqa: E402
from wargame_core.state import GameState, Objective, Unit  # noqa: E402


def make_map(rows=5, cols=5, elev=None, cond=None, roads=None):
    elev = elev or {}
    cond = cond or {}
    roads = roads or {}
    data = []
    for r in range(rows):
        row = []
        for c in range(cols):
            p = hexmap.encode(r, c)
            row.append({
                "elev": elev.get(p, 0),
                "cond": cond.get(p, hexmap.OPEN),
                "roads": roads.get(p, [0, 0, 0, 0, 0, 0]),
                "rivers": [0, 0, 0, 0, 0, 0],
            })
        data.append(row)
    return hexmap.HexMap({
        "map_id": 1, "max_row": rows, "max_column": cols,
        "scale": 200, "ele_grade": 10, "map_data": data,
    })


def make_unit(uid, color, pos, name="重型坦克", blood=3, armor=3):
    profile = weapons.weapons_for(name, 2, 0)
    return Unit(
        uid=uid, name=name, color=color, op_type=2, sub_type=0, pos=pos,
        blood=blood, max_blood=blood, armor=armor, speed_kmh=36.0,
        value=10, unit_type="vehicle", domain="land",
        weapons=profile, ammo={w.key: w.ammo for w in profile},
    )


class HexTests(unittest.TestCase):
    def test_encode_decode(self):
        self.assertEqual(hexmap.encode(25, 46), 2546)
        self.assertEqual(hexmap.decode(2546), (25, 46))

    def test_neighbors_order(self):
        # Even row: right, upper-right, upper-left, left, lower-left, lower-right
        self.assertEqual(hexmap.neighbors(24, 45, 100, 100),
                         [2446, 2345, 2344, 2444, 2544, 2545])

    def test_distance(self):
        self.assertEqual(hexmap.distance(2546, 2546), 0)
        self.assertEqual(hexmap.distance(2546, 2547), 1)
        self.assertEqual(hexmap.distance(2546, 2646), 1)

    def test_line_inclusive(self):
        line = hexmap.line(2444, 2446)
        self.assertEqual(line[0], 2444)
        self.assertEqual(line[-1], 2446)
        self.assertIn(2445, line)


class LosTests(unittest.TestCase):
    def test_adjacent_always_visible(self):
        m = make_map(elev={hexmap.encode(2, 2): 100})
        los = LineOfSight(m)
        self.assertTrue(los.visible(hexmap.encode(2, 2), hexmap.encode(2, 3)))

    def test_ridge_blocks(self):
        # A high ridge between (2,2) and (2,4) blocks the low line.
        m = make_map(rows=5, cols=7,
                     elev={hexmap.encode(2, 3): 100})
        los = LineOfSight(m)
        self.assertFalse(los.visible(hexmap.encode(2, 2), hexmap.encode(2, 4)))

    def test_high_observer_sees_over(self):
        m = make_map(rows=5, cols=7,
                     elev={hexmap.encode(2, 2): 100, hexmap.encode(2, 3): 20})
        los = LineOfSight(m)
        self.assertTrue(los.visible(hexmap.encode(2, 2), hexmap.encode(2, 4)))


class MovementTests(unittest.TestCase):
    def test_roadblock_blocks_vehicle_but_not_infantry(self):
        blocked = hexmap.encode(2, 3)
        m = make_map(cond={blocked: hexmap.ROADBLOCK})
        self.assertFalse(m.passable(blocked, "vehicle"))
        self.assertTrue(m.passable(blocked, "infantry"))
        vpath = movement.find_path(m, hexmap.encode(2, 2),
                                   hexmap.encode(2, 4), "vehicle")
        self.assertTrue(vpath)
        self.assertNotIn(blocked, vpath)
        self.assertTrue(movement.find_path(m, hexmap.encode(2, 2),
                                           hexmap.encode(2, 4), "infantry"))

    def test_road_is_cheaper(self):
        a, b = hexmap.encode(2, 2), hexmap.encode(2, 3)
        m = make_map(cond={b: hexmap.FOREST},
                     roads={b: [2, 0, 0, 0, 0, 0]})
        self.assertEqual(m.move_cost(a, b, "vehicle"), 1.0)
        self.assertGreater(make_map(cond={b: hexmap.FOREST}).move_cost(a, b, "vehicle"), 1.0)

    def test_astar_reaches_goal(self):
        m = make_map(rows=6, cols=6)
        path = movement.find_path(m, hexmap.encode(0, 0), hexmap.encode(5, 5), "vehicle")
        self.assertEqual(path[0], hexmap.encode(0, 0))
        self.assertEqual(path[-1], hexmap.encode(5, 5))
        self.assertGreater(len(path), 1)


class DetectionTests(unittest.TestCase):
    def test_observation_table(self):
        m = make_map(rows=12, cols=12)
        a = make_unit(1, 0, hexmap.encode(5, 5))
        b = make_unit(2, 1, hexmap.encode(5, 10))
        from wargame_core import detection
        self.assertTrue(detection.can_observe(a, b, m))
        b.pos = hexmap.encode(5, 11)
        # distance 6 > vehicle->vehicle 25? no, still <=25; use a far map
        self.assertTrue(detection.can_observe(a, b, m))

    def test_forest_halves_range(self):
        m = make_map(rows=30, cols=30, cond={hexmap.encode(5, 20): hexmap.FOREST})
        a = make_unit(1, 0, hexmap.encode(5, 5))
        b = make_unit(2, 1, hexmap.encode(5, 20))
        from wargame_core import detection
        self.assertFalse(detection.can_observe(a, b, m))


class CombatTests(unittest.TestCase):
    def _state(self):
        m = make_map(rows=10, cols=10)
        a = make_unit(1, 0, hexmap.encode(4, 4))
        b = make_unit(2, 1, hexmap.encode(4, 6), armor=1)
        return GameState(m, {1: a, 2: b}, [Objective(hexmap.encode(4, 6), 10)]), a, b

    def test_direct_fire_event_and_ammo(self):
        state, a, b = self._state()
        before = dict(a.ammo)
        ev = combat.resolve_direct_fire(a, b, state)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["type"], "direct_fire")
        self.assertEqual(ev["source"], "engine")
        self.assertIn(ev["result"], ("damage", "suppressed", "none"))
        spent = [k for k in before if a.ammo[k] < before[k]]
        self.assertEqual(len(spent), 1)

    def test_deterministic_same_seed(self):
        s1, a1, b1 = self._state()
        s2, a2, b2 = self._state()
        e1 = combat.resolve_direct_fire(a1, b1, s1)
        e2 = combat.resolve_direct_fire(a2, b2, s2)
        self.assertEqual(e1["roll"], e2["roll"])
        self.assertEqual(e1["result"], e2["result"])
        self.assertEqual(b1.blood, b2.blood)

    def test_objective_capture(self):
        m = make_map(rows=5, cols=5)
        a = make_unit(1, 0, hexmap.encode(2, 2))
        s = GameState(m, {1: a}, [Objective(hexmap.encode(2, 2), 10, -1)])
        changes = s.control_objectives()
        self.assertEqual(s.objectives[0].flag, 0)
        self.assertEqual(len(changes), 1)

    def test_suppression_expires(self):
        m = make_map(rows=5, cols=5)
        a = make_unit(1, 0, hexmap.encode(2, 2))
        a.suppressed = True
        a.suppressed_ticks = 1
        s = GameState(m, {1: a}, [])
        from wargame_core.engine import WargameEngine, Order
        WargameEngine(s).step([Order(1, "hold")])
        self.assertFalse(a.suppressed)


if __name__ == "__main__":
    unittest.main()
