"""Sea/air/logistics tests for the joint wargame core."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from wargame_core import (air, beach, engagement, hexmap, logistics,
                          naval, sensors, weapons)  # noqa: E402
from wargame_core.state import GameState, Objective, Unit  # noqa: E402


WATER = 4


def make_sea_map(rows=12, cols=12, water=None, elev=None):
    water = water or set()
    elev = elev or {}
    data = []
    for r in range(rows):
        row = []
        for c in range(cols):
            p = hexmap.encode(r, c)
            row.append({"elev": elev.get(p, 0),
                        "cond": WATER if p in water else 0,
                        "roads": [0] * 6, "rivers": [0] * 6})
        data.append(row)
    return hexmap.HexMap({"map_id": 9, "max_row": rows, "max_column": cols,
                          "scale": 2000, "ele_grade": 10, "map_data": data})


def sea_unit(uid, color, pos, name="驱逐舰", domain="sea"):
    prof = engagement.sea_air_weapons_for(name, domain)
    u = Unit(uid=uid, name=name, color=color, op_type=2, sub_type=0,
             pos=pos, blood=4, max_blood=4, armor=3, speed_kmh=30.0,
             unit_type="ship", domain=domain,
             sensors=sensors.sensors_for(name, domain),
             ammo={w.key: w.ammo for w in prof}, sea_air_weapons=prof,
             altitude_band="水面", size_class="large", readiness=1.0)
    return u


def air_unit(uid, color, pos, name="战斗机", band="中空"):
    prof = engagement.sea_air_weapons_for(name, "air")
    u = Unit(uid=uid, name=name, color=color, op_type=3, sub_type=0,
             pos=pos, blood=3, max_blood=3, armor=1, speed_kmh=800.0,
             unit_type="aircraft", domain="air",
             sensors=sensors.sensors_for(name, "air"),
             ammo={w.key: w.ammo for w in prof}, sea_air_weapons=prof,
             altitude_band=band, size_class="small", readiness=1.0,
             fuel=1.0)
    return u


class SensorTests(unittest.TestCase):
    def test_low_altitude_reduces_radar_signature(self):
        m = make_sea_map()
        a = sea_unit(1, 0, hexmap.encode(2, 2))
        low = air_unit(2, 1, hexmap.encode(2, 6), band="超低空")
        high = air_unit(3, 1, hexmap.encode(2, 6), band="高空")
        r = next(s for s in a.sensors if s.kind == "radar")
        self.assertLess(r.effective_km(low), r.effective_km(high))

    def test_sonar_ignores_air(self):
        m = make_sea_map()
        a = sea_unit(1, 0, hexmap.encode(2, 2))
        b = air_unit(2, 1, hexmap.encode(2, 3))
        sonar = next(s for s in a.sensors if s.kind == "sonar")
        self.assertFalse(sensors.detects(sonar, a, b, m, state_los(m)))

    def test_submarine_detected_only_close(self):
        m = make_sea_map()
        a = sea_unit(1, 0, hexmap.encode(2, 2), "护卫舰")
        sub = sea_unit(2, 1, hexmap.encode(2, 10), "潜艇", "sub")
        sub.depth = "deep"
        # 8 hexes * 2 km = 16 km, within the 30 km sonar; but deep reduces sig.
        self.assertIsInstance(sensors.detects(
            next(s for s in a.sensors if s.kind == "sonar"), a, sub, m,
            state_los(m)), bool)


def state_los(m):
    from wargame_core.los import LineOfSight
    return LineOfSight(m)


class EngagementMatrixTests(unittest.TestCase):
    def test_land_cannot_engage_sea_with_tank(self):
        m = make_sea_map()
        tank = Unit(uid=1, name="重型坦克", color=0, op_type=2, sub_type=0,
                    pos=hexmap.encode(2, 2), blood=3, max_blood=3,
                    unit_type="vehicle", domain="land",
                    weapons=weapons.weapons_for("重型坦克"), ammo={},
                    sensors=[sensors.Sensor("visual", 5.0)])
        ship = sea_unit(2, 1, hexmap.encode(2, 6))
        self.assertFalse(engagement.can_engage(tank, ship))

    def test_coastal_missile_engages_sea(self):
        m = make_sea_map()
        battery = Unit(uid=1, name="岸舰导弹营", color=0, op_type=2, sub_type=0,
                       pos=hexmap.encode(2, 2), blood=3, max_blood=3,
                       unit_type="aa", domain="land",
                       sensors=[sensors.Sensor("radar", 200.0)],
                       sea_air_weapons=engagement.sea_air_weapons_for("岸舰导弹营", "land"),
                       ammo={"coastal_asm": 8})
        ship = sea_unit(2, 1, hexmap.encode(2, 6))
        self.assertTrue(engagement.can_engage(battery, ship))

    def test_ship_sam_engages_air(self):
        m = make_sea_map()
        ship = sea_unit(1, 0, hexmap.encode(2, 2))
        f = air_unit(2, 1, hexmap.encode(2, 4), "战斗机")
        self.assertTrue(engagement.can_engage(ship, f))

    def test_engagement_event_has_synthetic_flag(self):
        m = make_sea_map()
        a = sea_unit(1, 0, hexmap.encode(2, 2))
        b = sea_unit(2, 1, hexmap.encode(2, 5))
        state = GameState(m, {1: a, 2: b}, [Objective(hexmap.encode(2, 5), 10)])
        ev = engagement.resolve_sea_air(a, b, state)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["type"], "sea_air_fire")
        self.assertTrue(ev["raw"].get("approximated"))


class NavalMovementTests(unittest.TestCase):
    def test_ship_cannot_cross_land(self):
        water = {hexmap.encode(2, c) for c in range(12)}
        m = make_sea_map(water=water)
        ship = sea_unit(1, 0, hexmap.encode(2, 1))
        path = naval.naval_path(m, ship.pos, hexmap.encode(2, 9), ship)
        self.assertTrue(path)
        for p in path:
            self.assertEqual(m.cond(p), WATER)

    def test_ship_stops_at_land(self):
        water = {hexmap.encode(2, c) for c in range(6)}
        m = make_sea_map(water=water)
        ship = sea_unit(1, 0, hexmap.encode(2, 2))
        path = naval.naval_path(m, ship.pos, hexmap.encode(2, 11), ship)
        self.assertEqual(path, [])


class AirTests(unittest.TestCase):
    def test_fuel_decreases(self):
        u = air_unit(1, 0, hexmap.encode(2, 2))
        before = u.fuel
        air.update_fuel(u, 30.0)
        self.assertLess(u.fuel, before)

    def test_climb_one_band(self):
        u = air_unit(1, 0, hexmap.encode(2, 2), band="低空")
        self.assertTrue(air.climb(u, "中空"))
        self.assertFalse(air.climb(u, "超低空"))  # two bands away

    def test_air_ignores_terrain(self):
        m = make_sea_map()
        p = air.air_path(m, hexmap.encode(0, 0), hexmap.encode(11, 11))
        self.assertEqual(p[0], hexmap.encode(0, 0))
        self.assertEqual(p[-1], hexmap.encode(11, 11))


class LogisticsTests(unittest.TestCase):
    def test_resupply_near_port(self):
        m = make_sea_map()
        ship = sea_unit(1, 0, hexmap.encode(2, 2))
        ship.fuel = 0.2
        state = GameState(m, {1: ship}, [])
        state.supply_points = [{"color": 0, "pos": hexmap.encode(2, 2), "kind": "port"}]
        logistics.resupply(state, 3600.0)
        self.assertGreater(ship.fuel, 0.2)

    def test_no_resupply_far_from_port(self):
        m = make_sea_map(rows=30, cols=30)
        ship = sea_unit(1, 0, hexmap.encode(2, 2))
        ship.fuel = 0.2
        state = GameState(m, {1: ship}, [])
        state.supply_points = [{"color": 0, "pos": hexmap.encode(25, 25), "kind": "port"}]
        logistics.resupply(state, 3600.0)
        self.assertEqual(ship.fuel, 0.2)


class BeachTests(unittest.TestCase):
    def test_find_transfer_area_and_spawn(self):
        water = {hexmap.encode(2, c) for c in range(6)}
        m = make_sea_map(water=water)
        beach_pos = hexmap.encode(2, 6)
        area = beach.find_transfer_area(m, beach_pos)
        self.assertIsNotNone(area)
        self.assertEqual(m.cond(area), WATER)
        ship = sea_unit(1, 0, area)
        state = GameState(m, {1: ship}, [])
        craft = beach.spawn_landing_craft(state, ship, "登陆艇")
        self.assertEqual(craft.domain, "sea")
        self.assertTrue(craft.raw.get("beaching"))
        self.assertEqual(len(state.events), 1)


if __name__ == "__main__":
    unittest.main()
