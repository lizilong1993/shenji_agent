"""P2 skeleton tests: rule grid, legacy adapter, authoritative weapon tables.

Run from ``situation3d/``:
    python3 -m unittest tests/test_geo_engine_skeleton.py
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from compat.legacy import LegacyHexGrid, legacy_scenario_positions  # noqa: E402
from geo.geodesy import haversine_m  # noqa: E402
from geo.rule_grid import RuleGrid  # noqa: E402
from rules.weapons import (  # noqa: E402
    TAR_TYPE_AIR,
    TAR_TYPE_PERSONNEL,
    TAR_TYPE_VEHICLE,
    WeaponDataError,
    WeaponTables,
    attack_level,
    attack_level_for,
)
from sim.entity import Entity, Pose  # noqa: E402
from sim.world import World  # noqa: E402
from env.service import FlatEnvironment  # noqa: E402


class TestRuleGrid(unittest.TestCase):
    def setUp(self):
        # Small synthetic AOI near the Taiwan Strait.
        self.grid = RuleGrid(119.9, 23.1, cols=40, rows=30)

    def test_rule_cell_size_is_200m(self):
        self.assertEqual(self.grid.cell_m, 200.0)
        a = self.grid.cell_center_lonlat(5, 5)
        b = self.grid.cell_center_lonlat(5, 6)
        self.assertAlmostEqual(haversine_m(a, b), 200.0, delta=1.0)
        # Every hex neighbour centre is 200 m away (odd-r stagger included).
        c = self.grid.cell_center_lonlat(6, 5)
        self.assertAlmostEqual(haversine_m(a, c), 200.0, delta=1.0)

    def test_cell_roundtrip_within_one_metre(self):
        for cell in [(0, 0), (1, 1), (10, 20), (29, 39), (15, 0)]:
            lon, lat = self.grid.cell_center_lonlat(*cell)
            self.assertEqual(self.grid.cell_of(lon, lat), cell)
            back_lon, back_lat = self.grid.cell_center_lonlat(
                *self.grid.cell_of(lon, lat))
            self.assertLess(haversine_m((lon, lat), (back_lon, back_lat)), 1.0)

    def test_hex_distance(self):
        self.assertEqual(self.grid.distance((0, 0), (0, 1)), 1)
        self.assertEqual(self.grid.distance((0, 0), (1, 0)), 1)
        self.assertEqual(self.grid.distance((0, 0), (2, 0)), 2)
        # Adjacent cell centres are 200 m apart, but a distance-2 zigzag pair
        # is 346 m apart; Euclidean floor(346/200)=1 would be wrong.
        a = self.grid.cell_center_lonlat(0, 0)
        b = self.grid.cell_center_lonlat(1, 1)
        self.assertEqual(self.grid.distance((0, 0), (1, 1)), 2)
        self.assertGreater(haversine_m(a, b), 340.0)

    def test_neighbors(self):
        nbs = self.grid.neighbors((5, 5))
        self.assertEqual(len(nbs), 6)
        for nb in nbs:
            self.assertEqual(self.grid.distance((5, 5), nb), 1)

    def test_covering_bbox(self):
        g = RuleGrid.covering_bbox(119.5, 22.8, 120.1, 23.4)
        self.assertEqual(g.cell_m, 200.0)
        self.assertTrue(g.in_bounds(g.cell_of(119.8, 23.1)))


class TestLegacyAdapter(unittest.TestCase):
    GRID_META = {
        "center_lon": 119.95,
        "center_lat": 23.10,
        "cols": 80,
        "rows": 60,
        "width_m": 1500.0,
    }

    def test_legacy_roundtrip(self):
        grid = LegacyHexGrid.from_grid_meta(self.GRID_META)
        self.assertEqual(grid.cell_m, 1500.0)
        for pos in (1010, 2020, 3030, 5967):
            lon, lat = grid.to_lonlat(pos)
            self.assertEqual(grid.from_lonlat(lon, lat), pos)

    def test_missing_grid_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            LegacyHexGrid.from_map_doc({"map_data": []})

    def test_scenario_positions(self):
        grid = LegacyHexGrid.from_grid_meta(self.GRID_META)
        scenario = {"operators": [
            {"obj_id": 1, "cur_hex": 1010},
            {"obj_id": 2, "cur_hex": 2020},
        ]}
        out = legacy_scenario_positions(scenario, grid)
        self.assertEqual(sorted(out), [1, 2])
        self.assertEqual(len(out[1]), 2)

    def test_rule_and_legacy_grids_are_separate(self):
        rule = RuleGrid(self.GRID_META["center_lon"], self.GRID_META["center_lat"],
                        self.GRID_META["cols"], self.GRID_META["rows"])
        legacy = LegacyHexGrid.from_grid_meta(self.GRID_META)
        self.assertEqual(rule.cell_m, 200.0)
        self.assertEqual(legacy.cell_m, 1500.0)
        # Same lon/lat must map to different row/col on the two grids.
        lon, lat = legacy.to_lonlat(3030)
        self.assertNotEqual(rule.cell_of(lon, lat), legacy.from_lonlat(lon, lat))


class TestAuthoritativeWeapons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = WeaponTables(load_wpdata_json())

    def test_tables_are_well_formed(self):
        self.assertEqual(self.tables.validate_all(), [])
        self.assertGreaterEqual(len(self.tables.weapon_ids()), 20)

    def test_tar_type_mapping_matches_observed_engine(self):
        # Weapon 36 vs vehicle: observed judge_info 6/5/4/2 at 14/15/16/18.
        for distance, expected in [(14, 6), (15, 5), (16, 4), (18, 2)]:
            self.assertEqual(
                attack_level_for(self.tables, 36, "vehicle", 0, distance),
                expected)
        # TarType=1 is personnel, indexed by ObjBlood (blood=1 -> constant 2).
        table = self.tables.table(36, TAR_TYPE_PERSONNEL, 1)
        self.assertEqual(table.att_max, 10)
        self.assertEqual(table.values[:11], (2,) * 11)

    def test_vehicle_table_not_truncated(self):
        table = self.tables.table(36, TAR_TYPE_VEHICLE, 0)
        self.assertEqual(table.att_max, 18)
        self.assertEqual(len(table.values), 19)
        self.assertEqual(table.values[-1], 2)

    def test_air_table(self):
        table = self.tables.table(1, TAR_TYPE_AIR, 0)
        self.assertEqual(table.att_max, 20)
        self.assertEqual(table.values[0], 9)
        self.assertEqual(table.values[-1], 2)

    def test_out_of_range_returns_none(self):
        table = self.tables.table(36, TAR_TYPE_VEHICLE, 0)
        self.assertIsNone(attack_level(table, 19))
        self.assertIsNone(attack_level(table, -1))

    def test_missing_combination_raises(self):
        with self.assertRaises(WeaponDataError):
            self.tables.table(36, TAR_TYPE_VEHICLE, 3)


class TestWorldSkeleton(unittest.TestCase):
    def test_stacking_uses_rule_cell(self):
        grid = RuleGrid(119.9, 23.1, cols=20, rows=20)
        world = World(FlatEnvironment(), grid)
        lon, lat = grid.cell_center_lonlat(5, 5)
        a = Entity(1, "A", 0, "land", "mbt", Pose(lon, lat))
        b = Entity(2, "B", 0, "land", "mbt", Pose(lon, lat))
        c = Entity(3, "C", 1, "land", "mbt", Pose(lon, lat))
        for e in (a, b, c):
            world.add_entity(e)
        self.assertTrue(world.stacked(a))
        self.assertFalse(world.stacked(c))
        self.assertEqual(world.rule_distance(a, c), 0)
        self.assertEqual(len(world.entities_in_cell(world.rule_cell(a))), 3)

    def test_step_advances_time(self):
        grid = RuleGrid(119.9, 23.1, cols=10, rows=10)
        world = World(FlatEnvironment(), grid, dt_s=1.0)
        self.assertEqual(world.step(), [])
        self.assertAlmostEqual(world.time_s, 1.0)


def load_wpdata_json():
    from rules.weapons import load_wpdata
    return load_wpdata()


if __name__ == "__main__":
    unittest.main()
