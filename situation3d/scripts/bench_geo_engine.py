#!/usr/bin/env python3
"""Benchmark the geo engine at 300/500 entities.

Usage: python3 scripts/bench_geo_engine.py --entities 300 --ticks 100
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim.scenario import build_world  # noqa: E402

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 60, "rows": 60, "width_m": 200.0}


def make_basic():
    cells = []
    for r in range(60):
        row = []
        for c in range(60):
            row.append({"pos": r * 100 + c, "elev": 100 + (r % 5),
                        "cond": 0, "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": GRID, "map_data": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=100)
    args = ap.parse_args()
    basic = make_basic()
    ops = []
    rng = random.Random(1)
    for i in range(args.entities):
        side = i % 2
        pos = rng.randrange(10, 50) * 100 + rng.randrange(10, 50)
        ops.append({"obj_id": i + 1, "color": side, "type": 2, "name": "重型坦克",
                    "sub_type": 0, "blood": 3, "max_blood": 3, "armor": 4,
                    "cur_hex": pos, "carry_weapon_ids": [36], "value": 10})
    scenario = {"scenario_id": 1, "operators": ops, "cities": [],
                "time": {"cur_step": 0, "tick": 1.0, "max_time": 9999}}
    world = build_world(scenario, basic, seed=1)
    times = []
    for _ in range(args.ticks):
        t = time.perf_counter()
        world.step()
        times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    print("entities", args.entities, "ticks", args.ticks,
          "p50", round(statistics.median(times), 2),
          "p95", round(times[int(len(times) * 0.95)], 2),
          "max", round(times[-1], 2),
          "events", len(world.events))


if __name__ == "__main__":
    main()
