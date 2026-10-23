"""Regression gate for engine throughput (TODO #3).

Measures steady-state tick times (warm-up excluded) for 300, 500 and 800 entities
and pins the event hash so future optimisations must preserve adjudication
semantics (same seed -> same events).  The hashes were re-pinned for P0.2:
physical spawn de-overlap and kinematic motion change adjudication ordering.
"""
import hashlib
import json
import os
import random
import statistics
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim.scenario import build_world

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 60, "rows": 60,
        "width_m": 200.0}
# Acceptance target is 100 ms; shared-host scheduling noise pushes 500-entity
# p95 into the 90-110 ms range, so the gate allows a small margin.
P95_LIMIT_MS = 100.0  # acceptance target; 800-entity standalone ~60-66 ms
P99_LIMIT_MS = 150.0  # covers the scan spikes p95 skips
TICKS = 70  # covers >=2 scan periods so the periodic scan spike is gated


def make_basic():
    cells = []
    for r in range(60):
        row = []
        for c in range(60):
            row.append({"pos": r * 100 + c, "elev": 100 + (r % 5),
                        "cond": 0, "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": GRID, "map_data": cells}


def build_bench_world(entities):
    basic = make_basic()
    rng = random.Random(1)
    ops = []
    for i in range(entities):
        pos = rng.randrange(10, 50) * 100 + rng.randrange(10, 50)
        ops.append({"obj_id": i + 1, "color": i % 2, "type": 2,
                    "name": "重型坦克", "sub_type": 0, "blood": 3,
                    "max_blood": 3, "armor": 4, "cur_hex": pos,
                    "carry_weapon_ids": [36], "value": 10})
    scenario = {"scenario_id": 1, "operators": ops, "cities": [],
                "time": {"cur_step": 0, "tick": 1.0, "max_time": 9999}}
    return build_world(scenario, basic, seed=1)


def run_bench(entities, ticks, warmup):
    world = build_bench_world(entities)
    times = []
    for i in range(ticks):
        t = time.perf_counter()
        world.step()
        if i >= warmup:
            times.append((time.perf_counter() - t) * 1000.0)
    digest = hashlib.sha256(
        json.dumps(world.events, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    times.sort()
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
    # p95 deliberately excludes the two lockstep scan spikes in a 60-sample
    # window; p99 keeps the worst step so a spike regression is gated.
    p99 = times[min(len(times) - 1, int(len(times) * 0.99))]
    return {"p50": statistics.median(times), "p95": p95, "p99": p99,
            "events": len(world.events), "hash": digest}


class TestBenchRegression(unittest.TestCase):
    def test_300_entities(self):
        result = run_bench(300, ticks=TICKS, warmup=10)
        self.assertEqual(result["hash"], "628135508d4b7f18")
        self.assertLess(result["p95"], P95_LIMIT_MS,
                        "300-entity p95 %.1f ms" % result["p95"])
        self.assertLess(result["p99"], P99_LIMIT_MS,
                        "300-entity p99 %.1f ms" % result["p99"])

    def test_500_entities(self):
        result = run_bench(500, ticks=TICKS, warmup=10)
        self.assertEqual(result["hash"], "e72dcc0f9f349180")
        self.assertLess(result["p95"], P95_LIMIT_MS,
                        "500-entity p95 %.1f ms" % result["p95"])
        self.assertLess(result["p99"], P99_LIMIT_MS,
                        "500-entity p99 %.1f ms" % result["p99"])

    def test_800_entities(self):
        result = run_bench(800, ticks=TICKS, warmup=10)
        self.assertEqual(result["hash"], "6f07cad39f8c94bc")
        self.assertLess(result["p95"], P95_LIMIT_MS,
                        "800-entity p95 %.1f ms" % result["p95"])
        self.assertLess(result["p99"], P99_LIMIT_MS,
                        "800-entity p99 %.1f ms" % result["p99"])


if __name__ == "__main__":
    unittest.main()
