#!/usr/bin/env python3
"""Run a minimal clean-room land wargame demo.

Usage:
    python3 situation3d/scripts/run_wargame_core_demo.py \
        --scenario land_wargame_sdk/Data/Data/scenarios/1231.json \
        --map land_wargame_sdk/Data/Data/maps/map_123/basic.json \
        --ticks 60 --out /tmp/wargame_core_demo.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from wargame_core import engine, scenario  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--map", required=False)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--tick-seconds", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    sc = scenario.load_json(args.scenario)
    map_path = args.map
    if not map_path:
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.scenario)))
        map_path = scenario.find_map_for_scenario(sc, data_dir)
    if not map_path or not os.path.isfile(map_path):
        print("ERROR: map not found; pass --map", file=sys.stderr)
        return 2

    state = scenario.load_scenario(args.scenario, map_path,
                                   seed=args.seed,
                                   tick_seconds=args.tick_seconds)
    result = engine.run_demo(state, max_ticks=args.ticks, verbose=args.verbose)
    payload = {
        "schema": "situation3d.wargame_core.demo.v1",
        "scenario_path": os.path.abspath(args.scenario),
        "map_path": os.path.abspath(map_path),
        "seed": args.seed,
        "tick_seconds": args.tick_seconds,
        "result": result,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("written:", args.out)
    else:
        print(text)
    print("winner=%s reason=%s score=%s events=%d" % (
        result["winner"], result["reason"], result["score"],
        len(result["events"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
