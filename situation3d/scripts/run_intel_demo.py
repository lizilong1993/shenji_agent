#!/usr/bin/env python3
"""Run the geo engine with advisory JEV intent/behavior prediction.

The predictions are written to ``World.intel`` and included in
``observation(world, seat)``; they never feed rules or adjudication.

Usage:
    python3 scripts/run_intel_demo.py --case taiwan_landing --ticks 60 \
        --decision-every 10 --out /tmp/intel.json
    # real model (CPU; slow):
    PYTHONPATH=/tmp/jevlibs:$PYTHONPATH \
    python3 scripts/run_intel_demo.py --model /tmp/jev_model/qwen3.5-0.8b-nli-v2s-long ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from geoio.observation import observation  # noqa: E402
from policy.intent import BehaviorPredictor, IntentEstimator  # noqa: E402
from policy.jev import LocalOpenJevClient, MockJevClient  # noqa: E402
from sim.scenario import build_world  # noqa: E402


def _case(name: str):
    spec = importlib.util.spec_from_file_location(
        "intel_case", os.path.join(SITUATION3D, "campaigns", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    basic = json.loads(open(mod.map_path(), encoding="utf-8").read())
    return mod.scenario_dict(), basic


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="taiwan_landing")
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--decision-every", type=int, default=10)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    scenario, basic = _case(args.case)
    world = build_world(scenario, basic, seed=20261021, dt_s=30.0)
    client = LocalOpenJevClient(args.model) if args.model else MockJevClient()
    estimator = IntentEstimator(client, window=6)
    predictor = BehaviorPredictor(client)
    snapshots = []
    for i in range(args.ticks):
        if i % args.decision_every == 0:
            obs = observation(world, seat=args.seat)
            estimator.update(obs, world.events[-50:])
            intents = estimator.estimate(obs, args.seat)
            behaviors = predictor.predict(obs, args.seat)
            payload = {"time_s": round(world.time_s, 1),
                       "intent": estimator.format(intents),
                       "behavior": predictor.format(behaviors)}
            world.set_intel(args.seat, payload)
            snapshots.append(payload)
        world.step()
    result = {"case": args.case, "ticks": args.ticks,
              "model": args.model or "mock", "snapshots": snapshots}
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(json.dumps({"snapshots": len(snapshots),
                      "last": snapshots[-1] if snapshots else None},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
