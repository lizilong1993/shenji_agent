#!/usr/bin/env python3
"""P3 perception calibration harness: analytic p vs Monte-Carlo + ROC/AUC.

Usage:
    python3 scripts/calibrate_perception.py [--seeds 2000] [--out calib.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from env.service import FlatEnvironment  # noqa: E402
from mobility.profile import PROFILES  # noqa: E402
from sim import perception  # noqa: E402
from sim.entity import Entity, Pose  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = FlatEnvironment()
    obs = Entity(1, "observer", 0, "land", "tracked", Pose(120.0, 23.0))
    obs.profile = PROFILES["tracked"]
    tgt = Entity(2, "target", 1, "land", "tracked", Pose(120.0, 23.02))
    tgt.profile = PROFILES["tracked"]
    spec = perception.specs_for(obs)[0]

    rows = []
    pos = []
    neg = []
    for distance in range(100, 4600, 100):
        p = perception.detection_probability(
            spec, obs, tgt, float(distance), env)
        hits = sum(1 for seed in range(args.seeds)
                   if perception.detection_roll(seed, 0, 1, 2) < p)
        empirical = hits / float(args.seeds)
        rows.append({"distance_m": distance, "p": round(p, 4),
                     "empirical": round(empirical, 4),
                     "abs_error": round(abs(p - empirical), 4)})
        (pos if distance <= 3000 else neg).append(p)
    wins = sum(1 for a in pos for b in neg if a > b)
    ties = sum(1 for a in pos for b in neg if a == b)
    auc = (wins + 0.5 * ties) / max(1, len(pos) * len(neg))
    summary = {
        "weapon_profile": "tracked optics",
        "seeds_per_point": args.seeds,
        "max_abs_error": max(r["abs_error"] for r in rows),
        "auc_present_le_3000m": round(auc, 4),
        "rows": rows,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
