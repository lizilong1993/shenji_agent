#!/usr/bin/env python3
"""Generate the demonstration situation dataset consumed by the web front end.

Usage (run on the remote container, from the repository root):

    python3 situation3d/scripts/gen_demo_trajectory.py

Writes ``situation3d/web/data/demo_taiwan_strait.json``.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "situation3d"))

from engine import scenario  # noqa: E402


def main() -> int:
    dataset = scenario.build_dataset()

    out_dir = os.path.join(REPO, "situation3d", "web", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_taiwan_strait.json")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024.0
    frames = len(dataset["frames"])
    targets = len(dataset["targets"])

    print("scenario_id : %s" % dataset["scenario_id"])
    print("region      : %s" % dataset["region"])
    print("targets     : %d" % targets)
    print("frames      : %d  (tick=%ss, total=%sh)"
          % (frames, dataset["tick_seconds"], frames * dataset["tick_seconds"] / 3600.0))
    print("hotspots    : %d" % len(dataset["hotspots"]))
    print("written     : %s (%.1f KB)" % (out_path, size_kb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
