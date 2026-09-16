#!/usr/bin/env python3
"""Generate the demonstration situation datasets consumed by the web front end.

One dataset is written per registered demo case, plus a ``cases.json`` index
that drives the case menu.

Usage (from the repository root):

    python3 situation3d/scripts/gen_demo_trajectory.py            # every case
    python3 situation3d/scripts/gen_demo_trajectory.py sea_convoy # one case

Writes ``situation3d/web/data/<case_id>.json`` and ``.../cases.json``.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "situation3d"))

from engine import cases, scenario  # noqa: E402


def write_json(path: str, payload) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(path)


def main(argv) -> int:
    wanted = argv[1:] or [c.case_id for c in cases.CASES]
    unknown = [c for c in wanted if c not in cases.CASES_BY_ID]
    if unknown:
        print("ERROR: unknown case(s): %s" % ", ".join(unknown), file=sys.stderr)
        return 2

    out_dir = os.path.join(REPO, "situation3d", "web", "data")
    os.makedirs(out_dir, exist_ok=True)

    for case_id in wanted:
        case = cases.get_case(case_id)
        dataset = scenario.build_dataset(case)
        out_path = os.path.join(out_dir, case_id + ".json")
        size = write_json(out_path, dataset)

        summary = dataset["inference_summary"]
        print("case        : %s  (%s)" % (case_id, case.name))
        print("  targets   : %d  (sea/land/air = %s)"
              % (len(dataset["targets"]), _domain_counts(dataset["targets"])))
        print("  hotspots  : %d" % len(dataset["hotspots"]))
        print("  frames    : %d  (tick=%ss, total=%sh)"
              % (len(dataset["frames"]), dataset["tick_seconds"],
                 len(dataset["frames"]) * dataset["tick_seconds"] / 3600.0))
        print("  inference : peak=%s (tick %d), mean threat=%.1f, histogram=%s"
              % (summary["peak_level_label"], summary["peak_tick"],
                 summary["mean_threat"], summary["level_histogram"]))
        print("  written   : %s (%.1f KB)" % (out_path, size / 1024.0))

    index_path = os.path.join(out_dir, "cases.json")
    write_json(index_path, cases.index())
    print("index       : %s" % index_path)
    return 0


def _domain_counts(targets) -> str:
    counts = {}
    for t in targets:
        counts[t["domain"]] = counts.get(t["domain"], 0) + 1
    return "/".join("%s=%d" % (d, counts.get(d, 0)) for d in ("sea", "land", "air"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
