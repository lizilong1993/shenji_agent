# -*- coding: utf-8 -*-
"""Check target OBB overlaps in a generated situation.

Usage:
  PYTHONPATH=. python3 situation3d/tools/check_collisions.py \
      --case taiwan_dense_transfer_v1 [--swept] [--domains land]

This is a detector, not a resolver: it reports pairs so a lane-change or
navigation layer can decide what to do.  The shipped TW2 case intentionally
keeps the raw road movement, so overlaps at junctions are expected until that
layer is added.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "web", "data", "situations"))


def load_case(case_id: str) -> dict:
    path = os.path.join(DATA, case_id + ".json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="taiwan_dense_transfer_v1")
    ap.add_argument("--swept", action="store_true",
                    help="also sample between 30 s frames (slower)")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--domains", default="land")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from situation3d.engine import collision

    doc = load_case(args.case)
    targets = {t["id"]: t for t in doc.get("targets", [])}
    domains = tuple(d.strip() for d in args.domains.split(",") if d.strip())
    if args.swept:
        hits = collision.swept_collisions(
            doc.get("frames", []), targets, substeps=args.substeps, domains=domains)
    else:
        hits = []
        for frame in doc.get("frames", []):
            hits.extend(collision.frame_collisions(frame, targets, domains=domains))
    summary = collision.summarize(hits)
    summary["case"] = args.case
    summary["swept"] = bool(args.swept)
    summary["domains"] = list(domains)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("case=%s swept=%s domains=%s pairs=%d entries=%d"
              % (args.case, args.swept, ",".join(domains),
                 summary["pairs"], summary["entries"]))
        for row in summary["by_pair"][:20]:
            print("  %-8s %-8s first=%4d count=%4d max_depth=%.2fm"
                  % (row["a"], row["b"], row["first_tick"], row["count"],
                     row["max_depth_m"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
