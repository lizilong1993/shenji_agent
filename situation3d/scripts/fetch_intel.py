#!/usr/bin/env python3
"""Fetch public VGI sources into one immutable IntelBundle.

This is the P0 external runtime: it pulls USGS / GDACS / FIRMS, archives raw
responses, normalizes observations and writes a bundle plus a source-health
file.  It never runs inside the simulation clock.

Usage:
    python3 scripts/fetch_intel.py --sources usgs,gdacs,firms \
        --bbox 119.0,21.5,123.0,26.5 --hours 24 \
        --out var/intel/bundle.json --archive-root var/intel/archive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

SITUATION3D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.connectors import build_bundle  # noqa: E402
from intel.model import iso, parse_time  # noqa: E402


def _write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _bbox(text: str):
    values = [float(item) for item in str(text).split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            "bbox must be min_lon,min_lat,max_lon,max_lat")
    return tuple(values)


def _sources(text: str):
    values = [item.strip().lower() for item in str(text).split(",")
              if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one source is required")
    return tuple(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=_sources,
                        default=("usgs", "gdacs"),
                        help="usgs,gdacs,firms (FIRMS also needs FIRMS_MAP_KEY)")
    parser.add_argument("--bbox", type=_bbox,
                        default=(119.0, 21.5, 123.0, 26.5))
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--start", default=None,
                        help="window start (ISO 8601); overrides --hours")
    parser.add_argument("--end", default=None,
                        help="window end (ISO 8601); defaults to now")
    parser.add_argument("--scenario-id", default="vgi_live")
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--out", default=os.path.join(
        SITUATION3D, "var", "intel", "bundle.json"))
    parser.add_argument("--archive-root", default=os.path.join(
        SITUATION3D, "var", "intel", "archive"))
    parser.add_argument("--health-out", default=os.path.join(
        SITUATION3D, "var", "intel", "health.json"))
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--allow-partial", action="store_true",
                        help="write the bundle even if a source failed")
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    end = parse_time(args.end) if args.end else now
    start = (parse_time(args.start) if args.start
             else end - timedelta(hours=max(0.0, args.hours)))
    try:
        result = build_bundle(
            source_ids=args.sources,
            bbox=args.bbox,
            start=start,
            end=end,
            fetched_at=now,
            bundle_id=args.bundle_id,
            scenario_id=args.scenario_id,
            archive_root=None if args.no_archive else args.archive_root,
        )
    except ValueError as exc:
        print("fetch_intel: %s" % (exc,), file=sys.stderr)
        return 2
    qualified = not result.errors or args.allow_partial
    bundle_out = args.out if qualified else args.out + ".partial"
    health_out = args.health_out if qualified else args.health_out + ".partial"
    _write_json(bundle_out, result.bundle.as_dict())
    _write_json(health_out, {
        "generated_at": iso(now),
        "bbox": list(args.bbox),
        "time_window": [iso(start), iso(end)],
        "sources": [status.as_dict() for status in result.statuses],
        "archives": [record.as_dict() for record in result.archives],
        "errors": list(result.errors),
    })
    summary = {
        "bundle_id": result.bundle.bundle_id,
        "hash": result.bundle.hash,
        "envelope_hash": result.bundle.envelope_hash,
        "observations": len(result.bundle.observations),
        "out": bundle_out,
        "health_out": health_out,
        "partial": bool(result.errors),
        "sources": {
            status.source_id: {
                "state": status.state,
                "count": status.count,
                "error": status.error,
            }
            for status in result.statuses
        },
        "errors": list(result.errors),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True))
    if not qualified:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
