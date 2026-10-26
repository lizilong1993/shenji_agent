#!/usr/bin/env python3
"""Fetch case-scoped VGI sources into one immutable IntelBundle.

The case profile supplies the AOI, source allow-list, domains and optional
text/country terms.  CLI flags override the profile for one run; the same
profile can be attached to ``IntelGateway`` for runtime filtering.

Examples:
    python3 scripts/fetch_intel.py --case taiwan_landing
    python3 scripts/fetch_intel.py --case india_china_west --hours 48
    python3 scripts/fetch_intel.py --case global --sources usgs,gdacs,eonet
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

from intel.case import list_case_profiles, load_case_profile  # noqa: E402
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


def _terms(text: str):
    return tuple(item.strip() for item in str(text).split(",")
                 if item.strip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="global",
                        help="case profile id (%s)"
                        % ", ".join(list_case_profiles()))
    parser.add_argument("--profile", default=None,
                        help="path to a case profile JSON")
    parser.add_argument("--list-cases", action="store_true",
                        help="list available case profiles and exit")
    parser.add_argument("--sources", type=_sources, default=None,
                        help="override profile sources")
    parser.add_argument("--bbox", type=_bbox, default=None,
                        help="override profile bbox (use --bbox=-180,...)")
    parser.add_argument("--hours", type=float, default=None,
                        help="lookback hours; defaults to profile setting")
    parser.add_argument("--start", default=None,
                        help="window start (ISO 8601); overrides --hours")
    parser.add_argument("--end", default=None,
                        help="window end (ISO 8601); defaults to now")
    parser.add_argument("--domains", type=_terms, default=None,
                        help="override domains, comma separated")
    parser.add_argument("--keywords", type=_terms, default=None,
                        help="override keywords, comma separated")
    parser.add_argument("--countries", type=_terms, default=None,
                        help="override countries, comma separated")
    parser.add_argument("--buffer-km", type=float, default=None,
                        help="override AOI buffer in km")
    parser.add_argument("--max-items", type=int, default=None,
                        help="override maximum observations in the bundle")
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--archive-root", default=None)
    parser.add_argument("--health-out", default=None)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--allow-partial", action="store_true",
                        help="write the bundle even if a source failed")
    args = parser.parse_args(argv)

    if args.list_cases:
        print("\n".join(list_case_profiles()))
        return 0

    try:
        profile = load_case_profile(args.profile or args.case)
        profile = profile.with_overrides(
            case_id=None if args.profile else args.case,
            bbox=args.bbox,
            domains=args.domains,
            sources=args.sources,
            keywords=args.keywords,
            countries=args.countries,
            buffer_km=args.buffer_km,
            max_items=args.max_items,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("fetch_intel: %s" % (exc,), file=sys.stderr)
        return 2

    sources = args.sources or profile.sources or ("usgs", "gdacs", "eonet")
    now = datetime.now(timezone.utc)
    end = parse_time(args.end) if args.end else now
    hours = args.hours if args.hours is not None else profile.lookback_hours
    start = (parse_time(args.start) if args.start
             else end - timedelta(hours=max(0.0, hours)))
    base = os.path.join(SITUATION3D, "var", "intel", profile.case_id)
    out = args.out or os.path.join(base, "bundle.json")
    archive_root = args.archive_root or os.path.join(base, "archive")
    health_out = args.health_out or os.path.join(base, "health.json")

    try:
        result = build_bundle(
            source_ids=sources,
            bbox=profile.bbox,
            start=start,
            end=end,
            fetched_at=now,
            bundle_id=args.bundle_id,
            scenario_id=args.scenario_id or profile.case_id,
            archive_root=None if args.no_archive else archive_root,
            case_profile=profile,
        )
    except ValueError as exc:
        print("fetch_intel: %s" % (exc,), file=sys.stderr)
        return 2
    qualified = not result.errors or args.allow_partial
    bundle_out = out if qualified else out + ".partial"
    health_file = health_out if qualified else health_out + ".partial"
    _write_json(bundle_out, result.bundle.as_dict())
    _write_json(health_file, {
        "generated_at": iso(now),
        "case": profile.to_dict(),
        "case_filter_hash": profile.filter_hash,
        "bbox": list(profile.bbox),
        "expanded_bbox": list(profile.expanded_bbox),
        "time_window": [iso(start), iso(end)],
        "sources": [status.as_dict() for status in result.statuses],
        "archives": [record.as_dict() for record in result.archives],
        "errors": list(result.errors),
        "truncated": result.truncated,
    })
    summary = {
        "case_id": profile.case_id,
        "case_filter_hash": profile.filter_hash,
        "bundle_id": result.bundle.bundle_id,
        "hash": result.bundle.hash,
        "envelope_hash": result.bundle.envelope_hash,
        "observations": len(result.bundle.observations),
        "truncated": result.truncated,
        "out": bundle_out,
        "health_out": health_file,
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
