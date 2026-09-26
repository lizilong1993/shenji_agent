#!/usr/bin/env python3
"""Download and store a TLE catalogue for offline satellite-overpass queries.

Why download instead of querying live
-------------------------------------
Pass prediction needs the whole element set for the groups we care about, and
CelesTrak is reachable from the build machine but **not** from an isolated
deployment. So the elements are fetched once, slimmmed, and committed to the
tree -- the same offline-first pattern as the imagery, terrain and building
tiles. Runtime then reads a local file; no network, no API key.

Output layout::

    engine/data/tle/<group>.tle      raw 3-line records, verbatim
    engine/data/tle/catalog.json     {group: {fetched_utc, count, ...}} metadata

The raw files are kept verbatim (not converted to JSON) so they can be fed
straight back to `engine.sgp4.parse_tles` and diffed against a later download.

Notes
-----
* CelesTrak asks that clients not hammer it; this tool makes one request per
  group with a delay between them and sends no credentials.
* Element sets decay. `--max-age-days` (default 7) makes the tool refuse to
  treat an existing file as fresh, so a stale catalogue is visible rather than
  silently used for a prediction that is hours off.

Usage::

    python3 tools/fetch_tle.py --list
    python3 tools/fetch_tle.py                      # default groups
    python3 tools/fetch_tle.py --group stations --group weather
    python3 tools/fetch_tle.py --max-age-days 1     # force refresh

Standard library only: the target container has no pip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "engine", "data", "tle")

BASE = "https://celestrak.org/NORAD/elements/gp.php"

# Groups worth having for a Taiwan-Strait situational picture. Kept deliberately
# small: each group is a separate request and the whole set is well under 1 MB,
# so there is no reason to download the entire catalogue.
DEFAULT_GROUPS = ["resource", "geo", "weather", "science", "stations", "military"]

# CelesTrak rate-limits; one request per group with a pause is polite and also
# avoids being throttled mid-run.
DELAY_S = 2.0


def fetch_group(group: str, timeout: int = 40) -> str:
    url = "%s?GROUP=%s&FORMAT=tle" % (BASE, group)
    req = Request(url, headers={"User-Agent": "situation3d-tle-fetch/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", "replace")
    if "No GP data found" in text:
        raise ValueError("CelesTrak returned no data for group %r" % group)
    return text


def count_tles(text: str) -> int:
    """Count records without importing the engine (keeps this tool standalone)."""
    lines = [l for l in text.splitlines() if l.strip()]
    n = 0
    for i, line in enumerate(lines):
        if line.startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            n += 1
    return n


def age_days(path: str) -> float:
    if not os.path.exists(path):
        return float("inf")
    return (time.time() - os.path.getmtime(path)) / 86400.0


def load_catalog(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_catalog(path: str, catalog: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="Fetch and store TLE catalogues")
    ap.add_argument("--group", action="append", default=None,
                    help="group to fetch (repeatable); default: %s" % ",".join(DEFAULT_GROUPS))
    ap.add_argument("--out", default=OUT_DIR, help="output directory")
    ap.add_argument("--max-age-days", type=float, default=7.0,
                    help="re-download a group older than this (default 7)")
    ap.add_argument("--list", action="store_true", help="list available groups and exit")
    ap.add_argument("--force", action="store_true", help="ignore age and re-download")
    args = ap.parse_args(argv)

    if args.list:
        for g in DEFAULT_GROUPS:
            print(g)
        return 0

    groups = args.group or DEFAULT_GROUPS
    os.makedirs(args.out, exist_ok=True)
    catalog_path = os.path.join(args.out, "catalog.json")
    catalog = load_catalog(catalog_path)

    failures = []
    for i, group in enumerate(groups):
        path = os.path.join(args.out, "%s.tle" % group)
        age = age_days(path)
        if not args.force and age <= args.max_age_days:
            n = catalog.get(group, {}).get("count")
            print("%-12s fresh (%.1f d old, %s records) -- skipped"
                  % (group, age, n if n is not None else "?"))
            continue

        if i:
            time.sleep(DELAY_S)
        try:
            text = fetch_group(group)
        except (HTTPError, URLError, ValueError, OSError) as e:
            # A failed group must not discard the groups that did succeed.
            print("%-12s FAILED: %s" % (group, e), file=sys.stderr)
            failures.append((group, str(e)))
            continue

        n = count_tles(text)
        if n == 0:
            print("%-12s FAILED: no TLE records parsed" % group, file=sys.stderr)
            failures.append((group, "no records"))
            continue

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
        os.replace(tmp, path)

        catalog[group] = {
            "count": n,
            "bytes": os.path.getsize(path),
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "%s?GROUP=%s&FORMAT=tle" % (BASE, group),
        }
        print("%-12s %4d records  %6.1f KB  -> %s"
              % (group, n, os.path.getsize(path) / 1024.0, path))

    save_catalog(catalog_path, catalog)
    total = sum(v.get("count", 0) for v in catalog.values())
    print("catalog     : %s (%d groups, %d records)" % (catalog_path, len(catalog), total))
    if failures:
        print("FAILED groups: %s" % ", ".join(g for g, _ in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))