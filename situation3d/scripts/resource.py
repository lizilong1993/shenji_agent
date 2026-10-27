#!/usr/bin/env python3
"""Resource registry CLI (P0).

Examples:
    python3 scripts/resource.py scan
    python3 scripts/resource.py list --kind envpack
    python3 scripts/resource.py verify --all
    python3 scripts/resource.py pin envpack:taiwan-tw2-v1
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

from resources import providers  # noqa: E402
from resources.acquire import (AcquireError, acquire_tiles,  # noqa: E402
                               probe_report)
from resources.adapters import (acquire_buildings, acquire_canopy,  # noqa: E402
                                acquire_landcover, acquire_sentinel,
                                DatasetError)
from resources.vectorproc import DatasetSpec  # noqa: E402
from resources.enrichment import enrich_envpack  # noqa: E402
from resources.envpack_layers import (add_canopy_layer,  # noqa: E402
                                      add_landcover_layer)
from resources.processing import TilesetSpec  # noqa: E402
from resources.sources import raster_candidate  # noqa: E402
from resources.catalog import KINDS  # noqa: E402
from resources.scanners import scan_repo  # noqa: E402
from resources.store import ResourceError, ResourceStore  # noqa: E402


def _store(args):
    return ResourceStore(args.root)


def _print_entry(entry, verbose=False):
    line = "%-34s %-12s %s  %s  %s" % (
        entry.entry_id, entry.kind,
        entry.as_of or "-", entry.quality,
        entry.source[:60])
    print(line)
    if verbose:
        print("  path:    %s" % entry.path)
        print("  hash:    %s" % entry.content_hash)
        print("  version: %s  schema: %s  size: %d"
              % (entry.version, entry.schema, entry.size))
        print("  fetched: %s  license: %s  pinned: %s"
              % (entry.fetched_at or "-", entry.license or "-", entry.pinned))


def cmd_scan(args):
    entries, errors = scan_repo(SITUATION3D)
    store = _store(args)
    added = updated = skipped = 0
    for entry in entries:
        try:
            old = store.get(entry.entry_id)
        except ResourceError:
            old = None
        store.register(entry, replace=True)
        if old is None:
            added += 1
        elif old.content_hash != entry.content_hash:
            updated += 1
        else:
            skipped += 1
    print(json.dumps({"scanned": len(entries), "added": added,
                      "updated": updated, "unchanged": skipped,
                      "errors": errors}, ensure_ascii=False, indent=1))
    return 1 if errors else 0


def cmd_list(args):
    rows = _store(args).entries(args.kind)
    if args.json:
        print(json.dumps([e.as_dict() for e in rows], ensure_ascii=False, indent=1))
        return 0
    if not rows:
        print("(no resources registered)")
        return 0
    for entry in rows:
        _print_entry(entry, verbose=args.long)
    return 0


def cmd_show(args):
    entry = _store(args).get(args.entry_id)
    print(json.dumps(entry.as_dict(), ensure_ascii=False, indent=1))
    return 0


def cmd_verify(args):
    store = _store(args)
    if args.entry_id:
        result = store.verify(args.entry_id)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0 if result["ok"] else 1
    result = store.verify_all(args.kind)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        for item in result["results"]:
            print("%-34s %s" % (item["entry_id"], item["status"]))
        print("checked=%d ok=%s" % (result["checked"], result["ok"]))
    return 0 if result["ok"] else 1


def cmd_pin(args):
    entry = _store(args).set_pin(args.entry_id, not args.unpin)
    print(json.dumps({"entry_id": entry.entry_id, "pinned": entry.pinned}))
    return 0


def cmd_remove(args):
    _store(args).remove(args.entry_id, force=args.force)
    print(json.dumps({"removed": args.entry_id}))
    return 0


def _progress(done, total):
    if done % (50 << 20) < (1 << 20):
        print("[download] %.1f MB%s" % (
            done / 1048576.0,
            (" / %.1f MB" % (int(total) / 1048576.0)) if total else ""),
            file=sys.stderr, flush=True)


def cmd_fetch(args):
    store = _store(args)
    entry = store.get(args.entry_id)
    cache_root = args.cache_root or os.path.join(str(store.root), "cache")
    pbf = args.osm_pbf
    try:
        if args.pbf_url:
            name = os.path.basename(args.pbf_url.split("?")[0]) \
                or "download.osm.pbf"
            dest = providers.pbf_cache_path(cache_root, name)
            print("downloading %s -> %s" % (args.pbf_url, dest),
                  file=sys.stderr, flush=True)
            providers.download_osm_pbf(args.pbf_url, dest,
                                       timeout_s=args.pbf_timeout,
                                       progress=_progress)
            pbf = dest
        if pbf and not os.path.isfile(pbf):
            print("error: osm pbf not found: %s" % pbf, file=sys.stderr)
            return 1
        new = providers.refresh_envpack(
            store, entry, cache_root, mode=args.mode,
            timeout_s=args.timeout, osm_pbf=pbf,
            allow_degraded=args.allow_degraded)
    except providers.ProviderError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - download/network failures
        print("error: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print(json.dumps(new.as_dict(), ensure_ascii=False, indent=1))
    return 0


def cmd_download_pbf(args):
    store = _store(args)
    cache_root = args.cache_root or os.path.join(str(store.root), "cache")
    name = os.path.basename(args.url.split("?")[0]) or "download.osm.pbf"
    dest = args.out or providers.pbf_cache_path(cache_root, name)
    try:
        providers.download_osm_pbf(args.url, dest, timeout_s=args.timeout,
                                   progress=_progress)
    except Exception as exc:  # noqa: BLE001
        print("error: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print(json.dumps({"path": dest, "size": os.path.getsize(dest)}))
    return 0


def cmd_provide(args):
    store = _store(args)
    try:
        entry = providers.register_provided_envpack(store, args.path,
                                                    args.entry_id)
    except providers.ProviderError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(entry.as_dict(), ensure_ascii=False, indent=1))
    return 0


def _parse_bbox(text):
    values = [float(v) for v in str(text).split(",")]
    if len(values) != 4:
        raise ValueError("bbox must be west,south,east,north")
    return tuple(values)


def _parse_zooms(text):
    value = str(text).strip()
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    return int(value), int(value)


def _spec_from_args(args, layer=None):
    lo, hi = _parse_zooms(args.zooms)
    return TilesetSpec(name=args.name, bbox=_parse_bbox(args.bbox),
                       min_zoom=lo, max_zoom=hi,
                       layer=(layer or getattr(args, "layer", "imagery")))


def cmd_tiles(args):
    store = _store(args)
    try:
        spec = _spec_from_args(args)
        sources = [s.strip() for s in (args.sources or "").split(",") if s.strip()]
        result = acquire_tiles(
            store, spec, mode=args.mode,
            source_ids=sources or None,
            raster_urls=args.raster_url or None,
            min_bps=args.min_bps, force=args.force,
            workers=args.workers, probe_timeout_s=args.probe_timeout,
            timeout_s=args.timeout)
    except (AcquireError, KeyError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=1))
    else:
        manifest = result.manifest or {}
        print("tileset: %s" % (result.entry.entry_id if result.entry else spec.name))
        print("  path:    %s" % (result.entry.path if result.entry else "-"))
        print("  source:  %s" % (result.source.get("source_id"),
                                 result.source.get("method")))
        print("  cached:  %s  tiles: %s  nonempty: %s" % (
            result.cached, manifest.get("tile_count"),
            manifest.get("nonempty_tiles")))
        print("  elapsed: %.1f s" % result.elapsed_s)
        for attempt in result.attempts:
            print("  attempt: %s probe=%sms speed=%.0fKB/s result=%s" % (
                attempt.get("source_id"), attempt.get("latency_ms"),
                float(attempt.get("speed_bps") or 0) / 1024.0,
                attempt.get("result")))
    return 0


def cmd_probe_tiles(args):
    try:
        spec = _spec_from_args(args)
        sources = [s.strip() for s in (args.sources or "").split(",") if s.strip()]
        rows = probe_report(spec, sources or None, args.raster_url or None,
                            timeout_s=args.probe_timeout)
    except (KeyError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    for row in rows:
        print("%-4s %-12s %-6s ok=%-5s latency=%8.1fms speed=%8.0fKB/s %s" % (
            row.get("rank") or "-", row.get("source_id"), row.get("method"),
            row.get("ok"), float(row.get("latency_ms") or 0),
            float(row.get("speed_bps") or 0) / 1024.0,
            row.get("detail") or ""))
    return 0


def cmd_raster_tiles(args):
    store = _store(args)
    try:
        spec = _spec_from_args(args, layer=args.layer)
        result = acquire_tiles(
            store, spec, mode=args.mode,
            candidates=[raster_candidate(args.url)],
            force=args.force, workers=args.workers,
            probe_timeout_s=args.probe_timeout, timeout_s=args.timeout)
    except (AcquireError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps({
        "entry": result.entry.as_dict() if result.entry else None,
        "manifest": result.manifest,
        "cached": result.cached,
        "attempts": result.attempts,
        "elapsed_s": result.elapsed_s,
    }, ensure_ascii=False, indent=1))
    return 0


def cmd_buildings(args):
    store = _store(args)
    try:
        spec = DatasetSpec(name=args.name, bbox=_parse_bbox(args.bbox),
                           kind="building_footprints")
        sources = [s.strip() for s in (args.sources or "gba,overture").split(",")
                   if s.strip()]
        result = acquire_buildings(
            store, spec, sources,
            args.cache_root or os.path.join(str(store.root), "cache"),
            mode=args.mode, force=args.force)
    except (DatasetError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps({
        "entry": result["entry"].as_dict(),
        "manifest": result["manifest"],
        "stats": result["stats"],
        "source": result["source"],
        "errors": result["errors"],
    }, ensure_ascii=False, indent=1))
    return 0


def cmd_canopy(args):
    store = _store(args)
    try:
        spec = DatasetSpec(name=args.name, bbox=_parse_bbox(args.bbox),
                           kind="canopy_height",
                           resolution_m=args.resolution)
        result = acquire_canopy(
            store, spec,
            args.cache_root or os.path.join(str(store.root), "cache"),
            mode=args.mode, resolution_m=args.resolution, force=args.force)
    except (DatasetError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps({"entry": result["entry"].as_dict(),
                      "manifest": result["manifest"],
                      "source": result["source"]},
                     ensure_ascii=False, indent=1))
    return 0


def cmd_enrich(args):
    store = _store(args)
    try:
        env = store.get(args.envpack)
        buildings = store.get(args.buildings)
        result = enrich_envpack(store, env, buildings, name=args.name,
                                cache_root=args.cache_root)
    except Exception as exc:  # noqa: BLE001
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps({
        "entry": result["entry"].as_dict(),
        "stats": result["stats"],
    }, ensure_ascii=False, indent=1))
    return 0


def cmd_landcover(args):
    store = _store(args)
    try:
        spec = DatasetSpec(name=args.name, bbox=_parse_bbox(args.bbox),
                           kind="landcover")
        result = acquire_landcover(
            store, spec,
            args.cache_root or os.path.join(str(store.root), "cache"),
            mode=args.mode)
    except (DatasetError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps({"entry": result["entry"].as_dict(),
                      "manifest": result["manifest"],
                      "source": result["source"]},
                     ensure_ascii=False, indent=1))
    return 0


def cmd_add_canopy(args):
    store = _store(args)
    try:
        env = store.get(args.envpack)
        canopy = store.get(args.canopy)
        result = add_canopy_layer(
            store, env, canopy, name=args.name,
            resolution_m=args.resolution, cache_root=args.cache_root)
    except Exception as exc:  # noqa: BLE001
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps({"entry": result["entry"].as_dict(),
                      "grid": result["grid"]}, ensure_ascii=False, indent=1))
    return 0


def cmd_add_landcover(args):
    store = _store(args)
    try:
        env = store.get(args.envpack)
        lc = store.get(args.landcover)
        result = add_landcover_layer(
            store, env, lc, name=args.name,
            resolution_m=args.resolution, cache_root=args.cache_root)
    except Exception as exc:  # noqa: BLE001
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps({"entry": result["entry"].as_dict(),
                      "grid": result["grid"]}, ensure_ascii=False, indent=1))
    return 0


def cmd_sentinel(args):
    store = _store(args)
    try:
        spec = DatasetSpec(name=args.name, bbox=_parse_bbox(args.bbox),
                           kind="satellite_imagery",
                           resolution_m=args.resolution,
                           query={"date": args.date,
                                  "window_days": args.window_days,
                                  "max_cloud": args.max_cloud})
        result = acquire_sentinel(
            store, spec,
            args.cache_root or os.path.join(str(store.root), "cache"),
            target_date=args.date, window_days=args.window_days,
            max_cloud=args.max_cloud, resolution_m=args.resolution,
            force=args.force)
    except (DatasetError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps({"entry": result["entry"].as_dict(),
                      "manifest": result["manifest"],
                      "source": result["source"],
                      "cached": result.get("cached", False)},
                     ensure_ascii=False, indent=1))
    return 0


def cmd_kinds(args):
    rows = [KINDS[k].as_dict() for k in sorted(KINDS)]
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None,
                    help="index root (default: situation3d/var/resources)")
    sub = ap.add_subparsers(dest="command")

    p = sub.add_parser("scan", help="register artifacts found under situation3d")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("list", help="list registered resources")
    p.add_argument("--kind", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--long", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show one entry as JSON")
    p.add_argument("entry_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("verify", help="verify one entry or all entries")
    p.add_argument("entry_id", nargs="?")
    p.add_argument("--kind", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("pin", help="pin an entry")
    p.add_argument("entry_id")
    p.set_defaults(func=cmd_pin, unpin=False)

    p = sub.add_parser("unpin", help="unpin an entry")
    p.add_argument("entry_id")
    p.set_defaults(func=cmd_pin, unpin=True)

    p = sub.add_parser("remove", help="remove an entry from the index")
    p.add_argument("entry_id")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("fetch", help="refresh an envpack from online sources")
    p.add_argument("entry_id")
    p.add_argument("--mode", choices=("online", "auto", "offline"),
                   default="auto")
    p.add_argument("--cache-root", default=None)
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--osm-pbf", default=None,
                   help="build from a local .osm.pbf (needs pyosmium)")
    p.add_argument("--pbf-url", default=None,
                   help="download this .osm.pbf first, then build")
    p.add_argument("--pbf-timeout", type=int, default=900)
    p.add_argument("--allow-degraded", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("download-pbf",
                       help="download a public .osm.pbf into the cache")
    p.add_argument("url")
    p.add_argument("--out", default=None)
    p.add_argument("--cache-root", default=None)
    p.add_argument("--timeout", type=int, default=900)
    p.set_defaults(func=cmd_download_pbf)

    p = sub.add_parser("provide", help="register a user-supplied envpack")
    p.add_argument("path")
    p.add_argument("--entry-id", default=None)
    p.set_defaults(func=cmd_provide)

    p = sub.add_parser("buildings",
                       help="acquire building footprints (GBA/Overture)")
    p.add_argument("--name", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--sources", default="gba,overture")
    p.add_argument("--mode", choices=("auto", "offline", "online"),
                   default="auto")
    p.add_argument("--cache-root", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_buildings)

    p = sub.add_parser("canopy", help="acquire CHMv2 canopy height raster")
    p.add_argument("--name", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--resolution", type=float, default=1.2)
    p.add_argument("--mode", choices=("auto", "offline", "online"),
                   default="auto")
    p.add_argument("--cache-root", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_canopy)

    p = sub.add_parser("landcover",
                       help="acquire ESA WorldCover land-cover raster")
    p.add_argument("--name", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--mode", choices=("auto", "offline", "online"),
                   default="auto")
    p.add_argument("--cache-root", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_landcover)

    p = sub.add_parser("sentinel",
                       help="acquire a Sentinel-2 L2A true-colour raster")
    p.add_argument("--name", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--date", default=None,
                   help="ISO date/datetime (default: now)")
    p.add_argument("--window-days", type=int, default=60)
    p.add_argument("--max-cloud", type=float, default=30.0)
    p.add_argument("--resolution", type=float, default=10.0)
    p.add_argument("--cache-root", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_sentinel)

    p = sub.add_parser("tiles", help="acquire an XYZ tileset from parallel sources")
    p.add_argument("--name", required=True)
    p.add_argument("--bbox", required=True,
                   help="west,south,east,north")
    p.add_argument("--zooms", default="6-10", help="min-max or single zoom")
    p.add_argument("--layer", choices=("imagery", "terrain"), default="imagery")
    p.add_argument("--sources", default="esri,eox-s2,gibs-modis")
    p.add_argument("--raster-url", action="append", default=None,
                   help="also consider a full raster URL (repeatable)")
    p.add_argument("--mode", choices=("auto", "offline", "online"),
                   default="auto")
    p.add_argument("--min-bps", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--probe-timeout", type=float, default=5.0)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_tiles)

    p = sub.add_parser("probe-tiles",
                       help="probe tile sources in parallel and rank them")
    p.add_argument("--name", default="probe")
    p.add_argument("--bbox", required=True)
    p.add_argument("--zooms", default="6-10")
    p.add_argument("--layer", choices=("imagery", "terrain"), default="imagery")
    p.add_argument("--sources", default="esri,eox-s2,gibs-modis")
    p.add_argument("--raster-url", action="append", default=None)
    p.add_argument("--probe-timeout", type=float, default=5.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_probe_tiles)

    p = sub.add_parser("raster-tiles",
                       help="download a full raster and tile it locally")
    p.add_argument("--name", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--zooms", default="6-10")
    p.add_argument("--layer", choices=("imagery", "terrain"), default="terrain")
    p.add_argument("--mode", choices=("auto", "offline", "online"),
                   default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--probe-timeout", type=float, default=6.0)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_raster_tiles)

    p = sub.add_parser("enrich-envpack",
                       help="fill envpack default building heights from a footprint resource")
    p.add_argument("--envpack", required=True)
    p.add_argument("--buildings", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--cache-root", default=None)
    p.set_defaults(func=cmd_enrich)

    p = sub.add_parser("add-canopy",
                       help="add a canopy grid layer to an envpack")
    p.add_argument("--envpack", required=True)
    p.add_argument("--canopy", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--resolution", type=float, default=30.0)
    p.add_argument("--cache-root", default=None)
    p.set_defaults(func=cmd_add_canopy)

    p = sub.add_parser("add-landcover",
                       help="add a WorldCover class grid layer to an envpack")
    p.add_argument("--envpack", required=True)
    p.add_argument("--landcover", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--resolution", type=float, default=10.0)
    p.add_argument("--cache-root", default=None)
    p.set_defaults(func=cmd_add_landcover)

    p = sub.add_parser("kinds", help="list resource kinds")
    p.set_defaults(func=cmd_kinds)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    try:
        return args.func(args)
    except ResourceError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
