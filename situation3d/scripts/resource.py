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
