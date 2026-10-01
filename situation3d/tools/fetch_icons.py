#!/usr/bin/env python3
"""Fetch the game-icons bundle used for the platform glyphs inside the frames.

`tools/gen_symbols.py` reads the icon path data straight out of the tarball, so
this only has to download it once. Standard library only -- the deployment
container has no node/npm and no pip.

Licence: **CC-BY-3.0** (icons by the game-icons.net contributors). Attribution is
required and is recorded in `web/assets/symbols/ATTRIBUTION.md`, which
gen_symbols.py writes.

Usage:
    python3 tools/fetch_icons.py [--dest /tmp/iconprobe/gi.tgz]
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

# Pinned: a different version would change the glyph shapes, so the generated
# SVGs must be regenerated in the same commit.
VERSION = "1.2.4"
URL = ("https://registry.npmjs.org/@iconify-json/game-icons/-/"
       "game-icons-%s.tgz" % VERSION)
DEFAULT_DEST = "/tmp/iconprobe/gi.tgz"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dest) and not args.force:
        print("[icons] 已存在，跳过：%s（%d 字节）"
              % (args.dest, os.path.getsize(args.dest)))
        return 0

    d = os.path.dirname(args.dest)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    print("[icons] 下载 %s" % URL)
    try:
        with urllib.request.urlopen(URL, timeout=120) as resp:
            blob = resp.read()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("[icons] 下载失败：%s\n"
                         "  需要能访问 registry.npmjs.org；"
                         "也可手工下载后放到 --dest 指定的路径。\n" % exc)
        return 1
    with open(args.dest, "wb") as fh:
        fh.write(blob)
    print("[icons] 完成：%s（%d 字节）" % (args.dest, len(blob)))
    return 0


if __name__ == "__main__":
    sys.exit(main())