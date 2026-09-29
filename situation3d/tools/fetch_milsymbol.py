#!/usr/bin/env python3
"""Fetch the pinned milsymbol bundle into web/vendor/milsymbol/.

milsymbol (https://github.com/spatialillusions/milsymbol, MIT) generates
MIL-STD-2525 / APP-6 compliant symbols as SVG. We use it **offline, once**, to
pre-generate the symbol SVGs that ship in web/assets/symbols/ -- see
tools/gen_symbols.py. The bundle itself is NOT shipped to the browser.

Why a fetch script instead of committing the file
-------------------------------------------------
Same reason as Cesium: `web/vendor/` is gitignored, and a 862 KB minified
third-party bundle does not belong in the repository. Pinning the version here
keeps regeneration reproducible.

Why Python and not npm
----------------------
The deployment container has **no node and no npm** (python3.8, stdlib only, no
pip). The npm registry is reachable, and a tarball is just gzip+tar, so stdlib
`tarfile` is enough. Nothing here needs a JS runtime -- the JS only has to run
once, in the browser (gen_symbols.py drives headless Chrome).

Usage:
    python3 tools/fetch_milsymbol.py            # fetch if missing
    python3 tools/fetch_milsymbol.py --force    # re-fetch
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import urllib.request

# Pinned. Bumping this changes the rendered symbols, so the generated SVGs in
# web/assets/symbols/ must be regenerated in the same commit -- gen_symbols.py
# records this version in its manifest, and a mismatch is reported as an error.
MILSYMBOL_VERSION = "3.0.4"
TARBALL = ("https://registry.npmjs.org/milsymbol/-/milsymbol-%s.tgz"
           % MILSYMBOL_VERSION)
# Only the CommonJS/UMD bundle is needed; src/ + source maps are ~4 MB of noise.
MEMBERS = {
    "package/dist/milsymbol.js": "milsymbol.js",
    "package/package.json": "package.json",
    "package/LICENSE": "LICENSE",
}

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(os.path.dirname(HERE), "web", "vendor", "milsymbol")


def fetch(force=False):
    target = os.path.join(VENDOR, "milsymbol.js")
    if os.path.exists(target) and not force:
        print("[milsymbol] 已存在，跳过：%s" % target)
        return 0

    print("[milsymbol] 下载 %s" % TARBALL)
    try:
        with urllib.request.urlopen(TARBALL, timeout=120) as resp:
            blob = resp.read()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("[milsymbol] 下载失败：%s\n" % exc)
        sys.stderr.write("  本机/容器需要能访问 registry.npmjs.org；"
                         "也可以手工下载后解出 dist/milsymbol.js 放到 %s\n" % VENDOR)
        return 1

    print("[milsymbol] %d 字节，解包中" % len(blob))
    try:
        tar = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
    except tarfile.TarError as exc:
        sys.stderr.write("[milsymbol] 不是有效的 tar.gz：%s\n" % exc)
        return 1

    if not os.path.isdir(VENDOR):
        os.makedirs(VENDOR)

    written = 0
    for member_name, out_name in MEMBERS.items():
        try:
            fh = tar.extractfile(member_name)
        except KeyError:
            sys.stderr.write("[milsymbol] 包里没有 %s（版本变了？）\n" % member_name)
            return 1
        if fh is None:
            continue
        with open(os.path.join(VENDOR, out_name), "wb") as out:
            out.write(fh.read())
        written += 1
        print("  -> %s" % out_name)

    # A marker so gen_symbols.py and the humans can both see what is installed.
    with open(os.path.join(VENDOR, "VERSION"), "w") as fh:
        fh.write(MILSYMBOL_VERSION + "\n")

    print("[milsymbol] 完成：%d 个文件写入 %s" % (written, VENDOR))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()
    return fetch(args.force)


if __name__ == "__main__":
    sys.exit(main())