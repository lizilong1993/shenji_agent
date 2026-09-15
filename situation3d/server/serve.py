#!/usr/bin/env python3
"""Static server for the situation3d web front end.

No third-party dependencies — uses only the Python standard library so it runs
on the remote container's system python3 (3.8).

Usage:
    python3 situation3d/server/serve.py --port 8848
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
except ImportError:  # pragma: no cover - python2 guard
    sys.stderr.write("python3 is required\n")
    raise SystemExit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.join(os.path.dirname(HERE), "web")

EXTRA_TYPES = {
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".xml": "application/xml",
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_ROOT, **kwargs)

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in EXTRA_TYPES:
            return EXTRA_TYPES[ext]
        return super().guess_type(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] %s\n" % (fmt % args))


def main() -> int:
    ap = argparse.ArgumentParser(description="situation3d static server")
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    if not os.path.isdir(WEB_ROOT):
        sys.stderr.write("web root not found: %s\n" % WEB_ROOT)
        return 1

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write("serving %s on http://%s:%d/\n" % (WEB_ROOT, args.host, args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
