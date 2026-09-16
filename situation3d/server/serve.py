#!/usr/bin/env python3
"""Static server for the situation3d web front end.

No third-party dependencies -- uses only the Python standard library so it runs
on the remote container's system python3 (3.8).

Why this is more than `python -m http.server`
---------------------------------------------
The asset set is roughly 10k tiles and the per-case situation data is several
megabytes. Two things dominate load time at that size, and both are decided
here rather than in the browser:

* **Caching.** Tiles are immutable, so they are served with a long max-age and
  an ETag. The original version sent `no-store` on everything, which forced a
  full re-download of every tile on every reload.
* **Compression.** The situation JSON is repetitive and gzips to about a sixth
  of its size. Text responses are gzipped when the client asks for it.

Source files (html/js/css) keep `no-store` so edits show up on reload.

Usage:
    python3 situation3d/server/serve.py --port 8848
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import sys
import time

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
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".xml": "application/xml",
}

# Compressible text payloads above this size.
COMPRESS_MIN_BYTES = 1024
COMPRESSIBLE = ("application/json", "application/javascript", "text/html",
                "text/css", "image/svg+xml", "application/xml")

# Immutable asset roots: everything under web/tiles plus the vendored Cesium
# bundle. These never change without a re-fetch, so they can be cached hard.
IMMUTABLE_PREFIXES = ("/tiles/", "/vendor/")
IMMUTABLE_MAX_AGE = 7 * 24 * 3600

# Shared across handler instances. BaseHTTPRequestHandler runs the whole request
# from inside __init__ (setup -> handle -> finish), so anything assigned after
# the super().__init__ call does not exist yet when the request is served --
# instance state therefore has to live in class-level or module-level objects.
_ETAG_CACHE = {}


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "situation3d/1.1"

    def __init__(self, *args, **kwargs):
        # `directory` has to be passed into super().__init__ rather than set
        # afterwards: BaseHTTPRequestHandler serves the request from inside
        # __init__, so anything assigned after the super call happens too late.
        kwargs["directory"] = WEB_ROOT
        super().__init__(*args, **kwargs)

    # -- content type ----------------------------------------------------

    def guess_type(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in EXTRA_TYPES:
            return EXTRA_TYPES[ext]
        return super().guess_type(path)

    # -- helpers ---------------------------------------------------------

    def _is_immutable(self):
        return self.path.startswith(IMMUTABLE_PREFIXES)

    @staticmethod
    def _etag_for(fs_path, st):
        """Cheap ETag: size + mtime, which is enough for a build-time asset set."""
        key = (st.st_size, int(st.st_mtime))
        cached = _ETAG_CACHE.get(fs_path)
        if cached and cached[0] == key:
            return cached[1]
        raw = "%d-%d" % key
        tag = '"%s"' % hashlib.md5(raw.encode("ascii")).hexdigest()[:16]
        _ETAG_CACHE[fs_path] = (key, tag)
        return tag

    # -- responses -------------------------------------------------------

    def end_headers(self):
        # Cache policy is per-path, so it is decided here rather than in a
        # blanket header.
        if self._is_immutable():
            self.send_header("Cache-Control", "public, max-age=%d, immutable"
                             % IMMUTABLE_MAX_AGE)
        else:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        try:
            return self._send_head()
        except Exception:  # noqa: BLE001 - never leave the client with no reply
            import traceback
            traceback.print_exc()
            try:
                self.send_error(500, "internal error")
            except Exception:  # noqa: BLE001
                pass
            return None

    def _send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        try:
            st = os.stat(path)
        except OSError:
            return super().send_head()

        etag = self._etag_for(path, st)
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        ctype = self.guess_type(path)
        try:
            fh = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        # Decide on compression before writing any headers.
        gz_len = None
        wants_gzip = "gzip" in (self.headers.get("Accept-Encoding") or "")
        compressible = any(ctype.startswith(t) for t in COMPRESSIBLE)
        if wants_gzip and compressible and st.st_size >= COMPRESS_MIN_BYTES:
            raw = fh.read()
            fh.close()
            buf = io.BytesIO()
            # mtime=0 keeps the gzip output byte-stable for a given input, so
            # the ETag stays meaningful across restarts.
            with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
                gz.write(raw)
            body = buf.getvalue()
            fh = io.BytesIO(body)
            gz_len = len(body)

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if etag:
            self.send_header("ETag", etag)
        self.send_header("Vary", "Accept-Encoding")
        if gz_len is not None:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(gz_len))
        else:
            self.send_header("Content-Length", str(st.st_size))
        self.send_header("Last-Modified", self.date_time_string(int(st.st_mtime)))
        self.end_headers()
        return fh

    def log_message(self, fmt, *args):
        # Tile traffic is thousands of lines; only log non-asset requests.
        if not self._is_immutable():
            sys.stderr.write("[serve] %s\n" % (fmt % args))


class Server(ThreadingHTTPServer):
    """Threaded server sized for a tile-heavy client.

    The default accept backlog is 5. A browser opens up to six sockets per host
    and Cesium fires dozens of tile requests at once, so connections were being
    dropped at the door -- which the page sees as `Failed to fetch` and silently
    degrades to a coarser basemap.
    """

    daemon_threads = True
    request_queue_size = 256
    allow_reuse_address = True


def main() -> int:
    ap = argparse.ArgumentParser(description="situation3d static server")
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    if not os.path.isdir(WEB_ROOT):
        sys.stderr.write("web root not found: %s\n" % WEB_ROOT)
        return 1

    httpd = Server((args.host, args.port), Handler)
    sys.stderr.write("serving %s on http://%s:%d/  (started %s)\n"
                     % (WEB_ROOT, args.host, args.port,
                        time.strftime("%Y-%m-%d %H:%M:%S")))
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
