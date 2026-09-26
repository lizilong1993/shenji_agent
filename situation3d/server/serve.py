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
import base64
import gzip
import hashlib
import io
import json
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

# --------------------------------------------------------------------------
# Cesium ion token
# --------------------------------------------------------------------------
# The online 3D-reality layer (Google Photorealistic 3D Tiles, ion asset
# 2275207) needs an ion access token. The token lives in web/data/ion.json,
# which the repo's `web/data/*.json` ignore rule already covers, so it never
# enters git. It is handed to the browser over these endpoints rather than
# being written into a tracked html/js file.
#
# Note this layer is **online only**: ion reports the asset as
# `archivable:false, exportable:false`, the tile URLs carry one-shot
# `?session=` tokens, and the response headers say `Cache-Control: private`.
# So nothing here is a caching proxy -- it only ships the credential.
ION_ASSET_ID = 2275207
ION_CONFIG_PATH = os.path.join(WEB_ROOT, "data", "ion.json")


def _b64url_decode(seg):
    """
    Decode one base64url JWT segment. Standard library only (no `jwt` package).
    """
    seg = seg.replace("-", "+").replace("_", "/")
    seg += "=" * (-len(seg) % 4)
    return base64.b64decode(seg.encode("ascii"))


def read_ion_token():
    """Return the configured token, or None. Never raises."""
    try:
        with open(ION_CONFIG_PATH, "r") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    token = data.get("token") if isinstance(data, dict) else None
    return token if isinstance(token, str) and token.strip() else None


def write_ion_token(token):
    """Persist the token. 0600 + best-effort directory: it is a credential."""
    d = os.path.dirname(ION_CONFIG_PATH)
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = ION_CONFIG_PATH + ".tmp"
    # Beware umask: create with restrictive mode, then write.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump({"token": token, "_comment":
                       "Cesium ion access token for ion asset %d "
                       "(Google Photorealistic 3D Tiles). Ignored by git." % ION_ASSET_ID,
                       "assetId": ION_ASSET_ID}, fh, indent=2)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, ION_CONFIG_PATH)
    try:
        os.chmod(ION_CONFIG_PATH, 0o600)
    except OSError:
        pass


def validate_ion_token(token):
    """Loose sanity check: shape, issuer, and not-expired.

    Deliberately does not call the ion API -- this only guards against a typo
    or an unrelated JWT being saved. The real check is ion answering 401 when
    the layer tries to load.
    """
    if not isinstance(token, str):
        return "令牌必须是字符串"
    parts = token.strip().split(".")
    if len(parts) != 3 or not all(parts):
        return "不是合法的 JWT（应为三段，用点分隔）"
    try:
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except Exception:  # noqa: BLE001 - any decode failure means "not a JWT"
        return "JWT 载荷无法解码"
    if not isinstance(payload, dict):
        return "JWT 载荷不是对象"
    iss = str(payload.get("iss") or "")
    if "cesium.com" not in iss:
        return "这个 JWT 的签发者不是 Cesium ion（iss=%s）" % (iss or "缺失")
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and exp < time.time():
        return "令牌已过期（exp=%d）" % int(exp)
    return None


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

    # -- ion token API ---------------------------------------------------

    def _is_api_path(self):
        return self.path.split("?")[0].startswith("/api/")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # HTTP/1.1 keep-alive: the body must be written to keep the socket
        # framing valid, exactly like _send_head does for files.
        self.wfile.write(body)

    def _read_json_body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if n <= 0 or n > 65536:
            return None
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    @staticmethod
    def _is_loopback(addr):
        return addr in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

    def _api_get(self, path):
        if path == "/api/ion/status":
            configured = read_ion_token() is not None
            self._send_json({
                "configured": configured,
                "rail": "ion",
                "assetId": ION_ASSET_ID,
                # Telling the browser *whether* a token exists is fine; never
                # send the value itself on the status endpoint.
                "hint": ("已配置令牌" if configured
                         else "未配置：在 图层管理 里填入 Cesium ion 令牌，"
                              "或把令牌写进 web/data/ion.json"),
            })
            return
        if path == "/api/ion/token":
            token = read_ion_token()
            if token is None:
                # After the browser saves a token it has it in localStorage, so
                # 404 here only means "server side has nothing", not "no token".
                self._send_json({"error": "服务端未配置令牌"}, 404)
                return
            self._send_json({"token": token, "assetId": ION_ASSET_ID})
            return
        self._send_json({"error": "未知接口"}, 404)

    def _api_post(self, path):
        if path != "/api/ion/token":
            self._send_json({"error": "未知接口"}, 404)
            return
        # The server binds 0.0.0.0, and this endpoint writes a credential, so
        # do not accept writes from the network -- only from the operator on
        # the box itself. Remote deployments seed web/data/ion.json instead.
        if not self._is_loopback(self.client_address[0]):
            self._send_json({
                "error": "仅允许本机写入令牌；远程请直接写 web/data/ion.json",
            }, 403)
            return
        body = self._read_json_body()
        token = (body or {}).get("token") if isinstance(body, dict) else None
        if not token:
            self._send_json({"error": "请求体缺少 token 字段"}, 400)
            return
        why = validate_ion_token(token)
        if why:
            self._send_json({"error": why}, 400)
            return
        try:
            write_ion_token(token.strip())
        except OSError as e:
            self._send_json({"error": "写入失败：%s" % e}, 500)
            return
        self._send_json({"ok": True, "assetId": ION_ASSET_ID})

    def do_GET(self):
        if self._is_api_path():
            try:
                self._api_get(self.path.split("?")[0])
            except Exception:  # noqa: BLE001 - never leave the client hanging
                import traceback
                traceback.print_exc()
                try:
                    self._send_json({"error": "内部错误"}, 500)
                except Exception:  # noqa: BLE001
                    pass
            return
        # Everything else is a static file: let send_head do the work. This
        # mirrors SimpleHTTPRequestHandler.do_GET exactly, minus a
        # double-handled HEAD (HEAD is routed to send_head by the base class).
        f = self.send_head()
        if f:
            try:
                self.copyfile(f, self.wfile)
            finally:
                f.close()

    def do_POST(self):
        if self._is_api_path():
            try:
                self._api_post(self.path.split("?")[0])
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                try:
                    self._send_json({"error": "内部错误"}, 500)
                except Exception:  # noqa: BLE001
                    pass
            return
        self.send_error(405, "Method Not Allowed")

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
