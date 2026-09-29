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
import urllib.error
import urllib.request

try:
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
except ImportError:  # pragma: no cover - python2 guard
    sys.stderr.write("python3 is required\n")
    raise SystemExit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.join(os.path.dirname(HERE), "web")

# The engine package lives beside `server/`, and this file is run as a script
# (`python3 server/serve.py`), so the package root is not on sys.path by default.
# `/wfs` needs `engine.wfs`, so add it explicitly rather than relying on the
# caller's cwd.
_PKG_ROOT = os.path.dirname(HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

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
# 凭据：一律从 .env 读取
# --------------------------------------------------------------------------
# 第三方 key（天地图、Cesium ion）从仓库根目录的 `.env` 读，`.env` 进 .gitignore，
# 源码里**不留任何默认 key**。
#
# 为什么必须这么改：此前天地图 key 硬编码在 assets.js 里，配额耗尽后服务端返回
# 429，而代码对这个失败完全静默——整个底图变成一片深蓝，很久没人发现。凭据写死在
# 源码里还会随 git 泄漏，且换 key 要改代码。
#
# 优先级：**真实环境变量 > .env**。环境变量优先是为了让容器/CI 能在不落文件的情况下
# 覆盖，也让"临时用另一个 key 启动一次"不必动 .env。
#
# 只用标准库：运行环境是 python3.8 且没有 pip，装不了 python-dotenv。
ENV_PATH = os.path.join(os.path.dirname(HERE), ".env")
_ENV_CACHE = None


def _parse_env_file(path):
    """
    解析 KEY=VALUE。支持 `#` 注释、空行、`export ` 前缀、值两端成对的引号。

    格式不对的行**只告警不抛**：一个笔误不该让服务器起不来；把它打出来，
    比让整个服务崩掉更有用。
    """
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return out
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            print("[env] %s:%d 不是 KEY=VALUE，已跳过" % (path, lineno))
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # 只去掉成对引号，不做转义展开：可预测比"聪明"重要。
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            print("[env] %s:%d 键名为空，已跳过" % (path, lineno))
            continue
        out[key] = value
    return out


def env(key, default=None):
    """读一个配置项；进程环境变量优先于 .env。空字符串视为未设置。"""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _parse_env_file(ENV_PATH)
        # 只打印**键名**，绝不打印值——这些是凭据。
        # flush=True：这段在模块导入期执行，而服务器随后会一直跑下去；不 flush 的话
        # 这行会卡在缓冲区里，日志上看不到，等于没打印。
        if _ENV_CACHE:
            print("[env] 已载入 %s（%d 项：%s）"
                  % (ENV_PATH, len(_ENV_CACHE), ", ".join(sorted(_ENV_CACHE))),
                  flush=True)
        else:
            print("[env] 未找到 %s —— 第三方 key 视为未配置" % ENV_PATH, flush=True)
    value = os.environ.get(key)
    if value is None:
        value = _ENV_CACHE.get(key)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def tianditu_key():
    """天地图 key，未配置返回 None。"""
    return env("TIANDITU_KEY")


# --------------------------------------------------------------------------
# 天地图瓦片代理
# --------------------------------------------------------------------------
# 为什么必须走代理，而不是把 key 发给浏览器：
#
# 天地图按 **User-Agent** 判断调用方类型，并据此校验 key 类型：
#   浏览器 UA   → 只接受「浏览器端」key，否则 301012「权限类型错误」
#   非浏览器 UA → 只接受「服务器端」key，否则 301013「权限类型错误」
# （两条都是 2026-09-29 在本容器实测的结果。）
#
# 于是「把 key 下发到前端用 UrlTemplateImageryProvider 直连」这条路**只对浏览器端
# key 有效**；服务器端 key 从浏览器发出去必然 403。而 key 类型是用户在天地图后台
# 自己选的，我们不该要求用户去改。
#
# 代理成 `/tdt/<layer>/<z>/<x>/<y>` 后：
#   * 两种 key 都能用——先按非浏览器 UA 请求，被 301012 拒绝就换浏览器 UA 重试，
#     并把成功的组合记住（_TDT_UA_MODE）；
#   * key 始终留在服务端，不下发、不进 git、不出现在浏览器网络面板里；
#   * 顺带能做磁盘缓存，弱网/离线时已看过的区域仍然有图。
#
# 缓存目录在 web/tiles/ 下，该目录已被 .gitignore 覆盖。
TDT_BASE = "http://t0.tianditu.gov.cn"
# 允许的图层白名单：既是功能约束，也挡住把 layer 拼进 URL 造成的注入。
TDT_LAYERS = ("img", "cia", "vec", "cva", "ter")
TDT_CACHE_DIR = os.path.join(WEB_ROOT, "tiles", "tdt")
TDT_TIMEOUT = 12.0
TDT_MAX_LEVEL = 18
# 探测到的 UA 组合（'server' = 非浏览器 UA）。默认按服务器端 key 起步，因为
# 服务器端 key 更常见于后端代理场景。
_TDT_UA_MODE = ["server"]
# 最近一次失败原因，供 /api/tdt/status 查询——静默失败是这个项目已经吃过一次亏的地方。
_TDT_LAST_ERROR = [None]
# 上一次已经打过日志的原因，避免每块失败的瓦片都刷一行。
_TDT_LOG_STATE = [None]
# 上游并发上限。天地图会限流（实测遇到过 429 配额耗尽），而 Cesium 一次会抛来
# 几十个瓦片请求。不设上限的话代理会替浏览器把并发放大，反而更快触发限流。
# 4 是在本地缓存热起来之前，速度与"别把对方惹毛"之间的折中。
_TDT_MAX_UPSTREAM = 4
_TDT_SLOTS = None  # threading.Semaphore，用到时才建（模块导入期不建线程原语）


def _tdt_slot():
    global _TDT_SLOTS
    if _TDT_SLOTS is None:
        import threading
        _TDT_SLOTS = threading.Semaphore(_TDT_MAX_UPSTREAM)
    return _TDT_SLOTS


def _tdt_user_agent(mode):
    if mode == "browser":
        return ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    # 非浏览器 UA。用自述式而不是伪装成 curl/python：真出问题时对方日志里能认出我们。
    return "situation3d/1.1 (tile-proxy)"


def _tdt_error_detail(exc):
    """天地图的错误体是 JSON：{"code":301012,"msg":...,"resolve":...}。"""
    try:
        body = exc.read().decode("utf-8", "replace")
        data = json.loads(body)
        return data.get("code"), (data.get("msg") or data.get("resolve") or "")
    except Exception:  # noqa: BLE001 - 拿不到结构化错误就退回字符串
        return None, str(exc)


def _tdt_cache_write(path, data):
    """尽力写缓存。写不进去不影响本次返回（可能是只读挂载或磁盘满）。"""
    try:
        d = os.path.dirname(path)
        if not os.path.isdir(d):
            os.makedirs(d)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError:
        pass


def _tdt_upstream(layer, z, x, y, mode):
    key = tianditu_key()
    if not key:
        raise RuntimeError("未配置 TIANDITU_KEY")
    url = ("%s/%s_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
           "&LAYER=%s&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles"
           "&TILEMATRIX=%d&TILEROW=%d&TILECOL=%d&tk=%s"
           % (TDT_BASE, layer, layer, z, y, x, key))
    req = urllib.request.Request(url, headers={"User-Agent": _tdt_user_agent(mode)})
    with urllib.request.urlopen(req, timeout=TDT_TIMEOUT) as resp:
        return resp.read(), (resp.headers.get("Content-Type") or "image/jpeg")


def fetch_tdt_tile(layer, z, x, y):
    """
    取一块瓦片，返回 (bytes, content_type)。命中磁盘缓存则直接返回。

    key 类型未知，所以两种 UA 都试一遍：先用记忆中的组合，失败再换另一种。
    成功的组合被记住，后续请求不再多花一次往返。
    """
    cache = os.path.join(TDT_CACHE_DIR, layer, str(z), str(x), "%d.jpg" % y)
    try:
        with open(cache, "rb") as fh:
            return fh.read(), "image/jpeg"
    except OSError:
        pass

    order = (["server", "browser"] if _TDT_UA_MODE[0] == "server"
             else ["browser", "server"])
    last = "未知错误"
    slot = _tdt_slot()
    for mode in order:
        # 上游请求计数上限：缓存命中不会走到这里，所以只在真正要联网时才占坑。
        with slot:
            try:
                data, ctype = _tdt_upstream(layer, z, x, y, mode)
            except urllib.error.HTTPError as exc:
                code, msg = _tdt_error_detail(exc)
                last = "HTTP %s code=%s %s" % (exc.code, code, msg)
                # 301012 = 这是浏览器端 key；301013 = 这是服务器端 key。换另一种 UA 再试。
                if code in (301012, 301013):
                    continue
                break
            except Exception as exc:  # noqa: BLE001 - 网络层什么都可能抛
                last = "%s: %s" % (type(exc).__name__, exc)
                break
        _TDT_UA_MODE[0] = mode
        _TDT_LAST_ERROR[0] = None
        _tdt_cache_write(cache, data)
        return data, ctype

    _TDT_LAST_ERROR[0] = last
    raise RuntimeError(last)


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
    """
    Return the configured token, or None. Never raises.

    `.env` 里的 `CESIUM_ION_TOKEN` 优先于 web/data/ion.json：key 统一从 .env 读之后，
    这里是"配置"与"界面上临时填的"两条路，前者应当赢——否则 .env 改了却不生效，
    比没有 .env 更让人困惑。ion.json 保留为兼容路径（界面上的保存钮写的就是它）。
    """
    from_env = env("CESIUM_ION_TOKEN")
    if from_env:
        return from_env
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
        # `/wfs` is a public OGC endpoint rather than an app API, so it is called
        # out separately: a stock GIS client points at it directly, and it would
        # otherwise fall through to static file serving and 404.
        p = self.path.split("?")[0]
        return p.startswith("/api/") or p == "/wfs"

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

    @staticmethod
    def _query(handler_path):
        """Parse the query string of a request path into a dict."""
        try:
            from urllib.parse import urlparse, parse_qs
        except ImportError:  # pragma: no cover - python2 guard
            return {}
        q = parse_qs(urlparse(handler_path).query)
        # parse_qs returns lists; every parameter here is single-valued.
        return {k: v[0] for k, v in q.items() if v}

    def _api_get(self, path, query=None):
        query = query or {}
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
        if path == "/api/tdt/status":
            # **绝不下发 key 本身**：key 只存在服务端，浏览器用的是 /tdt/ 代理。
            # 这里只回答"配没配、通不通、按哪种 UA"。keyLen 用于区分"没配"与
            # "配错成空串"，不足以还原 key。
            key = tianditu_key()
            self._send_json({
                "configured": key is not None,
                "keyLen": len(key) if key else 0,
                "uaMode": _TDT_UA_MODE[0],
                "lastError": _TDT_LAST_ERROR[0],
                "proxyTemplate": "/tdt/{layer}/{z}/{x}/{y}",
                "layers": list(TDT_LAYERS),
                "cacheDir": os.path.relpath(TDT_CACHE_DIR, WEB_ROOT),
                "hint": ("已配置天地图 key（服务端代理取瓦片，key 不下发到浏览器）"
                         if key else
                         "未配置：把 TIANDITU_KEY 写进仓库根目录的 .env"),
            })
            return
        if path == "/api/satellites/passes":
            self._api_satellite_passes(query)
            return
        if path == "/api/intent/ontology":
            self._api_intent_ontology()
            return
        if path == "/api/health":
            self._api_health()
            return
        if path == "/wfs":
            self._wfs(query)
            return
        self._send_json({"error": "未知接口"}, 404)

    def _wfs(self, query):
        """OGC WFS 2.0.0 over the offline vector datasets, GeoJSON out.

        Lives at `/wfs` rather than under `/api/` so a stock GIS client can be
        pointed at it directly. Parameters are the standard WFS ones
        (service/version/request/typeNames/bbox/count/startIndex), with the WFS 1.1
        `maxFeatures` accepted as an alias for `count`.
        """
        import engine.wfs as wfs

        service = (query.get("service") or "WFS").upper()
        request = (query.get("request") or "GetCapabilities").strip()
        version = (query.get("version") or wfs.WFS_VERSION).strip()
        type_names = query.get("typeNames") or query.get("typenames") or query.get("typeName")

        def send_xml(text, status=200):
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/xml; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # WFS responses are live-ish (datasets can be re-fetched), so do not
            # let a proxy pin them.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Client navigated away mid-response; not an error worth a traceback.
                pass

        if service != "WFS":
            send_xml(wfs.exception_report("only service=WFS is supported", "InvalidParameterValue"),
                     400)
            return
        if version not in ("2.0.0", "2.0", "1.1.0", "1.0.0"):
            send_xml(wfs.exception_report("unsupported version %r" % version,
                                          "InvalidParameterValue"), 400)
            return

        req = request.lower()
        try:
            if req == "getcapabilities":
                names = [t.strip() for t in type_names.split(",")] if type_names else None
                send_xml(wfs.get_capabilities(
                    base_url="http://%s/wfs" % (self.headers.get("Host") or "localhost"),
                    type_names=names))
                return
            if req == "describefeaturetype":
                if not type_names:
                    send_xml(wfs.exception_report("DescribeFeatureType needs typeNames"), 400)
                    return
                send_xml(wfs.describe_feature_type(type_names.split(",")[0].strip()))
                return
            if req == "getfeature":
                if not type_names:
                    send_xml(wfs.exception_report("GetFeature needs typeNames"), 400)
                    return
                # One type per request: a merged multi-type response would have to
                # invent a feature-type column, and the front end does not need it.
                name = type_names.split(",")[0].strip()
                result = wfs.get_feature(
                    name,
                    bbox_raw=query.get("bbox"),
                    bbox_crs=query.get("bboxCrs"),
                    count_raw=query.get("count"),
                    max_features=query.get("maxFeatures"),
                    start_index_raw=query.get("startIndex"),
                    property_names=query.get("propertyName"),
                )
                self._send_json(result)
                return
            send_xml(wfs.exception_report("unsupported request %r" % request,
                                          "OperationNotSupported"), 400)
        except wfs.WfsError as e:
            send_xml(wfs.exception_report(str(e)), 400)
        except FileNotFoundError as e:
            send_xml(wfs.exception_report(str(e), "NoApplicableCode"), 503)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            send_xml(wfs.exception_report("internal error: %s" % e, "NoApplicableCode"), 500)

    def _api_satellite_passes(self, query):
        """Retired. Kept as an explicit 410 so an old client gets a clear answer
        instead of a confusing 404 "unknown endpoint"."""
        self._send_json({
            "error": "卫星过境接口已移除",
            "detail": "该模块已按要求下线；engine/sgp4.py 与 engine/satellites.py 也已删除。",
        }, 410)

    def _api_intent_ontology(self):
        """The 22-intent tactical ontology, served so the front end has one source.

        The front end must not carry a copy of 22 intent names: when the experts
        refine the ontology, the UI has to follow automatically. The file itself
        is vendored into `engine/data/` from the agent checkout
        (`code/shenji_agent/Shenji/ontology/intent_ontology_v2.json`) -- see
        engine/intent.py for why it is vendored rather than imported.
        """
        import engine.intent as intent

        try:
            full = intent.load_ontology()
        except intent.IntentError as e:
            self._send_json({"error": str(e)}, 503)
            return
        payload = dict(intent.ontology_summary())
        # confusion_matrix and the per-intent evidence criteria are useful to a
        # reviewer reading the UI, and they are already in the file.
        payload["confusion_matrix"] = full.get("confusion_matrix") or []
        payload["status_layer_definitions"] = full.get("status_layers") or []
        payload["supported_rules"] = intent.SUPPORTED_RULES
        payload["underivable"] = intent.underivable_report()
        self._send_json(payload)

    def _api_health(self):
        """Cheap liveness/config probe for the front end and for scripts."""
        import engine.wfs as wfs
        try:
            types = {k: v.get("features") for k, v in wfs.feature_types().items()}
        except Exception:  # noqa: BLE001 - health must never raise
            types = None
        self._send_json({
            "ok": True,
            "ion_configured": read_ion_token() is not None,
            "tianditu_configured": tianditu_key() is not None,
            "env_file": ENV_PATH,
            "wfs_types": types,
            "web_root": WEB_ROOT,
        })

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

    def _tdt_tile(self):
        """
        GET /tdt/<layer>/<z>/<x>/<y> —— 天地图瓦片代理（见 fetch_tdt_tile 的说明）。

        放在 `/tdt/` 而不是 `/api/` 下：它是给 Cesium 的 URL 模板直接用的。
        任何失败都回 404 而不是 500 —— 底下的离线底图仍在，缺一块瓦片不该让整个
        图层报错或让控制台刷满异常。
        """
        parts = self.path.split("?")[0].strip("/").split("/")
        if len(parts) != 5 or parts[0] != "tdt":
            self.send_error(404, "bad tile path")
            return
        layer = parts[1]
        try:
            z, x, y = int(parts[2]), int(parts[3]), int(parts[4])
        except ValueError:
            self.send_error(400, "bad tile coords")
            return
        if (layer not in TDT_LAYERS or not (0 <= z <= TDT_MAX_LEVEL)
                or x < 0 or y < 0):
            self.send_error(404, "bad tile")
            return
        if not tianditu_key():
            self.send_error(503, "TIANDITU_KEY 未配置")
            return
        try:
            data, ctype = fetch_tdt_tile(layer, z, x, y)
        except Exception as exc:  # noqa: BLE001
            # 每个坐标都打日志会刷屏；只在原因变化时打一条，避免淹没其它日志。
            reason = str(exc)
            if _TDT_LAST_ERROR[0] != reason or _TDT_LOG_STATE[0] != reason:
                print("[tdt] %s z=%d x=%d y=%d 取瓦片失败：%s"
                      % (layer, z, x, y, reason))
                _TDT_LOG_STATE[0] = reason
            self.send_error(404, "tile unavailable")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype or "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=%d" % IMMUTABLE_MAX_AGE)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.split("?")[0].startswith("/tdt/"):
            try:
                self._tdt_tile()
            except Exception:  # noqa: BLE001 - never leave the client hanging
                import traceback
                traceback.print_exc()
                try:
                    self.send_error(500, "tdt proxy error")
                except Exception:  # noqa: BLE001
                    pass
            return
        if self._is_api_path():
            try:
                self._api_get(self.path.split("?")[0], self._query(self.path))
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

    # 启动时就把 .env 读掉：语法有问题的行会在开机日志里立刻报出来，而不是等到
    # 第一个请求来了才发现（惰性读取会让"配错了"和"没人访问"看起来一模一样）。
    env("_WARM_")

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
