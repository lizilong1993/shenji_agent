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
import re
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

# Server-side mode and run store live beside this file.  `serve.py` is run as a
# script, so the server directory is on sys.path at runtime; tests add it too.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from modes import (  # noqa: E402
    DEFAULT_MODE, MODES, assert_no_simulation, capabilities as mode_capabilities,
    normalize_mode,
)
from run_store import RunError, RunStore  # noqa: E402

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
#
# `/tiles_db/`（mbtiles 直发）与 `/tdt/`（天地图代理）也在这里：瓦片内容不会原地变化。
# 这两个路径原先**不在**名单里，而它们的处理函数又各自 `send_header("Cache-Control", …)`
# 了一遍 —— 于是响应里出现两个互相矛盾的 Cache-Control（`public, max-age` 与后面的
# `no-store`），浏览器按最后一个走。所以这里既要加进名单，也要删掉那两处手动设置。
IMMUTABLE_PREFIXES = ("/tiles/", "/vendor/", "/tiles_db/", "/tdt/", "/tiles3d/")
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

# ---------------------------------------------------------------------------
# 离线瓦片库（mbtiles）—— 发布的瓦片**不摊成文件树**，直接按库发
# ---------------------------------------------------------------------------
# 为什么不摊开：这台机器的文件系统**建小文件极慢**（2026-10-14 实测：
# 顺序读 405 MB/s，但新建 12 KB 文件只有 **17 个/秒**、空文件 59 个/秒，
# 当时 load average ≈ 100）。20 万张瓦片摊开要三个多小时，还要多出 20 万个 inode；
# 而 mbtiles 本来就是"瓦片按行存"的容器 —— 磁盘占用相同甚至更小，
# 发布动作从"写 20 万个文件"变成"放一个文件"。
#
# 目录由 `.env` 的 `TILE_DB_DIR` 指定（默认 <repo>/tiles_db）。
# `<source>` 只接受 `[A-Za-z0-9_]+` 且必须命中目录里真实存在的 `<source>.mbtiles`：
# 这既挡住路径穿越，也让"库不在"表现为 404 而不是 500。
TILE_DB_DIR = env("TILE_DB_DIR") or os.path.join(os.path.dirname(HERE), "tiles_db")
TILE_DB_MAX_LEVEL = 24   # 再高就没有实际意义，且是明显的异常输入

# 天地图图层 → 离线瓦片库：断网或未配 key 时也能出图（见 fetch_tdt_tile）。
TDT_DB_SOURCE = {
    "img": "tianditu_satellite",
    "cia": "tianditu_satellite_label",
    "cva": "tianditu_vector_label",
    "ter": "tianditu_terrain_label",
}

# 连接与格式缓存。ThreadingHTTPServer 每个连接一个线程，按线程号存连接会随线程
# 回收而泄漏，所以这里**每库一条连接 + 一把锁**：sqlite3 的 execute 会放掉 GIL，
# 一次点查是几十微秒量级，串行化不影响这个量级的并发。
_TILE_DB_CONN = {}
_TILE_DB_LOCK = [None]

# ---------------------------------------------------------------------------
# TS 案例目录
# ---------------------------------------------------------------------------
#
# 案例**不再靠一份硬编码清单**（`engine/cases.py` 的 `CASES` + 生成出来的
# `web/data/cases.json`），而是扫描这个目录里的每一个 `*.json`。
# 往目录里放一份文件就多一个案例、删掉就少一个，**不需要动任何索引文件**，
# 也不需要跑生成器 —— 这是本次改动的核心目的。
#
# 用环境变量可覆盖（SITUATION_DIR），默认值即下面的路径。
SITUATIONS_DIR = env("SITUATION_DIR") or os.path.join(WEB_ROOT, "data", "situations")

# 默认案例：环境变量 SITUATION_DEFAULT 指定；不在扫描结果里时退回按 id 排序的第一个。
SITUATION_DEFAULT = env("SITUATION_DEFAULT") or None

# 测试端/演示端模式。默认 demo 是安全默认：未显式开启就没有推演能力。
SERVER_MODE = normalize_mode(env("SITUATION_MODE") or DEFAULT_MODE)
# 运行库只属于测试端；演示端即使目录存在也不会被创建/读取。
RUNS_DIR = env("SITUATION_RUNS_DIR") or os.path.join(_PKG_ROOT, "var", "sim_runs")
RUN_STORE = None

# `_api_situations` 的索引缓存：键 = 目录里各 .json 的 (名字, 大小, mtime_ns)，值 = 响应体。
# 见该方法的注释：不缓存的话，每次进系统都要白白解析 53.6 MB 的案例数据（10.5 s）。
_SITUATIONS_CACHE = {"key": None, "body": None}
# 持久化的那一份：让**重启服务后的第一个请求**也是毫秒级（内存缓存随进程消失）。
# 放 /tmp 而不是 web/data：这是服务端的派生数据，不该被当作项目数据文件。
_SITUATIONS_INDEX_PATH = "/tmp/situation3d_situations_index.json"

# 案例分片缓存（core / inference）。理由见 _case_parts()：39.6 MB 的案例里
# inference_frames 占 35 MB，首屏不需要它；但**不能让每次请求都重新解析**，
# 所以首次解析后把两个分片写到 /tmp，重启后直接读字节。
CASE_PARTS_DIR = "/tmp/situation3d_case_parts"
_CASE_PARTS = {}
_CASE_PARTS_LOCK = [None]


def _case_parts_lock():
    global _CASE_PARTS_LOCK
    if _CASE_PARTS_LOCK[0] is None:
        import threading
        _CASE_PARTS_LOCK[0] = threading.Lock()
    return _CASE_PARTS_LOCK[0]


def _write_case_part(path, data):
    """原子写；失败只当没写成（内存缓存仍然有效）。"""
    try:
        os.makedirs(CASE_PARTS_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError:
        pass


def _case_parts(path):
    """把案例拆成 (core, inference) 两份 JSON 字节。

    core = 原文档去掉 `inference_frames`；inference = 那一帧数组本身。
    前端先拿 core 立即渲染场景，再后台取 inference 补右侧 TS 面板 ——
    首屏不必等 35 MB 的推理帧。
    """
    lock = _case_parts_lock()
    with lock:
        try:
            st = os.stat(path)
        except OSError:
            raise IOError("案例文件不可读")
        key = "%x-%x" % (st.st_size, st.st_mtime_ns)
        ent = _CASE_PARTS.get(path)
        if ent and ent["key"] == key:
            return ent

        import hashlib
        stem = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12] + "-" + key
        core_p = os.path.join(CASE_PARTS_DIR, stem + ".core.json")
        inf_p = os.path.join(CASE_PARTS_DIR, stem + ".inference.json")
        core_gz_p = core_p + ".gz"
        inf_gz_p = inf_p + ".gz"
        if (os.path.isfile(core_p) and os.path.isfile(inf_p)
                and os.path.isfile(core_gz_p) and os.path.isfile(inf_gz_p)):
            with open(core_p, "rb") as fh:
                core = fh.read()
            with open(inf_p, "rb") as fh:
                inf = fh.read()
            with open(core_gz_p, "rb") as fh:
                core_gz = fh.read()
            with open(inf_gz_p, "rb") as fh:
                inf_gz = fh.read()
        else:
            if os.path.isfile(core_p) and os.path.isfile(inf_p):
                # 旧缓存（只有原文）：直接压缩，不必重新解析 39 MB。
                with open(core_p, "rb") as fh:
                    core = fh.read()
                with open(inf_p, "rb") as fh:
                    inf = fh.read()
            else:
                with open(path, "r") as fh:
                    doc = json.load(fh)
                frames = None
                if isinstance(doc, dict):
                    frames = doc.pop("inference_frames", None)
                core = json.dumps(doc, ensure_ascii=False).encode("utf-8")
                inf = json.dumps(frames if frames is not None else [],
                                 ensure_ascii=False).encode("utf-8")
                _write_case_part(core_p, core)
                _write_case_part(inf_p, inf)
            # **预压缩**：服务器 load average ~90 且进程 nice 10，请求时现压
            # 4.6 MB 实测要 5-11 s；压一次存起来，之后请求只写字节。
            core_gz = gzip.compress(core, 6)
            inf_gz = gzip.compress(inf, 6)
            _write_case_part(core_gz_p, core_gz)
            _write_case_part(inf_gz_p, inf_gz)

        ent = {"key": key, "core": core, "core_gz": core_gz,
               "inference": inf, "inference_gz": inf_gz}
        _CASE_PARTS[path] = ent
        return ent


def warm_case_parts():
    """启动后台把每个案例的 core/inference 分片准备好（每个 ~8 s，解析一次）。"""
    import threading

    def run():
        try:
            names = sorted(os.listdir(SITUATIONS_DIR))
        except OSError:
            return
        for name in names:
            if not name.endswith(".json"):
                continue
            full = os.path.join(SITUATIONS_DIR, name)
            if not os.path.isfile(full):
                continue
            t0 = time.time()
            try:
                _case_parts(full)
                print("[situations] 分片预热 %s 完成（%.1f s）"
                      % (name, time.time() - t0), flush=True)
            except Exception as exc:      # noqa: BLE001 - 预热失败不影响其它案例
                print("[situations] 分片预热 %s 失败：%s" % (name, exc), flush=True)

    threading.Thread(target=run, name="case-parts-warm", daemon=True).start()


def _situations_key():
    """
    目录里每个 `.json` 的 `(名字, 大小, mtime_ns)`，外加原始名字列表与 listdir 是否成功。

    为什么用 stat 而不是内容哈希：40 MB 文件哈希一次也要 ~0.1 s，而 stat 是常数代价，
    且这里的失效判据只需覆盖"文件被重新生成"这一种实际情况。
    """
    try:
        names = sorted(os.listdir(SITUATIONS_DIR))
        listed = True
    except OSError:
        names = []
        listed = False
    key = []
    for name in names:
        if not name.endswith(".json"):
            continue
        full = os.path.join(SITUATIONS_DIR, name)
        try:
            st = os.stat(full)
        except OSError:
            continue
        if os.path.isfile(full):
            key.append((name, st.st_size, st.st_mtime_ns))
    return tuple(key), names, listed


def _natural_key(text):
    """Stable, numeric-aware sort key for case names (登台2 before 登台10).

    Unicode code-point order is used for text segments, not pinyin, so the
    result is deterministic across servers and matches the case-id order
    (中印1 → 登台1 → 登台2).
    """
    out = []
    for part in re.split(r"(\d+)", str(text or "")):
        if part.isdigit():
            out.append((1, int(part)))
        elif part:
            out.append((0, part))
    return tuple(out)


def _case_sort_key(case):
    return (_natural_key(case.get("name") or case.get("id")),
            str(case.get("id") or ""))


def _build_situations_index(names, listed):
    """
    真正去解析案例文件，产出索引响应体。

    这一步就是那 10.5 s 的来源（`inference_frames`/`frames` 占了 46 MB，而索引只用到
    6 个字段），所以它只在缓存与持久化索引都未命中时才被调用。
    """
    cases, errors = [], []
    if not listed:
        errors.append({"file": os.path.relpath(SITUATIONS_DIR, WEB_ROOT),
                       "error": "目录不存在或不可读"})
    seen = {}
    for name in names:
        if not name.endswith(".json"):
            continue
        full = os.path.join(SITUATIONS_DIR, name)
        if not os.path.isfile(full):
            continue
        try:
            with open(full, "r") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            errors.append({"file": name, "error": str(exc)})
            continue
        if not isinstance(doc, dict):
            errors.append({"file": name, "error": "顶层不是对象"})
            continue
        cid = doc.get("case_id") or os.path.splitext(name)[0]
        if cid in seen:
            # 冲突时按文件名排序取第一个（确定性），另一个如实报出来
            errors.append({"file": name,
                           "error": "case_id 与 %s 冲突" % seen[cid]})
            continue
        seen[cid] = name
        targets = doc.get("targets") or []
        hotspots = doc.get("hotspots") or []
        cases.append({
            "id": cid,
            "name": doc.get("case_name") or cid,
            "description": doc.get("case_description") or "",
            "target_count": len(targets) if isinstance(targets, list) else 0,
            "hotspot_count": len(hotspots) if isinstance(hotspots, list) else 0,
            "file": "data/situations/" + name,
            "region": doc.get("region"),
            "theatre": doc.get("theatre"),
        })
    cases.sort(key=_case_sort_key)
    ids = {c["id"] for c in cases}
    default = SITUATION_DEFAULT if SITUATION_DEFAULT in ids else (
        cases[0]["id"] if cases else None)
    return {
        "default": default,
        "cases": cases,
        "errors": errors,
        "dir": os.path.relpath(SITUATIONS_DIR, WEB_ROOT),
    }


def _case_file_for_id(cid):
    """案例 id → 文件绝对路径。**不重新解析案例**：复用索引缓存/持久化索引。"""
    if not cid or "/" in cid or "\\" in cid:
        return None
    body = _SITUATIONS_CACHE["body"]
    if body is None:
        key, names, listed = _situations_key()
        body = _read_persisted_situations(key)
        if body is None:
            body = _build_situations_index(names, listed)
            _write_persisted_situations(key, body)
        _SITUATIONS_CACHE["key"] = key
        _SITUATIONS_CACHE["body"] = body
    for c in body.get("cases") or []:
        if c.get("id") != cid:
            continue
        rel = c.get("file") or ""
        if not rel.startswith("data/situations/"):
            return None
        full = os.path.join(SITUATIONS_DIR, os.path.basename(rel))
        return full if os.path.isfile(full) else None
    return None


def _read_persisted_situations(key):
    """读 /tmp 的持久化索引；键不匹配就返回 None（当作没命中）。"""
    try:
        with open(_SITUATIONS_INDEX_PATH, "r") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict):
            return None
        saved = tuple(tuple(x) for x in (doc.get("key") or []))
        if saved == key and isinstance(doc.get("body"), dict):
            return doc["body"]
    except (OSError, ValueError, TypeError):
        pass
    return None


def _write_persisted_situations(key, body):
    """原子写回持久化索引；失败只当没写成（不影响本次响应）。"""
    try:
        tmp = _SITUATIONS_INDEX_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"key": [list(k) for k in key], "body": body},
                      fh, ensure_ascii=False)
        os.replace(tmp, _SITUATIONS_INDEX_PATH)
    except OSError:
        pass


def warm_situations_index():
    """
    启动时在后台把案例索引准备好。

    没有这一步，"重启服务后的第一个请求"仍要现场付那 10.5 s —— 而那正是用户
    "进系统要十几秒"的场景。预热失败只打印一行，绝不影响服务启动。
    """
    try:
        key, names, listed = _situations_key()
        if _SITUATIONS_CACHE["key"] == key and _SITUATIONS_CACHE["body"] is not None:
            return
        body = _read_persisted_situations(key)
        if body is None:
            t0 = time.time()
            body = _build_situations_index(names, listed)
            print("[situations] 索引已构建（%.1f s，%d 个案例）"
                  % (time.time() - t0, len(body.get("cases") or [])), flush=True)
            _write_persisted_situations(key, body)
        _SITUATIONS_CACHE["key"] = key
        _SITUATIONS_CACHE["body"] = body
    except Exception as exc:            # noqa: BLE001 —— 预热不该让服务起不来
        print("[situations] 预热失败：%s" % exc, flush=True)
TDT_TIMEOUT = 12.0
TDT_MAX_LEVEL = 18
# 天地图"此级别下，该区域无影像"占位图。
#
# 天地图在区域/层级没有影像时**不回 404**，而是回一张固定的灰色 PNG（图上是
# "此级别下，该区域无影像，请放大或缩小地图后再浏览"），且对所有缺图瓦片都是
# **同一份字节**（2026-10-21 实测：4769 B，md5 c0edbdcb…；台海与印度方向、z16–z18
# 多点抽样一致）。把它当有效瓦片发出去，Cesium 会用它在父级影像之上覆盖一块灰，
# 整屏只剩那句提示。代理把它判成缺失、回 404 后，Cesium 对缺失瓦片的处理是保留
# 父级（更低层级）已加载的影像并拉伸——正是"维持相邻层级影像"的效果。
#
# 离线瓦片库 tianditu_satellite.mbtiles 里也烘焙进了 5 万多张这种占位瓦片
# （z14–z17），所以 /tdt 的瓦片库命中、上游取回，以及 /tiles_db 直发三条路径都要判。
TDT_BLANK_TILE_MD5 = "c0edbdcb2c8ddd3e6a5cf09348c0fcb4"
TDT_BLANK_TILE_SIZE = 4769
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


def tile_db_path(source):
    """把 `<source>` 解析成 tiles_db 下真实存在的 `<source>.mbtiles`，否则 None。"""
    import re
    if not source or not re.match(r"^[A-Za-z0-9_]+$", source or ""):
        return None
    p = os.path.join(TILE_DB_DIR, source + ".mbtiles")
    return p if os.path.isfile(p) else None


def _tile_db_lock():
    global _TILE_DB_LOCK
    if _TILE_DB_LOCK[0] is None:
        import threading
        _TILE_DB_LOCK[0] = threading.Lock()
    return _TILE_DB_LOCK[0]


def tile_db_sources():
    """当前可用的瓦片库源名（按文件名列出，供 /api/status 与 /api/health 用）。"""
    try:
        return sorted(f[:-len(".mbtiles")] for f in os.listdir(TILE_DB_DIR)
                      if f.endswith(".mbtiles"))
    except OSError:
        return []


def tile_db_ctype(data):
    """按**字节魔数**判瓦片类型，而不是信 mbtiles 的 metadata.format。

    实测教训：`tianditu_satellite.mbtiles` 的 metadata 写 `format=jpg`，
    但库里有相当一部分瓦片其实是 **PNG**（天地图对部分层级/瓦片返回带透明的 PNG）。
    照 metadata 发 `image/jpeg` 就会出现"PNG 内容配 JPEG 类型"——浏览器多数能靠
    嗅探显示出来，所以它是个**不会报错的错**，正因如此更该按字节判。
    """
    return "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"


def _tile_db_conn(path):
    """取（必要时建）该库的只读连接。调用方必须已持有 _tile_db_lock()。"""
    import sqlite3
    conn = _TILE_DB_CONN.get(path)
    if conn is None:
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True,
                               check_same_thread=False)
        _TILE_DB_CONN[path] = conn
    return conn


def tile_db_get(path, z, x, y):
    """取一张瓦片，命中返回 bytes，否则 None。

    **mbtiles 的行号是 TMS（自南向北）**，而 URL 里的 y 是 XYZ（自北向南），
    所以这里必须翻转：row = 2^z - 1 - y。翻错的表现是整幅图南北镜像 ——
    不报错，只是"看着不对"。
    """
    if z < 0 or z > TILE_DB_MAX_LEVEL or x < 0 or y < 0 or x >= (1 << z) or y >= (1 << z):
        return None
    try:
        with _tile_db_lock():
            row = _tile_db_conn(path).execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=?"
                " AND tile_row=?", (z, x, (1 << z) - 1 - y)).fetchone()
    except Exception:  # noqa: BLE001 - 库损坏/被换掉都不该让请求 500
        return None
    return row[0] if row else None


# ---------------------------------------------------------------------------
# 3D Tiles / quantized-mesh 地形：**zip 直发**，不摊成文件树
# ---------------------------------------------------------------------------
# 这批数据集是"几十万个小文件"的形态（2026-10-19 上传：TAIBEI101 36.4 万、
# TAIZHONG 26.1 万、TAOYUAN 22.7 万、TW2024 2.4 万、台湾 20m DEM 7.6 万），
# 而本机文件系统建小文件只有 17-59 个/秒（见上面 mbtiles 的注释）—— 摊开要
# 数小时且白白多出近百万 inode。zip 本身就是"按中央目录索引"的容器，和 mbtiles
# 同理：**保留上传的 zip 原样，请求时按条目取**。
#
# 目录由 `.env` 的 `TILES3D_DIR` 指定（默认 <repo>/tiles3d_db）。
# `<dataset>` 只接受 `[A-Za-z0-9_]+` 且必须命中真实存在的 `<dataset>.zip`。
TILES3D_DIR = env("TILES3D_DIR") or os.path.join(os.path.dirname(HERE), "tiles3d_db")

# 3D Tiles 与 quantized-mesh 的扩展名 → Content-Type。Cesium 对 .terrain 的
# 类型不挑剔，但按规范发 `application/vnd.quantized-mesh` 更利于排错。
TILES3D_TYPES = {
    ".json": "application/json; charset=utf-8",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".bin": "application/octet-stream",
    ".terrain": "application/vnd.quantized-mesh",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ktx2": "image/ktx2",
    ".webp": "image/webp",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}

# path -> {"key": (size, mtime_ns), "zf": ZipFile, "root": 顶层前缀}
_TILES3D_ZIPS = {}
# path -> threading.Lock。**每个 zip 一把锁**，不是全局一把：打开 taizhong
# 要几十秒，全局锁会把"读已经就绪的 taibei"也一起堵住（实测 13-34 s）。
_TILES3D_LOCKS = {}
_TILES3D_META_LOCK = [None]


def _tiles3d_meta_lock():
    global _TILES3D_META_LOCK
    if _TILES3D_META_LOCK[0] is None:
        import threading
        _TILES3D_META_LOCK[0] = threading.Lock()
    return _TILES3D_META_LOCK[0]


def _tiles3d_lock_for(path):
    """取（必要时建）该 zip 的专用锁。惰性建锁：导入期不建线程原语。"""
    with _tiles3d_meta_lock():
        lock = _TILES3D_LOCKS.get(path)
        if lock is None:
            import threading
            lock = threading.Lock()
            _TILES3D_LOCKS[path] = lock
        return lock


def tiles3d_dataset(name):
    """把 `<dataset>` 解析成 tiles3d_db 下真实存在的 `<dataset>.zip`，否则 None。"""
    import re
    if not name or not re.match(r"^[A-Za-z0-9_]+$", name or ""):
        return None
    p = os.path.join(TILES3D_DIR, name + ".zip")
    return p if os.path.isfile(p) else None


def tiles3d_sources():
    """当前可用的 3D 数据集名（供 /api/status 与 /api/health 用）。"""
    try:
        return sorted(f[:-len(".zip")] for f in os.listdir(TILES3D_DIR)
                      if f.endswith(".zip"))
    except OSError:
        return []


def _tiles3d_zip_locked(path):
    """取（必要时重开）该数据集的 ZipFile，并探测导出包的顶层目录前缀。

    调用方必须已持有 _tiles3d_lock_for(path)，且**在持有锁期间**完成所有
    对该 ZipFile 的读取（ZipFile 共用文件句柄，并发读会踩位置）。
    """
    import zipfile
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_size, st.st_mtime_ns)
    ent = _TILES3D_ZIPS.get(path)
    if ent and ent["key"] == key:
        return ent

    zf = zipfile.ZipFile(path)
    tops = set()
    common = True
    for n in zf.namelist():
        if n.endswith("/"):          # 目录占位条目不参与前缀判定
            continue
        if "/" not in n:             # 有顶层散文件 -> 没有统一前缀
            common = False
            break
        tops.add(n.split("/", 1)[0])
        if len(tops) > 1:
            common = False
            break
    root = (tops.pop() + "/") if (common and len(tops) == 1) else ""

    old = _TILES3D_ZIPS.get(path)
    if old:
        try:
            old["zf"].close()
        except Exception:  # noqa: BLE001 - 旧句柄关不掉不该影响新请求
            pass
    ent = {"key": key, "zf": zf, "root": root}
    _TILES3D_ZIPS[path] = ent
    return ent


def tiles3d_get(path, rel):
    """取一个条目，返回 (bytes, content_type, etag)；不存在返回 None。"""
    lock = _tiles3d_lock_for(path)
    with lock:
        ent = _tiles3d_zip_locked(path)
        if not ent:
            return None
        name = ent["root"] + rel
        try:
            info = ent["zf"].getinfo(name)
            data = ent["zf"].read(info)
        except KeyError:
            # 顶层文件（tileset.json / layer.json）缺失只可能是前缀探测或数据本身
            # 的问题，值得留一条日志；瓦片级缺失是常态（域外请求），不打。
            if "/" not in rel:
                print("[tiles3d] 缺少顶层条目 %s（root=%r）" % (name, ent["root"]),
                      flush=True)
            return None
        except Exception:  # noqa: BLE001 - 坏条目/坏 zip 都回 404，不 500
            return None
    ext = os.path.splitext(name)[1].lower()
    ctype = TILES3D_TYPES.get(ext) or "application/octet-stream"
    etag = '"%x-%x"' % (info.CRC, info.file_size)
    return data, ctype, etag


def warm_tiles3d():
    """启动时在后台打开所有 zip，把**中央目录解析**的代价从第一个请求里挪走。

    实测（2026-10-19，load ≈ 90）：`ZipFile('taibei101.zip')` 打开要 **19-50 s**
    （3.5 GB / 36.4 万条目的中央目录，在慢盘上读+解析），而打开之后单条目读取
    只要 0.01 s。Cesium 打开图层时先取 tileset.json —— 不预热的话用户点开关
    要干等几十秒且没有任何提示。

    每个数据集**一个线程**（各自持自己的锁）：顺序预热的话最后一个数据集要等到
    前四个都开完；并行之后全部就绪的时间约等于最慢的那一个。
    """
    import threading

    def one(name, path):
        t0 = time.time()
        try:
            with _tiles3d_lock_for(path):
                _tiles3d_zip_locked(path)
            print("[tiles3d] 预热 %s 完成（%.1f s）"
                  % (name, time.time() - t0), flush=True)
        except Exception as exc:  # noqa: BLE001 - 预热失败不影响其它数据集
            print("[tiles3d] 预热 %s 失败：%s" % (name, exc), flush=True)

    for name in tiles3d_sources():
        path = tiles3d_dataset(name)
        if not path:
            continue
        threading.Thread(target=one, args=(name, path),
                         name="tiles3d-warm-" + name, daemon=True).start()


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
    取一块瓦片，返回 (bytes, content_type)：**磁盘缓存 → 上游**。

    （离线瓦片库在上游之前由 `_tdt_tile` 先查 —— 那里查是为了让"库里命中"
    不必先满足"配了 key"这个前置条件。）

    key 类型未知，所以两种 UA 都试一遍：先用记忆中的组合，失败再换另一种。
    成功的组合被记住，后续请求不再多花一次往返。
    """
    cache = os.path.join(TDT_CACHE_DIR, layer, str(z), str(x), "%d.jpg" % y)
    try:
        with open(cache, "rb") as fh:
            data = fh.read()
        # 缓存里既有 jpg 也有 png（cia/cva/ter 上游本来就是 PNG）。按魔数判类型：
        # 原先把一切都当 image/jpeg 发，PNG 内容配 JPEG 类型，属于"碰巧能显示"。
        ctype = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
        return data, ctype
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


def tdt_blank_tile(data):
    """是否天地图"此级别下，该区域无影像"的占位图（见 TDT_BLANK_TILE_MD5）。

    先比长度再算 md5：正常影像瓦片几乎不会正好 4769 字节，绝大多数请求只付一次
    整数比较；长度命中才做哈希（几百微秒）。占位图同样写进磁盘缓存，重复请求
    只花一次读盘 + 哈希，不会反复打上游。
    """
    return (len(data) == TDT_BLANK_TILE_SIZE
            and hashlib.md5(data).hexdigest() == TDT_BLANK_TILE_MD5)


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


def read_ion_token():
    """
    Return the configured token, or None. Never raises.

    `.env` 里的 `CESIUM_ION_TOKEN` 优先于 `web/data/ion.json`：key 统一从 .env 读之后，
    "配置"与"历史遗留文件"两条路里前者应当赢——否则 .env 改了却不生效，比没有 .env
    更让人困惑。ion.json 保留为兼容读取（旧部署可能还在用），但界面上已经没有写它的
    入口了（令牌输入框与保存按钮已随 UI 精简移除）。

    ⚠ 这个函数被 `/api/ion/status`、`/api/ion/token` 与 `/api/health` 三处调用。
    它曾被按行号范围删"死代码"时误删过一次（连同 `_b64url_decode` / `write_ion_token` /
    `validate_ion_token` 一起），结果三个接口全部 NameError。**教训：删代码前先按名字
    grep 全部调用点，不要凭"这一段看起来都是 ion 相关"就整段删。**
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
        self._send_json_bytes(body, status)

    def _send_json_bytes(self, body, status=200, gz_body=None):
        """发 JSON 字节；客户端支持 gzip 时优先用预压缩体（案例分片已预先压好）。"""
        enc = None
        wants = "gzip" in (self.headers.get("Accept-Encoding") or "")
        if wants and status == 200 and gz_body is not None:
            body = gz_body
            enc = "gzip"
        elif wants and status == 200 and len(body) >= 1024:
            # 小响应现压即可；大响应（案例分片）走上面的预压缩路径。
            body = gzip.compress(body, 6)
            enc = "gzip"
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if enc:
            self.send_header("Content-Encoding", enc)
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # HTTP/1.1 keep-alive: the body must be written to keep the socket
        # framing valid, exactly like _send_head does for files.
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

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
            db_map = {ly: s for ly, s in TDT_DB_SOURCE.items() if tile_db_path(s)}
            self._send_json({
                "configured": key is not None,
                "keyLen": len(key) if key else 0,
                "uaMode": _TDT_UA_MODE[0],
                "lastError": _TDT_LAST_ERROR[0],
                "proxyTemplate": "/tdt/{layer}/{z}/{x}/{y}",
                "layers": list(TDT_LAYERS),
                "cacheDir": os.path.relpath(TDT_CACHE_DIR, WEB_ROOT),
                # 离线库里有哪些层：命中它们时**不需要 key、不需要联网**。
                "offlineLayers": sorted(db_map),
                "offlineDbDir": os.path.relpath(TILE_DB_DIR, os.path.dirname(HERE)),
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
        if path == "/api/situations":
            self._api_situations()
            return
        if path == "/api/situation":
            # 案例分片：?id=<case_id>&part=core|inference。前端先 core 后 inference，
            # 首屏不再等整包 39.6 MB（inference_frames 占 35 MB）。
            cid = (query or {}).get("id") or ""
            part = (query or {}).get("part") or "core"
            if part not in ("core", "inference"):
                self._send_json({"error": "part 只能是 core 或 inference"}, 400)
                return
            full = _case_file_for_id(cid)
            if not full:
                self._send_json({"error": "未知案例：%s" % cid}, 404)
                return
            try:
                parts = _case_parts(full)
            except Exception as exc:      # noqa: BLE001
                self._send_json({"error": "案例解析失败：%s" % exc}, 500)
                return
            if part == "core":
                self._send_json_bytes(parts["core"], gz_body=parts["core_gz"])
            else:
                self._send_json_bytes(parts["inference"],
                                      gz_body=parts["inference_gz"])
            return
        if path == "/api/sim/runs":
            # 只在测试端存在；演示端连路由都不暴露。
            if SERVER_MODE != "test" or RUN_STORE is None:
                self._send_json({"error": "not found"}, 404)
                return
            cid = (query or {}).get("case_id") or ""
            if not cid:
                self._send_json({"error": "case_id 必填"}, 400)
                return
            try:
                runs = RUN_STORE.list_runs(cid)
            except RunError as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            self._send_json({"mode": "test", "case_id": cid, "runs": runs})
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

    def _api_situations(self):
        """
        扫描 `SITUATIONS_DIR`，把目录里的每一份 TS 汇成案例索引。

        **解析失败的文件不静默**：进 `errors` 并如实报出文件名与原因，其余案例照常
        可用 —— 一份坏文件不该让整个菜单打不开。目录不存在或为空时返回空清单而不是
        500：那是一个正常状态（还没放案例），不是服务端故障。

        字段名与旧的 `web/data/cases.json` **保持一致**（含 `file`），这样前端的
        `state.caseIndex.cases[]`、菜单渲染与 `?case=` 深链都不用改。

        ⚠ 2026-10-17：加**按 stat 的索引缓存**。理由是实测的启动成本 ——
        两个案例文件合计 53.6 MB（`taiwan_landing.json` 39.6 MB，其中
        `inference_frames` 占 35 MB / 1441 帧），而本接口只用到 6 个字段。
        把它们整个 `json.load` 一次实测 **10.5 s**（14 MB→2.30 s、39.6 MB→8.18 s），
        HTTP 侧 TTFB 19.9~25.9 s；启动时前端卡在
        `Promise.all([loadBase(), loadCaseIndex()])` 上，用户看到的就是"进系统十几秒"。
        现在文件没变（名字/大小/mtime 都没动）就直接复用上次的响应体，
        只有案例文件真的变了才重算一次。命中路径分三级：
        **进程内存（~10 ms）→ /tmp 的持久化索引（省掉解析）→ 真去解析（10.5 s）**。
        第三级只在"索引文件缺失/不匹配"时才走到；启动时还有一个后台预热线程
        （见 warm_situations_index）把这一级也基本消掉 —— 否则**每次重启服务后的
        第一个请求**仍要等 20 多秒。
        """
        key, names, listed = _situations_key()
        if _SITUATIONS_CACHE["key"] == key and _SITUATIONS_CACHE["body"] is not None:
            self._send_json(_SITUATIONS_CACHE["body"])
            return
        body = _read_persisted_situations(key)
        if body is None:
            body = _build_situations_index(names, listed)
        _SITUATIONS_CACHE["key"] = key
        _SITUATIONS_CACHE["body"] = body
        _write_persisted_situations(key, body)
        self._send_json(body)

    def _api_health(self):
        """Liveness/config probe.

        I08 步骤 3：在**保留原有布尔字段**的前提下增加进程健康维度 ——
        `level`（整体级别）、`subchecks`（逐子项状态）、`cache`（离线缓存状态）。
        保留原字段是【风险与回滚】的明确要求：扩展不得破坏既有调用方
        （`dataHealth.js` 与脚本都按 `ion_configured` / `tianditu_configured` 取值）。

        ⚠ 本函数**不得抛异常**（健康检查自己失败就失去意义），
        故每个子项各自 try/except 并把失败降级成该子项的 `error` 状态，
        而不是让整条接口 500。
        """
        import engine.wfs as wfs

        subchecks = {}

        def _check(name, fn, *, required=True):
            """跑一个子项检查，把结果记进 subchecks。异常即该项 error。"""
            try:
                value = fn()
                ok = bool(value)
                subchecks[name] = {
                    "status": "ok" if ok else ("error" if required else "warn"),
                    "ok": ok, "detail": None, "required": required,
                }
            except Exception as exc:  # noqa: BLE001 - health must never raise
                subchecks[name] = {
                    "status": "error", "ok": False,
                    "detail": "%s: %s" % (type(exc).__name__, exc), "required": required,
                }

        _check("ion_token", lambda: read_ion_token() is not None)
        _check("tianditu_key", lambda: tianditu_key() is not None)
        # 离线瓦片库：有它就说明"上传的瓦片已经发布成服务"，且天地图/离线底图的
        # 高倍段即便没有 key 也能出图。缺库只算 warn（低倍静态树仍在）。
        _check("tile_dbs", lambda: len(tile_db_sources()) > 0, required=False)
        # 3D Tiles / quantized-mesh 数据集（zip 直发）。缺它只影响三维城市与
        # 台湾 20m 地形这两个可选图层，所以是 warn 而不是 error。
        _check("tiles3d_sets", lambda: len(tiles3d_sources()) > 0, required=False)

        types = None
        try:
            types = {k: v.get("features") for k, v in wfs.feature_types().items()}
            subchecks["wfs_types"] = {
                "status": "ok" if types else "warn", "ok": bool(types),
                "detail": None, "required": False,
            }
        except Exception as exc:  # noqa: BLE001
            subchecks["wfs_types"] = {
                "status": "error", "ok": False,
                "detail": "%s: %s" % (type(exc).__name__, exc), "required": False,
            }

        # 离线缓存状态：瓦片集是否在位（断网期前端靠它回放）。
        # 只数顶层目录与 manifest 是否存在，**不做全量遍历** —— 健康检查必须廉价。
        try:
            tiles_root = os.path.join(WEB_ROOT, "tiles")
            manifest = os.path.join(tiles_root, "manifest.json")
            subchecks["offline_tiles"] = {
                "status": "ok" if os.path.isdir(tiles_root) else "warn",
                "ok": os.path.isdir(tiles_root),
                "detail": None, "required": False,
            }
            subchecks["offline_manifest"] = {
                "status": "ok" if os.path.isfile(manifest) else "warn",
                "ok": os.path.isfile(manifest),
                "detail": None, "required": False,
            }
        except Exception as exc:  # noqa: BLE001
            subchecks["offline_tiles"] = {"status": "error", "ok": False,
                                          "detail": str(exc), "required": False}

        # 整体级别：任一**必需**子项 error → critical；有任意 error → error；
        # 有 warn → warn；否则 ok。这样级别的来源可解释，不是拍脑袋。
        statuses = [v["status"] for v in subchecks.values()]
        if any(v["status"] == "error" and v.get("required") for v in subchecks.values()):
            level = "critical"
        elif "error" in statuses:
            level = "error"
        elif "warn" in statuses:
            level = "warn"
        else:
            level = "ok"

        self._send_json({
            # ── 原有字段（**保持不变**，扩展不破坏既有调用方）──
            "ok": True,
            "mode": SERVER_MODE,
            "capabilities": mode_capabilities(SERVER_MODE),
            "ion_configured": read_ion_token() is not None,
            "tianditu_configured": tianditu_key() is not None,
            "env_file": ENV_PATH,
            "wfs_types": types,
            "web_root": WEB_ROOT,
            # 离线瓦片库（mbtiles 直发）里有哪些源；空列表=还没发布任何库。
            "tile_dbs": tile_db_sources(),
            # 3D Tiles / quantized-mesh 数据集（zip 直发）。
            "tiles3d_sets": tiles3d_sources(),
            # ── I08 新增 ──
            "level": level,
            "subchecks": subchecks,
            "cache": {
                "offline_tiles_present": subchecks.get("offline_tiles", {}).get("ok", False),
                "manifest_present": subchecks.get("offline_manifest", {}).get("ok", False),
                "note": "断网期前端用这些离线资产回放；恢复后应标记陈旧区间",
            },
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def _api_post(self, path):
        if path.startswith("/api/sim/"):
            if SERVER_MODE != "test" or RUN_STORE is None:
                self._send_json({"error": "simulation API is disabled in demo mode"}, 403)
                return
            self._send_json({"error": "simulation runner not implemented yet"}, 501)
            return
        # 令牌写入接口已移除：ion token 现在只从 `.env` 的 CESIUM_ION_TOKEN 读取，
        # 界面上那个输入框与「保存令牌」按钮也已删掉。唯一的调用方消失后，这条写入
        # 路径就是死代码——留着它等于留一个能改凭据的接口，没有必要。
        #
        # GET /api/ion/token **保留**：前端 buildings3d.js 靠它取 token。
        # 换 token 的方式：改 .env 的 CESIUM_ION_TOKEN，重启服务。
        self._send_json({
            "error": "令牌写入接口已移除：请改 .env 的 CESIUM_ION_TOKEN 后重启服务",
        }, 404)
        return

    def _send_tile(self, data, ctype):
        """发一张瓦片。Cache-Control 由 end_headers() 按 IMMUTABLE_PREFIXES 统一加，
        这里**不再重复设置**（两处都设会出现两个矛盾的 Cache-Control）。"""
        self.send_response(200)
        self.send_header("Content-Type", ctype or "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _tile_db_tile(self):
        """GET /tiles_db/<source>/<z>/<x>/<y>.<ext> —— 从 mbtiles 直接发瓦片。

        路径形状与静态瓦片树 `/<dir>/<z>/<x>/<y>.<ext>` 保持一致，这样
        `UrlTemplateImageryProvider` 的 url 模板只需换个前缀。
        任何失败都回 404：缺一块瓦片不该让整个图层报错。
        """
        parts = self.path.split("?")[0].strip("/").split("/")
        if len(parts) != 5 or parts[0] != "tiles_db":
            self.send_error(404, "bad tile path")
            return
        source = parts[1]
        try:
            z, x, y = int(parts[2]), int(parts[3]), int(parts[4].split(".")[0])
        except ValueError:
            self.send_error(400, "bad tile coords")
            return
        path = tile_db_path(source)
        if not path:
            self.send_error(404, "unknown tile db")
            return
        data = tile_db_get(path, z, x, y)
        if not data:
            self.send_error(404, "no such tile")
            return
        # 与 /tdt 同理：库里烘焙进去的天地图占位图也按缺失处理（见 TDT_BLANK_TILE_MD5）。
        if tdt_blank_tile(data):
            self.send_error(404, "no imagery at this level")
            return
        self._send_tile(data, tile_db_ctype(data))

    def _tiles3d_item(self):
        """GET /tiles3d/<dataset>/<路径> —— 从 zip 直发 3D Tiles / quantized-mesh。

        路径即 zip 内相对路径（顶层同名目录已被 `_tiles3d_zip` 去掉），查询串
        （Cesium 的 `?v=版本`）忽略。任何失败回 404：缺一块内容不该让整个
        tileset 报错。
        """
        parts = self.path.split("?")[0].strip("/").split("/")
        if len(parts) < 3 or parts[0] != "tiles3d":
            self.send_error(404, "bad tiles3d path")
            return
        name = parts[1]
        rel = "/".join(parts[2:])
        if ".." in rel.split("/"):
            self.send_error(400, "bad tiles3d path")
            return
        path = tiles3d_dataset(name)
        if not path:
            self.send_error(404, "unknown tiles3d dataset")
            return
        hit = tiles3d_get(path, rel)
        if not hit:
            self.send_error(404, "no such tiles3d item")
            return
        data, ctype, etag = hit
        # Cache-Control 由 end_headers() 按 IMMUTABLE_PREFIXES 统一加（/tiles3d/ 在名单里）。
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if etag:
            self.send_header("ETag", etag)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _tdt_tile(self):
        """
        GET /tdt/<layer>/<z>/<x>/<y> —— 天地图瓦片代理（见 fetch_tdt_tile 的说明）。

        放在 `/tdt/` 而不是 `/api/` 下：它是给 Cesium 的 URL 模板直接用的。
        任何失败都回 404 而不是 500 —— 底下的离线底图仍在，缺一块瓦片不该让整个
        图层报错或让控制台刷满异常。

        **顺序**：离线瓦片库 → 磁盘缓存/上游。库放最前，是为了让"有库里那块瓦片"
        不需要任何前置条件（key 没配、断网都照样出图）—— 这正是离线底图的意义。
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

        src = TDT_DB_SOURCE.get(layer)
        db_path = tile_db_path(src) if src else None
        if db_path:
            data = tile_db_get(db_path, z, x, y)
            if data:
                # 库里可能烘焙进了占位图（见 TDT_BLANK_TILE_MD5），一样按缺失处理。
                if tdt_blank_tile(data):
                    self.send_error(404, "no imagery at this level")
                    return
                self._send_tile(
                    data, tile_db_ctype(data))
                return

        if not tianditu_key():
            self.send_error(503, "TIANDITU_KEY 未配置，且离线瓦片库里没有这块")
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
        if tdt_blank_tile(data):
            self.send_error(404, "no imagery at this level")
            return
        self._send_tile(data, ctype)

    def do_GET(self):
        if self.path.split("?")[0].startswith("/tiles3d/"):
            try:
                self._tiles3d_item()
            except Exception:  # noqa: BLE001 - never leave the client hanging
                import traceback
                traceback.print_exc()
                try:
                    self.send_error(500, "tiles3d error")
                except Exception:  # noqa: BLE001
                    pass
            return
        if self.path.split("?")[0].startswith("/tiles_db/"):
            try:
                self._tile_db_tile()
            except Exception:  # noqa: BLE001 - never leave the client hanging
                import traceback
                traceback.print_exc()
                try:
                    self.send_error(500, "tile db error")
                except Exception:  # noqa: BLE001
                    pass
            return
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
    ap.add_argument("--mode", choices=MODES, default=None,
                    help="test = live simulation + run store; demo = read-only replay")
    ap.add_argument("--runs-dir", default=None,
                    help="run store root (test mode); default var/sim_runs")
    args = ap.parse_args()

    global SERVER_MODE, RUNS_DIR, RUN_STORE
    try:
        SERVER_MODE = normalize_mode(args.mode or env("SITUATION_MODE") or DEFAULT_MODE)
    except ValueError as exc:
        sys.stderr.write("%s\n" % exc)
        return 2
    RUNS_DIR = (args.runs_dir or env("SITUATION_RUNS_DIR")
                or os.path.join(_PKG_ROOT, "var", "sim_runs"))
    if SERVER_MODE == "demo":
        try:
            assert_no_simulation(SERVER_MODE)
        except RuntimeError as exc:
            sys.stderr.write("refusing to start: %s\n" % exc)
            return 3
    else:
        RUN_STORE = RunStore(RUNS_DIR)

    if not os.path.isdir(WEB_ROOT):
        sys.stderr.write("web root not found: %s\n" % WEB_ROOT)
        return 1

    # 启动时就把 .env 读掉：语法有问题的行会在开机日志里立刻报出来，而不是等到
    # 第一个请求来了才发现（惰性读取会让"配错了"和"没人访问"看起来一模一样）。
    env("_WARM_")

    httpd = Server((args.host, args.port), Handler)

    # 启动即预热案例索引（后台线程）。没有它，"重启服务后的第一个请求"仍要现场解析
    # 53.6 MB 的案例数据（10.5 s），用户看到的就是"进系统卡十几秒"。
    # 放后台线程：预热要几秒到十几秒，不该拖住服务开始监听。
    try:
        import threading
        threading.Thread(target=warm_situations_index, daemon=True).start()
    except Exception as exc:            # noqa: BLE001
        print("[situations] 预热线程未启动：%s" % exc, flush=True)

    # 案例分片（core/inference）同样后台预热：每个案例解析一次 ~8 s，放在首请求里
    # 就会变成"打开案例等 8 秒"。
    try:
        warm_case_parts()
    except Exception as exc:            # noqa: BLE001
        print("[situations] 分片预热线程未启动：%s" % exc, flush=True)

    # 3D Tiles 的 zip 中央目录解析要 ~20 s/个（见 warm_tiles3d 的注释），同样放后台。
    try:
        warm_tiles3d()
    except Exception as exc:            # noqa: BLE001
        print("[tiles3d] 预热线程未启动：%s" % exc, flush=True)

    sys.stderr.write("serving %s on http://%s:%d/  mode=%s%s  (started %s)\n"
                     % (WEB_ROOT, args.host, args.port, SERVER_MODE,
                        ("  runs=%s" % RUNS_DIR) if SERVER_MODE == "test" else "",
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
