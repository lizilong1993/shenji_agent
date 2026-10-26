"""Online acquisition (P3).

The engine still runs offline.  This module only refreshes *inputs*:

* ``refresh_envpack`` rebuilds an envpack from the same basic map / grid as an
  existing pack, using the online OSM + Terrarium sources already implemented
  in ``geodata``.  The result is written to a timestamped cache directory and
  registered as a new (versioned) entry; the old entry is left intact so
  historical runs keep their pinned path/hash.
* The refresh refuses to register a pack whose road layer fell back to the
  basic map (``allow_degraded=False`` by default) -- a failed Overpass fetch
  must not silently replace better offline data.

Network access is only attempted by explicit ``fetch`` (CLI) or by the runner
in ``auto``/``online`` resource mode when a critical envpack gap exists.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from compat.legacy import LegacyHexGrid
from geodata.cache import CacheStore
from geodata.dem import DemSource
from geodata.osm import OsmSource
from geodata.pipeline import build_envpack

from .model import ResourceEntry
from .scanners import scan_envpack
from .store import ResourceStore


class ProviderError(RuntimeError):
    """Online acquisition failed or produced unusable data."""


def probe_online(host: str = "overpass-api.de", port: int = 443,
                 timeout_s: float = 3.5) -> bool:
    """Cheap reachability check using a tiny Overpass query.

    A TCP/TLS connect is not enough here: the egress proxy may complete the
    handshake and then stall the request.  The probe therefore POSTs a
    one-node query and treats any HTTP answer / quick JSON response as online.
    """
    if host == "overpass-api.de" and port == 443:
        url = "https://overpass-api.de/api/interpreter"
        data = urllib.parse.urlencode(
            {"data": "[out:json][timeout:3];node(1);out;"}).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"User-Agent": "situation3d-geo-engine/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return 200 <= getattr(resp, "status", 200) < 500
        except urllib.error.HTTPError as exc:
            return exc.code != 406
        except Exception:  # noqa: BLE001 - any failure means "not usable"
            return False
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def download_osm_pbf(url: str, dest: str, timeout_s: int = 900,
                     progress=None) -> str:
    """Download a public .osm.pbf to ``dest`` atomically.

    Geofabrik redirects to a dated mirror; urllib follows it and honours the
    standard ``https_proxy`` environment variables.  Content is written to
    ``dest.part`` and renamed only after a complete read.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(
        url, headers={"User-Agent": "situation3d-geo-engine/1.0"})
    done = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            total = resp.headers.get("Content-Length")
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
        if done <= 0:
            raise ProviderError("downloaded .osm.pbf is empty")
        os.replace(tmp, dest)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def pbf_cache_path(cache_root: str, name: str = "download.osm.pbf") -> str:
    return os.path.join(cache_root, "osm_pbf", name)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ProviderError("expected a JSON object: %s" % path)
    return data


def refresh_envpack(store: ResourceStore, entry: ResourceEntry,
                    cache_root: str, mode: str = "auto",
                    timeout_s: int = 20,
                    osm_source=None, dem_source=None,
                    osm_pbf: Optional[str] = None,
                    allow_degraded: bool = False) -> ResourceEntry:
    """Rebuild ``entry`` from live sources and register a versioned copy."""
    if entry.kind != "envpack":
        raise ProviderError("refresh_envpack only supports envpack entries")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for existing in store.entries("envpack"):
        if (existing.notes.get("refreshed_from") == entry.entry_id
                and existing.notes.get("refresh_policy") == "fallback-template-v1"
                and existing.as_of == today):
            # One refresh per template per day: repeated auto runs must not
            # rebuild 38 MB every time or grow the registry without bound.
            return existing
    if not os.path.isdir(entry.path):
        raise ProviderError("envpack path missing: %s" % entry.path)
    manifest_path = os.path.join(entry.path, "manifest.json")
    manifest = _load_json(manifest_path)
    base_rel = manifest.get("fallback_basic_json") or "base_map.json"
    base_path = os.path.join(entry.path, base_rel)
    if not os.path.isfile(base_path):
        raise ProviderError(
            "envpack has no basic map (%s); cannot rebuild online" % base_rel)
    basic = _load_json(base_path)
    try:
        grid = LegacyHexGrid.from_grid_meta(manifest["grid"])
    except Exception as exc:  # noqa: BLE001
        raise ProviderError("envpack grid metadata unusable: %s" % exc)
    name = str(manifest.get("pack_id") or manifest.get("theatre")
               or os.path.basename(entry.path))
    stamp = _stamp()
    out_root = os.path.join(cache_root, "envpacks")
    out = os.path.join(out_root, "%s@%s" % (name, stamp))
    cache_dir = os.path.join(cache_root, "_cache")
    cache = CacheStore(cache_dir, mode=mode)
    if osm_pbf:
        if not os.path.isfile(osm_pbf):
            raise ProviderError("osm pbf not found: %s" % osm_pbf)
        try:
            from geodata.osm_pbf import PbfOsmSource
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ProviderError(
                "pyosmium is required to build from .osm.pbf: %s" % exc)
        try:
            osm = PbfOsmSource(cache, osm_pbf)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError("cannot open .osm.pbf: %s" % exc)
    else:
        osm = osm_source or OsmSource(cache, timeout_s=timeout_s)
    dem = dem_source or DemSource(cache, zoom=int(manifest.get("dem_zoom") or 10),
                                  timeout_s=timeout_s)
    new_manifest = build_envpack(
        out, basic, grid,
        theatre=str(manifest.get("theatre") or "demo"),
        mode=mode, cache_dir=cache_dir,
        resolution_m=float(manifest.get("resolution_m") or 100.0),
        dem_zoom=int(manifest.get("dem_zoom") or 10),
        osm_source=osm, dem_source=dem, bathymetry_source=None,
        use_bathymetry=False, bbox=manifest.get("bbox"))
    # Keep the template's fallback policy.  Packs built for the engine use
    # ``fallback_basic_json: null`` so the coarse 2 km basic map cannot
    # re-introduce building/terrain classes inside the AOI; the pipeline
    # default would otherwise point at base_map.json.
    if "fallback_basic_json" in manifest \
            and manifest.get("fallback_basic_json") is None:
        new_manifest["fallback_basic_json"] = None
        with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(new_manifest, fh, ensure_ascii=False, indent=1)
    layers = new_manifest.get("layers") or {}
    roads_source = str((layers.get("roads") or {}).get("source") or "")
    if roads_source.startswith("fallback-basic") and not allow_degraded:
        raise ProviderError(
            "在线 OSM 获取不完整（roads=fallback-basic），拒绝覆盖现有环境包；"
            "请检查网络/代理后重试，或使用离线包")
    scanned = scan_envpack(out)
    versioned_id = "%s@%s" % (entry.entry_id, stamp)
    data = dict(scanned.as_dict(), bbox=scanned.bbox,
                entry_id=versioned_id,
                provider="online-refresh",
                source="online refresh of %s (built %s)" % (entry.entry_id, stamp),
                notes=dict(scanned.notes, refreshed_from=entry.entry_id,
                           roads_source=roads_source,
                           refresh_policy="fallback-template-v1"))
    new_entry = ResourceEntry(**data)
    store.register(new_entry, replace=True)
    return new_entry


def register_provided_envpack(store: ResourceStore, path: str,
                              entry_id: Optional[str] = None,
                              allowed_roots=None) -> ResourceEntry:
    """Register a user-supplied envpack path (manual override)."""
    path = os.path.realpath(os.path.abspath(path))
    if allowed_roots:
        roots = [os.path.realpath(os.path.abspath(root))
                 for root in allowed_roots]
        if not any(path == root or path.startswith(root + os.sep)
                   for root in roots):
            raise ProviderError(
                "提供的路径不在允许目录内：%s（允许：%s）"
                % (path, ", ".join(roots)))
    if entry_id and not re.match(
            r"^[a-z0-9][a-z0-9_.-]*:[A-Za-z0-9][A-Za-z0-9_.:@-]*$",
            entry_id):
        raise ProviderError("entry_id 格式无效: %r" % entry_id)
    if not os.path.isdir(path):
        raise ProviderError("provided envpack path not found: %s" % path)
    try:
        scanned = scan_envpack(path)
    except Exception as exc:  # noqa: BLE001
        raise ProviderError("provided envpack invalid: %s" % exc)
    data = dict(scanned.as_dict(), bbox=scanned.bbox,
                entry_id=entry_id or scanned.entry_id,
                provider="provided-file",
                notes=dict(scanned.notes, manual_override=True))
    entry = ResourceEntry(**data)
    store.register(entry, replace=True)
    return entry
