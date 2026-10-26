"""Parallel multi-source acquisition with automatic fallback.

Pipeline::

    candidates -> parallel probe (availability/latency/throughput/quality)
               -> rank (quality, then speed)
               -> try in order (fetch + normalise); on failure/slow -> next
               -> cache + register a standard tileset

"Download whole raster and tile locally" and "download XYZ tiles directly"
are both first-class methods, so a slow/failing tile service can be replaced
by a raster + local `gdal2tiles` path (and vice versa).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .model import ResourceEntry, hash_path, slugify, utc_now
from .processing import (TilesetSpec, crop_raster_window, download_file,
                         image_magic_ok, raster_to_tiles, tileset_stats,
                         verify_tileset, write_manifest)
from .sources import SourceCandidate, candidates_for_tiles
from .store import ResourceError, ResourceStore
from .tilemath import lonlat_to_tile, tiles_for_bbox

QUALITY_RANK = {"measured": 0, "derived": 1, "default": 2, "unknown": 3}


class AcquireError(RuntimeError):
    pass


class SourceFailed(AcquireError):
    pass


class SlowSource(SourceFailed):
    pass


@dataclass
class ProbeResult:
    candidate: SourceCandidate
    ok: bool
    latency_ms: float = 0.0
    speed_bps: float = 0.0
    bytes_read: int = 0
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.candidate.source_id,
            "method": self.candidate.method,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 1),
            "speed_bps": round(self.speed_bps, 1),
            "bytes_read": self.bytes_read,
            "detail": self.detail,
        }


@dataclass
class AcquireResult:
    entry: Optional[ResourceEntry]
    manifest: Dict[str, Any]
    source: Dict[str, Any]
    cached: bool
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.as_dict() if self.entry else None,
            "manifest": self.manifest,
            "source": self.source,
            "cached": bool(self.cached),
            "attempts": list(self.attempts),
            "elapsed_s": round(self.elapsed_s, 2),
        }


def _sample_tile(spec: TilesetSpec) -> Tuple[int, int, int]:
    z = max(int(spec.min_zoom), min(int(spec.max_zoom), 10))
    w, s, e, n = spec.bbox
    lon = (w + e) / 2.0
    lat = (s + n) / 2.0
    xf, yf = lonlat_to_tile(lon, lat, z)
    return z, int(xf), int(yf)


def _http_get(url: str, timeout_s: float, headers: Optional[Dict[str, str]] = None,
              read_bytes: Optional[int] = None) -> Tuple[int, bytes, float]:
    started = time.perf_counter()
    req = Request(url, headers=dict(
        {"User-Agent": "situation3d-geo-engine/1.0"}, **(headers or {})))
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            data = resp.read(read_bytes) if read_bytes else resp.read()
            elapsed = time.perf_counter() - started
            status = getattr(resp, "status", 200)
            return status, data, elapsed
    except HTTPError as exc:
        elapsed = time.perf_counter() - started
        return exc.code, b"", elapsed


def probe_candidate(candidate: SourceCandidate, spec: TilesetSpec,
                    timeout_s: float = 5.0) -> ProbeResult:
    try:
        if candidate.method == "xyz":
            params = candidate.params
            z, x, y = _sample_tile(spec)
            max_zoom = int(params.get("max_zoom") or 22)
            if z > max_zoom:
                z = min(z, max_zoom)
                x, y = int((x % (2 ** z))), int((y % (2 ** z)))
            url = str(params["url"]).format(z=z, x=x, y=y)
            status, data, elapsed = _http_get(url, timeout_s)
            ext = str(params.get("ext") or "jpg")
            if status != 200:
                return ProbeResult(candidate, False, elapsed * 1000.0,
                                   detail="HTTP %s" % status)
            if not image_magic_ok(data, ext):
                return ProbeResult(candidate, False, elapsed * 1000.0,
                                   len(data), detail="payload is not %s" % ext)
            speed = len(data) / elapsed if elapsed > 0 else 0.0
            return ProbeResult(candidate, True, elapsed * 1000.0, speed,
                               len(data), "HTTP 200")
        if candidate.method == "raster":
            url = str(candidate.params.get("url") or "")
            status, data, elapsed = _http_get(
                url, timeout_s, headers={"Range": "bytes=0-262143"},
                read_bytes=262144)
            if status not in (200, 206):
                return ProbeResult(candidate, False, elapsed * 1000.0,
                                   detail="HTTP %s" % status)
            tiff = data[:4] in (b"II*\x00", b"MM\x00*")
            speed = len(data) / elapsed if elapsed > 0 else 0.0
            return ProbeResult(candidate, bool(tiff), elapsed * 1000.0, speed,
                               len(data), "TIFF" if tiff else "not a TIFF")
        if candidate.method == "file":
            path = str(candidate.params.get("path") or "")
            ok = os.path.exists(path)
            return ProbeResult(candidate, ok, 0.0, 1e12, detail="local path")
        return ProbeResult(candidate, False, detail="unknown method")
    except Exception as exc:  # noqa: BLE001 - a probe failure is data
        return ProbeResult(candidate, False,
                           detail="%s: %s" % (type(exc).__name__, exc))


def probe_parallel(candidates: Sequence[SourceCandidate], spec: TilesetSpec,
                   timeout_s: float = 5.0, workers: int = 8
                   ) -> List[ProbeResult]:
    if not candidates:
        return []
    workers = max(1, min(int(workers), len(candidates)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(probe_candidate, c, spec, timeout_s)
                   for c in candidates]
        return [fut.result() for fut in futures]


def rank_probes(probes: Iterable[ProbeResult]) -> List[ProbeResult]:
    nodes = [p for p in probes if p.ok]
    nodes.sort(key=lambda p: (QUALITY_RANK.get(p.candidate.quality, 3),
                              -float(p.speed_bps),
                              int(p.candidate.priority),
                              p.candidate.source_id))
    return nodes


def probe_report(spec: TilesetSpec,
                 source_ids: Optional[Sequence[str]] = None,
                 raster_urls: Optional[Sequence[str]] = None,
                 timeout_s: float = 5.0) -> List[Dict[str, Any]]:
    probes = probe_parallel(candidates_for_tiles(source_ids, raster_urls),
                            spec, timeout_s)
    ranked = rank_probes(probes)
    ranked_ids = [p.candidate.source_id for p in ranked]
    out = []
    for probe in probes:
        item = probe.as_dict()
        item["rank"] = (ranked_ids.index(probe.candidate.source_id) + 1
                        if probe.candidate.source_id in ranked_ids else None)
        out.append(item)
    out.sort(key=lambda item: (item["rank"] is None, item["rank"] or 0))
    return out


def _cache_dir(store: ResourceStore, spec: TilesetSpec) -> str:
    return os.path.join(str(store.root), "cache", "tiles", spec.slug,
                        spec.hash()[:12])


def _entry_id(spec: TilesetSpec) -> str:
    return "tile_layer:" + slugify(spec.name)


def _cached_result(store: ResourceStore, spec: TilesetSpec) -> Optional[AcquireResult]:
    try:
        entry = store.get(_entry_id(spec))
    except ResourceError:
        return None
    if not os.path.isdir(entry.path):
        return None
    check = verify_tileset(entry.path)
    if not check["ok"]:
        return None
    manifest = check["manifest"]
    if manifest.get("spec_hash") != spec.hash():
        return None
    verify = store.verify(entry.entry_id)
    if not verify["ok"]:
        return None
    return AcquireResult(entry=entry, manifest=manifest,
                         source=manifest.get("source") or {}, cached=True)


def _url_for(candidate: SourceCandidate, z: int, x: int, y: int) -> str:
    return str(candidate.params["url"]).format(z=z, x=x, y=y)


def _fetch_xyz(candidate: SourceCandidate, spec: TilesetSpec, out_dir: str,
               workers: int = 8, timeout_s: float = 30.0,
               fail_ratio: float = 0.1) -> Dict[str, Any]:
    ext = str(candidate.params.get("ext") or "jpg")
    jobs = []
    for z, x, y in tiles_for_bbox(spec.bbox, spec.min_zoom, spec.max_zoom):
        if z > int(candidate.params.get("max_zoom") or 22):
            continue
        jobs.append((_url_for(candidate, z, x, y),
                     os.path.join(out_dir, str(z), str(x), "%d.%s" % (y, ext))))
    if not jobs:
        raise SourceFailed("no tiles in zoom range for %s" % candidate.source_id)
    ok = absent = failed = 0
    written = 0
    errors: List[str] = []
    started = time.perf_counter()

    def fetch(url: str, dest: str) -> Tuple[str, Optional[bytes], str]:
        try:
            status, data, _elapsed = _http_get(url, timeout_s)
        except Exception as exc:  # noqa: BLE001
            return "failed", None, str(exc)
        if status == 404:
            return "absent", None, ""
        if status != 200:
            return "failed", None, "HTTP %s" % status
        if not image_magic_ok(data, ext):
            return "failed", None, "bad payload"
        return "ok", data, ""

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(fetch, url, dest): (url, dest)
                   for url, dest in jobs}
        for fut in as_completed(futures):
            url, dest = futures[fut]
            state, data, detail = fut.result()
            if state == "ok":
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                tmp = dest + ".part"
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, dest)
                written += len(data)
                ok += 1
            elif state == "absent":
                absent += 1
            else:
                failed += 1
                if len(errors) < 5:
                    errors.append("%s: %s" % (url, detail))
    elapsed = time.perf_counter() - started
    total = len(jobs)
    if ok == 0:
        raise SourceFailed("no usable tiles from %s (%d failed, %d absent): %s"
                           % (candidate.source_id, failed, absent,
                              "; ".join(errors[:3])))
    if failed > max(1, int(total * fail_ratio)):
        # Partial failure above the tolerance: prefer another source.
        raise SourceFailed("%d/%d tile requests failed from %s"
                           % (failed, total, candidate.source_id))
    return {"tiles": total, "ok": ok, "absent": absent, "failed": failed,
            "elapsed_s": elapsed, "bytes": written,
            "tiles_per_s": (ok / elapsed) if elapsed > 0 else 0.0}


def _download_raster(candidate: SourceCandidate, spec: TilesetSpec,
                     work_dir: str, min_bps: float = 0.0,
                     timeout_s: int = 900) -> Tuple[str, str]:
    url = str(candidate.params.get("url") or "")
    ext = ".tif"
    for suffix in (".tif", ".tiff", ".png"):
        if url.lower().split("?")[0].endswith(suffix):
            ext = suffix
            break
    dest = os.path.join(work_dir, "source" + ext)
    mode = "full"
    if candidate.params.get("cog", True) and url.startswith(("http://", "https://")):
        try:
            crop_raster_window(url, spec.bbox, dest, timeout_s=timeout_s)
            mode = "cog"
        except Exception:
            # Not every raster is a proper COG; fall back to the full download.
            mode = "full"
    if mode == "full":
        download_file(url, dest, timeout_s=timeout_s, min_bps=min_bps)
    if os.path.getsize(dest) < 1024:
        raise SourceFailed("downloaded raster too small")
    return dest, mode


def _quality_ok(stats: Dict[str, Any]) -> bool:
    return stats.get("tile_count", 0) > 0 and stats.get("nonempty_ratio", 0.0) >= 0.01


def _install(work_tiles: str, final_dir: str) -> None:
    parent = os.path.dirname(final_dir)
    os.makedirs(parent, exist_ok=True)
    backup = final_dir + ".old"
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)
    if os.path.exists(final_dir):
        os.replace(final_dir, backup)
    try:
        os.replace(work_tiles, final_dir)
    except Exception:
        if os.path.exists(backup):
            os.replace(backup, final_dir)
        raise
    if os.path.exists(backup):
        shutil.rmtree(backup, ignore_errors=True)


def acquire_tiles(store: ResourceStore, spec: TilesetSpec, mode: str = "auto",
                  source_ids: Optional[Sequence[str]] = None,
                  raster_urls: Optional[Sequence[str]] = None,
                  candidates: Optional[Sequence[SourceCandidate]] = None,
                  min_bps: float = 0.0, force: bool = False,
                  workers: int = 8, probe_timeout_s: float = 5.0,
                  timeout_s: int = 900) -> AcquireResult:
    spec.validate()
    if mode not in ("auto", "offline", "online"):
        raise ValueError("mode must be auto/offline/online")
    if mode == "online":
        force = True
    cached = None if force else _cached_result(store, spec)
    if cached is not None:
        return cached
    if mode == "offline":
        raise AcquireError("offline mode and no valid cached tileset for %s"
                           % spec.name)
    pool = list(candidates) if candidates is not None else candidates_for_tiles(
        source_ids, raster_urls)
    if not pool:
        raise AcquireError("no acquisition source for %s" % spec.name)
    started = time.perf_counter()
    probes = probe_parallel(pool, spec, timeout_s=probe_timeout_s,
                            workers=workers)
    ranked = [p for p in rank_probes(probes)
              if not min_bps or p.speed_bps >= min_bps or p.candidate.method == "file"]
    attempts: List[Dict[str, Any]] = []
    attempt_by_id: Dict[str, Dict[str, Any]] = {}
    ranked_ids = [p.candidate.source_id for p in ranked]
    for probe in probes:
        item = dict(probe.as_dict(),
                    result=("probe_ok" if probe.ok else "probe_failed"))
        if probe.candidate.source_id not in ranked_ids:
            item["error"] = ("probe failed" if not probe.ok
                             else "below min_bps")
            item["result"] = "skipped"
        attempts.append(item)
        attempt_by_id[probe.candidate.source_id] = item
    if not ranked:
        raise AcquireError("no usable source for %s (probes: %s)"
                           % (spec.name,
                              "; ".join("%s=%s" % (p.candidate.source_id,
                                                   p.detail or ("ok" if p.ok else "fail"))
                                        for p in probes)))
    final_dir = _cache_dir(store, spec)
    work_root = os.path.join(str(store.root), "cache", "tiles", "_tmp")
    os.makedirs(work_root, exist_ok=True)
    for probe in ranked:
        candidate = probe.candidate
        work = tempfile.mkdtemp(prefix="acquire-", dir=work_root)
        attempt = attempt_by_id.get(candidate.source_id) or \
            dict(probe.as_dict(), result="started")
        attempt["result"] = "started"
        try:
            tiles_dir = os.path.join(work, "tiles")
            os.makedirs(tiles_dir, exist_ok=True)
            source_meta = {"source_id": candidate.source_id,
                           "method": candidate.method,
                           "url": candidate.params.get("url"),
                           "license": candidate.params.get("license") or "unknown",
                           "attribution": candidate.params.get("attribution") or ""}
            if candidate.method == "xyz":
                fetch_stats = _fetch_xyz(candidate, spec, tiles_dir, workers,
                                         timeout_s=timeout_s)
                stats = tileset_stats(tiles_dir)
                source_meta["method"] = "xyz"
                source_meta["fetch"] = fetch_stats
            elif candidate.method == "raster":
                src, download_mode = _download_raster(
                    candidate, spec, work, min_bps=min_bps,
                    timeout_s=timeout_s)
                info = raster_to_tiles(
                    src, spec, tiles_dir,
                    hillshade=(spec.layer == "terrain"),
                    timeout_s=timeout_s, crop=(download_mode != "cog"))
                source_meta.update({"method": "raster",
                                    "download_mode": download_mode,
                                    "raster": info.get("raster"),
                                    "hillshade": info.get("hillshade")})
                stats = tileset_stats(tiles_dir)
            elif candidate.method == "file":
                src = str(candidate.params.get("path") or "")
                if not os.path.isdir(src):
                    raise SourceFailed("local tileset path missing")
                shutil.copytree(src, tiles_dir, dirs_exist_ok=True)
                stats = tileset_stats(tiles_dir)
                source_meta["method"] = "file"
            else:
                raise SourceFailed("unsupported method %s" % candidate.method)
            if not _quality_ok(stats):
                raise SourceFailed("tileset quality check failed: %r" % stats)
            quality = ("derived" if source_meta.get("hillshade") else
                       candidate.quality)
            attempt.update(result="ok", elapsed_s=time.perf_counter() - started)
            write_manifest(tiles_dir, spec, source_meta, stats, attempts)
            _install(tiles_dir, final_dir)
            digest, size = hash_path(final_dir)
            entry = ResourceEntry(
                entry_id=_entry_id(spec),
                kind="tile_layer",
                path=final_dir,
                content_hash=digest,
                size=size,
                version=digest[:12],
                schema="s3d.tileset.v1",
                provider="acquire:" + candidate.source_id,
                source="%s via %s" % (spec.name, candidate.source_id),
                fetched_at=utc_now(),
                as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                bbox=spec.bbox,
                license=str(source_meta.get("license") or "unknown"),
                quality=quality,
                notes={"spec_hash": spec.hash(),
                       "manifest": "manifest.json",
                       "method": candidate.method,
                       "zoom_range": [spec.min_zoom, spec.max_zoom],
                       "attempts": list(attempts)},
            ).validate()
            store.register(entry, replace=True)
            if attempt not in attempts:
                attempts.append(attempt)
            manifest = verify_tileset(final_dir)["manifest"]
            return AcquireResult(entry=entry, manifest=manifest,
                                 source=source_meta, cached=False,
                                 attempts=attempts,
                                 elapsed_s=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - try the next source
            attempt.update(result="failed", error="%s: %s"
                           % (type(exc).__name__, exc),
                           elapsed_s=time.perf_counter() - started)
            if attempt not in attempts:
                attempts.append(attempt)
            continue
        finally:
            shutil.rmtree(work, ignore_errors=True)
    raise AcquireError("all sources failed for %s: %s"
                       % (spec.name, "; ".join(
                           "%s=%s(%s)" % (a.get("source_id"),
                                           a.get("result"),
                                           a.get("error") or "")
                           for a in attempts)))
