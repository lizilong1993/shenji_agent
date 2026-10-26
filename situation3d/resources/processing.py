"""Normalise downloaded rasters/tiles into a standard tileset.

Output layout is standard XYZ::

    <root>/manifest.json
    <root>/<z>/<x>/<y>.png

gdal2tiles in GDAL 3.0 writes TMS (bottom-up y); this module flips the y
directory after rendering so consumers (Cesium, Leaflet, our server) all see
the same XYZ scheme.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

from .model import canonical_json, hash_path, sha256_bytes, slugify, utc_now
from .tilemath import TILE_SIZE, tile_count


@dataclass(frozen=True)
class TilesetSpec:
    name: str
    bbox: Tuple[float, float, float, float]
    min_zoom: int
    max_zoom: int
    format: str = "png"
    tile_size: int = TILE_SIZE
    layer: str = "imagery"

    def validate(self) -> "TilesetSpec":
        if not self.name:
            raise ValueError("tileset name is required")
        if len(self.bbox) != 4:
            raise ValueError("bbox must be west,south,east,north")
        w, s, e, n = [float(v) for v in self.bbox]
        if not (w < e and s < n):
            raise ValueError("bbox must be west<south<east<north")
        if not all(math.isfinite(v) for v in (w, s, e, n)):
            raise ValueError("bbox values must be finite")
        if not (0 <= int(self.min_zoom) <= int(self.max_zoom) <= 22):
            raise ValueError("zoom range must satisfy 0 <= min <= max <= 22")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "s3d.tileset.spec.v1",
            "name": self.name,
            "bbox": [float(v) for v in self.bbox],
            "min_zoom": int(self.min_zoom),
            "max_zoom": int(self.max_zoom),
            "format": self.format,
            "tile_size": int(self.tile_size),
            "layer": self.layer,
        }

    def hash(self) -> str:
        return sha256_bytes(canonical_json(self.as_dict()).encode("utf-8"))

    @property
    def slug(self) -> str:
        return slugify(self.name)


def image_magic_ok(data: bytes, ext: str) -> bool:
    if not data or len(data) < 64:
        return False
    if ext in ("jpg", "jpeg"):
        return data[:2] == b"\xff\xd8"
    if ext == "png":
        return data[:8] == b"\x89PNG\r\n\x1a\n"
    if ext in ("webp",):
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return True


def tileset_stats(root: str, ext: Optional[str] = None) -> Dict[str, Any]:
    count = 0
    total = 0
    nonempty = 0
    zooms = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith((".part", ".absent", ".xml")) or name == "manifest.json":
                continue
            path = os.path.join(dirpath, name)
            size = os.path.getsize(path)
            count += 1
            total += size
            suffix = os.path.splitext(name)[1].lstrip(".").lower()
            if size >= 128:
                try:
                    with open(path, "rb") as fh:
                        head = fh.read(16)
                    if image_magic_ok(head + b"\0" * 64, suffix):
                        nonempty += 1
                except OSError:
                    pass
            try:
                zooms.append(int(os.path.basename(os.path.dirname(os.path.dirname(path)))))
            except ValueError:
                pass
    return {
        "tile_count": count,
        "total_bytes": total,
        "nonempty_tiles": nonempty,
        "nonempty_ratio": (nonempty / float(count)) if count else 0.0,
        "zoom_min": min(zooms) if zooms else None,
        "zoom_max": max(zooms) if zooms else None,
    }


def write_manifest(root: str, spec: TilesetSpec, source: Dict[str, Any],
                   stats: Dict[str, Any], attempts: Sequence[Dict[str, Any]],
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    manifest = {
        "schema": "s3d.tileset.v1",
        "name": spec.name,
        "kind": "tile_layer",
        "scheme": "xyz",
        "bbox": [float(v) for v in spec.bbox],
        "min_zoom": int(spec.min_zoom),
        "max_zoom": int(spec.max_zoom),
        "tile_size": int(spec.tile_size),
        "format": spec.format,
        "layer": spec.layer,
        "spec_hash": spec.hash(),
        "built": utc_now(),
        "source": dict(source),
        "attempts": [dict(item) for item in attempts],
        "tile_count": stats.get("tile_count"),
        "total_bytes": stats.get("total_bytes"),
        "nonempty_tiles": stats.get("nonempty_tiles"),
        "nonempty_ratio": stats.get("nonempty_ratio"),
        "zoom_min": stats.get("zoom_min"),
        "zoom_max": stats.get("zoom_max"),
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(root, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return manifest


def verify_tileset(root: str) -> Dict[str, Any]:
    path = os.path.join(root, "manifest.json")
    if not os.path.isfile(path):
        return {"ok": False, "status": "missing_manifest"}
    try:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        return {"ok": False, "status": "bad_manifest", "detail": str(exc)}
    stats = tileset_stats(root)
    ok = bool(stats["tile_count"]) and stats["nonempty_ratio"] >= 0.01
    return {"ok": ok, "status": "ok" if ok else "empty_tileset",
            "manifest": manifest, "stats": stats}


def download_file(url: str, dest: str, timeout_s: int = 900,
                  progress=None, min_bps: float = 0.0,
                  grace_s: float = 3.0, grace_bytes: int = 2 << 20) -> str:
    """Atomically download a raster (honours https_proxy); supports file://."""
    from urllib.request import Request, urlopen

    if url.startswith("file://"):
        src = url[7:]
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        shutil.copyfile(src, dest)
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    done = 0
    started = __import__("time").perf_counter()
    req = Request(url, headers={"User-Agent": "situation3d-geo-engine/1.0"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
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
                    elapsed = __import__("time").perf_counter() - started
                    if (min_bps > 0 and elapsed >= grace_s
                            and done >= grace_bytes
                            and done / elapsed < min_bps):
                        raise RuntimeError(
                            "download below min_bps: %.0f < %.0f B/s"
                            % (done / elapsed, min_bps))
        if done <= 0:
            raise RuntimeError("downloaded raster is empty")
        os.replace(tmp, dest)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def _run(cmd, timeout_s=900):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError("%s failed: %s" % (
            cmd[0], proc.stderr.decode("utf-8", "replace")[-800:]))
    return proc


def crop_raster_window(url: str, bbox: Sequence[float], dest: str,
                       timeout_s: int = 180) -> str:
    """Read only the bbox window from a (remote) GeoTIFF via GDAL /vsicurl.

    This is the Cloud-Optimized-GeoTIFF pattern: HTTP range requests instead
    of downloading the whole tile.  ``gdalwarp -te_srs`` accepts WGS84 bounds
    regardless of the source CRS and writes a small local GeoTIFF.
    """
    if shutil.which("gdalwarp") is None:
        raise RuntimeError("gdalwarp not found")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    src = url
    if url.startswith(("http://", "https://")):
        src = "/vsicurl/" + url
    w, s, e, n = [float(v) for v in bbox]
    env = dict(os.environ)
    env.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    env.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.zip")
    env.setdefault("GDAL_HTTP_MAX_RETRY", "2")
    env.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
    cmd = ["gdalwarp", "-q", "-te_srs", "EPSG:4326",
           "-te", "%.8f" % w, "%.8f" % s, "%.8f" % e, "%.8f" % n,
           "-of", "GTiff", src, dest]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout_s, env=env)
    if proc.returncode != 0 or not os.path.isfile(dest) \
            or os.path.getsize(dest) < 1024:
        raise RuntimeError("COG window crop failed: %s"
                           % proc.stderr.decode("utf-8", "replace")[-600:])
    return dest


def raster_info(path: str) -> Dict[str, Any]:
    if shutil.which("gdalinfo") is None:
        raise RuntimeError("gdalinfo not found")
    proc = _run(["gdalinfo", "-json", path], timeout_s=180)
    doc = json.loads(proc.stdout.decode("utf-8"))
    bands = doc.get("bands") or []
    size = doc.get("size") or [0, 0]
    return {
        "width": int(size[0]) if size else 0,
        "height": int(size[1]) if size else 0,
        "band_count": len(bands),
        "band_types": [b.get("type") for b in bands],
        "crs": ((doc.get("coordinateSystem") or {}).get("wkt") or "")[:200],
        "bounds": doc.get("cornerCoordinates"),
    }


def _flip_tms_to_xyz(root: str) -> int:
    """gdal2tiles 3.0 renders TMS y; rename directories to XYZ y."""
    renamed = 0
    for zname in sorted(os.listdir(root)):
        if not zname.isdigit():
            continue
        z = int(zname)
        zdir = os.path.join(root, zname)
        if not os.path.isdir(zdir):
            continue
        side = 2 ** z
        for xname in sorted(os.listdir(zdir)):
            xdir = os.path.join(zdir, xname)
            if not os.path.isdir(xdir) or not xname.isdigit():
                continue
            for yname in sorted(os.listdir(xdir)):
                stem, ext = os.path.splitext(yname)
                if not stem.isdigit() or not ext:
                    continue
                new_y = str(side - 1 - int(stem))
                if new_y == stem:
                    continue
                src = os.path.join(xdir, yname)
                dst = os.path.join(xdir, new_y + ext)
                if os.path.exists(dst):
                    os.unlink(dst)
                os.replace(src, dst)
                renamed += 1
    return renamed


def raster_to_tiles(src: str, spec: TilesetSpec, out_dir: str,
                    hillshade: bool = False, processes: int = 2,
                    timeout_s: int = 900, crop: bool = True) -> Dict[str, Any]:
    """Render a GeoTIFF/COG window into an XYZ tileset (PNG)."""
    spec.validate()
    info = raster_info(src)
    if info["width"] <= 0 or info["height"] <= 0:
        raise RuntimeError("raster has no pixels: %s" % src)
    os.makedirs(out_dir, exist_ok=True)
    work = src
    tmp_work = None
    tmp_crop = None
    try:
        if crop:
            w, s, e, n = [float(v) for v in spec.bbox]
            fd, tmp_crop = tempfile.mkstemp(suffix=".tif", dir=out_dir)
            os.close(fd)
            _run(["gdalwarp", "-q", "-te_srs", "EPSG:4326",
                  "-te", "%.8f" % w, "%.8f" % s, "%.8f" % e, "%.8f" % n,
                  "-of", "GTiff", src, tmp_crop], timeout_s=timeout_s)
            work = tmp_crop
        if hillshade or info["band_count"] == 1:
            if shutil.which("gdaldem") is None:
                raise RuntimeError("gdaldem not found")
            fd, tmp_work = tempfile.mkstemp(suffix=".tif", dir=out_dir)
            os.close(fd)
            _run(["gdaldem", "hillshade", "-compute_edges", "-of", "GTiff",
                  "-alg", "Horn", work, tmp_work], timeout_s=timeout_s)
            work = tmp_work
        if shutil.which("gdal2tiles.py") is None:
            raise RuntimeError("gdal2tiles.py not found")
        _run(["gdal2tiles.py", "-p", "mercator",
              "-z", "%d-%d" % (spec.min_zoom, spec.max_zoom),
              "-r", "bilinear", "--processes", str(max(1, int(processes))),
              "-w", "none", work, out_dir], timeout_s=timeout_s)
        _flip_tms_to_xyz(out_dir)
        for name in ("tilemapresource.xml",):
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                os.unlink(path)
        return {"raster": info, "hillshade": bool(hillshade or info["band_count"] == 1)}
    finally:
        for path in (tmp_work, tmp_crop):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
