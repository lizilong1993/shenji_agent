"""Local scanners: turn existing artifacts into ``ResourceEntry`` records.

P0 deliberately scans in place instead of copying payloads into a cache
directory.  The entries carry a content hash so ``store.verify`` can still
detect drift; P3 providers will write fetched artifacts under
``var/resources/cache`` and register them the same way.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .model import ResourceEntry, hash_path


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9_.-]+", "-", str(text).strip().lower())
    return value.strip("-.") or "unnamed"


def _iso_date(value) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(value or ""))
    return match.group(1) if match else ""


def _mtime_iso(path: str) -> str:
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return ""
    return datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _newest_date(root: str) -> str:
    newest = ""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            stamp = _mtime_iso(os.path.join(dirpath, name))
            if stamp > newest:
                newest = stamp
    return newest[:10]



def _repo_stamp(path: str):
    """Commit hash + date for a tracked path, or empty strings outside git."""
    try:
        out = subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(path)), "log", "-1",
             "--format=%H %cI", "--", os.path.abspath(path)],
            stderr=subprocess.DEVNULL, timeout=5)
        parts = out.decode("utf-8", "replace").strip().split(" ", 1)
        if len(parts) == 2:
            return parts[0][:12], parts[1]
    except Exception:  # noqa: BLE001 - timestamp fallback below
        pass
    return "", ""

def _load_json(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object: %s" % path)
    return data


def _make(entry_id: str, kind: str, path: str, doc_schema: str,
          provider: str, source: str, as_of: str, fetched_at: str,
          bbox=None, license: str = "", quality: str = "measured",
          notes: Optional[Dict[str, object]] = None) -> ResourceEntry:
    digest, size = hash_path(path)
    return ResourceEntry(
        entry_id=entry_id,
        kind=kind,
        path=os.path.abspath(path),
        content_hash=digest,
        size=size,
        version=digest[:12],
        schema=doc_schema,
        provider=provider,
        source=source,
        fetched_at=fetched_at,
        as_of=as_of,
        bbox=tuple(float(v) for v in bbox) if bbox else None,
        license=license,
        quality=quality,
        notes=dict(notes or {}),
    ).validate()


def scan_envpack(path: str) -> ResourceEntry:
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise ValueError("envpack manifest not found: %s" % manifest_path)
    doc = _load_json(manifest_path)
    schema = str(doc.get("schema") or "")
    if not schema.startswith("msgeo.envpack."):
        raise ValueError("unsupported envpack schema %r in %s" % (schema, path))
    name = str(doc.get("pack_id") or doc.get("theatre") or os.path.basename(path))
    as_of = _iso_date(doc.get("as_of") or doc.get("built")) or _mtime_iso(manifest_path)[:10]
    fetched_at = str(doc.get("fetched_at") or "") or _mtime_iso(manifest_path)
    provenance = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
    layers = doc.get("layers") if isinstance(doc.get("layers"), dict) else {}
    layer_sources = [str((meta or {}).get("source") or "")
                     for meta in layers.values() if isinstance(meta, dict)]
    licenses = [str((meta or {}).get("license") or "")
                for meta in layers.values() if isinstance(meta, dict)]
    licenses = sorted(set(item for item in licenses if item))
    if doc.get("license"):
        license_text = str(doc["license"])
    elif licenses:
        license_text = ", ".join(licenses)
    else:
        license_text = "unknown"
    if not layer_sources and not provenance:
        quality = "unknown"
    elif provenance.get("fallback") or any(
            src.lower().startswith("fallback") or src.lower() in ("default", "none")
            for src in layer_sources if src):
        quality = "derived"
    else:
        quality = "measured"
    source_bits = []
    for key in ("osm", "dem", "bathymetry", "roads"):
        if provenance.get(key):
            source_bits.append("%s=%s" % (key, provenance[key]))
    source = "envpack %s (%s)" % (schema, ", ".join(source_bits) or "no provenance")
    return _make(
        entry_id="envpack:" + _slug(name),
        kind="envpack",
        path=path,
        doc_schema=schema,
        provider="local-scan",
        source=source,
        as_of=as_of,
        fetched_at=fetched_at,
        bbox=doc.get("bbox"),
        license=license_text,
        quality=quality,
        notes={"manifest": "manifest.json",
               "provenance": dict(provenance),
               "layers": sorted(layers)},
    )


def scan_rolecards(data_dir: str) -> ResourceEntry:
    classes = os.path.join(data_dir, "classes.json")
    if not os.path.isfile(classes):
        raise ValueError("role-card data missing classes.json: %s" % data_dir)
    doc = _load_json(classes)
    commit, stamp = _repo_stamp(classes)
    return _make(
        entry_id="rolecards:repo",
        kind="rolecards",
        path=data_dir,
        doc_schema=str(doc.get("schema") or "situation3d.rolecard"),
        provider="local-scan",
        source="repo/engine/rolecards/data",
        as_of=(stamp or _newest_date(data_dir))[:10],
        fetched_at=stamp or _mtime_iso(classes),
        license="internal",
        notes={"files": sorted(os.listdir(data_dir)), "commit": commit},
    )


def scan_platforms(platforms_dir: str) -> List[ResourceEntry]:
    out = []
    for path in sorted(glob.glob(os.path.join(platforms_dir, "*.json"))):
        doc = _load_json(path)
        key = str(doc.get("key") or os.path.splitext(os.path.basename(path))[0])
        commit, stamp = _repo_stamp(path)
        out.append(_make(
            entry_id="platforms:" + _slug(key),
            kind="platforms",
            path=path,
            doc_schema=str(doc.get("schema") or ""),
            provider="local-scan",
            source="repo/platforms/" + os.path.basename(path),
            as_of=(stamp or _mtime_iso(path))[:10],
            fetched_at=stamp or _mtime_iso(path),
            license="internal-synthetic",
            notes={"key": key, "name": (doc.get("identity") or {}).get("name_zh"),
                   "commit": commit},
        ))
    if not out:
        raise ValueError("no platform profiles found under %s" % platforms_dir)
    return out


def scan_rules(rules_dir: str) -> ResourceEntry:
    manifest = os.path.join(rules_dir, "manifest.json")
    if not os.path.isfile(manifest):
        raise ValueError("rules manifest not found: %s" % manifest)
    doc = _load_json(manifest)
    commit, stamp = _repo_stamp(manifest)
    return _make(
        entry_id="weapons:repo",
        kind="weapons",
        path=rules_dir,
        doc_schema=str(doc.get("schema") or ""),
        provider="local-scan",
        source="repo/rules/data",
        as_of=_iso_date(doc.get("source_date") or doc.get("generated"))
              or (stamp or _mtime_iso(manifest))[:10],
        fetched_at=stamp or _mtime_iso(manifest),
        license="internal",
        notes={"files": list(doc.get("files") or []), "commit": commit},
    )


def scan_roads(roads_dir: str) -> List[ResourceEntry]:
    out = []
    for path in sorted(glob.glob(os.path.join(roads_dir, "*.json"))):
        doc = _load_json(path)
        theatre = str(doc.get("theatre") or os.path.splitext(os.path.basename(path))[0])
        out.append(_make(
            entry_id="case_roads:" + _slug(theatre),
            kind="case_roads",
            path=path,
            doc_schema=str(doc.get("schema") or ""),
            provider="local-scan",
            source=str(doc.get("mirror") or "repo/engine/data/roads"),
            as_of=_iso_date(doc.get("osm_base_utc")) or _mtime_iso(path)[:10],
            fetched_at=str(doc.get("fetched_utc") or "") or _mtime_iso(path),
            bbox=doc.get("bbox"),
            license="ODbL-1.0",
            notes={"theatre": theatre,
                   "osm_base_utc": doc.get("osm_base_utc"),
                   "classes": doc.get("classes")},
        ))
    if not out:
        raise ValueError("no road networks found under %s" % roads_dir)
    return out


def scan_repo(pkg_root: str) -> Tuple[List[ResourceEntry], List[Dict[str, str]]]:
    """Scan the known artifact locations under ``situation3d``.

    Returns ``(entries, errors)``; missing optional directories are skipped,
    malformed artifacts are reported instead of aborting the whole scan.
    """
    root = os.path.abspath(pkg_root)
    entries: List[ResourceEntry] = []
    errors: List[Dict[str, str]] = []

    envpacks = os.path.join(root, "envpacks")
    if os.path.isdir(envpacks):
        for child in sorted(os.listdir(envpacks)):
            path = os.path.join(envpacks, child)
            if not os.path.isfile(os.path.join(path, "manifest.json")):
                continue
            try:
                entries.append(scan_envpack(path))
            except Exception as exc:  # noqa: BLE001 - report, keep scanning
                errors.append({"path": path, "error": "%s: %s"
                               % (type(exc).__name__, exc)})

    steps = [
        ("rolecards", os.path.join(root, "engine", "rolecards", "data"),
         lambda p: [scan_rolecards(p)]),
        ("platforms", os.path.join(root, "platforms"), scan_platforms),
        ("weapons", os.path.join(root, "rules", "data"),
         lambda p: [scan_rules(p)]),
        ("case_roads", os.path.join(root, "engine", "data", "roads"),
         scan_roads),
    ]
    for _name, path, fn in steps:
        if not os.path.isdir(path):
            continue
        try:
            entries.extend(fn(path))
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": path, "error": "%s: %s"
                           % (type(exc).__name__, exc)})
    return entries, errors
