"""Immutable raw-response archive for VGI connectors."""

from __future__ import annotations

import gzip
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

from ..model import iso, parse_time


@dataclass(frozen=True)
class ArchiveRecord:
    source_id: str
    path: str
    sha256: str
    bytes: int
    fetched_at: str

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "fetched_at": self.fetched_at,
        }


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "source"))
    return text.strip("._") or "source"


def archive_raw(root: str, source_id: str, payload: bytes,
                fetched_at: Any, ext: str = "json") -> ArchiveRecord:
    """Write ``payload`` as ``root/<source>/<utc>.<ext>.gz`` and return its hash."""
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes")
    body = bytes(payload)
    stamp = parse_time(fetched_at)
    directory = os.path.join(str(root), _safe_name(source_id))
    os.makedirs(directory, exist_ok=True)
    base = "%s.%s" % (stamp.strftime("%Y%m%dT%H%M%SZ"), _safe_name(ext))
    path = os.path.join(directory, base + ".gz")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(directory, "%s-%d.gz" % (base, counter))
        counter += 1
    temporary = path + ".tmp"
    with open(temporary, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           mtime=0) as handle:
            handle.write(body)
    os.replace(temporary, path)
    return ArchiveRecord(
        source_id=str(source_id),
        path=path,
        sha256=hashlib.sha256(body).hexdigest(),
        bytes=len(body),
        fetched_at=iso(stamp),
    )


def read_archive(path: str) -> bytes:
    """Read a raw response back, for replay tests and manual audits."""
    with gzip.open(path, "rb") as handle:
        return handle.read()
