"""Disk cache shared by the online and offline data paths.

Online sources write here; offline builds read the same keys.  ``mode``
controls behaviour when a key is missing: ``online`` fetches, ``offline``
raises ``OfflineCacheMiss``, ``auto`` fetches and falls back to the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class OfflineCacheMiss(FileNotFoundError):
    pass


class CacheCorrupt(ValueError):
    pass


class CacheStore(object):
    def __init__(self, root: str, mode: str = "auto"):
        if mode not in ("online", "offline", "auto"):
            raise ValueError("mode must be online/offline/auto")
        self.root = Path(root)
        self.mode = mode
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, kind: str, name: str) -> Path:
        safe = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
        return self.root / kind / safe

    def has(self, kind: str, name: str) -> bool:
        p = self.key(kind, name)
        try:
            return p.is_file() and p.stat().st_size > 0
        except OSError:
            return False

    def read_bytes(self, kind: str, name: str):
        p = self.key(kind, name)
        if not p.is_file():
            raise OfflineCacheMiss("cache miss: %s/%s" % (kind, name))
        return p.read_bytes()

    def write_bytes(self, kind: str, name: str, data: bytes) -> None:
        p = self.key(kind, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def read_json(self, kind: str, name: str):
        data = self.read_bytes(kind, name)
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            self.quarantine(kind, name)
            raise CacheCorrupt("corrupt JSON cache %s/%s: %s" % (kind, name, exc))

    def quarantine(self, kind: str, name: str) -> None:
        p = self.key(kind, name)
        if p.is_file():
            try:
                os.replace(p, p.with_suffix(p.suffix + ".corrupt"))
            except OSError:
                pass

    def write_json(self, kind: str, name: str, obj) -> None:
        self.write_bytes(kind, name,
                         json.dumps(obj, ensure_ascii=False).encode("utf-8"))
